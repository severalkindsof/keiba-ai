"""
第13波: 荒れるレース判定モジュール

「このレースは荒れそう / 堅そう」を 0〜100 のスコアで返す。

2 段階の実装:
    a. ヒューリスティクス版: race_meta から即計算（学習不要）
    b. LightGBM 版: train_volatility_model.py で学習されたモデルを使う

公開関数:
    compute_volatility(race_meta: dict) -> dict
        race_meta:
            field_size, distance, surface, venue, race_class, track_condition,
            top_popularity_odds (任意), race_no (任意), entry_elo_std (任意)
        Returns:
            {
              "score": 0-100,
              "label": "爆穴向き" / "堅軸向き" / "〜中間〜",
              "engine": "heuristic" or "lgbm",
              "components": {key: contribution, ...},
            }
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_DATA = Path(__file__).parent / "data"
_LGB_MODEL_PATH = _DATA / "volatility_lgb.txt"
_LGB_FEAT_PATH = _DATA / "volatility_features.json"

_lgb_model = None
_lgb_features: list[str] | None = None


def _load_lgb():
    """LightGBM モデルを遅延ロード。なければ None。"""
    global _lgb_model, _lgb_features
    if _lgb_model is not None:
        return _lgb_model, _lgb_features
    if not (_LGB_MODEL_PATH.exists() and _LGB_FEAT_PATH.exists()):
        return None, None
    try:
        import lightgbm as lgb
        _lgb_model = lgb.Booster(model_file=str(_LGB_MODEL_PATH))
        with open(_LGB_FEAT_PATH, encoding="utf-8") as f:
            _lgb_features = json.load(f)["features"]
        return _lgb_model, _lgb_features
    except Exception:
        return None, None


# ============================================================
# ヒューリスティクス版
# ============================================================

def _heuristic_score(meta: dict) -> tuple[float, dict]:
    """
    軽量ルールベースで荒れ度スコアを算出（0〜100 にクリップ）。
    基準値 = 40 から要素ごとに加減算。
    """
    score = 40.0
    comp: dict[str, float] = {}

    # 出走頭数
    fs = meta.get("field_size") or 16
    try:
        fs = int(fs)
    except Exception:
        fs = 16
    if fs >= 18:
        comp["頭数(多頭数)"] = +18
    elif fs >= 16:
        comp["頭数(16-17頭)"] = +10
    elif fs <= 10:
        comp["頭数(少頭数)"] = -12

    # 距離
    dist = meta.get("distance") or 1800
    try:
        dist = int(dist)
    except Exception:
        dist = 1800
    if dist <= 1200:
        comp["距離(短距離)"] = +12
    elif dist >= 2400:
        comp["距離(長距離)"] = -8

    # 馬場状態
    tc = (meta.get("track_condition") or "良").strip()
    if tc in ("重", "不良"):
        comp["馬場(道悪)"] = +20
    elif tc == "稍重":
        comp["馬場(稍重)"] = +8

    # クラス（race_class or race_name から推定）
    rc = (meta.get("race_class") or meta.get("race_name") or "")
    rc_str = str(rc)
    if any(g in rc_str for g in ["G1", "GI"]):
        comp["クラス(G1)"] = -15
    elif any(g in rc_str for g in ["G2", "G3", "GII", "GIII"]):
        comp["クラス(重賞)"] = -8
    elif "ハンデ" in rc_str:
        comp["クラス(ハンデ戦)"] = +15
    elif any(k in rc_str for k in ["未勝利", "新馬", "1勝", "2勝", "3勝", "未出走"]):
        comp["クラス(条件戦)"] = +10

    # 1人気オッズ（混戦度）
    pop1 = meta.get("top_popularity_odds")
    if pop1 is not None:
        try:
            pop1 = float(pop1)
            if pop1 >= 5.0:
                comp["混戦(1人気5倍超)"] = +18
            elif pop1 >= 3.5:
                comp["混戦(1人気3.5倍超)"] = +10
            elif pop1 < 2.0:
                comp["本命濃厚"] = -10
        except Exception:
            pass

    # Elo 分散（出走馬の Elo がバラついていれば実力差大→堅め、Elo が拮抗していれば荒れる）
    elo_std = meta.get("entry_elo_std")
    if elo_std is not None:
        try:
            elo_std = float(elo_std)
            if elo_std < 30:
                comp["Elo拮抗(横並び)"] = +12
            elif elo_std > 80:
                comp["Elo分散(実力差大)"] = -10
        except Exception:
            pass

    # 会場別バイアス（course_bias.parquet を参照、外枠/内枠有利は若干荒れやすい）
    venue = meta.get("venue")
    surface = meta.get("surface", "芝")
    cb_path = _DATA / "course_bias.parquet"
    if venue and dist and cb_path.exists():
        try:
            cb = pd.read_parquet(cb_path)
            row = cb[(cb["venue"] == venue) & (cb["surface"] == surface)
                     & (cb["distance"] == int(dist))]
            if not row.empty:
                gb = abs(float(row["gate_bias"].iloc[0]))
                if gb > 0.20:
                    comp[f"コースバイアス強({venue})"] = +6
        except Exception:
            pass

    # 集計
    for v in comp.values():
        score += v
    return float(np.clip(score, 0, 100)), comp


# ============================================================
# 公開 API
# ============================================================

def _label_for_score(score: float) -> str:
    # 第34波: LGBM単独スコアの実測分位で再校正（8,425レース）
    #   >=52: レースの7%が該当・実荒れ率66% / <=40: 21%が該当・実荒れ率21%
    if score >= 52:
        return "爆穴向き"
    if score <= 40:
        return "堅軸向き"
    return "〜中間〜"


def compute_volatility(meta: dict) -> dict:
    """
    レースの荒れ度を計算。LightGBM モデルがあればそれを使い、
    なければヒューリスティクスにフォールバック。
    """
    h_score, h_comp = _heuristic_score(meta)

    model, feats = _load_lgb()
    if model is None or feats is None:
        return {
            "score":   round(h_score, 1),
            "label":   _label_for_score(h_score),
            "engine":  "heuristic",
            "components": {k: round(v, 1) for k, v in h_comp.items()},
        }

    # LightGBM 推論
    try:
        # (第20波 U1 修正) meta は venue/surface/track_condition を文字列で持つが、
        # 学習特徴量は venue_code/surface_code/track_cond_code。コード化が抜けて
        # 全レースがデフォルト0（札幌/芝/良）扱いになり、LGBM が会場・馬場を
        # 完全無視していた（東京芝良と小倉ダ不良で予測同一を実測確認）。
        # 訓練時（train_volatility_model.py）と同一のマッピングでコード化する。
        _SURFACE_CODE = {"芝": 0, "ダート": 1, "障害": 2}
        _VENUE_CODE = {"札幌": 0, "函館": 1, "福島": 2, "新潟": 3, "東京": 4,
                       "中山": 5, "中京": 6, "京都": 7, "阪神": 8, "小倉": 9}
        _TRACK_COND_CODE = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3}
        _defaults = {"field_size": 16, "distance": 1800, "race_no": 10,
                     "month": 6, "field_avg_pop": 8.5, "field_pop_std": 4.5}
        row = {f: meta.get(f, _defaults.get(f, 0)) for f in feats}
        row["surface_code"]    = _SURFACE_CODE.get(str(meta.get("surface", "芝")).strip(), 0)
        row["venue_code"]      = _VENUE_CODE.get(str(meta.get("venue", "")).strip(), 4)
        row["track_cond_code"] = _TRACK_COND_CODE.get(str(meta.get("track_condition", "良")).strip(), 0)
        if not row.get("month"):
            from datetime import date as _d
            row["month"] = _d.today().month
        for k, dv in _defaults.items():
            if k in row and (row[k] is None or row[k] == 0):
                row[k] = dv
        X = pd.DataFrame([row])[feats]
        prob = float(model.predict(X.values)[0])
        # 確率 → 0-100 スコアに変換
        lgb_score = float(np.clip(prob * 100, 0, 100))
        # 第34波: 8,425レース実測で LGBM単独 AUC 0.6607 > 70/30ブレンド 0.6323 >
        # ヒューリスティクス単独 0.5861 — 混ぜるほど劣化するため LGBM 100% に変更
        # （ヒューリスティクスはモデル無し時のフォールバックと components 表示用に温存）
        blended = lgb_score
        return {
            "score":   round(blended, 1),
            "label":   _label_for_score(blended),
            "engine":  "lgbm+heuristic",
            "lgb_prob": round(prob, 3),
            "components": {k: round(v, 1) for k, v in h_comp.items()},
        }
    except Exception:
        return {
            "score":   round(h_score, 1),
            "label":   _label_for_score(h_score),
            "engine":  "heuristic (lgbm load failed)",
            "components": {k: round(v, 1) for k, v in h_comp.items()},
        }


def rank_races_by_volatility(
    race_list: list[dict],
    mode: str = "堅軸",
) -> pd.DataFrame:
    """
    複数レースに荒れ度スコアを付与し、モードに応じて並び替えて返す。

    Args:
        race_list: dict のリスト。各 dict は compute_volatility の meta を含む。
                   さらに表示用の race_name / race_no / venue が含まれることを想定。
        mode: "堅軸" → 荒れ度の低い順、"爆穴" → 高い順

    Returns:
        pd.DataFrame（venue / race_no / race_name / score / label / engine 列）
    """
    rows = []
    for race in race_list:
        v = compute_volatility(race)
        rows.append({
            "venue":     race.get("venue", ""),
            "race_no":   race.get("race_no", ""),
            "race_name": race.get("race_name", ""),
            "score":     v["score"],
            "label":     v["label"],
            "engine":    v["engine"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ascending = (mode == "堅軸")
    return df.sort_values("score", ascending=ascending).reset_index(drop=True)
