"""WIN5 統合予測モジュール。

1レースの (馬名, 1着確率) を、調教ボーナス・Isotonic校正・sum-normalize 込みで返す。
evaluate_race の最小再現＋WIN5に効く追加処理のみ。
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def predict_race_win_probs(
    race_id: str,
    win_rate_table: pd.DataFrame,
    sire_stats: pd.DataFrame,
    jockey_stats: pd.DataFrame,
    training_results: dict | None = None,
    apply_calibration: bool = True,
) -> list[tuple[str, float]]:
    """1レースの (馬名, 1着確率) リストを返す。確率合計=1。

    Parameters
    ----------
    training_results : tfjv_training_cache の results dict（{馬名: {bonus,label,...}}）
    apply_calibration : True なら Isotonic + sum-normalize を適用
    """
    from scraper import fetch_race_entries, fetch_race_meta
    from ev_calculator import evaluate_race

    entries = fetch_race_entries(race_id)
    meta = fetch_race_meta(race_id) or {}

    # 第46波: netkeibaでオッズ・人気が未確定（朝など）だと全馬同評価=一様分布になる
    # → TFJV出馬表分析CSV（est_odds付き）にフォールバック
    def _entries_lack_market(es):
        return not es or all(
            (e.get("odds") in (None, 0, "")) and (e.get("popularity") in (None, 0, ""))
            for e in es
        )

    if _entries_lack_market(entries):
        tfjv_info = _tfjv_fallback(race_id)
        if tfjv_info:
            entries = tfjv_info["entries"]
            meta = {
                "surface": tfjv_info["surface"],
                "distance": tfjv_info["distance"],
                "track_condition": "良",
            }

    if not entries:
        return []

    surface = meta.get("surface", "芝")
    distance = meta.get("distance", 2000)
    condition = meta.get("track_condition", "良")

    # メタ情報 + 調教ボーナスを各馬に貼る
    for e in entries:
        e["surface"] = surface
        e["distance"] = distance
        e["track_condition"] = condition
        name = str(e.get("horse_name", "")).strip()
        if training_results and name in training_results:
            tr = training_results[name]
            e["training_bonus"] = float(tr.get("bonus", 0)) * 10.0  # 第28波スケール変換
            e["training_label"] = tr.get("label", "")
            e["training_detail"] = tr.get("detail", "")
        else:
            e["training_bonus"] = 0.0
            e["training_label"] = "調教未取得"
            e["training_detail"] = ""

    eval_df = evaluate_race(entries, win_rate_table, sire_stats, jockey_stats)
    if eval_df.empty:
        return []

    # 確率カラム選択（LGBM 優先）
    prob_col = "lgbm_win_rate" if (
        "lgbm_win_rate" in eval_df.columns
        and eval_df["lgbm_win_rate"].notna().any()
    ) else "est_win_rate"

    raw = eval_df[prob_col].fillna(0).values.astype(float) / 100.0
    names = eval_df["horse_name"].astype(str).tolist()

    # Isotonic 校正（適用可能なら）
    if apply_calibration and prob_col == "lgbm_win_rate":
        try:
            from ev_calculator import _CALIBRATOR
            if _CALIBRATOR is not None:
                raw = _CALIBRATOR.transform(raw)
        except Exception:
            pass

    # sum-normalize（1レース内合計=1）
    total = raw.sum()
    if total > 0:
        probs = (raw / total).tolist()
    else:
        # 全部0なら一様分布
        probs = [1.0 / len(names)] * len(names)

    return list(zip(names, probs))


def _tfjv_fallback(race_id_nk: str) -> dict:
    """netkeiba 12桁 race_id → TFJV出馬表分析CSV の該当レース情報を返す。

    netkeiba形式: YYYY + 会場(2) + 開催回(2) + 開催日(2) + R(2)
    会場コードとR番号でマッチング（最新の出馬表分析CSVを探索）。
    """
    try:
        from pathlib import Path
        from tfjv_entries import load_tfjv_entries, VENUE_CODES
        if len(str(race_id_nk)) != 12:
            return {}
        code2v = {v: k for k, v in VENUE_CODES.items()}
        venue = code2v.get(str(race_id_nk)[4:6], "")
        rno = int(str(race_id_nk)[-2:])
        if not venue:
            return {}
        files = sorted(Path("C:/TFJV/TXT").glob("出馬表分析*.CSV"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            data = load_tfjv_entries(f)
            for _rid, info in data.items():
                if info["venue"] == venue and int(info["race_no"]) == rno:
                    return info
    except Exception:
        pass
    return {}


def load_training_cache() -> dict:
    """tfjv_training_cache.json から結果を読み込む。失敗時 {}。"""
    try:
        from tfjv_training import load_training_cache as _load
        return _load() or {}
    except Exception:
        return {}
