"""
総合信頼スコア（Confluence Score）エンジン v2。

全14ファクターを統合して 0〜100 点のスコアを生成する。
複数のプラスファクターが重なった時だけ「買い」判断を出す設計思想。

重み付け（合計100%）:
- EV・期待値ベース      : 22%
- ペース・展開恩恵      : 10%
- 枠順バイアス          :  7%
- 騎手乗り替わり        : 10%
- ローテーション        :  8%
- クラスドロップ        :  7%
- 馬体重変化            :  3%
- 前走位置取り補正      :  8%  ← NEW
- 斤量馬体重比          :  5%  ← NEW
- ニックス              :  5%  ← NEW
- 季節・馬場状態適性    :  5%  ← NEW
- 前走レースレベル      :  5%  ← NEW
- ラップ適性            :  5%  ← NEW (race_level.py)
- リアルタイムバイアス  : 10%  ← NEW（当日情報）
- 調教タイム            :  5%  ← NEW (bonus when available)
"""
import numpy as np
import pandas as pd

WEIGHTS = {
    # EVは除外: 総合スコアは「馬の実力評価」であり、市場評価(EV)は別表示
    "pace":          0.10,  # ペース・展開恩恵
    "draw":          0.06,  # 枠順バイアス
    "jockey":        0.10,  # 騎手乗り替わり
    "rotation":      0.08,  # ローテーション
    "class_drop":    0.06,  # クラス変動
    "weight":        0.03,  # 馬体重変化
    "position":      0.08,  # 前走位置取り補正
    "weight_ratio":  0.05,  # 斤量馬体重比
    "nicks":         0.05,  # ニックス
    "season":        0.05,  # 季節・馬場適性
    "race_level":    0.05,  # 前走レースレベル
    "lap":           0.05,  # ラップ適性
    "realtime_bias": 0.05,  # 当日バイアス
    # DEAD-2: 第4-5波で追加した新ボーナスを統合
    "elo":           0.04,  # 馬 Elo レーティング差
    "pair":          0.04,  # 厩舎×騎手ペア相性
    "pci":           0.03,  # ペース変動耐性
    "speed":         0.05,  # 自作タイム指数
    "speed_idx":     0.03,  # ベスト指数（推奨補強）
    # 合計: 1.00
}

# IMPROVE-Y2: WEIGHTS 合計を起動時にチェック
_W_SUM = sum(WEIGHTS.values())
if abs(_W_SUM - 1.0) > 0.001:
    import warnings as _w
    _w.warn(f"confluence.WEIGHTS 合計が 1.0 から外れています: sum={_W_SUM:.4f}",
            RuntimeWarning, stacklevel=2)


SCORE_LABELS = [
    (85, "◎◎ 最強推奨", "全ファクターが揃った最高信頼度。積極的に買い推奨。"),
    (75, "◎ 強推奨",    "複数プラス要因が重なっている。自信を持って買い推奨。"),
    (62, "○ 推奨",      "有力候補。EV+かつ主要ファクターが揃っている。"),
    (50, "△ 検討可",    "プラス要因はあるが決め手に欠ける。"),
    (38, "▲ 様子見",    "ファクター混在。見送りも選択肢。"),
    ( 0, "✕ 見送り推奨","マイナス要因が多い、またはロマン爆死リスク大。"),
]


def _norm_ev(ev: float) -> float:
    if np.isnan(ev):
        return 0.3
    return float(np.clip((ev + 0.5) / 1.0, 0.0, 1.0))


def _norm(bonus: float, max_val: float = 0.03) -> float:
    # NaN/None: 情報なし = 中立よりやや低め（0.5→0.35 に変更）
    if bonus is None or (isinstance(bonus, float) and np.isnan(bonus)):
        return 0.35
    return float(np.clip((bonus + max_val) / (2 * max_val), 0.0, 1.0))


def calc_confluence_score(horse: dict) -> dict:
    """全ファクターを統合して総合信頼スコアを計算する。"""

    # 既存ファクター
    ev            = float(horse.get("ev", -0.2))
    pace          = float(horse.get("pace_benefit", 0.0))
    # 第36波 (77): course_bias_bonus（第10波 B-3 の97コースパターン、±0.02級）が
    # 死に装飾だった → 同スケールの draw 系として合算（新潟千直の外枠有利等が初めて反映）
    draw          = (float(horse.get("draw_bonus", 0.0) or 0.0)
                     + float(horse.get("course_bias_bonus", 0.0) or 0.0))
    jockey        = float(horse.get("jockey_change_bonus", 0.0))
    rotation      = float(horse.get("rotation_bonus", 0.0)) + float(horse.get("tatakidai_bonus", 0.0))
    class_drop    = float(horse.get("class_bonus", 0.0))
    weight_chg    = float(horse.get("weight_bonus", 0.0))

    # 新ファクター
    position      = float(horse.get("position_correction_bonus", 0.0))
    weight_ratio  = float(horse.get("weight_ratio_bonus", 0.0))
    nicks_b       = float(horse.get("nicks_bonus", 0.0))
    # LATENT-3: season系ボーナスは単純加算せず上位2要因平均（過大評価防止）
    _season_items = [
        float(horse.get("season_bonus", 0.0)),
        float(horse.get("condition_apt_bonus", 0.0)),
        float(horse.get("surface_change_bonus", 0.0)),
        float(horse.get("weight_trend_bonus", 0.0)),
        float(horse.get("handicap_trend_bonus", 0.0)),
        float(horse.get("turn_dir_bonus", 0.0)),
        float(horse.get("hurdle_to_flat_bonus", 0.0)),
        float(horse.get("short_term_foreign_bonus", 0.0)),
        float(horse.get("proverb_bonus", 0.0)),
        float(horse.get("first_time_bonus", 0.0)),
        float(horse.get("stable_bonus", 0.0)),
        float(horse.get("field_size_bonus", 0.0)),
        float(horse.get("pace_fit_bonus", 0.0)),
    ]
    _season_top2 = sorted(_season_items, reverse=True)[:2]
    season_b = sum(_season_top2) / 2.0  # 上位2要因の平均
    race_level_b  = (float(horse.get("race_level_bonus", 0.0))
                   + float(horse.get("beat_bonus", 0.0))        # 有力馬撃破
                   + float(horse.get("resume_bonus_total", 0.0))) # 近走言い訳巻き返し
    lap_b         = (float(horse.get("lap_bonus", 0.0))
                   + float(horse.get("last3f_bonus", 0.0))
                   + float(horse.get("time_rank_bonus", 0.0)))
    bias_b        = float(horse.get("realtime_bias_bonus", 0.0))

    # DEAD-2: 新ボーナス系列を取得
    pair_b   = float(horse.get("pair_bonus", 0.0))
    pci_b    = float(horse.get("pci_bonus", 0.0))
    # Elo ボーナス: race平均 Elo との差を 0.01〜0.05 に変換
    _elo = float(horse.get("elo", 1500))
    _elo_avg = float(horse.get("_race_elo_avg", 1500))   # 後段でレース内平均を補完
    elo_b = max(-0.05, min(0.05, (_elo - _elo_avg) / 400.0 * 0.10))
    # Speed Index ボーナス: best 値を 0.01〜0.05 にスケーリング
    _spd_best = horse.get("speed_index_best")
    spd_b = 0.0
    if _spd_best is not None and not (isinstance(_spd_best, float) and np.isnan(_spd_best)):
        spd_b = max(-0.05, min(0.05, float(_spd_best) / 30.0 * 0.04))
    _spd_avg = horse.get("speed_index_avg")
    spd_avg_b = 0.0
    if _spd_avg is not None and not (isinstance(_spd_avg, float) and np.isnan(_spd_avg)):
        spd_avg_b = max(-0.04, min(0.04, float(_spd_avg) / 25.0 * 0.03))

    # 各スコアを 0〜1 に正規化（EVは除外: 実力評価のみ）
    s = {
        "pace":          _norm(pace,         0.05),
        "draw":          _norm(draw,         0.02),
        "jockey":        _norm(jockey,       0.05),
        "rotation":      _norm(rotation,     0.05),
        "class_drop":    _norm(class_drop,   0.05),
        "weight":        _norm(weight_chg,   0.025),
        "position":      _norm(position,     0.05),
        "weight_ratio":  _norm(weight_ratio, 0.04),
        "nicks":         _norm(nicks_b,      0.03),
        "season":        _norm(season_b,     0.05),
        "race_level":    _norm(race_level_b, 0.05),
        "lap":           _norm(lap_b,        0.04),
        "realtime_bias": _norm(bias_b,       0.03),
        # DEAD-2: 新追加
        "elo":           _norm(elo_b,        0.05),
        "pair":          _norm(pair_b,       0.03),
        "pci":           _norm(pci_b,        0.04),
        "speed":         _norm(spd_b,        0.05),
        "speed_idx":     _norm(spd_avg_b,    0.04),
    }

    # 加重平均
    raw_score = sum(s[k] * WEIGHTS[k] for k in WEIGHTS)
    score = int(round(raw_score * 100))

    # プラスファクター数カウント（LATENT-2: "ev"をWEIGHTSと整合させ除去）
    thresholds = {
        "pace": 0.005, "draw": 0.005, "jockey": 0.005,
        "rotation": 0.005, "class_drop": 0.005, "weight": 0.0,
        "position": 0.005, "weight_ratio": 0.005, "nicks": 0.005,
        "season": 0.005, "race_level": 0.005, "lap": 0.005, "realtime_bias": 0.005,
        "elo": 0.005, "pair": 0.005, "pci": 0.005,
        "speed": 0.01, "speed_idx": 0.005,
    }
    raw_vals = {
        "pace": pace, "draw": draw, "jockey": jockey,
        "rotation": rotation, "class_drop": class_drop, "weight": weight_chg,
        "position": position, "weight_ratio": weight_ratio, "nicks": nicks_b,
        "season": season_b, "race_level": race_level_b, "lap": lap_b,
        "realtime_bias": bias_b,
        "elo": elo_b, "pair": pair_b, "pci": pci_b,
        "speed": spd_b, "speed_idx": spd_avg_b,
    }
    plus_factors = sum(1 for k, thr in thresholds.items() if raw_vals[k] > thr)
    # EV+ は独立フラグとして管理（スコア計算外）
    ev_plus_flag = 1 if ev > 0 else 0
    plus_factors += ev_plus_flag  # EV+は引き続き plus_factors に1加算

    # 調教ボーナス（TFJVタイム評価 優先 → 旧スコア fallback）
    _tfjv_bonus = horse.get("training_bonus")
    if _tfjv_bonus is not None and isinstance(_tfjv_bonus, (int, float)) and abs(float(_tfjv_bonus)) > 0.1:
        # TFJVボーナスは既に±5点スケール
        score = int(np.clip(score + float(_tfjv_bonus), 0, 100))
        if float(_tfjv_bonus) >= 2:
            plus_factors += 1
    else:
        # 旧方式（netkeiba scraper）のフォールバック
        training_score = horse.get("training_score")
        if training_score is not None:
            t_bonus = (int(training_score) - 50) / 50 * 5
            score = int(np.clip(score + t_bonus, 0, 100))

    # 第36波 (76): horse_stats ボーナス（距離適性・道悪・上がり実績・コース特性）は
    # 計算・マージ済みなのに誰も読まない死に装飾だった → ±2点クリップで接続
    # （生スケールは ±0.25 級と未検証に大きいため控えめに変換: ×8, cap ±2）
    _hsb = horse.get("horse_stats_bonus")
    if _hsb is not None and isinstance(_hsb, (int, float)) and not np.isnan(float(_hsb)):
        score = int(np.clip(score + float(np.clip(_hsb * 8.0, -2.0, 2.0)), 0, 100))

    # 土曜勝ち馬との併せ馬ボーナス（最大+8点）
    partner_won_sat = horse.get("partner_won_sat", False)
    won_awase       = horse.get("won_awase")
    if partner_won_sat:
        awase_pts = 8 if won_awase is True else 5
        score = min(100, score + awase_pts)

    # プラスファクター多数ボーナス
    if plus_factors >= 8:
        score = min(100, score + 10)
    elif plus_factors >= 6:
        score = min(100, score + 6)
    elif plus_factors >= 4:
        score = min(100, score + 3)

    # value_ratioは実力スコアに含めない（市場割安/割高はEVで評価）
    # → confidence_score は純粋な「実力・適性・状態」の指標として機能させる

    # ラベル
    label, description = "", ""
    for threshold, lbl, desc in SCORE_LABELS:
        if score >= threshold:
            label, description = lbl, desc
            break

    # ---- 推奨理由を一言で生成 ----
    reason_parts = []

    # EV（0以上ならEV+、それ以下でも-0.15より大きければ数値表示）
    if ev > 0.0:
        reason_parts.append("EV+")
    elif ev > -0.15:
        reason_parts.append(f"EV{ev:+.2f}")

    # ローテーション（rotation_signalを直接使う）
    rot_sig = horse.get("rotation_signal", "")
    if "叩き2走目" in rot_sig:    reason_parts.append("叩き2走目")
    elif "叩き1走目" in rot_sig:  reason_parts.append("休養明け初戦")
    elif "長期休養" in rot_sig:   reason_parts.append("長期休養明け")
    elif "標準間隔" in rot_sig:   reason_parts.append("標準ローテ")
    elif rotation > 0.01:         reason_parts.append("ローテ良")

    # 騎手・脚質・展開・適性
    if jockey > 0.01:             reason_parts.append("騎手強化")
    if lap_b > 0.01:              reason_parts.append("末脚◎")
    if draw > 0.01:               reason_parts.append("枠有利")
    if season_b > 0.01:           reason_parts.append("適性◎")
    _rl = float(horse.get("race_level_bonus", 0.0))
    _bb = float(horse.get("beat_bonus", 0.0))
    _rb = float(horse.get("resume_bonus_total", 0.0))
    if _rl > 0.01:   reason_parts.append("前走強敵相手")
    if _bb > 0.01:   reason_parts.append("強敵撃破歴")
    if _rb > 0.02:   reason_parts.append("一変巻き返し")
    if bias_b > 0.01:             reason_parts.append("バイアス◎")
    if horse.get("exhaustion_comeback"): reason_parts.append("前走消耗→巻返し")

    # horse_stats由来のボーナス
    hs_details = horse.get("horse_stats_details", {})
    if isinstance(hs_details, dict):
        if any("距離適性◎" in k for k in hs_details): reason_parts.append("距離実績◎")
        if any("道悪◎" in k for k in hs_details):     reason_parts.append("道悪得意")
        if any("末脚◎" in k for k in hs_details):     reason_parts.append("上がり上位")
        if any("外国人" in k for k in hs_details):     reason_parts.append("外国人騎手")

    # 調教
    _tr_lbl = horse.get("training_label", "")
    if any(kw in _tr_lbl for kw in ("調教◎", "調教○", "好仕上がり", "普通以上", "状態良好")):
        reason_parts.append("追い切り良")

    # マイナス要因
    if jockey < -0.01:   reason_parts.append("鞍上弱化")
    if rotation < -0.01: reason_parts.append("ローテ難")

    # 何も該当しなければEV数値を表示（「判断材料不足」を廃止）
    if not reason_parts:
        reason_parts.append(f"EV{ev:+.2f}")

    recommend_reason = " / ".join(reason_parts)

    # NEW-9: 非推奨理由（特に上位5人気馬の「人気先行」を検出）
    popularity = int(horse.get("popularity") or 9)
    avoid_parts = []
    if ev < -0.20:                                    avoid_parts.append("EV過大評価")
    if rotation < -0.01:                              avoid_parts.append("ローテ難")
    if jockey < -0.01:                                avoid_parts.append("鞍上弱化")
    # 第45波: LGBM 予測がない（新馬・データ不足）馬で「過大評価」を誤って出すバグを修正
    # → _model_pct が有効値(>2%以上)のときのみ「過大評価」判定する
    _model_pct = float(horse.get("lgbm_norm_pct", 0) or 0)
    _market_pct = 100.0 / float(horse.get("odds", 10) or 10)
    if _model_pct >= 2.0:  # モデル予測が有効な場合のみ
        value_ratio = _model_pct / _market_pct if _market_pct > 0 else 1.0
        if popularity <= 5 and value_ratio < 0.7:  # 0.8 → 0.7 に閾値緩和
            avoid_parts.append("モデル評価＜市場評価")
    # 第45波: NaN safety — pd.NaN は truthy なので `or 99` で吸収できずクラッシュしていた
    import math as _math
    _rot = horse.get("rotation_days")
    try:
        _rot_n = int(_rot) if _rot is not None and not (isinstance(_rot, float) and _math.isnan(_rot)) else 99
    except (TypeError, ValueError):
        _rot_n = 99
    if horse.get("exhaustion_comeback") is False and _rot_n < 10:
        avoid_parts.append("連闘リスク")
    avoid_reason = " / ".join(avoid_parts) if avoid_parts else ""

    return {
        "confidence_score":  score,
        "confidence_label":  label,
        "confidence_desc":   description,
        "plus_factors":      plus_factors,
        "recommend_reason":  recommend_reason,
        "avoid_reason":      avoid_reason,
        "factor_breakdown": {
            "EV期待値":         round(_norm_ev(ev) * 100),  # 表示用のみ（スコア計算には含まない）
            "ペース展開":       round(s["pace"] * 100),
            "枠順":             round(s["draw"] * 100),
            "騎手":             round(s["jockey"] * 100),
            "ローテ":           round(s["rotation"] * 100),
            "クラス変動":       round(s["class_drop"] * 100),
            "馬体重変化":       round(s["weight"] * 100),
            "位置取り補正":     round(s["position"] * 100),
            "斤量比":           round(s["weight_ratio"] * 100),
            "ニックス":         round(s["nicks"] * 100),
            "季節・馬場適性":   round(s["season"] * 100),
            "前走レベル":       round(s["race_level"] * 100),
            "ラップ適性":       round(s["lap"] * 100),
            "当日バイアス":     round(s["realtime_bias"] * 100),
        },
    }


def add_confluence_to_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    """evaluate_race()の出力DataFrameにConfluenceスコアを付与して返す。"""
    if eval_df.empty:
        return eval_df
    # DEAD-2: レース内 Elo 平均を補完（elo_bonus 計算に必要）
    eval_df = eval_df.copy()
    if "elo" in eval_df.columns and eval_df["elo"].notna().any():
        eval_df["_race_elo_avg"] = float(eval_df["elo"].mean())
    scores = [calc_confluence_score(row.to_dict()) for _, row in eval_df.iterrows()]
    score_df = pd.DataFrame(scores)
    out = pd.concat([eval_df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)
    return out.sort_values("confidence_score", ascending=False).reset_index(drop=True)


def get_race_quality_score(eval_df: pd.DataFrame) -> dict:
    """レース全体の「買い頃度」を評価する（race_selector向け）。"""
    if eval_df.empty:
        return {"race_score": 0, "top_score": 0, "ev_plus_count": 0, "verdict": "データなし"}

    top_score  = int(eval_df["confidence_score"].max()) if "confidence_score" in eval_df.columns else 0
    ev_plus    = int((eval_df["ev"] > 0).sum()) if "ev" in eval_df.columns else 0
    avg_score  = float(eval_df["confidence_score"].mean()) if "confidence_score" in eval_df.columns else 0
    race_score = int(top_score * 0.7 + avg_score * 0.3)

    if race_score >= 70:   verdict = "◎ 狙い目レース"
    elif race_score >= 55: verdict = "○ 検討レース"
    elif race_score >= 40: verdict = "△ 様子見"
    else:                  verdict = "✕ 見送り推奨"

    return {"race_score": race_score, "top_score": top_score,
            "ev_plus_count": ev_plus, "verdict": verdict}
