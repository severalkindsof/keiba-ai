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
    # "ev": 0.22,  ← 除外
    "pace":          0.12,  # ペース・展開恩恵
    "draw":          0.08,  # 枠順バイアス
    "jockey":        0.12,  # 騎手乗り替わり
    "rotation":      0.10,  # ローテーション
    "class_drop":    0.08,  # クラス変動
    "weight":        0.04,  # 馬体重変化
    "position":      0.10,  # 前走位置取り補正
    "weight_ratio":  0.06,  # 斤量馬体重比
    "nicks":         0.06,  # ニックス
    "season":        0.06,  # 季節・馬場適性
    "race_level":    0.06,  # 前走レースレベル
    "lap":           0.06,  # ラップ適性
    "realtime_bias": 0.06,  # 当日バイアス
    # 合計: 1.00 (100%)
}

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
    if bonus is None or (isinstance(bonus, float) and np.isnan(bonus)):
        return 0.5
    return float(np.clip((bonus + max_val) / (2 * max_val), 0.0, 1.0))


def calc_confluence_score(horse: dict) -> dict:
    """全ファクターを統合して総合信頼スコアを計算する。"""

    # 既存ファクター
    ev            = float(horse.get("ev", -0.2))
    pace          = float(horse.get("pace_benefit", 0.0))
    draw          = float(horse.get("draw_bonus", 0.0))
    jockey        = float(horse.get("jockey_change_bonus", 0.0))
    rotation      = float(horse.get("rotation_bonus", 0.0)) + float(horse.get("tatakidai_bonus", 0.0))
    class_drop    = float(horse.get("class_bonus", 0.0))
    weight_chg    = float(horse.get("weight_bonus", 0.0))

    # 新ファクター
    position      = float(horse.get("position_correction_bonus", 0.0))
    weight_ratio  = float(horse.get("weight_ratio_bonus", 0.0))
    nicks_b       = float(horse.get("nicks_bonus", 0.0))
    season_b      = (float(horse.get("season_bonus", 0.0))
                   + float(horse.get("condition_apt_bonus", 0.0))
                   + float(horse.get("surface_change_bonus", 0.0))
                   + float(horse.get("weight_trend_bonus", 0.0))
                   + float(horse.get("handicap_trend_bonus", 0.0))
                   + float(horse.get("turn_dir_bonus", 0.0))           # 右/左回り適性
                   + float(horse.get("hurdle_to_flat_bonus", 0.0))     # 障害叩き後
                   + float(horse.get("short_term_foreign_bonus", 0.0)) # 短期外国人騎手
                   + float(horse.get("proverb_bonus", 0.0))            # レース格言
                   + float(horse.get("first_time_bonus", 0.0))         # 初距離・初馬場
                   + float(horse.get("stable_bonus", 0.0))             # 厩舎近況
                   + float(horse.get("field_size_bonus", 0.0))         # 頭数変化
                   + float(horse.get("pace_fit_bonus", 0.0)))          # ペース適合
    race_level_b  = (float(horse.get("race_level_bonus", 0.0))
                   + float(horse.get("beat_bonus", 0.0))        # 有力馬撃破
                   + float(horse.get("resume_bonus_total", 0.0))) # 近走言い訳巻き返し
    lap_b         = (float(horse.get("lap_bonus", 0.0))
                   + float(horse.get("last3f_bonus", 0.0))
                   + float(horse.get("time_rank_bonus", 0.0)))
    bias_b        = float(horse.get("realtime_bias_bonus", 0.0))

    # 各スコアを 0〜1 に正規化（EVは除外: 実力評価のみ）
    s = {
        "pace":          _norm(pace,         0.03),
        "draw":          _norm(draw,         0.02),
        "jockey":        _norm(jockey,       0.03),
        "rotation":      _norm(rotation,     0.03),
        "class_drop":    _norm(class_drop,   0.03),
        "weight":        _norm(weight_chg,   0.025),
        "position":      _norm(position,     0.03),
        "weight_ratio":  _norm(weight_ratio, 0.04),
        "nicks":         _norm(nicks_b,      0.03),
        "season":        _norm(season_b,     0.03),
        "race_level":    _norm(race_level_b, 0.03),
        "lap":           _norm(lap_b,        0.025),
        "realtime_bias": _norm(bias_b,       0.03),
    }

    # 加重平均
    raw_score = sum(s[k] * WEIGHTS[k] for k in WEIGHTS)
    score = int(round(raw_score * 100))

    # プラスファクター数カウント
    thresholds = {
        "ev": 0, "pace": 0.005, "draw": 0.005, "jockey": 0.005,
        "rotation": 0.005, "class_drop": 0.005, "weight": 0.0,
        "position": 0.005, "weight_ratio": 0.005, "nicks": 0.005,
        "season": 0.005, "race_level": 0.005, "lap": 0.005, "realtime_bias": 0.005,
    }
    raw_vals = {
        "ev": ev, "pace": pace, "draw": draw, "jockey": jockey,
        "rotation": rotation, "class_drop": class_drop, "weight": weight_chg,
        "position": position, "weight_ratio": weight_ratio, "nicks": nicks_b,
        "season": season_b, "race_level": race_level_b, "lap": lap_b,
        "realtime_bias": bias_b,
    }
    plus_factors = sum(1 for k, thr in thresholds.items() if raw_vals[k] > thr)

    # 調教ボーナス（タイムスコア + 併せ馬ボーナス）
    training_score = horse.get("training_score")
    if training_score is not None:
        t_bonus = (int(training_score) - 50) / 50 * 5  # ±5点
        score = int(np.clip(score + t_bonus, 0, 100))

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
    if race_level_b > 0.01:       reason_parts.append("格上挑戦")
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
    if horse.get("training_label", "") in ("調教◎（状態良好）", "調教○（普通以上）"):
        reason_parts.append("追い切り良")

    # マイナス要因
    if jockey < -0.01:   reason_parts.append("⚠鞍上弱化")
    if rotation < -0.01: reason_parts.append("⚠ローテ難")

    # 何も該当しなければEV数値を表示（「判断材料不足」を廃止）
    if not reason_parts:
        reason_parts.append(f"EV{ev:+.2f}")

    recommend_reason = " / ".join(reason_parts)

    return {
        "confidence_score":  score,
        "confidence_label":  label,
        "confidence_desc":   description,
        "plus_factors":      plus_factors,
        "recommend_reason":  recommend_reason,
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
