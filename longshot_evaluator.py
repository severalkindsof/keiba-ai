"""
穴馬の「構造的根拠」vs「ロマン買い」判定モジュール。

設計思想:
- 完全に予測不可能な穴馬（テンハッピーローズ的）は存在する
- しかし「市場が見落とした理由が検出できる穴馬」と「ただのロマン」は区別できる
- 構造的根拠の数が多いほど「根拠ある穴」、少ないほど「ロマン買い」と判定

構造的根拠の条件（7項目）:
  1. オッズ歪み（推定勝率 > 暗示確率）
  2. クラスドロップ後に人気がついていない
  3. 前走位置取りミスによる着順以上の能力
  4. 今回のペース・バイアスが脚質にハマる
  5. 血統（父系/ニックス）の距離・馬場適性が今回マッチ
  6. 調教タイムの改善（前走比）
  7. 騎手の強化乗り替わり or 手戻り
"""
import pandas as pd
import numpy as np


# ============================================================
# 構造的根拠チェック
# ============================================================

def check_structural_reasons(horse: dict) -> dict:
    """
    1頭の馬の dict から構造的根拠の数を計算し、
    判定ラベル・根拠テキストを返す。

    horse dict に含まれることが望ましいキー:
      ev, implied_prob, est_win_rate, odds, popularity,
      class_signal, position_mismatch_flag, position_correction_msg,
      pace_benefit, draw_bonus, realtime_bias_bonus,
      nicks_bonus, nicks_label,
      training_score, training_label,
      jockey_change_signal, jockey_change_msg
    """
    reasons   = []   # プラスの根拠
    warnings  = []   # マイナス・警告
    ev        = float(horse.get("ev", -0.5))
    popularity = int(horse.get("popularity", 9))
    odds      = float(horse.get("odds", 10.0))

    # 穴馬でない場合はスキップ
    if popularity < 7:
        return {
            "is_longshot": False,
            "structural_count": 0,
            "romance_risk": "対象外（人気馬）",
            "verdict": "人気馬",
            "verdict_emoji": "🔵",
            "reasons": [],
            "warnings": [],
            "summary": "7番人気未満のため穴馬判定対象外",
        }

    # ---- 根拠①：オッズ歪み ---- #
    implied = 1.0 / odds if odds > 0 else 0
    est_wr  = float(horse.get("est_win_rate", 0)) / 100
    distortion = float(horse.get("odds_distortion", 0))
    if ev > 0:
        reasons.append(f"✅ EV+（{ev:+.3f}）：市場が過小評価している可能性")
    elif distortion > 2.0:
        reasons.append(f"✅ オッズ歪み+{distortion:.1f}%：推定勝率がオッズより高い")
    else:
        warnings.append(f"❌ EVマイナス（{ev:+.3f}）：市場評価の方が高い")

    # ---- 根拠②：クラスドロップ ---- #
    class_signal = horse.get("class_signal", "")
    if "クラスドロップ" in str(class_signal):
        reasons.append(f"✅ {class_signal}：能力的に格上の馬が割安で出走")
    elif "クラスアップ" in str(class_signal):
        warnings.append(f"⚠️ {class_signal}：格上挑戦は割引")

    # ---- 根拠③：前走位置取りミス ---- #
    if horse.get("position_mismatch_flag"):
        msg = horse.get("position_correction_msg", "")
        reasons.append(f"✅ 前走展開負け：{msg[:40] if msg else '位置取りミス検出'}")

    # ---- 根拠④：ペース・バイアス適合 ---- #
    pace_b = float(horse.get("pace_benefit", 0))
    bias_b = float(horse.get("realtime_bias_bonus", 0))
    draw_b = float(horse.get("draw_bonus", 0))
    layout_total = pace_b + bias_b + draw_b
    if layout_total > 0.02:
        details = []
        if pace_b > 0:       details.append(f"ペース展開+{pace_b:.3f}")
        if bias_b > 0:       details.append(f"当日バイアス+{bias_b:.3f}")
        if draw_b > 0:       details.append(f"枠順+{draw_b:.3f}")
        reasons.append(f"✅ 展開・コース条件がハマる（{', '.join(details)}）")
    elif layout_total < -0.02:
        warnings.append(f"❌ 展開・コース条件が向かない（合計{layout_total:+.3f}）")

    # ---- 根拠⑤：血統・ニックス適性 ---- #
    nicks_b = float(horse.get("nicks_bonus", 0))
    nicks_label = horse.get("nicks_label", "")
    season_b = float(horse.get("season_bonus", 0))
    cond_b   = float(horse.get("condition_apt_bonus", 0))
    bloodline_total = nicks_b + season_b + cond_b
    if bloodline_total > 0.02:
        parts = []
        if nicks_b > 0.01 and nicks_label: parts.append(nicks_label[:20])
        if season_b > 0:  parts.append(f"季節適性+{season_b:.3f}")
        if cond_b > 0:    parts.append(f"馬場適性+{cond_b:.3f}")
        reasons.append(f"✅ 血統・適性マッチ（{', '.join(parts)}）")
    elif bloodline_total < -0.02:
        warnings.append(f"❌ 血統・適性が合っていない（{bloodline_total:+.3f}）")

    # ---- 根拠⑥：調教タイム ---- #
    training_score = horse.get("training_score")
    training_label = horse.get("training_label", "")
    if training_score is not None:
        if int(training_score) >= 75:
            reasons.append(f"✅ 調教良好：{training_label}")
        elif int(training_score) <= 40:
            warnings.append(f"❌ 調教物足りない：{training_label}")

    # ---- 根拠⑦：騎手乗り替わり ---- #
    jockey_signal = horse.get("jockey_change_signal", "")
    jockey_msg    = horse.get("jockey_change_msg", "")
    if jockey_signal in ("鞍上強化", "手戻り"):
        reasons.append(f"✅ {jockey_signal}：{jockey_msg[:40] if jockey_msg else ''}")
    elif jockey_signal == "鞍上弱化":
        warnings.append(f"❌ {jockey_signal}：{jockey_msg[:40] if jockey_msg else ''}")

    # ---- 前走レースレベル ---- #
    race_level_label = horse.get("race_level_label", "")
    race_level_b     = float(horse.get("race_level_bonus", 0))
    if race_level_b > 0.01:
        reasons.append(f"✅ {race_level_label[:40]}")
    elif race_level_b < -0.01:
        warnings.append(f"⚠️ {race_level_label[:40]}")

    # ---- 判定 ---- #
    n_reasons  = len(reasons)
    n_warnings = len(warnings)

    if n_reasons >= 4:
        verdict       = "◎ 構造的穴馬（買い推奨）"
        verdict_emoji = "🟢"
        romance_risk  = "低（複数の根拠あり）"
    elif n_reasons >= 3:
        verdict       = "○ 根拠ある穴馬（検討推奨）"
        verdict_emoji = "🟡"
        romance_risk  = "中"
    elif n_reasons >= 2:
        verdict       = "△ 弱い根拠の穴馬（慎重に）"
        verdict_emoji = "🟠"
        romance_risk  = "高"
    elif n_reasons >= 1 and ev > -0.1:
        verdict       = "▲ ロマン要素あり（見送りも可）"
        verdict_emoji = "🟠"
        romance_risk  = "高"
    else:
        verdict       = "✕ ロマン買い（根拠なし）"
        verdict_emoji = "🔴"
        romance_risk  = "極高（見送り推奨）"

    # ---- 3行サマリー生成 ---- #
    summary = _generate_summary(
        horse, n_reasons, n_warnings, reasons, warnings, verdict, odds, popularity
    )

    return {
        "is_longshot":       True,
        "structural_count":  n_reasons,
        "warning_count":     n_warnings,
        "romance_risk":      romance_risk,
        "verdict":           verdict,
        "verdict_emoji":     verdict_emoji,
        "reasons":           reasons,
        "warnings":          warnings,
        "summary":           summary,
    }


def _generate_summary(
    horse: dict,
    n_reasons: int,
    n_warnings: int,
    reasons: list,
    warnings: list,
    verdict: str,
    odds: float,
    popularity: int,
) -> str:
    """AIっぽい3行サマリーを生成する。"""
    horse_name = horse.get("horse_name", "この馬")
    lines = [f"**{horse_name}**（{popularity}番人気/{odds}倍）：{verdict}"]

    if reasons:
        top_reason = reasons[0].replace("✅ ", "")
        lines.append(f"最大の根拠：{top_reason[:60]}")
    if warnings:
        top_warning = warnings[0].replace("❌ ", "").replace("⚠️ ", "")
        lines.append(f"注意点：{top_warning[:60]}")
    elif n_reasons >= 3:
        lines.append(f"根拠が{n_reasons}つ重なっており、市場の過小評価が疑われます。")

    return "\n".join(lines)


# ============================================================
# 全馬への適用
# ============================================================

def evaluate_all_longshots(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    evaluate_race() の出力 DataFrame に穴馬判定結果を追加する。
    7番人気以上の馬のみ詳細判定、それ以外は「人気馬」として返す。
    """
    if eval_df.empty:
        return eval_df

    results = []
    for _, row in eval_df.iterrows():
        result = check_structural_reasons(row.to_dict())
        results.append(result)

    result_df = pd.DataFrame(results)

    # DataFrameに結合（dict列は除外）
    simple_cols = ["is_longshot", "structural_count", "warning_count",
                   "romance_risk", "verdict", "verdict_emoji", "summary"]
    out = pd.concat(
        [eval_df.reset_index(drop=True),
         result_df[simple_cols].reset_index(drop=True)],
        axis=1,
    )
    return out


def get_top_structural_longshots(eval_df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    構造的根拠が多い穴馬（7番人気以上）を上位N頭返す。
    当日の穴馬候補として表示するために使う。
    """
    if eval_df.empty or "structural_count" not in eval_df.columns:
        return pd.DataFrame()

    longshots = eval_df[
        (eval_df["popularity"] >= 7) &
        (eval_df["structural_count"] >= 2)
    ].copy()

    if longshots.empty:
        return pd.DataFrame()

    return longshots.nlargest(top_n, "structural_count")
