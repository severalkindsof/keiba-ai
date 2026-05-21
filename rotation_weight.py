"""
馬体重・ローテーション・クラス変動分析モジュール。

分析項目:
1. 馬体重変化（前走比 ±10kg 閾値）
2. レース間隔スコア（3〜5週が理想）
3. 叩き台パターン（凡走後の適切な間隔）
4. クラスドロップ（重賞/G1から格下へ）
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ---- 馬体重変化分析 ---- #

def analyze_weight_change(current_weight: int | None, prev_weight: int | None) -> dict:
    """
    体重変化を評価する。

    Returns:
        signal: "増加懸念" / "減少懸念" / "適正範囲" / "不明"
        bonus: float（勝率への補正値）
        message: str
    """
    if current_weight is None or prev_weight is None:
        return {"signal": "不明", "bonus": 0.0, "message": "体重情報なし（当日要確認）"}

    diff = current_weight - prev_weight

    if diff >= 20:
        return {
            "signal": "大幅増加（要注意）",
            "bonus": -0.025,
            "message": f"前走比 {diff:+d}kg（過大な増加、調整不足の可能性）",
        }
    elif diff >= 10:
        return {
            "signal": "増加",
            "bonus": -0.01,
            "message": f"前走比 {diff:+d}kg（やや増）",
        }
    elif diff >= 4:
        return {
            "signal": "微増",
            "bonus": 0.005,
            "message": f"前走比 {diff:+d}kg（成長・充実の可能性）",
        }
    elif diff >= -4:
        return {
            "signal": "維持",
            "bonus": 0.01,
            "message": f"前走比 {diff:+d}kg（安定）",
        }
    elif diff >= -10:
        return {
            "signal": "微減",
            "bonus": 0.0,
            "message": f"前走比 {diff:+d}kg（仕上がり減の可能性）",
        }
    else:
        return {
            "signal": "大幅減少（要注意）",
            "bonus": -0.025,
            "message": f"前走比 {diff:+d}kg（過大な減少、体調不安の可能性）",
        }


# ---- レース間隔スコア ---- #

def analyze_rotation(
    prev_race_date: str | None,
    current_race_date: str | None,
    running_style: str = "不明",
) -> dict:
    """
    前走からの間隔を評価する。

    理想: 3〜5週間（21〜35日）
    叩き台パターン: 2〜3週 + 前走凡走 → 仕上がり狙い目
    """
    if prev_race_date is None or current_race_date is None:
        return {"signal": "不明", "bonus": 0.0, "days": None, "message": "日付情報なし"}

    try:
        prev = pd.to_datetime(prev_race_date)
        curr = pd.to_datetime(current_race_date)
        days = (curr - prev).days
    except Exception:
        return {"signal": "不明", "bonus": 0.0, "days": None, "message": "日付解析エラー"}

    if days <= 7:
        # 連闘（1週以内）
        bonus = -0.015 if running_style in ("差し・追込", "中団") else -0.005
        return {
            "signal": "連闘",
            "bonus": bonus,
            "days": days,
            "message": f"連闘（{days}日）：差し/追込型には特に注意",
        }
    elif days <= 14:
        return {
            "signal": "中1週",
            "bonus": -0.005,
            "days": days,
            "message": f"中1週（{days}日）：疲労残りに注意",
        }
    elif days <= 35:
        # 理想的なローテーション
        bonus = 0.01 if 21 <= days <= 35 else 0.005
        label = "叩き2走目" if days <= 28 else "標準間隔"
        return {
            "signal": label,
            "bonus": bonus,
            "days": days,
            "message": f"{label}（{days}日）：理想的なローテーション",
        }
    elif days <= 56:
        return {
            "signal": "やや間隔空き",
            "bonus": -0.01,
            "days": days,
            "message": f"やや間隔空き（{days}日）：状態確認推奨",
        }
    elif days <= 90:
        return {
            "signal": "休養明け（叩き1走目）",
            "bonus": -0.03,
            "days": days,
            "message": f"⛔ 休養明け（{days}日ぶり）：叩き1走目、調教内容を要確認",
        }
    else:
        return {
            "signal": "長期休養明け",
            "bonus": -0.05,
            "days": days,
            "message": f"⛔ 長期休養明け（{days}日ぶり）：本番仕上がり未知数",
        }


# ---- 叩き台パターン検出 ---- #

def detect_tatakidai(history: pd.DataFrame, current_race_date: str | None) -> dict:
    """
    前走凡走 + 適切な間隔 = 叩き台から本番パターンを検出する。
    """
    if history.empty or len(history) < 2:
        return {"flag": False, "message": ""}

    prev = history.iloc[0]
    prev2 = history.iloc[1]

    prev_rank = pd.to_numeric(prev.get("rank", 99), errors="coerce")
    prev2_rank = pd.to_numeric(prev2.get("rank", 99), errors="coerce")

    prev_date = prev.get("date")
    rotation = analyze_rotation(str(prev_date) if prev_date else None, current_race_date)
    days = rotation.get("days")

    # 叩き台パターン：前走凡走(5着以下) + 前走から2〜5週 + 前々走が好走(3着以内)
    if (
        pd.notna(prev_rank) and prev_rank >= 5
        and pd.notna(prev2_rank) and prev2_rank <= 3
        and days is not None and 14 <= days <= 42
    ):
        return {
            "flag": True,
            "message": f"叩き台パターン（前走{int(prev_rank)}着 → 前々走{int(prev2_rank)}着）",
            "bonus": 0.02,
        }
    return {"flag": False, "message": "", "bonus": 0.0}


# ---- クラス変動分析 ---- #

CLASS_RANK = {
    "G1": 7, "GI": 7,
    "G2": 6, "GII": 6,
    "G3": 5, "GIII": 5,
    "重賞": 5,
    "オープン": 4, "OP": 4,
    "3勝": 3, "1600万": 3,
    "2勝": 2, "1000万": 2,
    "1勝": 1, "500万": 1,
    "新馬": 0, "未勝利": 0,
}

def get_class_rank(race_class: str) -> int:
    """クラス名からランクを返す（大きいほど格上）"""
    if not race_class:
        return 3
    for key, rank in CLASS_RANK.items():
        if key in str(race_class):
            return rank
    return 3

def analyze_class_change(prev_class: str, current_class: str) -> dict:
    """
    クラス変動を分析する。
    格下げ（クラスドロップ）は穴馬候補として特にプラス。
    """
    prev_rank = get_class_rank(prev_class)
    curr_rank = get_class_rank(current_class)
    drop = prev_rank - curr_rank

    if drop >= 2:
        return {
            "signal": "大幅クラスドロップ",
            "bonus": 0.03,
            "message": f"前走{prev_class} → 今回{current_class}（能力的に格上、穴候補）",
        }
    elif drop == 1:
        return {
            "signal": "クラスドロップ",
            "bonus": 0.015,
            "message": f"前走{prev_class} → 今回{current_class}（格下げ出走）",
        }
    elif drop == 0:
        return {"signal": "同クラス", "bonus": 0.0, "message": ""}
    else:
        return {
            "signal": "クラスアップ",
            "bonus": -0.01,
            "message": f"前走{prev_class} → 今回{current_class}（格上挑戦）",
        }


# ---- 障害叩き → 平地シグナル ---- #

HURDLE_KEYWORDS = ["障害", "ジャンプ", "J・", "J・G"]

def detect_hurdle_to_flat(history: pd.DataFrame) -> dict:
    """
    前走が障害レースで、今回平地に戻る馬を検出する。

    なぜ注目か:
    - 障害レースは距離が長く（2000m以上）、心肺能力・体力が鍛えられる
    - 市場は「障害帰り」を過小評価しがちで、オッズに割安感が出やすい
    - 特に叩き2走目（障害→平地2走目）の好走例が多い

    Returns
    -------
    {
        "prev_was_hurdle":  bool,
        "hurdle_rank":      int | None,  障害での着順
        "bonus":            float,
        "label":            str,
        "message":          str,
    }
    """
    empty = {"prev_was_hurdle": False, "hurdle_rank": None,
             "bonus": 0.0, "label": "", "message": ""}

    if history.empty or "race_name" not in history.columns:
        return empty

    prev = history.iloc[0]
    prev_race_name = str(prev.get("race_name", ""))
    prev_is_hurdle = any(kw in prev_race_name for kw in HURDLE_KEYWORDS)

    if not prev_is_hurdle:
        return empty

    hurdle_rank = pd.to_numeric(prev.get("rank"), errors="coerce")
    hurdle_rank_int = int(hurdle_rank) if not pd.isna(hurdle_rank) else None

    bonus = 0.0
    label = ""
    message = ""

    # 障害で好走（3着以内）→ 体力充実でさらに期待
    if hurdle_rank_int is not None and hurdle_rank_int <= 3:
        bonus   = 0.025
        label   = f"◎ 障害{hurdle_rank_int}着後の平地転戦（体力充実）"
        message = (f"前走は障害レース「{prev_race_name}」で{hurdle_rank_int}着。"
                   f"障害で鍛えられた体力が平地で活きやすい。市場が過小評価する穴候補。")
    # 障害で完走（着外でも）→ スタミナ面は担保、平地では能力発揮の余地
    elif hurdle_rank_int is not None:
        bonus   = 0.015
        label   = f"○ 障害叩き後の平地転戦（スタミナ面◎）"
        message = (f"前走は障害「{prev_race_name}」で{hurdle_rank_int}着。"
                   f"スタミナは十分。平地に戻って見直しの余地あり。")
    else:
        bonus   = 0.01
        label   = "障害叩き後の平地転戦"
        message = f"前走は障害レース「{prev_race_name}」。平地での巻き返しに注目。"

    return {
        "prev_was_hurdle": True,
        "hurdle_rank":     hurdle_rank_int,
        "bonus":           round(bonus, 4),
        "label":           label,
        "message":         message,
    }


# ---- 全馬への適用 ---- #

def analyze_rotation_for_field(
    horses: list[dict],
    df_hist: pd.DataFrame,
    current_race_date: str | None = None,
) -> list[dict]:
    """
    出走馬全頭にローテーション・体重・クラス変動分析を付与する。
    """
    result = []
    for h in horses:
        h2 = dict(h)
        name = h2.get("horse_name", "")

        hist = pd.DataFrame()
        if not df_hist.empty and "horse_name" in df_hist.columns and name:
            hist = df_hist[df_hist["horse_name"].str.strip() == name.strip()]
            if "date" in df_hist.columns:
                hist = hist.sort_values("date", ascending=False)
            hist = hist.head(5)

        # 馬体重
        weight_result = analyze_weight_change(
            h2.get("horse_weight"),
            int(hist.iloc[0].get("horse_weight", 0)) if not hist.empty and "horse_weight" in hist.columns else None,
        )
        h2["weight_signal"] = weight_result["signal"]
        h2["weight_bonus"] = weight_result["bonus"]
        h2["weight_message"] = weight_result["message"]

        # ローテーション
        prev_date = str(hist.iloc[0].get("date", "")) if not hist.empty and "date" in hist.columns else None
        rot_result = analyze_rotation(prev_date, current_race_date, h2.get("running_style", "不明"))
        h2["rotation_signal"] = rot_result["signal"]
        h2["rotation_bonus"] = rot_result["bonus"]
        h2["rotation_days"] = rot_result.get("days")
        h2["rotation_message"] = rot_result["message"]

        # 叩き台パターン
        tataki = detect_tatakidai(hist, current_race_date)
        h2["tatakidai_flag"] = tataki["flag"]
        h2["tatakidai_bonus"] = tataki.get("bonus", 0.0)
        h2["tatakidai_message"] = tataki["message"]

        # クラス変動
        prev_class = str(hist.iloc[0].get("race_class", "")) if not hist.empty and "race_class" in hist.columns else ""
        curr_class = h2.get("race_class", "")
        class_result = analyze_class_change(prev_class, curr_class)
        h2["class_signal"] = class_result["signal"]
        h2["class_bonus"] = class_result["bonus"]
        h2["class_message"] = class_result["message"]

        # 障害叩き → 平地シグナル
        hurdle_result = detect_hurdle_to_flat(hist)
        h2["hurdle_to_flat"]         = hurdle_result["prev_was_hurdle"]
        h2["hurdle_to_flat_label"]   = hurdle_result["label"]
        h2["hurdle_to_flat_bonus"]   = hurdle_result["bonus"]
        h2["hurdle_to_flat_message"] = hurdle_result["message"]

        result.append(h2)
    return result
