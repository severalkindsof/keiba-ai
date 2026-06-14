"""
騎手乗り替わり・手戻り分析モジュール。

パターン:
- 鞍上強化：人気薄なのにリーディング上位騎手への乗り替わり → プラス
- 鞍上弱化：前走の優秀騎手から格下騎手へ → マイナス
- 手戻り：以前乗ったことのある騎手の再登板 → 中〜プラス
- 初騎乗：初コンビ → 中立（相性未確認）
"""
import pandas as pd
import numpy as np

# JRA リーディング上位騎手リスト（2024年シーズン基準）
TOP_JOCKEYS = {
    "C.ルメール", "川田将雅", "横山武史", "戸崎圭太", "松山弘平",
    "福永祐一", "M.デムーロ", "岩田康誠", "中山雄太", "吉田隼人",
    "浜中俊", "和田竜二", "池添謙一", "武豊", "岩田望来",
    "坂井瑠星", "北村友一", "横山和生", "団野大成", "西村淳也",
}

# NEW-5: 海外短期免許騎手（強化評価）
FOREIGN_SHORT_JOCKEYS = [
    "レーン", "ディー", "モレイラ", "ピース", "シュタルケ", "ビュイック",
    "マクドナルド", "テータム", "ヒューズ",
    # 第40波: 主要海外短期を補強（ヴェルテンベルク前走で実在確認した「キング」含む）
    "キング", "ムーア", "マーカンド", "スミヨン", "バルザローナ",
    "Ｃ．デム", "C.デム", "ボウマン", "マーフィー",
]

APPRENTICE_SUFFIX = ("見習い", "▲")  # 見習い騎手の識別


def is_top_jockey(name: str) -> bool:
    # 双方向マッチング: 「戸崎圭太」in「戸崎圭」も「戸崎圭」in「戸崎圭太」もチェック
    return any(top in name or name in top for top in TOP_JOCKEYS)


def analyze_jockey_change(
    horse_name: str,
    current_jockey: str,
    history: pd.DataFrame,
    jockey_stats: pd.DataFrame,
) -> dict:
    """
    騎手乗り替わりを分析して補正値とシグナルを返す。

    Returns dict:
        signal: "強化" / "弱化" / "手戻り" / "初騎乗" / "継続"
        bonus: float（勝率への補正値、-0.03〜+0.03）
        message: str（説明文）
    """
    if not current_jockey or history.empty or "jockey" not in history.columns:
        return {"signal": "不明", "bonus": 0.0, "message": "騎手情報なし"}

    prev_jockey = ""
    if len(history) >= 1:
        prev_jockey = str(history.iloc[0].get("jockey", ""))

    # 過去に乗ったことがあるか（手戻り判定）
    past_jockeys = history["jockey"].dropna().tolist()
    rode_before = current_jockey in past_jockeys[1:] if len(past_jockeys) > 1 else False

    # 継続乗り
    if current_jockey == prev_jockey:
        return {"signal": "継続", "bonus": 0.0, "message": f"{current_jockey}（継続）"}

    # 手戻り
    if rode_before:
        # 手戻り騎手で勝ったことがあるか確認
        past_with_this_jockey = history[history["jockey"] == current_jockey]
        best_rank = past_with_this_jockey["rank"].dropna().min() if "rank" in past_with_this_jockey.columns else 99
        bonus = 0.015 if best_rank <= 3 else 0.008
        return {
            "signal": "手戻り",
            "bonus": bonus,
            "message": f"{current_jockey}（手戻り：以前の最高{int(best_rank) if best_rank < 99 else '?'}着）",
        }

    # 鞍上強化/弱化
    prev_is_top = is_top_jockey(prev_jockey) if prev_jockey else False
    curr_is_top = is_top_jockey(current_jockey)

    # NEW-5: 海外短期免許騎手への乗り替わりは最強化評価
    curr_is_foreign = any(f in current_jockey for f in FOREIGN_SHORT_JOCKEYS)
    if curr_is_foreign:
        return {
            "signal": "外国人強化",
            "bonus": 0.030,
            "message": f"{prev_jockey or '?'} → {current_jockey}（海外短期騎手への乗り替わり）",
        }

    # 第40波: 海外短期 → 日本人テン乗りは「差し戻し」で大幅減点。
    # ユーザー経験則「海外一流は馬の実力を120%引き出す。日本人テン乗りに替わると
    # 100%すら出せない差し戻しが頻発」を実装。
    # ※お手馬戻り(rode_before=True)は上の「手戻り」分岐で先にプラス処理されるため、
    #   ここに到達するのはテン乗り(過去未騎乗)のみ。
    prev_is_foreign = any(f in prev_jockey for f in FOREIGN_SHORT_JOCKEYS) if prev_jockey else False
    if prev_is_foreign and not curr_is_foreign:
        return {
            "signal": "外国人→日本人差し戻し",
            "bonus": -0.030,
            "message": f"{prev_jockey} → {current_jockey}（海外短期→日本人テン乗り：実力差し戻し懸念）",
        }

    if curr_is_top and not prev_is_top:
        return {
            "signal": "鞍上強化",
            "bonus": 0.025,
            "message": f"{prev_jockey or '?'} → {current_jockey}（リーディング上位への強化）",
        }
    elif prev_is_top and not curr_is_top:
        return {
            "signal": "鞍上弱化",
            "bonus": -0.02,
            "message": f"{prev_jockey} → {current_jockey}（鞍上弱化、理由を要確認）",
        }
    else:
        bonus = 0.005 if curr_is_top else 0.0
        return {
            "signal": "初騎乗",
            "bonus": bonus,
            "message": f"{prev_jockey or '?'} → {current_jockey}（新コンビ）",
        }


def infer_jockey_change_reason(
    prev_jockey: str,
    current_jockey: str,
    history: pd.DataFrame,
    current_race_date: str = "",
) -> dict:
    """
    乗り替わりの理由を状況から推定する。

    推定パターン:
    1. 怪我の可能性: 前走から今走まで日数が短い（10日以内）のに変わった
    2. 陣営の意図的強化: トップ騎手への変更
    3. 前走失敗への不満: 前走着外直後にトップ騎手へ変更
    4. ダブルブッキング: 同日に同騎手が別馬に乗っている（理想はDBで確認、ここでは日付ベースで近似）

    Returns
    -------
    {
        "reason":  str,   推定理由
        "confidence": str, "高" | "中" | "低"
        "note":    str,
    }
    """
    empty = {"reason": "不明", "confidence": "低", "note": ""}

    if not prev_jockey or not current_jockey or prev_jockey == current_jockey:
        return empty

    if history.empty or "date" not in history.columns:
        return empty

    # 前走日付を取得
    prev_date = pd.to_datetime(history.iloc[0].get("date"), errors="coerce")
    if pd.isna(prev_date):
        return empty

    curr_date = pd.to_datetime(current_race_date, errors="coerce") if current_race_date else None

    days_since = (curr_date - prev_date).days if curr_date and not pd.isna(curr_date) else None

    # 前走着順
    prev_rank = pd.to_numeric(history.iloc[0].get("rank"), errors="coerce")

    curr_is_top  = is_top_jockey(current_jockey)
    prev_is_top  = is_top_jockey(prev_jockey)

    # パターン別判定
    if curr_is_top and not prev_is_top:
        if not pd.isna(prev_rank) and prev_rank >= 6:
            return {
                "reason": "前走凡走後の意図的強化",
                "confidence": "高",
                "note": f"前走{int(prev_rank)}着後にリーディング騎手へ変更。陣営の本気度が高い。",
            }
        return {
            "reason": "陣営の意図的強化",
            "confidence": "中",
            "note": "リーディング騎手への変更。陣営が積極的なサポートを求めている。",
        }

    if days_since is not None and days_since <= 10 and not curr_is_top:
        return {
            "reason": "怪我の可能性（短期間での変更）",
            "confidence": "中",
            "note": f"前走から{days_since}日という短期間での乗り替わり。騎手の体調不良・怪我の可能性あり。",
        }

    if prev_is_top and not curr_is_top:
        return {
            "reason": "鞍上弱化（理由不明）",
            "confidence": "中",
            "note": "トップ騎手から格下へ。ダブルブッキングか本馬への評価が下がった可能性。",
        }

    return {
        "reason": "不明（新コンビ）",
        "confidence": "低",
        "note": "乗り替わり理由は不明。初コンビの相性に注目。",
    }


def get_jockey_change_for_field(
    horses: list[dict],
    df_hist: pd.DataFrame,
    jockey_stats: pd.DataFrame,
) -> list[dict]:
    """
    出走馬全頭の乗り替わり情報を付与して返す。
    各馬dictに 'jockey_change_signal', 'jockey_change_bonus', 'jockey_change_msg' を追加。
    """
    result = []
    for h in horses:
        h2 = dict(h)
        name = h2.get("horse_name", "")
        current_jockey = h2.get("jockey", "")

        if not df_hist.empty and "horse_name" in df_hist.columns and name:
            # A-4: strip()で空白/encoding差異による不一致を防ぐ
            hist = df_hist[df_hist["horse_name"].str.strip() == name.strip()]
            if "date" in df_hist.columns:
                hist = hist.sort_values("date", ascending=False)
            hist = hist.head(10)
        else:
            hist = pd.DataFrame()

        analysis = analyze_jockey_change(name, current_jockey, hist, jockey_stats)
        h2["jockey_change_signal"] = analysis["signal"]
        h2["jockey_change_bonus"]  = analysis["bonus"]
        h2["jockey_change_msg"]    = analysis["message"]

        # 乗り替わり理由推定
        if analysis["signal"] not in ("継続", "不明"):
            prev_j = str(hist.iloc[0].get("jockey", "")) if not hist.empty else ""
            reason = infer_jockey_change_reason(
                prev_j, current_jockey, hist,
                current_race_date=h2.get("race_date", ""),
            )
            h2["jockey_change_reason"]     = reason["reason"]
            h2["jockey_change_reason_note"] = reason["note"]
        else:
            h2["jockey_change_reason"]      = ""
            h2["jockey_change_reason_note"] = ""

        result.append(h2)
    return result
