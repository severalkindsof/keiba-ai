"""
ペース・展開予想エンジン。

ロジック:
1. 出走馬の脚質（逃げ/先行/中団/差し追込）をカウント
2. 逃げ馬の頭数 → ペース予測（ハイ/ミドル/スロー）
3. 今回のペース × 各馬の脚質 → 展開恩恵スコア（+/-）
4. 前走ハイペース消耗馬の巻き返しフラグ
"""
import pandas as pd
import numpy as np


# ---- 脚質判定 ---- #

STYLE_MAP = {
    "逃げ": 1,
    "先行": 2,
    "中団": 3,
    "差し": 4,
    "追込": 5,
}

def infer_running_style(history: pd.DataFrame) -> str:
    """過去成績のコーナー通過順から脚質を推定"""
    if history.empty:
        return "不明"
    # TFJVデータはcorner4列、旧データはcorner_order列を使う
    for col in ["corner4", "corner_order"]:
        if col not in history.columns:
            continue
        try:
            if col == "corner4":
                # TFJV "5=3" 形式に対応: str.extract で先頭数字のみ取得（train_lgbm.pyと統一）
                _parsed = history[col].astype(str).str.extract(r"^(\d+)", expand=False)
                positions = pd.to_numeric(_parsed, errors="coerce").dropna().head(5).tolist()
            else:
                positions = []
                for val in history[col].dropna().head(5):
                    first = str(val).split("-")[0].split(",")[0].strip()
                    if first.isdigit():
                        positions.append(int(first))
            if not positions:
                continue
            # B-2: 2件以上のデータがある場合のみ判定（1件では誤判定が多い）
            if len(positions) < 2:
                continue
            # 直近2走を重視した加重平均
            weights = [2 if i < 2 else 1 for i in range(len(positions))]
            avg = np.average(positions[:len(weights)], weights=weights[:len(positions)])
            field_series = pd.to_numeric(history.get("field_size", pd.Series([16]*len(history))), errors="coerce")
            field = field_series.mean() if pd.notna(field_series.mean()) and field_series.mean() > 0 else 16
            ratio = avg / field
            # LATENT-6: 閾値を実態に合わせて修正（逃げ=ほぼ最前列）
            if ratio <= 0.08:
                return "逃げ"
            elif ratio <= 0.28:
                return "先行"
            elif ratio <= 0.60:
                return "中団"
            else:
                return "差し・追込"
        except Exception:
            continue
    # コーナーなければ上がり3Fで推定
    # B-2: TFJVのlast_3fは0.1秒単位（347 = 34.7秒）→ 自動判定して秒に変換
    if "last_3f" in history.columns:
        avg3f = pd.to_numeric(history["last_3f"], errors="coerce").mean()
        if pd.notna(avg3f):
            avg3f_sec = avg3f / 10 if avg3f > 100 else avg3f  # 100以上なら0.1秒単位
            if avg3f_sec <= 34.0:
                return "差し・追込"
            elif avg3f_sec <= 35.5:
                return "中団"
            else:
                return "先行"
    return "不明"


# ---- ペース予測 ---- #

def predict_pace(horses: list[dict]) -> dict:
    """
    出走馬リストからペースを予測する。

    horses: 各dictに 'running_style'（逃げ/先行/中団/差し・追込/不明）が入っている想定。
    """
    front_runners  = sum(1 for h in horses if h.get("running_style", "") == "逃げ")
    leaders_only   = sum(1 for h in horses if h.get("running_style", "") == "先行")
    leaders        = front_runners + leaders_only   # 逃げ+先行（ペース判定用）

    if front_runners >= 3:
        pace = "ハイペース"
        pace_score = 3
    elif front_runners == 2:
        pace = "ミドル〜ハイ"
        pace_score = 2
    elif front_runners == 1:
        pace = "ミドル"
        pace_score = 1
    else:
        # 逃げ馬なし = 先行馬がペースを作るためスロー傾向
        if leaders >= 4:
            pace = "ミドル"
            pace_score = 1
        else:
            pace = "スローペース"
            pace_score = 0

    return {
        "predicted_pace": pace,
        "pace_score": pace_score,  # 0=スロー, 1=ミドル, 2=ミドルハイ, 3=ハイ
        "front_runner_count": front_runners,
        "leader_count": leaders_only,
        "summary": f"逃げ{front_runners}頭 先行{leaders_only}頭 → {pace}予想",
    }


# ---- 展開恩恵スコア ---- #

PACE_STYLE_BENEFIT = {
    # pace_score → {脚質: スコア補正}
    3: {"逃げ": -2, "先行": -1, "中団": +1, "差し・追込": +2, "不明": 0},   # ハイ
    2: {"逃げ": -1, "先行":  0, "中団":  0, "差し・追込": +1, "不明": 0},   # ミドルハイ
    1: {"逃げ": +1, "先行": +1, "中団":  0, "差し・追込":  0, "不明": 0},   # ミドル
    0: {"逃げ": +2, "先行": +2, "中団": +1, "差し・追込": -1, "不明": 0},   # スロー
}

def calc_pace_benefit(running_style: str, pace_score: int) -> float:
    """
    ペース × 脚質の展開恩恵スコア。
    -2〜+2 の整数を 0.02〜0.05 の勝率補正値に変換。
    """
    raw = PACE_STYLE_BENEFIT.get(pace_score, {}).get(running_style, 0)
    return raw * 0.015  # 1段階あたり約1.5%の勝率補正


# ---- 前走消耗フラグ ---- #

def detect_prev_race_exhaustion(history: pd.DataFrame, current_pace_score: int) -> dict:
    """
    前走がハイペースで消耗した差し/追込馬かどうかを検出する。
    今回スロー〜ミドルなら「巻き返し候補」フラグを立てる。
    """
    if history.empty or len(history) < 1:
        return {"flag": False, "message": ""}

    prev = history.iloc[0]  # 直近1走
    prev_rank = pd.to_numeric(prev.get("rank", 99), errors="coerce")
    if pd.isna(prev_rank):
        prev_rank = 99

    # 前走着順が6着以下で、且つ今回のペース予測がスロー/ミドルなら巻き返し候補
    if prev_rank >= 6 and current_pace_score <= 1:
        return {
            "flag": True,
            "message": f"前走{int(prev_rank)}着（消耗の可能性）→ 今回スロー/ミドル予想で巻き返し候補",
        }
    return {"flag": False, "message": ""}


# ---- 全馬の展開スコア計算 ---- #

def analyze_field_pace(
    horses: list[dict],
    df_hist: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    """
    出走馬全頭の脚質を推定してペース予測し、
    各馬の展開恩恵スコアを付与して返す。

    Returns:
        pace_info: dict（ペース予測結果）
        horses_with_pace: list[dict]（各馬にrunning_style, pace_benefit, exhaustion_flagを追加）
    """
    horses_out = []
    for h in horses:
        h2 = dict(h)
        name = h2.get("horse_name", "")
        if not h2.get("running_style") and not df_hist.empty and "horse_name" in df_hist.columns:
            hist = df_hist[df_hist["horse_name"].str.strip() == name.strip()]
            if "date" in df_hist.columns:
                hist = hist.sort_values("date", ascending=False)
            hist = hist.head(5)
            h2["running_style"] = infer_running_style(hist)
        horses_out.append(h2)

    pace_info = predict_pace(horses_out)
    pace_score = pace_info["pace_score"]

    for h2 in horses_out:
        style = h2.get("running_style", "不明")
        h2["pace_benefit"] = calc_pace_benefit(style, pace_score)

        # 前走消耗フラグ
        if not df_hist.empty and "horse_name" in df_hist.columns:
            hist = df_hist[df_hist["horse_name"] == h2.get("horse_name", "")].head(3)
            exhaustion = detect_prev_race_exhaustion(hist, pace_score)
            h2["exhaustion_comeback"] = exhaustion["flag"]
            h2["exhaustion_message"] = exhaustion["message"]
        else:
            h2["exhaustion_comeback"] = False
            h2["exhaustion_message"] = ""

    return pace_info, horses_out
