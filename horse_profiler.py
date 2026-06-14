"""
馬別の多角分析プロファイル。
- 直近5走の成績
- コース・距離適性スコア
- 上がり3ハロン傾向（末脚型 vs 先行型）
- 血統スコア
- 馬体重トレンド（NEW）
- コーナー通過順位適性（NEW）
- 時計ランク比較（NEW）
- 有力馬撃破スコア（NEW）
"""
import pandas as pd
import numpy as np


def get_horse_history(df: pd.DataFrame, horse_name: str, n: int = 5) -> pd.DataFrame:
    """過去n走のレース成績を返す"""
    if "horse_name" not in df.columns or df.empty:
        return pd.DataFrame()
    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=False)
    return hist.head(n)


def calc_distance_aptitude(df: pd.DataFrame, horse_name: str, distance: int) -> dict:
    """
    距離適性スコアを算出。
    指定距離帯でのレース数と勝率・複勝率。
    """
    if df.empty or "horse_name" not in df.columns:
        return {"score": 50, "detail": "データなし"}

    from data_loader import categorize_distance
    dist_cat = categorize_distance(distance)
    hist = df[df["horse_name"].str.strip() == horse_name.strip()]

    if hist.empty:
        return {"score": 50, "detail": "出走履歴なし"}

    dist_hist = hist[hist.get("distance_cat", pd.Series(dtype=str)) == dist_cat] if "distance_cat" in hist.columns else hist
    total = len(dist_hist)
    if total == 0:
        return {"score": 50, "detail": f"{dist_cat}での出走なし"}

    wins = int(dist_hist["win_flag"].sum()) if "win_flag" in dist_hist.columns else 0
    places = int(dist_hist["place_flag"].sum()) if "place_flag" in dist_hist.columns else 0

    win_rate = wins / total
    place_rate = places / total
    score = int(win_rate * 60 + place_rate * 40)  # 最大100点

    return {
        "score": min(100, score),
        "races": total,
        "wins": wins,
        "places": places,
        "win_rate": round(win_rate * 100, 1),
        "place_rate": round(place_rate * 100, 1),
        "detail": f"{dist_cat}：{total}走{wins}勝（勝率{win_rate*100:.0f}%）",
    }


def calc_surface_aptitude(df: pd.DataFrame, horse_name: str, surface: str) -> dict:
    """馬場（芝/ダート）適性スコア"""
    if df.empty or "horse_name" not in df.columns or "surface" not in df.columns:
        return {"score": 50, "detail": "データなし"}

    hist = df[(df["horse_name"].str.strip() == horse_name.strip()) & (df["surface"] == surface)]
    total = len(hist)
    if total == 0:
        return {"score": 50, "detail": f"{surface}での出走なし"}

    wins = int(hist["win_flag"].sum()) if "win_flag" in hist.columns else 0
    places = int(hist["place_flag"].sum()) if "place_flag" in hist.columns else 0
    win_rate = wins / total
    place_rate = places / total
    score = int(win_rate * 60 + place_rate * 40)

    return {
        "score": min(100, score),
        "races": total,
        "wins": wins,
        "places": places,
        "win_rate": round(win_rate * 100, 1),
        "place_rate": round(place_rate * 100, 1),
        "detail": f"{surface}：{total}走{wins}勝",
    }


def calc_running_style(df: pd.DataFrame, horse_name: str) -> str:
    """
    上がり3ハロンタイムとコーナー通過順から脚質を判定。
    フォールバック：判定不能なら「不明」
    """
    if df.empty or "horse_name" not in df.columns:
        return "不明"
    hist = df[df["horse_name"].str.strip() == horse_name.strip()]
    if hist.empty:
        return "不明"

    # TFJVデータのcorner4（4コーナー位置）から判定
    for col in ["corner4", "corner_order"]:
        if col in hist.columns:
            try:
                if col == "corner_order":
                    pos = hist[col].dropna().apply(
                        lambda x: int(str(x).split("-")[-1]) if "-" in str(x)
                        else int(str(x).split(",")[-1]) if "," in str(x)
                        else int(str(x).strip()) if str(x).strip().isdigit() else None
                    ).dropna()
                else:
                    # TFJV "5=3" 形式に対応: str.extract で先頭数字のみ取得（pace_analyzer.pyと統一）
                    _parsed = hist[col].astype(str).str.extract(r"^(\d+)", expand=False)
                    pos = pd.to_numeric(_parsed, errors="coerce").dropna()

                if len(pos) == 0:
                    continue
                avg_pos = pos.mean()
                # 頭数比率で判定（絶対位置でなく相対位置）
                total = pd.to_numeric(hist.get("field_size", pd.Series(16)), errors="coerce").mean()
                total = total if pd.notna(total) and total > 0 else 16
                ratio = avg_pos / total
                if ratio <= 0.25:
                    return "逃げ・先行"
                elif ratio <= 0.5:
                    return "先行"
                elif ratio <= 0.70:
                    return "中団"
                else:
                    return "差し・追込"
            except Exception:
                continue

    if "last_3f" in hist.columns:
        avg_3f = pd.to_numeric(hist["last_3f"], errors="coerce").mean()
        if pd.notna(avg_3f):
            if avg_3f <= 34.5:
                return "差し・追込（末脚型）"
            elif avg_3f <= 36.0:
                return "中団（バランス型）"
            else:
                return "先行（持続力型）"

    return "不明"


def get_recent_form_score(history: pd.DataFrame) -> int:
    """
    直近5走の着順から状態スコアを算出（0〜100点）。
    1着=20点、2着=15点、3着=10点、4着=5点、5着以下=0点
    """
    if history.empty or "rank" not in history.columns:
        return 50
    scores = history["rank"].dropna().apply(
        lambda r: 20 if r == 1 else 15 if r == 2 else 10 if r == 3 else 5 if r <= 5 else 0
    )
    return min(100, int(scores.sum()))


def calc_venue_distance_aptitude(
    df: pd.DataFrame, horse_name: str, venue: str, surface: str, distance: int
) -> dict:
    """
    同会場×同距離に絞った過去成績。より精緻なコース適性判定。
    """
    if df.empty or "horse_name" not in df.columns:
        return {"score": 50, "detail": "データなし", "races": 0}

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if hist.empty:
        return {"score": 50, "detail": "出走履歴なし", "races": 0}

    # フィルター: 会場 × 距離帯（±200m）× 馬場
    from data_loader import categorize_distance
    dist_cat = categorize_distance(distance)

    filtered = hist.copy()
    if "venue" in filtered.columns:
        filtered = filtered[filtered["venue"] == venue]
    if "surface" in filtered.columns:
        filtered = filtered[filtered["surface"] == surface]
    if "distance_cat" in filtered.columns:
        filtered = filtered[filtered["distance_cat"] == dist_cat]
    elif "distance" in filtered.columns:
        filtered = filtered[
            (pd.to_numeric(filtered["distance"], errors="coerce") - distance).abs() <= 200
        ]

    total = len(filtered)
    if total == 0:
        # 距離帯のみ（会場不問）にフォールバック
        fb = hist.copy()
        if "surface" in fb.columns:
            fb = fb[fb["surface"] == surface]
        if "distance_cat" in fb.columns:
            fb = fb[fb["distance_cat"] == dist_cat]
        fb_total = len(fb)
        if fb_total == 0:
            return {"score": 50, "detail": f"{venue}×{surface}{distance}mの実績なし", "races": 0}
        wins = int(fb["win_flag"].sum()) if "win_flag" in fb.columns else 0
        places = int(fb["place_flag"].sum()) if "place_flag" in fb.columns else 0
        wr = wins / fb_total
        pr = places / fb_total
        score = int(wr * 60 + pr * 40)
        return {
            "score": min(100, score),
            "races": fb_total,
            "wins": wins,
            "detail": f"{venue}実績なし → {surface}{dist_cat}全体: {fb_total}走{wins}勝",
            "is_venue_exact": False,
        }

    wins = int(filtered["win_flag"].sum()) if "win_flag" in filtered.columns else 0
    places = int(filtered["place_flag"].sum()) if "place_flag" in filtered.columns else 0
    wr = wins / total
    pr = places / total
    score = int(wr * 60 + pr * 40)

    return {
        "score": min(100, score),
        "races": total,
        "wins": wins,
        "places": places,
        "win_rate": round(wr * 100, 1),
        "place_rate": round(pr * 100, 1),
        "detail": f"{venue}×{surface}{distance}m: {total}走{wins}勝（複{int(pr*100)}%）",
        "is_venue_exact": True,
    }


def calc_last3f_rank(
    df: pd.DataFrame,
    horse_name: str,
    surface: str,
    distance: int,
    n_races: int = 5,
) -> dict:
    """
    前走上がり3F を全出走馬と比較して順位付け。
    さらに同条件（馬場×距離帯）での上がり3F平均も返す。

    Returns
    -------
    {
        "last_3f":       float | None,   直近の上がり3F
        "avg_3f":        float | None,   同条件での自身の平均
        "race_avg_3f":   float | None,   同条件全馬の平均
        "is_fast":       bool,           自身の平均 < 全体平均
        "label":         str,
        "bonus":         float,
    }
    """
    from data_loader import categorize_distance
    dist_cat = categorize_distance(distance)

    if df.empty or "last_3f" not in df.columns:
        return {"last_3f": None, "avg_3f": None, "race_avg_3f": None,
                "is_fast": False, "label": "データなし", "bonus": 0.0}

    # 直近N走の自身の上がり3F
    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=False)
    recent = hist.head(n_races)

    last_3f_raw = recent["last_3f"].iloc[0] if not recent.empty else None
    last_3f = float(pd.to_numeric(last_3f_raw, errors="coerce")) if last_3f_raw is not None else None

    # 同条件（馬場×距離帯）での自身の平均
    cond_hist = hist.copy()
    if "surface" in cond_hist.columns:
        cond_hist = cond_hist[cond_hist["surface"] == surface]
    if "distance_cat" in cond_hist.columns:
        cond_hist = cond_hist[cond_hist["distance_cat"] == dist_cat]
    avg_3f_series = pd.to_numeric(cond_hist["last_3f"], errors="coerce").dropna()
    avg_3f = float(avg_3f_series.mean()) if len(avg_3f_series) >= 2 else None

    # 同条件の全馬の上がり3F平均（レースレベル参照）
    all_cond = df.copy()
    if "surface" in all_cond.columns:
        all_cond = all_cond[all_cond["surface"] == surface]
    if "distance_cat" in all_cond.columns:
        all_cond = all_cond[all_cond["distance_cat"] == dist_cat]
    all_3f_series = pd.to_numeric(all_cond["last_3f"], errors="coerce").dropna()
    race_avg = float(all_3f_series.mean()) if len(all_3f_series) >= 10 else None

    bonus = 0.0
    label = ""
    is_fast = False

    if avg_3f is not None and race_avg is not None:
        diff = avg_3f - race_avg  # マイナス = 速い
        is_fast = diff < -0.5
        if diff <= -1.5:
            label = f"末脚◎（平均比{abs(diff):.1f}秒速い）"
            bonus = 0.02
        elif diff <= -0.5:
            label = f"末脚○（平均比{abs(diff):.1f}秒速い）"
            bonus = 0.01
        elif diff >= 1.5:
            label = f"末脚△（平均比{abs(diff):.1f}秒遅い）"
            bonus = -0.01
        else:
            label = "末脚標準"
    elif last_3f is not None:
        label = f"前走上がり: {last_3f:.1f}秒"

    return {
        "last_3f": last_3f,
        "avg_3f": round(avg_3f, 2) if avg_3f else None,
        "race_avg_3f": round(race_avg, 2) if race_avg else None,
        "is_fast": is_fast,
        "label": label,
        "bonus": bonus,
    }


def build_horse_profile(
    df: pd.DataFrame,
    horse_name: str,
    surface: str,
    distance: int,
    sire: str = "",
    jockey: str = "",
    venue: str = "",
) -> dict:
    """1頭の総合プロファイルを構築"""
    history = get_horse_history(df, horse_name)
    dist_apt = calc_distance_aptitude(df, horse_name, distance)
    surf_apt = calc_surface_aptitude(df, horse_name, surface)
    venue_apt = calc_venue_distance_aptitude(df, horse_name, venue, surface, distance)
    last3f_info = calc_last3f_rank(df, horse_name, surface, distance)
    running_style = calc_running_style(df, horse_name)
    form_score = get_recent_form_score(history)

    # 総合適性スコア（0〜100）
    overall = int(
        dist_apt.get("score", 50) * 0.25
        + surf_apt.get("score", 50) * 0.25
        + venue_apt.get("score", 50) * 0.20   # 同会場実績を加味
        + form_score * 0.30
    )

    return {
        "horse_name": horse_name,
        "sire": sire,
        "jockey": jockey,
        "running_style": running_style,
        "form_score": form_score,
        "distance_aptitude": dist_apt,
        "surface_aptitude": surf_apt,
        "venue_distance_aptitude": venue_apt,
        "last3f_info": last3f_info,
        "overall_score": overall,
        "history": history,
    }


# ============================================================
# 2. 馬体重トレンド
# ============================================================

def get_weight_trend(df: pd.DataFrame, horse_name: str, n: int = 10) -> dict:
    """
    過去N走の馬体重推移を返す。
    増減傾向（線形回帰）と直近変化量を評価する。

    Returns
    -------
    {
        "weights": list[int],       日付順（古→新）
        "dates":   list[str],
        "trend":   "増加" | "減少" | "安定" | "不明",
        "slope":   float,           kg/走
        "last_change": int | None,  直近変化（前走比）
        "label":   str,
        "bonus":   float,
    }
    """
    if df.empty or "horse_name" not in df.columns:
        return {"weights": [], "dates": [], "trend": "不明", "slope": 0.0,
                "last_change": None, "label": "体重データなし", "bonus": 0.0}

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=True)

    if "horse_weight" not in hist.columns:
        return {"weights": [], "dates": [], "trend": "不明", "slope": 0.0,
                "last_change": None, "label": "体重列なし", "bonus": 0.0}

    hist = hist.tail(n)
    wt_series = pd.to_numeric(hist["horse_weight"], errors="coerce").dropna()
    dt_series = hist["date"].astype(str).iloc[-len(wt_series):] if "date" in hist.columns else []

    if len(wt_series) < 2:
        return {"weights": list(wt_series), "dates": list(dt_series),
                "trend": "不明", "slope": 0.0, "last_change": None,
                "label": "体重データ不足", "bonus": 0.0}

    weights = list(wt_series.astype(int))
    dates   = list(dt_series)

    # 線形回帰でトレンド計算
    x = np.arange(len(weights))
    slope = float(np.polyfit(x, weights, 1)[0])  # kg/走

    last_change = weights[-1] - weights[-2]

    # 評価
    if abs(slope) < 1.0:
        trend = "安定"
    elif slope > 1.0:
        trend = "増加傾向"
    else:
        trend = "減少傾向"

    bonus = 0.0
    label = f"体重{trend}（{slope:+.1f}kg/走）"

    # 急激な変化への警告
    if abs(last_change) >= 14:
        label += f" 前走比{last_change:+}kg（急変動）"
        bonus = -0.01
    elif last_change >= 6:
        label += f" 前走比{last_change:+}kg（増）"
        bonus = -0.005
    elif last_change <= -6:
        label += f" 前走比{last_change:+}kg（絞れた）"
        bonus = 0.005

    return {
        "weights": weights,
        "dates": dates,
        "trend": trend,
        "slope": round(slope, 2),
        "last_change": last_change,
        "label": label,
        "bonus": round(bonus, 4),
    }


# ============================================================
# 5. コーナー通過順位適性（詳細版）
# ============================================================

def get_corner_position_stats(
    df: pd.DataFrame,
    horse_name: str,
    surface: str = "",
    distance: int = 0,
) -> dict:
    """
    コーナー通過順位の統計を返す。
    勝ち試合・負け試合での平均ポジションを比較して「得意なポジション」を特定。

    Returns
    -------
    {
        "avg_1st_corner":  float | None,  1コーナー平均通過順
        "win_position":    float | None,  勝ち時の平均ポジション
        "lose_position":   float | None,  負け時の平均ポジション
        "preferred_style": str,           逃げ/先行/中団/差し
        "position_shift":  float,         勝ち時-負け時（マイナス=先行で強い）
        "label":           str,
    }
    """
    empty = {"avg_1st_corner": None, "win_position": None, "lose_position": None,
             "preferred_style": "不明", "position_shift": 0.0, "label": "コーナーデータなし"}

    if df.empty or "horse_name" not in df.columns or "corner_order" not in df.columns:
        return empty

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if surface and "surface" in hist.columns:
        hist = hist[hist["surface"] == surface]

    if hist.empty:
        return empty

    def _parse_first_corner(val):
        try:
            s = str(val)
            first = s.split("-")[0] if "-" in s else s.split(",")[0]
            return int(first.strip())
        except Exception:
            return np.nan

    hist["_pos"] = hist["corner_order"].apply(_parse_first_corner)
    hist = hist.dropna(subset=["_pos"])

    if hist.empty:
        return empty

    avg_pos = float(hist["_pos"].mean())

    # 勝ち/負けでの位置比較
    win_hist  = hist[hist["win_flag"] == 1] if "win_flag" in hist.columns else pd.DataFrame()
    lose_hist = hist[hist["win_flag"] == 0] if "win_flag" in hist.columns else hist

    win_pos  = float(win_hist["_pos"].mean())  if not win_hist.empty  else None
    lose_pos = float(lose_hist["_pos"].mean()) if not lose_hist.empty else None

    # 得意スタイル判定
    if avg_pos <= 2.5:
        style = "逃げ"
    elif avg_pos <= 5:
        style = "先行"
    elif avg_pos <= 9:
        style = "中団"
    else:
        style = "差し・追込"

    shift = (win_pos - lose_pos) if (win_pos and lose_pos) else 0.0
    shift_note = ""
    if shift < -2:
        shift_note = "（先行した時に強い）"
    elif shift > 2:
        shift_note = "（差してきた時に強い）"

    label = f"平均{avg_pos:.1f}番手（{style}）{shift_note}"

    return {
        "avg_1st_corner": round(avg_pos, 1),
        "win_position":   round(win_pos, 1) if win_pos else None,
        "lose_position":  round(lose_pos, 1) if lose_pos else None,
        "preferred_style": style,
        "position_shift": round(shift, 2),
        "label": label,
    }


# ============================================================
# 8. 時計ランク比較（レースレベル補正済み）
# ============================================================

# JRAコース基準タイム（代表的なコース）
COURSE_PAR_TIMES = {
    ("東京", "芝",   1400): 82.5,
    ("東京", "芝",   1600): 94.5,
    ("東京", "芝",   1800): 107.0,
    ("東京", "芝",   2000): 119.5,
    ("東京", "芝",   2400): 143.5,
    ("東京", "ダート", 1300): 79.5,
    ("東京", "ダート", 1400): 84.5,
    ("東京", "ダート", 1600): 97.5,
    ("東京", "ダート", 2100): 130.0,
    ("中山", "芝",   1600): 95.5,
    ("中山", "芝",   1800): 109.0,
    ("中山", "芝",   2000): 121.5,
    ("中山", "芝",   2200): 133.0,
    ("中山", "ダート", 1200): 73.0,
    ("中山", "ダート", 1800): 111.0,
    ("阪神", "芝",   1400): 83.0,
    ("阪神", "芝",   1600): 95.0,
    ("阪神", "芝",   2000): 120.0,
    ("阪神", "ダート", 1200): 73.5,
    ("阪神", "ダート", 1800): 110.5,
    ("京都", "芝",   1600): 94.5,
    ("京都", "芝",   2000): 120.0,
    ("京都", "芝",   3000): 183.0,
    ("中京", "芝",   1200): 69.5,
    ("中京", "芝",   2000): 120.5,
    ("中京", "ダート", 1800): 111.0,
    ("札幌", "芝",   1800): 110.5,
    ("函館", "芝",   1800): 110.0,
    ("小倉", "芝",   1200): 69.0,
    ("福島", "芝",   1800): 110.0,
    ("新潟", "芝",   1600): 95.0,
}


def _parse_time_to_seconds(time_str) -> float | None:
    """'1:34.5' または '94.5' を秒に変換"""
    if not time_str or pd.isna(time_str):
        return None
    s = str(time_str).strip()
    try:
        if ":" in s:
            parts = s.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(s)
    except Exception:
        return None


def calc_time_rank(
    df: pd.DataFrame,
    horse_name: str,
    venue: str,
    surface: str,
    distance: int,
    n_races: int = 5,
) -> dict:
    """
    過去N走のタイムをコース基準タイムで補正してランク付けする。
    異なるコースでの比較を可能にする。

    Returns
    -------
    {
        "best_time_raw":   float | None,   自己最高タイム（秒）
        "best_time_adj":   float | None,   基準タイム補正後
        "par_time":        float | None,   コース基準タイム
        "time_rank":       str,            "S" | "A" | "B" | "C" | "D"
        "time_rank_bonus": float,
        "label":           str,
        "races_used":      int,
    }
    """
    from data_loader import categorize_distance
    empty = {"best_time_raw": None, "best_time_adj": None, "par_time": None,
             "time_rank": "D", "time_rank_bonus": 0.0, "label": "タイムデータなし", "races_used": 0}

    if df.empty or "horse_name" not in df.columns:
        return empty

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if hist.empty:
        return empty

    # タイム列の候補
    time_col = None
    for col in ["time", "finish_time", "race_time"]:
        if col in hist.columns:
            time_col = col
            break
    if time_col is None:
        return empty

    # 同条件（馬場×距離帯）に絞る
    dist_cat = categorize_distance(distance)
    cond_hist = hist.copy()
    if "surface" in cond_hist.columns:
        cond_hist = cond_hist[cond_hist["surface"] == surface]
    if "distance" in cond_hist.columns:
        cond_hist = cond_hist[
            (pd.to_numeric(cond_hist["distance"], errors="coerce") - distance).abs() <= 200
        ]

    if "date" in cond_hist.columns:
        cond_hist = cond_hist.sort_values("date", ascending=False)
    cond_hist = cond_hist.head(n_races)

    times = cond_hist[time_col].apply(_parse_time_to_seconds).dropna()
    if times.empty:
        # 距離帯を広げてフォールバック
        times = hist[time_col].apply(_parse_time_to_seconds).dropna()
        if times.empty:
            return empty

    best_raw = float(times.min())

    # コース基準タイムを取得（完全一致 → 最近傍距離の順）
    par = COURSE_PAR_TIMES.get((venue, surface, distance))
    if par is None:
        # 最近傍距離でマッチ
        best_dist_diff = 999
        for (v, s, d), t in COURSE_PAR_TIMES.items():
            if v == venue and s == surface and abs(d - distance) < best_dist_diff:
                best_dist_diff = abs(d - distance)
                par = t if best_dist_diff <= 400 else None
    if par is None:
        # 馬場×距離帯の全馬平均で代用
        all_cond = df.copy()
        if "surface" in all_cond.columns:
            all_cond = all_cond[all_cond["surface"] == surface]
        if "distance" in all_cond.columns:
            all_cond = all_cond[
                (pd.to_numeric(all_cond["distance"], errors="coerce") - distance).abs() <= 200
            ]
        all_times = all_cond[time_col].apply(_parse_time_to_seconds).dropna() if time_col in all_cond else pd.Series()
        par = float(all_times.median()) if not all_times.empty else best_raw

    # 補正後タイム（基準タイムからの差：マイナスが速い）
    adj = best_raw - par

    # ランク付け（秒差）
    if adj <= -1.5:
        rank, bonus = "S", 0.025
        label = f"時計S級（基準比{adj:+.2f}秒）"
    elif adj <= -0.5:
        rank, bonus = "A", 0.015
        label = f"時計A（基準比{adj:+.2f}秒）"
    elif adj <= 0.5:
        rank, bonus = "B", 0.005
        label = f"時計B標準（基準比{adj:+.2f}秒）"
    elif adj <= 2.0:
        rank, bonus = "C", -0.005
        label = f"時計C（基準比{adj:+.2f}秒）"
    else:
        rank, bonus = "D", -0.015
        label = f"時計D低（基準比{adj:+.2f}秒）"

    return {
        "best_time_raw":   round(best_raw, 2),
        "best_time_adj":   round(adj, 3),
        "par_time":        par,
        "time_rank":       rank,
        "time_rank_bonus": bonus,
        "label":           label,
        "races_used":      len(times),
    }


# ============================================================
# 11. 類似レース検索
# ============================================================

def find_similar_races(
    df: pd.DataFrame,
    venue: str,
    surface: str,
    distance: int,
    race_class: str = "",
    n_horses: int = 16,
    n_results: int = 30,
) -> dict:
    """
    Kaggleデータから類似条件のレースを検索し、
    過去の傾向（荒れ率・人気別勝率・上がり3F傾向）を集計する。

    Returns
    -------
    {
        "total_races":     int,
        "upset_rate":      float,   10番人気以上が3着内に来た割合
        "fav_win_rate":    float,   1番人気勝率
        "top3_pop_avg":    float,   1〜3着の平均人気
        "fast_3f_wins":    float,   上がり最速馬の勝率
        "sample_df":       DataFrame,  上位30件のサンプル
        "pattern_summary": str,     ワンライン要約
    }
    """
    from data_loader import categorize_distance
    dist_cat = categorize_distance(distance)

    empty = {"total_races": 0, "upset_rate": 0.0, "fav_win_rate": 0.0,
             "top3_pop_avg": 5.0, "fast_3f_wins": 0.0, "sample_df": pd.DataFrame(),
             "pattern_summary": "データ不足"}

    if df.empty:
        return empty

    # フィルタリング
    cond = df.copy()
    if "surface" in cond.columns:
        cond = cond[cond["surface"] == surface]
    if "distance_cat" in cond.columns:
        cond = cond[cond["distance_cat"] == dist_cat]
    elif "distance" in cond.columns:
        cond = cond[(pd.to_numeric(cond["distance"], errors="coerce") - distance).abs() <= 200]
    if "venue" in cond.columns:
        cond = cond[cond["venue"] == venue]

    if len(cond) < 20:
        return empty

    # レース別集計（race_id or date+race_name）
    if "race_id" in cond.columns:
        grp = "race_id"
    elif "date" in cond.columns and "race_name" in cond.columns:
        cond = cond.copy()
        cond["_rkey"] = cond["date"].astype(str) + "_" + cond["race_name"].astype(str)
        grp = "_rkey"
    else:
        return empty

    total_races = cond[grp].nunique()

    # 1番人気勝率
    fav = cond[cond["popularity"] == 1] if "popularity" in cond.columns else pd.DataFrame()
    fav_win_rate = float(fav["win_flag"].mean()) if not fav.empty and "win_flag" in fav.columns else 0.0

    # 10番人気以上の3着内率（荒れ率）
    if "popularity" in cond.columns and "place_flag" in cond.columns:
        longshot = cond[cond["popularity"] >= 10]
        upset_rate = float(longshot["place_flag"].mean()) if not longshot.empty else 0.0
    else:
        upset_rate = 0.0

    # 1〜3着馬の平均人気
    top3 = cond[(cond["rank"] <= 3)] if "rank" in cond.columns else pd.DataFrame()
    top3_pop_avg = float(top3["popularity"].mean()) if not top3.empty and "popularity" in top3.columns else 5.0

    # 上がり3F最速馬の勝率
    fast_3f_wins = 0.0
    if "last_3f" in cond.columns and grp in cond.columns:
        cond_3f = cond.dropna(subset=["last_3f"]).copy()
        cond_3f["last_3f_num"] = pd.to_numeric(cond_3f["last_3f"], errors="coerce")
        cond_3f = cond_3f.dropna(subset=["last_3f_num"])
        if not cond_3f.empty:
            # レースごとに最速3F馬を特定
            fastest = cond_3f.loc[cond_3f.groupby(grp)["last_3f_num"].idxmin()]
            fast_3f_wins = float(fastest["win_flag"].mean()) if "win_flag" in fastest.columns else 0.0

    # パターンサマリー文
    parts = []
    if fav_win_rate < 0.25:
        parts.append(f"1番人気勝率{fav_win_rate*100:.0f}%（荒れやすい）")
    else:
        parts.append(f"1番人気勝率{fav_win_rate*100:.0f}%")
    parts.append(f"平均3着内人気{top3_pop_avg:.1f}番人気")
    if upset_rate > 0.15:
        parts.append(f"大穴率{upset_rate*100:.0f}%（10番人気以上が3着内）")
    if fast_3f_wins > 0.3:
        parts.append(f"上がり最速馬の勝率{fast_3f_wins*100:.0f}%（末脚重要）")
    pattern_summary = " / ".join(parts)

    # サンプルデータ（直近30件）
    if "date" in cond.columns:
        sample_df = cond.sort_values("date", ascending=False).head(n_results)
    else:
        sample_df = cond.head(n_results)

    return {
        "total_races":     total_races,
        "upset_rate":      round(upset_rate, 3),
        "fav_win_rate":    round(fav_win_rate, 3),
        "top3_pop_avg":    round(top3_pop_avg, 1),
        "fast_3f_wins":    round(fast_3f_wins, 3),
        "sample_df":       sample_df,
        "pattern_summary": pattern_summary,
    }


# ============================================================
# 初距離・初馬場チェック
# ============================================================

def check_first_time_conditions(
    df: pd.DataFrame,
    horse_name: str,
    surface: str,
    distance: int,
) -> dict:
    """
    今回が「初馬場」「初距離帯」かどうかを判定する。

    なぜ注目か:
    - 市場は「未知数」として嫌うためオッズが高め → 適性があれば割安
    - 特に「初ダート・芝実績あり馬」「初長距離・スタミナ型血統」は覚醒パターンになりやすい

    Returns
    -------
    {
        "is_first_surface":  bool,
        "is_first_distance": bool,   距離帯（短距離/マイル/中距離/長距離）として初めて
        "prev_surface":      str,    前走の馬場（芝↔ダート変わりの補足）
        "bonus":             float,
        "label":             str,
    }
    """
    from data_loader import categorize_distance
    dist_cat = categorize_distance(distance)

    empty = {"is_first_surface": False, "is_first_distance": False,
             "prev_surface": "", "bonus": 0.0, "label": ""}

    if df.empty or "horse_name" not in df.columns:
        return empty

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if hist.empty:
        return empty

    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=False)

    prev_surface = str(hist.iloc[0].get("surface", "")) if not hist.empty else ""

    # 初馬場（今回の馬場での出走が1回もない）
    is_first_surface = False
    if "surface" in hist.columns:
        past_surfaces = hist["surface"].dropna().unique().tolist()
        is_first_surface = surface not in past_surfaces

    # 初距離帯
    is_first_distance = False
    if "distance_cat" in hist.columns:
        past_dists = hist["distance_cat"].dropna().unique().tolist()
        is_first_distance = dist_cat not in past_dists
    elif "distance" in hist.columns:
        def _cat(d):
            try: return categorize_distance(int(d))
            except: return ""
        past_dists = hist["distance"].apply(_cat).unique().tolist()
        is_first_distance = dist_cat not in past_dists

    bonus = 0.0
    labels = []

    if is_first_surface and is_first_distance:
        bonus = 0.01
        labels.append(f"初{surface}＋初{dist_cat}（未知数・割安の可能性）")
    elif is_first_surface:
        bonus = 0.008
        surface_change = f"{prev_surface}→{surface}" if prev_surface else f"初{surface}"
        labels.append(f"初{surface}（{surface_change}）市場が過小評価しがち")
    elif is_first_distance:
        bonus = 0.005
        labels.append(f"初{dist_cat}（距離適性は未知数）")

    return {
        "is_first_surface":  is_first_surface,
        "is_first_distance": is_first_distance,
        "prev_surface":      prev_surface,
        "bonus":             round(bonus, 4),
        "label":             " / ".join(labels),
    }


# ============================================================
# 厩舎の近況勝率（直近2週間）
# ============================================================

def calc_stable_recent_form(
    df: pd.DataFrame,
    trainer: str,
    days: int = 14,
    race_date=None,
) -> dict:
    """
    厩舎の直近N日間の勝率を計算する。

    競馬は「厩舎の状態」が連動する。好調厩舎の馬は全体的に状態が良い。

    Returns
    -------
    {
        "recent_wins":   int,
        "recent_races":  int,
        "recent_wr":     float,
        "trend":         str,   "好調" | "不調" | "普通" | "データなし"
        "bonus":         float,
        "label":         str,
    }
    """
    empty = {"recent_wins": 0, "recent_races": 0, "recent_wr": 0.0,
             "trend": "データなし", "bonus": 0.0, "label": ""}

    if df.empty or "trainer" not in df.columns or not trainer:
        return empty

    from datetime import datetime, timedelta
    import pandas as pd

    if race_date is None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    else:
        cutoff = pd.Timestamp(race_date) - pd.Timedelta(days=days)

    if "date" not in df.columns:
        return empty

    trainer_df = df[df["trainer"] == trainer].copy()
    trainer_df["date"] = pd.to_datetime(trainer_df["date"], errors="coerce")
    recent = trainer_df[trainer_df["date"] >= cutoff]

    if len(recent) < 3:
        return {**empty, "trend": "データ不足（直近出走少）"}

    recent_races = len(recent)
    recent_wins  = int(recent["win_flag"].sum()) if "win_flag" in recent.columns else 0
    recent_wr    = recent_wins / recent_races

    # 全体平均との比較（厩舎単位でなく全体）
    all_wr = float(df["win_flag"].mean()) if "win_flag" in df.columns else 0.07

    bonus = 0.0
    if recent_wr >= all_wr * 2.0:
        trend, bonus = "絶好調", 0.02
        label = f"厩舎絶好調（直近{days}日: {recent_wins}/{recent_races}勝 {recent_wr*100:.0f}%）"
    elif recent_wr >= all_wr * 1.4:
        trend, bonus = "好調", 0.01
        label = f"厩舎好調（直近{days}日: {recent_wins}/{recent_races}勝）"
    elif recent_wr <= all_wr * 0.4 and recent_races >= 5:
        trend, bonus = "不調", -0.01
        label = f"厩舎不調（直近{days}日: {recent_wins}/{recent_races}勝 {recent_wr*100:.0f}%）"
    else:
        trend, bonus = "普通", 0.0
        label = f"厩舎普通（直近{days}日: {recent_wins}/{recent_races}勝）"

    return {
        "recent_wins":  recent_wins,
        "recent_races": recent_races,
        "recent_wr":    round(recent_wr, 3),
        "trend":        trend,
        "bonus":        round(bonus, 4),
        "label":        label,
    }


# ============================================================
# 出走頭数の増減チェック
# ============================================================

def check_field_size_change(
    df: pd.DataFrame,
    horse_name: str,
    current_n_horses: int,
    surface: str = "",
    n_races: int = 5,
) -> dict:
    """
    過去の平均出走頭数と今回の頭数を比較する。

    少頭数 → 先行・逃げ馬に有利（プレッシャーなく逃げやすい）
    多頭数 → 差し・外枠に有利（スローになりにくい）

    Returns
    -------
    {
        "avg_field":     float,    過去の平均頭数
        "current_field": int,
        "diff":          float,    今回 - 過去平均（プラス = 頭数増加）
        "trend":         str,
        "bonus":         float,    脚質×頭数変化のボーナス
        "label":         str,
    }
    """
    empty = {"avg_field": current_n_horses, "current_field": current_n_horses,
             "diff": 0.0, "trend": "変化なし", "bonus": 0.0, "label": ""}

    if df.empty or "horse_name" not in df.columns:
        return empty

    # race_idごとの出走頭数を取得
    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if surface and "surface" in hist.columns:
        hist = hist[hist["surface"] == surface]
    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=False)
    hist = hist.head(n_races)

    if hist.empty:
        return empty

    # レースごとの頭数（同一レースの出走頭数）
    grp_key = "race_id" if "race_id" in df.columns else None
    if grp_key:
        past_fields = []
        for rid in hist[grp_key].dropna().unique():
            n = len(df[df[grp_key] == rid])
            if n >= 3:
                past_fields.append(n)
    else:
        return empty

    if not past_fields:
        return empty

    avg_field = float(np.mean(past_fields))
    diff = current_n_horses - avg_field

    bonus = 0.0
    label = ""
    trend = "変化なし"

    if diff <= -4:
        trend = "頭数大幅減少"
        label = f"頭数大幅減（平均{avg_field:.0f}頭→今回{current_n_horses}頭）"
        bonus = 0.01  # 一般的に先行馬有利
    elif diff >= 4:
        trend = "頭数大幅増加"
        label = f"頭数大幅増（平均{avg_field:.0f}頭→今回{current_n_horses}頭）"
        bonus = -0.005  # 混戦・位置取りリスク増
    elif diff <= -2:
        trend = "頭数減少"
        label = f"頭数減（平均{avg_field:.0f}頭→今回{current_n_horses}頭）"
        bonus = 0.005
    else:
        label = f"頭数変化軽微（平均{avg_field:.0f}頭→今回{current_n_horses}頭）"

    return {
        "avg_field":     round(avg_field, 1),
        "current_field": current_n_horses,
        "diff":          round(diff, 1),
        "trend":         trend,
        "bonus":         round(bonus, 4),
        "label":         label,
    }


# ============================================================
# 前走上がり3F × 今回ペース適合チェック
# ============================================================

def check_closing_pace_fit(
    df: pd.DataFrame,
    horse_name: str,
    current_pace: str,
    surface: str = "",
    distance: int = 0,
) -> dict:
    """
    前走で上がり最速（または上位）だった馬が、今回のペース予測でも
    「末脚が活きる展開」に向いているかを判定する。

    ロジック:
    - 前走スロー → 今回もスロー: 末脚型は引き続き有利
    - 前走ハイペース→展開不利で上がり最速 → 今回スロー: 強烈な巻き返し候補
    - 前走スロー→今回ハイペース: 先行馬なら楽ではなくなる

    Returns
    -------
    {
        "prev_pace":       str,   前走のペース推定（ハイ/スロー/不明）
        "current_pace":    str,
        "was_fast_closer": bool,  前走で上がり上位（同レース2位以内）だったか
        "pace_shift":      str,   "好転" | "悪化" | "継続" | "不明"
        "bonus":           float,
        "label":           str,
    }
    """
    empty = {"prev_pace": "不明", "current_pace": current_pace,
             "was_fast_closer": False, "pace_shift": "不明", "bonus": 0.0, "label": ""}

    if df.empty or "horse_name" not in df.columns:
        return empty

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=False)

    if hist.empty:
        return empty

    prev_row = hist.iloc[0]
    my_3f = pd.to_numeric(prev_row.get("last_3f"), errors="coerce")

    # 前走の上がり3F順位（同レース内）
    was_fast_closer = False
    grp_key = "race_id" if "race_id" in df.columns else None
    if grp_key and not pd.isna(my_3f):
        rkey = prev_row.get(grp_key)
        if rkey:
            race_df = df[df[grp_key] == rkey]
            race_3f = pd.to_numeric(race_df["last_3f"], errors="coerce").dropna()
            if len(race_3f) >= 3:
                rank_in = int((race_3f < my_3f).sum()) + 1
                was_fast_closer = rank_in <= 2  # 上がり1〜2位

    # 前走ペース推定（コーナー通過順と上がり3Fから）
    prev_pace = "不明"
    corner_s = prev_row.get("corner_order", "")
    if corner_s:
        corners = [int(x) for x in str(corner_s).replace(",","-").split("-") if x.strip().isdigit()]
        if len(corners) >= 2:
            # 前半位置が後半より後退が多い = ハイペース
            front_pos = corners[0]
            back_pos  = corners[-1] if len(corners) > 1 else corners[0]
            if back_pos - front_pos >= 3:
                prev_pace = "ハイペース（後退）"
            elif front_pos - back_pos >= 3:
                prev_pace = "スローペース（進出）"
            else:
                prev_pace = "ミドル"

    # ペース適合判定
    is_cur_slow = "スロー" in current_pace
    is_cur_high = "ハイ" in current_pace
    is_prev_high = "ハイ" in prev_pace

    bonus = 0.0
    pace_shift = "不明"
    label = ""

    if was_fast_closer and is_prev_high and is_cur_slow:
        # 前走ハイペースで上がり最速 → 今回スロー = 最大の恩恵
        bonus = 0.03
        pace_shift = "好転"
        label = "◎◎ 前走ハイペース上がり最速→今回スロー予測（最強巻き返しパターン）"
    elif was_fast_closer and is_cur_slow:
        # 前走でも上がり使えた → 今回もスロー = 引き続き末脚活きる
        bonus = 0.015
        pace_shift = "継続"
        label = "◎ 前走上がり上位→今回スロー予測（末脚引き続き活きる）"
    elif was_fast_closer and is_cur_high:
        # 末脚型なのに今回ハイペース = 前半消耗リスク
        bonus = -0.005
        pace_shift = "悪化"
        label = "▲ 前走上がり上位だが今回ハイペース予測（前半消耗に注意）"
    elif not was_fast_closer and is_cur_high:
        # 先行型 × ハイペース = 消耗戦
        bonus = -0.01
        pace_shift = "悪化"
        label = "▲ 先行型×ハイペース予測（消耗戦になる可能性）"

    return {
        "prev_pace":       prev_pace,
        "current_pace":    current_pace,
        "was_fast_closer": was_fast_closer,
        "pace_shift":      pace_shift,
        "bonus":           round(bonus, 4),
        "label":           label,
    }


# ============================================================
# 右回り/左回り適性
# ============================================================

# JRA全会場の回り方向
VENUE_TURN_DIRECTION = {
    # 左回り
    "東京": "左", "新潟": "左", "中京": "左",
    # 右回り
    "中山": "右", "阪神": "右", "京都": "右",
    "函館": "右", "札幌": "右", "福島": "右", "小倉": "右",
}

def calc_turn_aptitude(
    df: pd.DataFrame,
    horse_name: str,
    current_venue: str,
    surface: str = "",
) -> dict:
    """
    右回り/左回りの得意不得意を過去成績から算出する。

    Returns
    -------
    {
        "current_direction": str,   "左" | "右" | "不明"
        "left_stats":  {"races": int, "wins": int, "places": int, "win_rate": float}
        "right_stats": {"races": int, "wins": int, "places": int, "win_rate": float}
        "preferred":   str,   "左回り得意" | "右回り得意" | "差なし"
        "is_mismatch": bool,  今回が苦手方向か
        "bonus":       float,
        "label":       str,
    }
    """
    current_dir = VENUE_TURN_DIRECTION.get(current_venue, "不明")
    empty = {
        "current_direction": current_dir,
        "left_stats":  {"races": 0, "wins": 0, "places": 0, "win_rate": 0.0},
        "right_stats": {"races": 0, "wins": 0, "places": 0, "win_rate": 0.0},
        "preferred": "差なし", "is_mismatch": False,
        "bonus": 0.0, "label": "",
    }

    if df.empty or "horse_name" not in df.columns or "venue" not in df.columns:
        return empty

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if surface and "surface" in hist.columns:
        hist = hist[hist["surface"] == surface]
    if hist.empty:
        return empty

    hist["_dir"] = hist["venue"].map(VENUE_TURN_DIRECTION)

    def _stats(direction: str) -> dict:
        sub = hist[hist["_dir"] == direction]
        races  = len(sub)
        wins   = int(sub["win_flag"].sum())   if "win_flag"   in sub.columns else 0
        places = int(sub["place_flag"].sum()) if "place_flag" in sub.columns else 0
        wr = wins / races if races > 0 else 0.0
        return {"races": races, "wins": wins, "places": places, "win_rate": round(wr, 3)}

    left_s  = _stats("左")
    right_s = _stats("右")

    # 比較（最低3走以上あるほうだけ使う）
    left_valid  = left_s["races"]  >= 3
    right_valid = right_s["races"] >= 3

    bonus = 0.0
    preferred = "差なし"
    is_mismatch = False
    label = ""

    if left_valid and right_valid:
        diff = left_s["win_rate"] - right_s["win_rate"]
        if diff >= 0.08:
            preferred = "左回り得意"
            if current_dir == "左":
                bonus, label = 0.015, f"◎ 左回り得意（左{left_s['win_rate']*100:.0f}%/右{right_s['win_rate']*100:.0f}%）"
            else:
                bonus, label = -0.01, f"▲ 左回り得意だが今回右回り（苦手の可能性）"
                is_mismatch = True
        elif diff <= -0.08:
            preferred = "右回り得意"
            if current_dir == "右":
                bonus, label = 0.015, f"◎ 右回り得意（右{right_s['win_rate']*100:.0f}%/左{left_s['win_rate']*100:.0f}%）"
            else:
                bonus, label = -0.01, f"▲ 右回り得意だが今回左回り（苦手の可能性）"
                is_mismatch = True
        else:
            label = f"左右差なし（左{left_s['win_rate']*100:.0f}%/右{right_s['win_rate']*100:.0f}%）"
    elif left_valid and current_dir == "左":
        label = f"左回り実績あり（{left_s['win_rate']*100:.0f}%）"
        bonus = 0.005
    elif right_valid and current_dir == "右":
        label = f"右回り実績あり（{right_s['win_rate']*100:.0f}%）"
        bonus = 0.005
    elif current_dir != "不明":
        label = f"{current_dir}回りの実績が少ない（要注意）"

    return {
        "current_direction": current_dir,
        "left_stats":  left_s,
        "right_stats": right_s,
        "preferred":   preferred,
        "is_mismatch": is_mismatch,
        "bonus":       round(bonus, 4),
        "label":       label,
    }


# ============================================================
# 近走詳細分析（展開・不利・馬場ミスマッチ自動判定）
# ============================================================

# 着差→秒換算の係数（JRA非公式だが実用的な近似）
MARGIN_TO_SEC = {
    "ハナ": 0.05, "アタマ": 0.1, "クビ": 0.2,
    "1/2": 0.3, "3/4": 0.4, "1": 0.6,
    "1.1/4": 0.8, "1.1/2": 0.9, "2": 1.2,
    "2.1/2": 1.5, "3": 1.8, "4": 2.4,
    "5": 3.0, "大差": 5.0,
}


def _parse_corner_positions(corner_str) -> list[int]:
    """'3-3-4-5' や '3,3,4,5' を [3,3,4,5] に変換"""
    if not corner_str or pd.isna(corner_str):
        return []
    try:
        s = str(corner_str).replace(",", "-")
        return [int(x.strip()) for x in s.split("-") if x.strip().isdigit()]
    except Exception:
        return []


def _get_race_group_key(df: pd.DataFrame) -> str | None:
    """レース識別キー列名を返す"""
    if "race_id" in df.columns:
        return "race_id"
    if "date" in df.columns and "race_name" in df.columns:
        return "_rkey"
    return None


def analyze_recent_races(
    df: pd.DataFrame,
    horse_name: str,
    current_surface: str = "",
    current_distance: int = 0,
    current_pace: str = "",
    n: int = 5,
) -> list[dict]:
    """
    過去N走を1走ごとに詳細分析し、「言い訳」「強調材料」を自動生成する。

    自動判定する内容:
    ① 上がり最速だったが負け → 展開負けシグナル
    ② コーナーで大きく順位を落とした → 道中不利の可能性
    ③ 不向き馬場（重・不良）で惨敗 → 今回良なら巻き返し候補
    ④ クラス大幅格上で惨敗 → 言い訳あり
    ⑤ 超ハイペース/スローに巻き込まれた先行/差し馬 → 展開負け
    ⑥ 今回条件が得意条件に近づいた → 狙い目
    ⑦ 着差が僅差（0.3秒以内）で負け → 実力は近い

    Returns
    -------
    list[dict]: 走ごとの分析結果（新しい順）
    {
        "date", "race_name", "surface", "distance", "track_condition",
        "rank", "popularity", "odds", "last_3f", "corner_order",
        "pace_cat",          予測ペース（あれば）
        "fast_3f_rank",      この馬の上がり3F順位（同レース内）
        "is_fastest_3f",     上がり最速か
        "corner_drop",       コーナーで落とした順位数（大きいほど不利の可能性）
        "bad_track",         苦手馬場で走ったか
        "class_up",          クラス格上だったか
        "close_finish",      僅差負けか（0.3秒以内）
        "signals":  list[str]  自動生成したシグナル文言
        "excuse":   str        まとめの「言い訳」文
        "plus":     str        まとめの「強調材料」文
        "resume_bonus": float  今回への巻き返しボーナス（この走から）
    }
    """
    if df.empty or "horse_name" not in df.columns:
        return []

    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if hist.empty:
        return []

    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=False)
    hist = hist.head(n)

    # レース識別キーを準備
    grp_key = _get_race_group_key(df)
    if grp_key == "_rkey":
        df = df.copy()
        df["_rkey"] = df["date"].astype(str) + "_" + df["race_name"].astype(str)
        hist["_rkey"] = hist["date"].astype(str) + "_" + hist["race_name"].astype(str)

    # 全体の上がり3F平均（コース×距離帯ごと）
    all_3f = pd.to_numeric(df.get("last_3f", pd.Series(dtype=float)), errors="coerce")
    global_3f_avg = float(all_3f.median()) if not all_3f.dropna().empty else 35.0

    results = []

    for _, row in hist.iterrows():
        rank      = pd.to_numeric(row.get("rank"),       errors="coerce")
        pop       = pd.to_numeric(row.get("popularity"), errors="coerce")
        my_3f     = pd.to_numeric(row.get("last_3f"),    errors="coerce")
        corner_s  = row.get("corner_order", "") or row.get("corner4", "")
        track_c   = str(row.get("track_condition", "良"))
        surface_r = str(row.get("surface", ""))
        distance_r = pd.to_numeric(row.get("distance", 0), errors="coerce")
        margin_s  = str(row.get("margin", ""))
        race_class = str(row.get("race_class", ""))
        finish_time = pd.to_numeric(row.get("finish_time"), errors="coerce")
        field_size_r = pd.to_numeric(row.get("field_size"), errors="coerce")
        speed_fig_r  = pd.to_numeric(row.get("speed_figure"), errors="coerce")

        signals      = []
        excuse_parts = []
        plus_parts   = []
        resume_bonus = 0.0

        # ---- ① 上がり3F順位（同レース内） ----
        fast_3f_rank  = None
        is_fastest_3f = False

        if grp_key and not pd.isna(my_3f):
            rkey_val = row.get(grp_key)
            if rkey_val is not None:
                race_df = df[df[grp_key] == rkey_val]
                race_3f = pd.to_numeric(race_df.get("last_3f", pd.Series()), errors="coerce").dropna()
                if len(race_3f) >= 3:
                    sorted_3f = race_3f.sort_values()
                    rank_in_race = int((sorted_3f < my_3f).sum()) + 1
                    fast_3f_rank  = rank_in_race
                    is_fastest_3f = (rank_in_race == 1)

                    if is_fastest_3f and not pd.isna(rank) and rank > 1:
                        # 1着以外で上がり最速 = 展開負け
                        signals.append(f"上がり最速({my_3f:.1f}秒)だが{int(rank)}着 → 展開負けの可能性大")
                        excuse_parts.append("上がり最速で負け（展開不利）")
                        resume_bonus += 0.025 if rank >= 4 else 0.015
                    elif fast_3f_rank == 2 and not pd.isna(rank) and rank > 2:
                        signals.append(f"上がり2位({my_3f:.1f}秒)だが{int(rank)}着 → 末脚は使えた")
                        excuse_parts.append("上がり2位で負け")
                        resume_bonus += 0.01

        # ---- ② コーナー通過順位の変化 ----
        corner_drop = 0
        corners     = _parse_corner_positions(corner_s)
        if len(corners) >= 2:
            first_pos = corners[0]
            last_pos  = corners[-1]
            corner_drop = last_pos - first_pos  # プラス = 後退、マイナス = 進出

            # 中間で大きく落とした（不利の可能性）
            if len(corners) >= 3:
                max_pos = max(corners[1:])
                mid_drop = max_pos - corners[0]
                if mid_drop >= 5:
                    signals.append(f"道中で{mid_drop}番手後退 → 不利・包まれた可能性")
                    excuse_parts.append(f"道中不利（{mid_drop}頭後退）")
                    resume_bonus += 0.015

            # 大外を回った可能性（最終コーナーで前より外に）
            if corner_drop >= 4 and not pd.isna(rank) and rank > 3:
                signals.append(f"最終コーナーで{corner_drop}番手後退 → 進路不利の可能性")
                excuse_parts.append("コーナーロス")
                resume_bonus += 0.01

        # ---- ③ 不向き馬場 ----
        bad_track = False
        if track_c in ("重", "不良") and surface_r == "芝":
            # 芝の重馬場は一般的に不得手な馬が多い
            if not pd.isna(rank) and rank > (pop if not pd.isna(pop) else 5) + 2:
                bad_track = True
                signals.append(f"芝の{track_c}馬場で大敗（人気{int(pop) if not pd.isna(pop) else '?'}着→{int(rank)}着）")
                excuse_parts.append(f"{track_c}馬場不向きの可能性")
                if current_surface == "芝" and track_c in ("重", "不良"):
                    pass  # 今回も重なら言い訳にならない
                elif current_surface == "芝":
                    resume_bonus += 0.015  # 今回が良ならプラス

        # ---- ④ クラス格上 ----
        class_up = False
        upper_keywords = ["G1", "G2", "G3", "重賞", "オープン", "3勝", "OPEN"]
        if any(kw in race_class for kw in upper_keywords[:3]):
            class_up = True
            signals.append(f"重賞・G1/G2/G3出走（格上挑戦）")
            excuse_parts.append("格上挑戦")
            resume_bonus += 0.01

        # ---- ⑤ 着差僅差 ----
        close_finish = False
        margin_sec   = MARGIN_TO_SEC.get(margin_s.strip())
        if margin_sec is not None and margin_sec <= 0.3 and not pd.isna(rank) and rank > 1:
            close_finish = True
            signals.append(f"着差{margin_s}（{margin_sec}秒差）の僅差負け")
            plus_parts.append(f"僅差{margin_s}負け→実力的には互角")
            resume_bonus += 0.008

        # ---- ⑥ 今回条件との比較 ----
        if current_surface and surface_r and surface_r != current_surface:
            signals.append(f"前走は{surface_r}（今回{current_surface}に変わり）")

        # ---- ⑦ 上がり3F が平均より大幅に遅い（マイナス材料） ----
        if not pd.isna(my_3f) and my_3f > global_3f_avg + 2.0:
            signals.append(f"上がり{my_3f:.1f}秒（全体平均より{my_3f - global_3f_avg:.1f}秒遅い）")

        # ---- ⑧ PCI（ペース変化指数）：展開との相性 ----
        # pci > 100 = 後半速い（差し有利ペース）/ pci < 100 = 前半速い（前残りペース）
        pci_val = None
        if (not pd.isna(finish_time) and not pd.isna(my_3f)
                and not pd.isna(distance_r) and distance_r > 600):
            ft_sec = (finish_time // 1000) * 60 + (finish_time % 1000) / 10
            last3f_sec = my_3f   # last_3f は秒単位（34.5秒など）、/10 不要
            early_sec  = ft_sec - last3f_sec
            early_dist = float(distance_r) - 600.0
            if early_dist > 0 and last3f_sec > 0:
                early_pace600 = early_sec / (early_dist / 600)
                pci_val = (early_pace600 / last3f_sec) * 100
                pci_val = max(60.0, min(140.0, pci_val))

        # コーナー1番手位置（先行 vs 差し判定）
        c1_pos = None
        corners_parsed = _parse_corner_positions(corner_s)
        if corners_parsed:
            c1_pos = corners_parsed[0]

        field_s = float(field_size_r) if not pd.isna(field_size_r) else None

        if pci_val is not None and not pd.isna(rank) and field_s:
            # 差し馬が前残りペースで差し届かず（一変候補度大）
            if pci_val < 90 and c1_pos is not None and c1_pos > field_s * 0.5 and rank > field_s * 0.4:
                signals.append(f"前残りペース(PCI={pci_val:.0f})で後方から差し届かず")
                excuse_parts.append("前残りペースで不発")
                resume_bonus += 0.022
            # 先行馬が差し有利ペースで垂れた
            elif pci_val > 115 and c1_pos is not None and c1_pos <= field_s * 0.4 and rank > field_s * 0.5:
                signals.append(f"差し有利ペース(PCI={pci_val:.0f})で先行して失速")
                excuse_parts.append("差し有利ペースで先行失速")
                resume_bonus += 0.015

        # ---- ⑨ 速度指数乖離（相手強化で凡走）----
        # 速度指数（speed_figure）は「同条件の標準タイムとの差」
        # 速度指数が平均的なのに着順が悪い = 相手が強かっただけ
        if (not pd.isna(speed_fig_r) and not pd.isna(rank) and field_s
                and speed_fig_r >= -3.0   # 自身のパフォーマンスは悪くない
                and rank > field_s * 0.45):  # 着順は後半
            # さらに強い条件: 今回よりランクが高い相手
            signals.append(f"速度指数{speed_fig_r:+.1f}（自力OK）で{int(rank)}着 → 相手強の可能性")
            excuse_parts.append("相手強で凡走")
            resume_bonus += 0.018

        # ---- ⑩ 条件改善チェック（今走との比較） ----
        def _dist_cat(d):
            if pd.isna(d): return ""
            d = int(d)
            if d <= 1400: return "短距離"
            if d <= 1800: return "マイル"
            if d <= 2200: return "中距離"
            return "長距離"

        if current_distance and not pd.isna(distance_r):
            if _dist_cat(distance_r) != _dist_cat(current_distance):
                signals.append(f"前走{_dist_cat(distance_r)}→今走{_dist_cat(current_distance)}（距離変更）")
                if not excuse_parts:  # 他に言い訳がない場合のみ追加
                    excuse_parts.append(f"距離変更（{_dist_cat(distance_r)}→{_dist_cat(current_distance)}）")
                resume_bonus += 0.008

        # ---- まとめ文 ----
        excuse = " / ".join(excuse_parts) if excuse_parts else ""
        plus   = " / ".join(plus_parts)   if plus_parts   else ""

        results.append({
            "date":           str(row.get("date", ""))[:10],
            "race_name":      str(row.get("race_name", "")),
            "surface":        surface_r,
            "distance":       int(distance_r) if not pd.isna(distance_r) else 0,
            "track_condition": track_c,
            "rank":           int(rank)   if not pd.isna(rank) else None,
            "popularity":     int(pop)    if not pd.isna(pop)  else None,
            "odds":           row.get("odds"),
            "last_3f":        round(float(my_3f), 1) if not pd.isna(my_3f) else None,
            "corner_order":   corner_s,
            "fast_3f_rank":   fast_3f_rank,
            "is_fastest_3f":  is_fastest_3f,
            "corner_drop":    corner_drop,
            "bad_track":      bad_track,
            "class_up":       class_up,
            "close_finish":   close_finish,
            "pci":            round(pci_val, 1) if pci_val is not None else None,
            "signals":        signals,
            "excuse":         excuse,
            "plus":           plus,
            "resume_bonus":   round(min(resume_bonus, 0.05), 4),
        })

    return results


def calc_resume_bonus_from_recent(
    recent_analyses: list[dict],
) -> dict:
    """
    近走分析から「今回への巻き返し期待ボーナス」を集計する。

    Returns
    -------
    {
        "total_bonus":    float,
        "summary":        str,
        "top_excuse":     str,   最も強い言い訳
        "ippen_candidate": bool  複数の言い訳が重なり一変可能性あり
        "excuse_flags":   list[str]  前走言い訳フラグ（表示用）
    }
    """
    if not recent_analyses:
        return {"total_bonus": 0.0, "summary": "", "top_excuse": "",
                "ippen_candidate": False, "excuse_flags": []}

    # 直近3走に絞り、新しいほど重みを高く
    weights = [1.0, 0.7, 0.5]
    total_bonus  = 0.0
    excuse_parts = []
    excuse_flags = []  # 直近1走の言い訳フラグ（表示用）

    for i, race in enumerate(recent_analyses[:3]):
        w = weights[i]
        total_bonus += race["resume_bonus"] * w
        if race["excuse"]:
            excuse_parts.append(f"[{i+1}走前]{race['excuse']}")
        if i == 0 and race.get("signals"):
            # UI-4: キーワードフィルターを除去し、全シグナルを excuse_flags として表示
            excuse_flags = race["signals"][:3]  # 最大3つ（フィルターなし）

    total_bonus = min(total_bonus, 0.07)  # 上限を 0.07 に引き上げ

    top_excuse = excuse_parts[0].replace("[1走前]", "") if excuse_parts else ""
    summary = " / ".join(excuse_parts) if excuse_parts else "特記なし"

    # A-5: signalsベースに変更（excuse_partsは空でもsignalsには発火がある場合が多い）
    # signals >= 1 かつ total_bonus >= 0.02 で一変候補
    ippen = (len(excuse_flags) >= 1 and total_bonus >= 0.02)

    return {
        "total_bonus":     round(total_bonus, 4),
        "summary":         summary,
        "top_excuse":      top_excuse,
        "ippen_candidate": ippen,
        "excuse_flags":    excuse_flags,
    }


# ============================================================
# 有力馬撃破スコア
# ============================================================

def calc_beaten_strong_horses(
    df: pd.DataFrame,
    horse_name: str,
    surface: str = "",
    distance: int = 0,
    n_races: int = 10,
    recency_weight: bool = True,
) -> dict:
    """
    過去レースで「人気馬（1〜3番人気）を直接下した」実績を評価する。

    設計思想:
    - 市場はその馬をまだ「穴馬」として見ているが、実際には強い馬を
      倒したことがある → 構造的な過小評価のシグナル
    - 直近ほど重み付けを高くする（古い実績は割引）

    判定方法:
    1. 同一レースIDで、この馬の着順 < 1〜3番人気馬の着順 を検出
    2. 撃破した人気馬の人気（1位 > 2位 > 3位）と新しさで加点

    Returns
    -------
    {
        "beat_count":    int,    過去N走で有力馬を上回った回数
        "best_victim":   str,    最も人気だった相手（例: "1番人気"）
        "beat_details":  list,   [{race_name, date, beaten_popularity, our_rank}, ...]
        "bonus":         float,
        "label":         str,
    }
    """
    empty = {
        "beat_count": 0, "best_victim": None,
        "beat_details": [], "bonus": 0.0, "label": "",
    }

    required = {"horse_name", "rank", "popularity"}
    if df.empty or not required.issubset(df.columns):
        return empty

    # この馬の過去成績
    hist = df[df["horse_name"].str.strip() == horse_name.strip()].copy()
    if hist.empty:
        return empty

    # 条件絞り（同馬場×近い距離帯）
    if surface and "surface" in hist.columns:
        hist = hist[hist["surface"] == surface]
    if distance and "distance" in hist.columns:
        dist_hist = hist[
            (pd.to_numeric(hist["distance"], errors="coerce") - distance).abs() <= 400
        ]
        if len(dist_hist) >= 2:
            hist = dist_hist  # 条件一致が少なければ全体を使う

    if "date" in hist.columns:
        hist = hist.sort_values("date", ascending=False)
    hist = hist.head(n_races)

    if hist.empty:
        return empty

    # NEW-1: レース識別キーを venue+race_no 方式に変更（date+race_name は重複する）
    # train_lgbm.py と同じ方式: date + "_" + venue + "_" + race_no
    if "date" in hist.columns and "venue" in hist.columns and "race_no" in hist.columns:
        if "_rkey" not in df.columns:
            df = df.copy()
            df["_rkey"] = (df["date"].astype(str) + "_"
                           + df["venue"].astype(str) + "_"
                           + df["race_no"].astype(str))
        hist = hist.copy()
        hist["_rkey"] = (hist["date"].astype(str) + "_"
                         + hist["venue"].astype(str) + "_"
                         + hist["race_no"].astype(str))
        race_key = "_rkey"
    elif "date" in hist.columns and "race_name" in hist.columns:
        # フォールバック: venue/race_noがない場合は date+race_name（精度低下あり）
        if "_rkey" not in df.columns:
            df = df.copy()
            df["_rkey"] = df["date"].astype(str) + "_" + df["race_name"].astype(str)
        hist = hist.copy()
        hist["_rkey"] = hist["date"].astype(str) + "_" + hist["race_name"].astype(str)
        race_key = "_rkey"
    else:
        return empty

    beat_details = []
    races_checked = hist[race_key].dropna().unique()

    for i, rkey in enumerate(races_checked):
        race_df = df[df[race_key] == rkey] if race_key in df.columns else pd.DataFrame()
        if race_df.empty:
            continue

        my_rows = race_df[race_df["horse_name"].str.strip() == horse_name.strip()]
        if my_rows.empty:
            continue
        my_rank = pd.to_numeric(my_rows["rank"].iloc[0], errors="coerce")
        if pd.isna(my_rank) or my_rank > 5:  # 6着以下は対象外
            continue

        # A-2: 未勝利/新馬戦を除外（弱い1番人気を「強敵撃破」とみなさない）
        _race_name_str = str(my_rows.iloc[0].get("race_name", ""))
        if any(kw in _race_name_str for kw in ["未勝利", "新馬"]):
            continue

        # NEW-1: 各人気レベル（1・2・3番人気）で最大1頭のみカウント（重複防止）
        recency = max(0.5, 1.0 - i * 0.05) if recency_weight else 1.0
        for pop_level in [1, 2, 3]:
            rivals_pop = race_df[
                (race_df["horse_name"].str.strip() != horse_name.strip()) &
                (pd.to_numeric(race_df["popularity"], errors="coerce") == pop_level)
            ].copy()
            if rivals_pop.empty:
                continue
            # 最も着順が良いライバルのみ対象
            rivals_pop["_rank_n"] = pd.to_numeric(rivals_pop["rank"], errors="coerce")
            best_rival = rivals_pop.sort_values("_rank_n").iloc[0]
            rival_rank = best_rival["_rank_n"]
            if pd.isna(rival_rank):
                continue
            if my_rank < rival_rank:  # この馬が上回った
                beat_details.append({
                    "race_name":        my_rows.iloc[0].get("race_name", ""),
                    "date":             str(my_rows.iloc[0].get("date", "")),
                    "beaten_popularity": pop_level,
                    "our_rank":         int(my_rank),
                    "rival_rank":       int(rival_rank),
                    "recency_weight":   round(recency, 2),
                })
                break  # 1レースで最上位の1頭だけカウント

    if not beat_details:
        return empty

    # ボーナス計算（NEW-1: 上限を 0.015 に引き下げ）
    POP_BONUS = {1: 0.015, 2: 0.008, 3: 0.004}
    total_bonus = 0.0
    for b in beat_details:
        pop = b["beaten_popularity"]
        recency = b["recency_weight"]
        total_bonus += POP_BONUS.get(pop, 0.003) * recency

    total_bonus = min(total_bonus, 0.015)  # 上限 0.015

    # 最も格上の相手
    best_victim_pop = min(b["beaten_popularity"] for b in beat_details)
    best_victim = f"{best_victim_pop}番人気"

    # beat_count は重複排除済み件数（上限10）
    beat_count = min(len(beat_details), 10)

    if best_victim_pop == 1:
        label = f"◎ 1番人気撃破（1勝C以上 {beat_count}回）"
    elif best_victim_pop == 2:
        label = f"○ 2番人気以上撃破（1勝C以上 {beat_count}回）"
    else:
        label = f"△ 3番人気以上撃破（1勝C以上 {beat_count}回）"

    return {
        "beat_count":   beat_count,
        "best_victim":  best_victim,
        "beat_details": beat_details[:5],
        "bonus":        round(total_bonus, 4),
        "label":        label,
    }
