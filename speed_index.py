"""
NEW-7: タイム指数（スピード指数）自作モジュール

公式:
    speed_index = (条件別基準タイム - 走破タイム) × 10
        条件 = (venue, surface, distance, track_condition)
        単位 = 0.1秒（正 = 基準より速い）

馬場差補正:
    track_condition が「重・不良」なら基準タイム自体が遅くなるので
    補正不要（条件別median を使うため自動的に補正される）

使い方:
    python speed_index.py   # 基準テーブル構築

    from speed_index import get_horse_speed_stats
    stats = get_horse_speed_stats(df_hist, horse_name, n=5)
"""
import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_BASE_PATH = _DATA_DIR / "speed_baseline.parquet"

_BASE_CACHE: pd.DataFrame | None = None


from utils import to_seconds as _to_seconds  # CLEAN-3: 共通ヘルパーを利用


def build_baseline_table(
    tfjv_path: Path | None = None,
    out_path: Path | None = None,
    min_samples: int = 5,
) -> pd.DataFrame:
    """(venue, surface, distance, track_condition) ごとの基準タイム（中央値）テーブル構築"""
    tfjv_path = tfjv_path or (_DATA_DIR / "tfjv_all.parquet")
    out_path  = out_path  or _BASE_PATH

    df = pd.read_parquet(tfjv_path)
    df = df.dropna(subset=["finish_time", "venue", "surface", "distance", "track_condition"]).copy()
    df["finish_time_sec"] = df["finish_time"].apply(_to_seconds)
    df = df.dropna(subset=["finish_time_sec"])

    g = df.groupby(["venue", "surface", "distance", "track_condition"]).agg(
        baseline_sec=("finish_time_sec", "median"),
        sample_n=("finish_time_sec", "size"),
    ).reset_index()
    g = g[g["sample_n"] >= min_samples].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out_path, index=False)
    return g


def _load_baselines() -> pd.DataFrame:
    global _BASE_CACHE
    if _BASE_CACHE is not None:
        return _BASE_CACHE
    if not _BASE_PATH.exists():
        _BASE_CACHE = build_baseline_table()
    else:
        _BASE_CACHE = pd.read_parquet(_BASE_PATH)
    return _BASE_CACHE


def calc_speed_index(
    finish_time: float,
    venue: str,
    surface: str,
    distance: int,
    track_condition: str,
) -> float | None:
    """
    1走のタイム指数を返す（0.1秒単位、正 = 基準より速い）。
    finish_time は TFJV 形式（MSST）も秒も自動判定。
    """
    ft_sec = _to_seconds(finish_time)
    if ft_sec is None or ft_sec <= 0:
        return None
    base = _load_baselines()
    hit = base[
        (base["venue"] == venue)
        & (base["surface"] == surface)
        & (base["distance"] == distance)
        & (base["track_condition"] == track_condition)
    ]
    if hit.empty:
        # フォールバック：条件をゆるめる
        hit = base[
            (base["venue"] == venue)
            & (base["surface"] == surface)
            & (base["distance"] == distance)
        ]
        if hit.empty:
            return None
    baseline = float(hit["baseline_sec"].mean())
    return (baseline - ft_sec) * 10.0


def get_horse_speed_stats(df_hist: pd.DataFrame, horse_name: str, n: int = 5) -> dict:
    """直近 n 走のタイム指数統計"""
    if df_hist is None or df_hist.empty or "horse_name" not in df_hist.columns:
        return _empty()
    sub = df_hist[df_hist["horse_name"].astype(str).str.strip() == str(horse_name).strip()]
    if sub.empty:
        return _empty()
    if "date" in sub.columns:
        sub = sub.sort_values("date", ascending=False).head(n)
    else:
        sub = sub.head(n)

    indices = []
    for _, r in sub.iterrows():
        idx = calc_speed_index(
            r.get("finish_time"), r.get("venue", ""), r.get("surface", ""),
            int(r.get("distance", 2000) or 2000), r.get("track_condition", "良"),
        )
        if idx is not None:
            indices.append(idx)

    if not indices:
        return _empty()
    avg  = float(np.mean(indices))
    best = float(max(indices))

    if best >= 30:
        label = "トップクラス"
    elif best >= 15:
        label = "重賞級"
    elif best >= 0:
        label = "標準"
    elif best >= -20:
        label = "やや遅い"
    else:
        label = "遅い"
    return {
        "speed_avg":   round(avg, 1),
        "speed_best":  round(best, 1),
        "speed_label": label,
        "n_races":     len(indices),
    }


def _empty() -> dict:
    return {"speed_avg": None, "speed_best": None, "speed_label": "データなし", "n_races": 0}


if __name__ == "__main__":
    print("Building speed baseline table ...")
    g = build_baseline_table()
    print(f"  {len(g):,} 条件パターン構築完了")
    print(g.head(15).to_string(index=False))

    # サンプル
    print("\n=== タイム指数サンプル ===")
    samples = [
        # (ft, venue, surface, distance, condition, expected)
        (1086, "東京", "芝", 1800, "良"),  # 1:48.6
        (1100, "東京", "芝", 1800, "良"),  # 1:50.0 = やや遅い
    ]
    for ft, v, sf, d, c in samples:
        idx = calc_speed_index(ft, v, sf, d, c)
        print(f"  ft={ft} {v}{sf}{d}m {c} → speed_index={idx}")
