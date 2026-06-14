"""
NEW-5: トラックバイアス（馬場差）時系列記録

開催日ごとに「内外有利・前後有利・速い遅い」のバイアスを定量化し時系列で蓄積。

指標:
    1. **タイムバイアス** = その日の (venue, surface) 全レースの speed_index 平均
       正 = 高速馬場 / 負 = 時計のかかる馬場
    2. **枠順バイアス** = 1着馬の平均 horse_no（中央値8.5から離れる量）
       低 = 内枠有利 / 高 = 外枠有利
    3. **脚質バイアス**（簡易版） = 1着馬の平均 corner4 位置 / field_size
       低 = 先行有利 / 高 = 差し追込有利

使い方:
    python track_variant.py           # 全期間で再構築
    python track_variant.py 2025-01-01 # この日以降のみ更新

    from track_variant import get_recent_variant
    bias = get_recent_variant(venue="東京", surface="芝", n_days=7)
"""
import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_VARIANT_PATH = _DATA_DIR / "track_variants.parquet"

_VARIANT_CACHE: pd.DataFrame | None = None


from utils import to_seconds as _to_seconds  # CLEAN-3: 共通ヘルパーを利用


def build_variant_table(
    tfjv_path: Path | None = None,
    out_path: Path | None = None,
    since_date: str | None = None,
) -> pd.DataFrame:
    """
    (date, venue, surface) ごとのバイアステーブル構築。
    speed_baseline.parquet を参照して各レースの speed_index を計算 → 日次平均。
    """
    tfjv_path = tfjv_path or (_DATA_DIR / "tfjv_all.parquet")
    out_path  = out_path  or _VARIANT_PATH

    from speed_index import _load_baselines
    base = _load_baselines()

    df = pd.read_parquet(tfjv_path)
    df = df.dropna(subset=["date", "venue", "surface", "rank"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if since_date:
        df = df[df["date"] >= pd.to_datetime(since_date)]
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce")
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce")
    df["corner4_pos"] = pd.to_numeric(
        df.get("corner4", pd.Series([np.nan] * len(df))).astype(str).str.extract(r"^(\d+)")[0],
        errors="coerce",
    )
    df["finish_time_sec"] = df["finish_time"].apply(_to_seconds)

    # 各レースの speed_index 計算
    base_indexed = base.set_index(["venue", "surface", "distance", "track_condition"])
    def _sidx(r):
        try:
            b = base_indexed.loc[(r["venue"], r["surface"], int(r["distance"]), r["track_condition"]), "baseline_sec"]
            if pd.isna(b) or pd.isna(r["finish_time_sec"]):
                return np.nan
            return (b - r["finish_time_sec"]) * 10.0
        except (KeyError, ValueError):
            return np.nan
    df["speed_index"] = df.apply(_sidx, axis=1)

    # tfjv_all.parquet の field_size 列はデータ不正（巨大値混入）あり
    # → race_no 単位で実際の頭数を再計算してから日次集計
    df["race_key"] = (
        df["date"].dt.strftime("%Y%m%d") + "_" + df["venue"].astype(str) + "_"
        + df.get("race_no", pd.Series([""] * len(df))).astype(str)
    )
    actual_fs = df.groupby("race_key")["horse_no"].transform("max")  # 最大馬番 ≒ 出走頭数
    df["actual_field_size"] = actual_fs

    # 日次集計（venue × surface）
    daily = df.groupby([df["date"].dt.date, "venue", "surface"]).agg(
        n_races=("rank", lambda x: x.eq(1).sum()),
        time_bias=("speed_index", "mean"),
        winner_horse_no_avg=("horse_no", lambda x: x[df.loc[x.index, "rank"] == 1].mean()),
        winner_corner4_pos=("corner4_pos", lambda x: x[df.loc[x.index, "rank"] == 1].mean()),
        avg_field_size=("actual_field_size", lambda x: x[df.loc[x.index, "rank"] == 1].mean()),
    ).reset_index().rename(columns={"date": "race_date"})

    # 枠順バイアス: 正なら外枠有利
    daily["gate_bias"] = (
        (daily["winner_horse_no_avg"] - daily["avg_field_size"] / 2)
        / (daily["avg_field_size"] / 2).clip(lower=1)
    ).round(3)
    # 脚質バイアス
    daily["pace_bias"] = (
        daily["winner_corner4_pos"] / daily["avg_field_size"].clip(lower=1)
    ).round(3)
    daily["time_bias"] = daily["time_bias"].round(1)

    daily = daily[daily["n_races"] >= 1].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, index=False)
    return daily


def _load_variants() -> pd.DataFrame:
    global _VARIANT_CACHE
    if _VARIANT_CACHE is not None:
        return _VARIANT_CACHE
    if not _VARIANT_PATH.exists():
        _VARIANT_CACHE = build_variant_table()
    else:
        _VARIANT_CACHE = pd.read_parquet(_VARIANT_PATH)
    return _VARIANT_CACHE


def get_recent_variant(venue: str, surface: str = "芝", n_days: int = 7) -> dict:
    """
    直近 n_days 日の venue × surface 平均バイアスを返す。
    """
    df = _load_variants()
    df = df[(df["venue"] == venue) & (df["surface"] == surface)].copy()
    if df.empty:
        return _empty()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.sort_values("race_date", ascending=False).head(n_days)
    return {
        "n_days":    len(df),
        "time_bias": round(float(df["time_bias"].mean()), 1) if df["time_bias"].notna().any() else None,
        "gate_bias": round(float(df["gate_bias"].mean()), 3) if df["gate_bias"].notna().any() else None,
        "pace_bias": round(float(df["pace_bias"].mean()), 3) if df["pace_bias"].notna().any() else None,
        "summary":   _describe(df["time_bias"].mean(), df["gate_bias"].mean(), df["pace_bias"].mean()),
    }


def get_variant_history(venue: str, surface: str = "芝", n_days: int = 30) -> pd.DataFrame:
    """
    指定 venue×surface の直近 n_days 日推移を返す（可視化用）
    """
    df = _load_variants()
    df = df[(df["venue"] == venue) & (df["surface"] == surface)].copy()
    if df.empty:
        return pd.DataFrame()
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.sort_values("race_date", ascending=False).head(n_days).sort_values("race_date")
    return df[["race_date", "time_bias", "gate_bias", "pace_bias", "n_races"]]


def _describe(tb, gb, pb) -> str:
    parts = []
    if pd.notna(tb):
        if tb > 3:    parts.append("高速馬場")
        elif tb < -5: parts.append("時計かかる")
        else:         parts.append("標準時計")
    if pd.notna(gb):
        if gb > 0.15:  parts.append("外枠有利")
        elif gb < -0.15: parts.append("内枠有利")
    if pd.notna(pb):
        if pb > 0.6:   parts.append("差し有利")
        elif pb < 0.35: parts.append("先行有利")
    return " / ".join(parts) if parts else "中立"


def _empty() -> dict:
    return {"n_days": 0, "time_bias": None, "gate_bias": None,
            "pace_bias": None, "summary": "データなし"}


# ============================================================
# C-5 (第12波): 当日の終了済みレース結果から動的バイアスを計算
# ============================================================
def compute_intraday_bias(
    today_results: list[dict],
    venue: str,
    surface: str = "芝",
    baseline_days: int = 7,
) -> dict:
    """
    当日 1R〜直前レースの結果から、その日の馬場バイアスを動的計算。

    Args:
        today_results: scraper.fetch_today_finished_results() の戻り値
        venue: 会場名（baseline 比較用）
        surface: "芝" or "ダート"
        baseline_days: 直近 N 日の値を基準とする

    Returns:
        dict: {
            n_races,                # 集計対象レース数
            today_time_bias,        # 当日の time_bias
            baseline_time_bias,     # 直近 baseline_days 日の time_bias
            delta_time_bias,        # 差分（正=今日は速い馬場）
            today_gate_bias,        # 当日の枠順バイアス
            today_pace_bias,        # 当日の脚質バイアス
            correction_factor,      # 1馬身=0.2秒換算の係数（speed_index 用）
            summary,                # 一言コメント
        }
    """
    # 当日結果を surface でフィルタ
    relevant = [r for r in today_results if r.get("surface") == surface
                and r.get("finish_time_sec") is not None
                and r.get("distance") is not None]
    if not relevant:
        return _empty_intraday(venue, surface, baseline_days)

    # speed_index 計算（baseline 比較）
    from speed_index import _load_baselines
    base = _load_baselines()
    base_idx = base.set_index(["venue", "surface", "distance", "track_condition"])

    speed_indices = []
    winner_nos = []
    corner4s = []
    field_sizes = []
    for r in relevant:
        try:
            b = base_idx.loc[
                (venue, surface, int(r["distance"]), r.get("track_condition") or "良"),
                "baseline_sec",
            ]
            if pd.notna(b):
                sidx = (b - r["finish_time_sec"]) * 10.0
                speed_indices.append(sidx)
        except (KeyError, ValueError):
            pass
        if r.get("winner_horse_no"):
            winner_nos.append(r["winner_horse_no"])
        if r.get("winner_corner4_pos"):
            corner4s.append(r["winner_corner4_pos"])
        if r.get("field_size"):
            field_sizes.append(r["field_size"])

    today_time_bias = round(float(np.mean(speed_indices)), 1) if speed_indices else None

    # 枠順・脚質バイアス
    today_gate_bias = None
    if winner_nos and field_sizes:
        avg_winner = float(np.mean(winner_nos))
        avg_fs = float(np.mean(field_sizes))
        today_gate_bias = round((avg_winner - avg_fs / 2) / max(avg_fs / 2, 1), 3)

    today_pace_bias = None
    if corner4s and field_sizes:
        today_pace_bias = round(float(np.mean(corner4s)) / max(float(np.mean(field_sizes)), 1), 3)

    # baseline（直近 N 日）
    baseline = get_recent_variant(venue=venue, surface=surface, n_days=baseline_days)
    baseline_time_bias = baseline.get("time_bias")

    delta = None
    if today_time_bias is not None and baseline_time_bias is not None:
        delta = round(today_time_bias - baseline_time_bias, 1)

    # 補正係数：speed_index 単位での差分（10×秒）→ 0.1 で 1 ポイント = 0.01秒/100m
    correction_factor = round(delta / 10.0, 2) if delta is not None else 0.0

    # サマリ
    if delta is None:
        summary = f"当日 {len(relevant)}R 集計済み（baseline なし）"
    elif delta > 5:
        summary = f"当日は高速馬場（baseline +{delta}）"
    elif delta < -5:
        summary = f"当日は時計かかる（baseline {delta:+}）"
    else:
        summary = f"baseline と同水準（{delta:+}）"

    return {
        "n_races": len(relevant),
        "today_time_bias": today_time_bias,
        "baseline_time_bias": baseline_time_bias,
        "delta_time_bias": delta,
        "today_gate_bias": today_gate_bias,
        "today_pace_bias": today_pace_bias,
        "correction_factor": correction_factor,
        "summary": summary,
    }


def _empty_intraday(venue: str, surface: str, baseline_days: int) -> dict:
    baseline = get_recent_variant(venue=venue, surface=surface, n_days=baseline_days)
    return {
        "n_races": 0,
        "today_time_bias": None,
        "baseline_time_bias": baseline.get("time_bias"),
        "delta_time_bias": None,
        "today_gate_bias": None,
        "today_pace_bias": None,
        "correction_factor": 0.0,
        "summary": "当日終了レースなし",
    }


if __name__ == "__main__":
    import sys
    since = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Building track variant table (since={since}) ...")
    d = build_variant_table(since_date=since)
    print(f"  {len(d):,} 行構築完了")
    print(f"\n=== 最新10件（東京芝）===")
    sub = d[(d["venue"] == "東京") & (d["surface"] == "芝")].sort_values("race_date", ascending=False).head(10)
    print(sub.to_string(index=False))
