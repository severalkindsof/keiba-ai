"""
B-3: コースバイアスの恒常推定

「日次の馬場状態」とは別に、コース固有の構造的バイアスを抽出する。
例：「京都2400外回りは外枠不利」「新潟1000直は外枠有利」など。

handles:
    - (venue, surface, distance) ごとに、1着馬の平均馬番偏差
    - 直線長・コーナー特性は別途定数表から
    - サンプル数 50 以上のみ採用

使い方:
    python course_bias.py            # 全期間で再構築
    from course_bias import get_course_bias
    bias = get_course_bias("京都", "芝", 2400)
"""
import json
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"
_OUT_PATH = _DATA_DIR / "course_bias.parquet"
_CACHE = None


def build_course_bias(
    tfjv_path: Path | None = None,
    out_path: Path | None = None,
    min_samples: int = 50,
) -> pd.DataFrame:
    """(venue, surface, distance) ごとの 1着馬偏差を集計"""
    tfjv_path = tfjv_path or (_DATA_DIR / "tfjv_all.parquet")
    out_path  = out_path  or _OUT_PATH

    df = pd.read_parquet(tfjv_path)
    df = df.dropna(subset=["venue", "surface", "distance", "rank", "horse_no"]).copy()
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df = df.dropna(subset=["rank", "horse_no"])

    # race_no 毎の field_size 再計算（winners 抽出前に df に追加）
    df["race_key"] = (
        df["date"].astype(str) + "_" + df["venue"].astype(str) + "_"
        + df.get("race_no", pd.Series([""] * len(df))).astype(str)
    )
    df["actual_field_size"] = df.groupby("race_key")["horse_no"].transform("max")

    winners = df[df["rank"] == 1].copy()
    winners["field_size"] = winners["actual_field_size"].fillna(12)

    # gate_bias = (winner_horse_no - field_size/2) / (field_size/2)
    winners["bias_raw"] = (
        (winners["horse_no"] - winners["field_size"] / 2)
        / (winners["field_size"] / 2).clip(lower=1)
    )

    g = winners.groupby(["venue", "surface", "distance"]).agg(
        n_wins=("bias_raw", "size"),
        gate_bias=("bias_raw", "mean"),
        gate_bias_std=("bias_raw", "std"),
    ).reset_index()
    g = g[g["n_wins"] >= min_samples].copy()
    g["gate_bias"]     = g["gate_bias"].round(3)
    g["gate_bias_std"] = g["gate_bias_std"].round(3)

    # ラベル
    def _label(b):
        if b >= 0.15:  return "外枠有利"
        if b <= -0.15: return "内枠有利"
        return "フラット"
    g["label"] = g["gate_bias"].apply(_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out_path, index=False)
    return g


def _load() -> pd.DataFrame:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not _OUT_PATH.exists():
        _CACHE = build_course_bias()
    else:
        _CACHE = pd.read_parquet(_OUT_PATH)
    return _CACHE


def get_course_bias(venue: str, surface: str, distance: int) -> dict:
    """指定コースの構造的バイアスを返す"""
    df = _load()
    hit = df[(df["venue"] == venue) & (df["surface"] == surface) & (df["distance"] == int(distance))]
    if hit.empty:
        return {"gate_bias": 0.0, "label": "データなし", "n_wins": 0}
    r = hit.iloc[0]
    return {
        "gate_bias": float(r["gate_bias"]),
        "label":     str(r["label"]),
        "n_wins":    int(r["n_wins"]),
    }


if __name__ == "__main__":
    print("Building course_bias.parquet ...")
    g = build_course_bias()
    print(f"  {len(g)} コースパターン")
    print("\n--- 内枠有利 TOP 5 ---")
    print(g[g["label"] == "内枠有利"].sort_values("gate_bias").head(5).to_string(index=False))
    print("\n--- 外枠有利 TOP 5 ---")
    print(g[g["label"] == "外枠有利"].sort_values("gate_bias", ascending=False).head(5).to_string(index=False))
