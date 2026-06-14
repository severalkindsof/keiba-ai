"""
NEW-8: 厩舎×騎手 ペア成績マトリクス

(trainer, jockey) ペアの過去成績を集計し、好相性/不相性/初コンビ判定を提供。

公式・出典:
    - 亀谷敬正氏「最重要ファクター」シリーズで「厩舎×騎手の固定パターン」が
      回収率に効くと指摘
    - 短期免許騎手や特定厩舎の若手が当たる頻度を捉える

使い方:
    python trainer_jockey_matrix.py   # 一括計算 → data/trainer_jockey_matrix.parquet 保存

    from trainer_jockey_matrix import get_pair_stats
    stats = get_pair_stats(trainer="○○", jockey="△△")
"""
import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_MATRIX_PATH = _DATA_DIR / "trainer_jockey_matrix.parquet"

# キャッシュ
_MATRIX_CACHE: pd.DataFrame | None = None
_OVERALL_AVG_WR: float | None = None


def build_matrix(
    tfjv_path: Path | None = None,
    out_path: Path | None = None,
    min_pairs: int = 5,
) -> pd.DataFrame:
    """
    厩舎×騎手ペアの集計テーブルを構築。
    Returns columns: trainer, jockey, rides, wins, places, win_rate, place_rate, lift, label
    """
    tfjv_path = tfjv_path or (_DATA_DIR / "tfjv_all.parquet")
    out_path  = out_path  or _MATRIX_PATH

    df = pd.read_parquet(tfjv_path)
    df = df.dropna(subset=["trainer", "jockey", "rank"]).copy()
    df["trainer"] = df["trainer"].astype(str).str.strip()
    df["jockey"]  = df["jockey"].astype(str).str.strip()
    df = df[(df["trainer"] != "") & (df["jockey"] != "")]

    win = (df["rank"].astype(int) == 1).astype(int)
    place = (df["rank"].astype(int) <= 3).astype(int)
    df["_win"] = win
    df["_place"] = place

    g = df.groupby(["trainer", "jockey"]).agg(
        rides=("_win", "size"),
        wins=("_win", "sum"),
        places=("_place", "sum"),
    ).reset_index()
    g = g[g["rides"] >= min_pairs].copy()
    g["win_rate"]   = g["wins"]   / g["rides"]
    g["place_rate"] = g["places"] / g["rides"]

    overall_wr = float(df["_win"].mean())
    g["lift"] = g["win_rate"] / overall_wr  # 1.5+ = ペアで好成績

    def _label(row):
        if row["rides"] < 10:
            return "サンプル少"
        if row["lift"] >= 2.0:
            return "黄金コンビ"
        if row["lift"] >= 1.4:
            return "好相性"
        if row["lift"] <= 0.5:
            return "不相性"
        return "標準"
    g["label"] = g.apply(_label, axis=1)

    g = g.sort_values("lift", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(out_path, index=False)

    # 全体平均勝率もメタとして保存
    meta_path = _DATA_DIR / "trainer_jockey_matrix_meta.json"
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"overall_win_rate": overall_wr, "n_pairs": len(g)}, f)
    return g


def _load_matrix() -> pd.DataFrame:
    global _MATRIX_CACHE, _OVERALL_AVG_WR
    if _MATRIX_CACHE is not None:
        return _MATRIX_CACHE
    if not _MATRIX_PATH.exists():
        _MATRIX_CACHE = build_matrix()
    else:
        _MATRIX_CACHE = pd.read_parquet(_MATRIX_PATH)
    # メタ
    meta_path = _DATA_DIR / "trainer_jockey_matrix_meta.json"
    if meta_path.exists():
        import json
        try:
            _OVERALL_AVG_WR = float(json.load(open(meta_path, encoding="utf-8")).get("overall_win_rate", 0.08))
        except Exception:
            _OVERALL_AVG_WR = 0.08
    else:
        _OVERALL_AVG_WR = 0.08
    return _MATRIX_CACHE


def get_pair_stats(trainer: str, jockey: str) -> dict:
    """
    特定の (trainer, jockey) ペアの統計を取得。
    Returns:
        {
            "is_known_pair": bool,
            "rides": int, "wins": int, "win_rate": float, "lift": float,
            "label": str, "bonus": float (0〜0.04),
            "note": str
        }
    """
    if not trainer or not jockey:
        return _empty_pair_result()
    df = _load_matrix()
    sub = df[(df["trainer"] == str(trainer).strip()) & (df["jockey"] == str(jockey).strip())]
    if sub.empty:
        return {
            "is_known_pair": False,
            "rides": 0, "wins": 0, "win_rate": 0.0, "lift": 1.0,
            "label": "初コンビ", "bonus": 0.0,
            "note": "過去5走以上の組み合わせ実績なし（初コンビ or 少データ）",
        }
    row = sub.iloc[0]
    lift = float(row["lift"])
    # ボーナス: lift 2.0+ で +0.03、1.5+ で +0.02、0.5- で -0.02
    if lift >= 2.0:
        bonus = 0.03
    elif lift >= 1.5:
        bonus = 0.02
    elif lift >= 1.2:
        bonus = 0.01
    elif lift <= 0.5:
        bonus = -0.02
    else:
        bonus = 0.0
    return {
        "is_known_pair": True,
        "rides": int(row["rides"]),
        "wins":  int(row["wins"]),
        "win_rate": round(float(row["win_rate"]), 3),
        "lift":     round(lift, 2),
        "label":    str(row["label"]),
        "bonus":    bonus,
        "note": (f"{row['rides']}回騎乗で勝率 {row['win_rate']*100:.1f}% "
                 f"（全体平均比 ×{lift:.2f}）"),
    }


def _empty_pair_result() -> dict:
    return {
        "is_known_pair": False, "rides": 0, "wins": 0,
        "win_rate": 0.0, "lift": 1.0, "label": "データなし",
        "bonus": 0.0, "note": "",
    }


if __name__ == "__main__":
    print("Building trainer × jockey matrix ...")
    g = build_matrix()
    print(f"  {len(g):,} ペア構築完了")
    print(f"\n=== Top 20 黄金コンビ（騎乗20回以上） ===")
    top = g[g["rides"] >= 20].head(20)
    for _, r in top.iterrows():
        print(f"  {r['trainer']:14s} × {r['jockey']:14s}  "
              f"勝率{r['win_rate']*100:5.1f}%  ×{r['lift']:.2f}  "
              f"({r['rides']:4d}騎乗)  [{r['label']}]")
    print(f"\nSaved to {_MATRIX_PATH}")
