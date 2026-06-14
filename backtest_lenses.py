"""レンズ・バックテスト基盤（過去データで各レンズの捕捉率・精度を測る）

配当データがないため、人気で「穴決着」を代理。
任意のレンズ関数について「フラグ馬の3着内率・勝率」をベースラインと比較する。

使い方:
    python -X utf8 backtest_lenses.py            # elite_neglect を評価
    from backtest_lenses import evaluate_lens
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

_DIR = Path(__file__).parent


def load_results(since: str = "2023-01-01", min_field: int = 10) -> pd.DataFrame:
    """着順・人気・Elo付きの結果データ（フルゲートのみ）。"""
    df = pd.read_parquet(_DIR / "data/tfjv_all.parquet",
                         columns=["race_id", "date", "rank", "popularity",
                                  "horse_name", "venue", "surface", "distance"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] >= since].copy()
    df["rkey"] = df["race_id"].astype(str).str[:8]
    df["rk"] = pd.to_numeric(df["rank"], errors="coerce")
    df["pop"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["hn"] = df["horse_name"].astype(str).str.strip()
    e = pd.read_parquet(_DIR / "data/horse_elo.parquet")
    emap = dict(zip(e["horse_name"].astype(str).str.strip(), e["elo"]))
    df["elo"] = df["hn"].map(emap)
    fs = df.groupby("rkey").size()
    df = df[df["rkey"].isin(fs[fs >= min_field].index)]
    return df


def upset_races(df: pd.DataFrame, pop_max_min: int = 10) -> pd.DataFrame:
    """穴決着レース（3着内に pop_max_min 人気以上が入ったレース）の rkey 一覧。"""
    top3 = df[df["rk"] <= 3]
    g = top3.groupby("rkey")["pop"].max()
    return g[g >= pop_max_min].index


def summary(flagged: pd.DataFrame, df: pd.DataFrame, label: str):
    """フラグ馬の成績サマリを表示。"""
    if len(flagged) == 0:
        print(f"{label}: フラグ0"); return
    base = df[df["pop"] >= 7]
    print(f"=== {label} ===")
    print(f"  フラグ数:{len(flagged)}  3着内率:{(flagged['rk']<=3).mean()*100:.1f}%  "
          f"勝率:{(flagged['rk']==1).mean()*100:.1f}%  平均人気:{flagged['pop'].mean():.1f}")
    print(f"  (基準)7人気以下: 3着内{(base['rk']<=3).mean()*100:.1f}% 勝{(base['rk']==1).mean()*100:.1f}%")


def evaluate_elite_neglect(elo_floor=2400, pop_min=7, top_k=2, since="2023-01-01"):
    df = load_results(since)
    neg = df[(df["pop"] >= pop_min) & (df["elo"] >= elo_floor)].copy()
    neg["nr"] = neg.groupby("rkey")["elo"].rank(method="first", ascending=False)
    flagged = neg[neg["nr"] <= top_k]
    summary(flagged, df, f"elite_neglect (elo>={elo_floor}, top{top_k}, {since}-)")
    return flagged


if __name__ == "__main__":
    evaluate_elite_neglect()
    print()
    # 穴決着レースに限った検証（10人気+が来たレースで筆頭フラグが当たったか）
    df = load_results()
    up = upset_races(df, 10)
    print(f"穴決着レース(3着内10人気+): {len(up)}レース")
    sub = df[df["rkey"].isin(up)]
    neg = sub[(sub["pop"] >= 7) & (sub["elo"] >= 2400)].copy()
    neg["nr"] = neg.groupby("rkey")["elo"].rank(method="first", ascending=False)
    top1 = neg[neg["nr"] == 1]
    print(f"  筆頭フラグの3着内率: {(top1['rk']<=3).mean()*100:.1f}% (穴決着限定)")
