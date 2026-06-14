"""
市場インプライド確率の経験値テーブル。

Benter Odds Blending の π_i（市場確率）として、人気順位ごとの実勝率を使う。
これは「公衆の知恵」を定量化したベースライン値。

使い方:
    from market_prob import get_market_prob
    prob = get_market_prob(popularity=3)   # 人気3位なら 0.13 など

注意:
    - 人気→実勝率の対応は過去16年分の TFJV データから計算
    - 「実勝率」とは「人気X位だった馬のうち実際に1着になった割合」
    - 単純なオッズ逆数より favorite-longshot bias が補正された値になる
"""
from pathlib import Path
import pandas as pd
import numpy as np

_DATA_DIR = Path(__file__).parent / "data"
_MARKET_PATH = _DATA_DIR / "market_prob_by_popularity.parquet"

# キャッシュ（初回ロードのみ）
_MARKET_TABLE: pd.DataFrame | None = None


def build_market_prob_table(
    tfjv_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    TFJVデータから「人気順位ごとの実勝率」を計算してテーブル化。
    出力: pop, n, wins, win_rate, market_prob（=win_rate）
    """
    tfjv_path = tfjv_path or (_DATA_DIR / "tfjv_all.parquet")
    out_path  = out_path  or _MARKET_PATH

    df = pd.read_parquet(tfjv_path)
    df = df[(df["popularity"] > 0) & (df["popularity"] <= 18) & df["rank"].notna()].copy()
    g = df.groupby("popularity").agg(
        n=("rank", "size"),
        wins=("rank", lambda x: (x == 1).sum()),
    )
    g["win_rate"] = g["wins"] / g["n"]
    g["market_prob"] = g["win_rate"]
    g = g.reset_index()

    out_path.parent.mkdir(exist_ok=True, parents=True)
    g.to_parquet(out_path, index=False)
    return g


def _load_table() -> pd.DataFrame:
    """テーブルをロード（無ければ生成）"""
    global _MARKET_TABLE
    if _MARKET_TABLE is not None:
        return _MARKET_TABLE
    if not _MARKET_PATH.exists():
        _MARKET_TABLE = build_market_prob_table()
    else:
        _MARKET_TABLE = pd.read_parquet(_MARKET_PATH)
    return _MARKET_TABLE


def get_market_prob(popularity: int | float, default: float = 0.05) -> float:
    """
    人気順位から市場インプライド確率を取得。
    人気不明（0 / None / NaN）の場合は default を返す。
    """
    try:
        pop = int(popularity)
    except (TypeError, ValueError):
        return default
    if pop < 1 or pop > 18:
        return default
    df = _load_table()
    row = df[df["popularity"] == pop]
    if row.empty:
        return default
    return float(row["market_prob"].iloc[0])


def get_market_prob_series(popularities) -> pd.Series:
    """ベクトル版"""
    return pd.Series(popularities).apply(get_market_prob)


if __name__ == "__main__":
    print("Building market_prob_by_popularity.parquet ...")
    t = build_market_prob_table()
    print(t.to_string(index=False))
    print(f"\nSaved to {_MARKET_PATH}")
