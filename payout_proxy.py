"""三連複 配当proxy（実配当データなしでHarvilleモデルで推定）

人気→実勝率(market_prob_by_popularity)から、top3の三連複出現確率を
Harville近似で計算し、配当 ≈ 控除後還元率0.75 ÷ 出現確率 で推定。
配当そのものは無いが「どれくらい荒れた決着か」を金額スケールで代理できる。
"""
from __future__ import annotations
import pandas as pd
from itertools import permutations
from pathlib import Path

_DIR = Path(__file__).parent
_PMAP = None


def _pmap():
    global _PMAP
    if _PMAP is None:
        m = pd.read_parquet(_DIR / "data/market_prob_by_popularity.parquet")
        _PMAP = dict(zip(m["popularity"].astype(int), m["win_rate"]))
    return _PMAP


def trifecta_box_prob(pops: list[int]) -> float:
    """top3の3頭(人気)から三連複出現確率をHarville近似で計算。"""
    pm = _pmap()
    p = [pm.get(int(x), 0.001) for x in pops]
    if any(v <= 0 for v in p):
        return 0.0
    total = 0.0
    for a, b, c in permutations(range(3)):
        pa, pb, pc = p[a], p[b], p[c]
        denom1 = 1 - pa
        denom2 = 1 - pa - pb
        if denom1 <= 0 or denom2 <= 0:
            continue
        total += pa * (pb / denom1) * (pc / denom2)
    return total


def proxy_payout(pops: list[int], takeout_return: float = 0.75) -> int:
    """三連複 推定配当（円/100円）。"""
    prob = trifecta_box_prob(pops)
    if prob <= 0:
        return 0
    return int(takeout_return / prob * 100)


if __name__ == "__main__":
    tests = [
        ("1-2-3 鉄板", [1, 2, 3]),
        ("安田2026 (8-1-7)", [8, 1, 7]),
        ("VM2024 (14-4-1)", [14, 4, 1]),
        ("超大荒れ (15-12-10)", [15, 12, 10]),
    ]
    for label, pops in tests:
        print(f"  {label}: 三連複proxy ≈ {proxy_payout(pops):,}円 (出現確率{trifecta_box_prob(pops)*100:.3f}%)")
