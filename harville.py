"""
SUPER-4: Harville (1973) モデルによる順位確率の計算。

単勝勝率 p_i から「i が 1着・j が 2着・k が 3着になる確率」を導出する標準的手法。
Plackett-Luce モデルとも呼ばれる。

公式:
    P(i wins) = p_i
    P(i 1st, j 2nd)         = p_i · p_j / (1 - p_i)
    P(i 1st, j 2nd, k 3rd)  = p_i · p_j/(1-p_i) · p_k/(1-p_i-p_j)

各馬券種:
    馬連 {i,j}    = P(i 1st, j 2nd) + P(j 1st, i 2nd)
    馬単 i→j      = P(i 1st, j 2nd)
    ワイド {i,j}  = P(両方 top3)
    三連複 {i,j,k}= 6順列の合計
    三連単 i→j→k = そのままの順列

Harville は「2着確率を過小推定、3着確率を過大推定」の既知バイアスあり。
本実装では Discounted Harville（補正項なし、純粋な Harville）を採用。

使い方:
    from harville import top_n_combinations
    combos = top_n_combinations(win_probs, ticket_type="trifecta", top_n=6)
    # → [{"horses": (1,3,5), "prob": 0.012, "fair_odds": 83.3}, ...]
"""
from itertools import permutations, combinations
import numpy as np


def _normalize(p):
    """確率配列を合計1に正規化"""
    p = np.asarray(p, dtype=float)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))


# ============================================================
# 基本: 単勝確率から各順位確率
# ============================================================

def exacta_prob(win_probs, i: int, j: int) -> float:
    """馬単: i 1着 → j 2着"""
    p = _normalize(win_probs)
    if i == j or p[i] >= 1.0:
        return 0.0
    denom = 1.0 - p[i]
    if denom <= 0:
        return 0.0
    return float(p[i] * p[j] / denom)


def quinella_prob(win_probs, i: int, j: int) -> float:
    """馬連: i, j が 1-2 着（順序不問）"""
    return exacta_prob(win_probs, i, j) + exacta_prob(win_probs, j, i)


def trifecta_prob(win_probs, i: int, j: int, k: int) -> float:
    """三連単: i → j → k"""
    p = _normalize(win_probs)
    if len({i, j, k}) < 3:
        return 0.0
    if p[i] >= 1.0:
        return 0.0
    d1 = 1.0 - p[i]
    d2 = 1.0 - p[i] - p[j]
    if d1 <= 0 or d2 <= 0:
        return 0.0
    return float(p[i] * (p[j] / d1) * (p[k] / d2))


def trio_prob(win_probs, i: int, j: int, k: int) -> float:
    """三連複: {i, j, k} が 1-3 着（順序不問）→ 6順列の和"""
    if len({i, j, k}) < 3:
        return 0.0
    return sum(
        trifecta_prob(win_probs, a, b, c)
        for a, b, c in permutations((i, j, k))
    )


def top3_prob(win_probs, i: int) -> float:
    """i が 1〜3 着以内に入る確率（複勝確率）"""
    n = len(win_probs)
    p = _normalize(win_probs)
    # 1着 or (他が1着) → (i が 2着) or (他2頭が1-2着) → (i が 3着)
    total = float(p[i])  # 1着
    for j in range(n):
        if j == i:
            continue
        total += exacta_prob(p, j, i)  # j 1着, i 2着
        for k in range(n):
            if k in (i, j):
                continue
            total += trifecta_prob(p, j, k, i)  # j → k → i
    return total


def wide_prob(win_probs, i: int, j: int) -> float:
    """ワイド: i と j が両方 1〜3 着以内"""
    if i == j:
        return 0.0
    p = _normalize(win_probs)
    n = len(p)
    # 6つのパターン: 2人で3つの席を埋める
    total = 0.0
    # i 1着 j 2or3着
    total += exacta_prob(p, i, j)  # i→j
    for k in range(n):
        if k in (i, j):
            continue
        total += trifecta_prob(p, i, k, j)  # i→k→j
    # j 1着 i 2or3着
    total += exacta_prob(p, j, i)
    for k in range(n):
        if k in (i, j):
            continue
        total += trifecta_prob(p, j, k, i)
    # 他が1着で i,j が 2,3着
    for k in range(n):
        if k in (i, j):
            continue
        total += trifecta_prob(p, k, i, j) + trifecta_prob(p, k, j, i)
    return total


# ============================================================
# 列挙: 上位N頭の組合せ全て + EV計算
# ============================================================

def _fair_odds(prob: float, take_rate: float = 0.20) -> float:
    """確率→公平オッズ（控除率込み）"""
    if prob <= 0:
        return float("inf")
    return (1.0 - take_rate) / prob


def top_n_combinations(
    win_probs,
    ticket_type: str = "trio",
    top_n: int = 8,
    horse_nos=None,
    take_rate: float = 0.20,
) -> list[dict]:
    """
    上位N頭から ticket_type の全組合せを列挙、Harville確率と公平オッズを返す。

    Args:
        win_probs: 全頭の単勝勝率配列（合計≒1、indexがそのまま馬index）
        ticket_type: "win" / "quinella" / "exacta" / "wide" / "trio" / "trifecta"
        top_n: 上位何頭を対象にするか
        horse_nos: 馬番リスト（表示用、Noneなら index+1）
        take_rate: 控除率（JRA は券種により 0.20〜0.275）

    Returns:
        [{"horses": tuple, "prob": float, "fair_odds": float}, ...] prob 降順
    """
    p = _normalize(win_probs)
    n = len(p)
    if horse_nos is None:
        horse_nos = list(range(1, n + 1))
    # 上位 top_n 頭の index
    order = np.argsort(-p)[:min(top_n, n)]

    combos = []
    if ticket_type == "win":
        for i in order:
            combos.append({"horses": (horse_nos[i],), "prob": float(p[i])})
    elif ticket_type == "quinella":
        for i, j in combinations(order, 2):
            pr = quinella_prob(p, i, j)
            combos.append({"horses": tuple(sorted([horse_nos[i], horse_nos[j]])), "prob": pr})
    elif ticket_type == "exacta":
        for i, j in permutations(order, 2):
            pr = exacta_prob(p, i, j)
            combos.append({"horses": (horse_nos[i], horse_nos[j]), "prob": pr})
    elif ticket_type == "wide":
        for i, j in combinations(order, 2):
            pr = wide_prob(p, i, j)
            combos.append({"horses": tuple(sorted([horse_nos[i], horse_nos[j]])), "prob": pr})
    elif ticket_type == "trio":
        for i, j, k in combinations(order, 3):
            pr = trio_prob(p, i, j, k)
            combos.append({"horses": tuple(sorted([horse_nos[i], horse_nos[j], horse_nos[k]])), "prob": pr})
    elif ticket_type == "trifecta":
        for i, j, k in permutations(order, 3):
            pr = trifecta_prob(p, i, j, k)
            combos.append({"horses": (horse_nos[i], horse_nos[j], horse_nos[k]), "prob": pr})
    else:
        raise ValueError(f"unknown ticket_type: {ticket_type}")

    # 公平オッズ計算
    for c in combos:
        c["fair_odds"] = _fair_odds(c["prob"], take_rate=take_rate)
    combos.sort(key=lambda x: -x["prob"])
    return combos


# 控除率（JRA 2026年時点）
# 第21波修正: JRA 公式払戻率に準拠（馬単・三連複は 25% — 22.5% は誤りで
# 公平オッズが甘く出て EV を過大評価していた）
TAKE_RATES = {
    "win":      0.20,   # 単勝 (払戻率80%)
    "place":    0.20,   # 複勝 (80%)
    "quinella": 0.225,  # 馬連 (77.5%)
    "exacta":   0.25,   # 馬単 (75%)
    "wide":     0.225,  # ワイド (77.5%)
    "trio":     0.25,   # 三連複 (75%)
    "trifecta": 0.275,  # 三連単 (72.5%)
}

TICKET_LABELS_JA = {
    "win":      "単勝",
    "place":    "複勝",
    "quinella": "馬連",
    "exacta":   "馬単",
    "wide":     "ワイド",
    "trio":     "三連複",
    "trifecta": "三連単",
}


def calc_ev_for_market_odds(prob: float, market_odds: float) -> float:
    """市場オッズに対する EV を計算"""
    if market_odds is None or market_odds <= 1.0 or prob <= 0:
        return float("nan")
    return prob * (market_odds - 1) - (1 - prob)


if __name__ == "__main__":
    # サンプル: 単勝勝率 [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]
    wp = [0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]
    print("=== 三連複 上位5（Harville） ===")
    for c in top_n_combinations(wp, "trio", top_n=5)[:5]:
        print(f"  {c['horses']}  prob={c['prob']:.4f}  fair_odds={c['fair_odds']:.1f}倍")
    print("\n=== 馬連 上位5 ===")
    for c in top_n_combinations(wp, "quinella", top_n=5)[:5]:
        print(f"  {c['horses']}  prob={c['prob']:.4f}  fair_odds={c['fair_odds']:.1f}倍")
    print("\n=== 馬単 上位5 ===")
    for c in top_n_combinations(wp, "exacta", top_n=5)[:5]:
        print(f"  {c['horses']}  prob={c['prob']:.4f}  fair_odds={c['fair_odds']:.1f}倍")
