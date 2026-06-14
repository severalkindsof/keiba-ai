"""WIN5 3点組合せ最適化。

5レース分の (馬名, 1着確率) リストを受け取り、合計 ≤ 3点 の組合せで
合算的中確率（5レースの選択馬1着確率の積、軸複数頭の場合は和）を最大化する。

許容パターン（積≤3）:
  1×1×1×1×1=1点   全本命1点流し
  2×1×1×1×1=2点   1レースで2頭軸
  3×1×1×1×1=3点   1レースで3頭軸
"""
from __future__ import annotations
from itertools import combinations


def _race_top_n(horses: list[tuple[str, float]], n: int) -> tuple[list[str], float]:
    """1着確率上位n頭の (馬名リスト, 和) を返す。"""
    sorted_h = sorted(horses, key=lambda x: -x[1])[:n]
    return [h[0] for h in sorted_h], sum(p for _, p in sorted_h)


def optimize_win5(races: list[list[tuple[str, float]]],
                  max_points: int = 3) -> dict:
    """5レース分の組合せを最適化。

    Parameters
    ----------
    races : 各レースの [(馬名, 1着確率), ...] のリスト（長さ5）
    max_points : 最大点数（デフォルト3）

    Returns
    -------
    {
      "selections": [[馬名,...], [馬名,...], ...],  # 各レースの軸馬
      "points":     int,                              # 総点数
      "hit_prob":   float,                            # 的中確率
      "cost_yen":   int,                              # 投資額（点数×100）
      "ev_pattern": str,                              # "3×1×1×1×1" 等
    }
    """
    assert len(races) == 5, "WIN5 は5レース固定"

    # 各レース × 軸頭数(1..max_points) の (馬名リスト, 確率和) を事前計算
    options = []
    for race in races:
        opts = {}
        for n in range(1, max_points + 1):
            if len(race) >= n:
                horses, prob = _race_top_n(race, n)
                opts[n] = (horses, prob)
            else:
                opts[n] = (None, 0.0)
        options.append(opts)

    best = None
    best_prob = -1.0

    # 軸頭数の組合せ：積 ≤ max_points
    # 効率的に全探索（max_points=3 なら 5^5 上限）
    def _iter_alloc(remaining_budget, idx, current):
        if idx == 5:
            yield current[:]
            return
        for n in range(1, remaining_budget + 1):
            current.append(n)
            yield from _iter_alloc(remaining_budget // n, idx + 1, current)
            current.pop()

    for alloc in _iter_alloc(max_points, 0, []):
        product = 1
        for n in alloc:
            product *= n
        if product > max_points:
            continue
        prob = 1.0
        sel = []
        for race_idx, n in enumerate(alloc):
            horses, p = options[race_idx][n]
            if horses is None:
                prob = 0.0
                break
            sel.append(horses)
            prob *= p
        if prob > best_prob:
            best_prob = prob
            best = (alloc, sel, product)

    if best is None:
        return {"selections": [], "points": 0, "hit_prob": 0.0,
                "cost_yen": 0, "ev_pattern": ""}

    alloc, sel, points = best
    return {
        "selections": sel,
        "points":     points,
        "hit_prob":   best_prob,
        "cost_yen":   points * 100,
        "ev_pattern": "×".join(str(n) for n in alloc),
    }


if __name__ == "__main__":
    # テスト：5レースの仮データ
    races = [
        [("A1", 0.40), ("A2", 0.30), ("A3", 0.10)],
        [("B1", 0.55), ("B2", 0.20)],
        [("C1", 0.25), ("C2", 0.22), ("C3", 0.20), ("C4", 0.15)],  # 混戦
        [("D1", 0.60), ("D2", 0.15)],
        [("E1", 0.45), ("E2", 0.25)],
    ]
    result = optimize_win5(races, max_points=3)
    print(f"配分: {result['ev_pattern']} = {result['points']}点 / {result['cost_yen']}円")
    print(f"的中確率: {result['hit_prob']*100:.3f}%")
    for i, horses in enumerate(result["selections"], 1):
        print(f"  R{i}: {horses}")
