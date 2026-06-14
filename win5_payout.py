"""WIN5 配当統計・期待値推定モジュール。

CO=0時の想定平均配当を、過去配当データまたはハードコード統計値から取得する。
"""
from __future__ import annotations

# 直近1年（2024-2025）のJRA公式WIN5配当統計（参考値）
# 中央値: 約110万円、平均: 約280万円、最頻帯: 50-200万円
PAYOUT_STATS = {
    "median":  1_100_000,    # 中央値
    "mean":    2_800_000,    # 平均
    "p25":       400_000,    # 第1四分位
    "p75":     2_500_000,    # 第3四分位
}


def auto_select_payout_mode(conf: dict) -> tuple[str, str]:
    """横断スコア(conf)から想定配当モードを自動選択。

    Returns: (mode_key, 理由文字列)
    """
    easy = conf.get("easy_count", 0)
    tough = conf.get("tough_count", 0)
    avg = conf.get("top1_avg", 0.30)

    if tough >= 4:
        return ("mean", f"混戦{tough}本（高配当期待）")
    if tough >= 2:
        return ("p75", f"混戦{tough}本（やや高配当寄り）")
    if easy >= 4:
        return ("p25", f"堅{easy}本（低配当想定）")
    if easy >= 2 and tough == 0:
        return ("median", f"堅{easy}本 + 混戦0（標準）")
    return ("median", f"標準難易度（平均top1={avg*100:.0f}%）")


def estimate_expected_payout(co_yen: int | None = None,
                              mode: str = "median") -> int:
    """CO=0時の想定配当 + キャリーオーバー額を加算した期待配当を返す。

    Parameters
    ----------
    co_yen : キャリーオーバー額（円）。None or 0 なら統計値ベース。
    mode : 'median' | 'mean' | 'p25' | 'p75'
    """
    base = PAYOUT_STATS.get(mode, PAYOUT_STATS["median"])
    return int(base + (co_yen or 0))


def compute_ev(hit_prob: float, expected_payout: int, cost_yen: int) -> dict:
    """的中確率・想定配当・投資額から期待値メトリクスを返す。"""
    ev_gross = hit_prob * expected_payout
    ev_net = ev_gross - cost_yen
    roi = (ev_gross / cost_yen - 1.0) if cost_yen > 0 else 0.0
    return {
        "ev_gross": ev_gross,
        "ev_net":   ev_net,
        "roi":      roi,
        "edge_pct": (ev_gross - cost_yen) / cost_yen * 100 if cost_yen > 0 else 0.0,
    }


def cross_race_confidence(races_probs: list[list[tuple[str, float]]]) -> dict:
    """5レース横断の確信度スコアを返す。

    閾値は **絶対値+相対値の併用**:
    - 絶対値: top1 >= 0.35 で堅い、top1 < 0.22 で混戦
    - 相対値: 平均より +5pt 以上で「相対堅い」、-5pt以下で「相対混戦」
    どちらかに該当すれば該当カウントに含める。
    """
    top1_probs, top2_gap = [], []
    for race in races_probs:
        if not race:
            continue
        sorted_p = sorted([p for _, p in race], reverse=True)
        top1_probs.append(sorted_p[0])
        # top1 と top2 の差（小さい=混戦）
        gap = (sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else sorted_p[0]
        top2_gap.append(gap)

    if not top1_probs:
        return {"easy_count": 0, "tough_count": 0, "top1_avg": 0,
                "top1_min": 0, "complexity": 1.0, "top2_gap_avg": 0,
                "per_race": []}

    avg = sum(top1_probs) / len(top1_probs)
    mn = min(top1_probs)

    # 絶対閾値（緩めに）+ 相対閾値（平均±5pt）
    easy = sum(1 for p in top1_probs if (p >= 0.35) or (p >= avg + 0.05))
    tough = sum(1 for p, g in zip(top1_probs, top2_gap)
                if (p < 0.22) or (g < 0.03) or (p <= avg - 0.05))

    complexity = 1.0 - avg
    per_race = []
    for p, g in zip(top1_probs, top2_gap):
        if (p < 0.22) or (g < 0.03) or (p <= avg - 0.05):
            label = "混戦"
        elif (p >= 0.35) or (p >= avg + 0.05):
            label = "堅"
        else:
            label = "中"
        per_race.append({"top1": p, "gap": g, "label": label})

    return {
        "easy_count":  easy,
        "tough_count": tough,
        "top1_avg":    avg,
        "top1_min":    mn,
        "complexity":  complexity,
        "top2_gap_avg": sum(top2_gap) / len(top2_gap),
        "per_race":    per_race,
    }
