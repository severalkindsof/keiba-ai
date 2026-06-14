"""
IMPROVE-5: Multi-Bet Portfolio Kelly（Benter 流）

同日複数レースを買う時、独立 Kelly では合計賭け率が予算の 30%超えるケースがある。
ポートフォリオ全体でのドローダウンを抑えるため、合計賭け率に上限を設ける。

公式:
    total_kelly = Σ f_i
    if total_kelly > MAX_PORTFOLIO_EXPOSURE:
        scale = MAX_PORTFOLIO_EXPOSURE / total_kelly
        f_i_adjusted = f_i * scale

使い方:
    from portfolio_kelly import normalize_portfolio
    adjusted_bets = normalize_portfolio(bets, max_exposure=0.20, fraction=0.5)
"""
from dataclasses import dataclass


@dataclass
class PortfolioBet:
    race_id: str
    horse_name: str
    win_prob: float        # 0〜1
    odds: float            # 単勝オッズ
    raw_kelly: float       # 単独ケリー値（0〜1）
    adjusted_kelly: float = 0.0  # ポートフォリオ調整後
    adjusted_amount: int = 0     # 円換算


def calc_kelly(win_prob: float, odds: float) -> float:
    """単独ケリー値（0〜1）。EV<=0 なら 0。"""
    if odds is None or odds <= 1.0 or win_prob <= 0 or win_prob >= 1:
        return 0.0
    b = odds - 1.0
    ev = win_prob * b - (1 - win_prob)
    if ev <= 0:
        return 0.0
    f = (win_prob * b - (1 - win_prob)) / b
    return max(0.0, min(1.0, f))


def normalize_portfolio(
    bets: list[PortfolioBet],
    bankroll: int,
    max_exposure: float = 0.20,
    fraction: float = 0.50,
    bet_unit: int = 100,
) -> list[PortfolioBet]:
    """
    ポートフォリオ Kelly：
        1. 各 bet の raw_kelly に fraction（分数ケリー）を掛ける
        2. 合計が max_exposure を超えるなら比例縮小
        3. bankroll × adjusted_kelly を bet_unit で丸めて adjusted_amount に格納

    Args:
        bets:        購入候補リスト（複数レース横断OK）
        bankroll:    総資金（円）
        max_exposure: 全体上限（0.20 = 資金の20%）
        fraction:    分数ケリー（0.50 = ハーフケリー）
        bet_unit:    最小賭け単位（100円）
    """
    if not bets:
        return bets

    # 分数ケリーを適用
    for b in bets:
        b.adjusted_kelly = b.raw_kelly * fraction

    total = sum(b.adjusted_kelly for b in bets)

    # ポートフォリオ上限を超えるなら全体を比例縮小
    if total > max_exposure and total > 0:
        scale = max_exposure / total
        for b in bets:
            b.adjusted_kelly *= scale

    # 円換算 → 最小単位で丸める
    for b in bets:
        raw_amount = bankroll * b.adjusted_kelly
        b.adjusted_amount = max(0, int(raw_amount // bet_unit) * bet_unit)

    return bets


def get_portfolio_summary(bets: list[PortfolioBet], bankroll: int) -> dict:
    """ポートフォリオ全体のサマリ"""
    total_amount = sum(b.adjusted_amount for b in bets)
    n_active = sum(1 for b in bets if b.adjusted_amount > 0)
    total_kelly = sum(b.adjusted_kelly for b in bets)
    return {
        "bankroll": bankroll,
        "total_amount": total_amount,
        "exposure_pct": round(total_amount / bankroll * 100, 1) if bankroll else 0,
        "n_active_bets": n_active,
        "total_kelly_pct": round(total_kelly * 100, 2),
        "expected_return": round(sum(
            b.win_prob * b.odds * b.adjusted_amount
            - (1 - b.win_prob) * b.adjusted_amount
            for b in bets
        )),
    }


if __name__ == "__main__":
    print("=== Portfolio Kelly テスト ===")
    # 同日 5 レースに賭ける、各 EV+
    sample = [
        PortfolioBet("R1", "馬A", 0.40, 3.0,  calc_kelly(0.40, 3.0)),   # f = 0.10
        PortfolioBet("R2", "馬B", 0.30, 4.5,  calc_kelly(0.30, 4.5)),   # f = 0.10
        PortfolioBet("R3", "馬C", 0.20, 7.0,  calc_kelly(0.20, 7.0)),   # f = 0.067
        PortfolioBet("R4", "馬D", 0.50, 2.5,  calc_kelly(0.50, 2.5)),   # f = 0.167
        PortfolioBet("R5", "馬E", 0.10, 15.0, calc_kelly(0.10, 15.0)),  # f = 0.036
    ]
    print(f"Raw Kelly 合計: {sum(b.raw_kelly for b in sample):.3f}")
    print(f"フルケリー: 合計 47%。ハーフ＋上限 20% で調整：")

    adjusted = normalize_portfolio(
        sample, bankroll=100000, max_exposure=0.20, fraction=0.50, bet_unit=100
    )
    for b in adjusted:
        print(f"  {b.horse_name}: 単独f={b.raw_kelly:.3f} → 調整後f={b.adjusted_kelly:.3f} "
              f"→ {b.adjusted_amount:,}円")
    print()
    print("Summary:", get_portfolio_summary(adjusted, 100000))
