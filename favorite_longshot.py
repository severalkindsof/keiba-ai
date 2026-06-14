"""
SUPER-5: Favorite-Longshot Bias 補正

学術的に知られた事実：
    競馬ファンは「人気薄を過大評価、人気馬を過小評価」する傾向がある。
    Snowberg & Wolfers (2010) ら多数の研究で確認済み。

本モジュールは「市場オッズから素朴に逆算した implied prob」を
「文献に基づく真の勝率」に補正する。

補正係数（Snowberg-Wolfers 2010 ベース）:
    ~1.5倍以下: factor 1.10  （本命は実勝率高い）
    1.5-5倍   : factor 1.05
    5-20倍    : factor 1.00  （中位は概ね適正）
    20-50倍   : factor 0.85
    50-100倍  : factor 0.70
    100倍超   : factor 0.50  （超大穴は実勝率が公示より低い）

使い方:
    from favorite_longshot import correct_from_odds
    true_prob = correct_from_odds(odds=3.5)

注意:
    market_prob.py（人気→経験勝率）は経験値ベースなので
    favorite-longshot 補正が既に暗黙的に含まれている。
    本モジュールは「人気が無い直接オッズ入力時」の代替手段。
"""
import numpy as np


# Snowberg-Wolfers (2010) 由来の補正テーブル
# (オッズ下限, オッズ上限, 補正係数)
FL_CORRECTION_TABLE = [
    (1.00,    1.50, 1.10),
    (1.50,    5.00, 1.05),
    (5.00,   20.00, 1.00),
    (20.00,  50.00, 0.85),
    (50.00, 100.00, 0.70),
    (100.00, 9999.0, 0.50),
]


def get_correction_factor(odds: float) -> float:
    """オッズ範囲から補正係数を取得"""
    if odds is None or odds <= 1.0 or np.isnan(odds):
        return 1.0
    for lo, hi, factor in FL_CORRECTION_TABLE:
        if lo <= odds < hi:
            return factor
    return 1.0


def correct_from_odds(odds: float, take_rate: float = 0.20) -> float:
    """
    オッズから補正後の真の勝率を取得。
    Returns:
        true_prob (0〜1)
    """
    if odds is None or odds <= 1.0 or np.isnan(odds):
        return 0.05
    raw_implied = (1.0 - take_rate) / odds
    cf = get_correction_factor(odds)
    return float(min(0.999, max(0.001, raw_implied * cf)))


def correct_implied_prob(implied_prob: float, odds: float | None = None) -> float:
    """
    インプライド勝率を補正。odds が分かるなら oddsベースで、
    分からなければ implied_prob → 推定オッズに変換してから補正。
    """
    if implied_prob is None or implied_prob <= 0 or np.isnan(implied_prob):
        return 0.05
    if odds is None or odds <= 1.0:
        odds = 0.8 / max(implied_prob, 1e-6)
    cf = get_correction_factor(odds)
    return float(min(0.999, max(0.001, implied_prob * cf)))


if __name__ == "__main__":
    print("=== Favorite-Longshot 補正サンプル ===")
    print(f"{'odds':>8} {'raw_implied':>12} {'corrected':>12} {'factor':>8}")
    for odds in [1.5, 2.5, 5.0, 10.0, 30.0, 80.0, 200.0]:
        raw = 0.8 / odds
        cor = correct_from_odds(odds)
        cf = get_correction_factor(odds)
        print(f"{odds:>8.1f} {raw:>12.4f} {cor:>12.4f} {cf:>8.2f}")
