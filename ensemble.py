"""
NEW-6: アンサンブル予測

単一の LightGBM ではなく、複数モデル/シグナルを加重平均で統合：

    ensemble = w_lgbm * f_lgbm + w_market * f_market + w_uniform * f_uniform
        f_lgbm    : LightGBM 予測（Benter blended 推奨）
        f_market  : 市場ベースライン（人気→経験勝率）
        f_uniform : 一様分布（1/N、レース全体への保険）

設計思想（業界標準）:
    - 単一モデルは過学習・分布シフトで悪化することがある
    - 異なる「視点」を加重平均すると安定化する
    - 重みは バックテスト or 経験値で決定（デフォルト 0.7 / 0.2 / 0.1）

使い方:
    from ensemble import ensemble_blend
    ens = ensemble_blend(model_probs, market_probs, weights=(0.7, 0.2, 0.1))
"""
import numpy as np
from pathlib import Path  # CLEAN: pandas 未使用のため削除

_DATA_DIR = Path(__file__).parent / "data"


def ensemble_blend(
    model_probs,
    market_probs,
    weights: tuple[float, float, float] = (0.7, 0.2, 0.1),
    eps: float = 1e-6,
) -> np.ndarray:
    """
    3経路アンサンブル: (LGBM, 市場, 一様) の加重和。

    Args:
        model_probs:  自モデル予測勝率 配列（合計≈1）
        market_probs: 市場インプライド勝率 配列（合計≈1）
        weights: (w_lgbm, w_market, w_uniform)
                 合計 ≈ 1.0 でなくても内部で正規化
        eps:    log(0) 回避

    Returns:
        統合後勝率配列（合計 = 1）
    """
    f = np.asarray(model_probs, dtype=float)
    pi = np.asarray(market_probs, dtype=float)
    n = len(f)
    if n == 0:
        return f

    # 正規化
    f = np.where(f < eps, eps, f)
    pi = np.where(pi < eps, eps, pi)
    f = f / f.sum()
    pi = pi / pi.sum()
    uniform = np.full(n, 1.0 / n)

    w1, w2, w3 = weights
    s = max(w1 + w2 + w3, eps)
    w1, w2, w3 = w1/s, w2/s, w3/s

    ens = w1 * f + w2 * pi + w3 * uniform
    return ens / ens.sum()


def get_default_weights() -> tuple[float, float, float]:
    """保存済みアンサンブル重みをロード（なければ既定値）"""
    import json
    p = _DATA_DIR / "ensemble_weights.json"
    if p.exists():
        try:
            d = json.load(open(p, encoding="utf-8"))
            return (float(d.get("w_lgbm", 0.7)),
                    float(d.get("w_market", 0.2)),
                    float(d.get("w_uniform", 0.1)))
        except Exception:
            pass
    return (0.7, 0.2, 0.1)


def save_weights(w_lgbm: float, w_market: float, w_uniform: float) -> None:
    import json
    p = _DATA_DIR / "ensemble_weights.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"w_lgbm": w_lgbm, "w_market": w_market, "w_uniform": w_uniform},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("=== アンサンブル動作確認 ===")
    # サンプル: 8頭、自モデルが本命を強推し、市場は中位を推す
    model = np.array([0.40, 0.20, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04])
    market = np.array([0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06, 0.04])

    print(f"  自モデル: {model.round(3)}")
    print(f"  市場    : {market.round(3)}")
    for w in [(1.0, 0.0, 0.0), (0.7, 0.2, 0.1), (0.5, 0.4, 0.1), (0.0, 1.0, 0.0)]:
        ens = ensemble_blend(model, market, w)
        print(f"  weights={w} → {ens.round(3)}")
