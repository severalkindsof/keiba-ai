"""
SUPER-2: Isotonic Regression キャリブレーターの再学習スクリプト
（および SUPER-1: Benter blending 重み α, β の最尤推定）

実行方法:
    python fit_calibration.py

何をするか:
    1. tfjv_all.parquet の最新N%（時系列で末尾）を「未来データ」として確保
    2. LightGBM の生スコアを「未来データ」に対して予測
    3. Isotonic Regression で「スコア → 実勝率」マッピングを学習
    4. Benter blending の α, β をロジスティック回帰で最尤推定
    5. data/lgbm_calibrator.pkl と data/benter_weights.json を保存

注意:
    - 既存モデルが「全期間」で学習されている場合、ホールドアウトに完全な過学習がある
    - その場合は train_lgbm.py 側で time split に切り替え後、再実行を推奨
    - とりあえずは「ベースライン値」として現状でフィット可
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DATA_DIR = Path(__file__).parent / "data"


def _load_holdout(holdout_months: int = 6) -> pd.DataFrame:
    """
    tfjv_all.parquet の末尾 holdout_months ヶ月を「未来データ」として返す。
    """
    df = pd.read_parquet(DATA_DIR / "tfjv_all.parquet")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "rank", "popularity"]).copy()
    df = df[(df["popularity"] >= 1) & (df["popularity"] <= 18)]
    df["win_flag"] = (df["rank"] == 1).astype(int)

    cutoff = df["date"].max() - pd.DateOffset(months=holdout_months)
    holdout = df[df["date"] >= cutoff].copy()
    print(f"  ホールドアウト: {len(holdout):,} 行 ({cutoff.date()} 〜 {df['date'].max().date()})")
    return holdout


def _predict_lgbm_on_holdout(holdout: pd.DataFrame) -> np.ndarray:
    """
    既存 LightGBM モデルを使ってホールドアウトの勝率を予測。
    特徴量は ev_calculator.predict_win_rate_lgbm と同じ揃え方をする。
    """
    from ev_calculator import predict_win_rate_lgbm, _load_lgbm_once
    _load_lgbm_once()
    preds = []
    for i, row in holdout.iterrows():
        horse = {
            "horse_name":      row.get("horse_name", ""),
            "surface":         row.get("surface", "芝"),
            "distance":        int(row.get("distance", 2000) or 2000),
            "venue":           row.get("venue", ""),
            "popularity":      int(row.get("popularity", 9) or 9),
            "horse_no":        row.get("horse_no", 8),
            "field_size":      int(row.get("field_size", 16) or 16),
            "weight_carried":  row.get("weight_carried", 55.0),
            "track_condition": row.get("track_condition", "良"),
            "sex":             row.get("sex", "牡"),
            "age":             row.get("age", 4),
            "horse_weight":    row.get("horse_weight", 480),
            "race_name":       row.get("race_name", ""),
        }
        # 第27波: 必ず生スコアで取得（校正済み出力への再フィット=循環校正を防止）
        p = predict_win_rate_lgbm(horse, raw_score=True)
        preds.append(p if p is not None else np.nan)
    return np.asarray(preds, dtype=float)


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """スコア→確率の単調マッピングをフィット"""
    mask = ~np.isnan(scores)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    ir.fit(scores[mask], labels[mask])
    return ir


def fit_beta_calibration(scores: np.ndarray, labels: np.ndarray):
    """
    IMPROVE-4: Beta Calibration（Kull et al. 2017）
    Platt scaling の一般化。階段関数の Isotonic より滑らかで頑健。

    f(s) = 1 / (1 + exp(-(a*log(s) - b*log(1-s) + c)))
    内部的にロジスティック回帰で 3パラメータ (a, b, c) を最尤推定。
    """
    from sklearn.linear_model import LogisticRegression
    mask = ~np.isnan(scores)
    s = np.clip(scores[mask], 1e-6, 1 - 1e-6)
    y = labels[mask]
    X = np.column_stack([np.log(s), -np.log(1 - s)])
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(X, y)

    class _BetaCalibrator:
        def __init__(self, lr):
            self.lr = lr
        def predict(self, scores):
            s = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
            X = np.column_stack([np.log(s), -np.log(1 - s)])
            return self.lr.predict_proba(X)[:, 1]

    return _BetaCalibrator(lr)


def fit_benter_weights(
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Benter (1994) の log-linear blend モデルを最尤推定。

        logit(P(win)) = α·log(f) + β·log(π) + const

    ロジスティック回帰で α, β を求める（softmax + race grouping は近似でスキップ、
    レース単位独立性は弱い仮定）。
    """
    eps = 1e-6
    f = np.clip(model_probs, eps, 1.0)
    pi = np.clip(market_probs, eps, 1.0)
    X = np.column_stack([np.log(f), np.log(pi)])
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], labels[mask]

    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, y)
    alpha, beta = float(lr.coef_[0, 0]), float(lr.coef_[0, 1])
    return alpha, beta


def main(holdout_months: int = 6):
    print("=" * 60)
    print("Calibration Fitting (Isotonic + Benter)")
    print("=" * 60)

    print("\n[1/4] ホールドアウト確保...")
    holdout = _load_holdout(holdout_months=holdout_months)

    print("\n[2/4] LightGBM 予測実行...")
    scores = _predict_lgbm_on_holdout(holdout)
    valid = ~np.isnan(scores)
    print(f"  有効予測: {valid.sum():,} / {len(scores):,}")

    # 市場確率（人気→過去実勝率）
    from market_prob import get_market_prob_series
    market = get_market_prob_series(holdout["popularity"].values).values
    labels = holdout["win_flag"].values.astype(int)

    print("\n[3/4] Isotonic Regression フィット...")
    ir = fit_isotonic(scores, labels)
    cal_path = DATA_DIR / "lgbm_calibrator.pkl"
    joblib.dump(ir, str(cal_path))
    print(f"  保存: {cal_path}")
    # サマリ
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    if valid.sum() > 100:
        cal_scores = ir.predict(scores[valid])
        df_eval = pd.DataFrame({"pred": cal_scores, "win": labels[valid]})
        df_eval["bin"] = pd.cut(df_eval["pred"], bins=bins, include_lowest=True)
        print("  キャリブレーション後の信頼性図:")
        for b, grp in df_eval.groupby("bin", observed=True):
            print(f"    {str(b):20s}  n={len(grp):6d}  pred_avg={grp['pred'].mean():.3f}  actual={grp['win'].mean():.3f}")

    print("\n[4/4] Benter blending α, β フィット...")
    # キャリブレーション済みスコアを f に使う
    cal_scores_full = np.where(valid, ir.predict(np.where(valid, scores, 0.001)), np.nan)
    alpha, beta = fit_benter_weights(cal_scores_full, market, labels)
    print(f"  α (自モデル重み) = {alpha:.4f}")
    print(f"  β (市場重み)     = {beta:.4f}")
    # 保存
    bw_path = DATA_DIR / "benter_weights.json"
    with open(bw_path, "w", encoding="utf-8") as f:
        json.dump({
            "alpha": alpha,
            "beta": beta,
            "holdout_months": holdout_months,
            "n_train": int(valid.sum()),
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {bw_path}")

    print("\n完了。アプリ再起動後、新しい値が反映されます。")


if __name__ == "__main__":
    import sys
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(holdout_months=months)
