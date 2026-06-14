"""
A-3: Conformal Prediction によるレース選択フィルター + 予測信頼区間

論文: Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
      [arXiv 2107.07511]

機能:
    1. valid セット上で「非整合スコア」を計算
    2. 推論時に各馬の予測勝率に **信頼区間 (1-α coverage)** を付与
    3. レース全体の「予測確信度」(=平均区間幅) を測定 → 区間が広いレースは見送り推奨

使い方:
    # 訓練時（fit_calibration.py からも呼ばれる）:
        python conformal.py
        → data/conformal_q.json に quantile (90% カバレッジ用) を保存

    # 推論時:
    from conformal import predict_interval
    lo, hi = predict_interval(model_prob, alpha=0.10)  # 90% 信頼区間
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
_Q_PATH = _DATA_DIR / "conformal_q.json"


def fit_conformal(
    preds: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.10,
) -> dict:
    """
    Split conformal prediction:
        非整合スコア = |y - p_hat|（回帰的扱い・勝率予測の絶対誤差）
        quantile = ceil((n+1)(1-α))/n の経験 quantile

    Args:
        preds:  valid セットの予測勝率（0〜1）
        labels: 実際の win_flag（0/1）
        alpha:  miscoverage rate（0.10 = 90%カバレッジ）

    Returns:
        {"q_alpha": float, "alpha": alpha, "n_calib": int}
    """
    preds  = np.asarray(preds,  dtype=float)
    labels = np.asarray(labels, dtype=float)
    mask = ~np.isnan(preds) & ~np.isnan(labels)
    preds, labels = preds[mask], labels[mask]
    n = len(preds)
    if n < 100:
        return {"q_alpha": 0.05, "alpha": alpha, "n_calib": n, "warning": "n<100"}

    scores = np.abs(labels - preds)
    # adjusted quantile for finite-sample validity
    q_rank = int(np.ceil((n + 1) * (1 - alpha))) - 1
    q_rank = min(q_rank, n - 1)
    q_alpha = float(np.sort(scores)[q_rank])
    return {"q_alpha": q_alpha, "alpha": alpha, "n_calib": n}


def save_conformal(q_dict: dict, path: Path | None = None):
    path = path or _Q_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(q_dict, f, ensure_ascii=False, indent=2)


_Q_CACHE: dict | None = None


def _load_q() -> dict:
    global _Q_CACHE
    if _Q_CACHE is not None:
        return _Q_CACHE
    if not _Q_PATH.exists():
        _Q_CACHE = {"q_alpha": 0.05, "alpha": 0.10, "n_calib": 0}
    else:
        with open(_Q_PATH, encoding="utf-8") as f:
            _Q_CACHE = json.load(f)
    return _Q_CACHE


def predict_interval(prob: float, q_alpha: float | None = None) -> tuple[float, float]:
    """
    予測勝率に信頼区間を付与。
    Returns: (lo, hi) ∈ [0, 1]
    """
    if q_alpha is None:
        q_alpha = float(_load_q().get("q_alpha", 0.05))
    lo = max(0.0, prob - q_alpha)
    hi = min(1.0, prob + q_alpha)
    return lo, hi


def race_confidence(probs: list[float], q_alpha: float | None = None) -> dict:
    """
    レース単位の予測確信度を計算。
    - 平均区間幅（大きい = 不確実 = 見送り推奨）
    - トップ確率と区間下限の重なり度

    Returns:
        {
            "mean_interval_width": float,    # 平均信頼区間幅
            "top_lo": float,                 # トップ馬の予測下限
            "second_hi": float,              # 2位馬の予測上限
            "overlap": bool,                 # トップと2位の区間が重なるか
            "confidence_label": str,         # "高" / "中" / "低"
            "recommend_skip": bool,          # True なら見送り推奨
        }
    """
    if not probs:
        return {"mean_interval_width": 0, "top_lo": 0, "second_hi": 0,
                "overlap": False, "confidence_label": "データなし",
                "recommend_skip": True}
    if q_alpha is None:
        q_alpha = float(_load_q().get("q_alpha", 0.05))
    intervals = [predict_interval(p, q_alpha) for p in probs]
    widths = [hi - lo for lo, hi in intervals]
    mean_w = float(np.mean(widths))

    # 参考情報: トップ2の区間重なり（binary残差由来の q_alpha は大きく
    # 区間は常に重なるため、判定には使わない — 表示用に残す）
    sorted_idx = sorted(range(len(probs)), key=lambda i: -probs[i])
    top_lo  = intervals[sorted_idx[0]][0] if len(sorted_idx) >= 1 else 0
    second_hi = intervals[sorted_idx[1]][1] if len(sorted_idx) >= 2 else 0
    overlap = (top_lo <= second_hi)

    # 第24波修正: 旧判定は q_alpha=0.336（binary 残差の分位点）に対し
    # 閾値 0.04/0.07 が非現実的で、1強でも大混戦でも常に「やや低/skip=False」
    # → 「見送り推奨」が一度も発火しない死に機能だった。
    # 確率分布の形状ベースに変更（valid 1,731 レースで実測校正）:
    #   高:  top1>=0.35 & gap>=0.15 → 31%のレース / 本命的中 45%
    #   中:  top1>=0.25 & gap>=0.08 → 46% / 30%
    #   低:  top1<0.28 & gap<0.04   →  6% / 28%（見送り推奨）
    ps = sorted([float(p) for p in probs], reverse=True)
    top1 = ps[0]
    gap = top1 - ps[1] if len(ps) >= 2 else top1
    if top1 >= 0.35 and gap >= 0.15:
        label = "高（決め打ち可）"; skip = False
    elif top1 >= 0.25 and gap >= 0.08:
        label = "中（本命押し）"; skip = False
    elif top1 < 0.28 and gap < 0.04:
        label = "低（混戦・見送り推奨）"; skip = True
    else:
        label = "やや低（上位拮抗）"; skip = False

    return {
        "mean_interval_width": round(mean_w, 3),
        "top_lo":  round(top_lo, 3),
        "second_hi": round(second_hi, 3),
        "overlap": overlap,
        "confidence_label": label,
        "recommend_skip": skip,
    }


def fit_from_training_cache(alpha: float = 0.10) -> dict:
    """
    train_lgbm.py が生成した training_cache から conformal を学習。
    valid_preproc.parquet が必要。
    """
    valid_path = _DATA_DIR / "training_cache" / "valid_preproc.parquet"
    if not valid_path.exists():
        print(f"valid_preproc.parquet が無い: {valid_path}")
        return {}
    # LightGBM モデル + キャリブレータでvalid の校正済み確率を計算
    import ev_calculator as _ev
    _ev._load_lgbm_once()
    if _ev._LGBM_MODEL is None:
        print("LightGBM モデルが見つからない")
        return {}
    _LGBM_MODEL = _ev._LGBM_MODEL
    _CALIBRATOR = _ev._CALIBRATOR
    _FEATURE_COLS = _ev._FEATURE_COLS

    valid_df = pd.read_parquet(valid_path)
    cat_cols = ["surface", "track_condition", "venue", "distance_cat", "sex"]
    for c in cat_cols:
        if c in valid_df.columns:
            valid_df[c] = valid_df[c].astype("category")

    feat_cols = [c for c in _FEATURE_COLS if c in valid_df.columns]
    X = valid_df[feat_cols]
    raw = _LGBM_MODEL.predict(X)
    if _CALIBRATOR is not None:
        cal = _CALIBRATOR.predict(raw)
    else:
        cal = raw
    cal = np.clip(cal, 0.001, 0.999)

    y = valid_df["win_flag"].astype(int).values
    result = fit_conformal(cal, y, alpha=alpha)
    save_conformal(result)
    print(f"Conformal フィット完了：q_alpha={result['q_alpha']:.4f} (n={result['n_calib']:,})")
    print(f"保存: {_Q_PATH}")
    return result


if __name__ == "__main__":
    fit_from_training_cache(alpha=0.10)
