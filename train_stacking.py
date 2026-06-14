"""
A-2: 3-way Stacking — LightGBM + XGBoost + CatBoost

train_lgbm.py が生成した前処理キャッシュ (data/training_cache/) を使い、
XGBoost と CatBoost の Ranker を並走学習。
3モデルの valid 予測を入力にロジスティック回帰でメタモデルをフィット。

実行:
    python train_lgbm.py        # 前提：前処理キャッシュ生成
    python train_stacking.py    # XGB + CatBoost 学習 + stacking
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "training_cache"
CAT_COLS = ["surface", "track_condition", "venue", "distance_cat", "sex"]


def _load_cached():
    train = pd.read_parquet(CACHE_DIR / "train_preproc.parquet")
    valid = pd.read_parquet(CACHE_DIR / "valid_preproc.parquet")
    with open(DATA_DIR / "lgbm_feature_cols.json", encoding="utf-8") as f:
        FEATURE_COLS = json.load(f)
    for c in CAT_COLS:
        if c in train.columns:
            train[c] = train[c].astype("category")
            valid[c] = valid[c].astype("category")
    train = train.sort_values("race_key").reset_index(drop=True)
    valid = valid.sort_values("race_key").reset_index(drop=True)
    return train, valid, FEATURE_COLS


def train_xgboost(train_df, valid_df, FEATURE_COLS):
    """XGBoost Ranker"""
    import xgboost as xgb
    print("\n[XGBoost] 学習開始...")
    # XGB は category dtype を扱えないので one-hot or label-encode
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    cat_maps = {}
    for c in CAT_COLS:
        if c in train_df.columns:
            cats = sorted(set(train_df[c].astype(str).unique()) | set(valid_df[c].astype(str).unique()))
            cat_maps[c] = {v: i for i, v in enumerate(cats)}
            train_df[c] = train_df[c].astype(str).map(cat_maps[c]).fillna(-1).astype(int)
            valid_df[c] = valid_df[c].astype(str).map(cat_maps[c]).fillna(-1).astype(int)

    X_train = train_df[FEATURE_COLS].astype(float).fillna(0)
    y_train = train_df["rank_label"].values
    X_valid = valid_df[FEATURE_COLS].astype(float).fillna(0)
    y_valid = valid_df["rank_label"].values

    train_groups = train_df.groupby("race_key", sort=False).size().values
    valid_groups = valid_df.groupby("race_key", sort=False).size().values

    m = xgb.XGBRanker(
        objective="rank:ndcg", eval_metric="ndcg@3",
        n_estimators=300, learning_rate=0.07, max_depth=6,
        subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=-1, tree_method="hist",
        early_stopping_rounds=30,
    )
    m.fit(X_train, y_train, group=train_groups,
          eval_set=[(X_valid, y_valid)], eval_group=[valid_groups],
          verbose=50)
    preds_valid = m.predict(X_valid)
    # Save
    m.save_model(str(DATA_DIR / "xgb_ranker.json"))
    with open(DATA_DIR / "xgb_cat_maps.json", "w", encoding="utf-8") as f:
        json.dump(cat_maps, f, ensure_ascii=False)
    print(f"  保存: {DATA_DIR / 'xgb_ranker.json'}")
    return preds_valid, cat_maps


def train_catboost(train_df, valid_df, FEATURE_COLS):
    """CatBoost Ranker"""
    from catboost import CatBoostRanker, Pool
    print("\n[CatBoost] 学習開始...")
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    cat_features = [c for c in CAT_COLS if c in FEATURE_COLS]
    # CatBoost は str カテゴリでOK
    for c in cat_features:
        train_df[c] = train_df[c].astype(str)
        valid_df[c] = valid_df[c].astype(str)

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["rank_label"].values
    X_valid = valid_df[FEATURE_COLS].copy()
    y_valid = valid_df["rank_label"].values
    # CatBoost は ArrowStringArray を受け付けないため tolist で純 Python list 化
    group_train = train_df["race_key"].astype(str).tolist()
    group_valid = valid_df["race_key"].astype(str).tolist()

    pool_train = Pool(X_train, y_train, group_id=group_train, cat_features=cat_features)
    pool_valid = Pool(X_valid, y_valid, group_id=group_valid, cat_features=cat_features)

    m = CatBoostRanker(
        loss_function="YetiRank",
        iterations=400, learning_rate=0.07, depth=6,
        random_seed=42, verbose=50,
        early_stopping_rounds=30,
    )
    m.fit(pool_train, eval_set=pool_valid)
    preds_valid = m.predict(X_valid)
    m.save_model(str(DATA_DIR / "catboost_ranker.cbm"))
    print(f"  保存: {DATA_DIR / 'catboost_ranker.cbm'}")
    return preds_valid


def predict_lgbm_on_valid(train_df, valid_df, FEATURE_COLS):
    """既存 LightGBM で valid を予測（メタモデル入力用）"""
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(DATA_DIR / "lgbm_win_model.txt"))
    preds = m.predict(valid_df[FEATURE_COLS])
    return preds


def _softmax_per_race(scores, race_keys):
    """レース内 softmax 正規化"""
    df = pd.DataFrame({"s": scores, "r": race_keys})
    return df.groupby("r", sort=False)["s"].transform(
        lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    ).values


def fit_meta_model(lgbm_p, xgb_p, cb_p, labels, race_keys):
    """3モデル予測を入力にロジスティック回帰で stacking 重みをフィット"""
    from sklearn.linear_model import LogisticRegression

    # 各モデルをレース内 softmax 正規化
    p_lgbm = _softmax_per_race(lgbm_p, race_keys)
    p_xgb  = _softmax_per_race(xgb_p,  race_keys)
    p_cb   = _softmax_per_race(cb_p,   race_keys)

    eps = 1e-6
    X = np.column_stack([
        np.log(np.clip(p_lgbm, eps, 1)),
        np.log(np.clip(p_xgb,  eps, 1)),
        np.log(np.clip(p_cb,   eps, 1)),
    ])
    y = labels.astype(int)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, y)
    w = lr.coef_[0]
    return {"w_lgbm": float(w[0]), "w_xgb": float(w[1]), "w_cb": float(w[2]),
            "intercept": float(lr.intercept_[0])}


def main():
    print("=" * 60)
    print("A-2: 3-way Stacking (LightGBM + XGBoost + CatBoost)")
    print("=" * 60)

    train_df, valid_df, FEATURE_COLS = _load_cached()
    print(f"  train: {len(train_df):,}, valid: {len(valid_df):,}, features: {len(FEATURE_COLS)}")

    print("\n[1/4] LightGBM valid 予測...")
    p_lgbm = predict_lgbm_on_valid(train_df, valid_df, FEATURE_COLS)

    print("\n[2/4] XGBoost 学習...")
    p_xgb, _cat_maps = train_xgboost(train_df, valid_df, FEATURE_COLS)

    print("\n[3/4] CatBoost 学習...")
    p_cb = train_catboost(train_df, valid_df, FEATURE_COLS)

    print("\n[4/4] メタモデル（stacking 重み）フィット...")
    win_labels = (valid_df["rank_label"] == 4).astype(int).values  # 1着
    race_keys  = valid_df["race_key"].values
    meta = fit_meta_model(p_lgbm, p_xgb, p_cb, win_labels, race_keys)
    print(f"  Stacking weights: {meta}")

    out_path = DATA_DIR / "stacking_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  保存: {out_path}")

    # 評価
    from sklearn.metrics import roc_auc_score, brier_score_loss
    for name, scores in [("LGBM", p_lgbm), ("XGB", p_xgb), ("CB", p_cb)]:
        try:
            auc = roc_auc_score(win_labels, scores)
            prob = _softmax_per_race(scores, race_keys)
            br = brier_score_loss(win_labels, prob)
            print(f"  {name}: AUC={auc:.4f}, Brier={br:.5f}")
        except Exception as _e:
            print(f"  {name}: 評価エラー {_e}")

    # stacking 評価
    eps = 1e-6
    p_lg = _softmax_per_race(p_lgbm, race_keys)
    p_xg = _softmax_per_race(p_xgb,  race_keys)
    p_cb_n = _softmax_per_race(p_cb, race_keys)
    log_combined = (
        meta["w_lgbm"] * np.log(np.clip(p_lg, eps, 1))
        + meta["w_xgb"] * np.log(np.clip(p_xg, eps, 1))
        + meta["w_cb"] * np.log(np.clip(p_cb_n, eps, 1))
    )
    stacked = _softmax_per_race(log_combined, race_keys)
    auc_s = roc_auc_score(win_labels, stacked)
    brier_s = brier_score_loss(win_labels, stacked)
    print(f"\n  Stacking: AUC={auc_s:.4f}, Brier={brier_s:.5f}")
    print("\n完了。app.py で ensemble モードに「stacking」オプションが使えるようになります。")


if __name__ == "__main__":
    main()
