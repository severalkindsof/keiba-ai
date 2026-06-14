"""
IMPROVE-2: Optuna ハイパーパラメータ最適化（オプション・夜間バッチ向け）

実行方法:
    python tune_lgbm.py            # 50試行
    python tune_lgbm.py --trials 200

前提:
    `python train_lgbm.py` を1回実行して
    data/training_cache/train_preproc.parquet, valid_preproc.parquet が存在すること

最適化対象:
    learning_rate, max_depth, num_leaves, min_child_samples,
    reg_alpha, reg_lambda, subsample, colsample_bytree

出力:
    data/lgbm_best_params.json
    train_lgbm.py 次回実行時に自動で適用される
"""
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = DATA_DIR / "training_cache"

CAT_COLS = ["surface", "track_condition", "venue", "distance_cat", "sex"]


def main(n_trials: int = 50):
    try:
        import optuna
    except ImportError:
        print("optuna 未インストール。`pip install optuna` を実行してください。")
        return

    train_path = CACHE_DIR / "train_preproc.parquet"
    valid_path = CACHE_DIR / "valid_preproc.parquet"
    if not (train_path.exists() and valid_path.exists()):
        print(f"前処理キャッシュ未生成。先に `python train_lgbm.py` を実行してください。")
        print(f"  期待パス: {train_path}, {valid_path}")
        return

    cols_path = DATA_DIR / "lgbm_feature_cols.json"
    with open(cols_path, encoding="utf-8") as f:
        FEATURE_COLS = json.load(f)

    print(f"前処理キャッシュ読み込み中...")
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    print(f"  train: {len(train_df):,}行 / valid: {len(valid_df):,}行")

    # カテゴリ復元
    for col in CAT_COLS:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype("category")
            valid_df[col] = valid_df[col].astype("category")

    # race_key ソート（LambdaRank 必須）
    train_df = train_df.sort_values("race_key").reset_index(drop=True)
    valid_df = valid_df.sort_values("race_key").reset_index(drop=True)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["rank_label"]
    X_valid = valid_df[FEATURE_COLS]
    y_valid = valid_df["rank_label"]
    train_groups = train_df.groupby("race_key", sort=False)["race_key"].count().values
    valid_groups = valid_df.groupby("race_key", sort=False)["race_key"].count().values

    print(f"\nOptuna チューニング開始（{n_trials} 試行）...")

    def objective(trial):
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3],
            "n_estimators": 600,
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth":         trial.suggest_int("max_depth", 4, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha":         trial.suggest_float("reg_alpha",  1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state":      42,
            "n_jobs": -1,
            "verbose": -1,
        }
        m = lgb.LGBMRanker(**params)
        m.fit(
            X_train, y_train, group=train_groups,
            eval_set=[(X_valid, y_valid)], eval_group=[valid_groups],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        return m.best_score_["valid_0"]["ndcg@3"]

    study = optuna.create_study(direction="maximize", study_name="lgbm_ranker_keiba")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n--- 最適パラメータ ---")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  best NDCG@3: {study.best_value:.5f}")

    out_path = DATA_DIR / "lgbm_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "params": study.best_params,
            "ndcg3":  study.best_value,
            "n_trials": n_trials,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  保存: {out_path}")
    print(f"  → 次回 train_lgbm.py 実行時に自動で適用されます。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(n_trials=args.trials)
