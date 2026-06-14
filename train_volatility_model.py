"""
第13波: 荒れるレース判定モデル学習スクリプト

target:
    is_volatile = 1 if (1着馬の人気 >= 6) or (3着内に8人気以下が1頭以上)
                  else 0

features (race 単位):
    field_size       - 出走頭数
    distance         - 距離
    surface_code     - 芝=0 / ダート=1 / 障害=2
    venue_code       - 会場コード（0-9）
    race_no          - レース番号
    track_cond_code  - 良=0 / 稍=1 / 重=2 / 不良=3
    month            - 月
    field_avg_pop    - 出走馬の人気平均
    field_pop_std    - 人気の分散（横並びほど荒れる）

評価: 学習 / 検証は時系列分割（< 2023-07 / >= 2024-01）
出力:
    data/volatility_lgb.txt           - LightGBM Booster
    data/volatility_features.json     - 使用特徴量リスト
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TFJV = DATA / "tfjv_all.parquet"
OUT_MODEL = DATA / "volatility_lgb.txt"
OUT_FEATS = DATA / "volatility_features.json"


SURFACE_CODE = {"芝": 0, "ダート": 1, "障害": 2}
VENUE_CODE = {"札幌": 0, "函館": 1, "福島": 2, "新潟": 3,
              "東京": 4, "中山": 5, "中京": 6, "京都": 7, "阪神": 8, "小倉": 9}
TRACK_COND_CODE = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3}


def build_dataset() -> pd.DataFrame:
    print("[1/4] tfjv_all 読み込み...")
    df = pd.read_parquet(TFJV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce")

    # レース単位の race_key
    df["race_key"] = (
        df["date"].dt.strftime("%Y%m%d") + "_" + df["venue"].astype(str)
        + "_" + df["race_no"].astype(str)
    )

    print("[2/4] レース集計と target 作成...")
    g = df.groupby("race_key")
    # is_volatile = 1着人気>=6 OR 上位3着に人気>=8 が含まれる
    rec_winner_pop = g.apply(
        lambda x: int(x.loc[x["rank"] == 1, "popularity"].iloc[0])
        if (x["rank"] == 1).any() and pd.notna(x.loc[x["rank"] == 1, "popularity"].iloc[0])
        else np.nan
    )
    rec_top3_max_pop = g.apply(
        lambda x: int(x.loc[x["rank"].between(1, 3), "popularity"].max())
        if (x["rank"].between(1, 3)).any() and x.loc[x["rank"].between(1, 3), "popularity"].notna().any()
        else np.nan
    )

    # レース特徴量
    rec = pd.DataFrame({
        "race_key":        rec_winner_pop.index,
        "winner_pop":      rec_winner_pop.values,
        "top3_max_pop":    rec_top3_max_pop.values,
    })
    meta = g.agg({
        "date":       "first",
        "venue":      "first",
        "surface":    "first",
        "distance":   "first",
        "race_no":    "first",
        "track_condition": "first",
        "horse_no":   "max",       # 出走頭数の代用
        "popularity": ["mean", "std"],
    })
    meta.columns = ["date", "venue", "surface", "distance", "race_no",
                    "track_condition", "field_size", "field_avg_pop", "field_pop_std"]
    meta = meta.reset_index()
    rec = rec.merge(meta, on="race_key", how="left")

    rec = rec.dropna(subset=["winner_pop", "field_size", "distance"])
    rec["is_volatile"] = (
        (rec["winner_pop"] >= 6) | (rec["top3_max_pop"] >= 8)
    ).astype(int)

    # 特徴量化
    rec["surface_code"]   = rec["surface"].map(SURFACE_CODE).fillna(0).astype(int)
    rec["venue_code"]     = rec["venue"].map(VENUE_CODE).fillna(0).astype(int)
    rec["track_cond_code"] = rec["track_condition"].map(TRACK_COND_CODE).fillna(0).astype(int)
    rec["month"]          = rec["date"].dt.month
    rec["race_no"]        = pd.to_numeric(rec["race_no"], errors="coerce").fillna(0).astype(int)
    rec["field_size"]     = rec["field_size"].fillna(16).astype(int)
    rec["distance"]       = rec["distance"].astype(int)
    rec["field_avg_pop"]  = rec["field_avg_pop"].fillna(8.5)
    rec["field_pop_std"]  = rec["field_pop_std"].fillna(4.0)

    print(f"  レース数: {len(rec):,}, 荒れた割合: {rec['is_volatile'].mean()*100:.1f}%")
    return rec


FEATURES = [
    "field_size", "distance", "surface_code", "venue_code",
    "race_no", "track_cond_code", "month",
    "field_avg_pop", "field_pop_std",
]


def train(rec: pd.DataFrame):
    print("[3/4] 時系列分割 & 学習...")
    train_mask = rec["date"] < "2023-07-01"
    valid_mask = (rec["date"] >= "2023-07-01") & (rec["date"] < "2024-01-01")
    test_mask  = rec["date"] >= "2024-01-01"

    Xtr, ytr = rec.loc[train_mask, FEATURES], rec.loc[train_mask, "is_volatile"]
    Xvl, yvl = rec.loc[valid_mask, FEATURES], rec.loc[valid_mask, "is_volatile"]
    Xts, yts = rec.loc[test_mask,  FEATURES], rec.loc[test_mask,  "is_volatile"]
    print(f"  train={len(Xtr):,} / valid={len(Xvl):,} / test={len(Xts):,}")

    dtr = lgb.Dataset(Xtr, ytr)
    dvl = lgb.Dataset(Xvl, yvl, reference=dtr)
    params = {
        "objective": "binary",
        "metric":    "auc",
        "learning_rate": 0.05,
        "num_leaves":    31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq":     5,
        "verbose": -1,
    }
    model = lgb.train(
        params, dtr,
        num_boost_round=500,
        valid_sets=[dvl],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    print("[4/4] 評価...")
    for name, X, y in [("train", Xtr, ytr), ("valid", Xvl, yvl), ("test", Xts, yts)]:
        p = model.predict(X)
        auc = roc_auc_score(y, p)
        ll  = log_loss(y, p)
        print(f"  {name}: AUC={auc:.4f}  LogLoss={ll:.4f}  baseline={y.mean()*100:.1f}%")

    return model


def main():
    rec = build_dataset()
    model = train(rec)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_MODEL))
    with open(OUT_FEATS, "w", encoding="utf-8") as f:
        json.dump({"features": FEATURES}, f, ensure_ascii=False, indent=2)
    print(f"\n保存完了:")
    print(f"  {OUT_MODEL}")
    print(f"  {OUT_FEATS}")


if __name__ == "__main__":
    main()
