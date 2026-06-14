"""
第13波・B: 馬券内特化サブモデル学習スクリプト

target:
    in_top3 = 1 if rank <= 3 else 0

特徴: 既存 horse_latest_features の数値特徴量をそのまま流用。
出力:
    data/place_lgb.txt              - LightGBM Booster (二項分類)
    data/place_features.json        - 使用特徴量リスト
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
OUT_MODEL = DATA / "place_lgb.txt"
OUT_FEATS = DATA / "place_features.json"


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    print("[1/3] データ読み込み...")
    df = pd.read_parquet(TFJV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["horse_weight"] = pd.to_numeric(df["horse_weight"], errors="coerce")
    df["last_3f"] = pd.to_numeric(df["last_3f"], errors="coerce")
    df["finish_time"] = pd.to_numeric(df["finish_time"], errors="coerce")

    df = df.dropna(subset=["rank", "popularity", "distance", "horse_no"])

    # target
    df["in_top3"] = (df["rank"] <= 3).astype(int)

    # ==========================================================
    # 第32波: フォーム系ローリング特徴量（人気薄内の識別力強化）
    # 帯内AUC診断で 6人気以下 0.60 と判明 — 基本属性9個だけでは
    # 「絶対に爆走する穴」と紙くずを区別できない。
    # train_lgbm.py と同じ shift(1) リーク防止方式で直近成績を注入。
    # ==========================================================
    df = df.sort_values(["horse_name", "date"]).reset_index(drop=True)
    grp = df.groupby("horse_name", sort=False)

    # スピードフィギュア（条件別基準タイムとの差）
    _base = df.groupby(["venue", "surface", "distance", "track_condition"],
                       observed=True)["finish_time"].transform("median")
    df["speed_figure"] = _base - df["finish_time"]

    df["rank_avg3"] = grp["rank"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["rank_best5"] = grp["rank"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).min())
    df["speed_fig_avg3"] = grp["speed_figure"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["last3f_avg3"] = grp["last_3f"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df["places_last5"] = grp["rank"].transform(lambda x: (x.shift(1) <= 3).rolling(5, min_periods=1).sum())
    df["days_since_prev"] = grp["date"].transform(lambda x: x.diff().dt.days)
    _td = pd.to_numeric(df["time_diff"], errors="coerce")
    df["prev_margin"] = _td.groupby(df["horse_name"]).shift(1).clip(-3, 5)
    df["is_west"] = df["stable"].astype(str).str.contains("栗", na=False).astype(int)

    # 第32波R2: 「不利で負けただけの穴」検出シグナル
    # 前走の上がり3F順位（上がり最速で負け = 展開が向かなかっただけ）
    df["last3f_rank_raw"] = df.groupby("race_id")["last_3f"].rank(method="min", ascending=True)
    df["prev_last3f_rank"] = grp["last3f_rank_raw"].shift(1)
    # 前走の追い込み度（4角位置 → 着順の改善幅）
    _c4 = pd.to_numeric(df["corner4"].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    _fs = df.groupby("race_id")["horse_no"].transform("max").clip(lower=1)
    df["closing_raw"] = (_c4 - df["rank"]) / _fs
    df["prev_closing"] = grp["closing_raw"].shift(1)
    # クラス変動（race_name パース、train_lgbm と同方式）
    def _cl(rn):
        s = str(rn)
        if "G1" in s or "Ｇ１" in s: return 8
        if "G2" in s or "Ｇ２" in s: return 7
        if "G3" in s or "Ｇ３" in s: return 6
        if "オープン" in s or "3勝" in s or "１６００万" in s: return 5
        if "2勝" in s or "１０００万" in s: return 4
        if "1勝" in s or "５００万" in s: return 3
        if "未勝利" in s: return 2
        if "新馬" in s: return 1
        return 3
    df["class_lv"] = df["race_name"].apply(_cl)
    df["prev_class_lv"] = grp["class_lv"].shift(1)
    df["class_drop"] = (df["prev_class_lv"] - df["class_lv"]).fillna(0)  # 正 = 格下げ戦
    # 距離変更
    df["prev_distance"] = grp["distance"].shift(1)
    df["dist_change"] = (df["distance"] - df["prev_distance"]).fillna(0)
    # 枠の正規化位置（大外・最内）
    df["gate_pos_norm"] = df["horse_no"] / _fs

    # surface / track_condition のコード化
    df["surface_code"] = df["surface"].map({"芝": 0, "ダート": 1, "障害": 2}).fillna(0).astype(int)
    df["track_cond_code"] = df["track_condition"].map(
        {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3}
    ).fillna(0).astype(int)
    df["month"] = df["date"].dt.month
    df["venue_code"] = df["venue"].map(
        {"札幌": 0, "函館": 1, "福島": 2, "新潟": 3, "東京": 4,
         "中山": 5, "中京": 6, "京都": 7, "阪神": 8, "小倉": 9}
    ).fillna(0).astype(int)

    # (第14波修正) last_3f はそのレースの上がり3F = レース結果のリーク → 除外。
    # レース前に確定している情報のみを使う（horse_weight は当日朝発表で事前確定扱い）
    feats = [
        "horse_no", "popularity", "distance", "surface_code", "venue_code",
        "track_cond_code", "month", "age", "horse_weight",
        # 第32波: フォーム系（人気薄内の識別力の主役）
        "rank_avg3", "rank_best5", "speed_fig_avg3", "last3f_avg3",
        "places_last5", "days_since_prev", "prev_margin", "is_west",
        # 第32波R2: 穴の爆走シグナル
        "prev_last3f_rank", "prev_closing", "class_drop", "dist_change", "gate_pos_norm",
    ]
    # フォーム系は初出走で NaN になるため、基本属性のみ必須・フォームは許容
    df = df.dropna(subset=["horse_no", "popularity", "distance", "surface_code",
                            "venue_code", "track_cond_code", "month", "age", "horse_weight"])
    print(f"  サンプル数: {len(df):,} / target=in_top3: {df['in_top3'].mean()*100:.1f}%")
    return df, feats


def train(df: pd.DataFrame, feats: list[str]):
    print("[2/3] 時系列分割 & 学習...")
    train_mask = df["date"] < "2023-07-01"
    valid_mask = (df["date"] >= "2023-07-01") & (df["date"] < "2024-01-01")
    test_mask  = df["date"] >= "2024-01-01"

    Xtr, ytr = df.loc[train_mask, feats], df.loc[train_mask, "in_top3"]
    Xvl, yvl = df.loc[valid_mask, feats], df.loc[valid_mask, "in_top3"]
    Xts, yts = df.loc[test_mask,  feats], df.loc[test_mask,  "in_top3"]
    print(f"  train={len(Xtr):,} / valid={len(Xvl):,} / test={len(Xts):,}")

    # (R3 の人気薄重み付け×3 + num_leaves 127 は帯内AUCを悪化させたためロールバック)
    dtr = lgb.Dataset(Xtr, ytr)
    dvl = lgb.Dataset(Xvl, yvl, reference=dtr)
    params = {
        "objective":     "binary",
        "metric":        "auc",
        "learning_rate": 0.05,
        "num_leaves":    63,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq":     5,
        "verbose": -1,
    }
    model = lgb.train(
        params, dtr,
        num_boost_round=800,
        valid_sets=[dvl],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(100)],
    )

    print("[3/3] 評価...")
    for name, X, y in [("train", Xtr, ytr), ("valid", Xvl, yvl), ("test", Xts, yts)]:
        p = model.predict(X)
        auc = roc_auc_score(y, p)
        ll  = log_loss(y, p)
        print(f"  {name}: AUC={auc:.4f}  LogLoss={ll:.4f}  baseline={y.mean()*100:.1f}%")

    # 第32波: 帯内 AUC（穴選び実力の直接測定 — このモデルの存在意義）
    print("  --- 人気帯別の帯内 AUC (test) ---")
    test_sub = df.loc[test_mask].copy()
    test_sub["p"] = model.predict(test_sub[feats])
    for lo, hi, label in [(6, 9, "6-9人気"), (10, 13, "10-13人気"), (14, 18, "14-18人気"), (6, 18, "6人気以下")]:
        s = test_sub[(test_sub["popularity"] >= lo) & (test_sub["popularity"] <= hi)]
        if s["in_top3"].nunique() == 2:
            a = roc_auc_score(s["in_top3"], s["p"])
            print(f"    {label}: {a:.4f} (n={len(s):,})")

        # 上位30%抽出時の的中率
        thr = np.quantile(p, 0.7)
        upper_hit = y[p >= thr].mean() * 100
        print(f"    予測上位30%の馬券内率 = {upper_hit:.1f}%")

    return model


def main():
    df, feats = build_dataset()
    model = train(df, feats)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUT_MODEL))
    with open(OUT_FEATS, "w", encoding="utf-8") as f:
        json.dump({"features": feats}, f, ensure_ascii=False, indent=2)
    # 第18波: テスト AUC をメトリクス保存（UI 表示用）
    from sklearn.metrics import roc_auc_score as _ras
    test_mask = df["date"] >= "2024-01-01"
    _p = model.predict(df.loc[test_mask, feats])
    _auc_t = float(_ras(df.loc[test_mask, "in_top3"], _p))
    with open(DATA / "place_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_auc": round(_auc_t, 4),
                   "trained_at": pd.Timestamp.now().isoformat()[:19]}, f, ensure_ascii=False, indent=2)
    print(f"\n保存完了:")
    print(f"  {OUT_MODEL}")
    print(f"  {OUT_FEATS}")


if __name__ == "__main__":
    main()
