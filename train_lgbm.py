"""
LightGBM 勝率予測モデルの学習スクリプト。

実行方法: python train_lgbm.py
所要時間: 約5〜10分（初回）
出力:
  data/lgbm_win_model.txt   … 学習済みモデル
  data/lgbm_feature_cols.json … 特徴量リスト（ev_calculator.pyが使う）

運用サイクル:
  毎週  → convert_tfjv.py を再実行（最新データに更新）
  月1回 → このスクリプトを再実行（モデルを最新データで再学習）
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent / "data"

# ============================================================
# 1. データ読み込み
# ============================================================
print("=" * 50)
print("STEP 1: データ読み込み")
print("=" * 50)
df = pd.read_parquet(DATA_DIR / "tfjv_all.parquet")
print(f"  読み込み完了: {len(df):,}行")

# 文字列列の前後空白を除去
for col in ["horse_name", "jockey", "trainer", "sire", "venue"]:
    if col in df.columns:
        df[col] = df[col].str.strip()

# 数値変換（文字列型のまま残っている列）
for col in ["weight_carried", "horse_no", "age"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 日付でソート（ローリング特徴量の計算に必須）
print("  日付ソート中...")
df = df.sort_values(["horse_name", "date"]).reset_index(drop=True)
print("  完了")

# ============================================================
# 2. スピードフィギュア（速度指数）の計算
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: スピードフィギュア計算")
print("=" * 50)
# 基準タイム = 会場×馬場×距離×馬場状態ごとの中央値（同条件レースの標準ペース）
# speed_figure の単位は「0.1秒」: +10 = 1秒速い
baseline = df.groupby(["venue", "surface", "distance", "track_condition"],
                       observed=True)["finish_time"].transform("median")
df["speed_figure"] = baseline - df["finish_time"]
# NaN（タイムなし）はそのまま保持（ローリングでmin_periods=1が処理）
valid_sf = df["speed_figure"].notna().sum()
print(f"  速度指数が計算できた行: {valid_sf:,} / {len(df):,}")

# ============================================================
# 3. ローリング特徴量（直近n走の集計）
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: ローリング特徴量（直近成績）")
print("=" * 50)
# shift(1) = 「そのレース自身の結果を除外」してデータリークを防ぐ

grp = df.groupby("horse_name", sort=False)

print("  直近3走・5走の平均着順...")
df["rank_avg3"]      = grp["rank"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["rank_avg5"]      = grp["rank"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df["rank_best5"]     = grp["rank"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).min())

print("  直近3走のスピードフィギュア・上がり3F...")
df["speed_fig_avg3"] = grp["speed_figure"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["last3f_avg3"]    = grp["last_3f"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())

print("  直近5走の勝利数・複勝数...")
df["wins_last5"]     = grp["win_flag"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum())
df["places_last5"]   = grp["place_flag"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).sum())

print("  出走間隔（日数）...")
df["days_since_prev"] = grp["date"].transform(
    lambda x: pd.to_datetime(x).diff().dt.days)
# diff()は現在行 - 前行なので、shift不要（前走からの間隔がそのまま入る）

print("  馬体重トレンド（直近3走）...")
df["weight_trend3"]  = grp["horse_weight"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=2).apply(
        lambda w: w.iloc[-1] - w.iloc[0] if len(w) >= 2 else np.nan, raw=False))

print("  ローリング特徴量完了")

# ============================================================
# 4. 騎手・調教師の統計を計算して結合
# ============================================================
print("\n" + "=" * 50)
print("STEP 4: 騎手・調教師統計の計算")
print("=" * 50)

# 騎手全体勝率・複勝率（全人気帯）
jockey_all = (df.groupby("jockey", sort=False)
              .agg(jockey_win_rate   = ("win_flag",   "mean"),
                   jockey_place_rate = ("place_flag", "mean"),
                   jockey_rides      = ("win_flag",   "count"))
              .reset_index())

# 騎手の人気薄（6番人気以下）での勝率
jockey_long = (df[df["popularity"] >= 6]
               .groupby("jockey", sort=False)
               .agg(jockey_longshot_win_rate   = ("win_flag",   "mean"),
                    jockey_longshot_place_rate  = ("place_flag", "mean"))
               .reset_index())

df = df.merge(jockey_all,  on="jockey",  how="left")
df = df.merge(jockey_long, on="jockey",  how="left")

# 調教師全体勝率
trainer_all = (df.groupby("trainer", sort=False)
               .agg(trainer_win_rate   = ("win_flag",   "mean"),
                    trainer_place_rate = ("place_flag", "mean"))
               .reset_index())
df = df.merge(trainer_all, on="trainer", how="left")

print(f"  騎手数: {len(jockey_all):,}")
print(f"  調教師数: {len(trainer_all):,}")

# ============================================================
# 5. 父系統計を結合
# ============================================================
sire_stats_path = DATA_DIR / "sire_stats.parquet"
if sire_stats_path.exists() and "sire" in df.columns:
    print("\n父系統計を結合中...")
    sire_df = pd.read_parquet(sire_stats_path)
    sire_df = sire_df.rename(columns={"win_rate": "sire_win_rate"})
    df = df.merge(sire_df[["sire", "distance_cat", "sire_win_rate"]],
                  on=["sire", "distance_cat"], how="left")
    print(f"  sire_win_rate 取得件数: {df['sire_win_rate'].notna().sum():,}")

# ============================================================
# 6. 特徴量の定義
# ============================================================
print("\n" + "=" * 50)
print("STEP 5: 特徴量の選択・前処理")
print("=" * 50)

# カテゴリ変数
CAT_COLS = ["surface", "track_condition", "venue", "distance_cat", "sex"]
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype("category")

# 使用する特徴量リスト
FEATURE_COLS = [
    # ---- レース条件 ----
    "surface",          # 芝 / ダート
    "track_condition",  # 良 / 稍重 / 重 / 不良
    "venue",            # 競馬場
    "distance",         # 距離（m）
    "distance_cat",     # 距離帯カテゴリ
    "field_size",       # 出走頭数
    "weight_carried",   # 斤量（kg）
    "horse_no",         # 馬番（枠順の代理）
    # ---- 馬の基本情報 ----
    "age",              # 馬齢
    "sex",              # 性別
    "horse_weight",     # 馬体重（kg）
    # weight_change は列マッピング不正のため除外（TODO: TFJV列定義確認後に復活）
    # ---- 市場評価 ----
    # popularity は除外: popularityが支配的になり他の特徴量が無効化されるため
    # EV計算には使うが、モデルの学習には使わない（馬の実力を純粋に評価するため）
    # ---- 直近成績（ローリング特徴量） ----
    "rank_avg3",        # 直近3走の平均着順
    "rank_avg5",        # 直近5走の平均着順
    "rank_best5",       # 直近5走の最高着順
    "speed_fig_avg3",   # 直近3走の平均スピードフィギュア
    "last3f_avg3",      # 直近3走の平均上がり3F（0.1秒単位）
    "wins_last5",       # 直近5走の勝利数
    "places_last5",     # 直近5走の複勝数
    "days_since_prev",  # 前走からの出走間隔（日数）
    "weight_trend3",    # 直近3走の馬体重変化トレンド
    # ---- 騎手 ----
    "jockey_win_rate",              # 騎手全体勝率
    "jockey_place_rate",            # 騎手全体複勝率
    "jockey_rides",                 # 騎手の総騎乗数（実績の信頼度）
    "jockey_longshot_win_rate",     # 騎手の人気薄での勝率
    "jockey_longshot_place_rate",   # 騎手の人気薄での複勝率
    # ---- 調教師 ----
    "trainer_win_rate",             # 調教師勝率
    "trainer_place_rate",           # 調教師複勝率
]

# 父系統計（あれば追加）
if "sire_win_rate" in df.columns:
    FEATURE_COLS.append("sire_win_rate")

# 実際に存在する列のみ使用
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
print(f"  使用特徴量数: {len(FEATURE_COLS)}")
for col in FEATURE_COLS:
    n_null = df[col].isna().sum()
    if n_null > 0:
        print(f"    {col}: 欠損 {n_null:,}件")

# ============================================================
# 7. 学習・テストデータ分割（時系列で分割）
# ============================================================
print("\n" + "=" * 50)
print("STEP 6: 学習/テストデータ分割")
print("=" * 50)
# 2010〜2023年 → 学習用、2024〜2025年 → テスト用（未来データで評価）
train_mask = df["date"] < "2024-01-01"
test_mask  = df["date"] >= "2024-01-01"
train_df   = df[train_mask].copy()
test_df    = df[test_mask].copy()

print(f"  学習データ: {len(train_df):,}行  ({train_df['date'].min()} 〜 {train_df['date'].max()})")
print(f"  テストデータ: {len(test_df):,}行  ({test_df['date'].min()} 〜 {test_df['date'].max()})")

X_train = train_df[FEATURE_COLS]
y_train = train_df["win_flag"].astype(int)
X_test  = test_df[FEATURE_COLS]
y_test  = test_df["win_flag"].astype(int)

# クラス不均衡の確認（勝つ馬は全体の約1/field_size）
pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"  正例:負例 比率 = 1 : {pos_ratio:.1f}  （負例が多いため scale_pos_weight で補正）")

# ============================================================
# 8. LightGBM学習
# ============================================================
print("\n" + "=" * 50)
print("STEP 7: LightGBM 学習")
print("=" * 50)

model = lgb.LGBMClassifier(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 31,
    min_child_samples = 50,
    scale_pos_weight  = pos_ratio,   # クラス不均衡補正
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

model.fit(
    X_train, y_train,
    eval_set          = [(X_test, y_test)],
    callbacks         = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ],
)

best_n = model.best_iteration_
print(f"\n  最適なtree数: {best_n}")

# ============================================================
# 9. 評価
# ============================================================
print("\n" + "=" * 50)
print("STEP 8: 評価")
print("=" * 50)

y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"  AUC-ROC: {auc:.4f}  （目安: 0.65以上なら良好、0.70以上なら優秀）")

# 人気別の予測精度確認
test_df = test_df.copy()
test_df["pred_win_rate"] = y_pred
for pop_label, grp_df in test_df.groupby("pop_bucket", observed=True):
    if len(grp_df) == 0:
        continue
    actual_wr  = grp_df["win_flag"].mean()
    predict_wr = grp_df["pred_win_rate"].mean()
    print(f"  [{pop_label}]  実際の勝率: {actual_wr:.3f}  予測平均: {predict_wr:.3f}  件数: {len(grp_df):,}")

# バックテスト（人気から推定オッズを使ったEV>0の馬のみ購入）
print("\n--- バックテスト（EV > 0 の馬だけ購入した場合） ---")
# 人気→典型オッズの変換（実際のオッズはnetkeibaから取得するが、
# 過去データにはないため人気帯の典型値を使った近似）
TYPICAL_ODDS = {1: 2.5, 2: 4.0, 3: 6.5, 4: 10.0, 5: 14.0,
                6: 20.0, 7: 27.0, 8: 35.0, 9: 45.0}
def pop_to_odds(p):
    try:
        p = int(p)
    except (ValueError, TypeError):
        return 10.0
    return TYPICAL_ODDS.get(p, 50.0 + max(0, p - 10) * 8.0)

test_df["est_odds"] = test_df["popularity"].apply(pop_to_odds)
test_df["pred_ev"]  = test_df["pred_win_rate"] * (test_df["est_odds"] - 1) - (1 - test_df["pred_win_rate"])

buy = test_df[test_df["pred_ev"] > 0].copy()
if len(buy) > 0:
    returns = (buy["win_flag"] * buy["est_odds"]).sum()
    roi     = returns / len(buy) * 100
    print(f"  購入数:  {len(buy):,}件")
    print(f"  的中数:  {int(buy['win_flag'].sum()):,}件")
    print(f"  回収率:  {roi:.1f}%  （※人気→推定オッズを使った近似値）")
else:
    print("  EV>0 の馬が見つかりませんでした")

# 特徴量重要度トップ15
print("\n--- 特徴量重要度 トップ15 ---")
fi = (pd.Series(model.feature_importances_, index=FEATURE_COLS)
      .sort_values(ascending=False))
for i, (fname, score) in enumerate(fi.head(15).items(), 1):
    print(f"  {i:2d}. {fname:35s} {score:6.0f}")

# ============================================================
# 10. モデルと特徴量リストを保存
# ============================================================
print("\n" + "=" * 50)
print("STEP 9: 保存")
print("=" * 50)

model_path  = DATA_DIR / "lgbm_win_model.txt"
cols_path   = DATA_DIR / "lgbm_feature_cols.json"

model.booster_.save_model(str(model_path))
with open(cols_path, "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)

print(f"  モデル保存:   {model_path}")
print(f"  特徴量リスト: {cols_path}")

# カテゴリ変数のマッピングを保存（予測時に訓練時と同じ変換をするため）
cat_maps = {}
for col in CAT_COLS:
    if col in df.columns and hasattr(df[col], "cat"):
        cat_maps[col] = df[col].cat.categories.tolist()
cat_maps_path = DATA_DIR / "lgbm_cat_mappings.json"
with open(cat_maps_path, "w", encoding="utf-8") as f:
    json.dump(cat_maps, f, ensure_ascii=False, indent=2)
print(f"  カテゴリマップ: {cat_maps_path}")

# 各馬の「最新レース時点での特徴量」を保存（予測時に使う直近成績）
print("\n馬ごとの最新特徴量を保存中...")
save_cols = ["horse_name"] + [c for c in FEATURE_COLS if c in df.columns]
# 各馬の最後のレース行を取得（= その馬の次レースで使える直近成績）
latest = (df.sort_values("date")
            .groupby("horse_name", sort=False)
            .last()
            .reset_index()
           [save_cols])
latest_path = DATA_DIR / "horse_latest_features.parquet"
latest.to_parquet(latest_path, index=False)
print(f"  保存完了: {latest_path}  ({len(latest):,}頭)")
print("\n全ステップ完了！ev_calculator.py にLightGBMが組み込まれます。")
