"""
horse_latest_features.parquet を高速更新するスクリプト。
モデルの再学習は不要。特徴量テーブルだけ更新する（所要時間 約1〜2分）。

実行方法: python update_features.py

使いどころ:
  - 今週のTFJVデータを convert_tfjv.py で取り込んだ後に実行
  - レース前に最新の「直前走」データを反映させたい時
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

print("=" * 50)
print("特徴量テーブル更新スクリプト")
print("=" * 50)

# ------------------------------------------------
# STEP 1: データ読み込み
# ------------------------------------------------
print("\nSTEP 1: データ読み込み")
df = pd.read_parquet(DATA_DIR / "tfjv_all.parquet")
print(f"  {len(df):,}行 読み込み完了")

for col in ["horse_name", "jockey", "trainer", "sire", "venue"]:
    if col in df.columns:
        df[col] = df[col].str.strip()

for col in ["weight_carried", "horse_no", "age"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values(["horse_name", "date"]).reset_index(drop=True)

# ------------------------------------------------
# STEP 2: スピードフィギュア
# ------------------------------------------------
print("STEP 2: スピードフィギュア計算")
baseline = df.groupby(
    ["venue", "surface", "distance", "track_condition"], observed=True
)["finish_time"].transform("median")
df["speed_figure"] = baseline - df["finish_time"]
df["win_flag"]   = (df["rank"] == 1).astype(int)
df["place_flag"] = (df["rank"] <= 3).astype(int)
df["distance_cat"] = df["distance"].apply(
    lambda d: "短距離" if d <= 1400 else "マイル" if d <= 1800 else "中距離" if d <= 2200 else "長距離"
)
print(f"  speed_figure: {df['speed_figure'].notna().sum():,}件")

# ------------------------------------------------
# STEP 3: ローリング特徴量
# ------------------------------------------------
print("STEP 3: ローリング特徴量")
grp = df.groupby("horse_name", sort=False)

df["rank_avg3"]      = grp["rank"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["rank_avg5"]      = grp["rank"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
df["rank_best5"]     = grp["rank"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).min())
df["speed_fig_avg3"] = grp["speed_figure"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["last3f_avg3"]    = grp["last_3f"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
df["wins_last5"]     = grp["win_flag"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
df["places_last5"]   = grp["place_flag"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
df["days_since_prev"] = grp["date"].transform(lambda x: pd.to_datetime(x).diff().dt.days)
df["weight_trend3"]  = grp["horse_weight"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=2).apply(
        lambda w: w.iloc[-1] - w.iloc[0] if len(w) >= 2 else np.nan, raw=False))
print("  ローリング特徴量完了")

# ------------------------------------------------
# STEP 3b: 言い訳分析特徴量
# ------------------------------------------------
print("STEP 3b: 言い訳分析特徴量")
if "field_size" in df.columns:
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce")

for c_col in ["corner1", "corner2", "corner3", "corner4"]:
    if c_col in df.columns:
        df[f"{c_col}_pos"] = pd.to_numeric(
            df[c_col].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")

if "corner4_pos" in df.columns and "field_size" in df.columns:
    fs = df["field_size"].clip(lower=1)
    df["closing_move_raw"] = (df["corner4_pos"] / fs) - (df["rank"] / fs)

c_pos_cols = [c for c in ["corner1_pos","corner2_pos","corner3_pos","corner4_pos"] if c in df.columns]
if len(c_pos_cols) >= 2:
    df["corner_pos_var_raw"] = df[c_pos_cols].var(axis=1)

if "last_3f" in df.columns and "race_id" in df.columns:
    df["last3f_rank_in_race_raw"] = df.groupby("race_id")["last_3f"].rank(
        method="min", ascending=True, na_option="bottom")

if "finish_time" in df.columns and "last_3f" in df.columns and "distance" in df.columns:
    ft      = df["finish_time"].copy()
    ft_sec  = (ft // 1000) * 60 + (ft % 1000) / 10
    l3f_sec = df["last_3f"]
    early_s = (ft_sec - l3f_sec).clip(lower=0.1)
    early_d = (df["distance"] - 600).clip(lower=100)
    df["pci_raw"] = (early_s / (early_d / 600.0)) / l3f_sec.clip(lower=0.1) * 100
    df["pci_raw"] = df["pci_raw"].clip(lower=60, upper=140)

grp2 = df.groupby("horse_name", sort=False)
for raw_col, feat_col in [
    ("closing_move_raw",       "closing_move"),
    ("corner_pos_var_raw",     "corner_pos_var"),
    ("last3f_rank_in_race_raw","last3f_rank_in_race"),
    ("pci_raw",                "pci"),
]:
    if raw_col in df.columns:
        df[feat_col] = grp2[raw_col].transform(lambda x: x.shift(1))
print("  言い訳分析特徴量完了")

# ------------------------------------------------
# STEP 3c: クラスレベル
# ------------------------------------------------
def _class_level(race_name) -> int:
    if pd.isna(race_name): return 3
    s = str(race_name)
    if "G1" in s:                                   return 8
    if "G2" in s:                                   return 7
    if "G3" in s or "重賞" in s:                   return 6
    if "オープン" in s or "OPEN" in s or "3勝" in s: return 5
    if "2勝" in s:                                  return 4
    if "1勝" in s:                                  return 3
    if "未勝利" in s:                               return 2
    if "新馬" in s:                                 return 1
    return 3

if "race_name" in df.columns:
    df["class_level"]      = df["race_name"].apply(_class_level)
    grp_cls                = df.groupby("horse_name", sort=False)
    df["prev_class_level"] = grp_cls["class_level"].transform(lambda x: x.shift(1))
    df["class_change"]     = df["class_level"] - df["prev_class_level"]
print("STEP 3c: クラスレベル完了")

# ------------------------------------------------
# STEP 3d: Phase H 追加特徴量
# ------------------------------------------------
grp_h = df.groupby("horse_name", sort=False)

if "speed_figure" in df.columns and "race_id" in df.columns:
    sf_sum = df.groupby("race_id")["speed_figure"].transform("sum")
    sf_cnt = df.groupby("race_id")["speed_figure"].transform("count")
    df["opponent_sf_avg_raw"]   = (sf_sum - df["speed_figure"]) / (sf_cnt - 1).clip(lower=1)
    df["prev_opponent_sf_avg"]  = grp_h["opponent_sf_avg_raw"].transform(lambda x: x.shift(1))

if "weight_carried" in df.columns:
    df["weight_carried_change"] = grp_h["weight_carried"].transform(lambda x: x - x.shift(1))

if "venue" in df.columns and "race_no" in df.columns:
    df["_ym"] = df["date"].str[:7]
    df["meet_race_seq"] = (
        df.groupby(["venue", "_ym"], sort=False)["race_no"]
          .transform(lambda x: pd.to_numeric(x, errors="coerce").rank(method="dense"))
          .astype("Int64")
    )
    df.drop(columns=["_ym"], inplace=True, errors="ignore")

TURN_DIR = {
    "東京":"左", "新潟":"左", "中京":"左", "函館":"左", "札幌":"左",
    "中山":"右", "阪神":"右", "京都":"右", "小倉":"右", "福島":"右",
}
if "venue" in df.columns:
    df["turn_dir"]         = df["venue"].map(TURN_DIR).fillna("不明")
    df["prev_turn_dir"]    = grp_h["turn_dir"].transform(lambda x: x.shift(1))
    df["turn_dir_changed"] = (
        (df["turn_dir"] != df["prev_turn_dir"]) & df["prev_turn_dir"].notna()
    ).astype("Int64")
print("STEP 3d: Phase H 特徴量完了")

# ------------------------------------------------
# STEP 4: 騎手・調教師統計の結合
# ------------------------------------------------
print("STEP 4: 騎手・調教師統計")
jockey_all  = (df.groupby("jockey", sort=False)
               .agg(jockey_win_rate=("win_flag","mean"),
                    jockey_place_rate=("place_flag","mean"),
                    jockey_rides=("win_flag","count"))
               .reset_index())
jockey_long = (df[df["popularity"] >= 6]
               .groupby("jockey", sort=False)
               .agg(jockey_longshot_win_rate=("win_flag","mean"),
                    jockey_longshot_place_rate=("place_flag","mean"))
               .reset_index())
trainer_all = (df.groupby("trainer", sort=False)
               .agg(trainer_win_rate=("win_flag","mean"),
                    trainer_place_rate=("place_flag","mean"))
               .reset_index())
df = df.merge(jockey_all,  on="jockey",  how="left")
df = df.merge(jockey_long, on="jockey",  how="left")
df = df.merge(trainer_all, on="trainer", how="left")

for stats_file, key_col, rate_col in [
    (DATA_DIR / "sire_stats.parquet",           "sire",    "sire_win_rate"),
    (DATA_DIR / "damsire_stats.parquet",         "damsire", "damsire_win_rate"),
    (DATA_DIR / "venue_distance_stats.parquet",  None,      "vd_win_rate"),
]:
    if stats_file.exists():
        _tmp = pd.read_parquet(stats_file)
        if "win_rate" in _tmp.columns:
            _tmp = _tmp.rename(columns={"win_rate": rate_col})
        if key_col and key_col in df.columns and key_col in _tmp.columns:
            df = df.merge(_tmp[[key_col, "distance_cat", rate_col]], on=[key_col, "distance_cat"], how="left")
        elif key_col is None:
            df = df.merge(_tmp[["venue", "distance_cat", rate_col]], on=["venue","distance_cat"], how="left")
print("  統計結合完了")

# ------------------------------------------------
# STEP 9: horse_latest_features 保存
# ------------------------------------------------
print("\nSTEP 9: horse_latest_features 保存")

# モデルの FEATURE_COLS を読み込む（あれば）
feature_cols_path = DATA_DIR / "lgbm_feature_cols.json"
if feature_cols_path.exists():
    with open(feature_cols_path) as f:
        FEATURE_COLS = json.load(f)
    print(f"  特徴量リスト読み込み: {len(FEATURE_COLS)}個")
else:
    # フォールバック: 主要列のみ
    FEATURE_COLS = [
        "surface","track_condition","venue","distance","distance_cat","field_size",
        "weight_carried","horse_no","age","sex","horse_weight","popularity",
        "rank_avg3","rank_avg5","rank_best5","speed_fig_avg3","last3f_avg3",
        "wins_last5","places_last5","days_since_prev","weight_trend3",
        "closing_move","corner_pos_var","pci","last3f_rank_in_race",
        "prev_class_level","class_change","prev_opponent_sf_avg",
        "weight_carried_change","meet_race_seq","turn_dir_changed",
        "jockey_win_rate","jockey_place_rate","jockey_rides",
        "jockey_longshot_win_rate","jockey_longshot_place_rate",
        "trainer_win_rate","trainer_place_rate",
    ]
    for _opt in ["sire_win_rate","damsire_win_rate","vd_win_rate"]:
        if _opt in df.columns:
            FEATURE_COLS.append(_opt)
    print("  特徴量リスト（フォールバック）")

EXCUSE_RAW_COLS = [
    "rank","speed_figure","last_3f","surface","distance","track_condition",
    "field_size","last3f_rank_in_race_raw","closing_move_raw","corner_pos_var_raw",
    "pci_raw","corner1_pos","popularity","class_level","opponent_sf_avg_raw",
]

# 第29波: 推論時供給用（train_lgbm と同じ定義）
if "stable" in df.columns:
    df["is_west"] = df["stable"].astype(str).str.contains("栗", na=False).astype(int)
if "time_diff" in df.columns:
    # horse_latest は「最新行 = その馬の前走そのもの」なので shift 不要
    df["prev_margin"] = pd.to_numeric(df["time_diff"], errors="coerce").clip(-3, 5)

save_cols = (["horse_name"]
             + [c for c in FEATURE_COLS if c in df.columns]
             + [c for c in EXCUSE_RAW_COLS if c in df.columns and c not in FEATURE_COLS])

latest = (df.sort_values("date")
            .groupby("horse_name", sort=False)
            .last()
            .reset_index()
           [[c for c in save_cols if c in df.columns]])

latest_path = DATA_DIR / "horse_latest_features.parquet"
latest.to_parquet(latest_path, index=False)
print(f"  保存完了: {latest_path}")
print(f"  馬数: {len(latest):,}頭")

# 最新データ日付を表示
if "date" in df.columns:
    print(f"  最新データ日付: {df['date'].max()}")

print("\n[完了] 特徴量テーブル更新完了！アプリを再起動してください。")
