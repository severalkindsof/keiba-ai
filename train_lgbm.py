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

# 第29波: 放置されていた TFJV 列の特徴量化
#   is_west: 栗東（関西）所属 = 1。関西馬優位は実在の定番ファクター
#   prev_margin: 前走の着差（秒）。勝ち方・負け方の質（shift(1) でリーク防止）
if "stable" in df.columns:
    df["is_west"] = df["stable"].astype(str).str.contains("栗", na=False).astype(int)
else:
    df["is_west"] = 0
if "time_diff" in df.columns:
    _td = pd.to_numeric(df["time_diff"], errors="coerce")
    df["prev_margin"] = _td.groupby(df["horse_name"]).shift(1).clip(-3, 5)
else:
    df["prev_margin"] = 0.0
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
# 3b. 言い訳分析用の派生特徴量（近走敗走の原因検出に使用）
# ============================================================
print("\n" + "=" * 50)
print("STEP 3b: 言い訳分析特徴量")
print("=" * 50)

# field_size を数値型に変換（文字列のまま残っている場合があるため）
if "field_size" in df.columns:
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce")

# [A] コーナー通過順位をパース（"5-3" → 5.0）
print("  コーナー位置をパース中...")
for c_col in ["corner1", "corner2", "corner3", "corner4"]:
    if c_col in df.columns:
        df[f"{c_col}_pos"] = pd.to_numeric(
            df[c_col].astype(str).str.extract(r"^(\d+)")[0], errors="coerce")

# --- 現在レースの「言い訳指標」を計算（horse_latest_features 保存用） ---
# ※ これらは現レースの結果を含むため、そのままでは LGBM 特徴量にできない（データリーク）
# ※ 訓練用は shift(1) した「前走の値」を使う（下記 [F] 参照）

# [B] 4コーナー→ゴール改善度（正 = 順位を上げた）
if "corner4_pos" in df.columns and "field_size" in df.columns:
    fs = df["field_size"].clip(lower=1)
    df["closing_move_raw"] = (df["corner4_pos"] / fs) - (df["rank"] / fs)

# [C] コーナー通過順位の分散（揉まれ/展開ロス指標）
c_pos_cols = [c for c in ["corner1_pos","corner2_pos","corner3_pos","corner4_pos"] if c in df.columns]
if len(c_pos_cols) >= 2:
    df["corner_pos_var_raw"] = df[c_pos_cols].var(axis=1)

# [D] 上がり3F レース内順位
if "last_3f" in df.columns and "race_id" in df.columns:
    df["last3f_rank_in_race_raw"] = df.groupby("race_id")["last_3f"].rank(
        method="min", ascending=True, na_option="bottom")

# [E] PCI（Pace Change Index）: 前半÷後半ペース比
# pci < 100 = 前半速い（前残りペース）/ pci > 100 = 後半速い（差し有利ペース）
if "finish_time" in df.columns and "last_3f" in df.columns and "distance" in df.columns:
    # POLISH-1: PCI 計算を pci_calculator に委譲（1ソース化、式変更時の不整合根絶）
    from pci_calculator import calc_pci_row as _calc_pci_row
    df["pci_raw"] = df.apply(
        lambda r: _calc_pci_row(r.get("finish_time"), r.get("last_3f"), r.get("distance")),
        axis=1,
    )
    df["pci_raw"] = pd.to_numeric(df["pci_raw"], errors="coerce")
    print(f"  pci_raw (pci_calculator 経由): 平均={df['pci_raw'].mean():.1f}, 中央値={df['pci_raw'].median():.1f}, notna={df['pci_raw'].notna().sum():,}")

# [F] 訓練用特徴量 = 各指標を shift(1)（前走の値）でリーク防止
grp2 = df.groupby("horse_name", sort=False)
print("  前走値にshift(1)適用中（データリーク防止）...")
for raw_col, feat_col in [
    ("closing_move_raw",       "closing_move"),
    ("corner_pos_var_raw",     "corner_pos_var"),
    ("last3f_rank_in_race_raw","last3f_rank_in_race"),
    ("pci_raw",                "pci"),
]:
    if raw_col in df.columns:
        df[feat_col] = grp2[raw_col].transform(lambda x: x.shift(1))
        print(f"  {feat_col}: {df[feat_col].notna().sum():,}件")

print("  言い訳分析特徴量完了")

# ============================================================
# 3c. クラスレベル特徴量（Phase B）
# ============================================================
print("\n" + "=" * 50)
print("STEP 3c: クラスレベル特徴量")
print("=" * 50)

def _class_level(race_name) -> int:
    """race_name からクラスを数値化（大きいほど格上）"""
    if pd.isna(race_name): return 3
    s = str(race_name)
    if "G1" in s:                              return 8
    if "G2" in s:                              return 7
    if "G3" in s or "重賞" in s:              return 6
    if "オープン" in s or "OPEN" in s or "3勝" in s: return 5
    if "2勝" in s:                             return 4
    if "1勝" in s:                             return 3
    if "未勝利" in s:                          return 2
    if "新馬" in s:                            return 1
    return 3  # デフォルト: 1勝クラス相当

if "race_name" in df.columns:
    df["class_level"] = df["race_name"].apply(_class_level)
    # 前走クラスレベル（shift(1)でリーク防止）
    grp_cls = df.groupby("horse_name", sort=False)
    df["prev_class_level"] = grp_cls["class_level"].transform(lambda x: x.shift(1))
    df["class_change"]     = df["class_level"] - df["prev_class_level"]
    print(f"  class_level: {df['class_level'].value_counts().to_dict()}")

# ============================================================
# 3d. 追加特徴量 Phase H（相手強度・斤量変化・馬場劣化・回り方向）
# ============================================================
print("\n" + "=" * 50)
print("STEP 3d: Phase H 追加特徴量")
print("=" * 50)

grp_h = df.groupby("horse_name", sort=False)

# [H-1] 前走相手強度スコア（同レース他馬の平均速度指数）
if "speed_figure" in df.columns and "race_id" in df.columns:
    sf_sum   = df.groupby("race_id")["speed_figure"].transform("sum")
    sf_cnt   = df.groupby("race_id")["speed_figure"].transform("count")
    df["opponent_sf_avg_raw"] = (sf_sum - df["speed_figure"]) / (sf_cnt - 1).clip(lower=1)
    df["prev_opponent_sf_avg"] = grp_h["opponent_sf_avg_raw"].transform(lambda x: x.shift(1))
    print(f"  prev_opponent_sf_avg: {df['prev_opponent_sf_avg'].notna().sum():,}件")

# [H-2] 斤量変化（前走比）
if "weight_carried" in df.columns:
    df["weight_carried_change"] = grp_h["weight_carried"].transform(lambda x: x - x.shift(1))
    print(f"  weight_carried_change: {df['weight_carried_change'].notna().sum():,}件")

# [H-3] 開催通算レース数（馬場劣化度）: venue × 年月 内の race_no 順位
if "venue" in df.columns and "race_no" in df.columns:
    df["_ym"] = df["date"].str[:7]  # "YYYY-MM"
    df["meet_race_seq"] = (
        df.groupby(["venue", "_ym"], sort=False)["race_no"]
          .transform(lambda x: pd.to_numeric(x, errors="coerce").rank(method="dense"))
          .astype("Int64")
    )
    df.drop(columns=["_ym"], inplace=True, errors="ignore")
    print(f"  meet_race_seq: {df['meet_race_seq'].notna().sum():,}件")

# [H-6] コース回り方向フラグ（右/左回り）
TURN_DIR = {
    "東京":"左", "新潟":"左", "中京":"左", "函館":"左", "札幌":"左",
    "中山":"右", "阪神":"右", "京都":"右", "小倉":"右", "福島":"右",
}
if "venue" in df.columns:
    df["turn_dir"] = df["venue"].map(TURN_DIR).fillna("不明")
    df["prev_turn_dir"] = grp_h["turn_dir"].transform(lambda x: x.shift(1))
    df["turn_dir_changed"] = (
        (df["turn_dir"] != df["prev_turn_dir"]) & df["prev_turn_dir"].notna()
    ).astype("Int64")
    print(f"  turn_dir_changed: {df['turn_dir_changed'].notna().sum():,}件")

print("  Phase H 特徴量完了")

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

# A-5: 調教師の直近 form（rolling 騎乗ベース） — 上り調子の厩舎を捕捉
# BUG-X2: date が str dtype のケースに対応、日付演算ではなく rolling のみで実装
print("\n  調教師 form 計算中...")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["trainer", "date"]).reset_index(drop=True)
g = df.groupby("trainer", sort=False)
# 直近50騎乗での勝率（30日相当の代用、shift(1)でリーク防止）
_t50_wins  = g["win_flag"].transform(lambda x: x.shift(1).rolling(50, min_periods=10).sum())
_t50_rides = g["win_flag"].transform(lambda x: x.shift(1).rolling(50, min_periods=10).count())
df["trainer_recent50_rides"]   = _t50_rides
df["trainer_recent50_winrate"] = (_t50_wins / _t50_rides.clip(lower=1)).fillna(0.08)
# 上り調子 trend: 直近10騎乗 vs 直近50騎乗 勝率差
_t10_wins = g["win_flag"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
df["trainer_trend"] = (
    (_t10_wins / 10.0) - (_t50_wins / _t50_rides.clip(lower=1))
).fillna(0.0)
print(f"  trainer_recent50_winrate / trainer_trend 計算完了 "
      f"(notna={df['trainer_recent50_winrate'].notna().sum():,})")

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

# [H-4] 母父（damsire）距離適性
damsire_path = DATA_DIR / "damsire_stats.parquet"
if damsire_path.exists() and "damsire" in df.columns:
    print("母父統計を結合中...")
    dam_df = pd.read_parquet(damsire_path).rename(columns={"win_rate": "damsire_win_rate"})
    df = df.merge(dam_df[["damsire", "distance_cat", "damsire_win_rate"]],
                  on=["damsire", "distance_cat"], how="left")
    print(f"  damsire_win_rate 取得件数: {df['damsire_win_rate'].notna().sum():,}")

# [H-5] venue × distance_cat 交差勝率
vd_path = DATA_DIR / "venue_distance_stats.parquet"
if vd_path.exists():
    print("会場×距離統計を結合中...")
    vd_df = pd.read_parquet(vd_path)
    df = df.merge(vd_df[["venue", "distance_cat", "vd_win_rate"]],
                  on=["venue", "distance_cat"], how="left")
    print(f"  vd_win_rate 取得件数: {df['vd_win_rate'].notna().sum():,}")

# ============================================================
# 5.5 欠損値修正(第88波): 0=データ無しをNaN化してLGBMの欠損処理に委ねる
#   horse_weight=0(不明)やjockey/血統rate=0(無join/極小標本)を0のまま学習すると
#   「超小型馬/最低騎手=悪」と誤学習し、実力ある馬に偽ペナルティ(ビザンチン例)。NaNなら中立扱い。
# ============================================================
import numpy as _np
_zero_to_nan = ["horse_weight", "jockey_win_rate", "jockey_place_rate",
                "trainer_win_rate", "trainer_place_rate", "sire_win_rate", "damsire_win_rate"]
for _c in _zero_to_nan:
    if _c in df.columns:
        _n0 = int((df[_c] == 0).sum())
        df[_c] = df[_c].replace(0, _np.nan)
        if _n0:
            print(f"  欠損修正 {_c}: {_n0:,}件の0をNaN化")

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
    "is_west",          # 第29波: 栗東（関西）所属フラグ
    "prev_margin",      # 第29波: 前走着差（秒、clip -3〜5）
    # weight_change は列マッピング不正のため除外（TODO: TFJV列定義確認後に復活）
    # ---- 市場評価 ----
    # 第88波: 本命用(lgbm_win_model)はpopularity込み=本命ゾーンで市場の正確さを活かす。
    #   穴用は別モデル(lgbm_independent.txt=popularity抜き)で役割分担(バックテストで穴lift1.75x実証)。
    "popularity",       # 人気順 - 本命ゾーンの精度に寄与
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
    # ---- 言い訳分析特徴量（直前走の展開・ペース） ----
    "closing_move",        # 4C→ゴール改善度（不利の代理変数）
    "corner_pos_var",      # 道中展開ロス（コーナー順位分散）
    "pci",                 # ペース傾向（>100=差し有利、<100=前残り）
    "last3f_rank_in_race", # 上がり3Fレース内順位（小=末脚強）
    # ---- クラスレベル（Phase B） ----
    "prev_class_level",    # 前走クラス（昇降級の判定基準）
    "class_change",        # クラス変化（正=昇級、負=降級）
    # ---- Phase H 追加特徴量 ----
    "prev_opponent_sf_avg",  # 前走レース相手強度（他馬の平均速度指数）
    "weight_carried_change", # 斤量変化（前走比）
    "meet_race_seq",         # 開催通算レース数（馬場劣化度）
    "turn_dir_changed",      # コース回り方向変更フラグ
    "damsire_win_rate",      # 母父距離適性
    "vd_win_rate",           # venue × distance_cat 交差勝率
    # ---- 騎手 ----
    "jockey_win_rate",              # 騎手全体勝率
    "jockey_place_rate",            # 騎手全体複勝率
    "jockey_rides",                 # 騎手の総騎乗数（実績の信頼度）
    "jockey_longshot_win_rate",     # 騎手の人気薄での勝率
    "jockey_longshot_place_rate",   # 騎手の人気薄での複勝率
    # ---- 調教師 ----
    "trainer_win_rate",             # 調教師勝率（通算）
    "trainer_place_rate",           # 調教師複勝率（通算）
    # A-5: 調教師 form (直近のトレンド)
    "trainer_recent50_rides",       # 直近50騎乗数
    "trainer_recent50_winrate",     # 直近50騎乗勝率
    "trainer_trend",                # 上り調子トレンド（正=上昇 / 負=下降）
]

# 父系統計（あれば追加）
if "sire_win_rate" in df.columns:
    FEATURE_COLS.append("sire_win_rate")

# 実際に存在する列のみ使用 + Y6: 重複除去（順序保持）
_seen = set()
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns and not (c in _seen or _seen.add(c))]
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

# ランキングラベル: 4段階評価（LightGBMのデフォルト上限31以内に収める）
# field_size - rank は大頭数（障害28頭等）で上限超過するため固定スケールを使用
# 1着=4, 2着=3, 3着=2, 4-5着=1, 6着以下=0
def _make_rank_label(rank):
    if pd.isna(rank): return 0
    r = int(rank)
    if r == 1: return 4
    if r == 2: return 3
    if r == 3: return 2
    if r <= 5: return 1
    return 0
df["rank_label"] = df["rank"].apply(_make_rank_label)

# SUPER-7: 3分割 (train < 2023-07-01 / valid 2023-07-01〜2024-01-01 / test >= 2024-01-01)
# early_stopping は valid を見て、test は純粋なホールドアウト評価用に。
train_mask = df["date"] <  "2023-07-01"
valid_mask = (df["date"] >= "2023-07-01") & (df["date"] < "2024-01-01")
test_mask  = df["date"] >= "2024-01-01"

# レースキー: date + venue + race_no の結合（race_id はTFJVでは馬単位のIDのため使用不可）
df["race_key"] = (df["date"].astype(str) + "_"
                  + df["venue"].astype(str).str.strip() + "_"
                  + df["race_no"].astype(str))

# LambdaRank: race_key でソート必須
train_df = df[train_mask].copy().sort_values("race_key").reset_index(drop=True)
valid_df = df[valid_mask].copy().sort_values("race_key").reset_index(drop=True)
test_df  = df[test_mask].copy().sort_values("race_key").reset_index(drop=True)

print(f"  学習データ:   {len(train_df):,}行  ({train_df['date'].min()} 〜 {train_df['date'].max()})")
print(f"  検証データ:   {len(valid_df):,}行  ({valid_df['date'].min()} 〜 {valid_df['date'].max()})")
print(f"  テストデータ: {len(test_df):,}行  ({test_df['date'].min()} 〜 {test_df['date'].max()})")

# IMPROVE-2: 前処理済みデータを tune_lgbm.py のために保存（リーケージ防止）
print("  前処理済みデータを保存中（tune_lgbm.py 用）...")
_cols_to_save = list(set(FEATURE_COLS + ["race_key", "rank_label", "win_flag", "date", "horse_name"]))
_cols_to_save = [c for c in _cols_to_save if c in train_df.columns]
# カテゴリ列はそのまま保存できないので文字列化
_save_dir = DATA_DIR / "training_cache"
_save_dir.mkdir(exist_ok=True)
def _to_saveable(d):
    d2 = d[_cols_to_save].copy()
    for c in CAT_COLS:
        if c in d2.columns and hasattr(d2[c], "cat"):
            d2[c] = d2[c].astype(str)
    return d2
_to_saveable(train_df).to_parquet(_save_dir / "train_preproc.parquet", index=False)
_to_saveable(valid_df).to_parquet(_save_dir / "valid_preproc.parquet", index=False)
print(f"  保存: {_save_dir}/train_preproc.parquet, valid_preproc.parquet")

X_train = train_df[FEATURE_COLS]
y_train = train_df["rank_label"]
X_valid = valid_df[FEATURE_COLS]
y_valid = valid_df["rank_label"]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df["rank_label"]

# グループ情報（race_key 単位の馬数）
train_groups = train_df.groupby("race_key", sort=False)["race_key"].count().values
valid_groups = valid_df.groupby("race_key", sort=False)["race_key"].count().values
test_groups  = test_df.groupby("race_key",  sort=False)["race_key"].count().values
print(f"  学習グループ数（レース数）: {len(train_groups):,} / 検証: {len(valid_groups):,} / テスト: {len(test_groups):,}")

# ============================================================
# 8. LightGBM LambdaRank 学習（Phase A）
# ============================================================
print("\n" + "=" * 50)
print("STEP 7: LightGBM LambdaRank 学習")
print("=" * 50)

# IMPROVE-2: Optuna で最適化済みパラメータがあれば自動適用
_best_params = {}
_best_params_path = DATA_DIR / "lgbm_best_params.json"
if _best_params_path.exists():
    try:
        with open(_best_params_path, encoding="utf-8") as _f:
            _best_params = json.load(_f).get("params", {})
        print(f"  Optuna 最適パラメータ適用: {_best_params}")
    except Exception:
        _best_params = {}

_default_params = dict(
    objective         = "lambdarank",
    metric            = "ndcg",
    ndcg_eval_at      = [1, 3, 5],
    n_estimators      = 1000,
    learning_rate     = 0.05,
    max_depth         = 6,
    num_leaves        = 31,
    min_child_samples = 20,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)
_default_params.update(_best_params)
model = lgb.LGBMRanker(**_default_params)

model.fit(
    X_train, y_train,
    group      = train_groups,
    eval_set   = [(X_valid, y_valid)],   # SUPER-7: 専用validで early_stop（test には触れない）
    eval_group = [valid_groups],
    callbacks  = [
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

# LambdaRank はスコアを出力（確率ではない）
raw_scores = model.predict(X_test)

# レース内 softmax で勝率に変換してバックテスト
test_df = test_df.copy()
test_df["raw_score"] = raw_scores

# AUC-ROC（スコアと実際の勝利フラグで計算）
from sklearn.metrics import roc_auc_score as _auc, brier_score_loss as _brier
auc = _auc(test_df["win_flag"].astype(int), raw_scores)
print(f"  AUC-ROC: {auc:.4f}  （参考: ランキングスコアと勝利フラグの相関）")

# IMPROVE-1: Brier Score（確率校正の標準指標、小さいほど良）
# 仮校正済み確率 = レース内 softmax → 0-1 確率値
def _softmax_per_race(scores):
    s = scores.values.astype(float)
    e = np.exp(s - s.max())
    return pd.Series(e / e.sum(), index=scores.index)
_prob_pre = (test_df.groupby("race_key", sort=False)["raw_score"]
             .transform(_softmax_per_race))
brier = _brier(test_df["win_flag"].astype(int), _prob_pre)
print(f"  Brier Score: {brier:.5f}  （小さいほど校正度↑、参考: ベースライン 0.0667）")

# 第18波: メトリクスをファイル保存（UI のヒーロー表示等が参照、ハードコード排除）
with open(DATA_DIR / "lgbm_metrics.json", "w", encoding="utf-8") as _fm:
    json.dump({"test_auc": round(float(auc), 4), "test_brier": round(float(brier), 5),
               "trained_at": pd.Timestamp.now().isoformat()[:19]}, _fm, ensure_ascii=False, indent=2)

# IMPROVE-1: Walk-Forward CV（検証期間を3分割して各窓で精度測定）
print("\n  --- Walk-Forward CV（valid期間を3窓に分割） ---")
valid_df_sorted = valid_df.sort_values("date").reset_index(drop=True)
n_v = len(valid_df_sorted)
wf_aucs = []
wf_briers = []
for i in range(3):
    start = i * (n_v // 3)
    end   = (i + 1) * (n_v // 3) if i < 2 else n_v
    chunk = valid_df_sorted.iloc[start:end]
    if len(chunk) == 0:
        continue
    try:
        scores_chunk = model.predict(chunk[FEATURE_COLS])
        chunk = chunk.copy()
        chunk["sc"] = scores_chunk
        prob_chunk = (chunk.groupby("race_key", sort=False)["sc"]
                      .transform(_softmax_per_race))
        a = _auc(chunk["win_flag"].astype(int), scores_chunk)
        b = _brier(chunk["win_flag"].astype(int), prob_chunk)
        wf_aucs.append(a); wf_briers.append(b)
        d0, d1 = chunk["date"].min().date(), chunk["date"].max().date()
        print(f"    Window {i+1} ({d0}〜{d1}, n={len(chunk):,}): "
              f"AUC={a:.4f}  Brier={b:.5f}")
    except Exception as _e_wf:
        print(f"    Window {i+1} エラー: {_e_wf}")
if wf_aucs:
    import numpy as _np
    print(f"  WF平均 AUC={_np.mean(wf_aucs):.4f} ± {_np.std(wf_aucs):.4f}  "
          f"Brier={_np.mean(wf_briers):.5f} ± {_np.std(wf_briers):.5f}")

# レース内 softmax → 勝率換算（transform で各行にスカラーを返す）
def _softmax_transform(scores):
    s = scores.values.astype(float)
    e = np.exp(s - s.max())
    return pd.Series(e / e.sum(), index=scores.index)

test_df["pred_win_rate"] = (
    test_df.groupby("race_key", sort=False)["raw_score"]
           .transform(_softmax_transform)
)

# 人気別の予測精度確認
for pop_label, grp_df in test_df.groupby("pop_bucket", observed=True):
    if len(grp_df) == 0:
        continue
    actual_wr  = grp_df["win_flag"].mean()
    predict_wr = grp_df["pred_win_rate"].mean()
    print(f"  [{pop_label}]  実際の勝率: {actual_wr:.3f}  予測平均: {predict_wr:.3f}  件数: {len(grp_df):,}")

# バックテスト
print("\n--- バックテスト（EV > 0 の馬だけ購入した場合） ---")
TYPICAL_ODDS = {1: 2.5, 2: 4.0, 3: 6.5, 4: 10.0, 5: 14.0,
                6: 20.0, 7: 27.0, 8: 35.0, 9: 45.0}
def pop_to_odds(p):
    try: p = int(p)
    except (ValueError, TypeError): return 10.0
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

# IMPROVE-3: 重要度が低い特徴量をリストアップ（ユーザーが手動で剪定判断）
print("\n--- 特徴量剪定候補（importance < 50） ---")
low_imp = fi[fi < 50]
if len(low_imp) > 0:
    for fname, score in low_imp.items():
        print(f"    {fname:35s} {score:6.0f}")
    print(f"  → 上記 {len(low_imp)} 件を FEATURE_COLS から除外して再学習で精度維持+推論高速化の可能性")
else:
    print("  すべての特徴量が importance >= 50。剪定不要。")

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

# モデルタイプをメタデータとして保存（ev_calculator.py が参照）
meta_path = DATA_DIR / "lgbm_model_meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({"model_type": "ranker"}, f)
print(f"  モデルメタ:   {meta_path}")

# ============================================================
# ROOT-1: Isotonic Regression キャリブレーター保存
# LambdaRankスコア → 勝率確率への変換（業界標準の事後キャリブレーション）
# ============================================================
print("\nIsotonic Regressionキャリブレーターを学習中...")
from sklearn.isotonic import IsotonicRegression
import joblib

# SUPER-7: valid セットでキャリブレータをフィット（test はホールドアウト維持）
raw_scores_val = model.predict(X_valid)
win_labels_val = (y_valid == 4).astype(int)   # rank_label==4 = 1着
print(f"  検証セット（valid）: {len(raw_scores_val):,}件  1着数: {win_labels_val.sum():,}件")
print(f"  スコア範囲: min={raw_scores_val.min():.3f}  max={raw_scores_val.max():.3f}  mean={raw_scores_val.mean():.3f}")

calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
calibrator.fit(raw_scores_val, win_labels_val)

CAL_PATH = DATA_DIR / "lgbm_calibrator.pkl"
joblib.dump(calibrator, CAL_PATH)
print(f"  キャリブレーター保存: {CAL_PATH}")

# キャリブレーション精度確認
cal_probs = calibrator.predict(raw_scores_val)
print(f"  キャリブレーション後 確率範囲: min={cal_probs.min():.4f}  max={cal_probs.max():.4f}  mean={cal_probs.mean():.4f}")

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
# FEATURE_COLS + 言い訳分析用の生値（モデル特徴量以外も保存）
EXCUSE_RAW_COLS = [
    "rank",                      # 前走着順（生値）
    "speed_figure",              # 前走スピードフィギュア（生値）
    "last_3f",                   # 前走上がり3F（生値）
    "surface",                   # 前走馬場種別
    "distance",                  # 前走距離
    "track_condition",           # 前走馬場状態
    "field_size",                # 前走頭数
    "last3f_rank_in_race_raw",   # 前走上がり3Fレース内順位（リーク防止のためraw列を使用）
    "closing_move_raw",          # 前走4C→ゴール改善度（raw）
    "corner_pos_var_raw",        # 前走コーナー順位分散（raw）
    "pci_raw",                   # 前走PCI（raw）
    "corner1_pos",               # 前走1コーナー位置（先行/差し判定）
    "popularity",                # 前走人気（生値）
    "class_level",               # 前走クラスレベル（生値）
    "opponent_sf_avg_raw",       # 前走レース相手強度（生値）
]
save_cols = (["horse_name"]
             + [c for c in FEATURE_COLS if c in df.columns]
             + [c for c in EXCUSE_RAW_COLS
                if c in df.columns and c not in FEATURE_COLS])
# 各馬の最後のレース行を取得（= その馬の次レースで使える直近成績）
latest = (df.sort_values("date")
            .groupby("horse_name", sort=False)
            .last()
            .reset_index()
           [[c for c in save_cols if c in df.columns]])
latest_path = DATA_DIR / "horse_latest_features.parquet"
latest.to_parquet(latest_path, index=False)
print(f"  保存完了: {latest_path}  ({len(latest):,}頭)")

# ============================================================
# SUPER-7 + SUPER-1: Benter blending α/β 自動フィット
# valid セット上で「キャリブレーション済み確率 × 市場確率（人気→経験勝率）」をロジスティックに統合
# ============================================================
print("\nBenter blending α/β を valid セットでフィット中...")
try:
    from market_prob import get_market_prob_series
    from sklearn.linear_model import LogisticRegression

    cal_probs_valid = calibrator.predict(raw_scores_val)
    market_probs_valid = get_market_prob_series(valid_df["popularity"].values).values
    win_y = win_labels_val.values

    eps = 1e-6
    f = np.clip(cal_probs_valid, eps, 1.0)
    pi = np.clip(market_probs_valid, eps, 1.0)
    X_bw = np.column_stack([np.log(f), np.log(pi)])
    mask = ~np.isnan(X_bw).any(axis=1)
    X_bw, y_bw = X_bw[mask], win_y[mask]

    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X_bw, y_bw)
    alpha, beta = float(lr.coef_[0, 0]), float(lr.coef_[0, 1])
    bw_path = DATA_DIR / "benter_weights.json"
    with open(bw_path, "w", encoding="utf-8") as bf:
        json.dump({
            "alpha": alpha,
            "beta": beta,
            "fit_source": "train_lgbm.py valid set",
            "n_train": int(mask.sum()),
        }, bf, ensure_ascii=False, indent=2)
    print(f"  α (自モデル重み) = {alpha:.4f}")
    print(f"  β (市場重み)     = {beta:.4f}")
    print(f"  保存: {bw_path}")
except Exception as _bw_e:
    print(f"  Benter フィット失敗: {_bw_e}")

# Y7: Conformal Prediction を新モデルに合わせて自動再フィット
print("\nConformal Prediction を新モデルで再フィット中...")
try:
    import importlib
    import conformal as _cf
    importlib.reload(_cf)
    _cf.fit_from_training_cache(alpha=0.10)
except Exception as _cfe:
    print(f"  Conformal フィット失敗: {_cfe}")

print("\n全ステップ完了！ev_calculator.py にLightGBMが組み込まれます。")
