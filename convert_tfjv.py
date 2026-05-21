"""
TFJV CSV → Parquet 変換スクリプト（初回一度だけ実行）
実行方法: python convert_tfjv.py
"""
import pandas as pd
from pathlib import Path

OUT_PATH = Path(__file__).parent / "data" / "tfjv_all.parquet"

# 52列形式（2016-2025）
COLS52 = [
    "year","month","day","col3","venue","col5","race_no","race_name",
    "field_size","surface","col10","distance","track_condition",
    "horse_name","sex","age","jockey","weight_carried","col18",
    "horse_no","rank","col21","col22","time_diff","popularity",
    "col25","finish_time","col27","corner1","corner2","corner3","corner4",
    "last_3f","horse_weight","trainer","stable","weight_change",
    "horse_id","jockey_id","trainer_id","race_id",
    "owner","breeder","sire","dam","damsire",
    "coat","birth_date","col48","col49","col50","col51"
]

# 45列形式（2026〜、血統情報なし）
COLS45 = [
    "year","month","day","col3","venue","col5","race_no","race_name",
    "field_size","surface","col10","distance","track_condition",
    "horse_name","sex","age","jockey","weight_carried","col18",
    "horse_no","rank","col21","col22","time_diff","popularity",
    "col25","finish_time","col27","corner1","corner2","corner3","corner4",
    "last_3f","horse_weight","trainer","stable","weight_change",
    "horse_id","jockey_id","trainer_id","race_id",
    "col41","col42","col43","col44"
]

KEEP = [
    "year","month","day","venue","race_no","race_name","surface","distance",
    "track_condition","horse_name","sex","age","jockey","weight_carried",
    "horse_no","rank","time_diff","popularity","finish_time",
    "corner1","corner2","corner3","corner4","last_3f","horse_weight",
    "trainer","stable","weight_change","horse_id","jockey_id","trainer_id",
    "race_id","sire","dam","damsire","birth_date"
]

FILES = {
    Path("C:/TFJV/TXT/2015_2010"):     COLS52,
    Path("C:/TFJV/TXT/2025_2016"):     COLS52,
    Path("C:/TFJV/TXT/2026_5_20.txt"): COLS52,
}

def categorize_distance(d):
    if d <= 1400: return "短距離"
    if d <= 1800: return "マイル"
    if d <= 2200: return "中距離"
    return "長距離"

print("読み込み中...")
dfs = []
for p, cols in FILES.items():
    if not p.exists():
        print(f"  {p.name} スキップ（見つからない）")
        continue
    print(f"  {p.name} ({len(cols)}列)...")
    keep_cols = [c for c in KEEP if c in cols]
    df_tmp = pd.read_csv(p, header=None, names=cols,
                         encoding="cp932", dtype=str, on_bad_lines="skip",
                         usecols=[cols.index(c) for c in keep_cols])
    # 不足列はNaNで補完
    for c in KEEP:
        if c not in df_tmp.columns:
            df_tmp[c] = None
    dfs.append(df_tmp[KEEP])

df = pd.concat(dfs, ignore_index=True).drop_duplicates()
print(f"合計行数: {len(df):,}")

# 日付
df["date"] = "20" + df["year"].str.zfill(2) + "-" + df["month"].str.zfill(2) + "-" + df["day"].str.zfill(2)

# 数値変換
for col in ["rank","distance","popularity","last_3f","horse_weight","weight_change"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["finish_time"] = pd.to_numeric(df["finish_time"].str.strip('"'), errors="coerce")

# surface統一
df["surface"] = df["surface"].str.strip().map({"芝":"芝","ダ":"ダート","ダート":"ダート"}).fillna(df["surface"])

# 派生列
df["distance_cat"] = df["distance"].apply(lambda x: categorize_distance(int(x)) if pd.notna(x) else "中距離")
df["win_flag"]     = (df["rank"] == 1).astype("int8")
df["place_flag"]   = (df["rank"] <= 3).astype("int8")
df["pop_bucket"]   = pd.cut(df["popularity"], bins=[0,1,3,5,9,99],
                             labels=["1番人気","2-3番人気","4-5番人気","6-9番人気","10番人気以下"])

OUT_PATH.parent.mkdir(exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"保存完了: {OUT_PATH}")
print(f"ファイルサイズ: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# ============================================================
# 馬別プロファイル指標の事前計算（horse_stats.parquet）
# ============================================================
print("\n馬別指標を計算中...")
hs = df.copy()
hs["horse_name"] = hs["horse_name"].str.strip()
hs["top5_flag"]  = (hs["rank"] <= 5).astype("int8")

# 上がり3F 同レース内順位
hs["last3f_rank"] = hs.groupby("race_id")["last_3f"].rank(
    method="min", ascending=True, na_option="bottom")
hs["last3f_top2"] = (hs["last3f_rank"] <= 2).astype("int8")

# 基本集計
base = hs.groupby("horse_name", sort=False).agg(
    total_races   = ("rank",        "count"),
    place_count   = ("place_flag",  "sum"),
    top5_count    = ("top5_flag",   "sum"),
    last3f_top2_c = ("last3f_top2", "sum"),
    last_date     = ("date",        "max"),
    age           = ("age",         "last"),
).reset_index()
base["place_rate"]  = base["place_count"]   / base["total_races"]
base["top5_rate"]   = base["top5_count"]    / base["total_races"]
base["last3f_rate"] = base["last3f_top2_c"] / base["total_races"]
import datetime as _dt
base["days_since_last"] = (
    pd.Timestamp.now() - pd.to_datetime(base["last_date"], errors="coerce")
).dt.days.fillna(999).astype(int)

# 道悪複勝率
wet = hs[hs["track_condition"].isin(["重","不良"])].groupby("horse_name", sort=False).agg(
    wet_races  = ("place_flag","count"),
    wet_places = ("place_flag","sum"),
).reset_index()
wet = wet[wet["wet_races"] >= 2].copy()
wet["wet_place_rate"] = wet["wet_places"] / wet["wet_races"]
base = base.merge(wet[["horse_name","wet_place_rate"]], on="horse_name", how="left")

# 距離帯別複勝率（列として展開）
dist_grp = hs.groupby(["horse_name","distance_cat"], sort=False, observed=True).agg(
    dr=("place_flag","count"), dp=("place_flag","sum")).reset_index()
dist_grp = dist_grp[dist_grp["dr"] >= 2].copy()
dist_grp["drate"] = dist_grp["dp"] / dist_grp["dr"]
dist_pivot = dist_grp.pivot_table(
    index="horse_name", columns="distance_cat", values="drate").reset_index()
dist_pivot.columns = ["horse_name"] + [f"dist_{c}" for c in dist_pivot.columns[1:]]
base = base.merge(dist_pivot, on="horse_name", how="left")

HS_PATH = OUT_PATH.parent / "horse_stats.parquet"
base.to_parquet(HS_PATH, index=False)
print(f"馬別指標保存完了: {HS_PATH}")
print(f"  馬数: {len(base):,}頭")

# ============================================================
# 勝率テーブル・父系統計・騎手統計の事前計算
# ============================================================
print("\n統計テーブルを計算中...")

# 勝率テーブル（surface × distance_cat × pop_bucket）
df["pop_bucket"] = pd.cut(df["popularity"], bins=[0,1,3,5,9,99],
    labels=["1番人気","2-3番人気","4-5番人気","6-9番人気","10番人気以下"])
wrt = (df.groupby(["surface","distance_cat","pop_bucket"], observed=True)
       .agg(races=("win_flag","count"), wins=("win_flag","sum"), places=("place_flag","sum"))
       .reset_index())
wrt["win_rate"]   = wrt["wins"]   / wrt["races"]
wrt["place_rate"] = wrt["places"] / wrt["races"]
wrt.to_parquet(OUT_PATH.parent / "win_rate_table.parquet", index=False)
print(f"  勝率テーブル: {len(wrt)}行")

# 父系統計
if "sire" in df.columns:
    sire = (df.groupby(["sire","distance_cat"], observed=True)
            .agg(races=("win_flag","count"), wins=("win_flag","sum"))
            .reset_index())
    sire = sire[sire["races"] >= 10]
    sire["win_rate"] = sire["wins"] / sire["races"]
    sire.to_parquet(OUT_PATH.parent / "sire_stats.parquet", index=False)
    print(f"  父系統計: {len(sire)}行")

# 騎手統計（穴騎手）
if "jockey" in df.columns:
    ls = df[df["popularity"] >= 10]
    jky = (ls.groupby("jockey")
           .agg(rides=("win_flag","count"), wins=("win_flag","sum"), places=("place_flag","sum"))
           .reset_index())
    jky = jky[jky["rides"] >= 20]
    jky["place_rate_longshot"] = jky["places"] / jky["rides"]
    jky["win_rate_longshot"]   = jky["wins"]   / jky["rides"]
    jky.to_parquet(OUT_PATH.parent / "jockey_stats.parquet", index=False)
    print(f"  騎手統計: {len(jky)}行")

print("\n全テーブル保存完了。次回起動から瞬時に読み込まれます。")
