"""
TFJV CSV → Parquet 変換スクリプト

使い方:
    python convert_tfjv.py              # フル変換（800k 行・約90秒）
    python convert_tfjv.py --incremental # 差分更新（既存 parquet の最終日以降のみ）

C-4 (第12波):
    --incremental では既存 parquet を読み、各 CSV ファイルのうち
    parquet の mtime より新しいファイルのみ再読込し、最終日以降の行のみ追加。
    集計テーブル（horse_stats / win_rate_table / 父系統計 / トラックバイアス）は
    全期間が必要なため毎回再構築する（差分にしない）。
"""
import argparse
import sys
import time
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--incremental", action="store_true",
                    help="既存 parquet の最終日以降のみ追加（mtime 比較で読込スキップ）")
args = parser.parse_args()

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
    "field_size",          # ← 追加: 出走頭数（一変候補・PCI計算に使用）
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
    # 2026-05-23 追加: オークス・平安S 出走馬の近走データ
    Path("C:/TFJV/TXT/okus.csv"):      COLS52,
    Path("C:/TFJV/TXT/heian.csv"):     COLS52,
    # 2026-06-10 追加: 2026年度全成績（5/24オークス〜6/7まで）
    Path("C:/TFJV/TXT/2026_06_10.txt"): COLS52,
    # 2026-06-12 追加: 6/13土・6/14日 出走馬の過去成績（宝塚記念週）
    Path("C:/TFJV/TXT/DS260613.CSV"): COLS52,
    Path("C:/TFJV/TXT/DS260614.CSV"): COLS52,
}

def categorize_distance(d):
    if d <= 1400: return "短距離"
    if d <= 1800: return "マイル"
    if d <= 2200: return "中距離"
    return "長距離"

t0 = time.time()

# --- C-4: 差分モード判定 ---
existing_df: pd.DataFrame | None = None
parquet_mtime = 0.0
existing_max_date = None
if args.incremental and OUT_PATH.exists():
    print("[incremental] 既存 parquet を読込中...")
    existing_df = pd.read_parquet(OUT_PATH)
    parquet_mtime = OUT_PATH.stat().st_mtime
    existing_max_date = pd.to_datetime(existing_df["date"], errors="coerce").max()
    print(f"  既存: {len(existing_df):,} 行 / 最終日={existing_max_date.date() if pd.notna(existing_max_date) else 'なし'}")

print("読み込み中...")
dfs = []
for p, cols in FILES.items():
    if not p.exists():
        print(f"  {p.name} スキップ（見つからない）")
        continue
    # 差分モード: parquet より古い CSV はスキップ
    if args.incremental and existing_df is not None:
        if p.stat().st_mtime <= parquet_mtime:
            print(f"  {p.name} スキップ (CSV mtime <= parquet mtime)")
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

if not dfs and existing_df is not None:
    # 差分モードで新規 CSV ゼロ → 既存をそのまま使う（集計だけ再構築）
    print("[incremental] 新規 CSV なし、既存 parquet を再利用して集計のみ再構築")
    df = existing_df.copy()
elif args.incremental and existing_df is not None:
    new_df = pd.concat(dfs, ignore_index=True)
    # 既存最終日より後の行のみ抽出
    new_df["_date_tmp"] = "20" + new_df["year"].str.zfill(2) + "-" + new_df["month"].str.zfill(2) + "-" + new_df["day"].str.zfill(2)
    new_df_dt = pd.to_datetime(new_df["_date_tmp"], errors="coerce")
    if existing_max_date is not None and pd.notna(existing_max_date):
        new_df = new_df[new_df_dt > existing_max_date]
    new_df = new_df.drop(columns=["_date_tmp"])
    print(f"  新規追加行: {len(new_df):,}")
    # 既存と結合 (existing_df は既に派生列含むので、KEEP のみ取り出す)
    existing_base = existing_df[KEEP] if all(c in existing_df.columns for c in KEEP) else existing_df
    df = pd.concat([existing_base, new_df], ignore_index=True)
    # race_id + horse_no で重複排除（後勝ち）
    if "race_id" in df.columns and "horse_no" in df.columns:
        df = df.drop_duplicates(subset=["race_id", "horse_no"], keep="last")
    else:
        df = df.drop_duplicates()
else:
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
print(f"合計行数: {len(df):,} (経過 {time.time()-t0:.1f}s)")

# 日付
# 第42波: 既存parquet(数値型)とCSV(文字列型)concat後の str accessor 対策
for _c in ("year", "month", "day"):
    df[_c] = df[_c].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
df["date"] = "20" + df["year"].str.zfill(2) + "-" + df["month"].str.zfill(2) + "-" + df["day"].str.zfill(2)

# 数値変換
for col in ["rank","distance","popularity","last_3f","horse_weight","weight_change"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# 第41波: 既存parquet(数値型)とCSV(文字列型)のconcat後に str.strip が落ちる対策
if df["finish_time"].dtype == object:
    df["finish_time"] = df["finish_time"].astype(str).str.strip('"')
df["finish_time"] = pd.to_numeric(df["finish_time"], errors="coerce")

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

# 母父（damsire）統計（Phase H-4）
if "damsire" in df.columns:
    damsire = (df[df["damsire"].notna() & (df["damsire"].str.strip() != "")]
               .groupby(["damsire","distance_cat"], observed=True)
               .agg(races=("win_flag","count"), wins=("win_flag","sum"))
               .reset_index())
    damsire = damsire[damsire["races"] >= 10]
    damsire["win_rate"] = damsire["wins"] / damsire["races"]
    damsire.to_parquet(OUT_PATH.parent / "damsire_stats.parquet", index=False)
    print(f"  母父統計: {len(damsire)}行")

# venue × distance_cat 交差勝率（Phase H-5）
vd_tbl = (df.groupby(["venue","distance_cat"], observed=True)
            .agg(races=("win_flag","count"), wins=("win_flag","sum"))
            .reset_index())
vd_tbl = vd_tbl[vd_tbl["races"] >= 30]
vd_tbl["vd_win_rate"] = vd_tbl["wins"] / vd_tbl["races"]
vd_tbl.to_parquet(OUT_PATH.parent / "venue_distance_stats.parquet", index=False)
print(f"  会場×距離統計: {len(vd_tbl)}行")

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

# ============================================================
# Phase E: 動的トラックバイアステーブル（venue × month × surface × 馬番帯）
# ============================================================
print("\nトラックバイアステーブルを計算中...")
df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
df["gate_band"] = pd.cut(
    pd.to_numeric(df["horse_no"], errors="coerce"),
    bins=[0, 4, 8, 12, 18], labels=["1-4", "5-8", "9-12", "13+"]
)

bias_tbl = (
    df.groupby(["venue", "month", "surface", "gate_band"], observed=True)
      .agg(races=("win_flag", "count"), wins=("win_flag", "sum"))
      .reset_index()
)
bias_tbl = bias_tbl[bias_tbl["races"] >= 30].copy()
bias_tbl["bias_win_rate"] = bias_tbl["wins"] / bias_tbl["races"]
overall_wr = df["win_flag"].mean()
bias_tbl["bias_score"] = (bias_tbl["bias_win_rate"] - overall_wr).round(4)

BIAS_PATH = OUT_PATH.parent / "track_bias_table.parquet"
bias_tbl.to_parquet(BIAS_PATH, index=False)
print(f"  トラックバイアステーブル: {len(bias_tbl)}行 → {BIAS_PATH}")

# ============================================================
# 第29波: データ健康診断（finish_time 79.7万行全滅の再発防止）
# ============================================================
print("\n=== データ健康診断 ===")
_CRITICAL_COLS = {"finish_time": 90, "rank": 95, "popularity": 95, "last_3f": 90,
                  "horse_weight": 90, "track_condition": 95, "corner4": 80}
_health_ng = []
for _c, _thr in _CRITICAL_COLS.items():
    if _c in df.columns:
        _pct = df[_c].notna().mean() * 100
        _status = "OK" if _pct >= _thr else "!!!! 異常 !!!!"
        print(f"  {_c}: {_pct:.1f}% (閾値{_thr}%) {_status}")
        if _pct < _thr:
            _health_ng.append(_c)
if _health_ng:
    print(f"\n[警告] {len(_health_ng)} 列が閾値未満: {_health_ng}")
    print("  → このまま学習するとモデルが空データで学習されます（第28波事故の再来）")
else:
    print("  全列正常")

print("\n全テーブル保存完了。次回起動から瞬時に読み込まれます。")
