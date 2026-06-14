# -*- coding: utf-8 -*-
"""
verify_orthogonal.py … Eloと直交する軸(展開バイアスの揺り戻し)が、複合スコアに真の上積みを
もたらすかを検証する。鍵: 素のリフトでなく「Elo分位内での上積み」を見る(=Eloと独立か)。

軸:
  A. 前走上がり3F順位(脚を余した/展開不利の巻き返し)
  B. PCI逆張り(前走ハイペースで先行して凡走→展開好転で巻き返し)
レースのペースは、同(距離帯×馬場)でのレース平均上がり3Fと比較して推定(遅い=ハイペース)。
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/tfjv_all.parquet")
for c in ("popularity", "rank", "last_3f", "corner4", "field_size"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["popularity", "rank", "horse_id", "date", "race_id"])
df = df[~df["race_name"].astype(str).str.contains("障害", na=False)]
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Eloマージ(strip)
df["hn"] = df["horse_name"].astype(str).str.strip()
yyyymmdd = ((2000 + pd.to_numeric(df["year"]).astype(int)).astype(str)
            + pd.to_numeric(df["month"]).astype(int).astype(str).str.zfill(2)
            + pd.to_numeric(df["day"]).astype(int).astype(str).str.zfill(2))
df["rk"] = yyyymmdd + "_" + df["venue"].astype(str) + "_" + pd.to_numeric(df["race_no"]).astype(int).astype(str).str.zfill(2)
elo = pd.read_parquet("data/horse_elo_pit.parquet")[["race_key", "horse_name", "pre_elo"]]
elo["hn"] = elo["horse_name"].astype(str).str.strip()
df = df.merge(elo[["race_key", "hn", "pre_elo"]], left_on=["rk", "hn"], right_on=["race_key", "hn"], how="left")

# レース内: 上がり3F順位、4角位置、レース平均上がり
# last_3f有効頭数で正規化(古いデータの欠損対策)。last_3f欠損行は検証から外れる
# 注: tfjv_allのrace_idは行ごとにユニーク(レース識別に使えない)。自作rkでgroupbyする
df["l3f_rank"] = df.groupby("rk")["last_3f"].rank()                   # 1=最速
df["l3f_n"] = df.groupby("rk")["last_3f"].transform("count")
df["l3f_pct"] = np.where(df["l3f_n"] >= 5, df["l3f_rank"] / df["l3f_n"], np.nan)
df["corner_pct"] = df["corner4"] / df["field_size"]                   # 小=前
race_l3f = df.groupby("rk")["last_3f"].mean().rename("race_l3f")
df = df.merge(race_l3f, on="rk", how="left")
# 同(距離帯×馬場)でのレース平均上がりの基準 → 遅いほどハイペース
key_mean = df.groupby(["distance_cat", "surface"])["race_l3f"].transform("mean")
df["pace_hi"] = df["race_l3f"] - key_mean                             # +ほどハイペース寄り

# 前走情報
df = df.sort_values(["horse_id", "date"])
g = df.groupby("horse_id")
df["prev_l3f_pct"] = g["l3f_pct"].shift()
df["prev_corner_pct"] = g["corner_pct"].shift()
df["prev_pace_hi"] = g["pace_hi"].shift()
df["prev_rank"] = g["rank"].shift()

ana = df[df["popularity"] >= 10].copy()
ana["place"] = (ana["rank"] <= 3).astype(int)
base = ana["place"].mean()
print(f"大穴基準複勝率 {base*100:.2f}%  (n={len(ana)})\n")

# --- A. 前走上がり3F順位 ---
print("=== A. 前走上がり3F順位 → 今走大穴リフト ===")
a = ana.dropna(subset=["prev_l3f_pct"])
for lab, m in [("前走上がり最速級(上位20%)", a.prev_l3f_pct <= 0.2),
               ("上位20-40%", (a.prev_l3f_pct > 0.2) & (a.prev_l3f_pct <= 0.4)),
               ("中位40-70%", (a.prev_l3f_pct > 0.4) & (a.prev_l3f_pct <= 0.7)),
               ("下位70%+", a.prev_l3f_pct > 0.7)]:
    gg = a[m]
    print(f"  {lab}: lift{gg['place'].mean()/base:.2f}x n{len(gg)}")

# --- B. PCI逆張り: 前走ハイペース×先行×凡走 ---
print("\n=== B. PCI逆張り(前走ハイペース×先行×凡走) → 今走大穴リフト ===")
b = ana.dropna(subset=["prev_pace_hi", "prev_corner_pct", "prev_rank"])
flag = (b.prev_pace_hi > 0.3) & (b.prev_corner_pct <= 0.4) & (b.prev_rank >= 6)
print(f"  逆張りフラグ該当: lift{b[flag]['place'].mean()/base:.2f}x n{flag.sum()}")
print(f"  非該当: lift{b[~flag]['place'].mean()/base:.2f}x n{(~flag).sum()}")

# --- ★直交性: Elo分位内での前走上がり最速級の上積み ---
print("\n=== ★直交性チェック: Elo分位ごとの『前走上がり上位20%』の上積み ===")
ae = ana.dropna(subset=["pre_elo", "prev_l3f_pct"]).copy()
ae["elo_q"] = pd.qcut(ae["pre_elo"], 4, labels=["Elo下", "Elo中下", "Elo中上", "Elo上"])
for q in ["Elo下", "Elo中下", "Elo中上", "Elo上"]:
    s = ae[ae["elo_q"] == q]
    bq = s["place"].mean()
    fast = s[s["prev_l3f_pct"] <= 0.2]
    if len(fast) > 50 and bq > 0:
        print(f"  {q}(base{bq*100:.1f}%): 前走上がり上位 lift{fast['place'].mean()/bq:.2f}x (n{len(fast)})")
