# -*- coding: utf-8 -*-
"""
verify_new_factors.py … エージェントが外部リサーチで洗い出した新規ファクター候補を、
手元の実データ(tfjv_all)で大穴(10番人気以下)の複勝リフトとして実測する。
�querルは「鵜呑みにせず実測」(feedback-subagent-audit-rules)。基準=大穴複勝率。
前走系は horse_id × date でソートしてshiftで前走情報を取得。
"""
import numpy as np
import pandas as pd

df = pd.read_parquet("data/tfjv_all.parquet")
for c in ("popularity", "rank", "weight_carried", "age", "month"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["popularity", "rank", "horse_id", "date"])
df = df[~df["race_name"].astype(str).str.contains("障害", na=False)]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["horse_id", "date"])

# 前走情報（shift）
g = df.groupby("horse_id")
df["prev_jockey"] = g["jockey"].shift()
df["prev_wc"] = g["weight_carried"].shift()
df["prev_date"] = g["date"].shift()
df["days"] = (df["date"] - df["prev_date"]).dt.days
df["kg_change"] = df["weight_carried"] - df["prev_wc"]
df["change_jockey"] = (df["jockey"] != df["prev_jockey"]) & df["prev_jockey"].notna()

ana = df[df["popularity"] >= 10].copy()
ana["place"] = (ana["rank"] <= 3).astype(int)
base = ana["place"].mean()
print(f"大穴基準複勝率 {base*100:.2f}%  (n={len(ana)})\n")


def show(label, mask, sub=None):
    d = sub if sub is not None else ana
    g = d[mask]
    if len(g) < 100:
        print(f"  {label}: n={len(g)} (少標本)")
        return
    print(f"  {label}: {g['place'].mean()*100:.2f}% (lift{g['place'].mean()/base:.2f}x n{len(g)})")


print("=== ① 乗り替わり vs 継続騎乗（×穴）===")
show("乗り替わり", ana["change_jockey"] == True)
show("継続騎乗", ana["change_jockey"] == False)

print("=== ② 斤量 前走比（×穴）===")
kc = ana.dropna(subset=["kg_change"])
for lab, m in [("減2kg超(<-2)", kc.kg_change < -2), ("微減(-2~-0.5)", (kc.kg_change >= -2) & (kc.kg_change < -0.5)),
               ("同(-0.5~0.5)", (kc.kg_change >= -0.5) & (kc.kg_change <= 0.5)),
               ("微増(0.5~2)", (kc.kg_change > 0.5) & (kc.kg_change <= 2)), ("増2kg超(>2)", kc.kg_change > 2)]:
    show(lab, m, kc)

print("=== ③ レース間隔 × 芝/ダート（×穴）===")
dd = ana.dropna(subset=["days"])
for surf in ["芝", "ダート"]:
    s = dd[dd["surface"] == surf]
    print(f" [{surf}]")
    for lab, m in [("連闘(~7)", s.days <= 7), ("中1-3週(8-27)", (s.days >= 8) & (s.days < 28)),
                   ("中4-7週(28-55)", (s.days >= 28) & (s.days < 56)), ("8-10週(56-76)", (s.days >= 56) & (s.days < 77)),
                   ("休明11週+(77+)", s.days >= 77)]:
        show(lab, m, s)

print("=== ④ 季節 × 性別（×穴）===")
for sx in ["牝", "牡", "セ"]:
    s = ana[ana["sex"] == sx]
    if len(s) < 500:
        continue
    summer = s[s["month"].isin([6, 7, 8])]
    winter = s[s["month"].isin([12, 1, 2])]
    print(f"  {sx}: 夏(6-8月)lift{summer['place'].mean()/base:.2f}x(n{len(summer)}) / "
          f"冬(12-2月)lift{winter['place'].mean()/base:.2f}x(n{len(winter)})")

print("=== ⑤ 前走着順 → 巻き返し（×穴）===")
df["prev_rank"] = g["rank"].shift()
ana2 = df[df["popularity"] >= 10].copy(); ana2["place"] = (ana2["rank"] <= 3).astype(int)
pr = ana2.dropna(subset=["prev_rank"])
for lab, m in [("前走1-3着", pr.prev_rank <= 3), ("前走4-5着", (pr.prev_rank > 3) & (pr.prev_rank <= 5)),
               ("前走6-9着", (pr.prev_rank > 5) & (pr.prev_rank <= 9)), ("前走10着以下(大敗)", pr.prev_rank >= 10)]:
    g2 = pr[m]
    print(f"  {lab}: lift{g2['place'].mean()/base:.2f}x n{len(g2)}")
