# -*- coding: utf-8 -*-
"""
optimize_gbdt.py … 複合スコアを勾配ブースティング(LightGBM)に拡張し精度を最大化する。
線形のロジ回帰(上位5% 2.76x)に対し、GBDTは要素の交互作用(脚質×馬場・血統×馬場・枠×コース等)を
自動学習する。過学習(=現実離れ)はnum_leaves/min_child/正則化で抑制し、リークなし時系列分割で検証。
鉄則: 学習(year<=24)でlift・モデルを作り、テスト(year>=25)で評価。リーク厳禁。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import anaba_composite as ac

MIN_CELL = 40
LIFT_DIMS = ["jockey", "sire", "damsire"]            # 高カーディナリティは学習期間liftで数値化
CAT = ["surface", "track_condition", "venue", "distance_cat", "style", "wbin"]
NUM = ["elo_pct", "distance", "horse_no", "field_size", "popularity", "prev_rank", "prev_l3f_pct"]

df = ac._prep(pd.read_parquet("data/tfjv_all.parquet"))
for c in ("year", "distance", "horse_no", "field_size", "popularity"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["prev_rank"] = df.groupby("horse_id")["rank"].shift()

ana = df[df["popularity"] >= ac.ANA_POP].copy()
ana["place"] = (ana["rank"] <= 3).astype(int)
tr = ana[ana["year"] <= 24].copy()
te = ana[ana["year"] >= 25].copy()
base_tr = tr["place"].mean()

# 学習期間で騎手/父/母父の穴liftを作りtr/teにmap（リークなし）
for c in LIFT_DIMS:
    g = tr.dropna(subset=[c]).groupby(c)["place"].agg(["mean", "size"])
    lift = {k: g.loc[k, "mean"] / base_tr for k in g.index if g.loc[k, "size"] >= MIN_CELL}
    tr[f"{c}_lift"] = tr[c].astype(str).map(lift).fillna(1.0)
    te[f"{c}_lift"] = te[c].astype(str).map(lift).fillna(1.0)
LIFT = [f"{c}_lift" for c in LIFT_DIMS]

feats = NUM + LIFT + CAT
for d in (tr, te):
    for c in CAT:
        d[c] = d[c].astype("category")

Xtr, ytr = tr[feats], tr["place"]
Xte = te[feats]

params = dict(objective="binary", num_leaves=15, min_child_samples=300,
              learning_rate=0.03, n_estimators=400, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=1.0, verbose=-1)
# 学習期間内で早期終了用に末尾を検証分割
cut = tr["year"] <= 23
m = lgb.LGBMClassifier(**params)
m.fit(Xtr[cut], ytr[cut], eval_set=[(Xtr[~cut], ytr[~cut])],
      eval_metric="auc", callbacks=[lgb.early_stopping(40, verbose=False)], categorical_feature=CAT)

te = te.copy()
te["gbdt"] = m.predict_proba(Xte)[:, 1]
base_te = te["place"].mean()

pop_place = ana.groupby("popularity")["place"].mean()
def est_fuku(p):
    pr = pop_place.get(p, np.nan)
    return 0.8 / pr if pr and pr > 0 else 0.0

print(f"=== LightGBM(交互作用自動・正則化) テスト期間 base複勝{base_te*100:.2f}% ===")
for qlab, q in [("上位20%", 0.80), ("上位5%", 0.95), ("上位2%", 0.98)]:
    sub = te[te["gbdt"] >= te["gbdt"].quantile(q)]
    ret = sub.apply(lambda r: est_fuku(r["popularity"]) if r["place"] else 0.0, axis=1)
    print(f"  {qlab}: 複勝率{sub['place'].mean()*100:.1f}% lift{sub['place'].mean()/base_te:.2f}x "
          f"推定回収率{ret.mean()*100:.0f}% (n{len(sub)})")
print("  (比較: ロジ回帰 上位5% 2.76x/190%, 単純積 2.59x/172%)")

imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
print("\n=== 特徴重要度(上位10) ===")
for f, v in imp.head(10).items():
    print(f"  {f}: {int(v)}")
