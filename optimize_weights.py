# -*- coding: utf-8 -*-
"""
optimize_weights.py … 複合スコアの要素重みを最適化する。
現状は単純積(各log-liftを等重み合計)。ロジスティック回帰で place(3着内) を各要素のlog-liftから
予測し、係数=最適重みを学習する(benter方式)。リーク回避: 学習期間でfit→テスト期間で評価。
過学習はL2正則化で抑制。等重み(単純積)と予測力・回収率を比較する。
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import anaba_composite as ac

CDIMS = ac.CDIMS  # style, jockey, sire, damsire, wbin, prank_cat, l3f_cat
MIN_CELL = 40

df = ac._prep(pd.read_parquet("data/tfjv_all.parquet"))
df["year"] = pd.to_numeric(df["year"], errors="coerce")
ana = df[df["popularity"] >= ac.ANA_POP].copy()
ana["place"] = (ana["rank"] <= 3).astype(int)
tr = ana[ana["year"] <= 24].copy()
te = ana[ana["year"] >= 25].copy()
base = tr["place"].mean()

# 学習期間でリフトテーブル
tables = {}
for c in CDIMS:
    g = tr.dropna(subset=[c]).groupby(c)["place"].agg(["mean", "size"])
    tables[c] = {str(k): g.loc[k, "mean"] / base for k in g.index if g.loc[k, "size"] >= MIN_CELL}
# Elo分位lift
tre = tr.dropna(subset=["elo_pct"])
_, bins = pd.qcut(tre["elo_pct"], 5, retbins=True, duplicates="drop")
elo_lift = []
for i in range(len(bins) - 1):
    seg = tre[(tre["elo_pct"] > (bins[i] if i > 0 else -1)) & (tre["elo_pct"] <= bins[i + 1])]
    elo_lift.append(seg["place"].mean() / base if len(seg) else 1.0)

FEATS = CDIMS + ["elo"]


def loglift_matrix(d):
    cols = {}
    for c in CDIMS:
        cols[c] = d[c].astype(str).map(tables[c]).fillna(1.0)
    eq = np.clip(np.searchsorted(bins, d["elo_pct"].values, side="right") - 1, 0, len(elo_lift) - 1)
    elo_col = pd.Series([elo_lift[q] for q in eq], index=d.index)
    elo_col[d["elo_pct"].isna()] = 1.0
    cols["elo"] = elo_col
    X = np.log(np.clip(np.column_stack([cols[f].values for f in FEATS]).astype(float), 0.05, None))
    return X


Xtr, ytr = loglift_matrix(tr), tr["place"].values
Xte = loglift_matrix(te)

clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(Xtr, ytr)

print("=== 学習した要素重み(ロジスティック回帰係数) ===")
for f, w in sorted(zip(FEATS, clf.coef_[0]), key=lambda x: -x[1]):
    print(f"  {f:12s}: {w:+.3f}")
print(f"  (等重み=単純積は全て1.0相当)")

te = te.copy()
te["w_score"] = clf.predict_proba(Xte)[:, 1]            # 最適重みスコア
te["eq_score"] = Xte.sum(axis=1)                         # 等重み(単純積=log和)
base_te = te["place"].mean()

# 人気別複勝率→複勝配当推定(控除20%)で回収率概算
pop_place = ana.groupby("popularity")["place"].mean()
def est_fuku(p):
    pr = pop_place.get(p, np.nan)
    return 0.8 / pr if pr and pr > 0 else 0.0

print(f"\n=== テスト期間 予測力比較 (base複勝{base_te*100:.2f}%) ===")
for name, col in [("単純積(等重み)", "eq_score"), ("最適重み(LogReg)", "w_score")]:
    for qlab, q in [("上位20%", 0.80), ("上位5%", 0.95)]:
        sub = te[te[col] >= te[col].quantile(q)]
        ret = sub.apply(lambda r: est_fuku(r["popularity"]) if r["place"] else 0.0, axis=1)
        print(f"  {name} {qlab}: 複勝率{sub['place'].mean()*100:.1f}% "
              f"lift{sub['place'].mean()/base_te:.2f}x 推定回収率{ret.mean()*100:.0f}% (n{len(sub)})")
