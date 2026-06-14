# -*- coding: utf-8 -*-
"""
anaba_composite.py … 穴馬の複合スコア（実戦用・LightGBM版）。
要素の交互作用(脚質×馬場・血統×馬場・枠×コース等)をGBDTで自動学習する。
全特徴を数値化(高カーディナリティ/カテゴリは学習期間の穴複勝liftで数値化)し、score時の再現を堅牢にする。

特徴: 数値[elo_pct, distance, horse_no, field_size, popularity, prev_rank, prev_l3f_pct]
    + lift[jockey, sire, damsire, surface, track_condition, venue, distance_cat, style, wbin]
鉄則: 直感係数なし。liftは実データの大穴(10番人気以下)複勝実測。過学習はnum_leaves/min_child/正則化で抑制。
予測力はbacktest_composite/optimize_gbdtでリークなし時系列分割により実証(テスト上位5% 3.83x)。

使い方:
  python -X utf8 anaba_composite.py build   # lift+GBDT学習・保存
  python -X utf8 anaba_composite.py check    # 判定別の複勝率(自己検証)
"""
import json
import sys
import numpy as np
import pandas as pd

ANA_POP = 10
MIN_CELL = 40
MODEL = "data/anaba_gbdt.txt"
META = "data/anaba_composite.json"
NUM = ["elo_pct", "distance", "horse_no", "field_size", "popularity", "prev_rank", "prev_l3f_pct"]
LIFT_DIMS = ["jockey", "sire", "damsire", "surface", "track_condition", "venue", "distance_cat", "style", "wbin"]
FEATS = NUM + [d + "_lift" for d in LIFT_DIMS]


def _prep(df):
    for c in ("popularity", "rank", "year", "month", "day", "race_no", "last_3f",
              "field_size", "corner4", "corner3", "corner2", "horse_weight", "distance", "horse_no"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["popularity", "rank", "horse_id"])
    df = df[~df["race_name"].astype(str).str.contains("障害", na=False)]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    yyyymmdd = ((2000 + df["year"].astype(int)).astype(str)
                + df["month"].astype(int).astype(str).str.zfill(2)
                + df["day"].astype(int).astype(str).str.zfill(2))
    df["rk"] = yyyymmdd + "_" + df["venue"].astype(str) + "_" + df["race_no"].astype(int).astype(str).str.zfill(2)
    df["hn"] = df["horse_name"].astype(str).str.strip()
    elo = pd.read_parquet("data/horse_elo_pit.parquet")[["race_key", "horse_name", "pre_elo"]]
    elo["hn"] = elo["horse_name"].astype(str).str.strip()
    df = df.merge(elo[["race_key", "hn", "pre_elo"]], left_on=["rk", "hn"], right_on=["race_key", "hn"], how="left")
    df["elo_pct"] = df.groupby("rk")["pre_elo"].rank(pct=True)
    pos = df["corner4"].where(df["corner4"] > 0)
    pos = pos.fillna(df["corner3"].where(df["corner3"] > 0)).fillna(df["corner2"].where(df["corner2"] > 0))
    r = pos / df["field_size"]
    df["style"] = np.select([r <= 0.35, r >= 0.65], ["逃先", "差追"], default="中団")
    df.loc[pos.isna() | (df["field_size"] <= 0), "style"] = "?"
    df["wbin"] = pd.cut(df["horse_weight"], [0, 430, 460, 490, 520, 9999],
                        labels=["~430", "430-460", "460-490", "490-520", "520+"]).astype("object")
    df = df.sort_values(["horse_id", "date"])
    df["prev_rank"] = df.groupby("horse_id")["rank"].shift()
    df["l3f_rank"] = df.groupby("rk")["last_3f"].rank()
    df["l3f_n"] = df.groupby("rk")["last_3f"].transform("count")
    df["l3f_pct"] = np.where(df["l3f_n"] >= 5, df["l3f_rank"] / df["l3f_n"], np.nan)
    df["prev_l3f_pct"] = df.groupby("horse_id")["l3f_pct"].shift()
    return df


def build():
    import lightgbm as lgb
    df = _prep(pd.read_parquet("data/tfjv_all.parquet"))
    ana = df[df["popularity"] >= ANA_POP].copy()
    ana["place"] = (ana["rank"] <= 3).astype(int)
    base = float(ana["place"].mean())
    # 全カテゴリ/高カーディナリティを穴複勝liftで数値化(全期間)
    lifts = {}
    for d in LIFT_DIMS:
        g = ana.dropna(subset=[d]).groupby(d)["place"].agg(["mean", "size"])
        lifts[d] = {str(k): g.loc[k, "mean"] / base for k in g.index if g.loc[k, "size"] >= MIN_CELL}
        ana[d + "_lift"] = ana[d].astype(str).map(lifts[d]).fillna(1.0)
    params = dict(objective="binary", num_leaves=15, min_child_samples=300, learning_rate=0.03,
                  n_estimators=400, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=1.0, verbose=-1)
    clf = lgb.LGBMClassifier(**params).fit(ana[FEATS], ana["place"])
    clf.booster_.save_model(MODEL)
    ana["score"] = clf.predict_proba(ana[FEATS])[:, 1]
    meta = {"base": base, "lifts": lifts, "feats": FEATS,
            "q_axis": float(ana["score"].quantile(0.80)),
            "q_keep": float(ana["score"].quantile(0.50)),
            "q_kill": float(ana["score"].quantile(0.20))}
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    imp = pd.Series(clf.feature_importances_, index=FEATS).sort_values(ascending=False)
    print(f"[build] base{base*100:.2f}% 重要度上位{list(imp.head(4).index)} -> {MODEL},{META}")
    return meta


_MODEL = None
_META = None


def _load():
    global _MODEL, _META
    if _META is None:
        import os
        if not (os.path.exists(MODEL) and os.path.exists(META)):
            build()
        import lightgbm as lgb
        _MODEL = lgb.Booster(model_file=MODEL)
        _META = json.load(open(META, encoding="utf-8"))
    return _MODEL, _META


def score(elo_pct=None, style=None, jockey=None, sire=None, damsire=None, wbin=None,
          surface=None, track_condition=None, venue=None, distance_cat=None, distance=None,
          horse_no=None, field_size=None, popularity=None, prev_rank=None, prev_l3f_pct=None):
    """出走馬の特徴から複合スコア(3着内確率)を返す。欠損はNaN/中立。GBDTが交互作用込みで判定。"""
    mdl, m = _load()
    lf = m["lifts"]
    raw = {"jockey": jockey, "sire": sire, "damsire": damsire, "surface": surface,
           "track_condition": track_condition, "venue": venue, "distance_cat": distance_cat,
           "style": style, "wbin": wbin}
    row = {"elo_pct": elo_pct, "distance": distance, "horse_no": horse_no, "field_size": field_size,
           "popularity": popularity, "prev_rank": prev_rank, "prev_l3f_pct": prev_l3f_pct}
    for d in LIFT_DIMS:
        v = raw[d]
        row[d + "_lift"] = lf[d].get(str(v), 1.0) if v is not None else 1.0
    X = pd.DataFrame([[row.get(f, np.nan) for f in FEATS]], columns=FEATS).astype(float)
    s = float(mdl.predict(X)[0])
    return {"score": s}


def classify(score_val):
    _, m = _load()
    if score_val >= m["q_axis"]:
        return "◎軸候補"
    if score_val >= m["q_keep"]:
        return "○押さえ"
    if score_val <= m["q_kill"]:
        return "✕消し"
    return "△中立"


_CTX = None


def horse_context(names):
    """出走馬名→最新 pre_elo・前走着順・前走上がり順位(pct)。race_briefへの特徴供給。"""
    global _CTX
    names = set(str(n).strip() for n in names)
    if _CTX is None:
        _CTX = _prep(pd.read_parquet("data/tfjv_all.parquet"))
    sub = _CTX[_CTX["hn"].isin(names)].sort_values("date").groupby("hn").tail(1)
    out = {}
    for _, r in sub.iterrows():
        out[r["hn"]] = {"pre_elo": (None if pd.isna(r["pre_elo"]) else float(r["pre_elo"])),
                        "prev_rank": (None if pd.isna(r["rank"]) else float(r["rank"])),
                        "prev_l3f_pct": (None if pd.isna(r["l3f_pct"]) else float(r["l3f_pct"]))}
    return out


def check():
    m = build()
    mdl, _ = _load()
    df = _prep(pd.read_parquet("data/tfjv_all.parquet"))
    ana = df[df["popularity"] >= ANA_POP].copy()
    ana["place"] = (ana["rank"] <= 3).astype(int)
    for d in LIFT_DIMS:
        ana[d + "_lift"] = ana[d].astype(str).map(m["lifts"][d]).fillna(1.0)
    ana["score"] = mdl.predict(ana[FEATS].astype(float))
    ana["cls"] = ana["score"].apply(classify)
    base = ana["place"].mean()
    print(f"=== 判定別 大穴複勝率(全期間・基準{base*100:.2f}%) ===")
    for c in ["◎軸候補", "○押さえ", "△中立", "✕消し"]:
        s = ana[ana["cls"] == c]
        if len(s):
            print(f"  {c}: 複勝{s['place'].mean()*100:.2f}% (lift{s['place'].mean()/base:.2f}x n{len(s)})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "check":
        check()
    else:
        print(__doc__)
