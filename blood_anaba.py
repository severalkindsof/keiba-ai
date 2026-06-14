# -*- coding: utf-8 -*-
"""
blood_anaba.py … 馬側の「穴血統」DB。父系(sire)・母父系(damsire)の両サイドで
大穴(10番人気以下)での複勝リフトを実績集計する。穴騎手DB(jockey_anaba)の馬版。

【鉄則】買い条件だけでなく消し条件・特異条件を必ず併記する([[feedback-list-conditions]])。
- 継続性: 年別lift(24/25/26)で「今年も過去も激走」する血統を判定し一発屋を分離。
- 条件別: 脚質/馬場/距離/状態/会場 ごとの買い◎(lift>=1.3)/消し✕(lift<=0.7)を網羅。
- 有名どころに限らないよう母数を MIN_RIDES=150 まで緩和。まぐれはWilson下限で抑制。

使い方:
  python -X utf8 blood_anaba.py build
  python -X utf8 blood_anaba.py fulllist sire      # 父系の継続穴血統フルリスト(買い/消し条件込み)
  python -X utf8 blood_anaba.py fulllist damsire    # 母父系
  python -X utf8 blood_anaba.py who ディープインパクト
"""
import sys
import numpy as np
import pandas as pd

DATA = "data/tfjv_all.parquet"
OUT_SUMMARY = "data/blood_anaba.parquet"
OUT_COND = "data/blood_anaba_cond.parquet"
OUT_META = "data/blood_anaba_meta.json"

ANA_POP = 10
MIN_RIDES = 150    # 血統別の最低・大穴騎乗数（有名どころ以外も拾うため緩和）
MIN_CELL = 40      # 条件別セルの最低サンプル
RECENT_YEAR = 24
DIMS = ("sire", "damsire")
CDIMS = ("style", "surface", "distance_cat", "track_condition", "venue")
CLABEL = {"style": "脚質", "surface": "馬場", "distance_cat": "距離",
          "track_condition": "状態", "venue": "会場"}


def _add_style(ana):
    for c in ("field_size", "corner4", "corner3", "corner2"):
        ana[c] = pd.to_numeric(ana[c], errors="coerce")
    pos = ana["corner4"].where(ana["corner4"] > 0)
    pos = pos.fillna(ana["corner3"].where(ana["corner3"] > 0))
    pos = pos.fillna(ana["corner2"].where(ana["corner2"] > 0))
    ratio = pos / ana["field_size"]
    ana["style"] = np.select([ratio <= 0.35, ratio >= 0.65], ["逃先", "差追"], default="中団")
    ana.loc[pos.isna() | (ana["field_size"] <= 0), "style"] = "?"
    return ana


def _load_ana():
    df = pd.read_parquet(DATA)
    for c in ("popularity", "rank", "year"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["popularity", "rank", "year"])
    df = df[~df["race_name"].astype(str).str.contains("障害", na=False)]
    ana = df[df["popularity"] >= ANA_POP].copy()
    ana["place"] = (ana["rank"] <= 3).astype(int)
    return _add_style(ana)


def _wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / d


def build():
    ana = _load_ana()
    base = ana["place"].mean()
    base_y = {y: ana[ana["year"] == y]["place"].mean() for y in (24, 25, 26)}

    sum_rows, cond_rows = [], []
    for dim in DIMS:
        g = ana.dropna(subset=[dim]).groupby(dim).agg(
            rides=("place", "size"), hit=("place", "sum"))
        g = g[g["rides"] >= MIN_RIDES]
        g["rate"] = g["hit"] / g["rides"]
        g["lift"] = g["rate"] / base
        g["lb_lift"] = [_wilson_lb(h, r) / base for h, r in zip(g["hit"], g["rides"])]
        # 年別lift・継続性
        for y in (24, 25, 26):
            gy = ana[ana["year"] == y].dropna(subset=[dim]).groupby(dim)["place"].agg(["mean", "size"])
            g[f"lift{y}"] = g.index.map(
                lambda v: (gy.loc[v, "mean"] / base_y[y]) if (v in gy.index and gy.loc[v, "size"] >= 25) else np.nan)
            g[f"n{y}"] = g.index.map(lambda v: int(gy.loc[v, "size"]) if v in gy.index else 0)
        g["cont"] = g[["lift24", "lift25", "lift26"]].mean(axis=1)
        g["stable"] = (g[["lift25", "lift26"]] >= 1.05).all(axis=1)
        g = g.reset_index().rename(columns={dim: "value"})
        g["dim"] = dim
        sum_rows.append(g)

        # 条件別（掲載血統のみ）
        keep = set(g["value"])
        sub = ana[ana[dim].isin(keep)]
        for cdim in CDIMS:
            cc = sub[sub[cdim] != "?"].dropna(subset=[dim, cdim]).groupby([dim, cdim]).agg(
                rides=("place", "size"), hit=("place", "sum")).reset_index()
            cc = cc[cc["rides"] >= MIN_CELL]
            cc["rate"] = cc["hit"] / cc["rides"]
            cc["lift"] = cc["rate"] / base
            cc["bloodtype"] = dim
            cc["cdim"] = cdim
            cc = cc.rename(columns={dim: "value", cdim: "cvalue"})
            cond_rows.append(cc[["bloodtype", "value", "cdim", "cvalue", "rides", "hit", "rate", "lift"]])

    summary = pd.concat(sum_rows, ignore_index=True)
    cond = pd.concat(cond_rows, ignore_index=True)
    summary.to_parquet(OUT_SUMMARY, index=False)
    cond.to_parquet(OUT_COND, index=False)
    import json
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({"base": float(base)}, f, ensure_ascii=False)
    print(f"[build] 大穴基準{base*100:.2f}%  父系{(summary['dim']=='sire').sum()}・"
          f"母父系{(summary['dim']=='damsire').sum()} 条件セル{len(cond)} -> {OUT_SUMMARY},{OUT_COND}")
    return summary, cond


def _ensure():
    import os
    if not (os.path.exists(OUT_SUMMARY) and os.path.exists(OUT_COND)):
        return build()
    return pd.read_parquet(OUT_SUMMARY), pd.read_parquet(OUT_COND)


def fulllist(which="sire", min_cont=1.10, min_n26=30):
    """継続穴血統フルリスト。年別で継続激走する血統を、◎買い/✕消し条件込みで網羅。"""
    summary, cond = _ensure()
    lab = "父系" if which == "sire" else "母父系"
    d = summary[summary["dim"] == which]
    cand = d[(d["cont"] >= min_cont) & (d["n26"] >= min_n26) & d["stable"]]
    cand = cand.sort_values("cont", ascending=False)
    print(f"=== {lab} 穴血統フルコンプリート（継続性>={min_cont} × 今年{min_n26}騎乗+ × 直近2年安定）===")
    print(f"{len(cand)}血統。lift=穴複勝リフト。◎買い=条件lift>=1.3 / ✕消し=<=0.7\n")
    for _, r in cand.iterrows():
        yrs = "/".join(f"{r[f'lift{y}']:.2f}" if pd.notna(r[f"lift{y}"]) else "－" for y in (24, 25, 26))
        print(f"■{r['value']} 総合lift{r['lift']:.2f}(下限{r['lb_lift']:.2f}) "
              f"年別24/25/26={yrs} 今年{int(r['n26'])}騎乗")
        c = cond[(cond["bloodtype"] == which) & (cond["value"] == r["value"])]
        buys, avoids = [], []
        for cdim in CDIMS:
            for _, cr in c[c["cdim"] == cdim].iterrows():
                tag = f"{CLABEL[cdim]}:{cr['cvalue']}({cr['lift']:.2f}x)"
                if cr["lift"] >= 1.3:
                    buys.append((cr["lift"], tag))
                elif cr["lift"] <= 0.7:
                    avoids.append((cr["lift"], tag))
        buys.sort(reverse=True)
        avoids.sort()
        print("   ◎買い: " + ("  ".join(b for _, b in buys[:8]) if buys else "—"))
        print("   ✕消し: " + ("  ".join(a for _, a in avoids[:6]) if avoids else "—"))
        print()


def rank(n=12):
    summary, _ = _ensure()
    for dim, lab in (("sire", "父系"), ("damsire", "母父系")):
        d = summary[summary["dim"] == dim]
        print(f"=== {lab} ◎買い血統（Wilson下限順・最低{MIN_RIDES}騎乗）===")
        for _, r in d.sort_values("lb_lift", ascending=False).head(n).iterrows():
            print(f"  {r['value']:18s} lift{r['lift']:.2f}x 下限{r['lb_lift']:.2f}x (n{int(r['rides'])})")
        print(f"--- {lab} ✕消し血統 ---")
        for _, r in d.sort_values("lb_lift").head(6).iterrows():
            print(f"  {r['value']:18s} lift{r['lift']:.2f}x 下限{r['lb_lift']:.2f}x (n{int(r['rides'])})")
        print()


def lookup(sire=None, damsire=None, surface=None, distance_cat=None, style=None, venue=None):
    """馬の父・母父から穴血統度を返す。条件(脚質/馬場/距離/会場)も加味可。
    戻り: dict(sire_lift, damsire_lift, lb, tags)"""
    summary, cond = _ensure()
    res = {"sire_lift": None, "damsire_lift": None, "lb": 1.0, "tags": []}
    lbs = []
    ctx = {"surface": surface, "distance_cat": distance_cat, "style": style, "venue": venue}
    for key, dim, lab in ((sire, "sire", "父"), (damsire, "damsire", "母父")):
        if not key:
            continue
        m = summary[(summary["dim"] == dim) & (summary["value"] == str(key).strip())]
        if m.empty:
            continue
        lf = float(m.iloc[0]["lb_lift"])
        res[f"{dim}_lift"] = lf
        lbs.append(lf)
        if lf >= 1.10:
            res["tags"].append(f"◎{lab}{key}{lf:.2f}x")
        elif lf <= 0.70:
            res["tags"].append(f"✕{lab}{key}{lf:.2f}x")
        # 条件別の特異/消し（顕著なものだけ）
        c = cond[(cond["bloodtype"] == dim) & (cond["value"] == str(key).strip())]
        for cdim, cval in ctx.items():
            if not cval:
                continue
            cm = c[(c["cdim"] == cdim) & (c["cvalue"] == cval)]
            if not cm.empty:
                clf = float(cm.iloc[0]["lift"])
                if clf >= 1.4 or clf <= 0.6:
                    mark = "◎" if clf >= 1.4 else "✕"
                    res["tags"].append(f"{mark}{lab}×{cval}{clf:.2f}x")
    if lbs:
        res["lb"] = max(lbs)
    return res


def who(name):
    summary, cond = _ensure()
    m = summary[summary["value"] == name.strip()]
    if m.empty:
        print(f"'{name}' は掲載対象外（{MIN_RIDES}騎乗未満）")
        return
    for _, r in m.iterrows():
        lab = "父系" if r["dim"] == "sire" else "母父系"
        yrs = "/".join(f"{r[f'lift{y}']:.2f}" if pd.notna(r[f"lift{y}"]) else "－" for y in (24, 25, 26))
        print(f"■{name}（{lab}）大穴{int(r['rides'])}騎乗 複勝{r['rate']*100:.1f}% "
              f"lift{r['lift']:.2f}x 下限{r['lb_lift']:.2f}x 年別{yrs}")
        c = cond[(cond["bloodtype"] == r["dim"]) & (cond["value"] == name.strip())]
        for cdim in CDIMS:
            cc = c[c["cdim"] == cdim].sort_values("lift", ascending=False)
            if cc.empty:
                continue
            cells = "  ".join(f"{cv}:{lf:.2f}x(n{int(n)})"
                              for cv, lf, n in zip(cc["cvalue"], cc["lift"], cc["rides"]))
            print(f"  [{CLABEL[cdim]}] {cells}")


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "rank"
    if cmd == "build":
        build()
    elif cmd == "rank":
        rank(int(args[1]) if len(args) > 1 else 12)
    elif cmd == "fulllist":
        fulllist(args[1] if len(args) > 1 else "sire")
    elif cmd == "who":
        who(args[1])
    else:
        print(__doc__)
