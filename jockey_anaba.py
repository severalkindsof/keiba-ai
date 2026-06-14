# -*- coding: utf-8 -*-
"""
jockey_anaba.py … 「穴をよく持ってくる騎手」を実績ベースで抽出するレンズ。

設計の要点（検証で確定した交絡を反映）:
- 大穴 = 10番人気以下（基準複勝率 約5%）。これを3着内に持ってくる率を騎手別に集計。
- 障害レースは除外（race_nameに「障害」を含む行。芝/ダート扱いで混入し長距離リフトを汚染するため）。
- 現役のみ（直近2年=2024-2026に騎乗あり）。引退・期間限定騎手の小標本まぐれを排除。
- ランキングは複勝率そのものではなく Wilson下限（95%CIの下側）でソート。母数の少ない騎手の過大評価を防ぐ。
- 「どの条件で穴を持ってくるか」は surface(芝/ダート) × distance_cat × track_condition のセル別リフトで保持。

生成物: data/jockey_anaba.parquet（騎手別サマリ）, data/jockey_anaba_cond.parquet（条件別リフト）

使い方:
  python -X utf8 jockey_anaba.py build           # プロファイル再生成
  python -X utf8 jockey_anaba.py rank [N]         # 現役穴職人ランキング上位N（既定20）
  python -X utf8 jockey_anaba.py who 西村淳也       # 指定騎手の条件別発火プロファイル
  python -X utf8 jockey_anaba.py race 出馬CSV       # 出走騎手×当該条件で穴フラグ付与（任意）
"""
import re
import sys
import numpy as np
import pandas as pd

DATA = "data/tfjv_all.parquet"
OUT_SUMMARY = "data/jockey_anaba.parquet"
OUT_COND = "data/jockey_anaba_cond.parquet"
OUT_META = "data/jockey_anaba_meta.json"


def _dist_cat(d):
    d = int(d)
    if d <= 1400:
        return "短距離"
    if d <= 1800:
        return "マイル"
    if d <= 2200:
        return "中距離"
    return "長距離"

ANA_POP = 10        # 大穴の定義: この人気以下
MIN_RIDES = 100     # サマリ掲載の最低・大穴騎乗数（埋もれた中堅も拾うため緩和）
MIN_CELL = 40       # 条件別セルの最低サンプル
RECENT_YEAR = 24    # 現役判定: この年(2桁)以降に騎乗あり

# データに最新年の騎乗が残っていても実際は引退済みの騎手。現役判定から手動除外。
# 注: 和田竜二はデータ上2026年騎乗記録があるがユーザー申告で引退扱い（要再確認）。
RETIRED = {"勝浦正樹", "和田竜二"}

# 別人として扱う注意名（4文字truncで紛らわしいが別騎手）:
#   ルメール = C.ルメール(日本通年) / ルメート = A.ルメートル(短期免許) → 統合しない


def _is_foreign(name):
    """漢字を含まない=外国人(短期免許含む)。検証で大穴複勝率1.52xと最も高いカテゴリ。"""
    return not bool(re.search(r"[一-龥]", str(name)))


def _load():
    df = pd.read_parquet(DATA)
    for c in ("popularity", "rank", "year"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["popularity", "rank", "jockey", "year"])
    # 障害レース除外
    df = df[~df["race_name"].astype(str).str.contains("障害", na=False)]
    return df


def _add_style(ana):
    """最終コーナー通過順/頭数から脚質を逃先/中団/差追に分類して 'style' 列を付与。
    検証: 穴は脚質に強く依存。逃先1.53x / 中団0.83x / 差追0.48x。"""
    for c in ("field_size", "corner4", "corner3", "corner2"):
        ana[c] = pd.to_numeric(ana[c], errors="coerce")
    pos = ana["corner4"].where(ana["corner4"] > 0)
    pos = pos.fillna(ana["corner3"].where(ana["corner3"] > 0))
    pos = pos.fillna(ana["corner2"].where(ana["corner2"] > 0))
    ratio = pos / ana["field_size"]
    ana["style"] = np.select(
        [ratio <= 0.35, ratio >= 0.65], ["逃先", "差追"], default="中団")
    ana.loc[pos.isna() | (ana["field_size"] <= 0), "style"] = "?"
    return ana


def _wilson_lb(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / d


def build():
    df = _load()
    ana = df[df["popularity"] >= ANA_POP].copy()
    ana["place"] = (ana["rank"] <= 3).astype(int)
    ana = _add_style(ana)
    base = ana["place"].mean()
    recent = set(df[df["year"] >= RECENT_YEAR]["jockey"].unique()) - RETIRED

    # --- 騎手サマリ ---
    g = ana.groupby("jockey").agg(rides=("place", "size"), hit=("place", "sum"))
    g = g[g["rides"] >= MIN_RIDES]
    g = g[g.index.isin(recent)]
    g["rate"] = g["hit"] / g["rides"]
    g["lift"] = g["rate"] / base
    g["wilson_lb"] = [_wilson_lb(h, r) for h, r in zip(g["hit"], g["rides"])]
    g["lb_lift"] = g["wilson_lb"] / base
    g["base"] = base
    g["foreign"] = g.index.map(_is_foreign)
    # --- 継続性: 年別lift(24/25/26)。一発屋(今年だけ高い)と本物(3年安定)を分離 ---
    for y in (24, 25, 26):
        by = ana[ana["year"] == y]["place"].mean()
        gy = ana[ana["year"] == y].groupby("jockey")["place"].agg(["mean", "size"])
        g[f"lift{y}"] = g.index.map(
            lambda j: (gy.loc[j, "mean"] / by) if (j in gy.index and gy.loc[j, "size"] >= 30) else np.nan)
        g[f"n{y}"] = g.index.map(lambda j: int(gy.loc[j, "size"]) if j in gy.index else 0)
    # 継続スコア=3年liftの平均(NaN無視)、安定フラグ=直近2年とも1.1以上
    g["cont"] = g[["lift24", "lift25", "lift26"]].mean(axis=1)
    g["stable"] = (g[["lift25", "lift26"]] >= 1.1).all(axis=1)
    g = g.sort_values("wilson_lb", ascending=False).reset_index()
    g.to_parquet(OUT_SUMMARY, index=False)

    # 外国人(短期免許含む)カテゴリの事前分布。個別母数が薄い新規騎手の素点に使う。
    fa = ana[ana["jockey"].map(_is_foreign)]
    f_rate = fa["place"].mean()
    import json
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({"base": float(base), "foreign_prior": float(f_rate),
                   "foreign_lift": float(f_rate / base)}, f, ensure_ascii=False)
    print(f"[build] 外国人カテゴリ事前: 大穴{len(fa)}騎乗 複勝{f_rate*100:.2f}% "
          f"(lift {f_rate/base:.2f}x) ← 短期免許新規はこの素点で評価可")

    # --- 条件別リフト（サマリ掲載騎手のみ） ---
    keep = set(g["jockey"])
    sub = ana[ana["jockey"].isin(keep)]
    rows = []
    for dim in ("surface", "distance_cat", "track_condition", "style", "venue"):
        cc = sub[sub[dim] != "?"].groupby(["jockey", dim]).agg(
            rides=("place", "size"), hit=("place", "sum")
        ).reset_index()
        cc = cc[cc["rides"] >= MIN_CELL]
        cc["rate"] = cc["hit"] / cc["rides"]
        cc["lift"] = cc["rate"] / base
        cc["dim"] = dim
        cc = cc.rename(columns={dim: "value"})
        rows.append(cc[["jockey", "dim", "value", "rides", "hit", "rate", "lift"]])
    cond = pd.concat(rows, ignore_index=True)
    cond.to_parquet(OUT_COND, index=False)
    print(f"[build] 障害除外後 大穴基準複勝率={base*100:.2f}%  "
          f"掲載騎手={len(g)}人  条件セル={len(cond)}  -> {OUT_SUMMARY}, {OUT_COND}")
    return g, cond


def _ensure():
    import os
    if not (os.path.exists(OUT_SUMMARY) and os.path.exists(OUT_COND)):
        return build()
    return pd.read_parquet(OUT_SUMMARY), pd.read_parquet(OUT_COND)


def rank(n=20):
    g, _ = _ensure()
    base = g["base"].iloc[0]
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"=== 中央現役・穴職人ランキング（障害除外/最低{MIN_RIDES}騎乗/"
          f"大穴=10人気↓ 基準{base*100:.2f}% / Wilson下限ソート）===")
    view = g.head(n)[["jockey", "rides", "hit", "rate", "lift", "lb_lift"]].copy()
    view["rate"] = (view["rate"] * 100).round(1)
    view[["lift", "lb_lift"]] = view[["lift", "lb_lift"]].round(2)
    print(view.to_string(index=False))


def who(name):
    g, cond = _ensure()
    row = g[g["jockey"] == name]
    if row.empty:
        cand = [j for j in g["jockey"] if name in j]
        print(f"'{name}' は掲載対象外（母数不足/非現役）。近い候補: {cand}")
        return
    r = row.iloc[0]
    print(f"■{name}  大穴{int(r['rides'])}騎乗  複勝{r['rate']*100:.1f}%  "
          f"lift={r['lift']:.2f}x  Wilson下限lift={r['lb_lift']:.2f}x")
    c = cond[cond["jockey"] == name]
    labels = {"style": "脚質", "surface": "馬場", "distance_cat": "距離",
              "track_condition": "状態", "venue": "会場"}
    for dim, lab in labels.items():
        d = c[c["dim"] == dim].sort_values("lift", ascending=False)
        if d.empty:
            continue
        cells = "  ".join(
            f"{v}:{rt*100:.0f}%({lf:.2f}x/n{int(n)})"
            for v, n, rt, lf in zip(d["value"], d["rides"], d["rate"], d["lift"])
        )
        print(f"  [{lab}] {cells}")


def fulllist(min_cont=1.15, min_n26=40):
    """穴騎手フルコンプリートリスト。今年も過去も継続して穴を持つ騎手を、
    買い条件(lift>=1.3)/消し条件(lift<=0.7)込みで網羅出力。"""
    g, cond = _ensure()
    base = g["base"].iloc[0]
    DIM = {"style": "脚質", "surface": "馬場", "distance_cat": "距離",
           "track_condition": "状態", "venue": "会場"}
    # 継続穴騎手: 継続スコア>=min_cont かつ 今年十分騎乗 かつ 直近2年安定 or 外国人
    cand = g[((g["cont"] >= min_cont) & (g["n26"] >= min_n26) & g["stable"]) |
             (g["foreign"] & (g["n26"] >= min_n26))]
    cand = cand.sort_values("cont", ascending=False)
    print(f"=== 穴騎手フルコンプリートリスト（大穴基準{base*100:.2f}% / 継続性>={min_cont} ×"
          f" 今年{min_n26}騎乗+ × 直近2年安定）===")
    print(f"{len(cand)}名。lift=穴での複勝リフト。買い◎=条件lift>=1.3 / 消し✕=<=0.7\n")
    for _, r in cand.iterrows():
        yrs = "/".join(f"{r[f'lift{y}']:.2f}" if pd.notna(r[f"lift{y}"]) else "－" for y in (24, 25, 26))
        foreign = "🌍短期" if r["foreign"] else ""
        print(f"■{r['jockey']} {foreign} 総合lift{r['lift']:.2f}(下限{r['lb_lift']:.2f}) "
              f"年別24/25/26={yrs} 今年{int(r['n26'])}騎乗")
        c = cond[cond["jockey"] == r["jockey"]]
        buys, avoids = [], []
        for dim, lab in DIM.items():
            for _, cr in c[c["dim"] == dim].iterrows():
                if cr["rides"] < MIN_CELL:
                    continue
                if cr["lift"] >= 1.3:
                    buys.append((cr["lift"], f"{lab}:{cr['value']}({cr['lift']:.2f}x)"))
                elif cr["lift"] <= 0.7:
                    avoids.append((cr["lift"], f"{lab}:{cr['value']}({cr['lift']:.2f}x)"))
        buys.sort(reverse=True)
        avoids.sort()
        print("   ◎買い: " + ("  ".join(b for _, b in buys[:8]) if buys else "—"))
        print("   ✕消し: " + ("  ".join(a for _, a in avoids[:6]) if avoids else "—"))
        print()


def lookup(jockey, surface=None, distance_cat=None, style=None, venue=None):
    """騎手の穴度を返す。掲載外の外国人は外国人カテゴリ事前(約1.52x)で代替。
    style(逃先/中団/差追)・venue も条件として加味可。穴は逃先脚質で激走(1.53x)。
    戻り: dict(jockey, lift, lb_lift, src, cond_lift, style_lift) — src= jockey / foreign_prior / none"""
    import json, os
    g, cond = _ensure()
    meta = {}
    if os.path.exists(OUT_META):
        meta = json.load(open(OUT_META, encoding="utf-8"))
    row = g[g["jockey"] == jockey]
    if not row.empty:
        r = row.iloc[0]
        c = cond[cond["jockey"] == jockey]
        cells = []
        style_lift = None
        for dim, val in (("surface", surface), ("distance_cat", distance_cat),
                         ("style", style), ("venue", venue)):
            if val is not None:
                m = c[(c["dim"] == dim) & (c["value"] == val)]
                if not m.empty:
                    lf = float(m.iloc[0]["lift"])
                    cells.append((val, lf))
                    if dim == "style":
                        style_lift = lf
        return {"jockey": jockey, "lift": float(r["lift"]),
                "lb_lift": float(r["lb_lift"]), "src": "jockey",
                "cond_lift": cells, "style_lift": style_lift}
    if _is_foreign(jockey) and meta:
        return {"jockey": jockey, "lift": meta["foreign_lift"],
                "lb_lift": meta["foreign_lift"], "src": "foreign_prior",
                "cond_lift": [], "style_lift": None}
    return {"jockey": jockey, "lift": 1.0, "lb_lift": 1.0, "src": "none",
            "cond_lift": [], "style_lift": None}


def race(csv_path=None):
    """当日出馬CSVを読み、各レースの穴騎手フラグ＋穴馬(10人気↓)×穴騎手の同時成立を抽出。"""
    from tfjv_entries import load_tfjv_entries
    if csv_path is None:
        import glob
        cands = sorted(glob.glob("C:/TFJV/TXT/出馬表分析*.CSV"))
        if not cands:
            print("出馬表分析CSVが見つかりません。パスを指定してください。")
            return
        csv_path = cands[-1]
    data = load_tfjv_entries(csv_path)
    print(f"[race] {csv_path}  {len(data)}レース")
    for rid, info in data.items():
        surface = info["surface"]
        dcat = _dist_cat(info["distance"])
        hits = []
        for e in info["entries"]:
            jk = e.get("jockey", "")
            res = lookup(jk, surface, dcat)
            pop = e.get("popularity")
            mark = ""
            # 注目: 穴騎手(lb_lift>=1.10) or 外国人事前、特に穴馬(10人気↓)に騎乗
            strong = res["lb_lift"] >= 1.10 or res["src"] == "foreign_prior"
            if strong:
                cond = " ".join(f"{v}{lf:.2f}x" for v, lf in res["cond_lift"])
                tag = "外国人事前" if res["src"] == "foreign_prior" else f"lb{res['lb_lift']:.2f}x"
                pm = f"{pop}人気" if pop else "人気?"
                ana = "★穴馬" if (pop and pop >= ANA_POP) else ""
                hits.append(f"    {e['horse_no']:>2}番 {e['horse_name']}({pm}) {jk} [{tag} {cond}]{ana}")
        if hits:
            print(f"\n■{info['venue']}{info['race_no']}R {info.get('race_class','')} {surface}{info['distance']} [{dcat}]")
            print("\n".join(hits))


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "rank"
    if cmd == "build":
        build()
    elif cmd == "rank":
        rank(int(args[1]) if len(args) > 1 else 20)
    elif cmd == "who":
        who(args[1])
    elif cmd == "fulllist":
        fulllist()
    elif cmd == "race":
        race(args[1] if len(args) > 1 else None)
    else:
        print(__doc__)
