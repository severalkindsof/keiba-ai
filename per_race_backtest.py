"""1レース精査バックテスト（穴決着レースで「勝った穴馬を拾えたか」を検証）

各穴決着レース（勝ち馬7人気以下）について、勝った穴馬の【レース前】シグナルを
日付カットオフで立て、どのレンズが発火したか／拾えなかったかを判定する。
レンズ別の捕捉率と、未捕捉レースの理由内訳を出す。
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

_DIR = Path(__file__).parent


def _load():
    df = pd.read_parquet(_DIR / "data/tfjv_all.parquet",
                         columns=["race_id", "date", "rank", "popularity", "horse_name",
                                  "venue", "surface", "distance", "race_name",
                                  "last_3f", "weight_carried"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rkey"] = df["race_id"].astype(str).str[:8]
    df["rk"] = pd.to_numeric(df["rank"], errors="coerce")
    df["pop"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["hn"] = df["horse_name"].astype(str).str.strip()
    e = pd.read_parquet(_DIR / "data/horse_elo.parquet")
    emap = dict(zip(e["horse_name"].astype(str).str.strip(), e["elo"]))
    return df, emap


def _class_level(name: str) -> float | None:
    s = str(name)
    for k, v in [("G1", 7), ("G2", 6), ("G3", 5), ("(L)", 4.5), ("オープ", 4),
                 ("3勝", 3), ("３勝", 3), ("1600万", 3), ("2勝", 2), ("２勝", 2),
                 ("1000万", 2), ("1勝", 1), ("１勝", 1), ("500万", 1),
                 ("未勝利", 0), ("新馬", 0)]:
        if k in s:
            return float(v)
    return None


def lenses_for_horse(name, race_date, venue, surface, distance, cur_class,
                     pop, df, emap, field_elos):
    """1頭のレース前シグナル（発火レンズの集合）を返す。"""
    past = df[(df["hn"] == name) & (df["date"] < race_date)].sort_values("date", ascending=False)
    fired = []
    if past.empty:
        return fired, "過去走なし(新馬・地方上がり等)"

    # L1 市場見限りエリート（自Elo高 × 人気薄） ※Eloリーク注意
    elo = emap.get(name)
    if elo is not None and pop >= 7 and elo >= 2400:
        # レース内で見限られ群の上位か
        neg = [(h, e) for h, e in field_elos.items()
               if e is not None and e >= 2400]
        neg.sort(key=lambda x: -x[1])
        rank = next((i for i, (h, _) in enumerate(neg, 1) if h == name), 99)
        if rank <= 2:
            fired.append("市場見限りエリート")

    # L2 同コース得意（同venue×surface×distance 2走+ 連対率50%+）
    sc = past[(past["venue"] == venue) & (past["surface"] == surface) & (past["distance"] == distance)]
    sc_r = pd.to_numeric(sc["rank"], errors="coerce").dropna()
    if len(sc_r) >= 2 and (sc_r <= 3).mean() >= 0.5:
        fired.append("同コース得意")

    # L3 近走復調（直近5走に1-3着が1回以上、かつ前走で大敗していない）
    r5 = pd.to_numeric(past.head(5)["rank"], errors="coerce").dropna()
    if len(r5) >= 1 and (r5 <= 3).sum() >= 1:
        fired.append("近走好走歴")

    # L4 格上挑戦帰り（前走G級→今回条件戦）
    prev_cls = _class_level(past.iloc[0]["race_name"])
    if cur_class is not None and prev_cls is not None and prev_cls >= 5 and cur_class <= 3:
        fired.append("格上挑戦帰り")

    # L5 同距離実績（同distanceで複勝率40%+、3走+）
    sd = past[past["distance"] == distance]
    sd_r = pd.to_numeric(sd["rank"], errors="coerce").dropna()
    if len(sd_r) >= 3 and (sd_r <= 3).mean() >= 0.4:
        fired.append("距離得意")

    # L6 上がり最速級経験（直近4走で同レース上がり1-2位）
    rkeys = past.head(4)["rkey"].tolist()
    if rkeys:
        sub = df[df["rkey"].isin(rkeys)][["rkey", "hn", "last_3f"]].copy()
        sub["l3f"] = pd.to_numeric(sub["last_3f"], errors="coerce")
        sub["lr"] = sub.groupby("rkey")["l3f"].rank(method="min", na_option="bottom")
        mine = sub[sub["hn"] == name]
        if (mine["lr"] <= 2).sum() >= 1:
            fired.append("決め手")

    return fired, "" if fired else "全レンズ不発(真の伏兵)"


def run(since="2024-01-01", min_field=12, max_races=None):
    df, emap = _load()
    d = df[df["date"] >= since]
    fs = d.groupby("rkey").size()
    d = d[d["rkey"].isin(fs[fs >= min_field].index)]
    wins = d[d["rk"] == 1]
    upset = wins[wins["pop"] >= 7].copy().sort_values("date")
    if max_races:
        upset = upset.tail(max_races)

    lens_names = ["市場見限りエリート", "同コース得意", "近走好走歴",
                  "格上挑戦帰り", "距離得意", "決め手"]
    catch = {ln: 0 for ln in lens_names}
    caught_any = 0
    miss_reasons = {}
    total = 0
    examples_caught, examples_missed = [], []

    for _, w in upset.iterrows():
        rkey = w["rkey"]; rdate = w["date"]
        venue, surf, dist = w["venue"], w["surface"], int(w["distance"])
        cur_class = _class_level(w["race_name"])
        field = d[d["rkey"] == rkey]
        field_elos = {h: emap.get(h) for h in field["hn"]}
        fired, reason = lenses_for_horse(w["hn"], rdate, venue, surf, dist,
                                         cur_class, int(w["pop"]), df, emap, field_elos)
        total += 1
        for ln in fired:
            catch[ln] += 1
        if fired:
            caught_any += 1
            if len(examples_caught) < 6:
                examples_caught.append((str(rdate)[:10], venue, w["hn"], int(w["pop"]), fired))
        else:
            miss_reasons[reason] = miss_reasons.get(reason, 0) + 1
            if len(examples_missed) < 6:
                examples_missed.append((str(rdate)[:10], venue, w["hn"], int(w["pop"]), reason))

    print(f"=== 穴決着({since}-, {min_field}頭+, 勝ち馬7人気以下) {total}レース 精査 ===")
    print(f"いずれかのレンズで捕捉: {caught_any}/{total} ({caught_any/total*100:.1f}%)")
    print("\nレンズ別 捕捉率（勝った穴馬を事前にフラグできた率）:")
    for ln in lens_names:
        print(f"  {ln:14s}: {catch[ln]:4d}/{total} ({catch[ln]/total*100:.1f}%)")
    print("\n未捕捉の理由内訳:")
    for r, c in sorted(miss_reasons.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")
    print("\n--- 捕捉できた穴の例 ---")
    for dt, v, h, p, f in examples_caught:
        print(f"  {dt} {v} {h}({p}人気) ← {' / '.join(f)}")
    print("\n--- 拾えなかった穴の例 ---")
    for dt, v, h, p, r in examples_missed:
        print(f"  {dt} {v} {h}({p}人気) ← {r}")


if __name__ == "__main__":
    run()
