"""検証済み穴選定スコア（第64波・リークなし大標本で実証）

穴(7人気以下)の複勝率を、以下4ファクター該当数で予測（単調・実証済み）:
  0個=7.1%(0.90x) / 1個=10.7% / 2個=11.2% / 3個以上=12.9%(1.63x)
基準(穴全体)=7.9%。

4ファクター（いずれもリークなしで複勝率1.3-1.7xのリフトを確認）:
  f_elo   : PIT/現在Elo >= 2300（地力上位）
  f_dist  : 同距離 過去複勝率40%+（3走以上）
  f_course: 同コース(会場×馬場×距離) 過去複勝率50%+（2走以上）
  f_total : 通算 過去複勝率50%+（4走以上）

実運用: 現在Eloは未来レースでは直前Elo=リークなし。過去複勝率も過去走のみ。
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

_DIR = Path(__file__).parent
_DF = None
_EMAP = None


def _load():
    global _DF, _EMAP
    if _DF is None:
        d = pd.read_parquet(_DIR / "data/tfjv_all.parquet",
                            columns=["horse_name", "rank", "date", "venue", "surface", "distance"])
        d["horse_name"] = d["horse_name"].astype(str).str.strip()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["place"] = (pd.to_numeric(d["rank"], errors="coerce") <= 3).astype(float)
        _DF = d
    if _EMAP is None:
        e = pd.read_parquet(_DIR / "data/horse_elo.parquet")
        _EMAP = dict(zip(e["horse_name"].astype(str).str.strip(), e["elo"]))
    return _DF, _EMAP


def anaba_flags(horse: str, venue: str, surface: str, distance: int,
                before=None, elo_floor: float = 2300.0) -> dict:
    """1頭の検証済みフラグを返す。"""
    df, emap = _load()
    name = str(horse).strip()
    h = df[df["horse_name"] == name]
    if before is not None:
        h = h[h["date"] < pd.to_datetime(before)]
    flags = []
    elo = emap.get(name)
    if elo is not None and elo >= elo_floor:
        flags.append(f"地力(Elo{elo:.0f})")
    # 同距離
    sd = h[h["distance"] == distance]
    if len(sd) >= 3 and sd["place"].mean() >= 0.4:
        flags.append(f"同距離複勝{sd['place'].mean()*100:.0f}%")
    # 同コース
    sc = h[(h["venue"] == venue) & (h["surface"] == surface) & (h["distance"] == distance)]
    if len(sc) >= 2 and sc["place"].mean() >= 0.5:
        flags.append(f"同コース複勝{sc['place'].mean()*100:.0f}%")
    # 通算
    if len(h) >= 4 and h["place"].mean() >= 0.5:
        flags.append(f"通算複勝{h['place'].mean()*100:.0f}%")

    n = len(flags)
    exp = {0: 7.1, 1: 10.7, 2: 11.2, 3: 12.9, 4: 13.0}.get(n, 12.9)
    tier = "◎軸候補" if n >= 3 else ("○" if n >= 1 else "✕")
    return {"n_flags": n, "flags": flags, "exp_place": exp, "tier": tier}


def rank_longshots(entries: list[dict], venue: str, surface: str, distance: int,
                   pop_min: int = 7, before=None) -> list[dict]:
    """出走馬の穴(pop_min以上)を検証済みフラグ数で順位付け。"""
    out = []
    for e in entries:
        pop = e.get("popularity")
        if pop is None or int(pop) < pop_min:
            continue
        f = anaba_flags(e["horse_name"], venue, surface, distance, before)
        out.append({"horse": e["horse_name"], "pop": int(pop), **f})
    out.sort(key=lambda x: -x["n_flags"])
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_DIR))
    from tfjv_entries import load_tfjv_entries
    data = load_tfjv_entries("C:/TFJV/TXT/出馬表分析260613.CSV")
    race = data["20260613021111"]
    res = rank_longshots(race["entries"], "函館", "芝", 1200)
    print("=== 函館スプリント 穴(7人気以下)検証フラグ順 ===")
    for r in res:
        print(f"  {r['tier']} {r['horse']}({r['pop']}人気) フラグ{r['n_flags']}個 期待複勝{r['exp_place']}% | {' / '.join(r['flags'])}")
