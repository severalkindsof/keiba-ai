"""レース選定（判断材料リスト・予測スコアではない）

第62波の検証で「妙味穴の有無で狙い目スコアを作る」設計は予測力ゼロと判明
（エリート在レースの人気BOX回収率61.5% < 不在66.6%）。
→ 予測スコアは撤去。各レースの【事実】だけを並べ、判断は対話で行う。

並べる事実:
  - 妙味穴(市場見限りエリート)の馬名（いれば）
  - 鉄板軸候補・危険な1人気（honmei判定）
  - 頭数
  - 重賞なら過去の荒れ度（1番人気勝率の実績）
"""
from __future__ import annotations
import sys
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))
import pandas as pd

_HIST = None


def _grade_upset_rate(race_name: str):
    """重賞なら過去開催の1番人気勝率を返す（荒れ度の実績）。"""
    global _HIST
    if _HIST is None:
        d = pd.read_parquet(_DIR / "data/tfjv_all.parquet",
                            columns=["race_name", "race_id", "rank", "popularity"])
        d["rk"] = pd.to_numeric(d["rank"], errors="coerce")
        d["pop"] = pd.to_numeric(d["popularity"], errors="coerce")
        d["rkey"] = d["race_id"].astype(str).str[:8]
        _HIST = d
    key = str(race_name)[:5]
    if not any(g in str(race_name) for g in ["G1", "G2", "G3"]):
        return None
    sub = _HIST[_HIST["race_name"].astype(str).str[:5] == key]
    wins = sub[sub["rk"] == 1]
    if len(wins) < 10:
        return None
    fav_win = (wins["pop"] == 1).mean()
    upset = (wins["pop"] >= 7).mean()
    return {"n": len(wins), "fav_win": fav_win, "upset": upset}


def describe_day(csv_path: str):
    from tfjv_entries import load_tfjv_entries
    from elite_neglect import build_elite_neglect
    from honmei import build_honmei_reliability
    data = load_tfjv_entries(csv_path)
    rows = []
    for rid, info in data.items():
        names = [e["horse_name"] for e in info["entries"]]
        pm = {e["horse_name"]: (int(e["popularity"]) if e.get("popularity") else None) for e in info["entries"]}
        elite = build_elite_neglect(names, pm)
        honmei = build_honmei_reliability(names, pm)
        rows.append({
            "venue": info["venue"], "race_no": info["race_no"],
            "race_class": info["race_class"], "surface": info["surface"],
            "distance": info["distance"], "field": len(info["entries"]),
            "elite": [h for h in names if elite.get(h, {}).get("flag")],
            "tekkan": [h for h in names if honmei.get(h, {}).get("tier") == "鉄板"],
            "trap": [h for h in names if honmei.get(h, {}).get("tier") == "罠"],
            "grade_hist": _grade_upset_rate(info["race_class"]),
        })
    rows.sort(key=lambda x: (x["venue"], x["race_no"]))
    return rows


if __name__ == "__main__":
    import glob
    csvs = sorted(glob.glob("C:/TFJV/TXT/出馬表分析*.CSV"))
    csv = sys.argv[1] if len(sys.argv) > 1 else csvs[-1]
    print(f"=== レース判断材料 ({Path(csv).name}) ※予測スコアではない・事実の列挙 ===")
    for r in describe_day(csv):
        parts = []
        if r["elite"]: parts.append(f"妙味穴:{'/'.join(r['elite'])}")
        if r["tekkan"]: parts.append(f"鉄板軸:{'/'.join(r['tekkan'])}")
        if r["trap"]: parts.append(f"危険1人気:{'/'.join(r['trap'])}")
        gh = r["grade_hist"]
        if gh: parts.append(f"重賞荒れ実績:1人気勝率{gh['fav_win']*100:.0f}%/穴勝率{gh['upset']*100:.0f}%(n={gh['n']})")
        info = " | ".join(parts) if parts else "特記なし"
        print(f"  {r['venue']}{r['race_no']:2d}R {r['race_class'][:9]:9s}{r['surface']}{r['distance']} {r['field']}頭 | {info}")
