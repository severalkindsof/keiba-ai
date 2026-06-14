"""WIN5戦略A（検証済みロジック・当日運用）

検証(第63波・リークなし): 鉄板1人気の勝率36.5% > 非鉄板29.8% > 全1人気32.1%。
→ 鉄板レッグはpin(1頭)、非鉄板/混戦レッグは広く流す、が盲目pinより合理的。
ただしWIN5自体は-EV(3-4点で回収率56-83%)。本戦略は点数配分の最適化であって
プラス化ではない。夢枠として割り切る前提。

各レッグを honmei で 鉄板/標準/混戦 に分類し、予算内で pin×spread を提案。
"""
from __future__ import annotations
import sys
from pathlib import Path
_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))


def classify_leg(info: dict) -> dict:
    """1レッグ(1レース)を pin/spread 判定。"""
    from honmei import build_honmei_reliability
    from elite_neglect import build_elite_neglect
    names = [e["horse_name"] for e in info["entries"]]
    pm = {e["horse_name"]: (int(e["popularity"]) if e.get("popularity") else None) for e in info["entries"]}
    hon = build_honmei_reliability(names, pm)
    elite = build_elite_neglect(names, pm)
    # 人気順
    ranked = sorted([e for e in info["entries"] if pm.get(e["horse_name"])],
                    key=lambda e: pm[e["horse_name"]])
    fav = ranked[0]["horse_name"] if ranked else None
    fav_tier = hon.get(fav, {}).get("tier", "") if fav else ""
    elite_horses = [h for h in names if elite.get(h, {}).get("flag")]

    if fav_tier == "鉄板":
        mode = "PIN"; picks = [fav]
        note = f"鉄板1人気(勝率36.5%級)→1頭pin"
    elif fav_tier == "罠" or elite_horses:
        mode = "SPREAD"; picks = [r["horse_name"] for r in ranked[:5]] + elite_horses
        picks = list(dict.fromkeys(picks))
        note = "波乱含み(罠1人気/妙味穴)→広く流す"
    else:
        mode = "MID"; picks = [r["horse_name"] for r in ranked[:3]]
        note = "標準→上位3頭"
    return {"venue": info["venue"], "race_no": info["race_no"],
            "race_class": info["race_class"], "mode": mode, "picks": picks,
            "note": note, "fav": fav}


def build_strategy(legs: list[dict], budget: int = 400) -> dict:
    """5レッグの分類から買い目を組む。budget(円)内に収まるよう調整。"""
    classified = [classify_leg(l) for l in legs]
    # 初期: PIN=1頭, MID=3頭, SPREAD=5頭
    import itertools
    def total_points(sel): 
        p = 1
        for s in sel: p *= max(1, len(s))
        return p
    sels = [c["picks"] for c in classified]
    # 予算(点数=budget/100)内に収まるまで、点数の多いレッグから1頭ずつ削る
    max_pts = budget // 100
    while total_points(sels) > max_pts:
        # 最も頭数の多いレッグを1つ削る
        i = max(range(5), key=lambda k: len(sels[k]))
        if len(sels[i]) <= 1:
            break
        sels[i] = sels[i][:-1]
    return {"classified": classified, "selections": sels,
            "points": total_points(sels), "cost": total_points(sels) * 100}


if __name__ == "__main__":
    import glob
    from tfjv_entries import load_tfjv_entries
    csv = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("C:/TFJV/TXT/出馬表分析*.CSV"))[-1]
    # WIN5対象レースを取得(なければ手動指定)
    try:
        from win5_fetcher import fetch_win5_races
        info = fetch_win5_races()
        rids = info.get("race_ids", [])
    except Exception:
        rids = []
    data = load_tfjv_entries(csv)
    # netkeiba race_id と TFJV を会場+R でマッチは別途。ここではCSV内の重賞5レースを仮利用デモ
    print("win5_strategy: WIN5対象5レッグのentriesを渡して build_strategy() を呼ぶ")
    print("当日は race_select と同様に 出馬表分析CSV + WIN5対象レース(venue,raceno) を指定")
