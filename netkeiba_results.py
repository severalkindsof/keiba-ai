# -*- coding: utf-8 -*-
"""
netkeiba_results.py … 当日の同会場・既走レース結果から「当日リアルタイム馬場バイアス」を実測する。

宿題3(handoff): track_bias.py は前日/当日のTFJV結果CSVから集計するが、当日途中の結果は
TFJVに即時反映されない。そこで netkeiba の確定結果を live で引き、レースが走り次第
「前残り/差し」「内/外」を集計する。宝塚(阪神11R)の前に走る1〜10Rで馬場傾向を掴む。

集計ロジックは track_bias.py と同じ(1〜3着馬の最終コーナー通過位置・馬番)。
通過順=最終コーナーでの位置 → 前(逃先)/後(差追)を頭数比で判定。馬番/頭数で内/外。

使い方:
  python -X utf8 netkeiba_results.py 20260614 阪神          # 当日阪神の既走全Rを集計
  python -X utf8 netkeiba_results.py 20260614 阪神 11       # 11Rより前の既走Rのみで集計(本番)
"""
import sys
import requests
from bs4 import BeautifulSoup
from netkeiba_odds import fetch_race_ids, HEADERS


def fetch_result(race_id: str) -> dict | None:
    """確定結果を返す。未確定/未発走なら None。
    返す: {'cond':馬場, 'weather':天候, 'field':頭数, 'top3':[(着,枠,馬番,通過pos), ...]}"""
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "euc-jp"
    soup = BeautifulSoup(r.content, "lxml")
    rows = soup.select("tr.HorseList")
    if not rows:
        return None  # 未発走
    data = soup.select_one(".RaceData01")
    txt = data.get_text() if data else ""
    cond = next((c for c in ("不良", "重", "稍重", "良") if c in txt), "不明")
    weather = "曇" if "曇" in txt else "晴" if "晴" in txt else "雨" if "雨" in txt else "不明"
    field = len(rows)
    top3 = []
    finished = 0
    for tr in rows:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 10:
            continue
        rank = cells[0]
        if not rank.isdigit():
            continue  # 中止/除外
        finished += 1
        if int(rank) <= 3:
            try:
                waku = int(cells[1]); no = int(cells[2])
                pos = cells[9]  # 最終コーナー通過位置
                ppos = int(pos.split("-")[-1]) if pos and pos.split("-")[-1].isdigit() else None
            except (ValueError, IndexError):
                continue
            top3.append((int(rank), waku, no, ppos))
    if finished == 0:
        return None
    return {"cond": cond, "weather": weather, "field": field, "top3": top3}


def analyze_day(date_yyyymmdd: str, venue: str, before_r: int | None = None):
    ids = fetch_race_ids(date_yyyymmdd)
    targets = sorted([rno for (v, rno) in ids if v == venue])
    if before_r:
        targets = [r for r in targets if r < before_r]
    front = closer = inner = outer = n = 0
    cond = weather = "不明"
    done = []
    for rno in targets:
        rid = ids[(venue, rno)]
        res = fetch_result(rid)
        if not res:
            continue
        done.append(rno)
        cond, weather = res["cond"], res["weather"]
        fs = res["field"]
        if fs < 8:
            continue  # 少頭数はバイアス判定から除外(track_bias準拠)
        for rank, waku, no, ppos in res["top3"]:
            n += 1
            rel = no / fs
            if rel <= 0.4:
                inner += 1
            elif rel >= 0.6:
                outer += 1
            if ppos is not None:
                prel = ppos / fs
                if prel <= 0.4:
                    front += 1
                elif prel >= 0.6:
                    closer += 1

    label = f"{venue}" + (f"(〜{before_r-1}R)" if before_r else "(全既走)")
    print(f"=== {date_yyyymmdd} {label} 当日live実測バイアス ===")
    print(f"  馬場:{cond} 天候:{weather} / 集計レース:{done}")
    if n == 0:
        print("  まだ確定結果なし(未発走)。レース後に再実行してください。")
        return
    pace = "前残り傾向" if front > closer * 1.3 else "差し有利傾向" if closer > front * 1.3 else "フラット"
    waku_b = "内有利" if inner > outer * 1.3 else "外差し有利" if outer > inner * 1.3 else "フラット"
    print(f"  通過位置[前{front}/後{closer}] → {pace}")
    print(f"  枠[内{inner}/外{outer}] → {waku_b}")
    print(f"  (好走{n}頭で集計)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
    else:
        date, venue = args[0], args[1]
        before = int(args[2]) if len(args) > 2 else None
        analyze_day(date, venue, before)
