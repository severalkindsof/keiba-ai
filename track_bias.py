# -*- coding: utf-8 -*-
"""
track_bias.py … 当日(または前日)の実レース結果から馬場バイアスを実測する。
憶測でなく、その日の各レースの1〜3着馬の脚質・枠位置を集計し「内有利/外有利」「前残り/差し有利」を客観判定。
TFJV結果CSV(今日の結果XXXX.csv・174列フォーマット)を読む。

使い方:
  python -X utf8 track_bias.py 今日の結果0613.csv 阪神
  python -X utf8 track_bias.py 今日の結果0613.csv          # 全会場
"""
import sys
import csv

VENUE = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
         "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
# 結果CSVの列インデックス: 1-3着の馬番・脚質・頭数
RANK_NO = [13, 14, 15]        # 1着/2着/3着の馬番
STYLE_COL = [98, 121, 144]    # 1着/2着/3着の脚質(逃/先/差/追)
FIELD = 7                     # 頭数


def analyze(csv_path, venue_filter=None):
    with open(csv_path, encoding="cp932", errors="replace") as f:
        rows = [r for r in csv.reader(f) if len(r) > 150]
    front, closer, inner, outer, total = 0, 0, 0, 0, 0
    by_venue = {}
    for r in rows:
        rid = r[0]
        ven = VENUE.get(rid[8:10], rid[8:10])
        if venue_filter and ven != venue_filter:
            continue
        try:
            n = int(r[FIELD])
        except ValueError:
            continue
        if n < 8:
            continue
        by_venue.setdefault(ven, {"front": 0, "closer": 0, "inner": 0, "outer": 0, "n": 0})
        for mc, sc in zip(RANK_NO, STYLE_COL):
            try:
                no = int(r[mc])
            except ValueError:
                continue
            style = r[sc].strip()
            pos = no / n
            v = by_venue[ven]
            v["n"] += 1
            total += 1
            if style in ("逃", "先"):
                front += 1; v["front"] += 1
            elif style in ("差", "追"):
                closer += 1; v["closer"] += 1
            if pos <= 0.4:
                inner += 1; v["inner"] += 1
            elif pos >= 0.6:
                outer += 1; v["outer"] += 1

    label = f"{venue_filter}" if venue_filter else "全会場"
    print(f"=== {csv_path} {label} 実測バイアス（1〜3着馬の脚質・枠）===")
    for ven, v in by_venue.items():
        if v["n"] == 0:
            continue
        fr, cl = v["front"], v["closer"]
        inn, out = v["inner"], v["outer"]
        pace_bias = ("前残り傾向" if fr > cl * 1.3 else "差し有利傾向" if cl > fr * 1.3 else "フラット")
        waku_bias = ("内有利" if inn > out * 1.3 else "外差し有利" if out > inn * 1.3 else "フラット")
        print(f"  {ven}: 連対脚質[逃先{fr}/差追{cl}]→{pace_bias} | "
              f"枠[内{inn}/外{out}]→{waku_bias} (好走{v['n']}頭)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        path = sys.argv[1]
        if "\\" not in path and "/" not in path:
            path = "C:/TFJV/TXT/" + path
        analyze(path, sys.argv[2] if len(sys.argv) > 2 else None)
