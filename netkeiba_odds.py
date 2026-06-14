# -*- coding: utf-8 -*-
"""
netkeiba_odds.py … netkeiba から確定/リアルタイム単勝オッズを取得する。

TFJVの出馬表エクスポートにはオッズが入らないため、レース選定の「段2(確定オッズ絞り)」は
netkeiba から実オッズを引く。レース直前のリアルタイムオッズもこのAPIで取れる。

netkeiba race_id 形式: 年(4) + 場(2) + 回(2) + 日(2) + R(2) 計12桁
  場コード: 01札幌 02函館 03福島 04新潟 05東京 06中山 07中京 08京都 09阪神 10小倉

使い方:
  python -X utf8 netkeiba_odds.py 20260614              # 当日の会場・R→race_id一覧
  python -X utf8 netkeiba_odds.py 20260614 阪神 11        # 指定レースの確定オッズ
"""
import sys
import re
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://race.netkeiba.com/",
}
VENUE = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


def fetch_race_ids(date_yyyymmdd: str) -> dict:
    """開催日(YYYYMMDD)の {(会場名, R番号): race_id} を返す。"""
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_yyyymmdd}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.encoding = "euc-jp"
    ids = sorted(set(re.findall(r"race_id=(\d{12})", r.text)))
    out = {}
    for i in ids:
        venue = VENUE.get(i[4:6])
        if venue:
            out[(venue, int(i[10:12]))] = i
    return out


def fetch_win_odds(race_id: str) -> dict:
    """race_id の単勝オッズ {馬番int: (オッズfloat, 人気int)} を返す。取消等は除外。"""
    url = (f"https://race.netkeiba.com/api/api_get_jra_odds.html"
           f"?race_id={race_id}&type=1&action=update")
    r = requests.get(url, headers=HEADERS, timeout=15)
    j = r.json()
    tan = j.get("data", {}).get("odds", {}).get("1", {})
    out = {}
    for no, v in tan.items():
        try:
            odds = float(v[0]) if v[0] not in ("", None) else None
            pop = int(v[2]) if len(v) > 2 and v[2] not in ("", None) else None
            # 999.9 は取消/除外馬のダミー値(netkeibaの上限キャップ)。実オッズではないので除外
            if odds and odds < 999:
                out[int(no)] = (odds, pop)
        except (ValueError, IndexError):
            continue
    return out


def fetch_track_condition(date_yyyymmdd: str, venue: str, race_no: int) -> str | None:
    """会場名・R番号から当日の実馬場(良/稍重/重/不良)を取得。取れなければNone。"""
    ids = fetch_race_ids(date_yyyymmdd)
    rid = ids.get((venue, race_no))
    if not rid:
        return None
    try:
        url = f"https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.encoding = "euc-jp"
        m = re.search(r"馬場:(不良|稍重|稍|重|良)", r.text)
        if not m:
            return None
        c = m.group(1)
        return "稍重" if c == "稍" else c
    except Exception:
        return None


def fetch_odds_for(date_yyyymmdd: str, venue: str, race_no: int) -> dict:
    """会場名・R番号から確定オッズを取得。見つからなければ {}。"""
    ids = fetch_race_ids(date_yyyymmdd)
    rid = ids.get((venue, race_no))
    if not rid:
        return {}
    return fetch_win_odds(rid)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1:
        ids = fetch_race_ids(args[0])
        print(f"{args[0]} 開催: {len(ids)}レース")
        for (v, rno), rid in sorted(ids.items()):
            print(f"  {v}{rno:>2}R  race_id={rid}")
    elif len(args) == 3:
        date, venue, rno = args[0], args[1], int(args[2])
        odds = fetch_odds_for(date, venue, rno)
        if not odds:
            print("該当レースが見つかりません")
        else:
            print(f"{venue}{rno}R 確定単勝オッズ（人気順）:")
            for no, (o, p) in sorted(odds.items(), key=lambda x: x[1][1] or 99):
                print(f"  {p:>2}人気  馬番{no:>2}  {o:>6.1f}倍")
            vals = [o for o, _ in odds.values()]
            print(f"  → 頭数{len(odds)} 最下位{max(vals):.0f}倍 1番人気{min(vals):.1f}倍")
    else:
        print(__doc__)
