"""
GitHub Actions 用 週末レースデータ事前取得スクリプト。
GitHubのサーバーIPはnetkeiba db.netkeiba.com へのアクセスが可能。
毎週火曜・水曜に実行し、coming weekendの全レース馬情報をJSONとして保存する。
"""
import sys
import time
import json
import re
import requests
from datetime import date, timedelta
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_INTERVAL = 2  # seconds
SAVE_DIR = Path(__file__).parent.parent.parent / "saved_sessions" / "horse_cache"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RACE_CACHE_FILE = Path(__file__).parent.parent.parent / "saved_sessions" / "_scan_cache.json"


def get_weekend_dates():
    today = date.today()
    days_until_sat = (5 - today.weekday()) % 7
    if days_until_sat == 0:
        days_until_sat = 7  # 今日が土曜なら来週
    saturday = today + timedelta(days=days_until_sat)
    return [saturday.strftime("%Y%m%d"), (saturday + timedelta(1)).strftime("%Y%m%d")]


def get(url):
    time.sleep(REQUEST_INTERVAL)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"  [WARN] GET失敗: {url} → {e}")
        return None


def fetch_race_list(date_str):
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date_str}"
    soup = get(url)
    if not soup:
        return []
    seen = set()
    races = []
    JRA = {str(i).zfill(2) for i in range(1, 11)}
    for a in soup.select("a[href*='shutuba.html'][href*='race_id']"):
        m = re.search(r"race_id=(\d{12})", a.get("href", ""))
        if not m:
            continue
        rid = m.group(1)
        if rid in seen or rid[4:6] not in JRA:
            continue
        seen.add(rid)
        races.append({"race_id": rid, "race_name": a.get_text(strip=True).split("\n")[0]})
    return races


def fetch_entries(race_id):
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = get(url)
    if not soup:
        return []
    entries = []
    for row in soup.select("tr.HorseList"):
        try:
            horse_link = row.select_one(".HorseName a")
            if not horse_link:
                continue
            horse_name = horse_link.get_text(strip=True)
            if not horse_name:
                continue
            m = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
            horse_id = m.group(1) if m else ""
            entries.append({"horse_name": horse_name, "horse_id": horse_id})
        except Exception:
            continue
    return entries


def fetch_horse_history(horse_id, horse_name):
    """db.netkeiba.com から過去成績を取得"""
    cache_path = SAVE_DIR / f"{horse_id}.json"
    # TTL: 7日以内のキャッシュは再取得しない
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            from datetime import datetime
            saved = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
            if (datetime.now() - saved).days < 7:
                print(f"    → キャッシュ使用: {horse_name}")
                return True
        except Exception:
            pass

    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    soup = get(url)
    if not soup:
        return False

    table = soup.select_one("table.db_h_race_results")
    if not table:
        return False

    headers = [th.get_text(strip=True) for th in table.select("thead tr th")]
    rows = []
    for tr in table.select("tbody tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) >= 10:
            rows.append(dict(zip(headers, tds)))

    if not rows:
        return False

    payload = {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "fetched_at": date.today().isoformat(),
        "records": rows,
    }
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )
    print(f"    → 取得完了: {horse_name} ({len(rows)}戦)")
    return True


def main():
    dates = get_weekend_dates()
    print(f"対象週末: {dates}")

    total_ok = 0
    total_fail = 0

    for date_str in dates:
        print(f"\n=== {date_str} ===")
        races = fetch_race_list(date_str)
        # 9R以降に絞る
        races_9r = [r for r in races if int(r["race_id"][-2:]) >= 9]
        print(f"  {len(races)}レース取得 → 9R以降 {len(races_9r)}レース対象")

        for race in races_9r:
            print(f"  {race['race_name']} ({race['race_id']})")
            entries = fetch_entries(race["race_id"])
            print(f"    出走馬: {len(entries)}頭")

            for e in entries:
                if not e.get("horse_id"):
                    continue
                ok = fetch_horse_history(e["horse_id"], e["horse_name"])
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1
                    print(f"    [FAIL] {e['horse_name']}")

    print(f"\n完了: 取得成功 {total_ok}頭 / 失敗 {total_fail}頭")
    if total_ok == 0 and total_fail > 0:
        print("ERROR: 全馬取得失敗 → GitHubからもdb.netkeiba.comがブロックされている可能性")
        sys.exit(1)


if __name__ == "__main__":
    main()
