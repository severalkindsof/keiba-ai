"""
JRA-VAN JV-Link から馬の過去成績を取得して
saved_sessions/horse_cache/ にJSONで保存するスクリプト。

実行方法: py -3.11-32 jvlink_fetch.py
"""
import win32com.client
import json
import re
import sys
import struct
from pathlib import Path
from datetime import datetime, timedelta

assert struct.calcsize("P") * 8 == 32, "32bit Pythonで実行してください: py -3.11-32 jvlink_fetch.py"

SAVE_DIR = Path(__file__).parent / "saved_sessions" / "horse_cache"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---- JV-Link 初期化 ----
jv = win32com.client.Dispatch("JVDTLab.JVLink")

SERVICE_KEY = "AUJC46NQ7ELLQX3B4"

def jv_init():
    """JV-Linkを初期化"""
    # まずキーを登録（初回のみ必要）
    try:
        ret_set = jv.JVSetServiceKey(SERVICE_KEY)
        print(f"JVSetServiceKey: {ret_set}")
    except Exception as e:
        print(f"JVSetServiceKey スキップ: {e}")

    ret = jv.JVInit(SERVICE_KEY)
    if ret != 0:
        print(f"JVInit エラー: {ret}")
        sys.exit(1)
    print("JV-Link 初期化完了")

def fetch_race_results_recent(days_back=365):
    """
    直近N日分のレース結果を取得する。
    データ種別: RA（レース詳細）、SE（馬毎レース情報）
    """
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d%H%M%S")

    print(f"レース結果取得開始（{days_back}日分）...")

    # SE: 馬毎レース情報（出走・着順・タイム等）
    # OUTパラメータなしで呼ぶパターンも試す
    try:
        result = jv.JVOpen("SE", from_date, 1)
    except Exception:
        result = jv.JVOpen("SE", from_date, 1, 0, 0, "")
    print(f"JVOpen 戻り値: {result}")

    # win32comはOUT引数をタプルで返す (ret, readCount, downloadCount, lastFileTimeStamp)
    if isinstance(result, tuple):
        ret        = result[0]
        read_count = result[1] if len(result) > 1 else 0
    else:
        ret        = result
        read_count = ret

    if ret < 0:
        print(f"JVOpen エラー: {ret}")
        return []

    total = read_count
    print(f"取得予定: {total}件")

    records = []
    count = 0
    while True:
        buff = " " * 10240
        ret, buff, filename = jv.JVRead(buff, 10240, "")
        if ret == 0:
            break  # 終了
        if ret < 0:
            print(f"JVRead エラー: {ret}")
            break
        if ret > 0:
            records.append(buff.strip())
            count += 1
            if count % 1000 == 0:
                print(f"  取得中... {count}/{total}")

    jv.JVClose()
    print(f"取得完了: {count}件")
    return records


def parse_se_record(record):
    """
    SE（馬毎レース情報）レコードをパースする。
    JV-Data仕様書に基づくフィールド位置。
    """
    try:
        if len(record) < 200:
            return None

        # SEレコードのフィールド（JV-Data仕様書 SE形式）
        record_type = record[0:2]
        if record_type != "SE":
            return None

        data_kubun     = record[2:3]
        race_id        = record[3:19]    # レースID 16桁
        kaisai_date    = record[3:11]    # 開催年月日
        venue_code     = record[11:13]   # 競馬場コード
        race_no        = record[15:17]   # レース番号

        horse_no       = record[17:19]   # 馬番
        wakuban        = record[19:20]   # 枠番
        horse_id       = record[20:30]   # 血統登録番号（馬ID）

        horse_name_raw = record[30:66].strip()  # 馬名（36文字）

        rank_raw       = record[66:68].strip()  # 着順

        distance_raw   = record[100:104].strip() # 距離
        surface_raw    = record[98:99]            # 芝ダ区分

        time_raw       = record[106:110].strip()  # タイム（1/10秒）

        jockey_id      = record[120:125].strip()

        last3f_raw     = record[130:133].strip()  # 上がり3F（1/10秒）

        horse_weight   = record[133:136].strip()  # 馬体重
        weight_diff    = record[136:139].strip()  # 体重増減

        odds_raw       = record[140:145].strip()  # 単勝オッズ
        popularity     = record[145:147].strip()  # 人気

        corner_pos     = record[150:156].strip()  # コーナー通過順

        return {
            "horse_id":      horse_id.strip(),
            "horse_name":    horse_name_raw,
            "race_id":       race_id.strip(),
            "date":          f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:8]}",
            "venue_code":    venue_code,
            "race_no":       race_no,
            "horse_no":      horse_no,
            "gate":          wakuban,
            "rank":          int(rank_raw) if rank_raw.isdigit() else None,
            "surface":       "芝" if surface_raw == "1" else "ダート" if surface_raw == "2" else "",
            "distance":      int(distance_raw) if distance_raw.isdigit() else None,
            "time":          int(time_raw) if time_raw.isdigit() else None,
            "last_3f":       round(int(last3f_raw) / 10, 1) if last3f_raw.isdigit() else None,
            "horse_weight":  int(horse_weight) if horse_weight.isdigit() else None,
            "weight_change": int(weight_diff) if weight_diff.lstrip("+-").isdigit() else None,
            "odds":          round(int(odds_raw) / 10, 1) if odds_raw.isdigit() else None,
            "popularity":    int(popularity) if popularity.isdigit() else None,
            "corner_order":  corner_pos,
        }
    except Exception as e:
        return None


def save_horse_caches(records):
    """馬IDごとにグループ化してJSONで保存"""
    from collections import defaultdict

    horse_data = defaultdict(lambda: {"horse_name": "", "records": []})

    for r in records:
        if r and r.get("horse_id"):
            hid = r["horse_id"]
            if r.get("horse_name"):
                horse_data[hid]["horse_name"] = r["horse_name"]
            horse_data[hid]["records"].append(r)

    saved = 0
    for horse_id, data in horse_data.items():
        if not data["records"]:
            continue

        path = SAVE_DIR / f"{horse_id}.json"
        payload = {
            "horse_id":   horse_id,
            "horse_name": data["horse_name"],
            "fetched_at": datetime.now().isoformat(),
            "records":    sorted(data["records"], key=lambda x: x.get("date", ""), reverse=True)
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )
        saved += 1

    print(f"馬別JSONを保存: {saved}頭")
    return saved


def main():
    jv_init()

    # 直近1年分のレース結果を取得
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    raw_records = fetch_race_results_recent(days_back=days)

    # パース
    parsed = [parse_se_record(r) for r in raw_records]
    parsed = [r for r in parsed if r]
    print(f"パース成功: {len(parsed)}件")

    # 保存
    save_horse_caches(parsed)
    print("完了!")


if __name__ == "__main__":
    main()
