"""
netkeiba から当日の重賞レース情報・出走馬・オッズを取得する。
対象：JRA中央競馬のみ。

利用上の注意：
- 個人の学習・研究目的に限定して使用してください。
- リクエスト間隔は最低3秒空けます（サーバー負荷軽減）。
- 取得できない場合は手動入力フォームにフォールバックします。
"""
import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date
import streamlit as st

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_INTERVAL = 3  # 秒

# netkeiba 認証Cookie — st.secrets["netkeiba"]["cookie"] からのみ読み込む
# （ISSUE-7: ハードコードを撤廃。流出リスク・期限切れリスクを排除）
_nk_cookie = ""
try:
    _nk_cookie = st.secrets["netkeiba"]["cookie"] or ""
except Exception:
    pass
if _nk_cookie:
    HEADERS["Cookie"] = _nk_cookie

# JRA会場コード
VENUE_CODES = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}


def _get(url: str) -> BeautifulSoup | None:
    """GETリクエストを送ってBeautifulSoupを返す。失敗時はNone。"""
    try:
        time.sleep(REQUEST_INTERVAL)
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        # 第45波: race_list_sub.html などUTF-8化に対応するためmeta charset自動検出
        # 旧来のEUC-JPページも meta から検出できれば正しく扱える
        import re as _re
        m = _re.search(rb'charset=["\']?([\w\-]+)', resp.content[:1024], _re.IGNORECASE)
        if m:
            resp.encoding = m.group(1).decode("ascii", errors="ignore").strip().upper()
        else:
            resp.encoding = "EUC-JP"  # 旧来動作のフォールバック
        return BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"[scraper] 失敗: {url}: {e}")
        return None


# BUG-X5: TTL 4階層統一（60秒=直前オッズ / 900=レース固有 / 3600=日次 / 86400=履歴）
@st.cache_data(ttl=900)
def fetch_today_races(target_date: str | None = None) -> list[dict]:
    """
    指定日（YYYYMMDD）の重賞レース一覧を取得する。
    デフォルトは今日。
    返す dict: {race_id, race_name, date_str}
    """
    if target_date is None:
        target_date = date.today().strftime("%Y%m%d")

    JRA_VENUE_CODES = {str(i).zfill(2) for i in range(1, 11)}
    seen = set()
    races = []

    # race_list_sub.html → race_list.html の順でフォールバック
    urls = [
        f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={target_date}",
        f"https://race.netkeiba.com/top/race_list.html?kaisai_date={target_date}",
    ]
    for url in urls:
        soup = _get(url)
        if soup is None:
            continue
        # shutuba.html 限定 → race_id を含む全リンクへ拡張（ページ構造変化に対応）
        for sel in ["a[href*='shutuba.html'][href*='race_id']", "a[href*='race_id']"]:
            for a in soup.select(sel):
                href = a.get("href", "")
                m = re.search(r"race_id=(\d{12})", href)
                if not m:
                    continue
                race_id = m.group(1)
                if race_id in seen:
                    continue
                seen.add(race_id)
                venue_code = race_id[4:6]
                if venue_code not in JRA_VENUE_CODES:
                    continue  # 地方競馬をスキップ
                name = a.get_text(strip=True).split("\n")[0].strip()
                races.append({
                    "race_id":   race_id,
                    "race_name": name,
                    "date_str":  target_date,
                })
            if races:
                break  # セレクタでヒットしたら次のセレクタは試さない
        # デバッグ出力（ターミナルで確認可能）
        if not races:
            _title = soup.title.string if soup.title else "no-title"
            _nlinks = len(soup.select("a[href*='race_id']"))
            print(f"[scraper] {url} → title={_title!r}, race_id_links={_nlinks}, found=0")
        if races:
            break  # URLでヒットしたら次のURLは試さない
    return races


def get_race_id_from_venue_date(venue: str, date_str: str, race_no: int | str) -> str | None:
    """
    NEW-7: 会場名・日付・レース番号から race_id を自動取得する。
    netkeibaのレース一覧ページをスクレイピングして race_id を逆引き。

    Args:
        venue:    会場名（"東京", "阪神" 等）
        date_str: "YYYY-MM-DD" または "YYYYMMDD"
        race_no:  レース番号（整数 or 文字列）
    Returns:
        12桁の race_id、見つからなければ None
    """
    try:
        yyyymmdd = date_str.replace("-", "")
        races = fetch_today_races(yyyymmdd)
        if not races:
            return None

        # race_no を2桁ゼロ埋め文字列に変換
        rno_str = str(int(race_no)).zfill(2)

        # netkeiba の race_id 形式: YYYY(4) + 会場コード(2) + 開催回(2) + 開催日(2) + レース番号(2)
        # レース番号は下2桁で判定
        for r in races:
            rid = str(r.get("race_id", ""))
            if len(rid) == 12 and rid[10:12] == rno_str:
                # 会場名が一致するか確認（会場コードで比較）
                vname = str(r.get("race_name", ""))
                if venue in vname or not venue:
                    return rid

        # 会場名マッチなしの場合はレース番号のみで返す（最初にマッチしたもの）
        for r in races:
            rid = str(r.get("race_id", ""))
            if len(rid) == 12 and rid[10:12] == rno_str:
                return rid

        return None
    except Exception:
        return None


@st.cache_data(ttl=900)  # 15分キャッシュ（オッズ変動を考慮）
def fetch_race_entries(race_id: str) -> list[dict]:
    """
    出走馬リストと単勝オッズを取得する。
    返す list of dict: {horse_no, horse_name, jockey, popularity, odds, weight_carried, ...}
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return []

    entries = []
    for row in soup.select("tr.HorseList"):
        try:
            # Umaban1〜Umaban18 に対応
            umaban_el = row.select_one("[class*='Umaban']")
            horse_no = umaban_el.get_text(strip=True) if umaban_el else ""
            horse_name = _text(row, ".HorseName")
            jockey = _text(row, ".Jockey")
            weight_carried = _text(row, ".Txt_C")
            # 枠番は tr の class に "Waku1"〜"Waku8" または "Waku01"〜"Waku08" として入っている
            gate = 0
            for cls in row.get("class", []):
                m = re.match(r"Waku0*(\d+)", cls)
                if m:
                    g = int(m.group(1))
                    if 1 <= g <= 8:
                        gate = g
                        break
            # フォールバック: horse_no から枠番を計算（18頭以下）
            if gate == 0:
                try:
                    hn = int(horse_no)
                    gate = (hn + 1) // 2 if hn <= 16 else (hn - 14) // 2 + 8
                    gate = max(1, min(8, gate))
                except Exception:
                    gate = 0
            horse_link = row.select_one(".HorseName a")
            horse_id = ""
            if horse_link:
                m_id = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
                if m_id:
                    horse_id = m_id.group(1)
            # 馬体重（発表後）: "500(+2)" のような形式から先頭3〜4桁を抽出
            horse_weight = 0
            hw_el = row.select_one(".Weight")
            if hw_el:
                hw_txt = hw_el.get_text(strip=True)
                m_hw = re.search(r"(\d{3,4})", hw_txt)
                if m_hw:
                    try:
                        horse_weight = int(m_hw.group(1))
                    except Exception:
                        horse_weight = 0
            entries.append({
                "horse_no": horse_no,
                "horse_name": horse_name,
                "jockey": jockey,
                "weight_carried": weight_carried,
                "gate": gate,
                "popularity": 9,
                "odds": 10.0,
                "sire": "",
                "horse_id": horse_id,
                "horse_weight": horse_weight,
            })
        except Exception:
            continue

    # オッズを別ページから補完
    entries = _enrich_odds(race_id, entries)
    # LATENT-4: 空の horse_name / horse_no エントリを除去（HTML誤パース行 + odds hit失敗防止）
    entries = [e for e in entries if e.get("horse_name") and str(e.get("horse_no", "")).strip()]
    return entries


def _fetch_odds_api(race_id: str, type_code: int, timeout: float = 10):
    """単一のオッズAPI呼び出し（並列化用に分離）"""
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type={type_code}&action=update"
    try:
        return requests.get(url, headers=HEADERS, timeout=timeout).json()
    except Exception:
        return None


def _enrich_odds(race_id: str, entries: list[dict]) -> list[dict]:
    """race.netkeiba.com の APIから単勝・複勝オッズを並列取得（LATENT-19）"""
    from concurrent.futures import ThreadPoolExecutor
    time.sleep(REQUEST_INTERVAL)  # 全体のリクエスト間隔は維持
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_tansho = ex.submit(_fetch_odds_api, race_id, 1, 10)   # 単勝
            # A-1 BUGFIX: 複勝は type=2（type=5 はワイド）
            f_fukusho = ex.submit(_fetch_odds_api, race_id, 2, 8)   # 複勝
            data = f_tansho.result()
            dp   = f_fukusho.result()
    except Exception:
        return _enrich_odds_fallback(race_id, entries)

    if data is None:
        return _enrich_odds_fallback(race_id, entries)

    # "yoso"（予想）でもデータがあれば使う。"NG"のみフォールバック
    if data.get("status") == "NG":
        return _enrich_odds_fallback(race_id, entries)
    if data.get("status") not in ("middle", "fixed", "yoso"):
        return _enrich_odds_fallback(race_id, entries)

    try:
        odds_dict = data["data"].get("odds", {})
        tansho = odds_dict.get("1", {})  # type=1 が単勝

        # 複勝オッズの結果（並列取得済み・A-1 修正：type=2 が複勝）
        fukusho = {}
        if dp and dp.get("status") in ("middle", "fixed", "yoso"):
            fukusho = dp.get("data", {}).get("odds", {}).get("2", {})

        for e in entries:
            umaban = str(e.get("horse_no", "")).zfill(2)
            row = tansho.get(umaban)
            if row and len(row) >= 1:
                # ISSUE-4: 取得成功時のみ値を入れ、失敗時は odds_confirmed=False で None 維持
                try:
                    if row[0] and row[0] != "---":
                        e["odds"] = float(row[0])
                        e["odds_confirmed"] = True
                    else:
                        e["odds"] = 10.0
                        e["odds_confirmed"] = False
                except Exception:
                    e["odds"] = 10.0
                    e["odds_confirmed"] = False
                try:
                    e["popularity"] = int(row[2]) if len(row) >= 3 and row[2] else 9
                except Exception:
                    e["popularity"] = 9
            else:
                # オッズ API ヒット失敗 → 暫定値 + 未確定マーキング
                if e.get("odds") is None:       e["odds"] = 10.0
                if e.get("popularity") is None: e["popularity"] = 9
                e["odds_confirmed"] = False

            # 複勝オッズ（min〜max の範囲）
            prow = fukusho.get(umaban)
            if prow and len(prow) >= 2:
                try:
                    e["place_odds_min"] = float(prow[0]) if prow[0] and prow[0] != "---" else None
                    e["place_odds_max"] = float(prow[1]) if prow[1] and prow[1] != "---" else None
                    # 中間値を place_odds として使用
                    if e["place_odds_min"] and e["place_odds_max"]:
                        e["place_odds"] = round((e["place_odds_min"] + e["place_odds_max"]) / 2, 1)
                except Exception:
                    pass
    except Exception:
        return _enrich_odds_fallback(race_id, entries)

    return entries


# BUG-X5: 多券種オッズは直前変動最重要 → 60秒
@st.cache_data(ttl=60)
def fetch_multi_odds(race_id: str) -> dict:
    """
    多券種オッズを並列取得：馬連(4) / 馬単(6) / 三連複(7) / 三連単(8)

    Returns:
        {
            "quinella":  {"01-02": 12.3, ...},    # 馬連 (key: "min-max" zero-padded)
            "exacta":    {"01-02": 18.5, ...},    # 馬単 (key: "first-second" zero-padded)
            "trio":      {"01-02-03": 45.2, ...}, # 三連複
            "trifecta":  {"01-02-03": 245.0, ...},# 三連単
        }
    """
    from concurrent.futures import ThreadPoolExecutor
    time.sleep(REQUEST_INTERVAL)
    result = {"quinella": {}, "exacta": {}, "trio": {}, "trifecta": {}}
    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {
                "quinella": ex.submit(_fetch_odds_api, race_id, 4, 12),
                "exacta":   ex.submit(_fetch_odds_api, race_id, 6, 12),
                "trio":     ex.submit(_fetch_odds_api, race_id, 7, 15),
                "trifecta": ex.submit(_fetch_odds_api, race_id, 8, 15),
            }
            data_by = {k: f.result() for k, f in futs.items()}
    except Exception:
        return result

    type_codes = {"quinella": "4", "exacta": "6", "trio": "7", "trifecta": "8"}
    for ticket_name, data in data_by.items():
        if not data or data.get("status") not in ("middle", "fixed", "yoso"):
            continue
        try:
            tc = type_codes[ticket_name]
            raw = data.get("data", {}).get("odds", {}).get(tc, {})
            for combo_key, row in raw.items():
                # API のキーは "01-02-03" 形式 or "010203" 連結（券種により）
                # row は典型的に [odds_str, "0", popularity]
                if not row:
                    continue
                try:
                    odds_val = float(row[0]) if row[0] and row[0] != "---" else None
                except Exception:
                    odds_val = None
                if odds_val is None:
                    continue
                # キー正規化：連結→ハイフン区切り
                if "-" not in combo_key and combo_key.isdigit():
                    digits = combo_key
                    chunk = 2
                    parts = [digits[i:i+chunk] for i in range(0, len(digits), chunk)]
                    combo_key = "-".join(parts)
                elif "-" in combo_key:
                    # 第35波 (G4): "1-2-3" 形式をゼロ埋め "01-02-03" に統一
                    # （Harville タブ照合キーと不一致 → 市場オッズが静かに不発見）
                    try:
                        combo_key = "-".join(p.zfill(2) for p in combo_key.split("-"))
                    except Exception:
                        pass
                result[ticket_name][combo_key] = odds_val
        except Exception:
            continue
    return result


def _enrich_odds_fallback(race_id: str, entries: list[dict]) -> list[dict]:
    """フォールバック: odds/index.html から単勝オッズをスクレイピング"""
    url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=1"
    soup = _get(url)
    if soup is None:
        for e in entries:
            e.setdefault("odds", 10.0)
            e.setdefault("popularity", 9)
        return entries

    odds_map = {}
    for row in soup.select("tr.HorseList, tr[id^='tr_']"):
        tds = row.find_all("td")
        if len(tds) >= 3:
            try:
                umaban = tds[0].get_text(strip=True)
                odds_val = float(tds[-2].get_text(strip=True).replace(",", "").replace("---", "10"))
                ninki_text = tds[-1].get_text(strip=True) if len(tds) > 2 else "9"
                ninki = int(re.sub(r"\D", "", ninki_text) or "9")
                if umaban.isdigit():
                    odds_map[umaban] = {"odds": odds_val, "popularity": ninki}
            except Exception:
                continue

    for e in entries:
        info = odds_map.get(str(e.get("horse_no", "")), {})
        e["odds"]       = info.get("odds", 10.0)
        e["popularity"] = info.get("popularity", 9)
        # 第35波 (G3): フォールバック経路で odds_confirmed が未付与だと
        # BUG-C の無効化（暫定オッズ馬の EV/buy 遮断）をすり抜けていた
        e["odds_confirmed"] = "odds" in info
    return entries


# ---- 競馬場の座標 ----
_VENUE_COORDS = {
    "東京":  (35.764, 139.490),
    "中山":  (35.778, 139.920),
    "阪神":  (34.838, 135.401),
    "京都":  (34.901, 135.723),
    "中京":  (35.098, 136.942),
    "小倉":  (33.868, 130.868),
    "新潟":  (37.866, 139.044),
    "福島":  (37.784, 140.466),
    "函館":  (41.768, 140.729),
    "札幌":  (43.058, 141.381),
}

# WMO天気コード → 日本語
_WMO_LABEL = {
    0: "快晴", 1: "晴れ", 2: "薄曇", 3: "曇り",
    45: "霧", 48: "霧氷",
    51: "霧雨(弱)", 53: "霧雨", 55: "霧雨(強)",
    61: "小雨", 63: "雨", 65: "大雨",
    71: "小雪", 73: "雪", 75: "大雪",
    80: "にわか雨(弱)", 81: "にわか雨", 82: "にわか雨(強)",
    95: "雷雨", 96: "雷雨+雹", 99: "激しい雷雨",
}

def _wmo_to_label(code: int) -> str:
    return _WMO_LABEL.get(code, f"コード{code}")

def _rain_to_condition(rain_mm: float) -> str:
    """予想降水量から馬場状態を推定"""
    if rain_mm >= 15: return "不良の可能性"
    if rain_mm >= 5:  return "重〜稍重の可能性"
    if rain_mm >= 1:  return "稍重の可能性"
    return "良馬場見込み"


@st.cache_data(ttl=3600)
def fetch_weather_forecast(venue: str, race_date: str) -> dict:
    """
    競馬場の週末天気予報を取得する（OpenMeteo API, 無料・APIキー不要）。

    Args:
        venue: 競馬場名（東京、阪神、中山 など）
        race_date: レース日（YYYY-MM-DD形式）

    Returns:
        {
          "date": "2026-05-25",
          "weather": "晴れ",
          "precipitation_mm": 0.0,
          "condition_forecast": "良馬場見込み",
          "raw_code": 1
        }
    """
    coords = _VENUE_COORDS.get(venue)
    if coords is None:
        return {"date": race_date, "weather": "不明", "precipitation_mm": 0.0,
                "condition_forecast": "データなし", "raw_code": -1}

    lat, lon = coords
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}"
           f"&daily=weathercode,precipitation_sum,temperature_2m_max"
           f"&forecast_days=10&timezone=Asia%2FTokyo")
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        if race_date in dates:
            idx = dates.index(race_date)
            code  = daily["weathercode"][idx]
            rain  = float(daily["precipitation_sum"][idx] or 0)
            return {
                "date": race_date,
                "weather": _wmo_to_label(code),
                "precipitation_mm": rain,
                "condition_forecast": _rain_to_condition(rain),
                "raw_code": code,
            }
        return {"date": race_date, "weather": "予報期間外", "precipitation_mm": 0.0,
                "condition_forecast": "データなし", "raw_code": -1}
    except Exception as e:
        return {"date": race_date, "weather": "取得失敗", "precipitation_mm": 0.0,
                "condition_forecast": "データなし", "raw_code": -1}


def fetch_horse_sire(horse_id: str) -> str:
    """馬の父名を血統ページから取得する"""
    url = f"https://db.netkeiba.com/horse/{horse_id}"
    soup = _get(url)
    if soup is None:
        return ""
    try:
        # 血統テーブルの最初のセル（父）
        sire_cell = soup.select_one("table.blood_table td")
        return sire_cell.get_text(strip=True) if sire_cell else ""
    except Exception:
        return ""


@st.cache_data(ttl=900)
def fetch_race_meta(race_id: str) -> dict:
    """
    レースのメタ情報（距離・馬場・馬場状態・会場・レース名）を取得する。
    取得失敗 or 必須項目欠落時は空 dict を返す（呼び出し側でガード必須）。
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return {}
    try:
        # race_id の[4:6]から会場名を逆引き
        venue_code = race_id[4:6] if len(race_id) >= 6 else ""
        code_to_venue = {v: k for k, v in VENUE_CODES.items()}
        venue = code_to_venue.get(venue_code, "")

        race_data = soup.select_one("div.RaceData01")
        if not race_data:
            return {}
        text = race_data.get_text()
        distance_m = re.search(r"(\d{3,4})m", text)
        if not distance_m:
            return {}  # 距離が取れないならメタ取得失敗とみなす
        surface = "芝" if "芝" in text else ("ダート" if ("ダート" in text or "ダ" in text) else "")
        if not surface:
            return {}
        condition_m = re.search(r"(良|稍重|重|不良)", text)

        # レース名（RaceName 内）
        race_name_el = soup.select_one("div.RaceName") or soup.select_one(".RaceName")
        race_name = race_name_el.get_text(strip=True) if race_name_el else ""

        return {
            "distance": int(distance_m.group(1)),
            "surface": surface,
            "track_condition": condition_m.group(1) if condition_m else "良",
            "venue": venue,
            "race_name": race_name,
        }
    except Exception:
        return {}


def _text(element, selector: str) -> str:
    el = element.select_one(selector)
    return el.get_text(strip=True) if el else ""


# ---- 馬の過去成績スクレイピング ---- #

_VENUE_NAMES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]


def _categorize_distance(dist: int) -> str:
    if dist <= 1400:
        return "短距離"
    elif dist <= 1800:
        return "マイル"
    elif dist <= 2200:
        return "中距離"
    return "長距離"


def _safe_float_s(s) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except Exception:
        return None


def _safe_int_s(s) -> int | None:
    try:
        return int(str(s))
    except Exception:
        return None


def _parse_past_results(raw: pd.DataFrame, horse_name: str) -> pd.DataFrame:
    """netkeibaの生テーブルをKaggle互換形式に変換する"""
    records = []
    for _, r in raw.iterrows():
        try:
            course = r.get("コース", "")
            if not course:
                continue
            surface = "芝" if "芝" in course else "ダート" if "ダ" in course else ""
            if not surface:
                continue
            dist_m = re.search(r"(\d{3,4})", course)
            if not dist_m:
                continue
            distance = int(dist_m.group(1))

            rank_raw = r.get("着順", "")
            try:
                rank = int(rank_raw)
            except (ValueError, TypeError):
                rank = None

            kaisai = r.get("開催", "")
            venue = next((v for v in _VENUE_NAMES if v in kaisai), "")

            weight_raw = r.get("馬体重", "")
            m_w = re.match(r"(\d+)", str(weight_raw))
            horse_weight = int(m_w.group(1)) if m_w else None

            records.append({
                "horse_name": horse_name,
                "date": r.get("日付", ""),
                "venue": venue,
                "surface": surface,
                "distance": distance,
                "track_condition": r.get("馬場", ""),
                "rank": rank,
                "jockey": r.get("騎手", ""),
                "weight_carried": r.get("斤量", ""),
                "horse_weight": horse_weight,
                "odds": _safe_float_s(r.get("オッズ", "")),
                "popularity": _safe_int_s(r.get("人気", "")),
                "last_3f": _safe_float_s(r.get("上がり", "")),
                "race_name": r.get("レース名", ""),
                "race_class": "",
                "gate": _safe_int_s(r.get("枠番", "")),
                "horse_no": _safe_int_s(r.get("馬番", "")),
                "corner_order": r.get("通過", ""),
            })
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["distance_cat"] = df["distance"].apply(
        lambda x: _categorize_distance(int(x)) if pd.notna(x) else "不明"
    )
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["win_flag"] = (df["rank"] == 1).astype(int)
    df["place_flag"] = (df["rank"] <= 3).astype(int)
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def fetch_horse_past_results(horse_id: str, horse_name: str) -> pd.DataFrame:
    """
    netkeibaの馬詳細ページから過去成績を取得し、Kaggle互換のDataFrameを返す。
    horse_id: netkeibaの馬ID（10桁数字）
    """
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    soup = _get(url)
    if soup is None:
        return pd.DataFrame()

    table = soup.select_one("table.db_h_race_results")
    if table is None:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.select("thead tr th")]
    if not headers:
        return pd.DataFrame()

    rows = []
    for tr in table.select("tbody tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) >= 10:
            rows.append(dict(zip(headers, tds)))

    if not rows:
        return pd.DataFrame()

    return _parse_past_results(pd.DataFrame(rows), horse_name)


# ---- 手動入力フォールバック ---- #

def manual_entry_template() -> list[dict]:
    """
    スクレイピング失敗時に手動入力できるサンプルテンプレートを返す。
    ユーザーがStreamlitのフォームで上書きして使う。
    """
    return [
        {
            "horse_no": str(i),
            "horse_name": f"馬{i}",
            "jockey": "",
            "gate": ((i - 1) // 2) + 1,  # 2頭ずつ同枠
            "odds": round(10.0 * i * 0.7, 1),
            "popularity": i,
            "sire": "",
            "weight_carried": "55.0",
            "horse_weight": 480,
        }
        for i in range(1, 9)
    ]


# ============================================================
# C-5 (第12波): 当日の終了済みレース結果取得（intraday time_bias 用）
# ============================================================
@st.cache_data(ttl=300)  # 5分キャッシュ（当日結果は逐次更新されるため短め）
def fetch_today_finished_results(date_str: str, venue: str) -> list[dict]:
    """
    指定日（YYYYMMDD）・会場のうち、既に終了したレースの結果を返す。

    Returns:
        list of dict: [{race_no, distance, surface, finish_time_sec, winner_horse_no,
                        winner_corner4_pos, field_size, track_condition}, ...]
    """
    venue_code = VENUE_CODES.get(venue)
    if not venue_code:
        return []
    # netkeiba 当日結果一覧
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str}"
    soup = _get(url)
    if soup is None:
        return []

    # 会場コードを含む race_id だけ抽出
    race_ids: list[str] = []
    seen: set[str] = set()
    for a in soup.select("a[href*='race_id']"):
        m = re.search(r"race_id=(\d{12})", a.get("href", ""))
        if not m:
            continue
        rid = m.group(1)
        if rid in seen:
            continue
        if rid[4:6] != venue_code:
            continue
        seen.add(rid)
        race_ids.append(rid)

    results: list[dict] = []
    for rid in race_ids:
        meta = _fetch_race_result_brief(rid)
        if meta is None:
            continue
        results.append(meta)
    return results


def _fetch_race_result_brief(race_id: str) -> dict | None:
    """単一レースの結果サマリ（1着馬の情報のみ）を取得。未終了レースは None。

    (第20波 U3 修正) セレクタを race_diary.fetch_race_result_from_netkeiba で
    実証済みの `table.RaceTable01 tr.HorseList` 方式に統一。
    旧実装の `tr.Rank01` は誤りで、常に None を返し当日補正が機能しなかった。
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return None
    rank1_cells = None
    for row in soup.select("table.RaceTable01 tr.HorseList"):
        cells = row.find_all("td")
        if len(cells) < 8:
            continue
        if cells[0].get_text(strip=True) == "1":
            rank1_cells = cells
            break
    if rank1_cells is None:
        # 未終了 or パース失敗
        return None
    try:
        tds = [c.get_text(strip=True) for c in rank1_cells]
        horse_no = int(tds[2]) if tds[2].isdigit() else None
        time_str = tds[7] if len(tds) > 7 else ""
        finish_sec = None
        m = re.match(r"(\d+):(\d+)\.(\d+)", time_str)
        if m:
            finish_sec = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10.0
        # コーナー通過順（"4-3-2-1" 形式のセルを探して最後を取る）
        corner4 = None
        for t in tds:
            if re.match(r"^[\d\-]+$", t) and "-" in t:
                parts = t.split("-")
                if parts[-1].isdigit():
                    corner4 = int(parts[-1])
                    break
    except Exception:
        return None
    # 距離・surface・field_size はレース見出しから
    title = soup.select_one("div.RaceData01, .RaceData01")
    distance = None
    surface = None
    track_cond = None
    field_size = None
    if title:
        ttxt = title.get_text(" ", strip=True)
        dm = re.search(r"(芝|ダ|障)(\d{3,4})m", ttxt)
        if dm:
            surface = {"芝": "芝", "ダ": "ダート", "障": "障害"}[dm.group(1)]
            distance = int(dm.group(2))
        cm = re.search(r"馬場:(良|稍重|重|不良)", ttxt)
        if cm:
            track_cond = cm.group(1)
        fm = re.search(r"(\d+)頭", ttxt)
        if fm:
            field_size = int(fm.group(1))
    return {
        "race_id": race_id,
        "race_no": int(race_id[-2:]),
        "distance": distance,
        "surface": surface,
        "track_condition": track_cond,
        "field_size": field_size,
        "finish_time_sec": finish_sec,
        "winner_horse_no": horse_no,
        "winner_corner4_pos": corner4,
    }


# ============================================================
# 第40波: 初ブリンカーアラート — 「新聞」タブ(shutuba_past.html)から
# ブリンカー装着馬を自動取得。実HTML構造を安田記念で確認済み:
#   span.Mark にテキスト "B" を持つ行の a[href*='/horse/'] が装着馬。
#   さらに各馬の過去5走欄に B が無ければ「初装着の可能性」と判定。
# ============================================================
@st.cache_data(ttl=900)
def fetch_blinker_horses(race_id: str) -> dict:
    """
    出馬表「新聞」タブからブリンカー装着馬を取得。
    Returns: {"all": [馬名...], "first_time": [初装着の可能性がある馬名...]}
    """
    url = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return {"all": [], "first_time": []}
    all_b, first_b = [], []
    seen = set()
    for row in soup.select("tr.HorseList"):
        name_el = row.select_one("a[href*='/horse/']")
        if not name_el:
            continue
        name = name_el.get_text(strip=True)
        if not name or name in seen:
            continue
        # 現在のブリンカー: 馬名セル（先頭近く）の span.Mark に "B"
        marks = [m.get_text(strip=True) for m in row.select("span.Mark")]
        has_b = "B" in marks
        if not has_b:
            continue
        seen.add(name)
        all_b.append(name)
        # 第40波修正: shutuba_past の B マークは「初/通常」の区別がHTMLに無い
        # （netkeibaはCSSで色付けしているだけ）。全馬の行内B数=1個で同一。
        # → 自動での「初」判定は不可能と確定。装着馬リストのみ返し、
        # 「初」判定は UI 側でユーザーがチェックする方式に変更。
        # first_b リストは互換のため空のまま返す。
    return {"all": all_b, "first_time": first_b}


def fetch_blinkers_for_date(date_str: str, venue: str | None = None,
                             min_race_no: int = 1) -> list[dict]:
    """
    指定日の全レース（min_race_no 以降）のブリンカー装着馬をまとめて取得。
    Returns: [{race_id, venue, race_no, all:[...], first_time:[...]}, ...]
    """
    races = fetch_today_races(date_str)
    out = []
    for r in races:
        rid = r.get("race_id", "")
        if len(rid) != 12:
            continue
        rno = int(rid[-2:])
        if rno < min_race_no:
            continue
        if venue:
            vcode = VENUE_CODES.get(venue)
            if vcode and rid[4:6] != vcode:
                continue
        b = fetch_blinker_horses(rid)
        if b["all"]:
            out.append({"race_id": rid, "race_no": rno,
                        "all": b["all"], "first_time": b["first_time"]})
    return out
