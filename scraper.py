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

# Streamlit Secrets から netkeiba の認証Cookie を読み込む
try:
    _nk_cookie = st.secrets["netkeiba"]["cookie"]
    if _nk_cookie:
        HEADERS["Cookie"] = _nk_cookie
except Exception:
    pass

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
        resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "lxml")
    except Exception as e:
        print(f"[scraper] 失敗: {url}: {e}")
        return None


@st.cache_data(ttl=1800)
def fetch_today_races(target_date: str | None = None) -> list[dict]:
    """
    指定日（YYYYMMDD）の重賞レース一覧を取得する。
    デフォルトは今日。
    返す dict: {race_id, race_name, venue, race_no, date_str}
    """
    if target_date is None:
        target_date = date.today().strftime("%Y%m%d")

    # race_list_sub.html を使用（race_list.htmlはJS動的生成に変更されたため）
    url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={target_date}"
    soup = _get(url)
    if soup is None:
        return []

    JRA_VENUE_CODES = {str(i).zfill(2) for i in range(1, 11)}

    seen = set()
    races = []
    # shutuba.html?race_id=... のリンクを取得（重複除外）
    for a in soup.select("a[href*='shutuba.html'][href*='race_id']"):
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
            "race_id": race_id,
            "race_name": name,
            "date_str": target_date,
        })
    return races


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
            # 枠番は tr の class に "Waku1"〜"Waku8" として入っている
            gate = 1
            for cls in row.get("class", []):
                m = re.match(r"Waku(\d)", cls)
                if m:
                    gate = int(m.group(1))
                    break
            horse_link = row.select_one(".HorseName a")
            horse_id = ""
            if horse_link:
                m_id = re.search(r"/horse/(\w+)", horse_link.get("href", ""))
                if m_id:
                    horse_id = m_id.group(1)
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
            })
        except Exception:
            continue

    # オッズを別ページから補完
    entries = _enrich_odds(race_id, entries)
    # 空の horse_name エントリを除去（HTML誤パース行）
    entries = [e for e in entries if e.get("horse_name")]
    return entries


def _enrich_odds(race_id: str, entries: list[dict]) -> list[dict]:
    """race.netkeiba.com の APIから単勝オッズ・人気を取得する"""
    import json as _json
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update"
    try:
        time.sleep(REQUEST_INTERVAL)
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
    except Exception:
        # JSONでなければフォールバック（odds/index.htmlから取得）
        return _enrich_odds_fallback(race_id, entries)

    if data.get("status") not in ("middle", "fixed"):
        return _enrich_odds_fallback(race_id, entries)

    # APIレスポンスのパース
    # odds = {"1": {"01": ["オッズ", "", "人気順位"], "02": [...], ...}}
    try:
        odds_dict = data["data"].get("odds", {})
        tansho = odds_dict.get("1", {})  # type=1 が単勝

        for e in entries:
            umaban = str(e.get("horse_no", "")).zfill(2)
            row = tansho.get(umaban)
            if row and len(row) >= 1:
                try:
                    e["odds"] = float(row[0]) if row[0] and row[0] != "---" else 10.0
                except Exception:
                    e["odds"] = 10.0
                try:
                    e["popularity"] = int(row[2]) if len(row) >= 3 and row[2] else 9
                except Exception:
                    e["popularity"] = 9
            else:
                if e.get("odds") is None:       e["odds"] = 10.0
                if e.get("popularity") is None: e["popularity"] = 9
    except Exception:
        return _enrich_odds_fallback(race_id, entries)

    return entries


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
    return entries


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
    レースのメタ情報（距離・馬場・馬場状態・会場）を取得する
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = _get(url)
    if soup is None:
        return {}
    try:
        race_data = soup.select_one("div.RaceData01")
        if not race_data:
            return {}
        text = race_data.get_text()
        distance_m = re.search(r"(\d{3,4})m", text)
        surface = "芝" if "芝" in text else "ダート" if "ダート" in text else "芝"
        condition_m = re.search(r"(良|稍重|重|不良)", text)
        return {
            "distance": int(distance_m.group(1)) if distance_m else 2000,
            "surface": surface,
            "track_condition": condition_m.group(1) if condition_m else "良",
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
