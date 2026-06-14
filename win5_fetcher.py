"""WIN5対象レース取得モジュール。

netkeiba の WIN5 ページから対象5レースの race_id を抽出する。
キャリーオーバー額もページから取得を試みる（失敗時は None）。
"""
from __future__ import annotations
import re
import requests
from bs4 import BeautifulSoup

try:
    import streamlit as st
    _cache = st.cache_data(ttl=3600)
except Exception:
    def _cache(fn): return fn  # streamlit 外でも使えるように

_URL = "https://race.netkeiba.com/top/win5.html"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


@_cache
def fetch_win5_races() -> dict:
    """WIN5対象5レースとキャリーオーバー情報を返す。

    Returns
    -------
    {
        "race_ids":      [str, ...],   # 12桁 race_id 5本（取得順）
        "carryover_yen": int | None,   # キャリーオーバー額（円）。なければ None
        "title":         str,          # ページタイトル（日付付き）
    }
    失敗時は race_ids=[] を返す。
    """
    try:
        r = requests.get(_URL, headers=_HEADERS, timeout=15)
        r.encoding = "EUC-JP"
        html = r.text
    except Exception:
        return {"race_ids": [], "carryover_yen": None, "title": ""}

    race_ids = sorted(set(re.findall(r"race_id=(\d{12})", html)))[:5]

    carry = None
    m = re.search(r"キャリーオーバー[^0-9]{0,30}([\d,]+)\s*円", html)
    if m:
        try:
            carry = int(m.group(1).replace(",", ""))
        except Exception:
            carry = None

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string or "").strip() if soup.title else ""

    return {"race_ids": race_ids, "carryover_yen": carry, "title": title}


if __name__ == "__main__":
    info = fetch_win5_races()
    print(f"タイトル: {info['title']}")
    print(f"対象レース: {info['race_ids']}")
    print(f"キャリーオーバー: {info['carryover_yen']:,}円" if info['carryover_yen'] else "キャリーオーバー: なし")
