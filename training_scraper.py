"""
調教タイム取得モジュール。

データソース: netkeiba の調教ページ
URL形式: https://race.netkeiba.com/race/shutuba_past.html?race_id=XXXX
         または馬個別ページの調教タブ

実現可能性:
- race_id が判明している場合は出馬表ページから horse_id を取得
- horse_id から調教ページをスクレイピング
- 失敗した場合は手動入力フォームにフォールバック

閾値（栗東基準）:
- 坂路: ≤53.4秒 → 仕上がり良好
- CW(ウッド): 6F ≤ 83.0秒 → 標準, ≤80.0秒 → 好仕上がり
- 単走 vs 併せ馬: 併せ馬は相対的な動きで評価
"""
import time
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}
REQUEST_INTERVAL = 3


def _get(url: str) -> BeautifulSoup | None:
    try:
        time.sleep(REQUEST_INTERVAL)
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = r.apparent_encoding
        return BeautifulSoup(r.content, 'lxml')
    except Exception:
        return None


def get_horse_ids_from_race(race_id: str) -> dict[str, str]:
    """
    出馬表ページから「馬名 → horse_id」の対応表を取得する。
    """
    url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    soup = _get(url)
    if soup is None:
        return {}

    id_map = {}
    for a in soup.select('a[href*="/horse/"]'):
        href = a.get('href', '')
        m = re.search(r'/horse/(\d+)', href)
        if m:
            name = a.get_text(strip=True)
            if name:
                id_map[name] = m.group(1)
    return id_map


@st.cache_data(ttl=1800)
def fetch_training_times(horse_id: str) -> list[dict]:
    """
    馬IDから調教タイムを取得する。
    Returns list of dicts: {date, course, style, time_6f, time_4f, time_1f, rank, note}
    """
    url = f'https://db.netkeiba.com/horse/training/{horse_id}/'
    soup = _get(url)
    if soup is None:
        return []

    training_data = []
    for row in soup.select('table.Training tr')[1:6]:  # 直近5本
        cells = row.find_all('td')
        if len(cells) < 5:
            continue
        try:
            training_data.append({
                'date': cells[0].get_text(strip=True),
                'course': cells[1].get_text(strip=True),
                'style': cells[2].get_text(strip=True),
                'time_raw': cells[3].get_text(strip=True),
                'note': cells[-1].get_text(strip=True) if len(cells) > 5 else '',
            })
        except Exception:
            continue
    return training_data


def evaluate_training(training_list: list[dict]) -> dict:
    """
    調教タイムリストを評価して状態スコアと補正値を返す。

    閾値:
    - 坂路4F: ≤52.0秒 → 絶好調, ≤53.4秒 → 良好, ≥55.0秒 → 普通以下
    - CW6F: ≤80.0秒 → 絶好調, ≤83.0秒 → 標準, ≥85.0秒 → 物足りない
    """
    if not training_list:
        return {'score': 50, 'bonus': 0.0, 'label': '調教データなし', 'message': ''}

    latest = training_list[0]
    course = latest.get('course', '')
    time_raw = latest.get('time_raw', '')

    # タイムのパース（例: '54.3-39.2-12.1' → 最初の数値が全体タイム）
    times = re.findall(r'\d+\.\d+', time_raw)
    if not times:
        return {'score': 50, 'bonus': 0.0, 'label': '調教データなし', 'message': ''}

    main_time = float(times[0])
    last_1f = float(times[-1]) if len(times) >= 2 else None

    # コース別の閾値評価
    if '坂路' in course:
        if main_time <= 52.0:
            score, bonus, label = 90, 0.025, '絶好調（坂路自己ベスト圏）'
        elif main_time <= 53.4:
            score, bonus, label = 75, 0.015, '好仕上がり（坂路基準クリア）'
        elif main_time <= 55.0:
            score, bonus, label = 55, 0.0, '標準的な仕上がり'
        else:
            score, bonus, label = 35, -0.01, '調教物足りない（坂路遅め）'
    elif 'CW' in course or 'ウッド' in course or 'W' == course:
        if main_time <= 80.0:
            score, bonus, label = 90, 0.025, '絶好調（CW自己ベスト圏）'
        elif main_time <= 83.0:
            score, bonus, label = 70, 0.01, '好仕上がり（CW基準クリア）'
        elif main_time <= 85.0:
            score, bonus, label = 50, 0.0, '標準的な仕上がり'
        else:
            score, bonus, label = 35, -0.01, '調教物足りない（CW遅め）'
    else:
        score, bonus, label = 50, 0.0, f'コース判別不能（{course}）'

    # 最終1F補正（12.5秒以下は切れる）
    if last_1f and last_1f <= 12.0:
        bonus += 0.005
        label += '・鋭い終い'

    return {
        'score': score,
        'bonus': round(bonus, 3),
        'label': label,
        'message': f'{course} {time_raw} → {label}',
        'main_time': main_time,
    }


def manual_training_input(horse_name: str) -> dict:
    """
    スクレイピングが失敗した場合のフォールバック用の手動入力テンプレート。
    Streamlit フォーム側でこの構造を参考に入力させる。
    """
    return {
        'horse_name': horse_name,
        'course': '',
        'time_raw': '',
        'evaluated': False,
    }


def fetch_all_training(race_id: str, horse_names: list[str]) -> dict[str, dict]:
    """
    レース内の全出走馬の調教タイムを取得して評価する。
    失敗した馬はデフォルト値を返す。
    """
    horse_ids = get_horse_ids_from_race(race_id)
    result = {}

    for name in horse_names:
        h_id = horse_ids.get(name)
        if not h_id:
            result[name] = {'score': 50, 'bonus': 0.0, 'label': 'ID不明（手動入力推奨）', 'message': ''}
            continue

        training = fetch_training_times(h_id)
        result[name] = evaluate_training(training)

    return result


# ============================================================
# 併せ馬パートナー分析
# ============================================================

@st.cache_data(ttl=3600)
def fetch_training_with_partner(horse_id: str) -> list[dict]:
    """
    調教データを取得し、併せ馬のパートナー名も抽出する。
    netkeibaの調教ページでは「併せ馬」の場合に相手馬名が記載される。

    Returns list of dicts including:
        {date, course, style, time_raw, note, partner_name, is_awase, won_awase}
    """
    url = f'https://db.netkeiba.com/horse/training/{horse_id}/'
    soup = _get(url)
    if soup is None:
        return []

    training_data = []
    for row in soup.select('table.Training tr')[1:6]:
        cells = row.find_all('td')
        if len(cells) < 5:
            continue
        try:
            style_text = cells[2].get_text(strip=True)
            note_text  = cells[-1].get_text(strip=True) if len(cells) > 5 else ''

            # 併せ馬判定：styleか備考に「併」「併せ」が含まれる
            is_awase = '併' in style_text or '併' in note_text

            # パートナー馬名の抽出（例: "馬名（5歳）に勝ち" "〇〇と併せ"）
            partner_name = None
            won_awase    = None  # 併せで勝ったか負けたか

            if is_awase:
                # 「に勝ち」「に先着」→ 自馬が勝ち
                m_win = re.search(r'(.+?)(?:に勝ち|に先着)', note_text or style_text)
                # 「に遅れ」「に負け」→ 相手が勝ち
                m_lose = re.search(r'(.+?)(?:に遅れ|に負け)', note_text or style_text)
                # 「と併せ」→ 引き分け的
                m_tie  = re.search(r'(.+?)と併せ', note_text or style_text)

                if m_win:
                    partner_name = m_win.group(1).strip()
                    won_awase    = True   # 自馬が上回った
                elif m_lose:
                    partner_name = m_lose.group(1).strip()
                    won_awase    = False  # 相手が上回った
                elif m_tie:
                    partner_name = m_tie.group(1).strip()
                    won_awase    = None   # 引き分け

            training_data.append({
                'date':         cells[0].get_text(strip=True),
                'course':       cells[1].get_text(strip=True),
                'style':        style_text,
                'time_raw':     cells[3].get_text(strip=True),
                'note':         note_text,
                'is_awase':     is_awase,
                'partner_name': partner_name,
                'won_awase':    won_awase,
            })
        except Exception:
            continue

    return training_data


@st.cache_data(ttl=900)
def fetch_saturday_winners(saturday_date_str: str) -> list[str]:
    """
    土曜日の全レース勝ち馬名リストを取得する。
    sunday race の前日（土曜）のnetkeibaレース一覧から結果を取得。

    Parameters
    ----------
    saturday_date_str : str  "YYYYMMDD" 形式

    Returns
    -------
    list[str]  勝ち馬名のリスト
    """
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={saturday_date_str}"
    try:
        time.sleep(REQUEST_INTERVAL)
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.content, 'lxml')
    except Exception:
        return []

    JRA_VENUE_CODES = {str(i).zfill(2) for i in range(1, 11)}
    race_ids = []
    for a in soup.select("a[href*='/race/result.html']"):
        href = a.get("href", "")
        m = re.search(r"race_id=(\d{12})", href)
        if m:
            rid = m.group(1)
            if rid[4:6] in JRA_VENUE_CODES:
                race_ids.append(rid)

    winners = []
    for race_id in race_ids[:20]:  # 土曜は最大20レースまで
        try:
            time.sleep(REQUEST_INTERVAL)
            result_url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
            r2 = requests.get(result_url, headers=HEADERS, timeout=12)
            r2.encoding = r2.apparent_encoding
            soup2 = BeautifulSoup(r2.content, 'lxml')

            # 1着馬を取得（着順1番のHorseName）
            for row in soup2.select('table.RaceTable01 tr, table.ResultTable tr'):
                rank_el = row.select_one('.Rank, td:first-child')
                name_el = row.select_one('.Horse_Name a, .HorseName a')
                if rank_el and name_el:
                    rank_text = rank_el.get_text(strip=True)
                    if rank_text == '1':
                        winners.append(name_el.get_text(strip=True))
                        break
        except Exception:
            continue

    return list(set(winners))  # 重複排除


def evaluate_training_partner(
    training_list: list[dict],
    saturday_winners: list[str],
) -> dict:
    """
    最終追い切りの併せ馬パートナーが土曜日の勝ち馬かどうかを判定する。

    Parameters
    ----------
    training_list    : fetch_training_with_partner() の出力
    saturday_winners : fetch_saturday_winners() の出力

    Returns
    -------
    {
        "partner_name":     str | None,
        "partner_won_sat":  bool,       土曜勝ち馬と同一パートナーか
        "won_awase":        bool | None, 併せ調教で勝ったか
        "bonus":            float,
        "label":            str,
        "message":          str,
    }
    """
    empty = {
        "partner_name": None, "partner_won_sat": False,
        "won_awase": None, "bonus": 0.0, "label": "", "message": "",
    }

    if not training_list:
        return empty

    # 最終追い切り（最新）を使う
    latest = training_list[0]
    if not latest.get("is_awase"):
        return {**empty, "message": "最終追い切りは単走"}

    partner = latest.get("partner_name")
    won_awase = latest.get("won_awase")

    if not partner:
        return {**empty, "message": "併せ馬だがパートナー名不明"}

    # 土曜勝ち馬との照合（部分一致も考慮）
    partner_won_sat = any(
        partner in w or w in partner
        for w in saturday_winners
    )

    bonus = 0.0
    label = ""
    message = ""

    if partner_won_sat:
        # 土曜勝ち馬と調教 → 高評価
        if won_awase is True:
            bonus   = 0.04
            label   = "◎◎ 土曜勝ち馬に併せ勝ち"
            message = f"最終追い切りで土曜勝ち馬「{partner}」に先着。仕上がり最上位。"
        elif won_awase is False:
            bonus   = 0.02
            label   = "◎ 土曜勝ち馬と併せ（先着された）"
            message = f"最終追い切りで土曜勝ち馬「{partner}」と調教。負けたが相手が強い。"
        else:
            bonus   = 0.025
            label   = "◎ 土曜勝ち馬と併せ調教"
            message = f"最終追い切りで土曜勝ち馬「{partner}」と調教実施。"
    else:
        # 土曜勝ち馬ではないが、併せ調教で勝ったことは評価
        if won_awase is True:
            bonus   = 0.015
            label   = "○ 併せ調教で先着"
            message = f"最終追い切りで「{partner}」に先着。状態良好。"
        elif won_awase is False:
            bonus   = 0.0
            label   = "△ 併せ調教で先着されず"
            message = f"最終追い切りで「{partner}」に遅れ。"
        else:
            bonus   = 0.005
            label   = "併せ調教実施"
            message = f"「{partner}」と併せ調教。"

    return {
        "partner_name":    partner,
        "partner_won_sat": partner_won_sat,
        "won_awase":       won_awase,
        "bonus":           round(bonus, 4),
        "label":           label,
        "message":         message,
    }


def fetch_all_training_with_partner(
    race_id: str,
    horse_names: list[str],
    saturday_date_str: str = "",
) -> dict[str, dict]:
    """
    全出走馬の調教評価＋土曜勝ち馬との併せ照合を一括実行する。

    Parameters
    ----------
    race_id              : 日曜のレースID
    horse_names          : 出走馬名リスト
    saturday_date_str    : "YYYYMMDD" 土曜日付（空なら前日を自動計算）

    Returns
    -------
    dict[horse_name, {score, bonus, label, message, partner_name, partner_won_sat, ...}]
    """
    from datetime import date, timedelta

    if not saturday_date_str:
        # 自動で前日（土曜）を計算
        today = date.today()
        sat = today - timedelta(days=(today.weekday() - 5) % 7 or 7)
        saturday_date_str = sat.strftime("%Y%m%d")

    # 土曜勝ち馬リストを取得
    saturday_winners = fetch_saturday_winners(saturday_date_str)

    horse_ids = get_horse_ids_from_race(race_id)
    result = {}

    for name in horse_names:
        h_id = horse_ids.get(name)
        if not h_id:
            result[name] = {
                'score': 50, 'bonus': 0.0,
                'label': 'ID不明', 'message': '',
                'partner_name': None, 'partner_won_sat': False,
                'won_awase': None,
            }
            continue

        training = fetch_training_with_partner(h_id)
        base_eval = evaluate_training(training)
        partner_eval = evaluate_training_partner(training, saturday_winners)

        # 調教タイムボーナスと併せ馬ボーナスを合算
        combined_bonus = round(base_eval['bonus'] + partner_eval['bonus'], 4)
        combined_label = base_eval['label']
        if partner_eval['label']:
            combined_label += f" / {partner_eval['label']}"

        result[name] = {
            **base_eval,
            'bonus':           combined_bonus,
            'label':           combined_label,
            'partner_name':    partner_eval['partner_name'],
            'partner_won_sat': partner_eval['partner_won_sat'],
            'won_awase':       partner_eval['won_awase'],
            'partner_message': partner_eval['message'],
        }

    return result
