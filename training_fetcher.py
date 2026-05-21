"""
training_fetcher.py
netkeibaの調教ページから追い切り情報を取得・評価するモジュール。

ワークフロー：
1. fetch_race_training(race_id) → 全馬の調教データを取得
2. evaluate_training(training_data) → 調教スコアを算出
3. 結果をsessions/training_cache/{race_id}.jsonに保存
"""
import time
import json
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import streamlit as st

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

CACHE_DIR = Path(__file__).parent / "sessions" / "training_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# ポジティブ/ネガティブキーワード辞書
# ============================================================
POSITIVE_KEYWORDS = [
    "動き良", "手応え十分", "絶好", "状態良好", "力強", "伸び良",
    "自己ベスト", "好時計", "余裕", "軽快", "自信", "仕上がり万全",
    "抜群", "最高", "好調", "いい動き", "鋭い", "反応良",
    "本追い切り", "強め", "一杯に追う",
]

NEGATIVE_KEYWORDS = [
    "物足りない", "やや", "もう一息", "課題", "重め", "消耗",
    "慎重", "様子見", "軽め", "馬なり", "普通", "並",
    "不安", "心配", "難しい", "今ひとつ",
]

# 追い切り強度スコア
WORK_INTENSITY = {
    "一杯":   0.15,
    "強め":   0.10,
    "馬なり": 0.0,
    "軽め":   -0.05,
}

# コースボーナス
COURSE_BONUS = {
    "坂路":  0.05,   # 坂路好時計は能力の証
    "CW":   0.03,
    "芝":   0.05,
    "DP":   0.02,
    "ウッド": 0.03,
}


def _get(url: str, sleep: float = 2.0):
    time.sleep(sleep)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"[training] GET失敗: {url} → {e}")
        return None


def fetch_race_training(race_id: str) -> dict:
    """
    netkeibaの調教ページから全馬の追い切り情報を取得する。
    race_id: 12桁（例: 202605250511）
    """
    cache_path = CACHE_DIR / f"{race_id}.json"
    # キャッシュがあれば使用（3日以内）
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(data.get("fetched_at", "2000-01-01"))
            if (datetime.now() - fetched).days < 3:
                print(f"[training] キャッシュ使用: {race_id}")
                return data
        except Exception:
            pass

    print(f"[training] 調教データ取得中: {race_id}")
    result = {
        "race_id":    race_id,
        "fetched_at": datetime.now().isoformat(),
        "horses":     {},
    }

    # ① 調教タイムページ
    url = f"https://race.netkeiba.com/race/oikiri.html?race_id={race_id}"
    soup = _get(url)
    if soup:
        _parse_oikiri_page(soup, result)

    # ② 馬別調教ページ（horse_id が必要なので shutuba から取得）
    _enrich_from_shutuba(race_id, result)

    cache_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )
    print(f"[training] 取得完了: {len(result['horses'])}頭")
    return result


def _parse_oikiri_page(soup: BeautifulSoup, result: dict):
    """調教ページから追い切り時計・コース・強度を取得"""
    # 馬名と調教データを探す
    for row in soup.select("tr.OikiriTr, tr[class*='Horse']"):
        try:
            name_el = row.select_one(".HorseName, td.horsename, .HorseNameSmall")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            if not name:
                continue

            tds = row.find_all("td")
            if len(tds) < 5:
                continue

            # 一般的な列順: 馬名/コース/強度/タイム/評価
            texts = [td.get_text(strip=True) for td in tds]

            horse_data = result["horses"].setdefault(name, {
                "workouts": [], "comments": [], "evaluation": "不明"
            })

            # コース検出
            course = ""
            for t in texts:
                for c in ["坂路", "CW", "芝", "DP", "ウッド", "W"]:
                    if c in t:
                        course = c
                        break

            # 強度検出
            intensity = "馬なり"
            for t in texts:
                if "一杯" in t:
                    intensity = "一杯"
                    break
                elif "強め" in t:
                    intensity = "強め"
                    break
                elif "軽め" in t:
                    intensity = "軽め"
                    break

            # タイム（数字5桁前後のパターン）
            time_str = ""
            for t in texts:
                m = re.search(r"\d{1,2}\.\d", t)
                if m:
                    time_str = m.group()
                    break

            horse_data["workouts"].append({
                "course":    course,
                "intensity": intensity,
                "time":      time_str,
            })
        except Exception:
            continue


def _enrich_from_shutuba(race_id: str, result: dict):
    """出馬表ページから馬IDを取得し、調教コメントを追加"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    soup = _get(url, sleep=1.5)
    if not soup:
        return

    for row in soup.select("tr.HorseList"):
        try:
            name_el = row.select_one(".HorseName a")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            m = re.search(r"/horse/(\w+)", name_el.get("href", ""))
            if not m:
                continue
            horse_id = m.group(1)

            horse_data = result["horses"].setdefault(name, {
                "workouts": [], "comments": [], "evaluation": "不明"
            })
            horse_data["horse_id"] = horse_id

            # 調教コメント列
            comment_el = row.select_one(".TrainingComment, td.comment")
            if comment_el:
                comment = comment_el.get_text(strip=True)
                if comment:
                    horse_data["comments"].append(comment)
        except Exception:
            continue


def evaluate_training(horse_name: str, training_data: dict) -> dict:
    """
    調教データから追い切りスコアを算出する。
    Returns: {"score": float, "label": str, "details": str}
    """
    horses = training_data.get("horses", {})

    # 名前の部分一致検索
    matched = None
    for key in horses:
        if key.strip() == horse_name.strip() or horse_name.strip() in key or key in horse_name.strip():
            matched = horses[key]
            break

    if not matched:
        return {"score": 0.0, "label": "調教データなし", "details": ""}

    workouts  = matched.get("workouts", [])
    comments  = matched.get("comments", [])
    score     = 0.0
    detail_parts = []

    # 最新の追い切りを評価
    if workouts:
        latest = workouts[-1]
        course    = latest.get("course", "")
        intensity = latest.get("intensity", "馬なり")
        time_str  = latest.get("time", "")

        # 強度ボーナス
        int_bonus = WORK_INTENSITY.get(intensity, 0.0)
        score += int_bonus
        if int_bonus > 0:
            detail_parts.append(f"{intensity}({course})")

        # コースボーナス
        c_bonus = COURSE_BONUS.get(course, 0.0)
        score += c_bonus

        if time_str:
            detail_parts.append(f"⏱{time_str}秒")

    # コメント評価
    all_comments = " ".join(comments)
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in all_comments)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in all_comments)

    if pos_count > neg_count:
        score += 0.10 * min(pos_count, 3)
        detail_parts.append(f"コメント◎")
    elif neg_count > pos_count:
        score -= 0.08 * min(neg_count, 2)
        detail_parts.append(f"コメント△")

    # ラベル
    if score >= 0.20:
        label = "調教◎（状態良好）"
    elif score >= 0.10:
        label = "調教○（普通以上）"
    elif score >= 0.0:
        label = "調教△（普通）"
    else:
        label = "調教▲（やや物足りない）"

    return {
        "score":   round(score, 3),
        "label":   label,
        "details": " / ".join(detail_parts),
    }


def get_cached_training(race_id: str) -> dict | None:
    """キャッシュされた調教データを返す（なければNone）"""
    cache_path = CACHE_DIR / f"{race_id}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_cached_races() -> list[str]:
    """調教データがキャッシュされているレースIDのリスト"""
    return [p.stem for p in CACHE_DIR.glob("*.json")]
