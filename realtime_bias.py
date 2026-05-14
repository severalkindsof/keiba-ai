"""
リアルタイム馬場バイアス収集モジュール。

【X（旧Twitter）の取得実現可能性について】
- X本体: SPA（React）のため直接スクレイピング不可
- nitter: 2024年以降ほぼ全インスタンスが停止
- X公式API: 有料（月$100〜）

【代替アプローチ（3段階フォールバック）】
1. netkeiba の当日レース情報から馬場状態（良/稍重/重/不良）を自動取得 ← 確実
2. ユーザーがXで「東京 馬場バイアス」などを検索してテキストを貼り付け
   → AIキーワード解析で内容を構造化（内前有利/外差し等を抽出）
3. 手動でバイアス方向を選択するUI

この設計により「取れる情報は自動、取れない情報はUIでアシスト」を実現する。
"""
import re
import requests
from bs4 import BeautifulSoup
import streamlit as st
from datetime import date
import time

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}

# ============================================================
# 馬場バイアスの構造定義
# ============================================================

BIAS_TYPES = {
    'inner_speed': {'label': '内前有利', 'desc': '内枠の逃げ・先行馬が有利。差しは届きにくい。',
                    'inner_bonus': 0.03, 'speed_bonus': 0.02},
    'outer_diff':  {'label': '外差し有利', 'desc': '外から差し馬が台頭しやすい。先行は苦しい。',
                    'inner_bonus': -0.02, 'speed_bonus': -0.01},
    'speed_track': {'label': '時計速い（高速馬場）', 'desc': '速い時計が出やすい。スピード型有利。',
                    'inner_bonus': 0.0, 'speed_bonus': 0.02},
    'heavy_track': {'label': '重馬場バイアス', 'desc': '道悪得意な馬・パワー型有利。',
                    'inner_bonus': 0.0, 'speed_bonus': -0.01},
    'neutral':     {'label': 'フラット（バイアスなし）', 'desc': 'バイアスなし、実力通りの決着が多い。',
                    'inner_bonus': 0.0, 'speed_bonus': 0.0},
    'unknown':     {'label': '不明', 'desc': 'バイアス未確認',
                    'inner_bonus': 0.0, 'speed_bonus': 0.0},
}

# キーワード → バイアスタイプのマッピング（テキスト解析用）
KEYWORD_MAP = [
    (['内前', '内有利', '先行有利', '逃げ有利', '前残り', '内が有利'], 'inner_speed'),
    (['外差し', '差し有利', '外が有利', '差し決まる', '追込', '外伸び'], 'outer_diff'),
    (['時計速い', '高速馬場', '速い馬場', 'タイム速', '超高速'], 'speed_track'),
    (['道悪', '重馬場', '不良馬場', '馬場悪', 'パワー型'], 'heavy_track'),
    (['フラット', 'バイアスなし', '公平', 'フェア', '差なし'], 'neutral'),
]


# ============================================================
# Step 1: netkeiba から馬場状態を自動取得
# ============================================================

@st.cache_data(ttl=900)
def fetch_track_condition_from_netkeiba(race_id: str) -> dict:
    """
    netkeibaの出馬表ページから馬場状態（良/稍重/重/不良）を取得する。
    確認済み: RaceData01 クラスに含まれる。
    """
    url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    try:
        time.sleep(2)
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.content, 'lxml')

        race_data = soup.select_one('.RaceData01')
        if not race_data:
            return {'condition': '不明', 'venue': '不明', 'surface': '不明'}

        text = race_data.get_text()
        surface = '芝' if '芝' in text else 'ダート' if 'ダート' in text else '不明'
        condition_m = re.search(r'(不良|重|稍重|良)', text)
        condition = condition_m.group(1) if condition_m else '不明'

        # 会場はrace_idから
        venue_code = race_id[4:6] if len(race_id) >= 6 else '00'
        venue_map = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
                     '05': '東京', '06': '中山', '07': '中京', '08': '京都',
                     '09': '阪神', '10': '小倉'}
        venue = venue_map.get(venue_code, '不明')

        return {'condition': condition, 'venue': venue, 'surface': surface}
    except Exception:
        return {'condition': '不明', 'venue': '不明', 'surface': '不明'}


# ============================================================
# Step 2: ユーザー貼り付けテキストからバイアス解析
# ============================================================

def parse_bias_from_text(text: str) -> dict:
    """
    X検索結果や競馬ブログのテキストを解析してバイアスタイプを抽出する。
    ユーザーがコピペした内容をNLP的にパースする。
    """
    if not text or len(text.strip()) < 5:
        return {'bias_type': 'unknown', 'confidence': 0, 'matched_keywords': []}

    text_lower = text.lower()
    scores = {bt: 0 for bt in BIAS_TYPES}
    matched = []

    for keywords, bias_type in KEYWORD_MAP:
        for kw in keywords:
            if kw in text:
                scores[bias_type] += 1
                matched.append(kw)

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    if best_score == 0:
        return {'bias_type': 'unknown', 'confidence': 0, 'matched_keywords': []}

    confidence = min(100, best_score * 25)  # 1キーワード=25点
    return {
        'bias_type': best_type,
        'confidence': confidence,
        'matched_keywords': matched,
        'label': BIAS_TYPES[best_type]['label'],
    }


# ============================================================
# バイアスを各馬の補正値に変換
# ============================================================

def calc_bias_bonus(
    bias_type: str,
    gate: int,
    total_horses: int,
    running_style: str,
    surface: str,
) -> float:
    """
    馬場バイアスと馬の特性を組み合わせて勝率補正値を返す。
    """
    if bias_type not in BIAS_TYPES or bias_type == 'unknown':
        return 0.0

    bias = BIAS_TYPES[bias_type]
    inner_bonus = bias['inner_bonus']
    speed_bonus = bias['speed_bonus']

    # 枠順による内外判定
    ratio = gate / max(total_horses, 1)
    is_inner = ratio <= 0.35
    is_outer = ratio >= 0.65

    # 脚質による補正
    is_speed = running_style in ('逃げ', '先行')
    is_closer = running_style in ('差し・追込',)

    pos_bonus = 0.0
    if inner_bonus > 0 and is_inner:
        pos_bonus += inner_bonus
    elif inner_bonus < 0 and is_outer:
        pos_bonus += abs(inner_bonus)  # 外差し有利なら外枠にプラス
    elif inner_bonus > 0 and is_outer:
        pos_bonus += inner_bonus  # 内有利なら外枠にマイナス（inner_bonusが正→外はマイナス）
        pos_bonus = -abs(inner_bonus)

    style_bonus = 0.0
    if speed_bonus > 0 and is_speed:
        style_bonus += speed_bonus
    elif speed_bonus < 0 and is_speed:
        style_bonus += speed_bonus
    elif speed_bonus > 0 and is_closer:
        style_bonus -= speed_bonus * 0.5

    return round(pos_bonus + style_bonus, 3)


def apply_realtime_bias(
    horses: list[dict],
    bias_type: str,
) -> list[dict]:
    """出走馬全頭にリアルタイムバイアス補正を付与する。"""
    result = []
    total = len(horses)
    for h in horses:
        h2 = dict(h)
        gate = int(h2.get('gate', h2.get('horse_no', 1)))
        style = h2.get('running_style', '不明')
        surface = h2.get('surface', '芝')

        bonus = calc_bias_bonus(bias_type, gate, total, style, surface)
        h2['realtime_bias_type'] = bias_type
        h2['realtime_bias_label'] = BIAS_TYPES.get(bias_type, {}).get('label', '不明')
        h2['realtime_bias_bonus'] = bonus
        result.append(h2)
    return result


# ============================================================
# Streamlit UI コンポーネント（app.py から呼び出す）
# ============================================================

def render_bias_input_panel(race_id: str | None = None) -> str:
    """
    馬場バイアス入力パネルをレンダリングし、選択されたバイアスタイプを返す。
    """
    st.markdown('#### 🌿 当日馬場バイアス')

    # Step1: 自動取得
    auto_condition = {}
    if race_id:
        with st.spinner('netkeiba から馬場状態を取得中...'):
            auto_condition = fetch_track_condition_from_netkeiba(race_id)
        if auto_condition.get('condition') != '不明':
            st.success(
                f"**{auto_condition['venue']} {auto_condition['surface']}** "
                f"馬場: **{auto_condition['condition']}**（自動取得）"
            )

    # Step2: Xテキスト貼り付け解析
    with st.expander('📋 X（旧Twitter）の投稿を貼り付けてバイアスを自動解析'):
        st.caption(
            'X で「東京 馬場バイアス」「中山 内前有利」などで検索し、'
            '関連するツイートをコピーしてここに貼り付けてください。'
        )
        pasted_text = st.text_area(
            'X検索結果・競馬ブログのテキスト',
            height=120,
            placeholder='例: 「今日の東京は明らかに内前有利。先行馬が残りまくっている。外差しは全く届かない。」',
            key='bias_text_input',
        )
        if pasted_text:
            parsed = parse_bias_from_text(pasted_text)
            if parsed['confidence'] > 0:
                st.info(
                    f"解析結果: **{parsed['label']}** "
                    f"（信頼度 {parsed['confidence']}%、"
                    f"キーワード: {', '.join(parsed['matched_keywords'][:3])}）"
                )
            else:
                st.warning('バイアスキーワードが検出できませんでした。下で手動選択してください。')

    # Step3: 手動選択（最終フォールバック）
    bias_labels = {bt: info['label'] for bt, info in BIAS_TYPES.items()}
    selected_label = st.selectbox(
        '馬場バイアスを選択（手動・最終確認）',
        list(bias_labels.values()),
        index=list(bias_labels.keys()).index('neutral'),
        key='bias_manual_select',
    )
    selected_type = next(bt for bt, info in BIAS_TYPES.items() if info['label'] == selected_label)

    # テキスト解析結果があればそちらを優先表示
    if pasted_text and 'parsed' in dir() and parsed.get('confidence', 0) > 50:
        return parsed['bias_type']
    return selected_type
