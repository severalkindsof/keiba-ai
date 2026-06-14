"""
季節・気温・馬場状態（芝/ダート逆転）適性モジュール。

データソース:
- Kaggleデータの date 列から月・季節を抽出
- ダートの馬場状態（重で速くなる）は静的ルールで実装
- 個馬の季節別成績は過去データから算出
"""
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date


# ---- 季節定義 ---- #
SEASON_MAP = {
    12: '冬', 1: '冬', 2: '冬',
    3: '春', 4: '春', 5: '春',
    6: '夏', 7: '夏', 8: '夏',
    9: '秋', 10: '秋', 11: '秋',
}

MONTH_TO_SEASON = {m: s for m, s in SEASON_MAP.items()}


def get_current_season(race_date: date | None = None) -> str:
    d = race_date or date.today()
    return MONTH_TO_SEASON.get(d.month, '春')


# ---- ダート馬場状態の逆転ロジック ---- #
# 芝: 重/不良 → 速度低下（馬によって得手不得手）
# ダート: 重/不良 → 水分で締まり速くなる（スピード馬有利）
DIRT_CONDITION_BONUS = {
    '良':   0.0,
    '稍重': 0.005,
    '重':   0.015,   # ダート重は速くなる
    '不良': 0.02,
}

TURF_CONDITION_PENALTY = {  # 重馬場を苦手とする馬へのペナルティ
    '良':   0.0,
    '稍重': -0.005,
    '重':   -0.015,
    '不良': -0.02,
}

def get_condition_surface_bonus(surface: str, condition: str) -> float:
    """
    芝/ダート × 馬場状態の補正値。
    ダートの重馬場は「速くなる」ため、ダート馬に有利。
    """
    if surface == 'ダート':
        return DIRT_CONDITION_BONUS.get(condition, 0.0)
    return 0.0  # 芝の補正は個馬ごとに計算


# ---- 個馬の季節別成績 ---- #

def calc_horse_season_stats(history: pd.DataFrame) -> dict:
    """
    過去成績から各季節の勝率・複勝率を算出する。
    """
    if history.empty or 'date' not in history.columns or 'win_flag' not in history.columns:
        return {}

    hist = history.copy()
    hist['date'] = pd.to_datetime(hist['date'], errors='coerce')
    hist = hist.dropna(subset=['date'])
    hist['season'] = hist['date'].dt.month.map(MONTH_TO_SEASON)

    stats = {}
    for season in ['春', '夏', '秋', '冬']:
        s_df = hist[hist['season'] == season]
        if len(s_df) < 2:
            continue
        wr = float(s_df['win_flag'].mean())
        pr = float(s_df['place_flag'].mean()) if 'place_flag' in s_df.columns else wr * 3
        stats[season] = {'win_rate': round(wr, 3), 'place_rate': round(pr, 3), 'races': len(s_df)}
    return stats


def get_season_bonus(
    history: pd.DataFrame,
    current_season: str,
) -> dict:
    """
    今の季節が得意かどうかを過去成績から評価する。
    """
    stats = calc_horse_season_stats(history)
    if not stats or current_season not in stats:
        return {'bonus': 0.0, 'message': '季節データなし', 'label': ''}

    season_wr = stats[current_season]['win_rate']
    all_wrs = [v['win_rate'] for v in stats.values()]
    overall_avg = np.mean(all_wrs) if all_wrs else 0.1
    deviation = season_wr - overall_avg

    if deviation >= 0.08:
        label = f'◎ {current_season}が得意'
        bonus = 0.02
    elif deviation >= 0.03:
        label = f'○ {current_season}でそこそこ'
        bonus = 0.01
    elif deviation <= -0.08:
        label = f'✕ {current_season}が苦手'
        bonus = -0.02
    elif deviation <= -0.03:
        label = f'▲ {current_season}はやや苦手'
        bonus = -0.01
    else:
        label = '季節影響なし'
        bonus = 0.0

    races = stats[current_season]['races']
    msg = f'{current_season}の成績: {races}走 勝率{season_wr*100:.0f}%（全季節平均{overall_avg*100:.0f}%）'
    return {'bonus': round(bonus, 3), 'message': msg, 'label': label}


# ---- 馬場状態（芝）の個馬適性 ---- #

def get_track_condition_aptitude(
    history: pd.DataFrame,
    surface: str,
    condition: str,
) -> dict:
    """
    過去の馬場状態別成績から、今回の馬場適性を評価する。
    """
    if history.empty or 'track_condition' not in history.columns:
        return {'bonus': 0.0, 'label': ''}

    hist = history.copy()
    condition_map = {'稍重': '重', '不良': '重'}  # サンプル増のために稍重・不良・重を統合
    hist['cond_group'] = hist['track_condition'].map(lambda x: condition_map.get(x, x))
    current_group = condition_map.get(condition, condition)

    cond_hist = hist[(hist.get('surface', pd.Series()) == surface) & (hist['cond_group'] == current_group)] \
        if 'surface' in hist.columns else hist[hist['cond_group'] == current_group]

    if len(cond_hist) < 2:
        return {'bonus': 0.0, 'label': '実績なし（この馬場状態での出走少）'}

    wr = float(cond_hist['win_flag'].mean()) if 'win_flag' in cond_hist.columns else 0.1
    overall_wr = float(hist['win_flag'].mean()) if 'win_flag' in hist.columns else 0.1
    dev = wr - overall_wr

    if dev >= 0.1:
        return {'bonus': 0.015, 'label': f'◎ {condition}馬場が得意'}
    elif dev >= 0.05:
        return {'bonus': 0.008, 'label': f'○ {condition}馬場でまずまず'}
    elif dev <= -0.1:
        return {'bonus': -0.015, 'label': f'✕ {condition}馬場が苦手'}
    elif dev <= -0.05:
        return {'bonus': -0.008, 'label': f'▲ {condition}馬場はやや苦手'}
    return {'bonus': 0.0, 'label': '馬場適性標準'}


# ---- 芝↔ダート変わり分析 ---- #

def analyze_surface_change(
    history: pd.DataFrame,
    current_surface: str,
) -> dict:
    """
    前走から今回の路面変更を分析する。

    データに基づく知見:
    - ダート→芝: 母父サンデー系など芝血統を持つ馬が初芝で覚醒するケースが穴になりやすい
      市場がダート馬として評価しているため、芝適性が過小評価されオッズが高くなる。
    - 芝→ダート: 芝で実績がなく、パワー型血統の場合は好走の可能性あり

    構造的穴馬としてのシグナルになるか判定する。
    """
    if history.empty or 'surface' not in history.columns:
        return {'signal': '初出走', 'bonus': 0.0, 'is_surface_change': False, 'message': ''}

    prev_surface = str(history.iloc[0].get('surface', ''))
    if not prev_surface or prev_surface == current_surface:
        return {'signal': '同路面', 'bonus': 0.0, 'is_surface_change': False, 'message': ''}

    # 変わり方向と過去の当該路面での実績
    direction = f'{prev_surface}→{current_surface}'
    current_surf_hist = history[history['surface'] == current_surface]

    if direction == 'ダート→芝':
        if len(current_surf_hist) == 0:
            # 初芝：市場が評価しきれていない → 構造的穴になりやすい
            return {
                'signal': '初芝転向',
                'bonus': 0.015,
                'is_surface_change': True,
                'message': '初芝転向：市場がダート馬として評価 → 芝適性が過小評価されやすい穴候補',
            }
        elif len(current_surf_hist) >= 2:
            # 芝の実績あり→芝適性を確認
            wr = float(current_surf_hist['win_flag'].mean()) if 'win_flag' in current_surf_hist.columns else 0
            if wr >= 0.15:
                return {
                    'signal': 'ダート→芝（実績あり）',
                    'bonus': 0.02,
                    'is_surface_change': True,
                    'message': f'◎ ダート→芝（芝勝率{wr*100:.0f}%）：芝での実績も十分',
                }
    elif direction == '芝→ダート':
        if len(current_surf_hist) == 0:
            # 初ダート：芝で実績がなかった馬の覚醒パターン
            prev_surf_wr = float(history[history['surface'] == prev_surface]['win_flag'].mean()) \
                if 'win_flag' in history.columns else 0
            if prev_surf_wr < 0.1:
                return {
                    'signal': '初ダート転向（芝で実績薄）',
                    'bonus': 0.01,
                    'is_surface_change': True,
                    'message': '初ダート：芝で結果が出なかった馬の一変の可能性あり',
                }

    return {
        'signal': f'路面変更（{direction}）',
        'bonus': 0.005,
        'is_surface_change': True,
        'message': f'路面変更（{direction}）：適性要確認',
    }


# ---- 全馬への適用 ---- #

def apply_season_climate(
    horses: list[dict],
    df_hist: pd.DataFrame,
    surface: str,
    condition: str,
    race_date: date | None = None,
) -> list[dict]:
    """出走馬全頭に季節・馬場状態適性を付与する。"""
    current_season = get_current_season(race_date)
    surface_cond_bonus = get_condition_surface_bonus(surface, condition)

    result = []
    for h in horses:
        h2 = dict(h)
        name = h2.get('horse_name', '')

        hist = pd.DataFrame()
        if not df_hist.empty and 'horse_name' in df_hist.columns:
            hist = df_hist[df_hist['horse_name'] == name]
            if 'date' in df_hist.columns:
                hist = hist.sort_values('date', ascending=False)
            hist = hist.head(20)

        season_info  = get_season_bonus(hist, current_season)
        cond_info    = get_track_condition_aptitude(hist, surface, condition)
        surface_info = analyze_surface_change(hist, surface)

        h2['current_season']      = current_season
        h2['season_bonus']        = season_info['bonus']
        h2['season_label']        = season_info['label']
        h2['season_message']      = season_info['message']
        h2['condition_apt_bonus'] = cond_info['bonus'] + surface_cond_bonus
        h2['condition_apt_label'] = cond_info['label']
        h2['surface_change_signal']  = surface_info['signal']
        h2['surface_change_bonus']   = surface_info['bonus']
        h2['surface_change_message'] = surface_info['message']
        h2['is_surface_change']      = surface_info['is_surface_change']
        result.append(h2)
    return result
