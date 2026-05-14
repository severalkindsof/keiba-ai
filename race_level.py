"""
前走レースレベル評価＆ラップ適性モジュール。

【前走レースレベル】
- 同レースの他馬の平均オッズ（低いほど強い相手）から「前走の質」を測定
- 強い相手に惨敗 → 能力的には過小評価されている可能性
- 弱い相手に圧勝 → 次走格上では過大評価のリスク

【ラップ適性ペース帯】
- 過去走の前半3F・後半3Fから「得意ペース帯」を算出
- 今回の予測ペースとの一致度で補正値を決定
- データ: Kaggleの time 列から前後3F推定、または直接の前後3F列
"""
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# 前走レースレベル評価
# ============================================================

@st.cache_data(ttl=3600)
def build_race_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    同一レース内の全馬の平均オッズから「レースレベルスコア」を構築する。
    低い平均オッズ = 強い馬が多い = ハイレベルレース

    race_id がない場合は date + race_name で代用する。
    """
    if df.empty or 'odds' not in df.columns:
        return pd.DataFrame()

    # レース識別キーを作成
    if 'race_id' in df.columns:
        group_key = 'race_id'
    elif 'race_name' in df.columns and 'date' in df.columns:
        df = df.copy()
        df['_race_key'] = df['date'].astype(str) + '_' + df['race_name'].astype(str)
        group_key = '_race_key'
    else:
        return pd.DataFrame()

    df2 = df.dropna(subset=['odds']).copy()
    df2['odds'] = pd.to_numeric(df2['odds'], errors='coerce')

    # レース別の平均オッズと「レベルスコア」を算出
    level = (
        df2.groupby(group_key)
        .agg(avg_odds=('odds', 'mean'), n_horses=('odds', 'count'))
        .reset_index()
    )
    # 平均オッズが低い = 人気馬が多い = レベルが高い
    # 正規化: avg_odds を 0(レベル低)〜1(レベル高) のスコアに変換
    level['level_score'] = 1.0 - (level['avg_odds'] - level['avg_odds'].min()) / \
                           (level['avg_odds'].max() - level['avg_odds'].min() + 0.01)
    return level


def get_prev_race_level_bonus(
    history: pd.DataFrame,
    race_level_table: pd.DataFrame,
) -> dict:
    """
    前走のレースレベルと着順を組み合わせて「実力補正値」を返す。

    パターン:
    - 高レベル + 惨敗 → 実力は評価より高い可能性 → プラス補正
    - 低レベル + 好走 → 次走格上は過大評価リスク → マイナス補正
    """
    if history.empty or race_level_table.empty:
        return {'bonus': 0.0, 'label': '', 'message': ''}

    prev = history.iloc[0]
    prev_rank = pd.to_numeric(prev.get('rank', 99), errors='coerce')
    if pd.isna(prev_rank):
        return {'bonus': 0.0, 'label': '', 'message': ''}

    # 前走のレベルスコアを取得
    level_score = _get_race_level_score(prev, race_level_table)

    if level_score is None:
        return {'bonus': 0.0, 'label': 'レベルデータなし', 'message': ''}

    # 高レベルで惨敗 → 実力者が競走で負けた可能性
    if level_score >= 0.7 and prev_rank >= 6:
        bonus = 0.02
        label = f'◎ 強豪相手に{int(prev_rank)}着 → 実力過小評価の可能性'
        msg = f'前走レベルスコア{level_score:.2f}（ハイレベル）で{int(prev_rank)}着。今走は格下なら有利。'

    # 高レベルで好走 → 実力は本物
    elif level_score >= 0.7 and prev_rank <= 3:
        bonus = 0.015
        label = '◎ 強豪相手に好走 → 実力証明済み'
        msg = f'前走レベルスコア{level_score:.2f}（ハイレベル）で{int(prev_rank)}着。実力は確か。'

    # 低レベルで好走 → 過大評価リスク
    elif level_score <= 0.3 and prev_rank <= 2:
        bonus = -0.015
        label = '▼ 弱い相手に好走 → 格上は割引'
        msg = f'前走レベルスコア{level_score:.2f}（低レベル）で{int(prev_rank)}着。格上での再現性に疑問。'

    else:
        bonus = 0.0
        label = '標準レベル'
        msg = f'前走レベルスコア{level_score:.2f}（標準）'

    return {'bonus': round(bonus, 3), 'label': label, 'message': msg}


def _get_race_level_score(prev_row: pd.Series, race_level_table: pd.DataFrame) -> float | None:
    """前走のレースIDまたはrace_keyからレベルスコアを引く。"""
    if race_level_table.empty:
        return None
    key_col = race_level_table.columns[0]  # race_id or _race_key

    if 'race_id' in prev_row.index and 'race_id' in race_level_table.columns:
        row = race_level_table[race_level_table['race_id'] == prev_row['race_id']]
    elif '_race_key' in race_level_table.columns:
        key_val = str(prev_row.get('date', '')) + '_' + str(prev_row.get('race_name', ''))
        row = race_level_table[race_level_table['_race_key'] == key_val]
    else:
        return None

    if row.empty:
        return None
    return float(row['level_score'].iloc[0])


# ============================================================
# ラップ適性ペース帯
# ============================================================

def estimate_lap_style(history: pd.DataFrame) -> dict:
    """
    過去成績の前後3Fタイムから「得意ペース帯」を推定する。

    前半3F（テン3F）と後半3F（上がり3F）の差分から:
    - 前傾（前半速）: テン型 = 逃げ先行が得意
    - 後傾（後半速）: 差し型 = スロー→ロングスパート or ハイ→差し
    - 均等: バランス型
    """
    result = {'preferred_pace': '不明', 'ten_avg': None, 'agari_avg': None, 'message': ''}

    if history.empty:
        return result

    # 前半3F: Kaggleでは 'first_3f', '前半3F', 'time'列から推定
    ten_col = next((c for c in ['first_3f', '前半3F', 'ten3f'] if c in history.columns), None)
    agari_col = next((c for c in ['last_3f', 'last3f', '上がり3F'] if c in history.columns), None)

    ten_vals, agari_vals = [], []

    if ten_col:
        ten_vals = pd.to_numeric(history[ten_col], errors='coerce').dropna().tolist()
    if agari_col:
        agari_vals = pd.to_numeric(history[agari_col], errors='coerce').dropna().tolist()

    if not agari_vals:
        return result

    agari_avg = np.mean(agari_vals)
    result['agari_avg'] = round(agari_avg, 2)

    if ten_vals:
        ten_avg = np.mean(ten_vals)
        result['ten_avg'] = round(ten_avg, 2)
        diff = ten_avg - agari_avg  # 正: 後半の方が速い(差し型), 負: 前半速(先行型)

        if diff >= 2.0:
            result['preferred_pace'] = 'スロー向き（溜め→爆発型）'
        elif diff >= 0.5:
            result['preferred_pace'] = 'ミドル〜スロー向き'
        elif diff <= -2.0:
            result['preferred_pace'] = 'ハイペース向き（粘り型）'
        elif diff <= -0.5:
            result['preferred_pace'] = 'ミドル〜ハイ向き'
        else:
            result['preferred_pace'] = 'オールラウンド型'
    else:
        # 上がり3Fのみで推定
        if agari_avg <= 34.0:
            result['preferred_pace'] = '差し・末脚型（スロー有利）'
        elif agari_avg >= 36.5:
            result['preferred_pace'] = '先行型（ハイペース耐性あり）'
        else:
            result['preferred_pace'] = 'バランス型'

    result['message'] = f"上がり平均{agari_avg:.1f}秒 → {result['preferred_pace']}"
    return result


def get_lap_pace_match_bonus(
    preferred_pace: str,
    predicted_pace: str,
) -> float:
    """
    馬の得意ペース帯 × 今回の予測ペース の一致度から補正値を返す。
    """
    MATCH_TABLE = {
        # (得意ペース, 予測ペース): 補正値
        ('スロー向き（溜め→爆発型）', 'スローペース'): 0.025,
        ('スロー向き（溜め→爆発型）', 'ミドル〜ハイ'): -0.01,
        ('スロー向き（溜め→爆発型）', 'ハイペース'): -0.02,
        ('ミドル〜スロー向き', 'スローペース'): 0.015,
        ('ミドル〜スロー向き', 'ミドル'): 0.01,
        ('ハイペース向き（粘り型）', 'ハイペース'): 0.025,
        ('ハイペース向き（粘り型）', 'ミドル〜ハイ'): 0.015,
        ('ハイペース向き（粘り型）', 'スローペース'): -0.015,
        ('ミドル〜ハイ向き', 'ハイペース'): 0.01,
        ('ミドル〜ハイ向き', 'ミドル〜ハイ'): 0.01,
        ('差し・末脚型（スロー有利）', 'スローペース'): 0.02,
        ('差し・末脚型（スロー有利）', 'ハイペース'): 0.01,  # ハイでも差しは来る
        ('先行型（ハイペース耐性あり）', 'ハイペース'): 0.015,
        ('先行型（ハイペース耐性あり）', 'スローペース'): 0.02,  # スローなら逃げ切り
    }
    return MATCH_TABLE.get((preferred_pace, predicted_pace), 0.0)


# ---- 全馬への適用 ---- #

def apply_race_level_and_lap(
    horses: list[dict],
    df_hist: pd.DataFrame,
    race_level_table: pd.DataFrame,
    predicted_pace: str = 'ミドル',
) -> list[dict]:
    """出走馬全頭に前走レースレベル・ラップ適性補正を付与する。"""
    result = []
    for h in horses:
        h2 = dict(h)
        name = h2.get('horse_name', '')

        hist = pd.DataFrame()
        if not df_hist.empty and 'horse_name' in df_hist.columns:
            hist = df_hist[df_hist['horse_name'] == name]
            if 'date' in df_hist.columns:
                hist = hist.sort_values('date', ascending=False)
            hist = hist.head(10)

        # 前走レースレベル
        level_info = get_prev_race_level_bonus(hist, race_level_table)
        h2['race_level_bonus'] = level_info['bonus']
        h2['race_level_label'] = level_info['label']
        h2['race_level_message'] = level_info['message']

        # ラップ適性
        lap_style = estimate_lap_style(hist)
        preferred_pace = lap_style['preferred_pace']
        lap_bonus = get_lap_pace_match_bonus(preferred_pace, predicted_pace)

        h2['preferred_pace'] = preferred_pace
        h2['lap_agari_avg'] = lap_style.get('agari_avg')
        h2['lap_bonus'] = lap_bonus
        h2['lap_message'] = lap_style['message']
        result.append(h2)
    return result
