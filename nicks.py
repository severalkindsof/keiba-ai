"""
ニックス（父×母父配合）分析モジュール。

「ニックスは成功例から傾向を割り出すもの」- 実際の過去成績から算出する。
データソース: Kaggleデータの sire（父）+ dam_sire（母父）列。
"""
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(ttl=3600)
def build_nicks_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去データから 父×母父×距離帯 の組み合わせ別勝率テーブルを構築する。
    サンプル数が少ない組み合わせは除外。
    """
    required = ['sire', 'dam_sire', 'distance_cat', 'win_flag']
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    df2 = df.dropna(subset=['sire', 'dam_sire']).copy()
    if df2.empty:
        return pd.DataFrame()

    tbl = (
        df2.groupby(['sire', 'dam_sire', 'distance_cat'], observed=True)
        .agg(
            races=('win_flag', 'count'),
            wins=('win_flag', 'sum'),
            places=('place_flag', 'sum') if 'place_flag' in df2.columns else ('win_flag', 'sum'),
        )
        .reset_index()
    )
    tbl = tbl[tbl['races'] >= 8]  # 最低8サンプル
    tbl['nicks_win_rate'] = tbl['wins'] / tbl['races']
    tbl['nicks_place_rate'] = tbl['places'] / tbl['races']

    # 全体平均との差分
    overall_avg = df2['win_flag'].mean()
    tbl['nicks_bonus'] = tbl['nicks_win_rate'] - overall_avg
    return tbl.sort_values('nicks_win_rate', ascending=False)


def get_nicks_bonus(
    nicks_table: pd.DataFrame,
    sire: str,
    dam_sire: str,
    distance_cat: str,
) -> dict:
    """
    特定の父×母父×距離帯のニックス補正値を返す。
    """
    if nicks_table.empty or not sire or not dam_sire:
        return {'bonus': 0.0, 'win_rate': None, 'sample': 0, 'label': ''}

    mask = (
        (nicks_table['sire'] == sire)
        & (nicks_table['dam_sire'] == dam_sire)
        & (nicks_table['distance_cat'] == distance_cat)
    )
    row = nicks_table[mask]

    if row.empty:
        # 距離帯なしで検索
        mask2 = (nicks_table['sire'] == sire) & (nicks_table['dam_sire'] == dam_sire)
        row = nicks_table[mask2]
        if row.empty:
            return {'bonus': 0.0, 'win_rate': None, 'sample': 0, 'label': 'データなし'}

    best = row.sort_values('races', ascending=False).iloc[0]
    bonus = float(best['nicks_bonus'])
    wr = float(best['nicks_win_rate'])
    sample = int(best['races'])

    if bonus >= 0.03:
        label = f'◎ 相性抜群ニックス（{sire}×{dam_sire}）'
    elif bonus >= 0.01:
        label = f'○ 好相性ニックス（{sire}×{dam_sire}）'
    elif bonus <= -0.02:
        label = f'▼ 相性不良（{sire}×{dam_sire}）'
    else:
        label = f'△ ニックス普通（{sire}×{dam_sire}）'

    return {
        'bonus': round(bonus, 4),
        'win_rate': round(wr, 4),
        'sample': sample,
        'label': label,
    }


# 主要種牡馬の距離・馬場適性（静的知識ベース補完用）
SIRE_COURSE_AFFINITY = {
    'ディープインパクト': {'芝': 0.02, 'ダート': -0.02, '長距離': 0.02, '短距離': -0.01},
    'キングカメハメハ': {'芝': 0.01, 'ダート': 0.01, '中距離': 0.02},
    'ハーツクライ': {'芝': 0.02, 'ダート': -0.02, '長距離': 0.03, 'マイル': 0.01},
    'ロードカナロア': {'芝': 0.01, 'ダート': 0.0, '短距離': 0.03, 'マイル': 0.02},
    'エピファネイア': {'芝': 0.02, 'ダート': -0.01, '中距離': 0.02, '長距離': 0.01},
    'モーリス': {'芝': 0.02, 'ダート': -0.01, 'マイル': 0.03, '中距離': 0.01},
    'スクリーンヒーロー': {'芝': 0.01, '中距離': 0.02, '長距離': 0.01},
    'ダイワメジャー': {'芝': 0.01, 'マイル': 0.02, '短距離': 0.01},
    'オルフェーヴル': {'芝': 0.02, '長距離': 0.03, '中距離': 0.02},
    'ステイゴールド': {'芝': 0.01, '長距離': 0.02},
    'ルーラーシップ': {'芝': 0.01, 'ダート': 0.01, '中距離': 0.02},
    'ゴールドシップ': {'芝': 0.01, '長距離': 0.02},
}

def get_static_sire_bonus(sire: str, surface: str, distance_cat: str) -> float:
    """
    静的知識ベースから父系の馬場・距離補正を返す（ニックスデータ不足時の補完）。
    """
    affinity = SIRE_COURSE_AFFINITY.get(sire, {})
    bonus = affinity.get(surface, 0.0) + affinity.get(distance_cat, 0.0)
    return round(bonus, 4)


def apply_nicks(
    horses: list[dict],
    nicks_table: pd.DataFrame,
    surface: str,
    distance_cat: str,
) -> list[dict]:
    """出走馬全頭にニックス補正を付与する。"""
    result = []
    for h in horses:
        h2 = dict(h)
        sire = h2.get('sire', '')
        dam_sire = h2.get('dam_sire', '')

        nicks = get_nicks_bonus(nicks_table, sire, dam_sire, distance_cat)
        static_b = get_static_sire_bonus(sire, surface, distance_cat)

        # ニックスデータがあればそちら優先、なければ静的補正
        final_bonus = nicks['bonus'] if nicks['bonus'] != 0.0 else static_b

        h2['nicks_bonus'] = final_bonus
        h2['nicks_win_rate'] = nicks['win_rate']
        h2['nicks_sample'] = nicks['sample']
        h2['nicks_label'] = nicks['label']
        result.append(h2)
    return result
