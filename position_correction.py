"""
前走位置取り補正モジュール。

「着順」だけでは見えない馬の真の能力を補正する。
例: 本来差し馬が前走で先行させられた → 着順より能力を過小評価されている

データソース: Kaggleデータの corner_order 列（コーナー通過順）
"""
import pandas as pd
import numpy as np


def parse_corner_positions(corner_str: str) -> list[int]:
    """
    'コーナー通過順' 文字列をパースして各コーナーでの位置リストを返す。
    例: '3-4-4-3' → [3, 4, 4, 3]
         '2,3,3' → [2, 3, 3]
    """
    if not corner_str or pd.isna(corner_str):
        return []
    s = str(corner_str).replace('　', '').strip()
    # セパレータを統一
    for sep in ['-', ',', '，', '・', ' ']:
        if sep in s:
            parts = s.split(sep)
            try:
                return [int(p.strip()) for p in parts if p.strip().isdigit()]
            except Exception:
                pass
    # 単一数字
    try:
        return [int(s)]
    except Exception:
        return []


def infer_natural_style_from_history(history: pd.DataFrame) -> str:
    """
    過去複数走のコーナー通過順の平均から「自然な脚質」を推定する。
    """
    if history.empty or 'corner_order' not in history.columns:
        return '不明'

    avg_positions = []
    for val in history['corner_order'].dropna().head(5):
        positions = parse_corner_positions(val)
        if positions:
            avg_positions.append(np.mean(positions))

    if not avg_positions:
        return '不明'

    overall_avg = np.mean(avg_positions)
    if overall_avg <= 2.5:
        return '逃げ'
    elif overall_avg <= 5.0:
        return '先行'
    elif overall_avg <= 8.0:
        return '中団'
    else:
        return '差し・追込'


def analyze_position_mismatch(
    history: pd.DataFrame,
    natural_style: str,
) -> dict:
    """
    前走の位置取りが自然な脚質と乖離していたかどうかを検出する。

    例:
    - 差し馬が前走で3番手先行 → スタイル矯正させられた → 今走は巻き返し候補
    - 逃げ馬が前走で後方から → 展開に恵まれなかった

    Returns:
        mismatch_flag: bool
        mismatch_degree: float（0〜1, 大きいほど乖離大）
        correction_bonus: float（勝率補正値, +0.01〜+0.03）
        message: str
    """
    if history.empty or 'corner_order' not in history.columns or natural_style == '不明':
        return {'mismatch_flag': False, 'mismatch_degree': 0.0,
                'correction_bonus': 0.0, 'message': ''}

    prev = history.iloc[0]
    prev_positions = parse_corner_positions(prev.get('corner_order', ''))
    if not prev_positions:
        return {'mismatch_flag': False, 'mismatch_degree': 0.0,
                'correction_bonus': 0.0, 'message': ''}

    prev_avg_pos = np.mean(prev_positions)
    prev_rank = pd.to_numeric(prev.get('rank', 99), errors='coerce')

    # 自然スタイルの「期待位置」
    expected = {'逃げ': 1.5, '先行': 3.5, '中団': 6.5, '差し・追込': 10.0}
    expected_pos = expected.get(natural_style, 6.0)

    # 乖離量（前走での実際の位置 - 期待位置）
    deviation = abs(prev_avg_pos - expected_pos)
    mismatch_degree = min(1.0, deviation / 6.0)

    # 強制先行（差し馬が前目に）or 強制後退（逃げ馬が後ろ）
    forced_forward = (natural_style in ('差し・追込', '中団')) and prev_avg_pos < expected_pos - 3
    forced_back = (natural_style in ('逃げ', '先行')) and prev_avg_pos > expected_pos + 3

    if not (forced_forward or forced_back) or mismatch_degree < 0.3:
        return {'mismatch_flag': False, 'mismatch_degree': mismatch_degree,
                'correction_bonus': 0.0, 'message': ''}

    # 前走で凡走していた場合のみ「巻き返し補正」を与える
    if pd.notna(prev_rank) and prev_rank >= 5:
        bonus = min(0.03, mismatch_degree * 0.04)
        direction = '強制先行' if forced_forward else '強制後退'
        msg = (f'前走{direction}（{prev_avg_pos:.1f}番手、本来{natural_style}）→'
               f' 前走{int(prev_rank)}着は展開負け。今走巻き返し候補。')
        return {'mismatch_flag': True, 'mismatch_degree': round(mismatch_degree, 2),
                'correction_bonus': round(bonus, 3), 'message': msg}

    return {'mismatch_flag': False, 'mismatch_degree': round(mismatch_degree, 2),
            'correction_bonus': 0.0, 'message': ''}


def apply_position_correction(
    horses: list[dict],
    df_hist: pd.DataFrame,
) -> list[dict]:
    """出走馬全頭に位置取り補正を付与する。"""
    result = []
    for h in horses:
        h2 = dict(h)
        name = h2.get('horse_name', '')

        hist = pd.DataFrame()
        if not df_hist.empty and 'horse_name' in df_hist.columns and name:
            hist = df_hist[df_hist['horse_name'] == name]
            if 'date' in df_hist.columns:
                hist = hist.sort_values('date', ascending=False)
            hist = hist.head(10)

        natural_style = h2.get('running_style') or infer_natural_style_from_history(hist)
        h2['natural_style'] = natural_style

        correction = analyze_position_mismatch(hist, natural_style)
        h2['position_mismatch_flag'] = correction['mismatch_flag']
        h2['position_mismatch_degree'] = correction['mismatch_degree']
        h2['position_correction_bonus'] = correction['correction_bonus']
        h2['position_correction_msg'] = correction['message']
        result.append(h2)
    return result
