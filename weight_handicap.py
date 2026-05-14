"""
斤量の相対評価（馬体重比）と頭数×本命信頼度モジュール。

データソース: 既存フィールド（weight_carried, horse_weight, popularity, 頭数）のみ。
新規データ取得不要。
"""
import numpy as np


# ============================================================
# 斤量の馬体重比分析
# ============================================================

WEIGHT_RATIO_THRESHOLD = 0.125  # 12.5% 超で勝率大幅低下（プロの閾値）
KG_TO_SECONDS = 0.2             # 1kg ≈ 0.2秒（JRA公式換算）

def calc_weight_ratio(weight_carried: float, horse_weight: int) -> dict:
    """
    斤量 ÷ 馬体重 の比率を計算し評価する。

    Args:
        weight_carried: 斤量（kg）例: 56.0
        horse_weight: 馬体重（kg）例: 480

    Returns:
        ratio: float
        signal: str
        bonus: float（勝率補正、重い斤量はマイナス）
        message: str
    """
    if not weight_carried or not horse_weight or horse_weight <= 0:
        return {'ratio': None, 'signal': '不明', 'bonus': 0.0, 'message': '体重・斤量データなし'}

    ratio = weight_carried / horse_weight

    if ratio > 0.135:
        return {
            'ratio': round(ratio, 4),
            'signal': '超重量（危険）',
            'bonus': -0.04,
            'message': f'斤量比{ratio*100:.1f}% → 13.5%超は勝率が著しく低下。軽視推奨。',
        }
    elif ratio > WEIGHT_RATIO_THRESHOLD:
        return {
            'ratio': round(ratio, 4),
            'signal': '重量注意',
            'bonus': -0.02,
            'message': f'斤量比{ratio*100:.1f}% → 12.5%超（標準体重の馬には厳しい斤量）',
        }
    elif ratio < 0.105:
        return {
            'ratio': round(ratio, 4),
            'signal': '軽量有利',
            'bonus': 0.01,
            'message': f'斤量比{ratio*100:.1f}% → 10.5%未満（体の大きい馬が軽い斤量）',
        }
    else:
        return {
            'ratio': round(ratio, 4),
            'signal': '適正範囲',
            'bonus': 0.0,
            'message': f'斤量比{ratio*100:.1f}% → 問題なし',
        }


def calc_weight_time_impact(
    weight_carried: float,
    standard_weight: float,
    distance: int,
) -> float:
    """
    斤量差による「タイム換算」補正値を返す（勝率ではなくタイム差として）。

    標準斤量（55kg）との差を 0.2秒/kg × 距離補正 で計算。
    長距離ほど影響大。
    """
    if not weight_carried or not standard_weight:
        return 0.0
    diff_kg = weight_carried - standard_weight
    distance_factor = 1.0 + (distance - 1600) / 3200  # 長距離ほど影響増
    time_diff = diff_kg * KG_TO_SECONDS * distance_factor
    # タイム差を勝率補正に変換（0.1秒 ≈ 勝率0.5%差）
    return round(-time_diff * 0.005, 4)


# ============================================================
# 頭数×本命信頼度評価
# ============================================================

# JRAデータ実績ベースの1番人気勝率テーブル
FAVORITE_WIN_RATES = {
    # 頭数: 勝率
    5:  0.55, 6:  0.52, 7:  0.50, 8:  0.48, 9:  0.46,
    10: 0.43, 11: 0.40, 12: 0.37, 13: 0.35, 14: 0.33,
    15: 0.31, 16: 0.30, 17: 0.29, 18: 0.28,
}

# 脚質×1番人気の勝率補正
STYLE_RELIABILITY = {
    '逃げ':      {'win_rate': 0.56, 'place_rate': 0.87, 'label': '最高信頼'},
    '先行':      {'win_rate': 0.48, 'place_rate': 0.80, 'label': '高信頼'},
    '中団':      {'win_rate': 0.40, 'place_rate': 0.72, 'label': '標準'},
    '差し・追込': {'win_rate': 0.32, 'place_rate': 0.65, 'label': 'やや低信頼'},
    '不明':      {'win_rate': 0.40, 'place_rate': 0.72, 'label': '標準'},
}


def eval_favorite_reliability(
    n_horses: int,
    popularity: int,
    running_style: str = '不明',
) -> dict:
    """
    1番人気馬の信頼度を評価する。

    穴狙いのユーザーにとっては「本命が脆弱なレース」を選ぶための指標。

    Returns:
        favorite_win_rate: float（本命の推定勝率）
        reliability_label: str
        is_weak_favorite: bool（本命が信頼できない = 荒れやすい）
        upset_score: int（0〜100, 高いほど穴馬チャンス大）
        message: str
    """
    base_wr = FAVORITE_WIN_RATES.get(min(18, max(5, n_horses)), 0.35)
    style_info = STYLE_RELIABILITY.get(running_style, STYLE_RELIABILITY['不明'])

    if popularity == 1:
        # 1番人気の場合、脚質補正を適用
        adj_wr = (base_wr + style_info['win_rate']) / 2
        reliability = style_info['label']
    else:
        adj_wr = base_wr
        reliability = '参考値'

    # 荒れやすさスコア（本命の弱さ + 頭数の多さ）
    upset_score = int(
        (1 - adj_wr) * 60          # 本命勝率が低いほど荒れやすい
        + min(n_horses, 18) * 2.2   # 頭数多いほど荒れやすい
    )
    upset_score = min(100, max(0, upset_score))

    is_weak = (n_horses >= 14 and adj_wr < 0.35) or running_style == '差し・追込'

    if is_weak:
        msg = f'頭数{n_horses}頭、{running_style}の本命→ 荒れやすいレース（穴狙いチャンス）'
    elif adj_wr >= 0.50:
        msg = f'頭数{n_horses}頭、{running_style}→ 本命鉄板の可能性（穴は薄い）'
    else:
        msg = f'頭数{n_horses}頭→ 標準的な荒れやすさ'

    return {
        'favorite_win_rate': round(adj_wr, 3),
        'reliability_label': reliability,
        'is_weak_favorite': is_weak,
        'upset_score': upset_score,
        'message': msg,
    }


# ============================================================
# 7. 斤量トレンド（ハンデ戦特化）
# ============================================================

def analyze_handicap_weight_trend(
    history_df,
    horse_name: str,
    current_weight_carried: float,
    race_name: str = "",
) -> dict:
    """
    ハンデ戦での斤量変化トレンドを分析する。
    「斤量が軽くなった」「以前より重い」等を過去成績から判定。

    Returns
    -------
    {
        "is_handicap":        bool,
        "trend":              str,   "軽量化" | "重量化" | "安定" | "初ハンデ"
        "handicap_win_rate":  float,
        "current_vs_best":    float | None,  今回斤量 - 過去最軽量
        "label":              str,
        "bonus":              float,
    }
    """
    import pandas as pd
    import numpy as np

    is_handicap = any(kw in (race_name or "") for kw in ["ハンデ", "ハンデキャップ", "H"])

    if not is_handicap:
        return {"is_handicap": False, "trend": "", "handicap_win_rate": 0.0,
                "current_vs_best": None, "label": "", "bonus": 0.0}

    empty_hc = {"is_handicap": True, "trend": "初ハンデ", "handicap_win_rate": 0.0,
                "current_vs_best": None, "label": "ハンデ戦初出走", "bonus": 0.0}

    if history_df is None or (hasattr(history_df, 'empty') and history_df.empty):
        return empty_hc

    hist = history_df
    if hasattr(hist, '__getitem__') and 'horse_name' in hist.columns:
        hist = hist[hist['horse_name'] == horse_name]

    if hist.empty or 'weight_carried' not in hist.columns:
        return empty_hc

    # ハンデ戦の過去成績に絞る（race_nameに"ハンデ"含む）
    if 'race_name' in hist.columns:
        hc_hist = hist[hist['race_name'].str.contains('ハンデ|ハンデキャップ', na=False)]
    else:
        hc_hist = hist  # フォールバック：全成績を使う

    if len(hc_hist) < 2:
        return empty_hc

    wc_series = pd.to_numeric(hc_hist['weight_carried'], errors='coerce').dropna()
    if wc_series.empty:
        return empty_hc

    best_weight = float(wc_series.min())  # 過去最軽量
    avg_weight  = float(wc_series.mean())
    current_vs_best = float(current_weight_carried) - best_weight if current_weight_carried else None

    # トレンド判定
    if len(wc_series) >= 3:
        slope = float(np.polyfit(np.arange(len(wc_series)), list(wc_series), 1)[0])
        if slope < -0.3:
            trend = "軽量化傾向"
        elif slope > 0.3:
            trend = "重量化傾向"
        else:
            trend = "斤量安定"
    else:
        trend = "参考"

    # ハンデ戦での勝率
    hc_win_rate = float(hc_hist['win_flag'].mean()) if 'win_flag' in hc_hist.columns else 0.0

    # 評価
    bonus = 0.0
    label_parts = [f"ハンデ戦{trend}"]

    if current_vs_best is not None:
        if current_vs_best <= -1:
            label_parts.append(f"今回{abs(current_vs_best):.0f}kg軽減（過去最軽）")
            bonus += 0.015
        elif current_vs_best >= 2:
            label_parts.append(f"今回過去最重比+{current_vs_best:.0f}kg")
            bonus -= 0.01

    if hc_win_rate >= 0.2:
        label_parts.append(f"ハンデ戦勝率{hc_win_rate*100:.0f}%")
        bonus += 0.01

    return {
        "is_handicap":        True,
        "trend":              trend,
        "handicap_win_rate":  round(hc_win_rate, 3),
        "current_vs_best":    round(current_vs_best, 1) if current_vs_best is not None else None,
        "label":              " / ".join(label_parts),
        "bonus":              round(bonus, 4),
    }


def apply_weight_handicap(
    horses: list[dict],
    distance: int,
    standard_weight: float = 55.0,
) -> list[dict]:
    """出走馬全頭に斤量・馬体重比補正を付与する。"""
    result = []
    for h in horses:
        h2 = dict(h)
        wc = h2.get('weight_carried')
        hw = h2.get('horse_weight')

        try:
            wc = float(wc) if wc else None
            hw = int(hw) if hw else None
        except Exception:
            wc, hw = None, None

        wr = calc_weight_ratio(wc, hw)
        time_adj = calc_weight_time_impact(wc, standard_weight, distance)

        h2['weight_ratio'] = wr['ratio']
        h2['weight_ratio_signal'] = wr['signal']
        h2['weight_ratio_bonus'] = wr['bonus'] + time_adj
        h2['weight_ratio_msg'] = wr['message']
        result.append(h2)
    return result
