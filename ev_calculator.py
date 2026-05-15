"""
期待値（EV）計算エンジン。
EV = 勝率 × (オッズ - 1) - (1 - 勝率)
"""
import pandas as pd
import numpy as np
from data_loader import get_win_rate_table, get_sire_stats, get_jockey_stats, categorize_distance
from knowledge_base import apply_kb_to_horse


# ---- 期待値計算 ---- #

def calc_ev(win_rate: float, odds: float) -> float:
    """単勝EV。win_rate は 0〜1 の小数。"""
    if odds <= 1.0 or np.isnan(win_rate) or np.isnan(odds):
        return np.nan
    return win_rate * (odds - 1) - (1 - win_rate)


def calc_ev_place(place_rate: float, place_odds: float) -> float:
    """複勝EV。"""
    if place_odds <= 1.0 or np.isnan(place_rate) or np.isnan(place_odds):
        return np.nan
    return place_rate * (place_odds - 1) - (1 - place_rate)


# ---- 条件別勝率ルックアップ ---- #

def lookup_win_rate(
    win_rate_table: pd.DataFrame,
    surface: str,
    distance: int,
    popularity: int,
) -> dict:
    """
    条件テーブルから勝率・複勝率を取得。
    見つからない場合は人気帯の全面的な平均にフォールバック。
    """
    dist_cat = categorize_distance(distance)
    pop_bucket = _popularity_bucket(popularity)

    mask = (
        (win_rate_table["surface"] == surface)
        & (win_rate_table["distance_cat"] == dist_cat)
        & (win_rate_table["pop_bucket"] == pop_bucket)
    )
    row = win_rate_table[mask]

    if row.empty:
        # フォールバック：人気帯のみで絞る
        mask2 = win_rate_table["pop_bucket"] == pop_bucket
        row = win_rate_table[mask2]

    if row.empty:
        return {"win_rate": np.nan, "place_rate": np.nan, "sample_size": 0}

    return {
        "win_rate": float(row["win_rate"].mean()),
        "place_rate": float(row["place_rate"].mean()),
        "sample_size": int(row["races"].sum()),
    }


def _popularity_bucket(popularity: int) -> str:
    if popularity <= 3:
        return "1〜3番人気"
    elif popularity <= 6:
        return "4〜6番人気"
    elif popularity <= 9:
        return "7〜9番人気"
    else:
        return "10番人気以下"


# ---- 血統補正 ---- #

def sire_bonus(sire_stats: pd.DataFrame, sire: str, distance: int) -> float:
    """
    父系の距離適性補正値を返す（±0.02 程度）。
    全体平均との差分を補正値として使う。
    """
    if sire_stats.empty or not sire:
        return 0.0
    dist_cat = categorize_distance(distance)
    row = sire_stats[(sire_stats["sire"] == sire) & (sire_stats["distance_cat"] == dist_cat)]
    if row.empty:
        return 0.0
    sire_wr = float(row["win_rate"].iloc[0])
    overall_avg = float(sire_stats[sire_stats["distance_cat"] == dist_cat]["win_rate"].mean())
    if np.isnan(overall_avg) or overall_avg == 0:
        return 0.0
    return round(sire_wr - overall_avg, 4)


# ---- 騎手補正 ---- #

def jockey_bonus(jockey_stats: pd.DataFrame, jockey: str, popularity: int) -> float:
    """
    穴馬（10番人気以下）での騎手補正値。
    穴複勝率が平均より高い騎手はプラス補正。
    """
    if jockey_stats.empty or not jockey or popularity < 10:
        return 0.0
    row = jockey_stats[jockey_stats["jockey"] == jockey]
    if row.empty:
        return 0.0
    avg_place = float(jockey_stats["place_rate_longshot"].mean())
    jockey_place = float(row["place_rate_longshot"].iloc[0])
    return round(jockey_place - avg_place, 4)


# ---- 馬場状態補正 ---- #

TRACK_CONDITION_MULTIPLIER = {
    "良": 1.0,
    "稍重": 1.05,   # 荒れやすく穴が出やすい
    "重": 1.10,
    "不良": 1.15,
}

def track_condition_bonus(condition: str, popularity: int) -> float:
    """馬場悪化時は穴馬の相対的な勝率が上昇する傾向（補正値）"""
    if popularity < 10:
        return 0.0
    multiplier = TRACK_CONDITION_MULTIPLIER.get(condition, 1.0)
    return round((multiplier - 1.0) * 0.5, 4)  # 穴馬の勝率にのみ半分加算


# ---- メイン評価関数 ---- #

def evaluate_horse(
    horse: dict,
    win_rate_table: pd.DataFrame,
    sire_stats: pd.DataFrame,
    jockey_stats: pd.DataFrame,
) -> dict:
    """
    1頭の馬を評価してEVスコア・各種指標を返す。

    horse dict の必須キー:
        horse_name, odds, popularity, surface, distance, jockey, sire, track_condition
    オプション:
        place_odds
    """
    surface = horse.get("surface", "芝")
    distance = int(horse.get("distance") or 2000)
    popularity = int(horse.get("popularity") or 9)
    odds = float(horse.get("odds") or 10.0)
    place_odds = float(horse.get("place_odds") or (odds / 3))
    jockey = horse.get("jockey", "")
    sire = horse.get("sire", "")
    condition = horse.get("track_condition", "良")

    # 基礎勝率・複勝率
    stats = lookup_win_rate(win_rate_table, surface, distance, popularity)
    win_rate = stats["win_rate"]
    place_rate = stats["place_rate"]

    if np.isnan(win_rate):
        win_rate = 1.0 / popularity if popularity > 0 else 0.05

    # 各種補正
    sb = sire_bonus(sire_stats, sire, distance)
    jb = jockey_bonus(jockey_stats, jockey, popularity)
    tb = track_condition_bonus(condition, popularity)

    # ナレッジベースボーナス
    kb_result = apply_kb_to_horse(horse, race_name=horse.get("race_name", ""))
    kb_b = kb_result.get("kb_bonus", 0.0)

    adjusted_win_rate = max(0.001, win_rate + sb + tb + kb_b)
    safe_place_rate = win_rate * 3 if (place_rate is None or np.isnan(place_rate)) else place_rate
    adjusted_place_rate = max(0.001, safe_place_rate + jb + tb)

    ev = calc_ev(adjusted_win_rate, odds)
    ev_place = calc_ev_place(adjusted_place_rate, place_odds)

    implied = 1.0 / odds
    odds_distortion = adjusted_win_rate - implied  # プラス = 過小評価（美味しい）

    # ロマン爆死スコア（高いほど「ロマンだけで買う危険な馬」）
    romance_danger = _romance_danger_score(popularity, adjusted_win_rate, odds)

    base = {
        "horse_name": horse.get("horse_name", ""),
        "odds": odds,
        "popularity": popularity,
        "implied_prob": round(implied * 100, 1),
        "est_win_rate": round(adjusted_win_rate * 100, 1),
        "est_place_rate": round(adjusted_place_rate * 100, 1),
        "odds_distortion": round(odds_distortion * 100, 1),
        "ev": round(ev, 3),
        "ev_place": round(ev_place, 3),
        "sire_bonus": sb,
        "jockey_bonus": jb,
        "track_bonus": tb,
        "kb_bonus": kb_b,
        "kb_notes": kb_result.get("kb_notes", []),
        "kb_avoids": kb_result.get("kb_avoids", []),
        "sample_size": stats["sample_size"],
        "romance_danger": romance_danger,
        "verdict": _verdict(ev, ev_place, popularity, adjusted_win_rate, odds),
        # 新ファクター（後から付与されるフィールドのデフォルト値）
        "pace_benefit": horse.get("pace_benefit", 0.0),
        "draw_bonus": horse.get("draw_bonus", 0.0),
        "draw_label": horse.get("draw_label", ""),
        "jockey_change_bonus": horse.get("jockey_change_bonus", 0.0),
        "jockey_change_signal": horse.get("jockey_change_signal", ""),
        "jockey_change_msg": horse.get("jockey_change_msg", ""),
        "rotation_bonus": horse.get("rotation_bonus", 0.0),
        "rotation_signal": horse.get("rotation_signal", ""),
        "rotation_days": horse.get("rotation_days"),
        "tatakidai_flag": horse.get("tatakidai_flag", False),
        "tatakidai_bonus": horse.get("tatakidai_bonus", 0.0),
        "tatakidai_message": horse.get("tatakidai_message", ""),
        "class_bonus": horse.get("class_bonus", 0.0),
        "class_signal": horse.get("class_signal", ""),
        "weight_bonus": horse.get("weight_bonus", 0.0),
        "weight_signal": horse.get("weight_signal", ""),
        "weight_message": horse.get("weight_message", ""),
        "exhaustion_comeback": horse.get("exhaustion_comeback", False),
        "exhaustion_message": horse.get("exhaustion_message", ""),
        "running_style": horse.get("running_style", "不明"),
    }
    return base


def _romance_danger_score(popularity: int, win_rate: float, odds: float) -> str:
    """
    穴馬を「ロマンだけで」買う危険度。
    EVがマイナスかつ人気が低い馬は危険ラベルをつける。
    """
    ev = calc_ev(win_rate, odds)
    if popularity < 7:
        return "低"
    if np.isnan(ev) or ev < -0.3:
        return "極高（要注意）"
    elif ev < -0.1:
        return "高"
    elif ev < 0:
        return "中"
    else:
        return "低（EV+）"


def _verdict(ev: float, ev_place: float, popularity: int, win_rate: float, odds: float) -> str:
    """総合判定コメント"""
    if np.isnan(ev):
        return "データ不足 → 見送り推奨"
    if ev > 0.1:
        return "◎ 買い推奨（EV+、期待値あり）"
    elif ev > 0:
        return "○ 検討可（わずかにEV+）"
    elif ev_place > 0 and popularity <= 9:
        return "△ 複勝・ワイドなら検討"
    elif popularity >= 10 and ev < -0.2:
        return "✕ ロマン爆死リスク大（見送り推奨）"
    else:
        return "▲ 様子見（EV微マイナス）"


# ---- レース全体の評価 ---- #

def evaluate_race(
    horses: list[dict],
    win_rate_table: pd.DataFrame,
    sire_stats: pd.DataFrame,
    jockey_stats: pd.DataFrame,
) -> pd.DataFrame:
    """出走馬リストを一括評価してDataFrameで返す"""
    results = [evaluate_horse(h, win_rate_table, sire_stats, jockey_stats) for h in horses]
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("ev", ascending=False).reset_index(drop=True)
    return df
