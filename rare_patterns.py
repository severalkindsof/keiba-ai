"""
第13波・C: レアパターン抽出（ルールベース穴ヒモ拾い）

「人気は無いが、馬の経歴・条件から激走の典型パターンに該当する馬」を抽出する
ヒューリスティクス・ルール集。

ロマン爆穴モードでの「穴ヒモ」候補抽出に使う。

公開関数:
    detect_rare_patterns(horse_dict, context) -> list[dict]
        horse_dict: 1頭の特徴量 dict
        context: レース全体の context dict (surface, distance, track_condition, etc)
        Returns: マッチしたパターンの list [{"name": ..., "score": +N, "reason": ...}, ...]

    score_rare_bonus(horse_dict, context) -> tuple[float, list[str]]
        合計加点 + 理由リストを返す（confluence の補助に使う想定）
"""
from __future__ import annotations
from typing import Any
import math


# ============================================================
# パターン定義
# ============================================================

# (第14波修正) キー名はすべて eval_df / horse_latest_features に実在するものを使う。
# 監査で 12 キー中 8 キーが架空名と判明（BUG-X1 と同型）→ 実在キーへ全面書き換え。

def _pat_class_dropper(horse: dict, ctx: dict) -> dict | None:
    """格上経験馬の人気落ち（class_level: 前走または近走で高クラス経験）"""
    cl = max(_f(horse.get("class_level")), _f(horse.get("prev_class_level")))
    if cl >= 8 and _f(horse.get("popularity"), 99) >= 8:  # 8=G3 級以上の想定
        return {
            "name":   "格上経験馬の人気落ち",
            "score":  +6,   # 実測 +17.7pp
            "reason": f"クラスレベル {cl:.0f}（重賞級経験）ながら人気薄",
        }
    return None


def _pat_wet_specialist(horse: dict, ctx: dict) -> dict | None:
    """道悪鬼（重・不良で複勝率高い）が当日道悪"""
    if ctx.get("track_condition") in ("重", "不良", "稍重"):
        wpr = _f(horse.get("wet_place_rate"))
        if wpr >= 0.45 and _f(horse.get("popularity"), 99) >= 6:
            return {
                "name":   "道悪鬼の人気薄",
                "score":  +6,
                "reason": f"道悪複勝率 {wpr*100:.0f}% / 当日 {ctx.get('track_condition')}馬場",
            }
    return None


# (第33波) コース巧者パターンは除去: vd_win_rate は「そのコースの平均勝率」
# （コース属性、82,730頭で40種しか値がない）であり馬の巧者性ではなかった。
# 列の意味を誤解した第14波の設計ミス。馬別コース成績列が存在しないため除去。


def _pat_trainer_signal(horse: dict, ctx: dict) -> dict | None:
    """厩舎の本気サイン（直近50走勝率 + 上り調子トレンド）"""
    tw = _f(horse.get("trainer_recent50_winrate"))
    trend = _f(horse.get("trainer_trend"))
    if tw >= 0.12 and trend > 0 and _f(horse.get("popularity"), 99) >= 7:
        return {
            "name":   "厩舎本気サイン",
            "score":  +2,   # 実測 +2.2pp（旧+5は過大）
            "reason": f"厩舎直近50走勝率 {tw*100:.0f}% + 上り調子",
        }
    return None


def _pat_top_jockey_change(horse: dict, ctx: dict) -> dict | None:
    """強い騎手への乗替（jockey_change_signal は app.py が生成する実在キー）"""
    sig = str(horse.get("jockey_change_signal", "") or "")
    if ("◎" in sig or "○" in sig) and _f(horse.get("popularity"), 99) >= 6:
        return {
            "name":   "トップ騎手乗替の人気薄",
            "score":  +5,
            "reason": f"騎手乗替: {sig}",
        }
    return None


def _pat_speed_hidden(horse: dict, ctx: dict) -> dict | None:
    """スピード指数の平均（speed_fig_avg3）は高いが直近凡走で人気落ち"""
    sf_avg = _f(horse.get("speed_fig_avg3"))
    rank_avg = _f(horse.get("rank_avg3"), 99)
    # 第33波: rank_avg>=6 は 82% 該当の死に条件 → >=8（真の凡走帯）に
    if sf_avg >= 95 and rank_avg >= 8 and _f(horse.get("popularity"), 99) >= 8:
        return {
            "name":   "スピード指数隠れ実力",
            "score":  +4,   # 実測 +5.7pp
            "reason": f"指数3走平均 {sf_avg:.0f} / 着順平均 {rank_avg:.0f} → 凡走で人気落ち",
        }
    return None


def _pat_closing_move(horse: dict, ctx: dict) -> dict | None:
    """前走で強い追い込み（closing_move）を見せた人気薄（ハイペース予想時）"""
    cm = _f(horse.get("closing_move"))
    predicted = ctx.get("pace_predicted", "")
    # 第33波: 旧 2.0 は (corner4-rank)/頭数 のスケールでほぼ不可能（該当0.04%）
    # → 実分布90%タイル付近の 0.25 に（実測で+15ppエッジ確認済みの有効パターン）
    if cm >= 0.25 and "ハイ" in str(predicted) and _f(horse.get("popularity"), 99) >= 7:
        return {
            "name":   "ハイペース×追込穴",
            "score":  +4,
            "reason": f"前走の追い上げ {cm:+.1f} / 展開予想 ハイ",
        }
    return None


def _pat_distance_revert(horse: dict, ctx: dict) -> dict | None:
    """大幅距離変更で得意距離帯に戻る馬（前走距離 = horse_latest の distance）"""
    last_dist = _f(horse.get("distance"))       # horse_latest_features は前走の距離
    this_dist = _f(ctx.get("distance"))
    if last_dist <= 0 or this_dist <= 0:
        return None
    if abs(this_dist - last_dist) >= 400 and _f(horse.get("popularity"), 99) >= 7:
        # 得意カテゴリ判定: rank_best5（過去5走ベスト着順）が良い馬のみ
        if _f(horse.get("rank_best5"), 99) <= 3:
            return {
                "name":   "距離替わり一変候補",
                "score":  +5,   # 実測 +8.8pp
                "reason": f"前走{last_dist:.0f}m → 今走{this_dist:.0f}m / 過去5走ベスト{_f(horse.get('rank_best5'), 0):.0f}着",
            }
    return None


def _f(v, default: float = 0.0) -> float:
    """NaN / None / 文字列安全な float 変換"""
    try:
        f = float(v)
        if math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


# ============================================================
# 公開 API
# ============================================================

_PATTERNS = [
    _pat_class_dropper,
    _pat_wet_specialist,
    _pat_trainer_signal,
    _pat_top_jockey_change,
    _pat_speed_hidden,
    _pat_closing_move,
    _pat_distance_revert,
]


def detect_rare_patterns(horse: dict, ctx: dict) -> list[dict]:
    """マッチしたレアパターンのリストを返す。"""
    matched = []
    for fn in _PATTERNS:
        try:
            r = fn(horse or {}, ctx or {})
            if r:
                matched.append(r)
        except Exception:
            continue
    return matched


def score_rare_bonus(horse: dict, ctx: dict) -> tuple[float, list[str]]:
    """
    レアパターン合計加点と、理由のショートラベルリストを返す。
    confluence_score の補助として +0〜+15 程度を返す想定。
    """
    matched = detect_rare_patterns(horse, ctx)
    total = sum(m["score"] for m in matched)
    labels = [m["name"] for m in matched]
    # 過度な加算を防ぐためキャップ
    total = min(total, 15.0)
    return float(total), labels


def format_for_ui(matched: list[dict]) -> str:
    """UI 表示用の短いマークダウン文字列に変換。"""
    if not matched:
        return ""
    parts = [f"{m['name']} (+{m['score']})" for m in matched]
    return " / ".join(parts)
