"""
馬券構成自動提案モジュール v2。

対応券種: 3連複・馬連・3連単（2〜3点）
バイアス反映: 当日のバイアス（内前有利等）を軸・紐の選定に反映する。
予算上限: 指定予算内に自動的に収まるよう調整。

ユーザーの好みに合わせたデフォルト構成:
  - メイン: 3連複（穴馬軸1頭 × 人気馬1頭 × 相手3〜5頭）
  - サブ:   馬連（単勝より馬連を好む）
  - 狙い:   3連単（2〜3点に絞る）
"""
import itertools
import pandas as pd
import numpy as np
from dataclasses import dataclass

TICKET_UNIT = 100


@dataclass
class Ticket:
    bet_type: str
    horses: list[str]
    amount: int
    expected_return: float
    ev: float
    reason: str = ""


def _round_to_unit(amount: int) -> int:
    return max(TICKET_UNIT, (amount // TICKET_UNIT) * TICKET_UNIT)


# ============================================================
# バイアス反映の軸・紐選定
# ============================================================

def select_axis_horses(
    eval_df: pd.DataFrame,
    bias_type: str = "neutral",
) -> dict:
    """
    バイアスを考慮して穴馬軸・人気馬軸・紐候補を選定する。

    Returns:
        longshot_axis: 穴馬軸（1頭）
        popular_axis:  人気馬軸（1頭）
        partners:      紐候補（3〜5頭）
        n_partners:    流す相手の数
    """
    if eval_df.empty:
        return {}

    df = eval_df.copy()
    df["ev"]           = pd.to_numeric(df.get("ev",           0), errors="coerce").fillna(-0.5)
    df["popularity"]   = pd.to_numeric(df.get("popularity",   9), errors="coerce").fillna(9)
    df["odds"]         = pd.to_numeric(df.get("odds",        10), errors="coerce").fillna(10)
    df["bias_bonus"]   = pd.to_numeric(df.get("realtime_bias_bonus", 0), errors="coerce").fillna(0)
    df["conf_score"]   = pd.to_numeric(df.get("confidence_score", 50), errors="coerce").fillna(50)

    # バイアス補正スコアで並べ替え
    df["selection_score"] = df["ev"] * 10 + df["conf_score"] / 100 + df["bias_bonus"] * 5

    # 穴馬軸：7番人気以上でselection_score最大
    longshots = df[df["popularity"] >= 7].sort_values("selection_score", ascending=False)
    ls_axis = longshots.iloc[0] if not longshots.empty else None

    # 人気馬軸：1〜5番人気でselection_score最大
    populars = df[df["popularity"] <= 5].sort_values("selection_score", ascending=False)
    pop_axis = populars.iloc[0] if not populars.empty else df.sort_values("popularity").iloc[0]

    # 紐候補：軸2頭以外でselection_score上位3〜5頭
    axis_names = set()
    if ls_axis is not None:  axis_names.add(ls_axis["horse_name"])
    axis_names.add(pop_axis["horse_name"])

    partners_df = df[~df["horse_name"].isin(axis_names)].sort_values(
        "selection_score", ascending=False
    )

    # バイアスによる紐の絞り込み（インデックスをリセットして比較）
    n_partners = 4
    partners_df = partners_df.reset_index(drop=True)
    if bias_type == "inner_speed":
        draw_ok  = partners_df["draw_bonus"].gt(0) if "draw_bonus" in partners_df.columns \
                   else pd.Series(False, index=partners_df.index)
        style_ok = partners_df["running_style"].str.contains("先行|逃げ", na=False) \
                   if "running_style" in partners_df.columns \
                   else pd.Series(False, index=partners_df.index)
        pref = partners_df[draw_ok | style_ok]
        rest = partners_df[~(draw_ok | style_ok)]
        partners_df = pd.concat([pref, rest], ignore_index=True)
    elif bias_type == "outer_diff":
        style_ok = partners_df["running_style"].str.contains("差し|追込", na=False) \
                   if "running_style" in partners_df.columns \
                   else pd.Series(False, index=partners_df.index)
        pref = partners_df[style_ok]
        rest = partners_df[~style_ok]
        partners_df = pd.concat([pref, rest], ignore_index=True)

    partners = partners_df.head(n_partners)["horse_name"].tolist()

    return {
        "longshot_axis": ls_axis["horse_name"] if ls_axis is not None else None,
        "popular_axis":  pop_axis["horse_name"],
        "partners":      partners,
        "n_partners":    len(partners),
        "ls_odds":       float(ls_axis["odds"]) if ls_axis is not None else 0,
        "pop_odds":      float(pop_axis["odds"]),
        "ls_ev":         float(ls_axis["ev"]) if ls_axis is not None else 0,
    }


# ============================================================
# メイン馬券構成関数
# ============================================================

def build_tickets(
    eval_df: pd.DataFrame,
    budget: int = 5000,
    surface: str = "芝",
    distance: int = 2000,
    bias_type: str = "neutral",
) -> dict:
    """
    評価済みDataFrame + バイアス情報から馬券構成案を構築する。
    """
    if eval_df.empty:
        return _empty_result("出走馬データがありません")

    axis = select_axis_horses(eval_df, bias_type)
    if not axis:
        return _empty_result("軸馬の選定に失敗しました")

    ls   = axis["longshot_axis"]   # 穴馬軸
    pop  = axis["popular_axis"]    # 人気馬軸
    partners = axis["partners"]    # 紐候補

    tickets  = []
    remaining = budget

    # ---- 3連複（メイン：穴馬軸 × 人気馬軸 × 相手N頭） ---- #
    if ls and pop and partners:
        combos = [p for p in partners if p not in (ls, pop)]
        n_combo = min(len(combos), 5)  # 最大5頭流し
        combo_cost = n_combo * TICKET_UNIT
        if combo_cost <= remaining * 0.6:
            unit = _round_to_unit(min(200, (remaining * 0.5) // max(n_combo, 1)))
            total_3f = unit * n_combo
            tickets.append(Ticket(
                bet_type="3連複",
                horses=[ls, pop, "流し"],
                amount=total_3f,
                expected_return=total_3f * 4.0,
                ev=axis["ls_ev"],
                reason=f"{ls}×{pop} 流し{n_combo}頭（{unit}円×{n_combo}点）",
            ))
            remaining -= total_3f

    # ---- 馬連（穴馬軸 × 人気馬軸） ---- #
    if ls and pop and remaining >= TICKET_UNIT * 2:
        amt_baren = _round_to_unit(min(500, remaining // 4))
        ls_odd   = axis["ls_odds"]
        pop_odd  = axis["pop_odds"]
        est_odds = (ls_odd * pop_odd) ** 0.35  # 馬連の簡易推定
        tickets.append(Ticket(
            bet_type="馬連",
            horses=[ls, pop],
            amount=amt_baren,
            expected_return=amt_baren * est_odds,
            ev=axis["ls_ev"] * 0.7,
            reason=f"{ls}×{pop}（軸2頭の直接対決）",
        ))
        remaining -= amt_baren

    # ---- 3連単（2〜3点に絞る） ---- #
    if ls and pop and partners and remaining >= TICKET_UNIT * 2:
        trifecta_count = 0
        # パターン1: 穴馬1着固定 → 人気馬2着 → 相手3着
        if len(partners) >= 1 and remaining >= TICKET_UNIT:
            third = next((p for p in partners if p not in (ls, pop)), None)
            if third:
                tickets.append(Ticket(
                    bet_type="3連単",
                    horses=[ls, pop, third],
                    amount=TICKET_UNIT,
                    expected_return=TICKET_UNIT * ls_odd_for_trifecta(axis["ls_odds"]),
                    ev=axis["ls_ev"],
                    reason=f"1着固定: {ls}（当たればラッキー）",
                ))
                remaining -= TICKET_UNIT
                trifecta_count += 1

        # パターン2: 人気馬1着 → 穴馬2着 → 相手3着
        if trifecta_count < 3 and len(partners) >= 1 and remaining >= TICKET_UNIT:
            third2 = next((p for p in partners if p not in (ls, pop)), None)
            if third2:
                tickets.append(Ticket(
                    bet_type="3連単",
                    horses=[pop, ls, third2],
                    amount=TICKET_UNIT,
                    expected_return=TICKET_UNIT * ls_odd_for_trifecta(axis["ls_odds"]) * 0.6,
                    ev=axis["ls_ev"] * 0.6,
                    reason=f"1着: {pop}、2着: {ls}（穴馬2着狙い）",
                ))
                remaining -= TICKET_UNIT

    total_cost = sum(t.amount for t in tickets)

    # ---- 旧来スタイルとの比較プラン ---- #
    romance_plan, romance_cost, brain_warning = _build_romance_plan(
        eval_df, budget, bias_type
    )

    return {
        "recommended":      tickets,
        "total_cost":       total_cost,
        "remaining_budget": budget - total_cost,
        "axis":             axis,
        "romance_plan":     romance_plan,
        "romance_cost":     romance_cost,
        "brain_warning":    brain_warning,
        "bias_applied":     bias_type,
    }


def ls_odd_for_trifecta(ls_odds: float) -> float:
    """穴馬オッズから3連単の概算倍率を推定"""
    return max(10.0, ls_odds * 8)


# ============================================================
# 旧来スタイル（大穴全流し）との比較
# ============================================================

def _build_romance_plan(
    eval_df: pd.DataFrame,
    budget: int,
    bias_type: str,
) -> tuple[list, int, str]:
    """2頭軸3連複全流しプランを再現して警告を出す。"""
    axis = select_axis_horses(eval_df, bias_type)
    if not axis or not axis.get("longshot_axis"):
        return [], 0, ""

    ls  = axis["longshot_axis"]
    pop = axis["popular_axis"]
    n_others = len(eval_df) - 2
    if n_others <= 0:
        return [], 0, ""

    total_cost = n_others * TICKET_UNIT
    ls_ev = axis["ls_ev"]

    warning = ""
    if ls_ev < -0.2:
        warning = (
            f"⚠️ ブレイン警告: {ls}（{eval_df[eval_df['horse_name']==ls]['popularity'].iloc[0] if ls in eval_df['horse_name'].values else '?'}番人気）"
            f"のEV={ls_ev:+.3f}。全流しは{total_cost:,}円かかり、期待値的に不利です。\n"
            f"推奨プランとの差額: {total_cost - sum([TICKET_UNIT * 3]):+,}円"
        )
    elif total_cost > budget:
        warning = f"⚠️ 全流しは{total_cost:,}円で予算{budget:,}円をオーバーします。"

    plan = [Ticket(
        bet_type="3連複全流し（比較）",
        horses=[ls, pop, "残り全頭"],
        amount=total_cost,
        expected_return=total_cost * 0.75,
        ev=ls_ev,
        reason=f"旧スタイル: {ls}×{pop} 全流し{n_others}点",
    )]
    return plan, total_cost, warning


def _empty_result(msg: str) -> dict:
    return {
        "recommended": [], "total_cost": 0, "remaining_budget": 0,
        "axis": {}, "romance_plan": [], "romance_cost": 0,
        "brain_warning": msg, "bias_applied": "unknown",
    }


# ============================================================
# 表示用フォーマット
# ============================================================

def format_tickets_for_display(tickets: list[Ticket]) -> pd.DataFrame:
    rows = []
    for t in tickets:
        horse_str = (
            " → ".join(t.horses) if t.bet_type == "3連単"
            else " / ".join(t.horses)
        )
        rows.append({
            "券種":     t.bet_type,
            "馬":       horse_str,
            "金額":     f"{t.amount:,}円",
            "期待回収": f"{int(t.expected_return):,}円",
            "EV":       f"{t.ev:+.3f}",
            "根拠":     t.reason,
        })
    return pd.DataFrame(rows)


def get_bet_summary(result: dict) -> str:
    """馬券構成のサマリー文字列を返す（AI相談への注入用）"""
    if not result.get("recommended"):
        return "推奨馬券なし（見送り推奨）"
    axis = result.get("axis", {})
    lines = [
        f"穴馬軸: {axis.get('longshot_axis','?')}（{axis.get('ls_odds','?')}倍）",
        f"人気馬軸: {axis.get('popular_axis','?')}（{axis.get('pop_odds','?')}倍）",
        f"合計: {result['total_cost']:,}円 / {len(result['recommended'])}点",
    ]
    return "\n".join(lines)
