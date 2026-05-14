"""
複数レース資金管理モジュール。

設計思想:
- 週予算（例: 10,000円）を土日の複数レースに最適配分する
- 信頼スコアが高いレースほど多く投資する（修正ケリー基準）
- 1レースへの最大投資は週予算の40%まで（過集中防止）
- 「今週どのレースに何円かけるか」をトップページに表示
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class RaceAllocation:
    race_id: str
    race_name: str
    date_str: str
    venue: str
    surface: str
    distance: int
    race_score: int          # Confluenceの最高スコア
    ev_plus_count: int       # EVプラス馬の頭数
    allocated_budget: int    # 配分予算（円）
    priority_rank: int       # 優先順位
    verdict: str
    top_horse: str
    top_ev: float


def kelly_fraction(win_rate: float, odds: float, fraction: float = 0.25) -> float:
    """
    修正ケリー基準による最適賭け比率。
    fraction=0.25 で 1/4ケリー（保守的）

    f* = (bp - q) / b  where b = odds-1, p = win_rate, q = 1-p
    """
    if odds <= 1.0 or win_rate <= 0:
        return 0.0
    b = odds - 1
    q = 1 - win_rate
    kelly = (b * win_rate - q) / b
    return max(0.0, min(kelly * fraction, 0.4))  # 最大40%


def allocate_weekend_budget(
    scan_df: pd.DataFrame,
    weekly_budget: int,
    max_races: int = 5,
    min_score: int = 50,
) -> list[RaceAllocation]:
    """
    週末のレーススキャン結果から最適な資金配分を計算する。

    Args:
        scan_df:       race_selector.scan_weekend_races() の出力
        weekly_budget: 週予算（円）
        max_races:     最大購入レース数
        min_score:     このスコア未満のレースは除外

    Returns:
        list of RaceAllocation（優先度順）
    """
    if scan_df.empty:
        return []

    # スコアでフィルタリング
    filtered = scan_df[scan_df["race_score"] >= min_score].copy()
    if filtered.empty:
        return []

    # 上位N件に絞る
    top = filtered.nlargest(max_races, "race_score").reset_index(drop=True)

    # スコアに基づく予算配分（スコアの比率で配分）
    scores = top["race_score"].values.astype(float)
    weights = scores / scores.sum()

    # 最小単位（100円）に丸める
    raw_budgets = (weights * weekly_budget / 100).astype(int) * 100
    # 端数調整
    diff = weekly_budget - raw_budgets.sum()
    raw_budgets[0] += diff  # 最高スコアレースに加算

    allocations = []
    for i, (_, row) in enumerate(top.iterrows()):
        alloc = RaceAllocation(
            race_id=str(row.get("race_id", "")),
            race_name=str(row.get("race_name", f"レース{i+1}")),
            date_str=str(row.get("date_str", "")),
            venue=str(row.get("surface", "")),
            surface=str(row.get("surface", "芝")),
            distance=int(row.get("distance", 2000)),
            race_score=int(row.get("race_score", 0)),
            ev_plus_count=int(row.get("ev_plus_count", 0)),
            allocated_budget=int(raw_budgets[i]),
            priority_rank=i + 1,
            verdict=str(row.get("verdict", "")),
            top_horse=str(row.get("top_horse", "")),
            top_ev=float(row.get("top_ev", 0.0)) if pd.notna(row.get("top_ev")) else 0.0,
        )
        allocations.append(alloc)

    return allocations


def format_allocation_table(allocations: list[RaceAllocation]) -> pd.DataFrame:
    """配分結果を表示用DataFrameに変換する。"""
    if not allocations:
        return pd.DataFrame()
    rows = []
    for a in allocations:
        rows.append({
            "優先": f"#{a.priority_rank}",
            "日付": a.date_str,
            "レース名": a.race_name,
            "会場/馬場": f"{a.venue}{a.surface}",
            "距離": f"{a.distance}m",
            "スコア": a.race_score,
            "判定": a.verdict,
            "注目馬": a.top_horse,
            "注目EV": f"{a.top_ev:+.3f}" if a.top_ev else "-",
            "配分予算": f"{a.allocated_budget:,}円",
        })
    return pd.DataFrame(rows)


def get_remaining_budget(
    allocations: list[RaceAllocation],
    weekly_budget: int,
) -> dict:
    """週予算の残高サマリーを返す。"""
    total_allocated = sum(a.allocated_budget for a in allocations)
    remaining = weekly_budget - total_allocated
    return {
        "weekly_budget": weekly_budget,
        "total_allocated": total_allocated,
        "remaining": remaining,
        "n_races": len(allocations),
        "avg_per_race": total_allocated // len(allocations) if allocations else 0,
    }


def suggest_single_race_budget(
    race_score: int,
    weekly_budget: int,
    races_this_week: int,
) -> int:
    """
    単一レースの推奨予算を計算する（スコア × 調整係数）。
    週のレース数が少ないほど1レースに多く使える。
    """
    base = weekly_budget / max(races_this_week, 1)
    multiplier = 1.0
    if race_score >= 80:
        multiplier = 1.4
    elif race_score >= 65:
        multiplier = 1.2
    elif race_score >= 50:
        multiplier = 1.0
    else:
        multiplier = 0.6

    suggested = int(base * multiplier / 100) * 100
    # 最大は週予算の50%まで
    max_budget = int(weekly_budget * 0.5 / 100) * 100
    return min(suggested, max_budget)


def render_bankroll_section(
    allocations: list[RaceAllocation],
    weekly_budget: int,
) -> None:
    """
    資金管理セクションをレンダリングする。
    app.py のトップページまたはバンクロールタブから呼び出す。
    """
    import streamlit as st
    import plotly.express as px

    if not allocations:
        st.info("レーススキャンを実行すると、週末の最適資金配分が表示されます。")
        return

    summary = get_remaining_budget(allocations, weekly_budget)

    # サマリーメトリクス
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("週予算", f"{summary['weekly_budget']:,}円")
    c2.metric("配分済み", f"{summary['total_allocated']:,}円")
    c3.metric("残高", f"{summary['remaining']:,}円")
    c4.metric("狙いレース数", f"{summary['n_races']}レース")

    # 配分テーブル
    st.dataframe(format_allocation_table(allocations),
                 use_container_width=True, hide_index=True)

    # 円グラフ
    if len(allocations) > 1:
        fig = px.pie(
            values=[a.allocated_budget for a in allocations],
            names=[f"#{a.priority_rank} {a.race_name[:10]}" for a in allocations],
            title="週末予算の配分",
        )
        st.plotly_chart(fig, use_container_width=True)
