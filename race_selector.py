"""
全レース横断選択モジュール。

土日の全JRAレース（平場含む）をスキャンして
「今週最もEV・Confluenceが高いレース」をランキングで提示する。
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from scraper import fetch_today_races, fetch_race_entries, fetch_race_meta
from ev_calculator import evaluate_race
from confluence import get_race_quality_score, add_confluence_to_eval


def scan_weekend_races(
    dates: list[str],
    win_rate_table: pd.DataFrame,
    sire_stats: pd.DataFrame,
    jockey_stats: pd.DataFrame,
    max_races: int = 30,
    progress_placeholder=None,
) -> pd.DataFrame:
    """
    指定日程の全JRAレースをスキャンして、レース別の狙い目スコアを返す。
    progress_placeholder: st.empty() を渡すとリアルタイム進捗を表示する。
    """
    all_races = []
    for date_str in dates:
        if progress_placeholder:
            progress_placeholder.info(f"📡 {date_str} のレース一覧を取得中...")
        races = fetch_today_races(date_str)
        for r in races:
            r["date_str"] = date_str
        all_races.extend(races)

    if not all_races:
        if progress_placeholder:
            progress_placeholder.warning("レース情報が取得できませんでした。出走表が公開されていない可能性があります（通常は木曜〜金曜に公開）。")
        return pd.DataFrame()

    total = min(len(all_races), max_races)
    if progress_placeholder:
        progress_placeholder.info(f"📋 {total}レースをスキャン開始...")

    results = []
    for i, race in enumerate(all_races[:max_races]):
        if progress_placeholder:
            pct = int((i + 1) / total * 100)
            progress_placeholder.info(
                f"🔍 スキャン中 {i+1}/{total}件 ({pct}%)  \n"
                f"「{race.get('race_name', '')}」を分析中..."
            )
        try:
            entries = fetch_race_entries(race["race_id"])
            meta = fetch_race_meta(race["race_id"])
            if not entries:
                continue

            surface = meta.get("surface", "芝")
            distance = meta.get("distance", 2000)
            condition = meta.get("track_condition", "良")

            for e in entries:
                e["surface"] = surface
                e["distance"] = distance
                e["track_condition"] = condition

            eval_df = evaluate_race(entries, win_rate_table, sire_stats, jockey_stats)
            if eval_df.empty:
                continue

            scored = add_confluence_to_eval(eval_df)
            quality = get_race_quality_score(scored)

            top_row = scored.iloc[0] if not scored.empty else {}
            results.append({
                "race_id": race["race_id"],
                "race_name": race.get("race_name", ""),
                "date_str": race.get("date_str", ""),
                "surface": surface,
                "distance": distance,
                "track_condition": condition,
                "race_score": quality["race_score"],
                "top_score": quality["top_score"],
                "ev_plus_count": quality["ev_plus_count"],
                "verdict": quality["verdict"],
                "top_horse": top_row.get("horse_name", ""),
                "top_confidence": top_row.get("confidence_score", 0),
                "top_ev": top_row.get("ev", float("nan")),
            })
        except Exception:
            continue

    if not results:
        if progress_placeholder:
            progress_placeholder.warning("出走表の取得に失敗しました。出走表が公開前か、通信エラーの可能性があります。")
        return pd.DataFrame()

    if progress_placeholder:
        progress_placeholder.success(f"✅ スキャン完了！ {len(results)}レースを分析しました")

    df = pd.DataFrame(results)
    df = df.sort_values("race_score", ascending=False).reset_index(drop=True)
    return df


def get_this_weekend_dates() -> list[str]:
    """今週末（土日）の日付を返す"""
    today = date.today()
    weekday = today.weekday()  # 0=月, 5=土, 6=日

    if weekday == 5:  # 今日が土曜
        saturday = today
    elif weekday == 6:  # 今日が日曜
        saturday = today - timedelta(days=1)
    else:
        # 平日 → 次の土曜
        days_until_sat = 5 - weekday
        saturday = today + timedelta(days=days_until_sat)

    sunday = saturday + timedelta(days=1)
    return [saturday.strftime("%Y%m%d"), sunday.strftime("%Y%m%d")]


def format_race_scan_display(df: pd.DataFrame) -> pd.DataFrame:
    """スキャン結果を表示用DataFrameに整形"""
    if df.empty:
        return df
    rename = {
        "race_name": "レース名",
        "date_str": "日付",
        "surface": "馬場",
        "distance": "距離",
        "race_score": "レーススコア",
        "verdict": "判定",
        "top_horse": "注目馬",
        "top_confidence": "注目馬スコア",
        "top_ev": "注目馬EV",
        "ev_plus_count": "EV+頭数",
    }
    cols = [c for c in rename if c in df.columns]
    return df[cols].rename(columns=rename)
