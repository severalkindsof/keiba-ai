"""
枠順バイアス分析モジュール。

アプローチ:
1. 静的テーブル：JRA各会場のコース・距離別の枠有利不利（定性的な傾向）
2. 動的計算：Kaggleの過去データから枠番別の勝率を実際に算出して補正
"""
import pandas as pd
import streamlit as st  # CLEAN: numpy 未使用のため削除


# ---- 静的バイアステーブル（調査ベース） ---- #
# bias: +2=強い内有利, +1=内有利, 0=フラット, -1=外有利, -2=強い外有利
# ダートは砂かぶり回避で外枠有利、芝は内枠ロス少なく内有利が基本

STATIC_BIAS = {
    # (会場, 馬場, 距離カテゴリ): (偏り方向, 強度)
    # 偏り方向: "inner"=内有利, "outer"=外有利, "flat"=フラット
    ("中山", "芝", "短距離"): ("inner", 2),   # 中山芝1200mは特に内有利
    ("中山", "芝", "マイル"):  ("inner", 1),
    ("中山", "芝", "中距離"): ("inner", 1),
    ("中山", "芝", "長距離"): ("flat",  0),
    ("中山", "ダート", "短距離"): ("outer", 1),
    ("中山", "ダート", "マイル"):  ("outer", 1),
    ("東京", "芝", "短距離"): ("inner", 1),
    ("東京", "芝", "マイル"):  ("flat",  0),   # 東京芝1600mはフラット
    ("東京", "芝", "中距離"): ("flat",  0),
    ("東京", "芝", "長距離"): ("flat",  0),
    ("東京", "ダート", "短距離"): ("outer", 1),
    ("東京", "ダート", "マイル"):  ("outer", 2),  # 東京ダート1600mは外枠強い
    ("阪神", "芝", "短距離"): ("inner", 2),
    ("阪神", "芝", "マイル"):  ("inner", 1),
    ("阪神", "芝", "中距離"): ("inner", 1),
    ("阪神", "ダート", "短距離"): ("outer", 1),
    ("京都", "芝", "中距離"): ("outer", 1),   # 京都外回りは外枠有利
    ("京都", "芝", "長距離"): ("flat",  0),
    ("中京", "芝", "マイル"):  ("outer", 1),
    ("中京", "ダート", "マイル"):  ("outer", 1),
    ("小倉", "芝", "短距離"): ("inner", 2),   # 小倉は小回りで内有利
    ("小倉", "芝", "マイル"):  ("inner", 1),
    ("福島", "芝", "短距離"): ("inner", 2),
    ("函館", "芝", "短距離"): ("inner", 1),
    ("札幌", "芝", "マイル"):  ("inner", 1),
    ("新潟", "芝", "短距離"): ("outer", 2),   # 新潟芝1000mは外枠有利
    ("新潟", "芝", "マイル"):  ("flat",  0),
}

N_BUCKETS = 3  # 内枠/中枠/外枠 の3分割


def get_draw_bucket(gate: int, total_horses: int) -> str:
    """頭数に応じて内/中/外枠を分類"""
    if total_horses <= 0:
        return "中"
    ratio = gate / total_horses
    if ratio <= 0.33:
        return "内"
    elif ratio <= 0.67:
        return "中"
    else:
        return "外"


def static_draw_bonus(
    venue: str,
    surface: str,
    distance_cat: str,
    gate: int,
    total_horses: int,
) -> float:
    """
    静的テーブルから枠順補正値を返す（勝率への加算値）。
    内枠有利なら内枠馬にプラス、外枠にマイナス、外枠有利ならその逆。
    """
    key = (venue, surface, distance_cat)
    bias_dir, strength = STATIC_BIAS.get(key, ("flat", 0))

    if strength == 0:
        return 0.0

    bucket = get_draw_bucket(gate, total_horses)
    base = strength * 0.01  # 1強度あたり1%の勝率差

    if bias_dir == "inner":
        if bucket == "内":
            return +base
        elif bucket == "中":
            return 0.0
        else:
            return -base
    elif bias_dir == "outer":
        if bucket == "外":
            return +base
        elif bucket == "中":
            return 0.0
        else:
            return -base
    return 0.0


@st.cache_data(ttl=3600)
def build_dynamic_draw_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去データから会場×馬場×距離帯×枠番の実際の勝率テーブルを構築する。
    （静的テーブルより精度が高いが、データが必要）
    """
    required = ["venue", "surface", "distance_cat", "gate", "win_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing or df.empty:
        return pd.DataFrame()

    df2 = df.dropna(subset=required).copy()
    df2["gate"] = pd.to_numeric(df2["gate"], errors="coerce")
    total_by_race = df2.groupby(["venue", "surface", "distance_cat"])["gate"].transform("max")
    df2["gate_bucket"] = [
        get_draw_bucket(int(g), int(t)) if pd.notna(g) and pd.notna(t) else "中"
        for g, t in zip(df2["gate"], total_by_race)
    ]

    tbl = (
        df2.groupby(["venue", "surface", "distance_cat", "gate_bucket"], observed=True)
        .agg(races=("win_flag", "count"), wins=("win_flag", "sum"))
        .reset_index()
    )
    tbl = tbl[tbl["races"] >= 20]
    tbl["win_rate"] = tbl["wins"] / tbl["races"]
    return tbl


def dynamic_draw_bonus(
    draw_table: pd.DataFrame,
    venue: str,
    surface: str,
    distance_cat: str,
    gate: int,
    total_horses: int,
) -> float:
    """動的テーブルから枠順補正値を返す。テーブルがなければ0。"""
    if draw_table.empty:
        return 0.0
    bucket = get_draw_bucket(gate, total_horses)
    key_rows = draw_table[
        (draw_table["venue"] == venue)
        & (draw_table["surface"] == surface)
        & (draw_table["distance_cat"] == distance_cat)
    ]
    if key_rows.empty:
        return 0.0
    overall_wr = key_rows["win_rate"].mean()
    target = key_rows[key_rows["gate_bucket"] == bucket]
    if target.empty:
        return 0.0
    target_wr = float(target["win_rate"].iloc[0])
    return round(target_wr - overall_wr, 4)


def get_draw_bonus(
    draw_table: pd.DataFrame,
    venue: str,
    surface: str,
    distance_cat: str,
    gate: int,
    total_horses: int,
) -> float:
    """動的テーブル優先、なければ静的テーブルを使う"""
    dynamic = dynamic_draw_bonus(draw_table, venue, surface, distance_cat, gate, total_horses)
    if dynamic != 0.0:
        return dynamic
    return static_draw_bonus(venue, surface, distance_cat, gate, total_horses)


def get_draw_label(venue: str, surface: str, distance_cat: str, gate: int, total_horses: int) -> str:
    """人間が読めるラベルを返す"""
    key = (venue, surface, distance_cat)
    bias_dir, strength = STATIC_BIAS.get(key, ("flat", 0))
    bucket = get_draw_bucket(gate, total_horses)
    if strength == 0:
        return "フラット"
    favored = "内" if bias_dir == "inner" else "外"
    if bucket == favored:
        return f"{'◎' if strength >= 2 else '○'} {bucket}枠有利"
    elif bucket == "中":
        return "△ 中枠"
    else:
        return f"{'▼' if strength >= 2 else '▲'} {bucket}枠不利"
