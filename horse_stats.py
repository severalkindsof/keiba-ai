"""
horse_stats.py
50万件のTFJVデータから各馬の個別指標を算出するモジュール。

算出指標：
- 同一条件（距離帯×馬場）での複勝率・5着以内率
- 道悪適性（重・不良での複勝率）
- 上がり3F優秀率（同レース内1-2位頻度）
- 年齢・鮮度ペナルティ
- コース直線長から差し有利判定
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ============================================================
# コース直線長テーブル（メートル）
# ============================================================
STRAIGHT_LENGTH = {
    # 芝
    ("新潟",  "芝"): 658.7,   # 最長、差し・追込天国
    ("東京",  "芝"): 525.9,
    ("阪神",  "芝"): 473.6,
    ("中京",  "芝"): 412.5,
    ("京都",  "芝"): 404.3,
    ("函館",  "芝"): 262.1,
    ("札幌",  "芝"): 264.0,
    ("福島",  "芝"): 295.7,
    ("中山",  "芝"): 310.0,
    ("小倉",  "芝"): 293.0,
    # ダート
    ("東京",  "ダート"): 501.6,
    ("阪神",  "ダート"): 352.0,
    ("中京",  "ダート"): 410.0,
    ("新潟",  "ダート"): 353.9,
    ("京都",  "ダート"): 329.1,
    ("中山",  "ダート"): 310.0,
    ("福島",  "ダート"): 295.7,
    ("札幌",  "ダート"): 264.0,
    ("函館",  "ダート"): 262.1,
    ("小倉",  "ダート"): 291.0,
}
# 直線長 > 400m → 差し・追込有利
DASH_ADVANTAGE_THRESHOLD = 400.0

def get_finishing_style_advantage(venue: str, surface: str) -> str:
    """コース直線長から有利な脚質タイプを返す"""
    key = (venue, surface)
    length = STRAIGHT_LENGTH.get(key, 350.0)
    if length >= 500:
        return "差し・追込大有利"
    elif length >= 400:
        return "差し有利"
    elif length >= 320:
        return "フラット"
    else:
        return "逃げ・先行有利"


# ============================================================
# 短期外国人ジョッキーリスト
# ============================================================
FOREIGN_SHORT_TERM_JOCKEYS = {
    # 常連・実績あり
    "モレイラ", "ジョアオモレイラ", "Ｊ．モレイラ",
    "レーン", "ダミアンレーン", "Ｄ．レーン",
    "シュタルケ", "ムルザバエフ", "Ｂ．ムルザバエフ",
    "レイチェルキング", "Ｒ．キング",
    "マイケルディー", "Ｍ．ディー",
    "ゴンサルベス", "Ｒ．ゴンサルベス",
    "ビュイック", "Ｗ．ビュイック",
    "マーカンド", "Ｔ．マーカンド",
    "ドイル", "Ｊ．ドイル",
    "ハリス",
    # 追加候補（随時更新）
}

def is_foreign_jockey(jockey_name: str) -> bool:
    """短期外国人ジョッキーかどうか判定"""
    if not jockey_name:
        return False
    name = jockey_name.strip()
    # リスト完全一致
    if name in FOREIGN_SHORT_TERM_JOCKEYS:
        return True
    # 部分一致（カタカナ＋ドット形式）
    for fj in FOREIGN_SHORT_TERM_JOCKEYS:
        if fj in name or name in fj:
            return True
    return False


# ============================================================
# 馬別指標の計算
# ============================================================

_HS_PARQUET = Path(__file__).parent / "data" / "horse_stats.parquet"

import streamlit as st

@st.cache_resource
def _load_horse_stats_parquet():
    print("[horse_stats] Parquetから読み込み中...")
    df_hs = pd.read_parquet(_HS_PARQUET)
    return df_hs.set_index("horse_name")

def build_horse_stats(df: pd.DataFrame) -> pd.DataFrame:
    """horse_stats.parquetから即時読み込み（初回のみ、以降はキャッシュ）"""
    if _HS_PARQUET.exists():
        return _load_horse_stats_parquet()
    return _build_vectorized(df)


def _build_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """ベクトル化された馬別指標計算（ループなし）"""
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["horse_name"] = df["horse_name"].str.strip()

    # --- 上がり3F 同レース内順位（ベクトル化）---
    df["last3f_rank"] = df.groupby("race_id")["last_3f"].rank(
        method="min", ascending=True, na_option="bottom"
    )
    df["last3f_top2"] = (df["last3f_rank"] <= 2).astype("int8")
    df["top5_flag"]   = (df["rank"] <= 5).astype("int8")

    # --- 基本集計（全条件）---
    base = df.groupby("horse_name", sort=False).agg(
        total_races   = ("rank",        "count"),
        place_count   = ("place_flag",  "sum"),
        top5_count    = ("top5_flag",   "sum"),
        last3f_top2_c = ("last3f_top2", "sum"),
        last_date     = ("date",        "max"),
        age           = ("age",         "last"),
    )
    base["place_rate"]      = base["place_count"]   / base["total_races"]
    base["top5_rate"]       = base["top5_count"]    / base["total_races"]
    base["last3f_rate"]     = base["last3f_top2_c"] / base["total_races"]
    base["days_since_last"] = (
        pd.Timestamp.now() - pd.to_datetime(base["last_date"], errors="coerce")
    ).dt.days.fillna(999).astype(int)

    # --- 道悪複勝率 ---
    wet = df[df["track_condition"].isin(["重", "不良"])].groupby("horse_name", sort=False).agg(
        wet_races  = ("place_flag", "count"),
        wet_places = ("place_flag", "sum"),
    )
    wet = wet[wet["wet_races"] >= 2]
    wet["wet_place_rate"] = wet["wet_places"] / wet["wet_races"]
    base = base.join(wet["wet_place_rate"], how="left")

    # --- 距離帯別複勝率（dictとして保持）---
    dist_grp = df.groupby(["horse_name", "distance_cat"], sort=False, observed=True).agg(
        dr = ("place_flag", "count"),
        dp = ("place_flag", "sum"),
    ).reset_index()
    dist_grp = dist_grp[dist_grp["dr"] >= 2]
    dist_grp["drate"] = dist_grp["dp"] / dist_grp["dr"]
    dist_dict = (
        dist_grp.groupby("horse_name")
        .apply(lambda x: dict(zip(x["distance_cat"].astype(str), x["drate"])))
    )
    base = base.join(dist_dict.rename("dist_stats"), how="left")

    stats_df = base
    return stats_df


def get_horse_score_bonus(
    horse_name: str,
    horse_stats: pd.DataFrame,
    distance_cat: str,
    surface: str,
    track_condition: str,
    jockey: str,
    venue: str,
    running_style: str = "不明",
) -> dict:
    """
    個別馬スコアのボーナス/ペナルティを計算する。
    Returns: {"bonus": float, "details": dict}
    """
    bonus = 0.0
    details = {}

    # --- 短期外国人ジョッキーボーナス ---
    if is_foreign_jockey(jockey):
        bonus += 0.25
        details["外国人騎手"] = "+0.25"

    if horse_stats.empty or horse_name not in horse_stats.index:
        # 直線長ボーナスだけ計算して返す
        style_adv = get_finishing_style_advantage(venue, surface)
        details["コース特性"] = style_adv
        return {"bonus": bonus, "details": details}

    row = horse_stats.loc[horse_name]
    age          = float(row.get("age", 0))
    days_since   = int(row.get("days_since_last", 0))
    place_rate   = float(row.get("place_rate", 0))
    top5_rate    = float(row.get("top5_rate", 0))
    last3f_rate  = float(row.get("last3f_rate", 0))
    wet_rate     = row.get("wet_place_rate")
    dist_stats    = row.get("dist_stats") or {}
    if not isinstance(dist_stats, dict): dist_stats = {}
    surface_stats = row.get("surface_stats") or {}
    if not isinstance(surface_stats, dict): surface_stats = {}

    # --- 年齢ペナルティ ---
    if age >= 7 and days_since > 365:
        bonus -= 0.15
        details["年齢ペナルティ"] = f"-0.15（{int(age)}歳、直近成績{days_since}日前）"
    elif age >= 8:
        bonus -= 0.10
        details["高齢馬"] = f"-0.10（{int(age)}歳）"

    # --- 同一距離帯複勝率ボーナス ---
    dist_rate = dist_stats.get(distance_cat, None)
    if dist_rate is not None:
        if dist_rate >= 0.40:
            bonus += 0.20
            details["距離適性◎"] = f"+0.20（{distance_cat}複勝率{dist_rate:.0%}）"
        elif dist_rate >= 0.25:
            bonus += 0.10
            details["距離適性○"] = f"+0.10（{distance_cat}複勝率{dist_rate:.0%}）"
        elif dist_rate < 0.10:
            bonus -= 0.10
            details["距離苦手"] = f"-0.10（{distance_cat}複勝率{dist_rate:.0%}）"

    # --- 道悪適性 ---
    if track_condition in ("重", "不良") and wet_rate is not None:
        if wet_rate >= 0.40:
            bonus += 0.20
            details["道悪◎"] = f"+0.20（道悪複勝率{wet_rate:.0%}）"
        elif wet_rate >= 0.25:
            bonus += 0.10
            details["道悪○"] = f"+0.10（道悪複勝率{wet_rate:.0%}）"
        elif wet_rate < 0.10:
            bonus -= 0.15
            details["道悪苦手"] = f"-0.15（道悪複勝率{wet_rate:.0%}）"

    # --- 上がり3F優秀率 ---
    if last3f_rate >= 0.35:
        bonus += 0.15
        details["末脚◎"] = f"+0.15（上がり1-2位率{last3f_rate:.0%}）"
    elif last3f_rate >= 0.20:
        bonus += 0.08
        details["末脚○"] = f"+0.08（上がり1-2位率{last3f_rate:.0%}）"

    # --- コース直線長 × 脚質マッチ ---
    style_adv = get_finishing_style_advantage(venue, surface)
    if style_adv in ("差し・追込大有利", "差し有利") and running_style in ("差し", "追込", "差し・追込"):
        bonus += 0.12
        details["コース×脚質◎"] = f"+0.12（{style_adv}×{running_style}）"
    elif style_adv in ("逃げ・先行有利") and running_style in ("逃げ", "先行"):
        bonus += 0.10
        details["コース×脚質◎"] = f"+0.10（{style_adv}×{running_style}）"
    details["コース特性"] = style_adv

    # --- 全体複勝率 ---
    if place_rate >= 0.35 and row.get("total_races", 0) >= 5:
        bonus += 0.10
        details["高複勝率"] = f"+0.10（複勝率{place_rate:.0%}）"

    return {"bonus": round(bonus, 3), "details": details}
