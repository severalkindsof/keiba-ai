"""
Kaggle JRA CSVデータの読み込みと前処理。
データセット: https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset
"""
import json
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent / "data"
HORSE_CACHE_DIR = Path(__file__).parent / "sessions" / "horse_cache"

# Google Drive フォルダID（Kaggleデータ置き場）
GDRIVE_FOLDER_ID = "1g6TvkHtM5Ubs8HjC3JeeKM2Y5IDzxwjR"

def _download_from_gdrive():
    """Google DriveからKaggle CSVを自動ダウンロードする（初回のみ）"""
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / "19860105-20210731_race_result.csv"
    if target.exists():
        return  # 既にある場合はスキップ

    try:
        import gdown
        print("[data_loader] Google Driveからダウンロード中...")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
            output=str(DATA_DIR),
            quiet=False,
        )
        print("[data_loader] ダウンロード完了")
    except Exception as e:
        print(f"[data_loader] ダウンロード失敗: {e}")

# 距離カテゴリ
def categorize_distance(dist):
    if dist <= 1400:
        return "短距離"
    elif dist <= 1800:
        return "マイル"
    elif dist <= 2200:
        return "中距離"
    else:
        return "長距離"

@st.cache_data(ttl=3600)
def load_race_results() -> pd.DataFrame:
    """レース結果CSVを読み込む（なければGoogle Driveから自動DL）"""
    # 注意: @st.cache_data 内では st.* 呼び出し禁止のため print のみ使用
    try:
        DATA_DIR.mkdir(exist_ok=True)
        candidates = list(DATA_DIR.glob("*.csv"))
        if not candidates:
            _download_from_gdrive()
            candidates = list(DATA_DIR.glob("*.csv"))
        if not candidates:
            print("[data_loader] CSVなし - 空DataFrameを返す")
            return pd.DataFrame()

        # 必要な列のみ読み込む（Kaggle JRA CSV の実際の列名）
        # 472MBを全列読むとメモリ超過するため usecols で絞る
        NEEDED_COLS = [
            "着順", "馬名", "距離(m)", "芝・ダート区分",
            "馬場状態1", "人気", "単勝", "騎手", "上り",
            "枠番", "馬番", "競馬場名", "レース日付",
            "馬体重", "場体重増減", "4コーナー",
            # 英語列名（他データセット用フォールバック）
            "rank", "horse_name", "distance", "surface",
            "track_condition", "popularity", "odds", "jockey",
            "last_3f", "gate", "horse_no", "venue", "date",
            "horse_weight", "weight_change", "corner_order",
        ]

        def _read_csv_slim(path):
            """必要な列だけ読み込む。列が存在しない場合は無視。"""
            # まずヘッダーだけ読んで存在する列を確認
            header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
            usecols = [c for c in NEEDED_COLS if c in header.columns]
            if not usecols:
                # 列名が全く合わない場合は全列読む（小さいCSVのみ）
                size_mb = path.stat().st_size / 1024 / 1024
                if size_mb > 100:
                    print(f"[data_loader] {path.name}: 列名不一致かつ大ファイル({size_mb:.0f}MB) → スキップ")
                    return None
                return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            print(f"[data_loader] {path.name}: {len(usecols)}列を選択読み込み")
            return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols, low_memory=False)

        priority = [
            "race_results", "results", "races",
            "19860105-20210731_race_result",
            "race_result",
        ]
        df = None
        for name in priority:
            path = DATA_DIR / f"{name}.csv"
            if path.exists():
                try:
                    df = _read_csv_slim(path)
                    if df is not None:
                        print(f"[data_loader] 読み込み成功: {path.name} ({len(df)}行 × {len(df.columns)}列)")
                        break
                except Exception as e:
                    print(f"[data_loader] {path.name} 読み込み失敗: {e}")
                    continue

        if df is None:
            try:
                # race_result を含むファイル名を優先
                race_files = [f for f in candidates if "race_result" in f.name]
                target = race_files[0] if race_files else max(candidates, key=lambda f: f.stat().st_size)
                df = _read_csv_slim(target)
                if df is None:
                    df = pd.DataFrame()
                else:
                    print(f"[data_loader] フォールバック使用: {target.name} ({len(df)}行)")
            except Exception as e:
                print(f"[data_loader] CSV読み込み失敗: {e}")
                return pd.DataFrame()

        df = _normalize_columns(df)
        df = _add_derived_columns(df)
        return df
    except Exception as e:
        print(f"[data_loader] 致命的エラー: {e}")
        return pd.DataFrame()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名を統一する（Kaggle JRA CSV・netkeiba両対応）"""
    # Kaggle JRA Dataset の正確な列名を優先マッピング
    # 実際の列名: 芝・ダート区分, 上り, 距離(m), 馬場状態1, 競馬場名, レース日付 など
    exact_map = {
        "芝・ダート区分":  "surface",
        "芝・ダート区分2": "surface_alt",
        "距離(m)":        "distance",
        "上り":           "last_3f",
        "馬場状態1":      "track_condition",
        "馬場状態2":      "track_condition2",
        "競馬場名":       "venue",
        "レース日付":     "date",
        "競争条件":       "race_class",
        "4コーナー":      "corner_order",
        "着順":           "rank",
        "馬名":           "horse_name",
        "枠番":           "gate",
        "馬番":           "horse_no",
        "斤量":           "weight_carried",
        "騎手":           "jockey",
        "タイム":         "time",
        "着差":           "margin",
        "単勝":           "odds",
        "人気":           "popularity",
        "馬体重":         "horse_weight",
        "場体重増減":     "weight_change",
        "調教師":         "trainer",
        "レース名":       "race_name",
    }
    # 完全一致マッピングを先に適用
    df = df.rename(columns={k: v for k, v in exact_map.items() if k in df.columns and v not in df.columns})

    # 部分一致フォールバック（netkeiba scraped data など）
    partial_map = {
        "着順": "rank",
        "馬名": "horse_name",
        "性齢": "sex_age",
        "斤量": "weight_carried",
        "騎手": "jockey",
        "タイム": "time",
        "着差": "margin",
        "オッズ": "odds",
        "人気": "popularity",
        "馬体重": "horse_weight",
        "調教師": "trainer",
        "父": "sire",
        "母父": "dam_sire",
        "レース名": "race_name",
        "距離": "distance",
        "馬場": "surface",
        "馬場状態": "track_condition",
        "日付": "date",
        "場名": "venue",
        "クラス": "race_class",
        "上がり": "last_3f",
        "コーナー": "corner_order",
        "枠番": "gate",
        "枠": "gate",
        "馬番": "horse_no",
        "前走クラス": "prev_class",
        "クラス名": "race_class",
    }
    cols = df.columns.tolist()
    for old, new in partial_map.items():
        matched = [c for c in cols if old in c]
        if matched and new not in df.columns:
            df = df.rename(columns={matched[0]: new})

    # surface の値を正規化（"ダ" → "ダート"）
    if "surface" in df.columns:
        df["surface"] = df["surface"].astype(str).str.strip()
        df["surface"] = df["surface"].replace({"ダ": "ダート", "D": "ダート", "T": "芝", "1": "芝", "2": "ダート"})

    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """分析用の派生列を追加"""
    if "distance" in df.columns:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df["distance_cat"] = df["distance"].apply(
            lambda x: categorize_distance(x) if pd.notna(x) else "不明"
        )
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["win_flag"] = (df["rank"] == 1).astype(int)
        df["place_flag"] = (df["rank"] <= 3).astype(int)
    if "odds" in df.columns:
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        df["implied_prob"] = 1.0 / df["odds"].replace(0, np.nan)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def get_win_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    条件別勝率テーブルを構築。
    グループ: surface × distance_cat × popularity_bucket
    """
    if df.empty or "win_flag" not in df.columns:
        return pd.DataFrame()

    cols_needed = ["surface", "distance_cat", "popularity", "win_flag", "place_flag"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    df2 = df.copy()
    df2["popularity"] = pd.to_numeric(df2["popularity"], errors="coerce")
    df2["pop_bucket"] = pd.cut(
        df2["popularity"],
        bins=[0, 3, 6, 9, 18],
        labels=["1〜3番人気", "4〜6番人気", "7〜9番人気", "10番人気以下"],
    )

    tbl = (
        df2.groupby(["surface", "distance_cat", "pop_bucket"], observed=True)
        .agg(
            races=("win_flag", "count"),
            wins=("win_flag", "sum"),
            places=("place_flag", "sum"),
        )
        .reset_index()
    )
    tbl["win_rate"] = tbl["wins"] / tbl["races"]
    tbl["place_rate"] = tbl["places"] / tbl["races"]
    return tbl


@st.cache_data(ttl=3600)
def get_sire_stats(df: pd.DataFrame) -> pd.DataFrame:
    """父系別・距離帯別の勝率（血統分析用）"""
    if "sire" not in df.columns or df.empty:
        return pd.DataFrame()
    tbl = (
        df.groupby(["sire", "distance_cat"], observed=True)
        .agg(races=("win_flag", "count"), wins=("win_flag", "sum"))
        .reset_index()
    )
    tbl = tbl[tbl["races"] >= 10]  # サンプル数が少なすぎる組み合わせを除外
    tbl["win_rate"] = tbl["wins"] / tbl["races"]
    return tbl.sort_values("win_rate", ascending=False)


@st.cache_data(ttl=3600)
def get_jockey_stats(df: pd.DataFrame) -> pd.DataFrame:
    """騎手別・人気帯別の複勝率（穴騎手リスト用）"""
    if "jockey" not in df.columns or df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    df2["popularity"] = pd.to_numeric(df2["popularity"], errors="coerce")
    df2["is_longshot"] = df2["popularity"] >= 10
    longshot = (
        df2[df2["is_longshot"]]
        .groupby("jockey")
        .agg(rides=("win_flag", "count"), wins=("win_flag", "sum"), places=("place_flag", "sum"))
        .reset_index()
    )
    longshot = longshot[longshot["rides"] >= 20]
    longshot["place_rate_longshot"] = longshot["places"] / longshot["rides"]
    longshot["win_rate_longshot"] = longshot["wins"] / longshot["rides"]
    return longshot.sort_values("place_rate_longshot", ascending=False)


# ---- netkeibaスクレイピング結果のキャッシュ管理 ---- #

def save_horse_cache(horse_id: str, horse_name: str, df: pd.DataFrame) -> None:
    """馬の過去成績をJSONキャッシュに保存する（sessions/horse_cache/）"""
    HORSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = HORSE_CACHE_DIR / f"{horse_id}.json"
    payload = {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "fetched_at": datetime.now().isoformat(),
        "records": df.to_dict(orient="records"),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
    )


def load_horse_cache(horse_id: str, ttl_days: int = 7) -> pd.DataFrame | None:
    """
    キャッシュから馬の過去成績を読み込む。
    TTL切れまたはファイルなしの場合は None を返す。
    """
    path = HORSE_CACHE_DIR / f"{horse_id}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now() - fetched_at > timedelta(days=ttl_days):
            return None
        records = payload.get("records", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return _add_derived_columns(df)
    except Exception:
        return None


def load_all_cached_horses() -> pd.DataFrame:
    """sessions/horse_cache/ の全キャッシュを結合して返す"""
    if not HORSE_CACHE_DIR.exists():
        return pd.DataFrame()
    dfs = []
    for p in HORSE_CACHE_DIR.glob("*.json"):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            records = payload.get("records", [])
            if records:
                dfs.append(pd.DataFrame(records))
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return _add_derived_columns(pd.concat(dfs, ignore_index=True))


def merge_with_horse_cache(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggleデータにキャッシュされたnetkeibaデータをマージして返す。
    - Kaggleに存在しない馬: 全行追加
    - Kaggleに存在する馬: 2022年以降の行のみ追加（Kaggleは2021年まで）
    """
    _KAGGLE_CUTOFF = pd.Timestamp("2022-01-01")

    cache_df = load_all_cached_horses()
    if cache_df.empty:
        return base_df
    if base_df.empty:
        return cache_df
    if "horse_name" not in base_df.columns or "horse_name" not in cache_df.columns:
        return base_df

    known = set(base_df["horse_name"].unique())

    # ① Kaggleに全くいない馬は全行追加
    completely_new = cache_df[~cache_df["horse_name"].isin(known)]

    # ② Kaggleにいる馬は2022年以降の行のみ追加
    existing_in_kaggle = cache_df[cache_df["horse_name"].isin(known)].copy()
    if not existing_in_kaggle.empty and "date" in existing_in_kaggle.columns:
        existing_in_kaggle["date"] = pd.to_datetime(existing_in_kaggle["date"], errors="coerce")
        new_races = existing_in_kaggle[existing_in_kaggle["date"] >= _KAGGLE_CUTOFF]
    else:
        new_races = pd.DataFrame()

    parts = [base_df]
    if not completely_new.empty:
        parts.append(completely_new)
    if not new_races.empty:
        parts.append(new_races)

    if len(parts) == 1:
        return base_df
    return pd.concat(parts, ignore_index=True)
