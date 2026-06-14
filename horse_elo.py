"""
SUPER-6: 馬 Elo レーティングシステム

各レースを「全頭の総当たり対戦」とみなし、レース結果に基づいて Elo を更新する。
Kovalchik (2020) の Margin-of-Victory 拡張も適用：着差が大きいほど Elo 更新量も大。

公式:
    R_new = R_old + K * MOV * (S - E)
        K   : 学習率（典型: 32）
        MOV : Margin-of-Victory 倍率（着差が大きいほど大）
        S   : 実際の結果（1=勝ち、0=負け）
        E   : 期待値 1 / (1 + 10^((R_opp - R_self)/400))

使い方:
    python horse_elo.py           # 一括計算 → data/horse_elo.parquet 保存
    from horse_elo import get_elo
    rating = get_elo("ホースネーム")  # 現在の Elo
"""
import pandas as pd
from pathlib import Path  # CLEAN: numpy 未使用のため削除

_DATA_DIR = Path(__file__).parent / "data"
_ELO_PATH = _DATA_DIR / "horse_elo.parquet"

INIT_RATING = 1500.0
K_FACTOR    = 32.0


def _expected_score(r_self: float, r_opp: float) -> float:
    """Elo 期待勝率"""
    return 1.0 / (1.0 + 10.0 ** ((r_opp - r_self) / 400.0))


def _mov_multiplier(time_diff: float | None) -> float:
    """着差秒数から Margin-of-Victory 倍率"""
    if time_diff is None or pd.isna(time_diff):
        return 1.0
    abs_diff = abs(float(time_diff))
    # 0秒=1.0, 0.5秒=1.3, 1秒=1.5, 2秒以上=1.8（上限）
    return float(min(1.8, 1.0 + abs_diff * 0.5))


def _horse_key(df: pd.DataFrame) -> pd.Series:
    """馬名衝突バグ対策の一意キー。horse_id を最優先。
    欠損/空のときのみ 馬名+生年(birth_date) で擬似一意化（別世代の同名馬を分離）。
    返り値は str の Series（df の index に整列）。"""
    if "horse_id" in df.columns:
        hid = df["horse_id"].astype(str).str.strip()
    else:
        hid = pd.Series([""] * len(df), index=df.index)
    bad = hid.isin(["", "nan", "None", "0", "<NA>"]) | hid.isna()
    if not bad.any():
        return hid  # horse_id が全て有効ならフォールバック計算は不要（高速）
    # 欠損行のみ 馬名+生年で擬似一意化（別世代の同名馬を分離）
    name = df["horse_name"].astype(str).str.strip()
    if "birth_date" in df.columns:
        yr = pd.to_datetime(df["birth_date"], errors="coerce").dt.year
    elif "date" in df.columns:
        yr = pd.to_datetime(df["date"], errors="coerce").dt.year
    else:
        yr = pd.Series([pd.NA] * len(df), index=df.index)
    fb = name + "@" + yr.astype("Int64").astype(str)
    return hid.where(~bad, fb)


def build_elo_table(
    tfjv_path: Path | None = None,
    out_path: Path | None = None,
    init_rating: float = INIT_RATING,
    k: float = K_FACTOR,
) -> pd.DataFrame:
    """
    tfjv_all.parquet の全レースを時系列順に処理して各馬の Elo を計算。
    """
    tfjv_path = tfjv_path or (_DATA_DIR / "tfjv_all.parquet")
    out_path  = out_path  or _ELO_PATH

    df = pd.read_parquet(tfjv_path)
    df = df.dropna(subset=["horse_name", "rank", "date"]).copy()
    df["horse_name"] = df["horse_name"].astype(str).str.strip()  # 元データの末尾空白でget_elo名前一致が外れるのを防ぐ
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # tfjv_all の race_id は 1行=1馬の連番なのでレースグルーピングに使えない
    # → (date, venue, race_no) を複合キーとして使う
    for col in ["venue", "race_no"]:
        if col not in df.columns:
            raise KeyError(f"必要な列がありません: {col}")
    df["race_key"] = (
        df["date"].dt.strftime("%Y%m%d") + "_" + df["venue"].astype(str)
        + "_" + df["race_no"].astype(str)
    )
    # 馬名衝突バグ対策: 集計は horse_id 基準（別世代の同名馬を分離）
    df["hid"] = _horse_key(df)
    df = df.sort_values(["date", "race_key", "rank"])
    print(f"  処理対象: {len(df):,}行 / 推定レース数: {df['race_key'].nunique():,}")

    ratings: dict[str, float] = {}
    last_race_date: dict[str, pd.Timestamp] = {}
    id_name: dict[str, str] = {}
    n_races = 0

    # レース単位で処理（複合キー）
    for race_id, race_df in df.groupby("race_key", sort=False):
        race_df = race_df.sort_values("rank")
        horses = race_df["hid"].tolist()          # horse_id基準キー
        names = race_df["horse_name"].tolist()
        ranks = race_df["rank"].astype(int).tolist()
        time_diffs = race_df.get("time_diff", pd.Series([0]*len(race_df))).tolist()
        race_date = race_df["date"].iloc[0]
        # 現在のレーティング取得
        cur = {h: ratings.get(h, init_rating) for h in horses}
        # 1着馬 vs 他全頭 / 2着 vs 3着以下 ... の総当たり処理
        new_ratings = {h: cur[h] for h in horses}
        for i in range(len(horses)):
            for j in range(i + 1, len(horses)):
                h_i, h_j = horses[i], horses[j]
                r_i, r_j = cur[h_i], cur[h_j]
                # 着順比較
                s_i = 1.0 if ranks[i] < ranks[j] else (0.5 if ranks[i] == ranks[j] else 0.0)
                e_i = _expected_score(r_i, r_j)
                # 着差（i基準）
                td_i = time_diffs[i] if i < len(time_diffs) else 0
                mov = _mov_multiplier(td_i)
                delta = k * mov * (s_i - e_i)
                new_ratings[h_i] += delta
                new_ratings[h_j] -= delta
        for h, nm in zip(horses, names):
            ratings[h] = new_ratings[h]
            last_race_date[h] = race_date
            id_name[h] = nm
        n_races += 1
        if n_races % 5000 == 0:
            print(f"  処理済み: {n_races:,}レース / 馬数={len(ratings):,}")

    print(f"  完了: {n_races:,}レース、{len(ratings):,}頭")

    out = pd.DataFrame([
        {"horse_id": h, "horse_name": id_name.get(h), "elo": round(r, 1),
         "last_race_date": last_race_date.get(h)}
        for h, r in ratings.items()
    ]).sort_values("elo", ascending=False).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


_ELO_CACHE: pd.DataFrame | None = None


def _load_elo() -> pd.DataFrame:
    global _ELO_CACHE
    if _ELO_CACHE is not None:
        return _ELO_CACHE
    if _ELO_PATH.exists():
        _ELO_CACHE = pd.read_parquet(_ELO_PATH)
    else:
        _ELO_CACHE = pd.DataFrame(columns=["horse_name", "elo"])
    return _ELO_CACHE


def get_elo(horse_name: str, default: float = INIT_RATING) -> float:
    """馬名から現在の Elo を取得。
    同名の別世代馬が複数いる場合は、直近に出走した馬(=現役)のEloを返す。"""
    if not horse_name:
        return default
    df = _load_elo()
    row = df[df["horse_name"] == str(horse_name).strip()]
    if row.empty:
        return default
    if len(row) > 1 and "last_race_date" in row.columns:
        row = row.sort_values("last_race_date").tail(1)
    return float(row["elo"].iloc[0])


def get_elo_by_id(horse_id: str, default: float = INIT_RATING) -> float:
    """horse_id から現在の Elo を取得（同名衝突なしの厳密版）。"""
    if not horse_id:
        return default
    df = _load_elo()
    if "horse_id" not in df.columns:
        return default
    row = df[df["horse_id"].astype(str) == str(horse_id).strip()]
    if row.empty:
        return default
    return float(row["elo"].iloc[0])


def get_elo_label(elo: float) -> str:
    """Elo 値から定性ラベル"""
    if elo >= 1750:
        return "トップクラス"
    if elo >= 1650:
        return "重賞級"
    if elo >= 1550:
        return "○ オープン級"
    if elo >= 1450:
        return "△ 平場標準"
    return "▼ 平均以下"


def get_elo_for_field(horse_names: list[str]) -> dict[str, dict]:
    """
    出走馬全頭の Elo を返す。出走馬間の相対比較に使う。
    Returns:
        {horse_name: {"elo": float, "label": str, "elo_rank_in_race": int}}
    """
    elos = {h: get_elo(h) for h in horse_names}
    sorted_h = sorted(elos.items(), key=lambda x: -x[1])
    rank_map = {h: i + 1 for i, (h, _) in enumerate(sorted_h)}
    return {
        h: {
            "elo": elos[h],
            "label": get_elo_label(elos[h]),
            "elo_rank_in_race": rank_map[h],
        }
        for h in horse_names
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Building Horse Elo Ratings")
    print("=" * 60)
    t = build_elo_table()
    print(f"\nSaved to {_ELO_PATH}")
    print(f"\n=== Top 20 ===")
    print(t.head(20).to_string(index=False))
