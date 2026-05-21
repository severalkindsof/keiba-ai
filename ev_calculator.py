"""
期待値（EV）計算エンジン。
EV = 勝率 × (オッズ - 1) - (1 - 勝率)

LightGBMモデルが存在する場合、勝率予測をAIに委譲する。
モデルがない場合は従来のルックアップテーブル方式にフォールバック。
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import get_win_rate_table, get_sire_stats, get_jockey_stats, categorize_distance
from knowledge_base import apply_kb_to_horse

# ============================================================
# LightGBM モデルのシングルトンロード（起動時に1回だけ）
# ============================================================
_DATA_DIR       = Path(__file__).parent / "data"
_LGBM_MODEL     = None   # lgb.Booster
_FEATURE_COLS   = None   # list[str]
_CAT_MAPS       = None   # dict[str, list]
_HORSE_FEATURES = None   # pd.DataFrame indexed by horse_name
_LGBM_READY     = False  # ロード済みフラグ

def _load_lgbm_once():
    """モデル・特徴量リスト・最新馬成績を一度だけ読み込む。"""
    global _LGBM_MODEL, _FEATURE_COLS, _CAT_MAPS, _HORSE_FEATURES, _LGBM_READY
    if _LGBM_READY:
        return
    _LGBM_READY = True  # 失敗しても再試行しない

    model_path  = _DATA_DIR / "lgbm_win_model.txt"
    cols_path   = _DATA_DIR / "lgbm_feature_cols.json"
    cat_path    = _DATA_DIR / "lgbm_cat_mappings.json"
    feats_path  = _DATA_DIR / "horse_latest_features.parquet"

    if not (model_path.exists() and cols_path.exists()):
        print("[LGBM] モデルファイルが見つかりません。ルックアップ方式で動作します。")
        return

    try:
        import lightgbm as lgb
        _LGBM_MODEL = lgb.Booster(model_file=str(model_path))
        with open(cols_path, encoding="utf-8") as f:
            _FEATURE_COLS = json.load(f)
        if cat_path.exists():
            with open(cat_path, encoding="utf-8") as f:
                _CAT_MAPS = json.load(f)
        if feats_path.exists():
            _HORSE_FEATURES = (pd.read_parquet(feats_path)
                               .set_index("horse_name"))
        print(f"[LGBM] モデルロード完了 ({len(_FEATURE_COLS)}特徴量, {len(_HORSE_FEATURES) if _HORSE_FEATURES is not None else 0}頭)")
    except Exception as e:
        print(f"[LGBM] ロードエラー: {e}  → ルックアップ方式で動作します。")
        _LGBM_MODEL = None


def predict_win_rate_lgbm(horse: dict) -> float | None:
    """
    LightGBMで勝率を予測して返す。
    モデルが未ロード・エラー時は None を返す（呼び出し元がフォールバック）。
    """
    _load_lgbm_once()
    if _LGBM_MODEL is None or _FEATURE_COLS is None:
        return None

    horse_name = str(horse.get("horse_name", "")).strip()

    # 馬の直近成績をベースとして取得
    base = {}
    if _HORSE_FEATURES is not None and horse_name in _HORSE_FEATURES.index:
        base = _HORSE_FEATURES.loc[horse_name].to_dict()

    dist = int(horse.get("distance") or base.get("distance", 2000))

    # 特徴量の組み立て（今回のレース情報 > 直近成績のデフォルト）
    row = {
        "surface":                    horse.get("surface",         base.get("surface", "芝")),
        "track_condition":            horse.get("track_condition", base.get("track_condition", "良")),
        "venue":                      horse.get("venue",           base.get("venue", "")),
        "distance":                   dist,
        "distance_cat":               categorize_distance(dist),
        "field_size":                 int(horse.get("field_size")     or base.get("field_size",    16)),
        "weight_carried":             float(horse.get("weight_carried") or base.get("weight_carried", 55.0)),
        "horse_no":                   float(horse.get("horse_no")      or base.get("horse_no",      8.0)),
        "age":                        float(horse.get("age")           or base.get("age",           4.0)),
        "sex":                        horse.get("sex",             base.get("sex", "牡")),
        "horse_weight":               float(horse.get("horse_weight")  or base.get("horse_weight",  480.0)),
        # weight_change はTFJVの列36の値（訓練データと同じ形式で一貫性を保つ）
        # ※ netkeibaの体重変化と混在させない
        "weight_change":              float(base.get("weight_change", 0.0) or 0.0),
        "popularity":                 int(horse.get("popularity")   or 9),
        # 直近成績（TFJV履歴から）
        "rank_avg3":                  float(base.get("rank_avg3",       6.0) or 6.0),
        "rank_avg5":                  float(base.get("rank_avg5",       6.0) or 6.0),
        "rank_best5":                 float(base.get("rank_best5",      5.0) or 5.0),
        "speed_fig_avg3":             float(base.get("speed_fig_avg3",  0.0) or 0.0),
        "last3f_avg3":                float(base.get("last3f_avg3",    35.5) or 35.5),
        "wins_last5":                 float(base.get("wins_last5",      0.0) or 0.0),
        "places_last5":               float(base.get("places_last5",    0.0) or 0.0),
        "days_since_prev":            float(horse.get("rotation_days")  or base.get("days_since_prev", 28.0) or 28.0),
        "weight_trend3":              float(base.get("weight_trend3",   0.0) or 0.0),
        # 騎手・調教師（直近成績から）
        "jockey_win_rate":            float(base.get("jockey_win_rate",           0.08) or 0.08),
        "jockey_place_rate":          float(base.get("jockey_place_rate",         0.25) or 0.25),
        "jockey_rides":               float(base.get("jockey_rides",            100.0) or 100.0),
        "jockey_longshot_win_rate":   float(base.get("jockey_longshot_win_rate",  0.02) or 0.02),
        "jockey_longshot_place_rate": float(base.get("jockey_longshot_place_rate",0.10) or 0.10),
        "trainer_win_rate":           float(base.get("trainer_win_rate",          0.08) or 0.08),
        "trainer_place_rate":         float(base.get("trainer_place_rate",        0.25) or 0.25),
        "sire_win_rate":              float(base.get("sire_win_rate") or np.nan),
    }

    X = pd.DataFrame([row])

    # カテゴリ変数を訓練時と同じ定義に揃える
    cat_cols = ["surface", "track_condition", "venue", "distance_cat", "sex"]
    for col in cat_cols:
        if col not in X.columns:
            continue
        if _CAT_MAPS and col in _CAT_MAPS:
            X[col] = pd.Categorical(X[col], categories=_CAT_MAPS[col])
        else:
            X[col] = X[col].astype("category")

    # モデルが要求する列のみ・順番通りに並べる
    X = X.reindex(columns=_FEATURE_COLS)

    try:
        prob = float(_LGBM_MODEL.predict(X)[0])
        return max(0.001, min(0.999, prob))
    except Exception as e:
        print(f"[LGBM] 予測エラー ({horse_name}): {e}")
        return None


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
    # win_rate_table が空または必要列がない場合は即フォールバック
    required_cols = {"surface", "distance_cat", "pop_bucket", "win_rate", "place_rate", "races"}
    if win_rate_table.empty or not required_cols.issubset(win_rate_table.columns):
        return {"win_rate": np.nan, "place_rate": np.nan, "sample_size": 0}

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

    # ---- 勝率の取得（LightGBM優先、なければルックアップテーブル）----
    lgbm_win_rate = predict_win_rate_lgbm(horse)
    lgbm_used = lgbm_win_rate is not None

    if lgbm_used:
        # LightGBM予測値をそのまま使用
        win_rate   = lgbm_win_rate
        place_rate = min(0.999, win_rate * 3.0)  # 複勝率の近似（単勝率の約3倍）
        sample_size = -1  # LightGBM使用を示す特別値
    else:
        # フォールバック: 条件別勝率テーブル
        stats = lookup_win_rate(win_rate_table, surface, distance, popularity)
        win_rate   = stats["win_rate"]
        place_rate = stats["place_rate"]
        sample_size = stats["sample_size"]
        if np.isnan(win_rate):
            win_rate = 1.0 / popularity if popularity > 0 else 0.05

    # 各種補正（LightGBM使用時は小さめに適用してダブルカウントを防ぐ）
    scale = 0.3 if lgbm_used else 1.0   # LightGBM既に血統・騎手を学習済みのため縮小
    sb = sire_bonus(sire_stats, sire, distance) * scale
    jb = jockey_bonus(jockey_stats, jockey, popularity) * scale
    tb = track_condition_bonus(condition, popularity) * scale

    # ナレッジベースボーナス（レース格言など・LightGBMに含まれない定性情報）
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
        "sample_size": sample_size,
        "lgbm_win_rate": round(lgbm_win_rate * 100, 1) if lgbm_used else None,
        "lgbm_used": lgbm_used,
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
        # 表示用フィールド
        "jockey": horse.get("jockey", ""),
        "gate":   int(horse.get("gate") or 0),
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
