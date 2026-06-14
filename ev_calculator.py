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
from data_loader import categorize_distance  # CLEAN: 未使用 import 削除
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
_MODEL_TYPE     = "classifier"  # "classifier" or "ranker"
_CALIBRATOR     = None   # IsotonicRegression（LambdaRankスコア→確率変換）

def get_model_type() -> str:
    """現在のモデルタイプを返す（_load_lgbm_once()後に正しい値になる）"""
    return _MODEL_TYPE

def _load_lgbm_once():
    """モデル・特徴量リスト・最新馬成績を一度だけ読み込む。"""
    global _LGBM_MODEL, _FEATURE_COLS, _CAT_MAPS, _HORSE_FEATURES, _LGBM_READY, _MODEL_TYPE, _CALIBRATOR
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
        # モデルタイプ（ranker / classifier）を読み込む
        meta_path = _DATA_DIR / "lgbm_model_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                _MODEL_TYPE = json.load(f).get("model_type", "classifier")
        # Isotonic Regressionキャリブレーター（ROOT-1）
        cal_path = _DATA_DIR / "lgbm_calibrator.pkl"
        if cal_path.exists():
            try:
                import joblib
                _CALIBRATOR = joblib.load(str(cal_path))
                print(f"[LGBM] キャリブレーターロード完了")
            except Exception as _ce:
                print(f"[LGBM] キャリブレーターロード失敗: {_ce}")
        print(f"[LGBM] モデルロード完了 ({len(_FEATURE_COLS)}特徴量, {len(_HORSE_FEATURES) if _HORSE_FEATURES is not None else 0}頭, type={_MODEL_TYPE}, calibrator={'あり' if _CALIBRATOR else 'なし'})")
    except Exception as e:
        print(f"[LGBM] ロードエラー: {e}  → ルックアップ方式で動作します。")
        _LGBM_MODEL = None


# ---- Phase B/H: 動的特徴量ヘルパー ----

_TURN_DIR_MAP = {
    "東京":"左", "新潟":"左", "中京":"左", "函館":"左", "札幌":"左",
    "中山":"右", "阪神":"右", "京都":"右", "小倉":"右", "福島":"右",
}

def _cls_level(race_name) -> int:
    """race_name からクラスレベルを数値化"""
    if not race_name: return 3
    s = str(race_name)
    if "G1" in s:                                  return 8
    if "G2" in s:                                  return 7
    if "G3" in s or "重賞" in s:                  return 6
    if "オープン" in s or "OPEN" in s or "3勝" in s: return 5
    if "2勝" in s:                                 return 4
    if "1勝" in s:                                 return 3
    if "未勝利" in s:                              return 2
    if "新馬" in s:                                return 1
    return 3

def _calc_class_and_weight_features(horse: dict, base: dict) -> dict:
    """
    Phase B/H の動的特徴量を計算して dict で返す。
    base は horse_latest_features の行 dict。
    """
    # --- クラスレベル ---
    prev_cl = float(base.get("class_level") or 3.0)   # 前走クラス（raw保存済み）
    curr_cl = float(_cls_level(horse.get("race_name", "")))

    # --- 斤量変化: 今走 - 前走 ---
    prev_wc = float(base.get("weight_carried") or 55.0)
    curr_wc = float(horse.get("weight_carried") or prev_wc)
    wc_change = curr_wc - prev_wc

    # --- 回り方向変更フラグ ---
    curr_dir = _TURN_DIR_MAP.get(str(horse.get("venue", "")).strip(), "")
    prev_dir = _TURN_DIR_MAP.get(str(base.get("venue",  "")).strip(), "")
    tdir_changed = float(1 if (curr_dir and prev_dir and curr_dir != prev_dir) else 0)

    return {
        "prev_class_level":     prev_cl,
        "class_change":         curr_cl - prev_cl,
        "weight_carried_change": wc_change,
        "turn_dir_changed":     tdir_changed,
        "meet_race_seq":        np.nan,   # 現走情報のため runtime 取得困難 → NaN
    }


def predict_win_rate_lgbm(horse: dict, raw_score: bool = False) -> float | None:
    """
    LightGBMで勝率を予測して返す。
    モデルが未ロード・エラー時は None を返す（呼び出し元がフォールバック）。

    raw_score=True: キャリブレーター適用前の生スコアを返す（第27波）。
        fit_calibration.py が校正器を再学習する時に使う。校正済み出力に
        再フィットすると「raw入力×校正済み定義域」の不一致で全確率が
        端値クリップされる循環校正バグになるため。
    """
    _load_lgbm_once()
    if _LGBM_MODEL is None or _FEATURE_COLS is None:
        return None

    horse_name = str(horse.get("horse_name", "")).strip()

    # 馬の直近成績をベースとして取得
    base = {}
    if _HORSE_FEATURES is not None and horse_name in _HORSE_FEATURES.index:
        base = _HORSE_FEATURES.loc[horse_name].to_dict()

    # BUG-D: 過去成績が無い馬は信頼できる予測ができないので予測しない
    # （horse_latest_features に存在しない = 過去レコード無し）
    if not base:
        return None
    # rank_avg3 が NaN なら過去3戦未満 → 予測精度低い
    _ravg = base.get("rank_avg3")
    if _ravg is None or (isinstance(_ravg, float) and pd.isna(_ravg)):
        return None

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
        # 第29波: 東西・前走着差（horse_latest の base から供給）
        "is_west":                    int(base.get("is_west", 0) or 0),
        "prev_margin":                float(base.get("prev_margin", 0.0) or 0.0),
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
        # ---- 言い訳分析特徴量（前走の展開・ペース）----
        # horse_latest_features には _raw サフィックスで保存されるため変換して渡す
        # _raw = 最新走の実値 → 次走予測の「前走特徴量」として使用
        "closing_move":        float(base["closing_move_raw"])        if base.get("closing_move_raw")        is not None else np.nan,
        "corner_pos_var":      float(base["corner_pos_var_raw"])      if base.get("corner_pos_var_raw")      is not None else np.nan,
        "pci":                 float(base["pci_raw"])                 if base.get("pci_raw")                 is not None else np.nan,
        "last3f_rank_in_race": float(base["last3f_rank_in_race_raw"]) if base.get("last3f_rank_in_race_raw") is not None else np.nan,
        # Phase H 追加特徴量
        "prev_opponent_sf_avg":  float(base["opponent_sf_avg_raw"])   if base.get("opponent_sf_avg_raw")   is not None else np.nan,
        "damsire_win_rate":      float(base.get("damsire_win_rate") or np.nan),
        "vd_win_rate":           float(base.get("vd_win_rate") or np.nan),
        # ---- Phase B: クラスレベル（動的計算）----
        # base["class_level"] = 前走のクラスレベル（EXCUSE_RAW_COLSで保存）
        # prev_class_level として使用し、今走との差分を class_change に
        **_calc_class_and_weight_features(horse, base),
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
        score = float(_LGBM_MODEL.predict(X)[0])
        if _MODEL_TYPE == "ranker":
            if raw_score:
                return score  # 校正前の生スコア（fit_calibration 用）
            # ROOT-1: Isotonic Regressionでスコア→確率に変換
            if _CALIBRATOR is not None:
                cal_prob = float(_CALIBRATOR.predict([score])[0])
                return float(np.clip(cal_prob, 0.001, 0.999))
            # キャリブレーターなし時はそのまま返す（app.pyでsoftmax正規化）
            return score
        else:
            # Classifier は確率を返す
            return max(0.001, min(0.999, score))
    except Exception as e:
        print(f"[LGBM] 予測エラー ({horse_name}): {e}")
        return None


# ============================================================
# Bill Benter Odds Blending (SUPER-1)
# 自モデル予測と市場（公衆）予測を統合する2段階モデル
# ============================================================

def blend_with_market(
    model_probs,
    market_probs,
    alpha: float = 0.7,
    beta: float = 0.3,
    eps: float = 1e-6,
):
    """
    Benter (1994) の式: c_i = softmax(α·log(f_i) + β·log(π_i))

    Args:
        model_probs:  自モデル予測勝率（配列、レース内合計≈1）
        market_probs: 市場インプライド勝率（配列、レース内合計≈1）
        alpha:        自モデル重み（典型値: 0.5〜0.8）
        beta:         市場重み（典型値: 0.2〜0.5）
        eps:          log(0) 回避用

    Returns:
        ブレンド後勝率配列（レース内合計=1）
    """
    f = np.asarray(model_probs, dtype=float)
    pi = np.asarray(market_probs, dtype=float)
    # 0 を eps に置換
    f = np.where(f < eps, eps, f)
    pi = np.where(pi < eps, eps, pi)
    # log-linear blending
    log_score = alpha * np.log(f) + beta * np.log(pi)
    # softmax (race-level normalize)
    log_score = log_score - log_score.max()
    exp_s = np.exp(log_score)
    return exp_s / exp_s.sum()


def get_benter_weights(default_alpha: float = 0.7, default_beta: float = 0.3) -> tuple[float, float]:
    """
    保存済みの Benter フィット重みをロード。
    なければデフォルト値（α=0.7, β=0.3）を返す。

    注意: フィット結果が β<0（市場と逆相関）になった場合は
    リーケージ気味のサインなので β=0 にクリップしてブレンドを安定化させる。
    """
    path = _DATA_DIR / "benter_weights.json"
    if not path.exists():
        return default_alpha, default_beta
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        alpha = float(d.get("alpha", default_alpha))
        beta  = float(d.get("beta",  default_beta))
        # 第34波修正: β<0 は「モデルが popularity 特徴量で市場情報を内包済み」の
        # 自然な帰結。旧実装は β だけ 0 にクリップし、ペアで打ち消し合うはずの
        # α=1.385 が単独暴走 → 1人気 32.2% を 43.1%（+10.9pt の幻勝率）に歪めていた。
        # β<0 時は α/β をペアで恒等化（=校正済みLGBM確率をそのまま使う。
        # 校正精度は実測で全人気帯 ±0.5pt 以内と EXCELLENT）
        if beta < 0:
            return 1.0, 0.0
        return alpha, beta
    except Exception:
        return default_alpha, default_beta


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

_MARKET_WR_CACHE: dict | None = None

def _market_winrate_by_pop(popularity: int) -> float:
    """実測の人気別勝率（market_prob_by_popularity.parquet）。無ければ控えめな近似。"""
    global _MARKET_WR_CACHE
    if _MARKET_WR_CACHE is None:
        try:
            _mp = pd.read_parquet(_DATA_DIR / "market_prob_by_popularity.parquet")
            _MARKET_WR_CACHE = dict(zip(_mp["popularity"].astype(int), _mp["win_rate"].astype(float)))
        except Exception:
            _MARKET_WR_CACHE = {}
    if popularity in _MARKET_WR_CACHE:
        return float(_MARKET_WR_CACHE[popularity])
    # テーブル外（18人気超等）: 実測下限相当の控えめな値
    return 0.002 if popularity >= 10 else max(0.02, 0.33 * (0.6 ** (popularity - 1)))


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
    # 第27波修正: 旧ラベル（1〜3番人気 等）はテーブル実値（1番人気/2-3番人気/
    # 4-5番人気/6-9番人気）と完全不一致で、10人気以下を除く全人気帯で
    # ルックアップが失敗 → フォールバック 1/popularity の幻勝率
    # （1番人気=100%扱い、9番人気=実際の5倍）で EV が出ていた。
    # convert_tfjv.py の pd.cut ラベルと同一の文字列に修正。
    if popularity <= 1:
        return "1番人気"
    elif popularity <= 3:
        return "2-3番人気"
    elif popularity <= 5:
        return "4-5番人気"
    elif popularity <= 9:
        return "6-9番人気"
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
    # 第28波修正: docstring の「±0.02 程度」が未クリップで、人気種牡馬は +0.07 など
    # 大幅加算 → 良血は人気に織込済みなので二重計上の楽観バイアスだった
    return float(np.clip(round(sire_wr - overall_avg, 4), -0.02, 0.02))


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
    # 第28波修正: 未クリップで最大 +0.102（12人気の複勝率実測7%を17%扱い）の
    # 楽観バイアスだった → ±0.04 にクリップ
    return float(np.clip(round(jockey_place - avg_place, 4), -0.04, 0.04))


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
    # 第28波修正: 旧 ×0.5 は不良馬場で勝率に +7.5pt 加算（実測1.6%の穴を9%扱い、
    # 5.6倍の楽観）→ ×0.05 に縮小（+0.75pt = 実測比1.5倍相当で現実的）
    return round((multiplier - 1.0) * 0.05, 4)


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
    # 第28波修正: 旧近似 odds/3 は単勝3倍以下で複勝1.0倍以下（EV常にNaN）、
    # 大穴では過大という二重の歪み → 第17波と同じ人気帯別逓減係数に統一
    _po_coef = float(np.clip(0.30 - 0.0075 * max(0.0, odds - 10.0), 0.15, 0.30))
    place_odds = float(horse.get("place_odds") or (1.0 + max(0.0, odds - 1.0) * _po_coef))
    jockey = horse.get("jockey", "")
    sire = horse.get("sire", "")
    condition = horse.get("track_condition", "良")

    # ---- 勝率の取得（LightGBM優先、なければルックアップテーブル）----
    lgbm_win_rate = predict_win_rate_lgbm(horse)
    lgbm_used = lgbm_win_rate is not None

    if lgbm_used:
        # LightGBM予測値をそのまま使用
        win_rate   = lgbm_win_rate
        place_rate = min(0.999, 1.0 - (1.0 - win_rate) ** 3)  # LATENT-5: 理論的複勝率推定
        sample_size = -1  # LightGBM使用を示す特別値
    else:
        # フォールバック: 条件別勝率テーブル
        stats = lookup_win_rate(win_rate_table, surface, distance, popularity)
        win_rate   = stats["win_rate"]
        place_rate = stats["place_rate"]
        sample_size = stats["sample_size"]
        if np.isnan(win_rate):
            # 第27波修正: 旧フォールバック 1/popularity は 1番人気=100%・
            # 9番人気=11%（実測の5倍）の幻勝率で EV を過大計上していた
            # → 実測の人気別勝率（market_prob_by_popularity）を使用
            win_rate = _market_winrate_by_pop(popularity)

    # BUG-B: LightGBM 使用時はボーナス完全ゼロ化（sire/jockey/track は既に学習済み）
    # ルックアップフォールバック時のみ補正を加算する
    scale = 0.0 if lgbm_used else 1.0
    sb = sire_bonus(sire_stats, sire, distance) * scale
    jb = jockey_bonus(jockey_stats, jockey, popularity) * scale
    tb = track_condition_bonus(condition, popularity) * scale

    # ナレッジベースボーナス（レース格言など・LightGBMに含まれない定性情報）
    kb_result = apply_kb_to_horse(horse, race_name=horse.get("race_name", ""))
    kb_b = kb_result.get("kb_bonus", 0.0)

    adjusted_win_rate = min(0.95, max(0.001, win_rate + sb + tb + kb_b))
    # 第28波修正: 一律 ×3 は実測倍率（1人気 2.0倍 / 5人気 4.2倍 / 10人気 5.4倍）と
    # 乖離し、上位人気の複勝率=複勝EVを過大評価していた → 人気帯別倍率に
    if place_rate is None or np.isnan(place_rate):
        _pr_mult = 2.0 if popularity <= 2 else (3.2 if popularity <= 5 else (4.5 if popularity <= 9 else 5.0))
        safe_place_rate = min(0.95, win_rate * _pr_mult)
    else:
        safe_place_rate = place_rate
    adjusted_place_rate = min(0.97, max(0.001, safe_place_rate + jb + tb))

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
        # 第30波→第35波: place_odds はローカル変数（実値優先・近似フォールバック）が
        # 下の行で返るため、ここの重複キーは削除（dict 後勝ちに依存する危険コードだった）
        "est_place_rate": round(adjusted_place_rate * 100, 1),
        "odds_distortion": round(odds_distortion * 100, 1),
        "ev": round(ev, 3),
        "ev_place": round(ev_place, 3),
        "place_odds": round(place_odds, 1),
        "ev_place_label": ev_label(ev_place),
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
        "ev_label":       ev_label(ev),
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
        "jockey":   horse.get("jockey", ""),
        "gate":     int(horse.get("gate") or 0),
        "horse_no": int(horse.get("horse_no") or 0),
        # BUG-Y1 二重保険: apply_xxx 系の戻り値 bonus を base dict にも拾う
        "condition_apt_bonus":       float(horse.get("condition_apt_bonus", 0.0) or 0.0),
        "surface_change_bonus":      float(horse.get("surface_change_bonus", 0.0) or 0.0),
        "hurdle_to_flat_bonus":      float(horse.get("hurdle_to_flat_bonus", 0.0) or 0.0),
        "weight_ratio_bonus":        float(horse.get("weight_ratio_bonus", 0.0) or 0.0),
        "nicks_bonus":               float(horse.get("nicks_bonus", 0.0) or 0.0),
        "season_bonus":              float(horse.get("season_bonus", 0.0) or 0.0),
        "position_correction_bonus": float(horse.get("position_correction_bonus", 0.0) or 0.0),
        "realtime_bias_bonus":       float(horse.get("realtime_bias_bonus", 0.0) or 0.0),
        "race_level_bonus":          float(horse.get("race_level_bonus", 0.0) or 0.0),
        "lap_bonus":                 float(horse.get("lap_bonus", 0.0) or 0.0),
        "won_awase":                 horse.get("won_awase"),
        "partner_won_sat":           bool(horse.get("partner_won_sat", False)),
    }
    return base


def _romance_danger_score(popularity: int, win_rate: float, odds: float,
                          confidence_score: int = 0, ev: float = float("nan")) -> str:
    """
    穴馬を「ロマンだけで」買う危険度。
    B-3: EV+の大穴は「極高」にしない（モデルが根拠を持っている）
    """
    # 人気馬は「ロマン爆死」リスクが低い
    if popularity <= 3:
        return "低（本命）"
    if popularity <= 6:
        return "低（中穴）"

    # NEW-6: 実力スコアが高い穴馬はAIが根拠を持っている → 危険度を下げる
    if confidence_score >= 60:
        return "低（実力馬）"
    if confidence_score >= 50:
        return "中（実力穴馬）" if popularity >= 13 else "中（根拠あり）"

    # B-3: EV+の大穴は「極高」にしない（モデルが割安と評価している根拠がある）
    _ev = ev if not np.isnan(ev) else calc_ev(win_rate, odds)
    ev_plus = not np.isnan(_ev) and _ev > 0.0

    if popularity >= 15:
        return "高（割安穴馬）" if ev_plus else "極高（超大穴）"
    if popularity >= 12 or odds >= 100:
        return "中（割安穴馬）" if ev_plus else "高（大穴）"

    # 7〜11番人気はEVで判断
    if np.isnan(_ev) or _ev < -0.3:
        return "高"
    elif _ev < -0.1:
        return "中"
    else:
        return "中（穴馬注意）"


def ev_label(ev: float) -> str:
    """EV値を市場評価ラベルに変換（NEW-3: 過小評価/過大評価に統一）"""
    if np.isnan(ev):
        return "？ データ不足"
    if ev >= 0.30:   return "◎ 過小評価"
    if ev >= 0.05:   return "○ やや過小評価"
    if ev >= -0.10:  return "△ 適正"
    if ev >= -0.25:  return "▲ やや過大評価"
    return "✕ 過大評価"


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
    # 第88波: 本番は人気込みモデル(本命用)に戻したので生EVで十分tame。穴の妙味は
    # independent_anaba(独立モデルタグ)で別判定する役割分担(バックテスト穴lift1.75x実証)。
    results = [evaluate_horse(h, win_rate_table, sire_stats, jockey_stats) for h in horses]
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("ev", ascending=False).reset_index(drop=True)
    return df
