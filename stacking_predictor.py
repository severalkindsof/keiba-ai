"""
A-2 完成版: 推論時の 3-way stacking 予測

train_stacking.py で学習した XGBoost / CatBoost / メタモデル重みを
推論時に使用して LightGBM 単独より高精度の予測を行う。

使い方:
    from stacking_predictor import predict_stacking
    blended_prob = predict_stacking(horse_dict)  # 0〜1
    # 失敗時は None → ensemble 側で LightGBM 単独にフォールバック
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

_XGB_MODEL = None
_XGB_CAT_MAPS = None
_CB_MODEL = None
_STACK_WEIGHTS = None
_LOADED = False
_FEATURE_COLS = None

CAT_COLS = ["surface", "track_condition", "venue", "distance_cat", "sex"]


def _load_once():
    global _XGB_MODEL, _XGB_CAT_MAPS, _CB_MODEL, _STACK_WEIGHTS, _LOADED, _FEATURE_COLS
    if _LOADED:
        return
    _LOADED = True
    try:
        with open(_DATA_DIR / "lgbm_feature_cols.json", encoding="utf-8") as f:
            _FEATURE_COLS = json.load(f)
    except Exception as e:
        print(f"[stacking] feature_cols ロード失敗: {e}")
        return

    # XGBoost
    try:
        import xgboost as xgb
        _XGB_MODEL = xgb.XGBRanker()
        _XGB_MODEL.load_model(str(_DATA_DIR / "xgb_ranker.json"))
        with open(_DATA_DIR / "xgb_cat_maps.json", encoding="utf-8") as f:
            _XGB_CAT_MAPS = json.load(f)
        print("[stacking] XGBoost ロード完了")
    except Exception as e:
        print(f"[stacking] XGBoost ロード失敗: {e}")
        _XGB_MODEL = None

    # CatBoost
    try:
        from catboost import CatBoostRanker
        _CB_MODEL = CatBoostRanker()
        _CB_MODEL.load_model(str(_DATA_DIR / "catboost_ranker.cbm"))
        print("[stacking] CatBoost ロード完了")
    except Exception as e:
        print(f"[stacking] CatBoost ロード失敗: {e}")
        _CB_MODEL = None

    # Stacking weights
    try:
        with open(_DATA_DIR / "stacking_weights.json", encoding="utf-8") as f:
            _STACK_WEIGHTS = json.load(f)
        print(f"[stacking] 重み: {_STACK_WEIGHTS}")
    except Exception as e:
        print(f"[stacking] weights ロード失敗: {e}")
        _STACK_WEIGHTS = None


def is_available() -> bool:
    """stacking 推論可能か"""
    _load_once()
    return all([_XGB_MODEL, _CB_MODEL, _STACK_WEIGHTS, _FEATURE_COLS])


def _build_row(horse: dict) -> pd.DataFrame:
    """LightGBM と同じ特徴量を組み立て"""
    import ev_calculator as _ev
    _ev._load_lgbm_once()
    base = {}
    if _ev._HORSE_FEATURES is not None:
        name = str(horse.get("horse_name", "")).strip()
        if name in _ev._HORSE_FEATURES.index:
            base = _ev._HORSE_FEATURES.loc[name].to_dict()
    row = {}
    for c in _FEATURE_COLS:
        row[c] = horse.get(c, base.get(c, 0))
    return pd.DataFrame([row])


def predict_xgb(horse: dict) -> float | None:
    """XGBoost 予測"""
    if _XGB_MODEL is None:
        return None
    try:
        X = _build_row(horse)
        for c, cm in (_XGB_CAT_MAPS or {}).items():
            if c in X.columns:
                X[c] = X[c].astype(str).map(cm).fillna(-1).astype(int)
        X = X[_FEATURE_COLS].astype(float).fillna(0)
        return float(_XGB_MODEL.predict(X)[0])
    except Exception as e:
        print(f"[stacking] XGB 推論エラー: {e}")
        return None


def predict_catboost(horse: dict) -> float | None:
    """CatBoost 予測"""
    if _CB_MODEL is None:
        return None
    try:
        X = _build_row(horse)
        for c in CAT_COLS:
            if c in X.columns:
                X[c] = X[c].astype(str)
        return float(_CB_MODEL.predict(X[_FEATURE_COLS])[0])
    except Exception as e:
        print(f"[stacking] CatBoost 推論エラー: {e}")
        return None


def stack_blend_probs(model_p_lgbm: np.ndarray, p_xgb: np.ndarray, p_cb: np.ndarray) -> np.ndarray:
    """
    レース内の3モデル確率配列を stacking 重みで合成。
    各配列はレース内 softmax 正規化済みとする。
    """
    _load_once()
    if _STACK_WEIGHTS is None:
        return model_p_lgbm
    eps = 1e-6
    w = _STACK_WEIGHTS
    log_combined = (
        w["w_lgbm"] * np.log(np.clip(model_p_lgbm, eps, 1))
        + w["w_xgb"] * np.log(np.clip(p_xgb, eps, 1))
        + w["w_cb"]  * np.log(np.clip(p_cb,  eps, 1))
    )
    log_combined -= log_combined.max()
    exp_s = np.exp(log_combined)
    return exp_s / exp_s.sum()
