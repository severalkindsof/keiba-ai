"""
第13波・B: 馬券内特化サブモデル推論 wrapper

公開関数:
    predict_place_prob(features_df: pd.DataFrame) -> np.ndarray
        馬券内（3着以内）に来る予測確率を返す。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

_DATA = Path(__file__).parent / "data"
_MODEL_PATH = _DATA / "place_lgb.txt"
_FEAT_PATH  = _DATA / "place_features.json"

_model = None
_feats: list[str] | None = None

_SURFACE_CODE = {"芝": 0, "ダート": 1, "障害": 2}
_VENUE_CODE = {"札幌": 0, "函館": 1, "福島": 2, "新潟": 3, "東京": 4,
               "中山": 5, "中京": 6, "京都": 7, "阪神": 8, "小倉": 9}
_TRACK_COND_CODE = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3}


def _load():
    global _model, _feats
    if _model is not None:
        return _model, _feats
    if not (_MODEL_PATH.exists() and _FEAT_PATH.exists()):
        return None, None
    try:
        import lightgbm as lgb
        _model = lgb.Booster(model_file=str(_MODEL_PATH))
        with open(_FEAT_PATH, encoding="utf-8") as f:
            _feats = json.load(f)["features"]
        return _model, _feats
    except Exception:
        return None, None


def predict_place_prob(features_df: pd.DataFrame) -> np.ndarray:
    """
    馬券内確率を予測。モデルがロードできない場合は popularity ベースのフォールバック。
    """
    model, feats = _load()
    n = len(features_df)
    if n == 0:
        return np.array([])

    if model is None or feats is None:
        # フォールバック：人気から複勝率を推定
        pop = pd.to_numeric(features_df.get("popularity", pd.Series([8]*n)), errors="coerce").fillna(8)
        return np.clip(0.85 - 0.075 * (pop - 1), 0.03, 0.85).values

    # 特徴量を埋める
    X = pd.DataFrame()
    for f in feats:
        if f in features_df.columns:
            X[f] = features_df[f]
        else:
            X[f] = 0
    # カテゴリ→コード変換
    if "surface_code" in feats and "surface" in features_df.columns and X["surface_code"].sum() == 0:
        X["surface_code"] = features_df["surface"].map(_SURFACE_CODE).fillna(0).astype(int)
    if "venue_code" in feats and "venue" in features_df.columns and X["venue_code"].sum() == 0:
        X["venue_code"] = features_df["venue"].map(_VENUE_CODE).fillna(0).astype(int)
    if "track_cond_code" in feats and "track_condition" in features_df.columns and X["track_cond_code"].sum() == 0:
        X["track_cond_code"] = features_df["track_condition"].map(_TRACK_COND_CODE).fillna(0).astype(int)
    # 必須列の数値化（第17波: 0 埋めは訓練分布外 → 列ごとの典型値で補完）
    _defaults = {"horse_no": 8, "popularity": 8, "distance": 1800,
                 "month": 6, "age": 4, "horse_weight": 470}
    for f in feats:
        X[f] = pd.to_numeric(X[f], errors="coerce")
        dv = _defaults.get(f, 0)
        X[f] = X[f].replace(0, np.nan).fillna(dv) if f in ("age", "horse_weight") else X[f].fillna(dv)
    X = X[feats]
    try:
        return model.predict(X.values)
    except Exception:
        pop = pd.to_numeric(features_df.get("popularity", pd.Series([8]*n)), errors="coerce").fillna(8)
        return np.clip(0.85 - 0.075 * (pop - 1), 0.03, 0.85).values
