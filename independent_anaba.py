# -*- coding: utf-8 -*-
"""独立実力モデル(popularity抜き)で穴の妙味を判定 — 第88波・役割分担。
バックテスト実証: 穴(7番人気↓)を独立モデルのレース内順位で分けると
  上位20%=3着内14.7%(lift1.75) / 下位40%=5.8%(lift0.69)で綺麗に単調。
本命側(1-5人気)では効かない(lift1.12)ので、本命は人気込みの旧モデルに任せる。
→ この関数は穴(7番人気↓)にのみ適用する。
"""
import json
import pandas as pd
import lightgbm as lgb

_M = _F = _HF = None
_CATS = ["surface", "venue", "track_condition", "distance_cat", "sex"]


def _load():
    global _M, _F, _HF
    if _M is None:
        _M = lgb.Booster(model_file="data/lgbm_independent.txt")
        _F = json.load(open("data/lgbm_independent_feature_cols.json", encoding="utf-8"))
        hf = pd.read_parquet("data/horse_latest_features.parquet")
        if "horse_name" in hf.columns:
            hf = hf.drop_duplicates("horse_name", keep="last").set_index("horse_name")
        _HF = hf
    return _M, _F, _HF


def score_race(horse_names, surface=None, track_condition=None, venue=None, distance=None):
    """出走馬を独立モデルでスコアし、レース内percentile(0-1, 1=最強)を返す。
    Returns: {horse_name: {"score": float, "pct": float}}  過去成績無しの馬は除外。"""
    M, F, HF = _load()
    rows, valid = [], []
    for h in horse_names:
        hn = str(h).strip()
        if hn in HF.index:
            r = HF.loc[hn]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[-1]
            # 過去3走未満(rank_avg3 NaN)は信頼できないので除外
            if pd.isna(r.get("rank_avg3")):
                continue
            rows.append(r)
            valid.append(hn)
    if not rows:
        return {}
    X = pd.DataFrame(rows)
    X = X.reindex(columns=F)
    # ★欠損値バグ修正: horse_weight=0/jockey rate=0 は「データ無し」であって実値0ではない。
    # 0のままだとモデルが「超小型馬/最低騎手」と誤解釈し巨大ペナルティ(ビザンチン例)。NaN化してLGBMの欠損処理に委ねる。
    for col in ("horse_weight",):
        if col in X.columns:
            X[col] = X[col].replace(0, float("nan"))
    for col in ("jockey_win_rate", "jockey_place_rate", "jockey_rides",
                "trainer_win_rate", "trainer_place_rate", "sire_win_rate", "damsire_win_rate"):
        if col in X.columns:
            X[col] = X[col].replace(0, float("nan"))
    # 今日のレース条件で上書き(任意)
    if surface is not None and "surface" in X:
        X["surface"] = surface
    if track_condition is not None and "track_condition" in X:
        X["track_condition"] = track_condition
    if venue is not None and "venue" in X:
        X["venue"] = venue
    if distance is not None and "distance" in X:
        X["distance"] = distance
    for c in _CATS:
        if c in X.columns:
            X[c] = X[c].astype("category")
    pred = M.predict(X)
    pct = pd.Series(pred).rank(pct=True).values
    return {valid[i]: {"score": float(pred[i]), "pct": float(pct[i])} for i in range(len(valid))}


def anaba_verdict_tag(pct):
    """穴のpercentileから妙味タグ。上位20%=買える穴/下位40%=消し。"""
    if pct is None:
        return ""
    if pct >= 0.8:
        return f"独立◎買える穴(上位{(1-pct)*100:.0f}%内・lift1.75)"
    if pct >= 0.6:
        return f"独立○穴(上位{(1-pct)*100:.0f}%内・lift1.55)"
    if pct < 0.4:
        return f"独立✕消し穴(下位40%・lift0.69)"
    return ""
