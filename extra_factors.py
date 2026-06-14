"""追加ファクター群（対話分析用レンズ）

過去走から複数の狙い目シグナルを抽出。スコアには畳み込まず、
race_brief の追加レンズ ⟨⟩ として併記して人間+AIで統合する。

ファクター:
1. class_move    : 格上挑戦帰り / 昇級初戦 / 格下げ
2. weight_drop   : 斤量大幅減（前走比）
3. course_record : 同コース(venue×surface×distance)連対率
4. layoff        : 休み明け週数（+ 叩き良化余地）
5. body_trend    : 馬体重の増減傾向
6. last3f_top    : 上がり最速級の経験（決め手）
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

_TFJV = Path(__file__).parent / "data" / "tfjv_all.parquet"
_SIRE_TRACK = Path(__file__).parent / "data" / "sire_track_stats.parquet"
_DF = None
_SIRE_MUD = None


def sire_mud_aptitude(sire: str, track_condition: str) -> dict:
    """父系の道悪適性。track_condition が 重/不(良) のときのみ意味を持つ。"""
    global _SIRE_MUD
    if not sire or track_condition not in ("重", "不", "不良", "稍", "稍重"):
        return {"tag": "", "detail": "", "wet_diff": None}
    if _SIRE_MUD is None:
        try:
            t = pd.read_parquet(_SIRE_TRACK)
            _SIRE_MUD = dict(zip(t["sire"].astype(str).str.strip(), t["wet_diff"]))
        except Exception:
            _SIRE_MUD = {}
    d = _SIRE_MUD.get(str(sire).strip())
    if d is None:
        return {"tag": "", "detail": "", "wet_diff": None}
    if d >= 0.05:
        return {"tag": "道悪巧者血統", "detail": f"{sire}産駒は道悪で複勝率+{d*100:.0f}pt", "wet_diff": d}
    if d <= -0.05:
        return {"tag": "道悪苦手血統", "detail": f"{sire}産駒は道悪で複勝率{d*100:.0f}pt", "wet_diff": d}
    return {"tag": "", "detail": "", "wet_diff": d}


def _load():
    global _DF
    if _DF is None:
        _DF = pd.read_parquet(_TFJV, columns=[
            "horse_name", "race_name", "venue", "surface", "distance",
            "weight_carried", "horse_weight", "rank", "last_3f", "race_id", "date"])
        _DF["horse_name"] = _DF["horse_name"].astype(str).str.strip()
        _DF["date"] = pd.to_datetime(_DF["date"], errors="coerce")
        _DF["rkey"] = _DF["race_id"].astype(str).str[:8]
    return _DF


def parse_class_level(race_name: str) -> float | None:
    """レース名からクラスレベルを推定。0=新馬未勝利 ... 7=G1。不明はNone。"""
    s = str(race_name)
    if "G1" in s or "Ｇ１" in s:
        return 7.0
    if "G2" in s or "Ｇ２" in s:
        return 6.0
    if "G3" in s or "Ｇ３" in s:
        return 5.0
    if "(L)" in s or "（L）" in s or "Ｌ" in s or "リステッド" in s:
        return 4.5
    if "オープン" in s or "オープ" in s or "ＯＰ" in s:
        return 4.0
    if "3勝" in s or "３勝" in s or "1600万" in s:
        return 3.0
    if "2勝" in s or "２勝" in s or "1000万" in s:
        return 2.0
    if "1勝" in s or "１勝" in s or "500万" in s:
        return 1.0
    if "未勝利" in s or "新馬" in s:
        return 0.0
    return None


def build_extra_factors(horse_name: str, current_class_level: float | None,
                        venue: str, surface: str, distance: int,
                        race_date=None, n_recent: int = 6) -> dict:
    """1頭の追加ファクターを集約して返す。"""
    df = _load()
    name = str(horse_name).strip()
    h = df[df["horse_name"] == name].sort_values("date", ascending=False)
    tags, details = [], []
    out = {"tags": [], "detail": "", "course_in2_rate": None,
           "layoff_days": None, "class_move": "", "weight_drop": None}
    if h.empty:
        return out

    recent = h.head(n_recent)
    prev = recent.iloc[0]  # 前走

    # --- 1. class_move（格上挑戦帰り / 昇級 / 格下げ） ---
    prev_cls = parse_class_level(prev["race_name"])
    if current_class_level is not None and prev_cls is not None:
        diff = current_class_level - prev_cls
        if prev_cls >= 5 and current_class_level <= 3:
            tags.append("格上挑戦帰り")
            details.append(f"前走G級(Lv{prev_cls:.0f})→今回条件戦(Lv{current_class_level:.0f})＝楽な相手")
            out["class_move"] = "格上挑戦帰り"
        elif diff >= 1:
            tags.append("昇級初戦")
            details.append(f"前走Lv{prev_cls:.0f}→今回Lv{current_class_level:.0f}（昇級）")
            out["class_move"] = "昇級"
        elif diff <= -1:
            tags.append("格下げ")
            details.append(f"前走Lv{prev_cls:.0f}→今回Lv{current_class_level:.0f}（相手弱化）")
            out["class_move"] = "格下げ"

    # --- 2. weight_drop（斤量大幅減・前走比） ---
    try:
        wc_now = None  # 今回斤量は出馬表側にあるが、ここでは前走斤量のみ参照可
        wc_prev = float(prev["weight_carried"])
        # 直近2走の斤量推移（減少基調か）
        wcs = pd.to_numeric(recent["weight_carried"], errors="coerce").dropna()
        if len(wcs) >= 2:
            out["weight_drop"] = round(float(wcs.iloc[0] - wcs.iloc[1]), 1)
    except Exception:
        pass

    # --- 3. course_record（同コース連対率） ---
    sc = h[(h["venue"] == venue) & (h["surface"] == surface) & (h["distance"] == distance)]
    sc_r = pd.to_numeric(sc["rank"], errors="coerce").dropna()
    if len(sc_r) >= 1:
        in2 = (sc_r <= 2).mean()
        in3 = (sc_r <= 3).mean()
        out["course_in2_rate"] = round(float(in2), 2)
        if len(sc_r) >= 2 and in3 >= 0.5:
            tags.append("同コース得意")
            details.append(f"{venue}{surface}{distance}m {len(sc_r)}走・連対率{in2*100:.0f}%・複勝{in3*100:.0f}%")

    # --- 4. layoff（休み明け週数） ---
    if race_date is not None:
        try:
            rd = pd.to_datetime(race_date)
            last = recent.iloc[0]["date"]
            if pd.notna(last):
                days = (rd - last).days
                out["layoff_days"] = int(days)
                if days >= 120:
                    tags.append(f"休み明け{days//7}週")
                    details.append(f"前走から{days}日（長期休養明け）")
                elif days >= 70:
                    details.append(f"前走から{days}日（やや間隔）")
        except Exception:
            pass

    # --- 5. body_trend（馬体重増減・直近2走） ---
    bw = pd.to_numeric(recent["horse_weight"], errors="coerce").dropna()
    if len(bw) >= 2:
        d = int(bw.iloc[0] - bw.iloc[1])
        if abs(d) >= 12:
            tags.append(f"馬体{'増' if d>0 else '減'}{abs(d)}kg")
            details.append(f"前走馬体重{int(bw.iloc[0])}（前々走比{d:+d}kg）")

    # --- 6. last3f_top（上がり最速級の決め手） ---
    # 直近走で同レース内の上がり順位を見る
    try:
        rkeys = recent["rkey"].dropna().tolist()[:4]
        df_l = df[df["rkey"].isin(rkeys)][["rkey", "horse_name", "last_3f"]].copy()
        df_l["l3f"] = pd.to_numeric(df_l["last_3f"], errors="coerce")
        df_l["l3f_rank"] = df_l.groupby("rkey")["l3f"].rank(method="min", na_option="bottom")
        mine = df_l[df_l["horse_name"] == name]
        top2 = (mine["l3f_rank"] <= 2).sum()
        if top2 >= 2:
            tags.append("決め手上位")
            details.append(f"直近で上がり2位以内{int(top2)}回（末脚信頼）")
    except Exception:
        pass

    out["tags"] = tags
    out["detail"] = " / ".join(details)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tfjv_entries import load_tfjv_entries
    data = load_tfjv_entries("C:/TFJV/TXT/出馬表分析260613.CSV")
    # 阪神7R（1勝クラス Lv1）でテスト＝格上挑戦帰り等が出やすい
    race = data["20260613090707"]
    cls = parse_class_level(race["race_class"])
    print(f"=== 阪神7R {race['race_class']} (Lv{cls}) 追加ファクター ===")
    for e in race["entries"][:10]:
        f = build_extra_factors(e["horse_name"], cls, "阪神", "ダート", 1800, "2026-06-13")
        if f["tags"]:
            print(f"  {e['horse_name']:14s}: {' / '.join(f['tags'])}")
