"""展開シミュレーション（脚質判定 + ペース予測）

対話分析で使った「2角平均位置」ベースの脚質判定をモジュール化。
既存 pace_analyzer は4角ベースで「全馬が逃げ先行に見える」誤誘導があったため、
序盤位置（2角）で脚質を正しく分類する。

脚質: 逃げ / 先行 / 中団 / 後方（直近5走の2角平均位置を頭数で正規化）
ペース: 逃げ・先行馬の頭数から予測
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

_TFJV = Path(__file__).parent / "data" / "tfjv_all.parquet"
_DF = None
_SIZE_MAP = None


def _load():
    global _DF, _SIZE_MAP
    if _DF is None:
        # 脚質は序盤位置(corner2)で測る。近年corner2は0%欠損なのでcorner4より正確。
        # corner2欠損時はcorner1→corner4の順でフォールバック。rkey実頭数で正規化。
        _DF = pd.read_parquet(_TFJV, columns=[
            "horse_name", "corner1", "corner2", "corner4", "race_id", "date"])
        _DF["horse_name"] = _DF["horse_name"].astype(str).str.strip()
        _DF["date"] = pd.to_datetime(_DF["date"], errors="coerce")
        _DF["rkey"] = _DF["race_id"].astype(str).str[:8]
        # 真の頭数 = rkey ごとの行数
        _SIZE_MAP = _DF.groupby("rkey").size().to_dict()
    return _DF, _SIZE_MAP


def classify_style(horse_name: str, n_recent: int = 5) -> dict:
    """1頭の脚質を直近n走の【序盤位置(corner2)】を実頭数で正規化して判定。

    序盤位置=2コーナー通過順。脚質(前に行くか後ろか)はここで決まる。
    corner2欠損時のみ corner1→corner4 にフォールバック。0/欠損は除外。
    標本2走未満は信頼度が低いので style に"(薄)"を付す。
    Returns: {style, c4_avg(序盤平均), c4_rel(正規化), n}
    """
    df, size_map = _load()
    h = df[df["horse_name"] == str(horse_name).strip()].sort_values("date", ascending=False).head(n_recent)
    if h.empty:
        return {"style": "不明", "c4_avg": None, "c4_rel": None, "n": 0}
    h = h.copy()
    for c in ("corner1", "corner2", "corner4"):
        h[c] = pd.to_numeric(h[c], errors="coerce")

    def _early(row):
        # 序盤位置: corner2優先→corner1→corner4。0/NaNは欠損として除外
        for c in ("corner2", "corner1", "corner4"):
            v = row[c]
            if pd.notna(v) and v > 0:
                return v
        return float("nan")

    h["early"] = h.apply(_early, axis=1)
    h["fs"] = h["rkey"].map(size_map)
    valid = h[(h["early"].notna()) & (h["fs"].notna()) & (h["fs"] > 1)]
    if valid.empty:
        return {"style": "不明", "c4_avg": None, "c4_rel": None, "n": 0}
    rels = (valid["early"] / valid["fs"])
    rel = rels.mean()
    avg = valid["early"].mean()
    # 脚質の安定度: 前1/3(rel<0.33)も後1/3(rel>0.66)も経験した馬＝先行↔後方を行き来＝前残りの罠
    spread = float(rels.max() - rels.min()) if len(rels) >= 2 else 0.0
    unstable = bool(len(rels) >= 2 and rels.min() < 0.33 and rels.max() > 0.66)

    # 序盤位置(2角)基準の閾値。序盤は隊列が伸びるのでcorner4より境界を絞る
    if rel <= 0.15:
        style = "逃げ"
    elif rel <= 0.35:
        style = "先行"
    elif rel <= 0.65:
        style = "中団"
    else:
        style = "後方"
    n = int(len(valid))
    return {"style": style, "c4_avg": round(float(avg), 1),
            "c4_rel": round(float(rel), 2), "n": n,
            "weak": n < 2, "unstable": unstable, "spread": round(spread, 2)}


def predict_pace(horse_names: list[str], n_recent: int = 5) -> dict:
    """出走馬の脚質構成からペースを予測。

    Returns: {
        pace: ハイ/ミドル/スロー,
        nige: 逃げ頭数, senko: 先行頭数, naka: 中団, ushiro: 後方,
        styles: {horse: style},
        advantage: 展開で有利になる脚質,
        detail: 説明文,
    }
    """
    styles = {h: classify_style(h, n_recent) for h in horse_names}
    nige = sum(1 for s in styles.values() if s["style"] == "逃げ")
    senko = sum(1 for s in styles.values() if s["style"] == "先行")
    naka = sum(1 for s in styles.values() if s["style"] == "中団")
    ushiro = sum(1 for s in styles.values() if s["style"] == "後方")
    n = len([s for s in styles.values() if s["style"] != "不明"])

    # 前に行く馬（逃げ+先行）の比率でペース判定
    front = nige + senko
    front_ratio = front / n if n else 0

    if nige >= 2 or front_ratio >= 0.45:
        pace = "ハイ"
        advantage = "差し・追込（前崩れ濃厚）"
    elif front_ratio <= 0.2 and nige <= 1:
        pace = "スロー"
        advantage = "逃げ・先行（前残り濃厚）"
    else:
        pace = "ミドル"
        advantage = "中団からの差し（標準）"

    # 単騎逃げ判定: 逃げ馬がちょうど1頭 = 楽逃げで残りやすい
    lone_leader = None
    if nige == 1:
        lone_leader = next((h for h, s in styles.items() if s["style"] == "逃げ"), None)

    detail = (f"逃げ{nige} 先行{senko} 中団{naka} 後方{ushiro} → "
              f"{pace}ペース予想（{advantage}が有利）")
    if lone_leader:
        detail += f" / 単騎逃げ濃厚: {lone_leader}（楽逃げ警戒）"
    elif nige >= 3:
        detail += f" / 逃げ{nige}頭で競り合い→前崩れ助長"

    return {
        "pace": pace, "nige": nige, "senko": senko, "naka": naka, "ushiro": ushiro,
        "styles": styles, "advantage": advantage, "detail": detail,
        "lone_leader": lone_leader,
    }


def pace_fit_bonus(horse_name: str, pace_info: dict) -> dict:
    """予測ペースと自分の脚質の噛み合いをボーナス化。

    ハイペース×差し追込 → +、スロー×逃げ先行 → + など。
    """
    s = pace_info["styles"].get(horse_name, {}).get("style", "不明")
    pace = pace_info["pace"]
    bonus = 0.0
    tag = ""
    if pace == "ハイ" and s in ("中団", "後方"):
        bonus = 0.015
        tag = "展開向く(差し有利)"
    elif pace == "ハイ" and s == "逃げ":
        bonus = -0.012
        tag = "展開不利(逃げ潰れ懸念)"
    elif pace == "スロー" and s in ("逃げ", "先行"):
        bonus = 0.015
        tag = "展開向く(前残り)"
    elif pace == "スロー" and s == "後方":
        bonus = -0.012
        tag = "展開不利(届かず懸念)"
    return {"bonus": round(bonus, 3), "tag": tag, "style": s}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tfjv_entries import load_tfjv_entries
    data = load_tfjv_entries("C:/TFJV/TXT/出馬表分析260613.CSV")
    race = data["20260613021111"]
    names = [e["horse_name"] for e in race["entries"]]
    pi = predict_pace(names)
    print(f"=== 函館スプリント 展開予測 ===")
    print(pi["detail"])
    print()
    for h in names:
        s = pi["styles"][h]
        fb = pace_fit_bonus(h, pi)
        print(f"  {h:14s} {s['style']:4s} 4角平均{s['c4_avg']}/{('%.2f'%s['c4_rel']) if s['c4_rel'] else '?'} {fb['tag']}")
