"""B[84] 相手強度（Opponent Strength）

「過去にどれだけ強い相手と戦ってきたか」を測る。
強敵相手に僅差で負けてきた馬は、格下相手のレースに替わると一変しやすい
（= 隠れ実力馬の検出）。

指標:
- opp_elo_avg : 直近N走で対戦した相手馬の平均Elo
- opp_elo_max : 直近N走で対戦した最強相手のElo
- tough_close : 強敵(相手平均Elo以上)に僅差(time_diff<=0.3秒)で負けた回数
- bonus       : 相手強度に応じた加点（隠れ実力馬ボーナス）
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

_TFJV = Path(__file__).parent / "data" / "tfjv_all.parquet"
_DF = None
_ELO_MAP = None


def _load():
    global _DF, _ELO_MAP
    if _DF is None:
        _DF = pd.read_parquet(_TFJV, columns=[
            "horse_name", "race_id", "rank", "time_diff", "date"])
        _DF["horse_name"] = _DF["horse_name"].astype(str).str.strip()
        _DF["date"] = pd.to_datetime(_DF["date"], errors="coerce")
        # TFJV race_id は末尾2桁が馬番。先頭8桁が真のレースキー
        _DF["rkey"] = _DF["race_id"].astype(str).str[:8]
    if _ELO_MAP is None:
        from horse_elo import _load_elo
        e = _load_elo()
        _ELO_MAP = dict(zip(e["horse_name"].astype(str).str.strip(), e["elo"]))
    return _DF, _ELO_MAP


def build_opponent_strength(horse_name: str, n_recent: int = 5) -> dict:
    """1頭の相手強度を返す。"""
    df, elo_map = _load()
    name = str(horse_name).strip()
    h = df[df["horse_name"] == name].sort_values("date", ascending=False)
    if h.empty:
        return {"tag": "", "bonus": 0.0, "detail": "",
                "opp_elo_avg": None, "opp_elo_max": None, "tough_close": 0}

    recent = h.head(n_recent)
    rkeys = recent["rkey"].dropna().tolist()
    if not rkeys:
        return {"tag": "", "bonus": 0.0, "detail": "",
                "opp_elo_avg": None, "opp_elo_max": None, "tough_close": 0}

    # 直近N走の全出走馬を一括取得（自分以外）
    field = df[df["rkey"].isin(rkeys)].copy()
    field = field[field["horse_name"] != name]
    field["elo"] = field["horse_name"].map(elo_map)

    opp_all = field["elo"].dropna()
    if opp_all.empty:
        return {"tag": "", "bonus": 0.0, "detail": "",
                "opp_elo_avg": None, "opp_elo_max": None, "tough_close": 0}

    opp_elo_avg = float(opp_all.mean())
    opp_elo_max = float(opp_all.max())

    # レースごとの相手平均Eloと、自分のそのレースの time_diff を突合
    tough_close = 0
    recent2 = recent.copy()
    recent2["td"] = pd.to_numeric(recent2["time_diff"], errors="coerce")
    recent2["rank_n"] = pd.to_numeric(recent2["rank"], errors="coerce")
    for _, r in recent2.iterrows():
        rid = r["rkey"]
        opp_this = field[field["rkey"] == rid]["elo"].dropna()
        if opp_this.empty:
            continue
        race_opp_avg = float(opp_this.mean())
        td = r["td"]
        rk = r["rank_n"]
        # 強敵(相手平均が高い)に僅差(0.3秒以内)で負けた = 善戦
        if (pd.notna(td) and 0 < td <= 0.3 and race_opp_avg >= opp_elo_avg
                and pd.notna(rk) and rk >= 2):
            tough_close += 1

    # ボーナス設計: 強敵僅差善戦が主軸（クラス下げ判定は field 側で付与）
    bonus = 0.0
    tags = []
    detail_parts = []

    if tough_close >= 2:
        bonus += 0.020
        tags.append("強敵僅差善戦")
        detail_parts.append(f"強敵に僅差負け{tough_close}回（地力上位の証拠）")
    elif tough_close == 1:
        bonus += 0.008
        detail_parts.append("強敵に僅差負け1回")

    return {
        "tag": " / ".join(tags),
        "bonus": round(bonus, 3),
        "detail": " / ".join(detail_parts),
        "opp_elo_avg": round(opp_elo_avg, 0),
        "opp_elo_max": round(opp_elo_max, 0),
        "tough_close": tough_close,
    }


def build_field_opponent_strength(horse_names: list[str], n_recent: int = 5) -> dict:
    """出走馬全頭の相手強度を一括計算し、レース内相対 + クラス下げ判定を付ける。

    クラス下げ判定: 各馬の過去対戦相手平均Elo が、今回の出走メンバー平均Elo を
    大きく上回る = 「格上のところで揉まれてきた馬が楽なメンバーに入った」= 強調材料。
    """
    df, elo_map = _load()
    results = {h: build_opponent_strength(h, n_recent) for h in horse_names}

    # 今回メンバーの Elo 平均（このレースの相手の強さ）
    field_elos = [elo_map.get(str(h).strip()) for h in horse_names]
    field_elos = [e for e in field_elos if e is not None]
    field_avg = float(np.mean(field_elos)) if field_elos else None

    # レース内で相手強度の高い順にランク付け
    valid = [(h, r["opp_elo_avg"]) for h, r in results.items() if r["opp_elo_avg"] is not None]
    valid.sort(key=lambda x: -x[1])
    for i, (h, _) in enumerate(valid, 1):
        results[h]["opp_strength_rank"] = i

    for h, r in results.items():
        r.setdefault("opp_strength_rank", None)
        # クラス下げ判定: 過去対戦相手平均 >> 今回メンバー平均
        if field_avg is not None and r["opp_elo_avg"] is not None:
            drop = r["opp_elo_avg"] - field_avg
            r["class_drop"] = round(drop, 0)
            if drop >= 150:
                r["bonus"] = round(r["bonus"] + 0.018, 3)
                _t = "格上挑戦帰り" if "強敵僅差善戦" not in r["tag"] else r["tag"]
                r["tag"] = (r["tag"] + " / 格下げ") if r["tag"] else "格下げ"
                _d = f"過去相手平均{r['opp_elo_avg']:.0f} > 今回平均{field_avg:.0f}（+{drop:.0f}・楽なメンバー）"
                r["detail"] = (r["detail"] + " / " + _d) if r["detail"] else _d
            elif drop <= -150:
                r["class_up"] = True
                _d = f"今回は格上挑戦（過去相手平均{r['opp_elo_avg']:.0f} < 今回平均{field_avg:.0f}）"
                r["detail"] = (r["detail"] + " / " + _d) if r["detail"] else _d
        else:
            r["class_drop"] = None
    return results


if __name__ == "__main__":
    # 函館スプリント出走馬で検証
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    targets = ["レイピア", "モズナナスター", "ダノンマッキンリー",
               "シュタールヴィント", "エーティーマクフィ"]
    res = build_field_opponent_strength(targets)
    for h in targets:
        r = res[h]
        print(f"{h:14s} 相手平均Elo={r['opp_elo_avg']} 最強={r['opp_elo_max']} "
              f"強敵僅差={r['tough_close']}回 bonus={r['bonus']:+.3f} | {r['detail']}")
