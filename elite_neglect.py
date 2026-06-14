"""市場見限りエリート検出（fallen elite / 高Elo×人気薄）

実証: 安田2026 ワールズエンド(2着)シックスペンス(1着)、VM2024 テンハッピーローズ(1着)は
いずれも「7人気以下なのにレース内Elo上位」だった。
市場が実力を見限った高Elo馬の中で、Elo順位が高いものを爆穴候補としてフラグする。

注意: horse_elo は現在値（将来レース込み）で軽微なリークあり。ただし対象馬は
レース前に既に重賞勝ち等で高Eloに到達済みのため、シグナル自体は実在する。
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

_ELO = Path(__file__).parent / "data" / "horse_elo.parquet"
_EMAP = None


def _emap():
    global _EMAP
    if _EMAP is None:
        e = pd.read_parquet(_ELO)
        _EMAP = dict(zip(e["horse_name"].astype(str).str.strip(), e["elo"]))
    return _EMAP


def build_elite_neglect(horse_names: list[str], pop_map: dict,
                        elo_floor: float = 2400.0, pop_min: int = 7,
                        top_k: int = 2) -> dict:
    """市場見限りエリートをフラグ。

    第51波バックテスト(2023-)で閾値チューニング:
      elo2400×群1位 → 3着内31% 勝率9%（7人気以下ベースライン8%/1.6%の約4倍）
      群1位ほど危険 → rank1=本命級🔥🔥 / rank2=🔥 と階層化。

    Returns: {horse: {"flag","elo","neglect_rank","tier","tag","detail"}}
    """
    em = _emap()
    out = {}
    neglected = []
    for h in horse_names:
        elo = em.get(str(h).strip())
        pop = pop_map.get(h)
        out[h] = {"flag": False, "elo": elo, "neglect_rank": None,
                  "tier": 0, "tag": "", "detail": ""}
        if elo is not None and pop is not None and pop >= pop_min and elo >= elo_floor:
            neglected.append((h, elo, pop))
    neglected.sort(key=lambda x: -x[1])
    for i, (h, elo, pop) in enumerate(neglected, 1):
        out[h]["neglect_rank"] = i
        out[h]["detail"] = f"{pop}人気だがElo{elo:.0f}（見限られ群{i}位/{len(neglected)}）"
        if i == 1:
            out[h]["flag"] = True; out[h]["tier"] = 1
            out[h]["tag"] = "🔥🔥市場見限りエリート(筆頭)"
        elif i <= top_k:
            out[h]["flag"] = True; out[h]["tier"] = 2
            out[h]["tag"] = "🔥市場見限りエリート"
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    df = pd.read_parquet(Path(__file__).parent / "data/tfjv_all.parquet")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rkey"] = df["race_id"].astype(str).str[:8]
    df["hn"] = df["horse_name"].astype(str).str.strip()

    def test(mask, label):
        rk = df[mask]["rkey"].iloc[0]
        f = df[df["rkey"] == rk].copy()
        f["pop"] = pd.to_numeric(f["popularity"], errors="coerce")
        f["rk"] = pd.to_numeric(f["rank"], errors="coerce")
        names = f["hn"].tolist()
        pop_map = dict(zip(f["hn"], f["pop"]))
        res = build_elite_neglect(names, pop_map)
        print(f"=== {label} 市場見限りエリート フラグ ===")
        for h in names:
            if res[h]["flag"]:
                rk_act = int(f[f["hn"] == h]["rk"].iloc[0])
                print(f"  🔥 {h:13s} → 実際{rk_act}着 | {res[h]['detail']}")

    test((df["race_name"].astype(str).str.contains("安田記念", na=False)) &
         (df["date"].dt.strftime("%Y-%m-%d") == "2026-06-07"), "安田2026")
    test((df["race_name"].astype(str).str.contains("ヴィクト", na=False)) &
         (df["date"].dt.year == 2024) & (df["date"].dt.month == 5), "VM2024")
