"""本命(軸)信頼度判定（PIT検証済みロジック）

第55-56波バックテスト(2023-, リークなし)で確立:
  1-3番人気 × Elo1位 × 過去複勝50%+ → 複勝69%/勝35%（鉄板）
  1番人気でも Elo3位以下 × 過去複勝40%未満 → 複勝59%（罠・相手厚く）

実運用は現在Elo=直前Elo(リークなし)、過去複勝率も過去走のみ=リークなし。
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

_DIR = Path(__file__).parent
_DF = None
_EMAP = None


def _load():
    global _DF, _EMAP
    if _DF is None:
        d = pd.read_parquet(_DIR / "data/tfjv_all.parquet",
                            columns=["horse_name", "rank", "date"])
        d["horse_name"] = d["horse_name"].astype(str).str.strip()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["place"] = (pd.to_numeric(d["rank"], errors="coerce") <= 3).astype(float)
        _DF = d
    if _EMAP is None:
        e = pd.read_parquet(_DIR / "data/horse_elo.parquet")
        _EMAP = dict(zip(e["horse_name"].astype(str).str.strip(), e["elo"]))
    return _DF, _EMAP


def prior_place_rate(name: str, before=None) -> tuple[float | None, int]:
    df, _ = _load()
    h = df[df["horse_name"] == str(name).strip()]
    if before is not None:
        h = h[h["date"] < pd.to_datetime(before)]
    n = len(h)
    if n == 0:
        return None, 0
    return float(h["place"].mean()), n


def build_honmei_reliability(horse_names: list[str], pop_map: dict, before=None) -> dict:
    """出走馬の軸信頼度を判定。Elo順位はフィールド内、過去複勝率は各馬の過去走から。

    Returns: {horse: {"tier","tag","elo_rank","prior_pr","detail"}}
      tier: 鉄板 / 標準 / 罠 / ''(対象外=4人気以下)
    """
    _, emap = _load()
    elos = {h: emap.get(str(h).strip()) for h in horse_names}
    ranked = sorted([(h, e) for h, e in elos.items() if e is not None], key=lambda x: -x[1])
    elo_rank = {h: i + 1 for i, (h, _) in enumerate(ranked)}

    out = {}
    for h in horse_names:
        pop = pop_map.get(h)
        er = elo_rank.get(h)
        pr, n = prior_place_rate(h, before)
        rec = {"tier": "", "tag": "", "elo_rank": er, "prior_pr": pr, "detail": ""}
        if pop is not None and 1 <= pop <= 3 and er is not None and n >= 4:
            if er == 1 and pr is not None and pr >= 0.5:
                rec["tier"] = "鉄板"; rec["tag"] = "◎鉄板軸"
                rec["detail"] = f"Elo1位×過去複勝{pr*100:.0f}%（複勝期待69%級）"
            elif pop == 1 and er >= 3 and pr is not None and pr < 0.4:
                rec["tier"] = "罠"; rec["tag"] = "⚠️危険な1人気"
                rec["detail"] = f"1人気だがElo{er}位×過去複勝{pr*100:.0f}%（信頼薄・相手厚く）"
            else:
                rec["tier"] = "標準"; rec["tag"] = "○標準軸"
                rec["detail"] = f"Elo{er}位×過去複勝{(pr*100 if pr else 0):.0f}%"
        out[h] = rec
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_DIR))
    from tfjv_entries import load_tfjv_entries
    data = load_tfjv_entries("C:/TFJV/TXT/出馬表分析260613.CSV")
    race = data["20260613021111"]
    names = [e["horse_name"] for e in race["entries"]]
    pm = {e["horse_name"]: e["popularity"] for e in race["entries"]}
    res = build_honmei_reliability(names, pm)
    print("=== 函館スプリント 軸信頼度 ===")
    for h in names:
        r = res[h]
        if r["tier"]:
            print(f"  {r['tag']} {h}（{pm[h]}人気）: {r['detail']}")
