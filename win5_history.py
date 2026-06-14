"""win5_2011_2026.txt をパースして構造化。
各開催: date, payout(円), 的中票数, 5レッグ(会場/距離/頭数/勝ち馬人気/オッズ)
"""
import re
import pandas as pd
from pathlib import Path

_F = Path("C:/TFJV/TXT/win5_2011_2026.txt")


def parse():
    txt = _F.read_text(encoding="cp932", errors="replace")
    blocks = txt.split("◆")
    rows = []
    for b in blocks[1:]:
        lines = b.splitlines()
        mdate = re.match(r"\s*(\d{4})年\s*(\d+)月\s*(\d+)日", lines[0])
        if not mdate:
            continue
        y, mo, d = mdate.groups()
        date = f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        # 配当・的中票
        payout = None; hit_votes = None
        for l in lines:
            mp = re.search(r"【\s*([\d,]+)円】\(的中\s*:\s*([\d,]+)票\)", l)
            if mp:
                payout = int(mp.group(1).replace(",", ""))
                hit_votes = int(mp.group(2).replace(",", ""))
                break
        # 各レッグ: 会場1文字+レース番号(1-2桁)+距離+頭数+勝ち馬人気
        # 1桁は "東 9R"(空白), 2桁は "阪10R"(詰まる)
        _VMAP = {"札":"札幌","函":"函館","福":"福島","新":"新潟","東":"東京",
                 "中":"中山","名":"中京","京":"京都","阪":"阪神","小":"小倉"}
        legs = []
        for l in lines:
            ml = re.match(r"\s*(\d)(st|nd|rd|th)\s+(.)\s*(\d{1,2})R\s+(.+?)\s+([芝ダ障])(\d+)\s+(\d+)頭\s+(\d+)人", l)
            if ml:
                legs.append({
                    "leg": int(ml.group(1)),
                    "venue": _VMAP.get(ml.group(3), ml.group(3)),
                    "race_no": int(ml.group(4)),
                    "surface": ml.group(6),
                    "distance": int(ml.group(7)),
                    "field": int(ml.group(8)),
                    "win_pop": int(ml.group(9)),
                })
        if payout and len(legs) == 5:
            rows.append({"date": date, "payout": payout, "hit_votes": hit_votes,
                         "pops": [lg["win_pop"] for lg in legs],
                         "legs": legs})
    return rows


if __name__ == "__main__":
    rows = parse()
    df = pd.DataFrame([{"date": r["date"], "payout": r["payout"],
                        "hit_votes": r["hit_votes"],
                        "max_pop": max(r["pops"]), "sum_pop": sum(r["pops"]),
                        "fav_legs": sum(1 for p in r["pops"] if p == 1),
                        "longshot_legs": sum(1 for p in r["pops"] if p >= 7)}
                       for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    print(f"WIN5 開催数: {len(df)} ({df['date'].min().date()} 〜 {df['date'].max().date()})")
    print(f"\n配当分布:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
        print(f"  {int(q*100)}%tile: {df['payout'].quantile(q):,.0f}円")
    print(f"  平均: {df['payout'].mean():,.0f}円  最高: {df['payout'].max():,.0f}円")
    print(f"\n1番人気で決まったレッグ数の分布（5レッグ中）:")
    print(df["fav_legs"].value_counts().sort_index().to_string())
    print(f"\n7人気以下が勝ったレッグ数の分布:")
    print(df["longshot_legs"].value_counts().sort_index().to_string())
    df.to_parquet(Path(__file__).parent/"data/win5_history.parquet", index=False)
    print("\n保存: data/win5_history.parquet")
