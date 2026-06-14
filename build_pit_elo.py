"""Point-in-Time Elo: 各レース直前(=予想時点)のEloを1行ずつ記録。

リーク除去版。backtest で「そのレースを予想する時点で知り得たElo」のみ使う。
出力: data/horse_elo_pit.parquet [race_key, horse_name, pre_elo, date]
race_key = YYYYMMDD_venue_raceno（horse_elo.build_elo_table と同一定義）
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from horse_elo import _expected_score, _mov_multiplier, _horse_key, INIT_RATING, K_FACTOR

_DATA = Path(__file__).parent / "data"


def build():
    # 馬名衝突バグ対策: Elo累積は horse_id 基準（別世代の同名馬を分離）。
    # 出力の horse_name 列はレース内一意なので下流のrk×horse_name結合はそのまま使える。
    df = pd.read_parquet(_DATA / "tfjv_all.parquet",
                         columns=["horse_name", "horse_id", "rank", "date", "venue", "race_no", "time_diff"])
    df = df.dropna(subset=["horse_name", "rank", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["horse_name"] = df["horse_name"].astype(str).str.strip()
    df["hid"] = _horse_key(df)
    df["race_key"] = (df["date"].dt.strftime("%Y%m%d") + "_" +
                      df["venue"].astype(str) + "_" + df["race_no"].astype(str))
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank"])
    df = df.sort_values(["date", "race_key", "rank"])

    ratings: dict[str, float] = {}
    out_rows = []
    k = K_FACTOR
    for race_key, rdf in df.groupby("race_key", sort=False):
        rdf = rdf.sort_values("rank")
        names = rdf["horse_name"].tolist()
        ids = rdf["hid"].tolist()
        ranks = rdf["rank"].astype(int).tolist()
        tds = pd.to_numeric(rdf.get("time_diff"), errors="coerce").fillna(0).tolist()
        rdate = rdf["date"].iloc[0]
        cur = {hid: ratings.get(hid, INIT_RATING) for hid in ids}
        # 直前Eloを記録（このレースの更新前）。出力キーは horse_name（rk内一意）。
        for hid, nm in zip(ids, names):
            out_rows.append((race_key, nm, round(cur[hid], 1), rdate))
        # Elo更新（horse_id基準）
        new = {hid: cur[hid] for hid in ids}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                hi, hj = ids[i], ids[j]
                s_i = 1.0 if ranks[i] < ranks[j] else (0.5 if ranks[i] == ranks[j] else 0.0)
                e_i = _expected_score(cur[hi], cur[hj])
                delta = k * _mov_multiplier(tds[i]) * (s_i - e_i)
                new[hi] += delta
                new[hj] -= delta
        for hid in ids:
            ratings[hid] = new[hid]

    out = pd.DataFrame(out_rows, columns=["race_key", "horse_name", "pre_elo", "date"])
    out.to_parquet(_DATA / "horse_elo_pit.parquet", index=False)
    print(f"PIT Elo 保存: {len(out):,}行 / {out['race_key'].nunique():,}レース")
    return out


if __name__ == "__main__":
    build()
