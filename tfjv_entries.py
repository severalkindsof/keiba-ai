"""TFJV「出馬表分析」CSV を取込み、各レースの出走馬リストを返す。

入力: C:/TFJV/TXT/出馬表分析YYMMDD.CSV（33列 cp932）

各レースは行順で連続。馬番(col3)が前行より小さくなった or
クラス/距離/会場が変わった = 新レース開始 と判定。

返り値は既存 scraper.fetch_race_entries() 互換の dict リスト形式：
[
  {race_id, horse_name, horse_no, jockey, weight_carried,
   sex_age, sire, dam, popularity(None), odds(None)},
  ...
]
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

VENUE_CODES = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}

COLS = [
    "date", "venue", "kaisai", "horse_no", "race_class", "surface", "distance",
    "horse_name", "sex", "age", "jockey", "weight_carried", "trainer",
    "east_west", "owner", "breeder", "sire", "dam", "horse_id",
    "col19", "damsire", "col21", "col22", "col23", "col24", "col25",
    "field_size", "idx1", "idx2", "col29", "est_odds", "col31", "internal_id",
]


def load_tfjv_entries(csv_path: str | Path) -> dict:
    """出馬表分析CSV を読み込み、レース毎の出走馬辞書を返す。

    Returns
    -------
    {
      race_id: {
        "venue": str, "race_no": int, "surface": str, "distance": int,
        "race_class": str, "field_size": int, "date_str": "YYYYMMDD",
        "entries": [
          {"horse_no": int, "horse_name": str, "sex_age": str,
           "jockey": str, "weight_carried": float, "trainer": str,
           "sire": str, "dam": str, "horse_id": str,
           "est_odds": float | None},
          ...
        ]
      }, ...
    }
    """
    df = pd.read_csv(csv_path, encoding="cp932", dtype=str,
                     header=None, on_bad_lines="skip")
    if len(df.columns) != 33:
        raise ValueError(f"列数が33ではない: {len(df.columns)}")
    df.columns = COLS
    df["horse_no_n"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")

    # レース区切り: 馬番が前行より小さい OR 距離が変わった OR 会場が変わった
    df["new_race"] = (
        (df["horse_no_n"] <= df["horse_no_n"].shift(1).fillna(0))
        | (df["distance"] != df["distance"].shift(1))
        | (df["venue"] != df["venue"].shift(1))
    )
    df["race_local_id"] = df["new_race"].cumsum()  # 同一CSV内の連番

    # 会場ごとにレース番号を振り直す（venue内で 1, 2, 3...）
    race_no_map = {}
    for vname, grp in df.groupby("venue", sort=False):
        for i, lid in enumerate(grp["race_local_id"].drop_duplicates(), 1):
            race_no_map[(vname, lid)] = i
    df["race_no"] = df.apply(
        lambda r: race_no_map.get((r["venue"], r["race_local_id"]), 0), axis=1
    )

    out: dict = {}
    for (vname, rno), grp in df.groupby(["venue", "race_no"], sort=False):
        v_code = VENUE_CODES.get(vname, "00")
        # 日付: YYMMDD → YYYYMMDD（20YY）
        yymmdd = str(grp.iloc[0]["date"]).zfill(6)
        yyyymmdd = "20" + yymmdd
        kaisai = str(grp.iloc[0]["kaisai"]).strip().zfill(2)
        # 簡易 race_id: YYYYMMDD + 会場 + 開催回 + R番号（本来のnetkeiba形式と異なる）
        race_id = f"{yyyymmdd}{v_code}{kaisai}{int(rno):02d}"

        first = grp.iloc[0]
        # est_odds の順位から popularity を推定
        _odds_rows = []
        for i, (_, row) in enumerate(grp.iterrows()):
            try:
                eo = float(row["est_odds"])
                if eo <= 0:
                    eo = 999.0
            except Exception:
                eo = 999.0
            _odds_rows.append((i, eo))
        _odds_rows.sort(key=lambda x: x[1])
        _pop_map = {idx: rank + 1 for rank, (idx, _) in enumerate(_odds_rows)}

        entries = []
        for i, (_, row) in enumerate(grp.iterrows()):
            try:
                est_odds = float(row["est_odds"])
                if est_odds <= 0:
                    est_odds = None
            except Exception:
                est_odds = None
            try:
                wc = float(row["weight_carried"])
            except Exception:
                wc = 0.0
            # 第45波: odds/popularity を est_odds から補完（netkeiba 取得失敗時のフォールバック）
            _odds_val = est_odds if est_odds is not None else 10.0
            _pop_val = _pop_map.get(i, 9)
            entries.append({
                "horse_no": int(row["horse_no_n"]) if pd.notna(row["horse_no_n"]) else 0,
                "horse_name": str(row["horse_name"]).strip(),
                "sex_age": f"{row['sex']}{row['age']}",
                "sex": str(row["sex"]).strip(),
                "age": int(row["age"]) if str(row["age"]).strip().isdigit() else 0,
                "jockey": str(row["jockey"]).strip(),
                "weight_carried": wc,
                "trainer": str(row["trainer"]).strip(),
                "sire": str(row["sire"]).strip(),
                "dam": str(row["dam"]).strip(),
                "damsire": str(row["damsire"]).strip(),
                "horse_id": str(row["horse_id"]).strip(),
                "est_odds": est_odds,
                "horse_weight": None,  # netkeiba 補完待ち
                "odds": _odds_val,      # est_odds をフォールバック値として
                "popularity": _pop_val, # est_odds順位から推定
            })
        out[race_id] = {
            "venue":      vname,
            "race_no":    int(rno),
            "surface":    str(first["surface"]).strip(),
            "distance":   int(first["distance"]),
            "race_class": str(first["race_class"]).strip(),
            "field_size": int(first.get("field_size", 0)) if str(first.get("field_size", "0")).strip().isdigit() else len(entries),
            "date_str":   yyyymmdd,
            "entries":    entries,
        }
    return out


def list_races(csv_path: str | Path) -> list[dict]:
    """レース一覧（既存 fetch_today_races 互換のサマリリスト）を返す。"""
    races = load_tfjv_entries(csv_path)
    out = []
    for rid, info in races.items():
        out.append({
            "race_id":   rid,
            "race_name": f"{info['venue']}{info['race_no']}R {info['race_class']}",
            "date_str":  info["date_str"],
            "venue":     info["venue"],
            "race_no":   info["race_no"],
            "surface":   info["surface"],
            "distance":  info["distance"],
            "field_size": info["field_size"],
        })
    return out


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "C:/TFJV/TXT/出馬表分析260613.CSV"
    races = list_races(p)
    print(f"取得レース数: {len(races)}")
    for r in races[:5]:
        print(f"  {r['race_id']} {r['race_name']} ({r['surface']}{r['distance']}m, {r['field_size']}頭)")
    print("...")
    for r in races[-3:]:
        print(f"  {r['race_id']} {r['race_name']} ({r['surface']}{r['distance']}m, {r['field_size']}頭)")
