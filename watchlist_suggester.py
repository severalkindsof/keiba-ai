# -*- coding: utf-8 -*-
"""
watchlist_suggester.py … レース結果から「次走でウォッチしたい馬」を機械抽出。

ユーザーの主観登録だけだと取りこぼすため、parquet の確定結果を走査して候補を自動提案。
keiba-pdca の Step 4 (watchlist 更新) で呼び出す想定。

検出ルール (v1):
  A. dark_horse  : 10人気以下で3着内 (大穴穴馬)
  B. overperform : 人気の1/3着以内かつ4着以内 (実力ありそうに見える)
  C. upset_3f    : 同レース上がり最速かつ rank<=5 かつ 4人気以下 (脚あり穴馬)

既に watchlist 登録済みの馬は除外。出力は markdown 形式で、
buy_condition のヒントを「過去走 (>=3走) の複勝率 >=50%」基準で示唆する。

使い方:
  python -X utf8 watchlist_suggester.py YYYY-MM-DD
  python -X utf8 watchlist_suggester.py YYYY-MM-DD,YYYY-MM-DD     # 複数日
  python -X utf8 watchlist_suggester.py YYYY-MM-DD --venue 東京   # 会場絞り
  python -X utf8 watchlist_suggester.py YYYY-MM-DD --min-reasons 2  # 検出理由を強める

出力後の運用:
  Claude (keiba-pdca) がこの出力を読み、ユーザーと対話して
  「どの馬を登録するか」「buy_condition をどう書くか」を確定 → watchlist.json に追記。
"""
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

_DIR = Path(__file__).parent
PARQUET = _DIR / "data" / "tfjv_all.parquet"
WATCHLIST = _DIR / "watchlist.json"


def _s(x):
    """str.strip safe."""
    if x is None:
        return ""
    return str(x).strip()


def load_watchlist():
    return json.loads(WATCHLIST.read_text(encoding="utf-8"))


def already_watched(name, wl):
    name = _s(name)
    return any(_s(h.get("name")) == name for h in wl.get("horses", []))


def detect_candidates(df_day):
    """対象日(複数日可)の全行から候補を抽出。"""
    out = []
    for race_id, g in df_day.groupby("race_id"):
        if g.empty:
            continue
        race_meta = {
            "race_name": _s(g["race_name"].iloc[0]),
            "venue": _s(g["venue"].iloc[0]),
            "race_no": _s(g["race_no"].iloc[0]),
            "surface": _s(g["surface"].iloc[0]),
            "distance": int(g["distance"].iloc[0]) if pd.notna(g["distance"].iloc[0]) else 0,
            "track_condition": _s(g["track_condition"].iloc[0]),
            "field_size": int(g["field_size"].iloc[0]) if pd.notna(g["field_size"].iloc[0]) else 0,
            "date": _s(g["date"].iloc[0]),
        }
        # 新馬戦は除外 (PDCA対象外)
        if "新馬" in race_meta["race_name"]:
            continue
        # 上がり最速値
        last3f_series = g["last_3f"].dropna()
        last3f_min = last3f_series.min() if not last3f_series.empty else None

        for _, row in g.iterrows():
            rank = int(row["rank"]) if pd.notna(row["rank"]) else 99
            pop = int(row["popularity"]) if pd.notna(row["popularity"]) else 99
            last3f = row["last_3f"] if pd.notna(row["last_3f"]) else None

            reasons = []
            # A. 大穴3着内
            if rank <= 3 and pop >= 10:
                reasons.append(f"大穴3着内({pop}人気→{rank}着)")
            # B. 人気以上の好走 (人気を3で割った着順以下、かつ4着以内)
            if rank <= 4 and pop >= 4 and rank <= max(2, pop // 3):
                reasons.append(f"人気以上({pop}人気→{rank}着)")
            # C. 上がり最速で着順イマイチ (次走以降に評価)
            if (last3f_min is not None and last3f is not None
                    and abs(last3f - last3f_min) < 1e-6
                    and rank <= 5 and pop >= 4):
                reasons.append(f"上がり最速({last3f}・{pop}人気{rank}着)")

            if reasons:
                out.append({
                    "name": _s(row["horse_name"]),
                    "jockey": _s(row["jockey"]),
                    "sire": _s(row["sire"]),
                    "rank": rank,
                    "popularity": pop,
                    "reasons": reasons,
                    "race": race_meta,
                })
    return out


def suggest_conditions(cand, df_all):
    """過去走(>=3走)を見て買い条件ヒントを返す。"""
    name = cand["name"]
    past = df_all[df_all["horse_name"].apply(_s) == name].copy()
    if len(past) < 3:
        return {}
    hints = {}
    # 馬場別 複勝率
    bg = past.groupby(past["track_condition"].apply(_s)).agg(
        n=("rank", "count"), top3=("place_flag", "sum")
    )
    good_tracks = []
    for cond, row in bg.iterrows():
        if row["n"] >= 2 and row["top3"] / row["n"] >= 0.5 and cond in ["稍重", "重", "不良"]:
            good_tracks.append(cond)
    if good_tracks:
        hints["track_condition"] = good_tracks
    # 騎手コンビ
    same_j = past[past["jockey"].apply(_s) == cand["jockey"]]
    if len(same_j) >= 2 and same_j["place_flag"].mean() >= 0.5:
        hints["jockey"] = cand["jockey"]
    # 会場相性
    same_v = past[past["venue"].apply(_s) == cand["race"]["venue"]]
    if len(same_v) >= 2 and same_v["place_flag"].mean() >= 0.5:
        hints["venue"] = cand["race"]["venue"]
    # 距離帯 (現在距離±200で過去複勝率)
    cur = cand["race"]["distance"]
    if cur:
        near = past[(past["distance"] >= cur - 200) & (past["distance"] <= cur + 200)]
        if len(near) >= 2 and near["place_flag"].mean() >= 0.5:
            hints["distance_range"] = [int(near["distance"].min()), int(near["distance"].max())]
    return hints


def render_markdown(cands, df_all, dates):
    lines = []
    lines.append(f"# watchlist 候補 ({', '.join(dates)})\n")
    lines.append(f"検出: **{len(cands)}頭** (既登録/新馬戦は除外済み)\n")
    if not cands:
        lines.append("候補なし。\n")
        return "\n".join(lines)

    # ソート: 理由が多い順 → 着順小さい順
    cands_sorted = sorted(cands, key=lambda x: (-len(x["reasons"]), x["rank"]))

    for i, c in enumerate(cands_sorted, 1):
        r = c["race"]
        hints = suggest_conditions(c, df_all)
        lines.append(f"## [{i}] {c['name']}")
        lines.append(f"- 検出理由: {' / '.join(c['reasons'])}")
        lines.append(
            f"- レース: {r['date']} {r['venue']}{r['race_no']}R "
            f"{r['race_name']} ({r['surface']}{r['distance']}m / "
            f"{r['track_condition']} / {r['field_size']}頭)"
        )
        lines.append(f"- 騎手: {c['jockey']} / 父: {c['sire']}")
        if hints:
            hint_strs = []
            for k, v in hints.items():
                hint_strs.append(f"{k}={v}")
            lines.append(f"- 過去走ヒント (>=3走・複勝率>=50%): {', '.join(hint_strs)}")
        else:
            lines.append(f"- 過去走ヒント: なし (過去走<3 or 顕著な偏りなし)")
        lines.append(f"- 提案 buy_condition: (対話で確定)")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dates", help="YYYY-MM-DD or YYYY-MM-DD,YYYY-MM-DD")
    p.add_argument("--venue", default=None, help="会場で絞る(例:東京)")
    p.add_argument("--min-reasons", default=1, type=int, help="最低検出理由数(>=1)")
    p.add_argument("--out", default=None, help="出力ファイル(省略時stdout)")
    args = p.parse_args()

    targets = [d.strip() for d in args.dates.split(",")]
    if not PARQUET.exists():
        sys.exit(f"parquet not found: {PARQUET}")
    df = pd.read_parquet(PARQUET)
    df_day = df[df["date"].isin(targets)]
    if args.venue:
        df_day = df_day[df_day["venue"].apply(_s) == args.venue]
    if df_day.empty:
        print(f"該当日のデータなし: {targets} / venue={args.venue}")
        print("convert_tfjv.py --incremental で取込済みか確認してください。")
        return

    wl = load_watchlist()
    cands = detect_candidates(df_day)
    cands = [c for c in cands if len(c["reasons"]) >= args.min_reasons]
    cands = [c for c in cands if not already_watched(c["name"], wl)]

    md = render_markdown(cands, df, targets)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"saved: {args.out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
