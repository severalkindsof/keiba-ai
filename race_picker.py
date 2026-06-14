# -*- coding: utf-8 -*-
"""
race_picker.py … 穴党スタイルのための「荒れ妙味レース」厳選ツール。

ユーザーのレース選定基準（今日の振り返りで確立・最重要ロジック）:
  1. 頭数: 16頭以上が理想。最低14頭。14頭以下は配当が薄く波乱も起きにくい → 原則やらない
  2. 最下位人気の想定オッズ >= 100倍。最下位が100倍未満のレースは妙味が乏しい
  3. オッズの分断が激しいほど旨味（上位は堅いが下位が極端に薄い二極化レース）
  → 「レース選びこそ回収率100%超の最重要ファクター。馬選び以上に大事」

est_odds は出馬表段階の想定オッズ（確定ではない）。これは事前スクリーニング用。
最終判断はレース直前の確定オッズで行う前提。

使い方:
  python -X utf8 race_picker.py [出馬表CSV]   # 省略時は最新の出馬表分析*.CSV
"""
import sys
import glob
import numpy as np
from tfjv_entries import load_tfjv_entries

# 選定しきい値（二段運用）
#   事前(想定オッズ): 16頭以上 × 中穴(50倍超)複数で「広めに候補」を拾う。
#       想定オッズは確定より圧縮され、最下位100倍を事前必須にすると荒れるレースを取りこぼすため。
#   確定(直前オッズ): 同ツールに確定オッズ版CSVを食わせ、最下位>=100倍・分断で最終的に絞る。
MIN_FIELD = 14          # これ未満は除外（少頭数は配当薄・波乱起きにくい）
IDEAL_FIELD = 16        # 事前候補の頭数下限（理想）
MIN_MID = 2             # 事前候補に必要な中穴(50倍超)頭数
MIN_BOTTOM_ODDS = 100   # 確定段階での最下位オッズ下限（本命候補の目安）


def _latest_csv():
    files = sorted(glob.glob("C:/TFJV/TXT/出馬表分析*.CSV"))
    return files[-1] if files else None


def _metrics_from_odds(odds, field):
    """オッズのリスト(想定 or 確定)と頭数から妙味メトリクスを算出。"""
    odds = [o for o in odds if o and o > 0]
    if len(odds) < 3:
        return None
    odds_sorted = sorted(odds)
    top = odds_sorted[0]              # 1番人気
    bottom = odds_sorted[-1]          # 最下位
    n_100 = sum(1 for o in odds if o >= 100)   # 100倍超(大穴)の頭数
    n_50 = sum(1 for o in odds if o >= 50)     # 50倍超(中穴以上)の頭数
    spread = bottom / top if top else 0        # 分断度(最下位/最上位)
    return {
        "field": field, "top": top, "bottom": bottom,
        "n_100": n_100, "n_50": n_50, "spread": spread,
    }


def _race_metrics(info):
    odds = [e.get("est_odds") for e in info["entries"]]
    return _metrics_from_odds(odds, len(info["entries"]))


def _score(m):
    """荒れ妙味スコア。頭数・最下位オッズ・分断度・大穴頭数を合成。"""
    field_s = min(m["field"], 18) / IDEAL_FIELD            # 頭数(16で1.0、18で1.125)
    bottom_s = min(m["bottom"], 500) / MIN_BOTTOM_ODDS     # 最下位オッズ(100で1.0、500で5.0)
    spread_s = min(m["spread"], 300) / 100                 # 分断度
    longshot_s = m["n_50"] / 6                             # 中穴以上の的の多さ
    return field_s * (bottom_s + spread_s + longshot_s)


def _line(info, m, sc=None, prime=False):
    star = "★" if prime else " "
    tail = f"| 妙味{sc:.2f}" if sc is not None else ""
    cls = info.get("race_class", "")[:8]
    sd = f"{info.get('surface','')}{info.get('distance','')}"
    return (f" {star}{info['venue']}{info['race_no']:>2}R {cls:8s} {sd:7s} {m['field']}頭 | "
            f"最下位{m['bottom']:.0f}倍 1人気{m['top']:.1f}倍 分断{m['spread']:.0f}x "
            f"大穴{m['n_100']}頭 中穴{m['n_50']}頭 {tail}")


def _render(rows, header, odds_kind):
    rows.sort(key=lambda x: -x[2])
    print(header)
    print(f"事前候補基準: 頭数>={IDEAL_FIELD} × 中穴(50倍超)>={MIN_MID}頭（広め拾い）/ "
          f"★=最下位>={MIN_BOTTOM_ODDS}倍の本命級（{odds_kind}）")
    print("=" * 82)
    print(f"◎狙い目（妙味順・★は最下位100倍超で荒れ濃厚）")
    any_pass = False
    for info, m, sc, passed, prime in rows:
        if not passed:
            continue
        any_pass = True
        print(_line(info, m, sc, prime))
    if not any_pass:
        print("  該当なし")
    print("-" * 82)
    print(f"△参考（14-15頭 or 中穴{MIN_MID}頭未満）")
    for info, m, sc, passed, prime in rows:
        if passed or m["field"] < MIN_FIELD:
            continue
        print(_line(info, m, None, prime))
    print("-" * 82)
    excl = [r for r in rows if r[1]["field"] < MIN_FIELD]
    if excl:
        print(f"✕除外（{MIN_FIELD}頭未満・少頭数）: " +
              ", ".join(f"{i['venue']}{i['race_no']}R({m['field']}頭)" for i, m, *_ in excl))


def _build_row(info, m):
    passed = (m["field"] >= IDEAL_FIELD and m["n_50"] >= MIN_MID)  # 事前候補(広め)
    prime = m["bottom"] >= MIN_BOTTOM_ODDS                          # 確定段階の本命級
    return (info, m, _score(m), passed, prime)


def pick(csv_path=None):
    """想定オッズ(est_odds)版: TFJV出馬表CSVから選定（段1=事前スクリーニング）。"""
    csv_path = csv_path or _latest_csv()
    if not csv_path:
        print("出馬表CSVが見つかりません")
        return
    data = load_tfjv_entries(csv_path)
    rows = [_build_row(info, m) for info in data.values()
            if (m := _race_metrics(info))]
    _render(rows, f"[race_picker/想定オッズ] {csv_path}", "想定オッズ・確定で開く可能性あり")


def pick_live(date_yyyymmdd):
    """確定/リアルタイムオッズ版: netkeibaから当日全レースを取得（段2=確定絞り）。"""
    import time
    from netkeiba_odds import fetch_race_ids, fetch_win_odds
    ids = fetch_race_ids(date_yyyymmdd)
    if not ids:
        print(f"{date_yyyymmdd} の開催が見つかりません")
        return
    rows = []
    for (venue, rno), rid in sorted(ids.items()):
        try:
            od = fetch_win_odds(rid)
        except Exception:
            continue
        time.sleep(0.3)  # netkeibaへの礼儀
        m = _metrics_from_odds([o for o, _ in od.values()], len(od))
        if not m:
            continue
        info = {"venue": venue, "race_no": rno}
        rows.append(_build_row(info, m))
    _render(rows, f"[race_picker/確定オッズ netkeiba] {date_yyyymmdd}", "確定/リアルタイムオッズ")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and arg.isdigit() and len(arg) == 8:
        pick_live(arg)            # YYYYMMDD → netkeiba確定オッズで選定
    else:
        pick(arg)                 # CSVパス or 省略 → 想定オッズで選定
