"""3連複2026_2008（実払戻ファイル）をパース。
固定位置: 三連複[179-181]+配当[182] / 三連単[194-196]+配当[197] / 馬単[155-156]+[157]
race_key = YYYYMMDD_venue_raceno（tfjv側と同一定義）で結合可能に。
"""
import pandas as pd
from pathlib import Path

_F = Path("C:/TFJV/TXT/3連複2026_2008")
_OUT = Path(__file__).parent / "data/payouts.parquet"


def parse():
    rows = []
    with open(_F, encoding="cp932", errors="replace") as f:
        for line in f:
            x = line.rstrip().split(",")
            if len(x) < 200:
                continue
            try:
                yy = int(x[0]); mm = int(x[1]); dd = int(x[2])
                year = 2000 + yy if yy < 50 else 1900 + yy
                venue = x[4]; raceno = int(x[6])
                rkey = f"{year:04d}{mm:02d}{dd:02d}_{venue}_{raceno}"
                # 三連複
                t3 = sorted([x[179], x[180], x[181]])
                try:
                    san_pay = int(x[182])
                except Exception:
                    san_pay = None
                # 三連単
                try:
                    tan3_pay = int(x[197])
                except Exception:
                    tan3_pay = None
                # 馬単
                try:
                    umatan_pay = int(x[157])
                except Exception:
                    umatan_pay = None
                if san_pay and san_pay > 0:
                    rows.append({"race_key": rkey,
                                 "sanrenpuku": san_pay,
                                 "sanrentan": tan3_pay,
                                 "umatan": umatan_pay})
            except Exception:
                continue
    df = pd.DataFrame(rows)
    df.to_parquet(_OUT, index=False)
    return df


if __name__ == "__main__":
    df = parse()
    print(f"払戻パース: {len(df):,}レース")
    print(f"三連複配当 中央値{df['sanrenpuku'].median():,.0f}円 平均{df['sanrenpuku'].mean():,.0f}円 最高{df['sanrenpuku'].max():,.0f}円")
    print(f"三連単配当 中央値{df['sanrentan'].median():,.0f}円")
    print("保存: data/payouts.parquet")
