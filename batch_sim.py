# -*- coding: utf-8 -*-
"""11時以降の全候補レースを一括精査(ライブオッズ+実力+近走)。エア馬券シミュ用。"""
import sys
import pandas as pd
from netkeiba_odds import fetch_race_ids, fetch_win_odds
import race_brief as rb

DF = pd.read_parquet('data/tfjv_all.parquet')
DF['horse_name'] = DF['horse_name'].astype(str).str.strip()


def last3(h):
    m = DF[DF['horse_name'] == h].sort_values('date').tail(3)
    if m.empty:
        return '新馬/無'
    out = []
    for _, r in m.iterrows():
        rk = pd.to_numeric(r['rank'], errors='coerce')
        out.append(f"{str(r['race_name'])[:7]}{int(rk) if pd.notna(rk) else '?'}着")
    return ' '.join(out)


def analyze(v, n):
    ids = fetch_race_ids('20260614')
    rid = ids.get((v, n))
    od = fetch_win_odds(rid)
    if not od:
        print(f'### {v}{n}R 取得不可'); return
    res = rb.analyze_race(venue=v, race_no=n)
    df = res['eval_df']
    L = res['adj']
    nm = {int(x.get('horse_no', 0)): x['horse_name'] for _, x in df.iterrows() if pd.notna(x.get('horse_no'))}
    rank = {x['horse_name']: i + 1 for i, (_, x) in enumerate(df.iterrows())}
    pop = {p: no for no, (o, p) in od.items()}
    print(f'### {v}{n}R {res["race_class"]} {res["surface"]}{res["distance"]} {len(od)}頭 ###')
    # 本命側: ライブ上位4人気のうち実力上位
    print(' [本命候補]')
    for p in range(1, 5):
        no = pop.get(p)
        if not no:
            continue
        h = nm.get(no, '?')
        print(f'   {p}人 {no:2d} {h:11s} {od.get(no,(0,0))[0]:5.1f}倍 実力{rank.get(h,99):2d}位 | {last3(h)}')
    # 大穴: 7人気以下で複合◎/穴騎手/elite
    print(' [大穴シグナル(7人気↓)]')
    for no, (o, p) in sorted(od.items(), key=lambda x: x[1][1] or 99):
        if p and p >= 7:
            h = nm.get(no, '?')
            l = L.get(h, {})
            sig = []
            if '◎軸候補' in l.get('composite', ''):
                sig.append('複合◎' + (l['composite'].split('内')[1][:5] if '内' in l.get('composite', '') else ''))
            if l.get('janaba'):
                sig.append(l['janaba'][:12])
            if 'エリート' in l.get('elite', ''):
                sig.append('elite')
            if l.get('blood') and '◎' in l.get('blood', ''):
                sig.append('血◎')
            if sig:
                print(f'   {p:2d}人 {no:2d} {h:11s} {o:6.1f}倍 実力{rank.get(h,99):2d}位 {" ".join(sig)} | {last3(h)}')
    print()


if __name__ == '__main__':
    races = []
    for a in sys.argv[1:]:
        v, n = a.split(':')
        races.append((v, int(n)))
    for v, n in races:
        try:
            analyze(v, n)
        except Exception as e:
            print(f'### {v}{n}R ERROR {e}\n')
