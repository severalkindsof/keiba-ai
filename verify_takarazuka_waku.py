# -*- coding: utf-8 -*-
"""宝塚記念 枠別再検証(第86波・宙づりバグ修正)。
旧バグ=field_size/horse_noが文字列のまま除算→全頭が内に分類。int化＋宝塚記念G1のみで再集計。
枠ゾーン: 馬番/頭数 で内(<=1/3)/中/外(>=2/3)。勝率・3着内率を集計。
比較対照に阪神芝2000-2200の重賞(母数補強)も出す。"""
import pandas as pd

df = pd.read_parquet('data/tfjv_all.parquet')
df['horse_name'] = df['horse_name'].astype(str).str.strip()
df['hno'] = pd.to_numeric(df['horse_no'], errors='coerce')
# field_size列は壊れている(全行ゴミ)。rk(日付_会場_R)ごとの出走頭数を真の頭数とする
df['rk'] = df['date'].str.replace('-', '', regex=False) + '_' + df['venue'].astype(str) + '_' + df['race_no'].astype(str)
df['fs'] = df.groupby('rk')['hno'].transform('count')
df['rank_n'] = pd.to_numeric(df['rank'], errors='coerce')
df['in3'] = (df['rank_n'] <= 3).astype(float)
df['win'] = (df['rank_n'] == 1).astype(float)
df = df.dropna(subset=['hno', 'fs'])
df['zone'] = df.apply(lambda r: '内' if r['hno'] <= r['fs']/3 else ('外' if r['hno'] >= r['fs']*2/3 else '中'), axis=1)


def zone_report(sub, title):
    print(f'\n=== {title} (n={len(sub)}, レース{sub["date"].nunique()}回) ===')
    base_w, base_p = sub['win'].mean(), sub['in3'].mean()
    print(f'  全体: 勝率{base_w:.3f} 3着内{base_p:.3f}')
    for z in ['内', '中', '外']:
        m = sub[sub['zone'] == z]
        if len(m):
            print(f'  {z}枠 n={len(m):4d} 勝率{m["win"].mean():.3f}({m["win"].mean()/base_w:.2f}x) '
                  f'3着内{m["in3"].mean():.3f}({m["in3"].mean()/base_p:.2f}x)')


# 宝塚記念G1のみ
tk = df[df['race_name'].astype(str) == '宝塚記念G1']
zone_report(tk, '宝塚記念G1 2010-2025')

# 対照: 阪神芝2000-2200の重賞(G1/G2/G3)で母数補強
hanshin = df[(df['venue'] == '阪神') & (df['surface'] == '芝') &
             (pd.to_numeric(df['distance'], errors='coerce').between(2000, 2200)) &
             (df['race_name'].astype(str).str.contains('G1|G2|G3'))]
zone_report(hanshin, '阪神芝2000-2200 重賞(母数補強)')

# 宝塚の勝ち馬・3着内馬の実馬番リスト(目視確認用)
w = tk[tk['rank_n'] <= 3].sort_values('date')
print('\n宝塚 3着内馬の[年/着/馬番/頭数/zone]:')
for _, r in w.iterrows():
    print(f'  20{r["year"]} {int(r["rank_n"])}着 馬番{int(r["hno"]):>2}/{int(r["fs"])} {r["zone"]} {r["horse_name"]}')
