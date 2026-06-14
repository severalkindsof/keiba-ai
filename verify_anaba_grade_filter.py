# -*- coding: utf-8 -*-
"""穴馬×格 を「選別フィルター」として再検証 — 第86波(ユーザー本丸)。

ユーザー思想: 格の本丸は穴馬の選別。
  (1)根拠ある穴=初挑戦でも過去に重賞/上位クラスで好走した実力裏付けがある → 買える穴
  (2)危ない穴=スティンガー型(格上挑戦かつ重賞好走経験ゼロ、Eloだけ高い) → 消すフィルター
平均リフトでなく「格なし穴を除外すると残った穴プールの精度が上がるか(フィルター効果)」を見る。
リークなしcredential(grade_class)+pre_elo。テスト25-26。重賞(G3含cls>=6)×7番人気↓。
"""
import pandas as pd
import numpy as np
from grade_class import build_credentials

df = pd.read_parquet('data/tfjv_all.parquet')
d = build_credentials(df)
elo = pd.read_parquet('data/horse_elo_pit.parquet')
elo['horse_name'] = elo['horse_name'].astype(str).str.strip()
d = d.merge(elo[['race_key', 'horse_name', 'pre_elo']],
            left_on=['rk', 'horse_name'], right_on=['race_key', 'horse_name'], how='left')
d['yr'] = pd.to_numeric(d['year'], errors='coerce')
d['pop'] = pd.to_numeric(d['popularity'], errors='coerce')

# 重賞×穴×テスト期間
ana = d[(d['yr'] >= 25) & (d['cls'] >= 6) & (d['pop'] >= 7)].copy()
ana['elo_pct'] = ana.groupby('rk')['pre_elo'].rank(pct=True)  # レース内Elo順位
base = ana['in3'].mean()
N = len(ana)
print(f'重賞×穴(7番人気↓) ベース3着内 {base:.4f} (n={N})')

# 格の定義
ana['格あり'] = ana['top_cls_in3'] >= 6          # 過去に重賞(G3以上)で3着内した実力裏付け
ana['G1G2格'] = ana['g12_in3'] == 1               # 過去G1/G2で好走
ana['格上挑戦'] = ana['cls'] > ana['top_cls_in3']  # 今回が過去好走最高クラス超え
# スティンガー型(消し対象)=格上挑戦 かつ 重賞好走経験ゼロ かつ レース内Elo上位(市場過剰評価)
ana['危険穴'] = ana['格上挑戦'] & (ana['top_cls_in3'] < 6) & (ana['elo_pct'] >= 0.6)


def line(mask, lab):
    m = ana[mask]
    if len(m) == 0:
        print(f'  {lab:32s} n=0'); return
    r = m['in3'].mean()
    print(f'  {lab:32s} n={len(m):4d} 3着内={r:.4f} lift={r/base:.3f}')


print('\n### 穴の中の格による選別 ###')
line(ana['格あり'], '買える穴=過去重賞好走あり')
line(ana['G1G2格'], '買える穴=過去G1/G2好走')
line(~ana['格あり'], '危ない穴=重賞好走経験ゼロ')
line(ana['危険穴'], 'スティンガー型(格上×経験ゼロ×高Elo)')
line(ana['危険穴'] == False, '└ スティンガー型を除外した残り穴プール')

print('\n### フィルター効果(危ない穴をどれだけ消せるか) ###')
removed = ana[~ana['格あり']]
kept = ana[ana['格あり']]
print(f'  「重賞好走経験ゼロ」を全消し: {len(removed)}頭除外({len(removed)/N:.0%})')
print(f'    除外群の3着内 {removed["in3"].mean():.4f} / 残存群 {kept["in3"].mean():.4f} '
      f'(残存/全体 lift={kept["in3"].mean()/base:.3f})')
print(f'  スティンガー型のみ消し: {ana["危険穴"].sum()}頭除外({ana["危険穴"].mean():.0%})')
st = ana[ana['危険穴']]; nost = ana[~ana['危険穴']]
print(f'    除外群の3着内 {st["in3"].mean():.4f} / 残存群 {nost["in3"].mean():.4f} '
      f'(残存/全体 lift={nost["in3"].mean()/base:.3f})')

# 配当を絡めた粗い回収率(複勝配当データ無→人気別複勝率の理論オッズ近似で参考値)
print('\n### 参考: 穴プールの質(3着内率の素点) ###')
print(f'  穴全体 {base:.4f} → 経験ゼロ消し後 {kept["in3"].mean():.4f} '
      f'(+{(kept["in3"].mean()-base)*100:.1f}pt)')
