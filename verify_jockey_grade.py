"""
宿題2: 乗り替わり×格 の実データ検証 — 第86波。

前提(handoff): 乗り替わり単独は穴に無効と実測済。
ユーザー要求: 「格」と組み合わせた評価。スティンガー(ルメール→岩田望来)・
  ビザンチン(C.デム→西村)型。乗り替わりが格(重賞/格上挑戦)の文脈で意味を持つか。
全てリークなし(prev_jockeyは過去走shift, credentialもshift)。テスト=25-26。
"""
import pandas as pd
import numpy as np
from grade_class import build_credentials

TOP = {"C.ルメール","ルメール","川田将雅","横山武史","戸崎圭太","松山弘平",
       "M.デムーロ","モレイラ","レーン","武豊","坂井瑠星","横山和生","西村淳也",
       "岩田望来","ムーア","マーフィー","ビュイック","北村友一","浜中俊","池添謙一","ダ"}

def is_top(n):
    n = str(n)
    return any(t in n or n in t for t in TOP)

df = pd.read_parquet('data/tfjv_all.parquet')
d = build_credentials(df)  # sorted by horse,date; has cls,in3,top_cls_in3,etc, rk
d['jk'] = d['jockey'].astype(str).str.strip()
g = d.groupby('horse_name', sort=False)
d['prev_jk'] = g['jk'].shift(1)
d['prev_in3'] = g['in3'].shift(1)
d['changed'] = (d['prev_jk'].notna()) & (d['jk'] != d['prev_jk'])
d['to_top'] = d['jk'].apply(is_top)
d['from_top'] = d['prev_jk'].fillna('').apply(is_top)
d['yr'] = pd.to_numeric(d['year'], errors='coerce')

test = d[(d['yr'] >= 25) & (d['prev_jk'].notna())].copy()

def line(sub, base, lab):
    if len(sub) == 0: return
    r = sub['in3'].mean()
    print(f'  {lab:34s} n={len(sub):5d} 3着内={r:.4f} lift={r/base:.3f}')

# ---- A: 重賞(G3含, cls>=6)での乗り替わり ----
print('### A: 重賞での乗り替わり単独/格組合せ ###')
gr = test[test['cls'] >= 6].copy()
b = gr['in3'].mean()
print(f'重賞ベース {b:.4f} (n={len(gr)})')
line(gr[~gr['changed']], b, '継続騎乗')
line(gr[gr['changed']], b, '乗り替わり(全)')
line(gr[gr['changed'] & gr['to_top']], b, '乗り替わり→トップ騎手')
line(gr[gr['changed'] & ~gr['to_top']], b, '乗り替わり→非トップ')
line(gr[gr['changed'] & gr['from_top'] & ~gr['to_top']], b, 'トップ→格下(差し戻し)')

# ---- B: 格上初挑戦 × 乗り替わり (スティンガー/ビザンチン型) ----
print('\n### B: 格上初挑戦(今回cls>過去好走最高)×乗り替わり ###')
up = gr[gr['cls'] > gr['top_cls_in3']].copy()
b2 = up['in3'].mean()
print(f'格上挑戦ベース {b2:.4f} (n={len(up)})')
line(up[~up['changed']], b2, '格上×継続')
line(up[up['changed'] & up['to_top']], b2, '格上×乗替→トップ騎手')
line(up[up['changed'] & ~up['to_top']], b2, '格上×乗替→非トップ')

# ---- C: 格通用済 × 乗り替わり ----
print('\n### C: 格通用済(今回cls<=過去好走最高)×乗り替わり ###')
ok = gr[gr['cls'] <= gr['top_cls_in3']].copy()
b3 = ok['in3'].mean()
print(f'格通用済ベース {b3:.4f} (n={len(ok)})')
line(ok[~ok['changed']], b3, '格通用×継続')
line(ok[ok['changed'] & ok['to_top']], b3, '格通用×乗替→トップ騎手')
line(ok[ok['changed'] & ~ok['to_top']], b3, '格通用×乗替→非トップ')

# ---- D: 穴(7番人気↓)重賞 × 乗り替わり×格 ----
print('\n### D: 重賞×穴(7番人気↓)×乗り替わり×格 ###')
ana = gr[pd.to_numeric(gr['popularity'], errors='coerce') >= 7].copy()
b4 = ana['in3'].mean()
print(f'重賞×穴ベース {b4:.4f} (n={len(ana)})')
line(ana[ana['changed'] & ana['to_top']], b4, '穴×乗替→トップ騎手')
line(ana[ana['g12_in3'] == 1], b4, '穴×過去G1/G2好走あり(格◎)')
line(ana[(ana['g12_in3'] == 1) & ana['changed'] & ana['to_top']], b4, '穴×格◎×乗替→トップ')
