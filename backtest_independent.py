# -*- coding: utf-8 -*-
"""独立実力モデルの穴選出力をバックテスト(第88波)。
検証=valid_preproc(2023下半期・学習対象外)。独立モデルでスコア→レース内rank。
穴(7番人気↓)の中で、モデル上位が実際に3着内に来るか=妙味検出力を測る。"""
import pandas as pd, json, lightgbm as lgb

va = pd.read_parquet('data/training_cache/valid_preproc.parquet')
feats = json.load(open('data/lgbm_feature_cols.json'))
cats = [c for c in ['surface', 'venue', 'track_condition', 'distance_cat', 'sex'] if c in feats]
for c in cats:
    va[c] = va[c].astype('category')
m = lgb.Booster(model_file='data/lgbm_win_model.txt')
va['mscore'] = m.predict(va[feats])

# 人気・着順をtfjv_allから(date+horse_name)
df = pd.read_parquet('data/tfjv_all.parquet')
df['horse_name'] = df['horse_name'].astype(str).str.strip()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['pop'] = pd.to_numeric(df['popularity'], errors='coerce')
df['rank_n'] = pd.to_numeric(df['rank'], errors='coerce')
key = df.dropna(subset=['date']).drop_duplicates(['date', 'horse_name'])[['date', 'horse_name', 'pop', 'rank_n']]
va['horse_name'] = va['horse_name'].astype(str).str.strip()
va['date'] = pd.to_datetime(va['date'], errors='coerce')
v = va.merge(key, on=['date', 'horse_name'], how='left').dropna(subset=['pop', 'rank_n'])
v['in3'] = (v['rank_n'] <= 3).astype(int)
v['win'] = (v['rank_n'] == 1).astype(int)
# レース内モデルrank(pct, 1=最上位)
v['mrank_pct'] = v.groupby('race_key')['mscore'].rank(pct=True, ascending=True)  # 高scoreほど大

print(f'検証 {len(v)}頭 / {v["race_key"].nunique()}レース  全体3着内={v["in3"].mean():.3f}')

print('\n### 穴(7番人気↓)で 独立モデルのレース内順位別 3着内率 ###')
ana = v[v['pop'] >= 7].copy()
b = ana['in3'].mean()
print(f'穴ベース 3着内={b:.4f} 勝率={ana["win"].mean():.4f} (n={len(ana)})')
for lab, mask in [
    ('モデル上位20%(妙味◎)', ana['mrank_pct'] >= 0.8),
    ('モデル上位20-40%', (ana['mrank_pct'] >= 0.6) & (ana['mrank_pct'] < 0.8)),
    ('モデル中位40-60%', (ana['mrank_pct'] >= 0.4) & (ana['mrank_pct'] < 0.6)),
    ('モデル下位40%(消し)', ana['mrank_pct'] < 0.4),
]:
    mm = ana[mask]
    if len(mm):
        print(f'  {lab:20s} n={len(mm):5d} 3着内={mm["in3"].mean():.4f} (lift{mm["in3"].mean()/b:.2f}) 勝率={mm["win"].mean():.4f}')

print('\n### 比較: 本命側(1-5人気)でもモデル上位が効くか(役割分担の検証) ###')
hon = v[v['pop'] <= 5].copy()
b2 = hon['in3'].mean()
print(f'本命側ベース 3着内={b2:.4f} (n={len(hon)})')
for lab, mask in [
    ('モデル上位20%', hon['mrank_pct'] >= 0.8),
    ('モデル下位40%', hon['mrank_pct'] < 0.4),
]:
    mm = hon[mask]
    if len(mm):
        print(f'  {lab:20s} n={len(mm):5d} 3着内={mm["in3"].mean():.4f} (lift{mm["in3"].mean()/b2:.2f})')

# 単勝回収率(穴・モデル上位): 的中時オッズ概算は人気別近似でなく実勝敗のみ示す
print('\n### 穴×モデル上位20% の勝率を人気平均と比較(妙味=同人気でも勝てるか) ###')
top = ana[ana['mrank_pct'] >= 0.8]
print(f'  穴×モデル上位20%: 勝率{top["win"].mean():.4f} 平均人気{top["pop"].mean():.1f}')
print(f'  穴全体          : 勝率{ana["win"].mean():.4f} 平均人気{ana["pop"].mean():.1f}')
