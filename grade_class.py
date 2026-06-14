"""
格(クラス)credential モジュール — 第86波。

狙い: Eloはクラス(G1/G2/G3)を区別しない盲点がある。
  例: G3勝ちたてでEloは高いが、G1で通用する「格」は未検証の馬(スティンガーグラス型)。
そこで馬ごとに「過去にどのクラスで好走したか」というcredentialをリークなしで構築し、
それがEloと独立に3着内率を持ち上げるか(特に重賞・穴で)を実データ検証する。

credential は全て「当該レースより前の成績のみ」で構築(expanding+shift)＝リークなし。
"""
import pandas as pd
import numpy as np


def class_level(race_name) -> int:
    """race_name からクラスを8段階で返す。update_features._class_level準拠＋旧称対応。"""
    if pd.isna(race_name):
        return 3
    s = str(race_name)
    if "G1" in s:                                          return 8
    if "G2" in s:                                          return 7
    if "G3" in s or "重賞" in s:                          return 6
    if "オープン" in s or "OPEN" in s or "3勝" in s or "1600万" in s: return 5
    if "2勝" in s or "1000万" in s:                       return 4
    if "1勝" in s or "500万" in s:                        return 3
    if "未勝利" in s:                                      return 2
    if "新馬" in s:                                        return 1
    return 3


def build_credentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    tfjv_all を受け取り、各行(馬×レース)に「その時点までの格credential」を付与。
    全て shift(1) でリークなし。
    返す列:
      cls            … 当該レースのクラスlevel
      top_cls_seen   … 過去に走った最高クラス
      top_cls_in3    … 過去に3着内に入った最高クラス  ← 「格」の本体
      n_g1_in3       … 過去G1で3着内回数
      n_g12_in3      … 過去G1/G2で3着内回数
      n_graded_run   … 過去重賞出走数
      g1_in3         … 過去G1好走フラグ(0/1)
      g12_in3        … 過去G1orG2好走フラグ(0/1)
    """
    from horse_elo import _horse_key
    d = df.copy()
    d['horse_name'] = d['horse_name'].astype(str).str.strip()
    # 馬名衝突バグ対策: credential累積は horse_id 基準（別世代の同名馬を分離）
    d['hid'] = _horse_key(d)
    # レース内順序のため日付＋レースで安定ソート
    d['rk'] = d['date'].str.replace('-', '', regex=False) + '_' + d['venue'].astype(str) + '_' + d['race_no'].astype(str)
    d = d.sort_values(['hid', 'date', 'rk']).reset_index(drop=True)
    d['cls'] = d['race_name'].apply(class_level)
    d['in3'] = (pd.to_numeric(d['rank'], errors='coerce') <= 3).astype(float)

    g = d.groupby('hid', sort=False)
    # 当該レースを含めない累積(=shift)。cummax/cumsum を1つ前まで。
    cls_in3 = d['cls'].where(d['in3'] == 1, other=0)
    d['top_cls_seen'] = g['cls'].transform(lambda x: x.shift(1).cummax())
    d['top_cls_in3']  = cls_in3.groupby(d['hid'], sort=False).transform(lambda x: x.shift(1).cummax())
    d['n_graded_run'] = (d['cls'] >= 6).astype(int).groupby(d['hid'], sort=False).transform(lambda x: x.shift(1).cumsum())
    g1_in3  = ((d['cls'] == 8) & (d['in3'] == 1)).astype(int)
    g12_in3 = ((d['cls'] >= 7) & (d['in3'] == 1)).astype(int)
    d['n_g1_in3']  = g1_in3.groupby(d['hid'], sort=False).transform(lambda x: x.shift(1).cumsum())
    d['n_g12_in3'] = g12_in3.groupby(d['hid'], sort=False).transform(lambda x: x.shift(1).cumsum())
    for c in ['top_cls_seen', 'top_cls_in3', 'n_graded_run', 'n_g1_in3', 'n_g12_in3']:
        d[c] = d[c].fillna(0)
    d['g1_in3']  = (d['n_g1_in3']  >= 1).astype(int)
    d['g12_in3'] = (d['n_g12_in3'] >= 1).astype(int)
    return d


def _lift(sub, mask, label):
    base = sub['in3'].mean()
    m = sub[mask]
    if len(m) == 0:
        return None
    r = m['in3'].mean()
    return (label, len(m), round(r, 4), round(r / base, 3))


if __name__ == '__main__':
    df = pd.read_parquet('data/tfjv_all.parquet')
    d = build_credentials(df)

    # Elo結合(リークなしpre_elo)で実力コントロール
    try:
        elo = pd.read_parquet('data/horse_elo_pit.parquet')
        elo['horse_name'] = elo['horse_name'].astype(str).str.strip()
        d = d.merge(elo[['race_key', 'horse_name', 'pre_elo']],
                    left_on=['rk', 'horse_name'], right_on=['race_key', 'horse_name'], how='left')
    except Exception as e:
        print('elo merge skip', e)
        d['pre_elo'] = np.nan

    # 時系列分割: テスト=2025-2026(year>=25)。credential自体はリークなしだが念のため未来でも検証。
    d['yr'] = pd.to_numeric(d['year'], errors='coerce')
    test = d[d['yr'] >= 25].copy()

    print('=== 全体baseline 3着内率 ===', round(d['in3'].mean(), 4))
    print('=== test(25-26) baseline ===', round(test['in3'].mean(), 4), 'n=', len(test))

    # ---- 検証1: 重賞レース(G2/G1, cls>=7)での格credentialリフト ----
    print('\n### 検証1: 重賞(G1/G2)での「過去G1/G2好走」格リフト ###')
    gr = test[test['cls'] >= 7].copy()
    base = gr['in3'].mean()
    print(f'重賞ベース3着内 {base:.4f} (n={len(gr)})')
    for lab, mask in [
        ('過去G1好走あり(格◎)', gr['g1_in3'] == 1),
        ('過去G1好走なし',       gr['g1_in3'] == 0),
        ('過去G1/G2好走あり',    gr['g12_in3'] == 1),
        ('過去重賞好走なし(格未検証)', gr['top_cls_in3'] < 6),
    ]:
        m = gr[mask]
        if len(m):
            print(f'  {lab:24s} n={len(m):5d} 3着内={m["in3"].mean():.4f} lift={m["in3"].mean()/base:.3f}')

    # ---- 検証2: スティンガー型(高Elo×G1格なし)は買えるか ----
    print('\n### 検証2: 重賞で高Elo×格の有無 (スティンガー仮説) ###')
    grp = gr.dropna(subset=['pre_elo']).copy()
    # レース内Eloパーセンタイル
    grp['elo_pct'] = grp.groupby('rk')['pre_elo'].rank(pct=True)
    hi = grp[grp['elo_pct'] >= 0.7]  # レース内Elo上位3割
    b2 = hi['in3'].mean()
    print(f'重賞×Elo上位3割ベース {b2:.4f} (n={len(hi)})')
    for lab, mask in [
        ('高Elo×G1格あり', hi['g1_in3'] == 1),
        ('高Elo×G1格なし(スティンガー型)', hi['g1_in3'] == 0),
        ('高Elo×重賞好走経験ゼロ', hi['top_cls_in3'] < 6),
    ]:
        m = hi[mask]
        if len(m):
            print(f'  {lab:30s} n={len(m):5d} 3着内={m["in3"].mean():.4f} lift={m["in3"].mean()/b2:.3f}')

    # ---- 検証3: 穴(7番人気↓)×格 ----
    print('\n### 検証3: 重賞×穴(7番人気↓)での格リフト ###')
    ana = gr[pd.to_numeric(gr['popularity'], errors='coerce') >= 7].copy()
    b3 = ana['in3'].mean()
    print(f'重賞×穴ベース {b3:.4f} (n={len(ana)})')
    for lab, mask in [
        ('穴×過去G1/G2好走あり(格◎)', ana['g12_in3'] == 1),
        ('穴×過去重賞好走なし',        ana['top_cls_in3'] < 6),
    ]:
        m = ana[mask]
        if len(m):
            print(f'  {lab:24s} n={len(m):5d} 3着内={m["in3"].mean():.4f} lift={m["in3"].mean()/b3:.3f}')

    # ---- 検証4: クラス昇級(格上挑戦)の壁 ----
    print('\n### 検証4: 当該クラス > 過去最高好走クラス (格上初挑戦) ###')
    gr2 = test[test['cls'] >= 6].copy()
    b4 = gr2['in3'].mean()
    print(f'重賞(G3含)ベース {b4:.4f} (n={len(gr2)})')
    for lab, mask in [
        ('今回クラス>過去好走最高(格上挑戦)', gr2['cls'] > gr2['top_cls_in3']),
        ('今回クラス<=過去好走最高(格通用済)', gr2['cls'] <= gr2['top_cls_in3']),
    ]:
        m = gr2[mask]
        if len(m):
            print(f'  {lab:30s} n={len(m):6d} 3着内={m["in3"].mean():.4f} lift={m["in3"].mean()/b4:.3f}')


# ============================================================
# race_brief 配線用: 馬ごとの格credentialルックアップ(キャッシュ生成)
# ============================================================
def build_credential_lookup(save=True):
    """全馬の現時点キャリアcredentialを集計し data/grade_credential.parquet に保存。
    今日の出走馬は今日のレースがtfjv_allに未収録なので、全履歴のmax/sumがそのまま
    『今日より前のcredential』になる(リークなし)。"""
    from horse_elo import _horse_key
    df = pd.read_parquet('data/tfjv_all.parquet')
    df = df.copy()
    df['horse_name'] = df['horse_name'].astype(str).str.strip()
    # 馬名衝突バグ対策: 集計は horse_id 基準（別世代の同名馬を分離）
    df['hid'] = _horse_key(df)
    df['cls'] = df['race_name'].apply(class_level)
    df['in3'] = (pd.to_numeric(df['rank'], errors='coerce') <= 3).astype(int)
    df['_d'] = pd.to_datetime(df['date'], errors='coerce')
    cls_in3 = df['cls'].where(df['in3'] == 1, other=0)
    out = pd.DataFrame({
        'top_cls_seen': df.groupby('hid', sort=False)['cls'].max(),
        'top_cls_in3': cls_in3.groupby(df['hid'], sort=False).max(),
        'n_graded_run': (df['cls'] >= 6).groupby(df['hid'], sort=False).sum(),
        'n_g1_in3': ((df['cls'] == 8) & (df['in3'] == 1)).groupby(df['hid'], sort=False).sum(),
        'n_g12_in3': ((df['cls'] >= 7) & (df['in3'] == 1)).groupby(df['hid'], sort=False).sum(),
    }).reset_index()
    # hid→馬名/最終出走日（grade_tagの名前フォールバック用に現役世代を判定）
    meta = df.groupby('hid', sort=False).agg(horse_name=('horse_name', 'last'),
                                             last_race_date=('_d', 'max')).reset_index()
    out = out.merge(meta, on='hid', how='left').rename(columns={'hid': 'horse_id'})
    if save:
        out.to_parquet('data/grade_credential.parquet')
    return out


def grade_tag(horse_name, today_cls, cred_df, horse_id=None):
    """人気馬〜中位の重賞用 格タグ。穴(7番人気↓)では呼ばないこと(穴では格無効)。
    返す: タグ文字列(該当なし時は '')。
    馬名衝突対策: horse_id があれば厳密一致。無ければ同名のうち現役(直近出走)を採用。"""
    row = None
    if horse_id is not None and 'horse_id' in cred_df.columns:
        row = cred_df[cred_df['horse_id'].astype(str) == str(horse_id).strip()]
    if row is None or row.empty:
        row = cred_df[cred_df['horse_name'] == str(horse_name).strip()]
        if len(row) > 1 and 'last_race_date' in row.columns:
            row = row.sort_values('last_race_date').tail(1)
    if row.empty:
        return ''
    r = row.iloc[0]
    tci = int(r['top_cls_in3']); g1 = int(r['n_g1_in3']); g12 = int(r['n_g12_in3'])
    name = {8: 'G1', 7: 'G2', 6: 'G3/重賞'}.get(tci, '条件戦')
    if today_cls <= tci and tci >= 6:
        if g1 >= 1:
            return f'格◎G1好走{g1}回'
        return f'格◎{name}通用済'
    if today_cls > tci:
        if tci >= 6:
            return f'格△{name}実績→格上挑戦(壁0.86x・トップ/海外Jなら1.33x補える)'
        return f'格▲重賞未経験で格上初挑戦(0.86x割引)'
    return ''


# ============================================================
# race_brief 配線用: 前走騎手ルックアップ + 乗替→トップ/非トップ判定(検証済1.26x/0.71x)
# ============================================================
_TOP_JK = {"C.ルメール","ルメール","川田将雅","横山武史","戸崎圭太","松山弘平","M.デムーロ",
           "モレイラ","レーン","武豊","坂井瑠星","横山和生","西村淳也","岩田望来","ムーア",
           "マーフィー","ビュイック","北村友一","浜中俊","池添謙一","レーヴ","Ｃ．デム","C.デム"}
_FOREIGN_JK = {"ルメール","モレイラ","レーン","ムーア","マーフィー","ビュイック","デム","デムーロ",
               "マクドナルド","スミヨン","バルザローナ","ボウマン","マーカンド","キング"}


def _is_top(n):
    n = str(n)
    return any(t in n or n in t for t in _TOP_JK)


def build_prev_jockey_lookup(save=True):
    """全馬の最新騎手(=今日から見た前走騎手)を集計し data/prev_jockey.parquet に保存。"""
    df = pd.read_parquet('data/tfjv_all.parquet').copy()
    df['horse_name'] = df['horse_name'].astype(str).str.strip()
    df = df.sort_values(['horse_name', 'date'])
    last = df.groupby('horse_name').tail(1)[['horse_name', 'jockey']]
    last = last.rename(columns={'jockey': 'prev_jockey'})
    if save:
        last.to_parquet('data/prev_jockey.parquet')
    return last


def jockey_change_tag(horse_name, current_jockey, prev_df):
    """人気馬〜中位の重賞用 乗替タグ。穴では無効なので呼ばないこと。
    →トップ騎手1.26x / →非トップ0.71x / 海外Jは格上補正(検証B 1.33x)。"""
    row = prev_df[prev_df['horse_name'] == str(horse_name).strip()]
    if row.empty:
        return ''
    prev = str(row.iloc[0]['prev_jockey']).strip()
    cur = str(current_jockey).strip()
    if not prev or not cur or prev == cur:
        return ''  # 継続 or 不明
    cur_top = _is_top(cur)
    cur_foreign = any(f in cur for f in _FOREIGN_JK)
    prev_top = _is_top(prev)
    # 検証済(重賞25-26)実測lift: prev/curの組合せで出し分け
    if cur_foreign and prev_top:
        return f'乗替トップ→海外トップ{prev}→{cur}(◎◎2.03x・勝負気配)'
    if cur_foreign:
        return f'乗替→海外トップ{prev}→{cur}(◎1.47x・格上補える)'
    if cur_top and prev_top:
        return f'乗替トップ→トップ{prev}→{cur}(◎横1.31x)'
    if cur_top:
        return f'乗替格下→トップ{prev}→{cur}(◎強化1.21x)'
    if prev_top and not cur_top:
        return f'乗替トップ→格下{prev}→{cur}(▲差し戻し0.80x)'
    return f'乗替→非トップ{prev}→{cur}(▲0.71x)'


# ============================================================
# 馬×騎手 個別コンビ実績(集計liftより優先する個別相性) — ユーザー指摘で追加
# ============================================================
def build_horse_jockey_combo(save=True):
    """(馬,騎手)ごとの過去コンビ成績を集計し data/horse_jockey_combo.parquet に保存。"""
    df = pd.read_parquet('data/tfjv_all.parquet').copy()
    df['horse_name'] = df['horse_name'].astype(str).str.strip()
    df['jk'] = df['jockey'].astype(str).str.strip()
    df['rank_n'] = pd.to_numeric(df['rank'], errors='coerce')
    df['in3'] = (df['rank_n'].between(1, 3)).astype(int)
    df['rank_pos'] = df['rank_n'].where(df['rank_n'] >= 1)  # 0(取消/除外)を最高着順から除く
    g = df.groupby(['horse_name', 'jk'])
    out = g.agg(n=('rank_pos', 'count'), best=('rank_pos', 'min'),
                n_in3=('in3', 'sum')).reset_index()
    out = out[out['n'] >= 1]
    if save:
        out.to_parquet('data/horse_jockey_combo.parquet')
    return out


def combo_tag(horse_name, jockey, combo_df):
    """今日の鞍上がこの馬に過去騎乗していれば、その個別相性を返す(集計liftより優先)。
    新コンビ(過去騎乗なし)なら'' を返し、呼び出し側で集計lift(jockey_change_tag)にフォールバック。"""
    r = combo_df[(combo_df['horse_name'] == str(horse_name).strip()) &
                 (combo_df['jk'] == str(jockey).strip())]
    if r.empty:
        return ''
    n = int(r.iloc[0]['n']); best = int(r.iloc[0]['best']); ni3 = int(r.iloc[0]['n_in3'])
    if best <= 2 or ni3 >= max(1, n - 1):
        return f'同コンビ◎好相性(過去{n}回 best{best}着/3着内{ni3})'
    if best >= 4 and ni3 == 0:
        return f'⚠同コンビ不振(過去{n}回 best{best}着・3着内0)=集計lift過信注意'
    return f'同コンビ△(過去{n}回 best{best}着/3着内{ni3})'
