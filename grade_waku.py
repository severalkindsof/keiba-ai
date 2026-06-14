# -*- coding: utf-8 -*-
"""重賞の枠ゾーン実測lift — 第86波。
race_briefの一般draw_bias(内有利前提)は重賞コース特性と逆になる(宝塚=外枠勝率1.82x)。
重賞は過去実測の枠ゾーン(内/中/外)lift を使う(鉄則: 重賞は過去枠順分析)。
2段ルックアップ: ①重賞名単体(母数十分なら最優先) ②会場×surface×距離帯(重賞のみ・母数補強)。
field_size列は壊れているのでrk(日付_会場_R)のcountを真の頭数とする。"""
import pandas as pd


def _zone(hno, fs):
    if fs <= 0 or pd.isna(hno):
        return None
    if hno <= fs / 3:
        return '内'
    if hno >= fs * 2 / 3:
        return '外'
    return '中'


def build(save=True):
    df = pd.read_parquet('data/tfjv_all.parquet')
    df = df.copy()
    df['hno'] = pd.to_numeric(df['horse_no'], errors='coerce')
    df['rk'] = df['date'].str.replace('-', '', regex=False) + '_' + df['venue'].astype(str) + '_' + df['race_no'].astype(str)
    df['fs'] = df.groupby('rk')['hno'].transform('count')
    df['rank_n'] = pd.to_numeric(df['rank'], errors='coerce')
    df['win'] = (df['rank_n'] == 1).astype(float)
    df['in3'] = (df['rank_n'] <= 3).astype(float)
    df['dist'] = pd.to_numeric(df['distance'], errors='coerce')
    df['distband'] = (df['dist'] // 200 * 200).astype('Int64')
    df['zone'] = df.apply(lambda r: _zone(r['hno'], r['fs']), axis=1)
    df = df.dropna(subset=['zone'])
    is_graded = df['race_name'].astype(str).str.contains('G1|G2|G3|重賞')
    g = df[is_graded].copy()

    def agg(keys):
        base = g.groupby(keys).agg(n_races=('rk', 'nunique'),
                                   win=('win', 'mean'), in3=('in3', 'mean')).reset_index()
        z = g.groupby(keys + ['zone']).agg(n=('win', 'size'),
                                           zwin=('win', 'mean'), zin3=('in3', 'mean')).reset_index()
        m = z.merge(base, on=keys)
        m['win_lift'] = m['zwin'] / m['win']
        m['in3_lift'] = m['zin3'] / m['in3']
        return m

    by_race = agg(['race_name'])
    by_course = agg(['venue', 'surface', 'distband'])
    if save:
        by_race.to_parquet('data/grade_waku_race.parquet')
        by_course.to_parquet('data/grade_waku_course.parquet')
    return by_race, by_course


_CACHE = {}


def _load():
    if not _CACHE:
        try:
            _CACHE['race'] = pd.read_parquet('data/grade_waku_race.parquet')
            _CACHE['course'] = pd.read_parquet('data/grade_waku_course.parquet')
        except Exception:
            _CACHE['race'] = _CACHE['course'] = None
    return _CACHE.get('race'), _CACHE.get('course')


def waku_tag(race_name, venue, surface, distance, hno, fs, min_races=12):
    """重賞の枠ゾーンliftタグを返す。母数不足なら会場×距離帯にフォールバック。該当なし''。"""
    zone = _zone(hno, fs)
    if zone is None:
        return ''
    br, bc = _load()
    src = None
    if br is not None:
        r = br[(br['race_name'].astype(str) == str(race_name)) & (br['zone'] == zone)]
        if not r.empty and int(r.iloc[0]['n_races']) >= min_races:
            src = r.iloc[0]
    if src is None and bc is not None:
        distband = int(distance // 200 * 200) if distance else None
        c = bc[(bc['venue'] == venue) & (bc['surface'] == surface) &
               (bc['distband'] == distband) & (bc['zone'] == zone)]
        if not c.empty and int(c.iloc[0]['n_races']) >= min_races:
            src = c.iloc[0]
    if src is None:
        return ''
    wl = float(src['win_lift'])
    if wl >= 1.25:
        return f'◎{zone}枠(重賞勝率{wl:.2f}x)'
    if wl <= 0.7:
        return f'▲{zone}枠(重賞勝率{wl:.2f}x)'
    return ''


if __name__ == '__main__':
    br, bc = build()
    print('race rows', len(br), 'course rows', len(bc))
    tk = br[br['race_name'].astype(str) == '宝塚記念G1'].sort_values('zone')
    print(tk[['race_name', 'zone', 'n_races', 'win_lift', 'in3_lift']].to_string())
    print('\n宝塚枠タグ例(18頭):')
    for hno in [1, 9, 17]:
        print(f'  馬番{hno}: {waku_tag("宝塚記念G1","阪神","芝",2200,hno,18)}')


# ============================================================
# 重賞の過去脚質傾向(前残り/差し)テーブル + ヘッダ用文言
# ============================================================
def build_style(save=True):
    """重賞名ごとに過去の1〜3着馬の脚質(最終コーナー位置)傾向を集計。"""
    df = pd.read_parquet('data/tfjv_all.parquet').copy()
    df['hno'] = pd.to_numeric(df['horse_no'], errors='coerce')
    df['rk'] = df['date'].str.replace('-', '', regex=False) + '_' + df['venue'].astype(str) + '_' + df['race_no'].astype(str)
    df['fs'] = df.groupby('rk')['hno'].transform('count')
    df['rank_n'] = pd.to_numeric(df['rank'], errors='coerce')
    df['c4'] = pd.to_numeric(df['corner4'], errors='coerce')
    w = df[(df['rank_n'] <= 3) & df['race_name'].astype(str).str.contains('G1|G2|G3')].copy()
    w['c4rel'] = w['c4'] / w['fs']
    w['st'] = w['c4rel'].apply(lambda x: '逃先' if x <= 0.35 else ('差追' if x >= 0.65 else '中団') if pd.notna(x) else None)
    w = w.dropna(subset=['st'])
    win = w[w['rank_n'] == 1]
    rows = []
    for rn, sub in w.groupby('race_name'):
        wsub = win[win['race_name'] == rn]
        n_races = sub['rk'].nunique()
        vc = sub['st'].value_counts()
        wvc = wsub['st'].value_counts()
        rows.append({'race_name': rn, 'n_races': n_races,
                     'in3_逃先': vc.get('逃先', 0), 'in3_中団': vc.get('中団', 0), 'in3_差追': vc.get('差追', 0),
                     'win_逃先': wvc.get('逃先', 0), 'win_中団': wvc.get('中団', 0), 'win_差追': wvc.get('差追', 0)})
    out = pd.DataFrame(rows)
    if save:
        out.to_parquet('data/grade_style_race.parquet')
    return out


def style_tendency(race_name, min_races=10):
    """重賞ヘッダ用: 過去の勝ち脚質傾向の一文。該当なし''。"""
    try:
        t = pd.read_parquet('data/grade_style_race.parquet')
    except Exception:
        return ''
    r = t[t['race_name'].astype(str) == str(race_name)]
    if r.empty or int(r.iloc[0]['n_races']) < min_races:
        return ''
    r = r.iloc[0]
    wf, wm, wc = int(r['win_逃先']), int(r['win_中団']), int(r['win_差追'])
    if_, im, ic = int(r['in3_逃先']), int(r['in3_中団']), int(r['in3_差追'])
    tot = wf + wm + wc
    lead = '逃げ先行' if wf >= max(wm, wc) else '差し追込' if wc >= max(wf, wm) else '中団'
    return (f"過去{int(r['n_races'])}年 勝ち脚質[逃先{wf}/中団{wm}/差追{wc}]→{lead}優勢 "
            f"(3着内 逃先{if_}/中団{im}/差追{ic})")
