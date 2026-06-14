"""
第40波: 会場・回り・コース適性モジュール

馬／騎手の「どの会場・どの回り・どの距離帯が得意か」を過去馬柱から自動算出。
TFJV データ（tfjv_all.parquet）に明示されない適性を、実績から推定する。

公開関数:
    horse_venue_aptitude(name) -> dict  馬の会場別・回り別 複勝率
    jockey_venue_aptitude(name) -> dict  騎手の会場別・回り別 複勝率
    build_aptitude_tag(name, venue, surface, distance) -> dict
        今走条件に対する適性タグ（"中山巧者" 等）と加点を返す
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

_DATA = Path(__file__).parent / "data"
_TFJV = _DATA / "tfjv_all.parquet"

# 左回り会場（東京・中京・新潟）/ それ以外は右回り
_LEFT = {"東京", "中京", "新潟"}

_df_cache: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        df = pd.read_parquet(_TFJV, columns=[
            "horse_name", "jockey", "venue", "surface", "distance",
            "rank", "popularity", "date", "race_id"])
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df["horse_name"] = df["horse_name"].astype(str).str.strip()
        df["jockey"] = df["jockey"].astype(str).str.strip()
        df["in3"] = (df["rank"] <= 3).astype(int)
        df["turn"] = df["venue"].apply(lambda v: "左" if v in _LEFT else "右")
        _df_cache = df
    return _df_cache


def _band(d) -> str:
    try:
        d = float(d)
    except Exception:
        return "?"
    if d <= 1400: return "短"
    if d <= 1800: return "マ"
    if d <= 2200: return "中"
    return "長"


def horse_venue_aptitude(name: str) -> dict:
    """馬の会場別・回り別・距離帯別 複勝率（n>=2 のみ）。"""
    df = _load()
    h = df[df["horse_name"] == str(name).strip()]
    if h.empty:
        return {}
    def agg(keycol):
        g = h.groupby(keycol).agg(n=("rank", "size"), in3=("in3", "sum"))
        return {k: (int(v["n"]), round(v["in3"]/v["n"], 3))
                for k, v in g.iterrows() if v["n"] >= 2}
    h = h.assign(band=h["distance"].apply(_band))
    return {
        "venue": agg("venue"),
        "turn":  agg("turn"),
        "band":  agg("band"),
        "total": (len(h), round(h["in3"].mean(), 3)),
    }


def jockey_venue_aptitude(name: str) -> dict:
    """騎手の会場別 複勝率（n>=10 のみ）。"""
    df = _load()
    j = df[df["jockey"] == str(name).strip()]
    if j.empty:
        return {}
    g = j.groupby("venue").agg(n=("rank", "size"), in3=("in3", "sum"))
    venue = {k: (int(v["n"]), round(v["in3"]/v["n"], 3))
             for k, v in g.iterrows() if v["n"] >= 10}
    gt = j.groupby(j["venue"].apply(lambda v: "左" if v in _LEFT else "右")).agg(
        n=("rank", "size"), in3=("in3", "sum"))
    turn = {k: (int(v["n"]), round(v["in3"]/v["n"], 3))
            for k, v in gt.iterrows() if v["n"] >= 10}
    return {"venue": venue, "turn": turn}


def build_aptitude_tag(name: str, venue: str, surface: str, distance) -> dict:
    """
    今走条件に対する馬の適性を判定。
    Returns: {"tags": [str], "bonus": float (-0.03〜+0.03), "detail": str}
    """
    apt = horse_venue_aptitude(name)
    if not apt:
        return {"tags": [], "bonus": 0.0, "detail": ""}
    tags, detail_parts = [], []
    bonus = 0.0
    total_n, total_rate = apt.get("total", (0, 0))

    # 会場適性: その会場で n>=2 かつ全体平均より明確に高い/低い
    v = apt.get("venue", {}).get(venue)
    if v:
        n, rate = v
        if n >= 2 and rate >= 0.5 and rate > total_rate + 0.1:
            tags.append(f"{venue}巧者")
            bonus += 0.025
            detail_parts.append(f"{venue}複勝{rate*100:.0f}%({n}走)")
        elif n >= 3 and rate == 0.0:
            tags.append(f"{venue}不振")
            bonus -= 0.02
            detail_parts.append(f"{venue}複勝0%({n}走)")

    # 回り適性
    cur_turn = "左" if venue in _LEFT else "右"
    t = apt.get("turn", {}).get(cur_turn)
    other = apt.get("turn", {}).get("右" if cur_turn == "左" else "左")
    if t and other:
        tn, trate = t; on, orate = other
        if tn >= 3 and on >= 3 and trate - orate >= 0.2:
            tags.append(f"{cur_turn}回り得意")
            bonus += 0.015
            detail_parts.append(f"{cur_turn}回り{trate*100:.0f}% vs 逆{orate*100:.0f}%")
        elif tn >= 3 and on >= 3 and orate - trate >= 0.25:
            tags.append(f"{cur_turn}回り苦手")
            bonus -= 0.015

    # 距離帯適性
    b = apt.get("band", {}).get(_band(distance))
    if b:
        n, rate = b
        if n >= 3 and rate >= 0.5 and rate > total_rate + 0.1:
            tags.append(f"{_band(distance)}距離得意")
            detail_parts.append(f"{_band(distance)}{rate*100:.0f}%")

    bonus = float(np.clip(bonus, -0.03, 0.03))
    return {"tags": tags, "bonus": round(bonus, 4), "detail": " / ".join(detail_parts)}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # コスモキュランダで実証
    for nm in ["コスモキュランダ", "マイユニバース"]:
        print(f"=== {nm} ===")
        apt = horse_venue_aptitude(nm)
        print("会場別:", apt.get("venue"))
        print("回り別:", apt.get("turn"))
        # 中山と東京で判定
        for v in ["中山", "東京", "阪神"]:
            tag = build_aptitude_tag(nm, v, "芝", 2200)
            print(f"  {v}2200m判定: {tag['tags']} bonus={tag['bonus']} ({tag['detail']})")


# ============================================================
# 第41波: クラスの壁判定 + 隠れ実力馬検出
# ============================================================

def _class_lv(rn) -> int:
    """race_name → 数値クラスレベル（1-8）"""
    s = str(rn)
    if "G1" in s or "Ｇ１" in s: return 8
    if "G2" in s or "Ｇ２" in s: return 7
    if "G3" in s or "Ｇ３" in s or "重賞" in s: return 6
    if "オープン" in s or "OPEN" in s or "3勝" in s or "１６００万" in s or "(L)" in s: return 5
    if "2勝" in s or "１０００万" in s: return 4
    if "1勝" in s or "５００万" in s: return 3
    if "未勝利" in s: return 2
    if "新馬" in s: return 1
    return 3


def build_class_wall_tag(name: str, current_class: int) -> dict:
    """
    馬の「クラスの壁」判定:
      - 過去最高クラスでの最良着順
      - 強い相手と当たって馬柱が汚れているだけの隠れ実力馬の検出
    Returns: {"tag": str, "bonus": float, "detail": str}
    """
    df = _load()
    h = df[df["horse_name"] == str(name).strip()].copy()
    if h.empty or len(h) < 3:
        return {"tag": "", "bonus": 0.0, "detail": ""}

    # 各過去走のクラスレベル（race_name から推定）
    # tfjv_all には race_name しか無いので _class_lv で推定
    h = h.sort_values("date", ascending=False)
    # 過去20走以内
    h = h.head(20).copy()
    # tfjv_all のロード時は race_name 含まれないので、ここで読み直し
    extra = pd.read_parquet(_TFJV, columns=["horse_name", "race_name", "date"])
    extra = extra[extra["horse_name"].astype(str).str.strip() == str(name).strip()]
    extra["date"] = pd.to_datetime(extra["date"], errors="coerce")
    extra = extra.set_index("date")["race_name"]
    h = h.assign(race_name=h["date"].map(extra))
    h["class_lv"] = h["race_name"].apply(_class_lv)

    # 過去最高クラス
    max_cl = int(h["class_lv"].max())
    # 過去最高クラスでの最良着順
    at_max = h[h["class_lv"] == max_cl]
    best_at_max = int(at_max["rank"].min()) if not at_max["rank"].isna().all() else 99

    tags = []
    bonus = 0.0
    detail_parts = []

    # 今走クラス
    cur = int(current_class) if current_class else 5

    # [84] クラスの壁判定
    if cur >= 7 and max_cl >= cur and best_at_max <= 3:
        # 今走と同等以上のクラスで3着内経験あり = クラスの器あり
        tags.append("クラスの器あり")
        bonus += 0.020
        detail_parts.append(f"過去同等クラスで{best_at_max}着実績")
    elif cur >= 7 and max_cl >= cur and best_at_max >= 8:
        # 今走相当クラスに出るが過去成績悪い = クラスの壁
        tags.append("クラスの壁")
        bonus -= 0.025
        detail_parts.append(f"上級クラスで{best_at_max}着大敗あり")
    elif cur >= 6 and max_cl < cur:
        # 今走で初の格上挑戦
        tags.append("格上初挑戦")
        # 中立（経験不足、人気にも織込）

    # [85] 隠れ実力馬の検出（2系統）
    recent = h.head(10).copy()
    if len(recent) >= 3 and "popularity" in recent.columns:
        pop_n = pd.to_numeric(recent["popularity"], errors="coerce")
        avg_pop = pop_n.mean()
        in3_count = (recent["rank"] <= 3).sum()
        # 系統1: 「人気薄続きだが実は実力で時々in3」
        if avg_pop >= 7 and in3_count >= 2:
            tags.append("隠れ実力馬")
            bonus += 0.015
            detail_parts.append(f"平均人気{avg_pop:.0f}番でin3が{in3_count}回")

    # [85深掘り] 系統2: 「展開で負けただけ」= 上がり最速級なのに人気以上に大敗
    # last_3f レース内順位を計算（全期間データから引く）
    try:
        rids = recent["race_id"].dropna().tolist()
        if rids:
            df_full = pd.read_parquet(_TFJV, columns=["race_id","horse_name","last_3f"])
            df_sub = df_full[df_full["race_id"].isin(rids)].copy()
            df_sub["l3f"] = pd.to_numeric(df_sub["last_3f"], errors="coerce")
            df_sub["l3f_rank"] = df_sub.groupby("race_id")["l3f"].rank(
                method="min", ascending=True, na_option="bottom")
            h_l3f = df_sub[df_sub["horse_name"].astype(str).str.strip() == str(name).strip()][["race_id","l3f_rank"]]
            recent2 = recent.merge(h_l3f, on="race_id", how="left")
            recent2["rank_n"] = pd.to_numeric(recent2["rank"], errors="coerce")
            recent2["pop_n"] = pd.to_numeric(recent2["popularity"], errors="coerce")
            unlucky = ((recent2["l3f_rank"]<=3) &
                       (recent2["rank_n"] >= recent2["pop_n"] + 5)).sum()
            if unlucky >= 2:
                tags.append("展開負け頻発")
                bonus += 0.020
                detail_parts.append(f"直近10走で上がり最速級でも人気以上大敗{int(unlucky)}回")
    except Exception:
        pass

    bonus = float(np.clip(bonus, -0.04, 0.04))
    return {"tag": " ".join(tags), "bonus": round(bonus, 4),
            "detail": " / ".join(detail_parts), "max_class": max_cl, "best_at_max": best_at_max}


if __name__ == "__main__":
    import sys as _s
    _s.stdout.reconfigure(encoding="utf-8", errors="replace")
    for nm in ["レガレイラ", "コスモキュランダ", "ミステリーウェイ",
               "シンエンペラー", "マイユニバース"]:
        r = build_class_wall_tag(nm, current_class=8)  # 宝塚=G1
        print(f"{nm} (G1出走想定): {r}")


# ============================================================
# 第41波: 騎手×距離 + 騎手×G1×人気帯マトリックス
# ============================================================

def jockey_distance_aptitude(jockey: str, distance: int) -> dict:
    """
    騎手の距離帯適性。「マイルの川田、長距離は微妙」等を判定。
    """
    df = _load()
    j = df[df["jockey"] == str(jockey).strip()].copy()
    if len(j) < 50:
        return {"tag": "", "bonus": 0.0, "detail": ""}
    j["band"] = j["distance"].apply(_band)
    band = _band(distance)
    same = j[j["band"] == band]
    other = j[j["band"] != band]
    if len(same) < 30 or len(other) < 30:
        return {"tag": "", "bonus": 0.0, "detail": ""}
    rate_same = same["in3"].mean()
    rate_other = other["in3"].mean()
    diff = rate_same - rate_other
    if diff >= 0.04:
        return {"tag": f"{band}距離得意騎手", "bonus": 0.015,
                "detail": f"{jockey} {band}距離 in3={rate_same*100:.0f}% vs 他{rate_other*100:.0f}%"}
    elif diff <= -0.04:
        return {"tag": f"{band}距離苦手騎手", "bonus": -0.020,
                "detail": f"{jockey} {band}距離 in3={rate_same*100:.0f}% vs 他{rate_other*100:.0f}%"}
    return {"tag": "", "bonus": 0.0, "detail": ""}


def jockey_grade_longshot_aptitude(jockey: str, race_name: str, popularity: int) -> dict:
    """
    重賞×人気薄での騎手成績。「G1人気薄の川田は警戒」等。
    重賞限定で人気6番以下での騎乗成績を集計。
    """
    if popularity < 6:
        return {"tag": "", "bonus": 0.0, "detail": ""}
    df = _load()
    j = df[df["jockey"] == str(jockey).strip()].copy()
    if j.empty:
        return {"tag": "", "bonus": 0.0, "detail": ""}
    # G1/G2/G3 限定で人気6+ をフィルタ
    # race_name は _load では取れていないので追加読込
    extra = pd.read_parquet(_TFJV, columns=["jockey", "race_name", "date"])
    extra["jockey"] = extra["jockey"].astype(str).str.strip()
    extra = extra[extra["jockey"] == str(jockey).strip()]
    extra["date"] = pd.to_datetime(extra["date"], errors="coerce")
    extra["is_grade"] = extra["race_name"].astype(str).str.contains(r"G[1-3]|Ｇ[１-３]|重賞", na=False, regex=True)
    # date型を揃えてマージ
    j["date"] = pd.to_datetime(j["date"], errors="coerce")
    j = j.merge(extra[["date", "is_grade"]].drop_duplicates("date"),
                on="date", how="left")
    j["popularity"] = pd.to_numeric(j["popularity"], errors="coerce")
    grade_long = j[(j["is_grade"] == True) & (j["popularity"] >= 6)]
    if len(grade_long) < 15:
        return {"tag": "", "bonus": 0.0, "detail": ""}
    in3_rate = grade_long["in3"].mean()
    # 全体ベースライン: 6人気以下の平均in3率は約15%
    if in3_rate >= 0.18:
        return {"tag": "重賞穴騎手", "bonus": 0.015,
                "detail": f"{jockey} 重賞×人気薄 in3={in3_rate*100:.0f}% (n={len(grade_long)})"}
    elif in3_rate <= 0.10:
        return {"tag": "重賞人気薄不振", "bonus": -0.010,
                "detail": f"{jockey} 重賞×人気薄 in3={in3_rate*100:.0f}% (n={len(grade_long)})"}
    return {"tag": "", "bonus": 0.0, "detail": f"{jockey} 重賞×人気薄 in3={in3_rate*100:.0f}%"}


if __name__ == "__main__":
    # 第41波の実証
    import sys as _s2
    _s2.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n=== 騎手×距離 ===")
    for jk in ["川田将雅", "横山典弘", "ルメール"]:
        for d in [1400, 1600, 2400, 3000]:
            r = jockey_distance_aptitude(jk, d)
            if r["tag"]:
                print(f"  {jk} {d}m: {r}")
    print("\n=== 騎手×重賞×人気薄 ===")
    for jk in ["川田将雅", "横山典弘", "菊沢一樹", "松若風馬", "丹内祐次"]:
        r = jockey_grade_longshot_aptitude(jk, "G1", 10)
        if r["detail"]:
            print(f"  {jk}: {r['detail']} → {r['tag']} bonus={r['bonus']}")


# ============================================================
# 第41波: C[91] 厩舎の重賞本気度
# ============================================================
_stable_grade_cache: pd.DataFrame | None = None

def _load_stable_grade() -> pd.DataFrame:
    global _stable_grade_cache
    if _stable_grade_cache is None:
        p = _DATA / "trainer_grade_stats.parquet"
        if p.exists():
            _stable_grade_cache = pd.read_parquet(p)
        else:
            _stable_grade_cache = pd.DataFrame()
    return _stable_grade_cache


def stable_grade_aptitude(trainer: str, race_name: str) -> dict:
    """
    厩舎の重賞in3率で「本気度」を推定（今走が重賞のときのみ発火）。
    全体平均18.5% → 25%以上で本気厩舎、12%以下で苦手厩舎。
    """
    if not any(g in str(race_name) for g in ["G1","G2","G3","Ｇ１","Ｇ２","Ｇ３","重賞"]):
        return {"tag":"", "bonus":0.0, "detail":""}
    df = _load_stable_grade()
    if df.empty:
        return {"tag":"", "bonus":0.0, "detail":""}
    row = df[df["trainer"]==str(trainer).strip()]
    if row.empty:
        return {"tag":"", "bonus":0.0, "detail":""}
    rate = float(row["grade_rate"].iloc[0])
    runs = int(row["grade_runs"].iloc[0])
    if rate >= 0.25 and runs >= 50:
        return {"tag":"重賞名門厩舎", "bonus":0.015,
                "detail":f"{trainer} 重賞in3={rate*100:.0f}%(n={runs})"}
    elif rate <= 0.12 and runs >= 30:
        return {"tag":"重賞苦手厩舎", "bonus":-0.010,
                "detail":f"{trainer} 重賞in3={rate*100:.0f}%(n={runs})"}
    return {"tag":"", "bonus":0.0, "detail":""}


def jockey_venue_distance_aptitude(jockey: str, venue: str, distance: int) -> dict:
    """
    第41波深掘り: 騎手×会場×距離。「東京1600のルメール」型の精細マトリックス。
    """
    df = _load()
    j = df[df["jockey"] == str(jockey).strip()].copy()
    if len(j) < 100:
        return {"tag": "", "bonus": 0.0, "detail": ""}
    j["band"] = j["distance"].apply(_band)
    band = _band(distance)
    same = j[(j["venue"] == venue) & (j["band"] == band)]
    other = j[(j["venue"] != venue) | (j["band"] != band)]
    if len(same) < 20:
        return {"tag": "", "bonus": 0.0, "detail": ""}
    rate_same = same["in3"].mean()
    rate_other = other["in3"].mean()
    diff = rate_same - rate_other
    if diff >= 0.05:
        return {"tag": f"{venue}{band}得意騎手", "bonus": 0.020,
                "detail": f"{jockey} {venue}{band} in3={rate_same*100:.0f}% vs 他{rate_other*100:.0f}% (n={len(same)})"}
    elif diff <= -0.05:
        return {"tag": f"{venue}{band}苦手騎手", "bonus": -0.020,
                "detail": f"{jockey} {venue}{band} in3={rate_same*100:.0f}% vs 他{rate_other*100:.0f}% (n={len(same)})"}
    return {"tag": "", "bonus": 0.0, "detail": ""}
