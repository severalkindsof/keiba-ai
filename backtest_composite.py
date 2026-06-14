# -*- coding: utf-8 -*-
"""
backtest_composite.py … 穴馬の複合スコアの予測力を時系列分割で検証する。

要素(全て乗法で掛け合わせ＝網目をくぐる。1つでも低いと落ちる):
  1. 実力     : レース内のpre_elo(リークなしElo)パーセンタイルの学習期間リフト
  2. 脚質     : 逃先/中団/差追
  3. ジョッキー: 騎手の大穴複勝リフト
  4. 血統     : 父系 × 母父系
  5. 馬格     : 馬体重ビン

鉄則: 直感スコア禁止。リフトは学習期間(year<=24)のみで作りテスト期間(year>=25)で検証(リークなし)。
合格基準: スコア上位分位ほど大穴複勝率が単調に上がり、最上位分位がベースを明確に上回ること。
"""
import numpy as np
import pandas as pd

ANA_POP = 10
MIN_J = 40


def _add_style(d):
    for c in ("field_size", "corner4", "corner3", "corner2"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    pos = d["corner4"].where(d["corner4"] > 0)
    pos = pos.fillna(d["corner3"].where(d["corner3"] > 0))
    pos = pos.fillna(d["corner2"].where(d["corner2"] > 0))
    r = pos / d["field_size"]
    d["style"] = np.select([r <= 0.35, r >= 0.65], ["逃先", "差追"], default="中団")
    d.loc[pos.isna() | (d["field_size"] <= 0), "style"] = "?"
    return d


def _weight_bin(w):
    w = pd.to_numeric(w, errors="coerce")
    return pd.cut(w, [0, 430, 460, 490, 520, 9999],
                  labels=["~430", "430-460", "460-490", "490-520", "520+"])


def main():
    df = pd.read_parquet("data/tfjv_all.parquet")
    for c in ("popularity", "rank", "year", "month", "day", "race_no", "last_3f"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["popularity", "rank", "year", "horse_name"])
    df = df[~df["race_name"].astype(str).str.contains("障害", na=False)]
    # race_key を組んで pre_elo をマージ
    yyyymmdd = ((2000 + df["year"].astype(int)).astype(str)
                + df["month"].astype(int).astype(str).str.zfill(2)
                + df["day"].astype(int).astype(str).str.zfill(2))
    df["rk"] = yyyymmdd + "_" + df["venue"].astype(str) + "_" + df["race_no"].astype(int).astype(str).str.zfill(2)
    # tfjv_allのhorse_nameは全角スペースでパディングされているためstripしてマージ（24%→100%）
    df["hn"] = df["horse_name"].astype(str).str.strip()
    elo = pd.read_parquet("data/horse_elo_pit.parquet")[["race_key", "horse_name", "pre_elo"]]
    elo["hn"] = elo["horse_name"].astype(str).str.strip()
    df = df.merge(elo[["race_key", "hn", "pre_elo"]], left_on=["rk", "hn"], right_on=["race_key", "hn"], how="left")
    print(f"Eloマージ率: {df['pre_elo'].notna().mean()*100:.1f}%")
    # 前走着順(第6要素): 検証で4-5着1.59x/大敗0.82x消しと最も強い新規シグナル
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["horse_id", "date"])
    df["prev_rank"] = df.groupby("horse_id")["rank"].shift()
    df["prank_cat"] = pd.cut(df["prev_rank"], [0, 3, 5, 9, 99],
                             labels=["前走1-3着", "前走4-5着", "前走6-9着", "前走10着↓"])
    # 第7要素: 前走上がり3F順位(Eloと直交=低中Elo帯で1.14-1.19x上積み)。race_idは行ユニークなのでrkで集計
    df["l3f_rank"] = df.groupby("rk")["last_3f"].rank()
    df["l3f_n"] = df.groupby("rk")["last_3f"].transform("count")
    df["l3f_pct"] = np.where(df["l3f_n"] >= 5, df["l3f_rank"] / df["l3f_n"], np.nan)
    df["prev_l3f_pct"] = df.groupby("horse_id")["l3f_pct"].shift()
    df["l3f_cat"] = pd.cut(df["prev_l3f_pct"], [0, 0.2, 0.4, 0.7, 1.01],
                           labels=["前走上り上位", "前走上り中上", "前走上り中", "前走上り下位"])
    # レース内Eloパーセンタイル(縦比較の実力)
    df["elo_pct"] = df.groupby("rk")["pre_elo"].rank(pct=True)
    df = _add_style(df)
    df["wbin"] = _weight_bin(df["horse_weight"])

    ana = df[df["popularity"] >= ANA_POP].copy()
    ana["place"] = (ana["rank"] <= 3).astype(int)
    tr = ana[ana["year"] <= 24].copy()
    te = ana[ana["year"] >= 25].copy()
    base_tr = tr["place"].mean()
    print(f"学習(〜24){len(tr)}件 base{base_tr*100:.2f}% / テスト(25-26){len(te)}件 base{te['place'].mean()*100:.2f}%")

    # --- 学習期間でリフトテーブル ---
    def lift_map(col, minN=MIN_J):
        g = tr.dropna(subset=[col]).groupby(col)["place"].agg(["mean", "size"])
        return {k: (g.loc[k, "mean"] / base_tr) for k in g.index if g.loc[k, "size"] >= minN}

    maps = {c: lift_map(c) for c in ["style", "jockey", "sire", "damsire", "wbin", "prank_cat", "l3f_cat"]}
    # Elo分位リフト(学習でqcut境界、テストに適用)
    tr_e = tr.dropna(subset=["elo_pct"])
    tr_e["eq"], bins = pd.qcut(tr_e["elo_pct"], 5, labels=False, retbins=True, duplicates="drop")
    elo_lift = {q: (tr_e[tr_e["eq"] == q]["place"].mean() / base_tr) for q in tr_e["eq"].unique()}

    def score_row(r):
        s = 1.0
        for c in ["style", "jockey", "sire", "damsire", "wbin", "prank_cat", "l3f_cat"]:
            s *= maps[c].get(r[c], 1.0)
        if pd.notna(r["elo_pct"]):
            q = min(np.searchsorted(bins, r["elo_pct"], side="right") - 1, len(bins) - 2)
            s *= elo_lift.get(max(q, 0), 1.0)
        return s

    te["score"] = te.apply(score_row, axis=1)
    base_te = te["place"].mean()
    te["sq"] = pd.qcut(te["score"], 5, labels=["Q1最低", "Q2", "Q3", "Q4", "Q5最高"], duplicates="drop")
    print("\n=== テスト期間: 複合スコア分位 × 大穴複勝率 ===")
    g = te.groupby("sq", observed=True)["place"].agg(["mean", "size"])
    for q in g.index:
        print(f"  {q}: 複勝{g.loc[q,'mean']*100:.2f}% (lift{g.loc[q,'mean']/base_te:.2f}x n{int(g.loc[q,'size'])})")
    # 最上位の更に上澄み(score上位5%)
    thr = te["score"].quantile(0.95)
    top = te[te["score"] >= thr]
    print(f"\n  スコア上位5%: 複勝{top['place'].mean()*100:.2f}% "
          f"(lift{top['place'].mean()/base_te:.2f}x n{len(top)})")
    # 単要素のテスト期間リフト(比較用)
    print("\n=== 参考: 単要素のテスト期間リフト(最上位カテゴリ) ===")
    for c in ["style", "jockey"]:
        gg = te.groupby(c, observed=True)["place"].mean()
    te_e = te.dropna(subset=["elo_pct"]); top_elo = te_e[te_e["elo_pct"] >= 0.8]
    print(f"  Elo上位20%(縦比較で実力上位)単独: lift{top_elo['place'].mean()/base_te:.2f}x n{len(top_elo)}")
    te_s = te[te["style"] == "逃先"]
    print(f"  逃先脚質 単独: lift{te_s['place'].mean()/base_te:.2f}x n{len(te_s)}")

    # === 複勝回収率の概算 ===
    # 複勝配当データが無いため、人気別の実複勝率から複勝配当を推定(控除20%仮定: payout≈0.8/複勝率)。
    # 無差別購入が約80%(控除分)に収束すれば推定は妥当。軸候補がそれを超えればエッジあり。
    pop_place = ana.groupby("popularity")["place"].mean()

    def est_fuku(p):
        pr = pop_place.get(p, np.nan)
        return 0.8 / pr if pr and pr > 0 else 0.0

    print("\n=== 複勝回収率の概算（複勝配当データ無→人気別複勝率から控除20%で推定）===")
    for lab, sub in [("全大穴 無差別", te),
                     ("複合スコア上位20%(軸候補)", te[te["score"] >= te["score"].quantile(0.80)]),
                     ("複合スコア上位5%", te[te["score"] >= te["score"].quantile(0.95)]),
                     ("複合スコア下位20%(消し)", te[te["score"] <= te["score"].quantile(0.20)])]:
        ret = sub.apply(lambda r: est_fuku(r["popularity"]) if r["place"] else 0.0, axis=1)
        print(f"  {lab}: 複勝率{sub['place'].mean()*100:.1f}% 推定回収率{ret.mean()*100:.0f}% (n{len(sub)})")


if __name__ == "__main__":
    main()
