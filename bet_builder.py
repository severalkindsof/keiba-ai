"""
馬券構成自動提案モジュール v2。

対応券種: 3連複・馬連・3連単（2〜3点）
バイアス反映: 当日のバイアス（内前有利等）を軸・紐の選定に反映する。
予算上限: 指定予算内に自動的に収まるよう調整。

ユーザーの好みに合わせたデフォルト構成:
  - メイン: 3連複（穴馬軸1頭 × 人気馬1頭 × 相手3〜5頭）
  - サブ:   馬連（単勝より馬連を好む）
  - 狙い:   3連単（2〜3点に絞る）
"""
import itertools
import pandas as pd
import numpy as np
from dataclasses import dataclass

TICKET_UNIT = 100


@dataclass
class Ticket:
    bet_type: str
    horses: list[str]
    amount: int
    expected_return: float
    ev: float
    reason: str = ""


def _round_to_unit(amount: int) -> int:
    return max(TICKET_UNIT, (amount // TICKET_UNIT) * TICKET_UNIT)


# ============================================================
# バイアス反映の軸・紐選定
# ============================================================

def select_axis_horses(
    eval_df: pd.DataFrame,
    bias_type: str = "neutral",
) -> dict:
    """
    バイアスを考慮して穴馬軸・人気馬軸・紐候補を選定する。

    Returns:
        longshot_axis: 穴馬軸（1頭）
        popular_axis:  人気馬軸（1頭）
        partners:      紐候補（3〜5頭）
        n_partners:    流す相手の数
    """
    if eval_df.empty:
        return {}

    df = eval_df.copy()

    def _col(df, name, default):
        """列が存在しない場合はデフォルト値のSeriesを返す"""
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    df["ev"]           = _col(df, "ev",                   -0.5)
    df["popularity"]   = _col(df, "popularity",             9)
    df["odds"]         = _col(df, "odds",                  10)
    df["bias_bonus"]   = _col(df, "realtime_bias_bonus",    0)
    df["conf_score"]   = _col(df, "confidence_score",      50)

    # バイアス補正スコアで並べ替え
    df["selection_score"] = df["ev"] * 10 + df["conf_score"] / 100 + df["bias_bonus"] * 5

    # 穴馬軸：7番人気以上でselection_score最大
    longshots = df[df["popularity"] >= 7].sort_values("selection_score", ascending=False)
    # 第34波 (P3): 少頭数戦で人気7位以下が不在だと None → 全主要券種が

    # スキップされていた → 相対的な穴（人気下位半分の最上位評価馬）にフォールバック

    if longshots.empty:

        _half = df[df["popularity"] >= df["popularity"].median()]

        ls_axis = _half.iloc[0] if not _half.empty else None

    else:

        ls_axis = longshots.iloc[0]

    # 人気馬軸：1〜5番人気でselection_score最大
    populars = df[df["popularity"] <= 5].sort_values("selection_score", ascending=False)
    pop_axis = populars.iloc[0] if not populars.empty else df.sort_values("popularity").iloc[0]

    # 紐候補：軸2頭以外でselection_score上位3〜5頭
    axis_names = set()
    if ls_axis is not None:  axis_names.add(ls_axis["horse_name"])
    axis_names.add(pop_axis["horse_name"])

    partners_df = df[~df["horse_name"].isin(axis_names)].sort_values(
        "selection_score", ascending=False
    )

    # バイアスによる紐の絞り込み（インデックスをリセットして比較）
    n_partners = 4
    partners_df = partners_df.reset_index(drop=True)
    if bias_type == "inner_speed":
        draw_ok  = partners_df["draw_bonus"].gt(0) if "draw_bonus" in partners_df.columns \
                   else pd.Series(False, index=partners_df.index)
        style_ok = partners_df["running_style"].str.contains("先行|逃げ", na=False) \
                   if "running_style" in partners_df.columns \
                   else pd.Series(False, index=partners_df.index)
        pref = partners_df[draw_ok | style_ok]
        rest = partners_df[~(draw_ok | style_ok)]
        partners_df = pd.concat([pref, rest], ignore_index=True)
    elif bias_type == "outer_diff":
        style_ok = partners_df["running_style"].str.contains("差し|追込", na=False) \
                   if "running_style" in partners_df.columns \
                   else pd.Series(False, index=partners_df.index)
        pref = partners_df[style_ok]
        rest = partners_df[~style_ok]
        partners_df = pd.concat([pref, rest], ignore_index=True)

    partners = partners_df.head(n_partners)["horse_name"].tolist()

    return {
        "longshot_axis": ls_axis["horse_name"] if ls_axis is not None else None,
        "popular_axis":  pop_axis["horse_name"],
        "partners":      partners,
        "n_partners":    len(partners),
        "ls_odds":       float(ls_axis["odds"]) if ls_axis is not None else 0,
        "pop_odds":      float(pop_axis["odds"]),
        "ls_ev":         float(ls_axis["ev"]) if ls_axis is not None else 0,
    }


# ============================================================
# Phase C: ケリー基準（賭け金最適化）
# ============================================================

def kelly_fraction(win_rate: float, odds: float, fraction: float = 0.25) -> float:
    """
    フラクショナルケリー基準による最適賭け比率。
    fraction=0.25 → 計算値の25%（過剰リスク回避のため推奨）

    Args:
        win_rate: 推定勝率（0〜1）
        odds:     単勝オッズ
        fraction: フラクション（デフォルト0.25）
    Returns:
        総資金に対する推奨賭け比率（0〜1）。0なら「見送り推奨」。
    """
    # LATENT-4: EV閾値チェック（EV<5%は賭けない）
    ev = win_rate * (odds - 1.0) - (1.0 - win_rate)
    if ev <= 0.05:
        return 0.0
    b = odds - 1.0
    if b <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    f = (win_rate * b - (1.0 - win_rate)) / b
    return max(0.0, round(f * fraction, 4))


def kelly_amount(win_rate: float, odds: float, total_funds: int,
                 fraction: float = 0.25) -> int:
    """ケリー基準に基づく推奨賭け金額（円）。"""
    f = kelly_fraction(win_rate, odds, fraction)
    raw = total_funds * f
    return _round_to_unit(int(raw))


def suggest_axis_and_partner(
    eval_df: pd.DataFrame,
    budget: int,
    bet_unit: int = 100,
    max_partners: int = 4,
) -> dict:
    """
    A-6: 軸 + 押さえ自動構成
        - 上位 confidence_score 1頭を「軸」
        - 2-5位を「押さえ」（最大 max_partners 頭）
        - 馬連 (axis × 各 partner) と 三連複 (axis × partner × partner) を提案

    Returns:
        {
          "axis": {horse_name, score, ev},
          "partners": [{horse_name, score}, ...],
          "umaren_bets": [{key, prob_est, amount}],     # 馬連
          "trio_bets":   [{key, prob_est, amount}],     # 三連複
          "total_amount": int,
        }
    """
    if eval_df.empty or "confidence_score" not in eval_df.columns:
        return {"axis": None, "partners": [], "umaren_bets": [], "trio_bets": [], "total_amount": 0}

    # 軸 = confidence_score トップ
    sorted_df = eval_df.sort_values("confidence_score", ascending=False).reset_index(drop=True)
    if len(sorted_df) < 3:
        return {"axis": None, "partners": [], "umaren_bets": [], "trio_bets": [], "total_amount": 0}

    axis_row = sorted_df.iloc[0]
    partner_rows = sorted_df.iloc[1: 1 + max_partners]

    # 軸と押さえの win prob を取得（blended_pct 優先）
    def _p(row):
        if "blended_pct" in row and pd.notna(row["blended_pct"]):
            return float(row["blended_pct"]) / 100
        if "lgbm_norm_pct" in row and pd.notna(row["lgbm_norm_pct"]):
            return float(row["lgbm_norm_pct"]) / 100
        return float(row.get("est_win_rate", 5)) / 100

    p_axis = _p(axis_row)
    p_partners = [(r["horse_name"], _p(r), r.to_dict()) for _, r in partner_rows.iterrows()]

    # ----- 馬連: 各 (axis, partner) ペアの Harville 近似確率 -----
    # P(axis 1, partner 2) + P(partner 1, axis 2)
    umaren = []
    for hp, pp, prow in p_partners:
        # Plackett-Luce 近似: 一方が1着、他方が2着の合計確率
        denom1 = max(1 - p_axis, 1e-6)
        denom2 = max(1 - pp,     1e-6)
        prob = p_axis * pp / denom1 + pp * p_axis / denom2
        umaren.append({
            "key": f"{axis_row['horse_name']} - {hp}",
            "horses": (axis_row["horse_name"], hp),
            "prob_est": prob,
        })

    # ----- 三連複: (axis, partner_i, partner_j) -----
    trios = []
    for i in range(len(p_partners)):
        for j in range(i + 1, len(p_partners)):
            h_i, p_i, _ = p_partners[i]
            h_j, p_j, _ = p_partners[j]
            # 6つの順列の合計
            prob = 0.0
            triple = [(axis_row["horse_name"], p_axis), (h_i, p_i), (h_j, p_j)]
            from itertools import permutations as _perms
            for a, b, c in _perms(triple):
                pa, pb, pc = a[1], b[1], c[1]
                d1 = max(1 - pa, 1e-6)
                d2 = max(1 - pa - pb, 1e-6)
                prob += pa * pb / d1 * pc / d2
            trios.append({
                "key": " - ".join(sorted([axis_row["horse_name"], h_i, h_j])),
                "horses": (axis_row["horse_name"], h_i, h_j),
                "prob_est": prob,
            })

    # ----- 予算配分: 確率に比例 -----
    # 馬連に予算の 60%、三連複に 40% を割り当て
    umaren_budget = int(budget * 0.6)
    trio_budget   = int(budget * 0.4)

    total_um = sum(u["prob_est"] for u in umaren) or 1.0
    for u in umaren:
        u["amount"] = max(0, int((umaren_budget * u["prob_est"] / total_um) // bet_unit) * bet_unit)

    total_tr = sum(t["prob_est"] for t in trios) or 1.0
    for t in trios:
        t["amount"] = max(0, int((trio_budget * t["prob_est"] / total_tr) // bet_unit) * bet_unit)

    total_amt = sum(u["amount"] for u in umaren) + sum(t["amount"] for t in trios)
    return {
        "axis": {"horse_name": axis_row["horse_name"],
                 "score": int(axis_row.get("confidence_score", 0)),
                 "odds": float(axis_row.get("odds", 0)),
                 "ev": float(axis_row.get("ev", 0))},
        "partners": [{"horse_name": h, "win_prob": p, "score": int(r.get("confidence_score", 0))}
                     for h, p, r in p_partners],
        "umaren_bets": [u for u in umaren if u["amount"] > 0],
        "trio_bets":   [t for t in trios   if t["amount"] > 0],
        "total_amount": total_amt,
    }


def variance_diversify_bets(
    win_prob: float,
    place_prob: float,
    win_odds: float,
    place_odds: float,
    wide_odds: float | None,
    wide_prob: float | None,
    budget: int,
    bet_unit: int = 100,
) -> dict:
    """
    A-7: 単一の軸馬に対して 単勝・複勝・ワイド の最適配分。
    各券種の EV を計算し、EV+ 券種のみに分散投資（variance ↓）。
    """
    bets = {}
    # 単勝 EV
    if win_odds and win_odds > 1:
        ev_w = win_prob * (win_odds - 1) - (1 - win_prob)
        if ev_w > 0:
            bets["単勝"] = {"odds": win_odds, "prob": win_prob, "ev": ev_w}
    # 複勝 EV
    if place_odds and place_odds > 1:
        ev_p = place_prob * (place_odds - 1) - (1 - place_prob)
        if ev_p > 0:
            bets["複勝"] = {"odds": place_odds, "prob": place_prob, "ev": ev_p}
    # ワイド（オプション）
    if wide_odds and wide_odds > 1 and wide_prob:
        ev_wd = wide_prob * (wide_odds - 1) - (1 - wide_prob)
        if ev_wd > 0:
            bets["ワイド"] = {"odds": wide_odds, "prob": wide_prob, "ev": ev_wd}

    if not bets:
        return {"bets": [], "total_amount": 0, "note": "+EVな券種なし"}

    # EV比例で配分（リスク分散）
    total_ev = sum(b["ev"] for b in bets.values())
    out = []
    for name, b in bets.items():
        share = b["ev"] / total_ev
        amount = max(0, int((budget * share) // bet_unit) * bet_unit)
        if amount > 0:
            out.append({
                "ticket": name, "odds": b["odds"], "prob": round(b["prob"], 3),
                "ev": round(b["ev"], 3), "share_pct": round(share * 100, 1),
                "amount": amount, "expected_return": int(b["prob"] * b["odds"] * amount),
            })
    return {"bets": out, "total_amount": sum(o["amount"] for o in out)}


def optimize_multi_bet_harville(
    combos: list[dict],
    real_odds_map: dict,
    budget: int,
    take_rate: float = 0.225,
    max_bets: int = 8,
    fraction: float = 0.50,
    bet_unit: int = 100,
) -> list[dict]:
    """
    BOOST-1: Harville 結果から +EV 組合せだけを抽出して予算内で最適配分。

    Args:
        combos:        harville.top_n_combinations 出力（'horses', 'prob', 'fair_odds'）
        real_odds_map: 組合せキー(zero-padded) → 実オッズ
        budget:        予算上限（円）
        take_rate:     控除率（参考表示用）
        max_bets:      最大点数
        fraction:      分数ケリー（0.50 = ハーフ）
        bet_unit:      最小賭け単位

    Returns:
        [{horses, prob, real_odds, ev, amount, kelly_pct}, ...] 期待リターン順
    """
    def _key(h):
        return "-".join(str(int(x)).zfill(2) for x in h)

    # 各組合せの EV を計算 → +EV だけ抽出
    plus = []
    for c in combos:
        ro = real_odds_map.get(_key(c["horses"]))
        if ro is None or ro <= 1.0:
            continue
        p = float(c["prob"])
        ev = p * (ro - 1) - (1 - p)
        if ev <= 0:
            continue
        k = (p * (ro - 1) - (1 - p)) / max(ro - 1, 0.01) * fraction
        plus.append({
            "horses": c["horses"],
            "prob": p,
            "real_odds": ro,
            "fair_odds": c.get("fair_odds"),
            "ev": ev,
            "raw_kelly": max(0.0, k),
        })

    if not plus:
        return []
    # ケリー値降順で並べる
    plus.sort(key=lambda x: -x["raw_kelly"])
    plus = plus[:max_bets]

    # 合計ケリーを予算に合わせて正規化
    total_k = sum(b["raw_kelly"] for b in plus)
    if total_k <= 0:
        return []
    for b in plus:
        b["adjusted_kelly"] = b["raw_kelly"] / total_k
        b["amount"] = max(0, int((budget * b["adjusted_kelly"]) // bet_unit) * bet_unit)
        b["kelly_pct"] = round(b["adjusted_kelly"] * 100, 1)
    # 配分0円のは削除
    plus = [b for b in plus if b["amount"] > 0]
    return plus


def apply_portfolio_kelly_to_df(
    eval_df: pd.DataFrame,
    bankroll: int,
    max_exposure: float = 0.20,
    fraction: float = 0.50,
    bet_unit: int = 100,
    prob_col: str = "blended_pct",
) -> pd.DataFrame:
    """
    IMPROVE-5: 同レース内の EV+ 馬を Portfolio Kelly でまとめて正規化。
    eval_df に 'portfolio_amount' 列を付与して返す。
    """
    from portfolio_kelly import PortfolioBet, normalize_portfolio
    if eval_df.empty:
        return eval_df
    if prob_col not in eval_df.columns:
        prob_col = "lgbm_norm_pct" if "lgbm_norm_pct" in eval_df.columns else None
    if not prob_col:
        eval_df["portfolio_amount"] = 0
        return eval_df

    df = eval_df.copy()
    df["portfolio_amount"] = 0
    # EV+ かつ買い対象のみ
    cand = df[(df.get("ev", -1) > 0) & (df.get("buy_flag", True) == True)].copy()
    if cand.empty:
        return df
    bets = []
    for idx, row in cand.iterrows():
        p = float(row[prob_col]) / 100 if prob_col.endswith("pct") else float(row[prob_col])
        odds = float(row.get("odds", 0))
        bets.append(PortfolioBet(
            race_id=str(row.get("race_id", "")),
            horse_name=str(row.get("horse_name", "")),
            win_prob=p,
            odds=odds,
            raw_kelly=kelly_fraction(p, odds, fraction=1.0),  # raw=フルケリー、fraction はあとで適用
        ))
    bets = normalize_portfolio(bets, bankroll=bankroll, max_exposure=max_exposure,
                                fraction=fraction, bet_unit=bet_unit)
    # 結果を eval_df に反映
    amount_map = {b.horse_name: b.adjusted_amount for b in bets}
    df["portfolio_amount"] = df["horse_name"].map(amount_map).fillna(0).astype(int)
    return df


# ============================================================
# Phase F: 見送り条件の最適化
# ============================================================

def dynamic_ev_threshold(popularity: int) -> float:
    """
    BOOST-2: 人気帯別の動的 EV 閾値。
    バックテスト経験値ベース：
      1-3番人気: +0.05（割安が小さくても回収率高い）
      4-6番人気: +0.10
      7-9番人気: +0.18
      10-13番人気: +0.25
      14番人気以下: +0.35（極大穴は高ハードル）
    """
    try:
        p = int(popularity)
    except (ValueError, TypeError):
        return 0.10
    if p <= 3:   return 0.05
    if p <= 6:   return 0.10
    if p <= 9:   return 0.18
    if p <= 13:  return 0.25
    return 0.35


def should_buy(ev: float, odds: float, confidence_score: int,
               ev_threshold: float = 0.10,
               odds_min: float = 2.5,
               score_min: int = 55,
               popularity: int | None = None,
               use_dynamic: bool = False) -> tuple[bool, str]:
    """
    購入推奨判定。

    先人事例: EV>130%（EV>0.30）かつオッズ2.5以上で回収率安定。
    デフォルトは EV>0.10（10%以上の期待値エッジ）を基本閾値とする。

    Returns:
        (True/False, 判定理由)
    """
    try:
        ev = float(ev)
    except (TypeError, ValueError):
        return False, "EV不明"
    if np.isnan(ev):
        return False, "EV不明"
    # BOOST-2: 動的 EV 閾値（人気帯別）
    if use_dynamic and popularity is not None:
        ev_threshold = dynamic_ev_threshold(popularity)
    if ev < ev_threshold:
        return False, f"EV{ev:+.2f}（閾値{ev_threshold:+.2f}未満）"
    if odds < odds_min:
        return False, f"オッズ{odds:.1f}（最低{odds_min:.1f}未満）"
    if confidence_score < score_min:
        return False, f"スコア{confidence_score}（閾値{score_min}未満）"
    return True, "買い推奨"


def apply_buy_filter(eval_df: pd.DataFrame,
                     ev_threshold: float = 0.10,
                     odds_min: float = 2.5,
                     score_min: int = 55,
                     use_dynamic: bool = False) -> pd.DataFrame:
    """
    eval_df に 'buy_flag'（True/False）と 'buy_reason' 列を付与して返す。
    use_dynamic=True で BOOST-2 の人気帯別動的閾値を使う。
    """
    if eval_df.empty:
        return eval_df
    results = [
        should_buy(
            float(row.get("ev", float("nan"))),
            float(row.get("odds", 0)),
            int(row.get("confidence_score", 0)),
            ev_threshold, odds_min, score_min,
            popularity=int(row.get("popularity", 9)),
            use_dynamic=use_dynamic,
        )
        for _, row in eval_df.iterrows()
    ]
    eval_df = eval_df.copy()
    eval_df["buy_flag"]   = [r[0] for r in results]
    eval_df["buy_reason"] = [r[1] for r in results]
    return eval_df


# ============================================================
# NEW-1: 勝率累乗→期待値差フィルター（hiyameshi66氏 81→124%）
# ============================================================

def apply_power_filter(
    eval_df: pd.DataFrame,
    power: float = 4.0,
    gap_threshold: float = 0.4,
    prob_col: str = "blended_pct",
) -> pd.DataFrame:
    """
    勝率を power 乗 → オッズ掛けて新EVを計算 → トップ2の差が gap_threshold 以上なら power_buy=True。

    背景: 勝率を累乗すると確信度の高い差が強調される。
    トップ予測との差が小さい馬は「混戦の中の偶然1位」 → 買わない方が良い。

    Args:
        eval_df: 評価済 DataFrame
        power: 累乗指数（典型: 2〜4）
        gap_threshold: トップ2のpower_EV差の閾値（典型: 0.3〜0.6）
        prob_col: 勝率列名（blended_pct 優先、無ければ lgbm_norm_pct or est_win_rate）

    Returns:
        eval_df + power_ev, power_top_gap, power_buy 列
    """
    if eval_df.empty:
        return eval_df
    df = eval_df.copy()
    if prob_col not in df.columns:
        for fallback in ("lgbm_norm_pct", "est_win_rate"):
            if fallback in df.columns:
                prob_col = fallback
                break
        else:
            df["power_ev"] = np.nan
            df["power_top_gap"] = np.nan
            df["power_buy"] = False
            return df
    if "odds" not in df.columns:
        df["power_ev"] = np.nan
        df["power_top_gap"] = np.nan
        df["power_buy"] = False
        return df

    p = (df[prob_col].fillna(0) / 100).clip(lower=1e-6, upper=1.0)
    odds = df["odds"].fillna(0).clip(lower=1.01)
    # 累乗確率（合計は1にならないが順序付けには十分）
    p_pow = p ** power
    df["power_ev"] = (p_pow * odds).round(3)
    # トップ2 の power_ev 差
    sorted_pe = df["power_ev"].sort_values(ascending=False).values
    if len(sorted_pe) >= 2:
        top_gap = float(sorted_pe[0] - sorted_pe[1])
    else:
        top_gap = float("inf")
    df["power_top_gap"] = top_gap
    # トップ馬のみ buy=True（gap が閾値超なら）
    top_idx = df["power_ev"].idxmax()
    df["power_buy"] = False
    if top_gap >= gap_threshold:
        df.loc[top_idx, "power_buy"] = True
    return df


# ============================================================
# NEW-4: 過去N戦未満の馬を見送り対象に
# ============================================================

def apply_data_quality_filter(
    eval_df: pd.DataFrame,
    df_hist: pd.DataFrame,
    min_past_races: int = 3,
) -> pd.DataFrame:
    """
    過去 min_past_races 戦未満の馬には data_quality_ok=False を付与。
    Mshimia氏「データ質確保で精度安定」。
    """
    if eval_df.empty:
        return eval_df
    df = eval_df.copy()
    if df_hist is None or df_hist.empty or "horse_name" not in df_hist.columns:
        df["past_race_count"] = -1
        df["data_quality_ok"] = True
        return df
    counts = df_hist.groupby("horse_name").size()
    df["past_race_count"] = df["horse_name"].map(counts).fillna(0).astype(int)
    df["data_quality_ok"] = df["past_race_count"] >= min_past_races
    return df


# ============================================================
# 第19波 (W3): 爆穴モード専用の買い目構成
# 複勝EV（place_ev）ベースで複勝・ワイドに配分する。
# 従来 build_tickets は単勝EVベースの軸選定のため、爆穴モードの
# 「複勝EVプラスの穴馬を買う」方針と不整合だった。
# ============================================================

def build_longshot_tickets(
    eval_df: pd.DataFrame,
    budget: int = 1000,
) -> dict:
    """
    爆穴モード用: 複勝EVプラスの穴馬（buy_flag=True）に複勝中心で配分。

    配分ロジック:
        - 各穴馬の複勝 Kelly 比率 = (place_prob × place_odds_est - 1) / (place_odds_est - 1)
        - Kelly 比率で予算を按分（1/2 Kelly 相当に抑制）
        - 上位 2 頭が両方 place_prob >= 0.15 ならワイド 1 点を追加
    """
    if eval_df.empty:
        return _empty_result("出走馬データがありません")
    need_cols = {"place_ev", "place_prob", "horse_name"}
    if not need_cols.issubset(eval_df.columns):
        return _empty_result("複勝EV列がありません（分析を先に実行してください）")

    # (第20波 U2 修正) buy_flag 列欠如時に df[False] で KeyError クラッシュしていた
    if "buy_flag" not in eval_df.columns:
        return _empty_result("buy_flag 列がありません（フィルタ適用前のデータです）")
    cand = eval_df[eval_df["buy_flag"] == True].copy()  # noqa: E712
    if cand.empty:
        return _empty_result("複勝EVプラスの穴馬がいません。このレースは爆穴向きではありません。")

    cand = cand.sort_values("place_ev", ascending=False).head(4)
    if "place_odds_est" in cand.columns:
        p_odds = cand["place_odds_est"]
    else:
        _w_bb = pd.to_numeric(cand.get("odds"), errors="coerce").fillna(10)
        _c_bb = np.clip(0.30 - 0.0075 * (_w_bb - 10).clip(lower=0), 0.15, 0.30)
        p_odds = 1.0 + (_w_bb - 1) * _c_bb

    # 複勝 Kelly（1/2 Kelly）
    b = (p_odds - 1.0).clip(lower=0.05)
    p = cand["place_prob"].clip(0.01, 0.95)
    kelly = ((p * (b + 1) - 1) / b).clip(lower=0) * 0.5
    if kelly.sum() <= 0:
        return _empty_result("Kelly 配分がゼロ（複勝EVが薄すぎます）")

    weights = kelly / kelly.sum()
    tickets: list[Ticket] = []
    used = 0
    for (_idx, row), w in zip(cand.iterrows(), weights):
        # 第33波修正: _round_to_unit は max(100, x) で 0 を 100 に引き上げるため、
        # Kelly=0（EVマイナス）の馬にも 100 円が配分されていた
        # → 切り捨て方式 + Kelly ゼロは明示スキップ
        if w <= 0 or float(row.get("place_ev", 0)) <= 0:
            continue
        _raw = int(budget * 0.8 * w)
        if _raw < TICKET_UNIT:
            continue
        amt = (_raw // TICKET_UNIT) * TICKET_UNIT
        try:
            po = float(p_odds.loc[_idx])
        except Exception:
            po = 3.0
        tickets.append(Ticket(
            bet_type="複勝",
            horses=[str(row["horse_name"])],
            amount=amt,
            expected_return=round(amt * po, 0),
            ev=float(row["place_ev"]),
            reason=f"複勝EV {row['place_ev']:+.2f} / 馬券内 {row['place_prob']*100:.0f}%",
        ))
        used += amt

    # ワイド 1 点（上位 2 頭が十分な馬券内確率を持つ場合）
    if len(cand) >= 2:
        top2 = cand.head(2)
        if (top2["place_prob"] >= 0.15).all():
            amt_w = _round_to_unit(int(budget * 0.2))
            if amt_w >= TICKET_UNIT and used + amt_w <= budget:
                tickets.append(Ticket(
                    bet_type="ワイド",
                    horses=[str(n) for n in top2["horse_name"]],
                    amount=amt_w,
                    expected_return=0.0,
                    ev=float(top2["place_ev"].mean()),
                    reason="穴馬上位2頭のワイド（両者とも馬券内15%以上）",
                ))
                used += amt_w

    if not tickets:
        return _empty_result("予算が小さすぎて買い目を構成できません")

    return {
        "recommended": tickets,
        "total_cost": used,
        "remaining_budget": budget - used,
        "axis": {"mode": "爆穴・複勝EVベース"},
        "romance_plan": [], "romance_cost": 0,
        "brain_warning": "",
        "bias_applied": "longshot_place_ev",
    }


# ============================================================
# メイン馬券構成関数
# ============================================================

def build_tickets(
    eval_df: pd.DataFrame,
    budget: int = 5000,
    surface: str = "芝",
    distance: int = 2000,
    bias_type: str = "neutral",
) -> dict:
    """
    評価済みDataFrame + バイアス情報から馬券構成案を構築する。
    """
    if eval_df.empty:
        return _empty_result("出走馬データがありません")

    axis = select_axis_horses(eval_df, bias_type)
    if not axis:
        return _empty_result("軸馬の選定に失敗しました")

    ls   = axis["longshot_axis"]   # 穴馬軸
    pop  = axis["popular_axis"]    # 人気馬軸
    partners = axis["partners"]    # 紐候補

    tickets  = []
    remaining = budget

    # ---- 3連複（メイン：穴馬軸 × 人気馬軸 × 相手N頭） ---- #
    if ls and pop and partners:
        combos = [p for p in partners if p not in (ls, pop)]
        n_combo = min(len(combos), 5)  # 最大5頭流し
        combo_cost = n_combo * TICKET_UNIT
        if combo_cost <= remaining * 0.6:
            unit = _round_to_unit(min(200, (remaining * 0.5) // max(n_combo, 1)))
            total_3f = unit * n_combo
            tickets.append(Ticket(
                bet_type="3連複",
                horses=[ls, pop, "流し"],
                amount=total_3f,
                expected_return=total_3f * 4.0,
                ev=axis["ls_ev"],
                reason=f"{ls}×{pop} 流し{n_combo}頭（{unit}円×{n_combo}点）",
            ))
            remaining -= total_3f

    # ---- 馬連（穴馬軸 × 人気馬軸） ---- #
    if ls and pop and remaining >= TICKET_UNIT * 2:
        amt_baren = _round_to_unit(min(500, remaining // 4))
        ls_odd   = axis["ls_odds"]
        pop_odd  = axis["pop_odds"]
        # 第34波 (P2): 旧 (ls×pop)^0.35 は 30×4倍で5.3倍と実勢(20-40倍)の1/4〜1/7の
        # 過小推定 → Harville 確率からの逆算（公平オッズ×控除率）に変更
        est_odds = (ls_odd * pop_odd) ** 0.5  # フォールバック（旧式より現実寄り）
        try:
            from harville import quinella_prob, TAKE_RATES
            _pcol = next((c for c in ("blended_pct", "lgbm_norm_pct", "est_win_rate")
                          if c in eval_df.columns), None)
            if _pcol:
                _ps = (pd.to_numeric(eval_df[_pcol], errors="coerce").fillna(0) / 100).values
                _names = eval_df["horse_name"].tolist()
                if ls in _names and pop in _names and _ps.sum() > 0.5:
                    _pq = quinella_prob(_ps, _names.index(ls), _names.index(pop))
                    if _pq > 1e-4:
                        est_odds = (1 - TAKE_RATES["quinella"]) / _pq
        except Exception:
            pass
        tickets.append(Ticket(
            bet_type="馬連",
            horses=[ls, pop],
            amount=amt_baren,
            expected_return=amt_baren * est_odds,
            ev=axis["ls_ev"] * 0.7,
            reason=f"{ls}×{pop}（軸2頭の直接対決）",
        ))
        remaining -= amt_baren

    # ---- 3連単（2〜3点に絞る） ---- #
    if ls and pop and partners and remaining >= TICKET_UNIT * 2:
        trifecta_count = 0
        # パターン1: 穴馬1着固定 → 人気馬2着 → 相手3着
        if len(partners) >= 1 and remaining >= TICKET_UNIT:
            third = next((p for p in partners if p not in (ls, pop)), None)
            if third:
                tickets.append(Ticket(
                    bet_type="3連単",
                    horses=[ls, pop, third],
                    amount=TICKET_UNIT,
                    expected_return=TICKET_UNIT * ls_odd_for_trifecta(axis["ls_odds"]),
                    ev=axis["ls_ev"],
                    reason=f"1着固定: {ls}（当たればラッキー）",
                ))
                remaining -= TICKET_UNIT
                trifecta_count += 1

        # パターン2: 人気馬1着 → 穴馬2着 → 相手3着
        if trifecta_count < 3 and len(partners) >= 1 and remaining >= TICKET_UNIT:
            third2 = next((p for p in partners if p not in (ls, pop)), None)
            if third2:
                tickets.append(Ticket(
                    bet_type="3連単",
                    horses=[pop, ls, third2],
                    amount=TICKET_UNIT,
                    expected_return=TICKET_UNIT * ls_odd_for_trifecta(axis["ls_odds"]) * 0.6,
                    ev=axis["ls_ev"] * 0.6,
                    reason=f"1着: {pop}、2着: {ls}（穴馬2着狙い）",
                ))
                remaining -= TICKET_UNIT

    total_cost = sum(t.amount for t in tickets)

    # ---- 旧来スタイルとの比較プラン ---- #
    romance_plan, romance_cost, brain_warning = _build_romance_plan(
        eval_df, budget, bias_type
    )

    return {
        "recommended":      tickets,
        "total_cost":       total_cost,
        "remaining_budget": budget - total_cost,
        "axis":             axis,
        "romance_plan":     romance_plan,
        "romance_cost":     romance_cost,
        "brain_warning":    brain_warning,
        "bias_applied":     bias_type,
    }


def ls_odd_for_trifecta(ls_odds: float) -> float:
    """穴馬オッズから3連単の概算倍率を推定（粗い参考値 — 表示専用、金額決定に不使用）"""
    return max(10.0, ls_odds * 8)


# ============================================================
# 旧来スタイル（大穴全流し）との比較
# ============================================================

def _build_romance_plan(
    eval_df: pd.DataFrame,
    budget: int,
    bias_type: str,
) -> tuple[list, int, str]:
    """2頭軸3連複全流しプランを再現して警告を出す。"""
    axis = select_axis_horses(eval_df, bias_type)
    if not axis or not axis.get("longshot_axis"):
        return [], 0, ""

    ls  = axis["longshot_axis"]
    pop = axis["popular_axis"]
    n_others = len(eval_df) - 2
    if n_others <= 0:
        return [], 0, ""

    total_cost = n_others * TICKET_UNIT
    ls_ev = axis["ls_ev"]

    warning = ""
    if ls_ev < -0.2:
        warning = (
            f"ブレイン警告: {ls}（{eval_df[eval_df['horse_name']==ls]['popularity'].iloc[0] if ls in eval_df['horse_name'].values else '?'}番人気）"
            f"のEV={ls_ev:+.3f}。全流しは{total_cost:,}円かかり、期待値的に不利です。\n"
            f"推奨プランとの差額: {total_cost - sum([TICKET_UNIT * 3]):+,}円"
        )
    elif total_cost > budget:
        warning = f"全流しは{total_cost:,}円で予算{budget:,}円をオーバーします。"

    plan = [Ticket(
        bet_type="3連複全流し（比較）",
        horses=[ls, pop, "残り全頭"],
        amount=total_cost,
        expected_return=total_cost * 0.75,
        ev=ls_ev,
        reason=f"旧スタイル: {ls}×{pop} 全流し{n_others}点",
    )]
    return plan, total_cost, warning


def _empty_result(msg: str) -> dict:
    return {
        "recommended": [], "total_cost": 0, "remaining_budget": 0,
        "axis": {}, "romance_plan": [], "romance_cost": 0,
        "brain_warning": msg, "bias_applied": "unknown",
    }


# ============================================================
# 表示用フォーマット
# ============================================================

def format_tickets_for_display(tickets: list[Ticket]) -> pd.DataFrame:
    rows = []
    for t in tickets:
        horse_str = (
            " → ".join(t.horses) if t.bet_type == "3連単"
            else " / ".join(t.horses)
        )
        rows.append({
            "券種":     t.bet_type,
            "馬":       horse_str,
            "金額":     f"{t.amount:,}円",
            "期待回収": f"{int(t.expected_return):,}円",
            "EV":       f"{t.ev:+.3f}",
            "根拠":     t.reason,
        })
    return pd.DataFrame(rows)


def get_bet_summary(result: dict) -> str:
    """馬券構成のサマリー文字列を返す（AI相談への注入用）"""
    if not result.get("recommended"):
        return "推奨馬券なし（見送り推奨）"
    axis = result.get("axis", {})
    lines = [
        f"穴馬軸: {axis.get('longshot_axis','?')}（{axis.get('ls_odds','?')}倍）",
        f"人気馬軸: {axis.get('popular_axis','?')}（{axis.get('pop_odds','?')}倍）",
        f"合計: {result['total_cost']:,}円 / {len(result['recommended'])}点",
    ]
    return "\n".join(lines)
