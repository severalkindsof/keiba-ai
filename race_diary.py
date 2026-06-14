"""
レース日記・振り返り・学習システム。

【手動入力が必要な項目（最小限）】
  1. 実際に買った馬券と金額（bet_builderの提案からワンクリック確定 or 修正）
  2. レース後の一言メモ（任意）

【自動取得する項目】
  - 着順・タイム → netkeibaレース結果ページ
  - 払い戻し金額（単勝/複勝/ワイド/3連複/3連単）→ netkeiba
  - ラップタイム → netkeiba
  - 馬体重 → netkeiba当日

【自動計算する項目】
  - 回収率 = 払い戻し / 投資 × 100
  - ファクター精度（予測したスコアが正しかったか）
  - 週次・月次ROI推移
"""
import json
import sqlite3
import time
import re
import requests
from bs4 import BeautifulSoup
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

DB_PATH = Path(__file__).parent / "race_diary.db"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


# ============================================================
# DB 初期化
# ============================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS race_records (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        race_id       TEXT,
        race_name     TEXT,
        race_date     TEXT,
        venue         TEXT,
        surface       TEXT,
        distance      INTEGER,
        track_condition TEXT,
        predicted_at  TEXT,
        confidence_top_score INTEGER,
        top_horse     TEXT,
        bias_type     TEXT,
        pace_predicted TEXT,
        budget        INTEGER,
        note          TEXT,
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS bets (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        race_record_id INTEGER,
        bet_type      TEXT,
        horses        TEXT,
        amount        INTEGER,
        is_hit        INTEGER DEFAULT 0,
        payout        INTEGER DEFAULT 0,
        FOREIGN KEY(race_record_id) REFERENCES race_records(id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS race_results (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        race_record_id INTEGER,
        rank          INTEGER,
        horse_name    TEXT,
        odds          REAL,
        popularity    INTEGER,
        time_str      TEXT,
        fetched_at    TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(race_record_id) REFERENCES race_records(id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS payouts (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        race_record_id INTEGER,
        bet_type      TEXT,
        combination   TEXT,
        payout_yen    INTEGER,
        FOREIGN KEY(race_record_id) REFERENCES race_records(id)
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS factor_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        race_record_id INTEGER,
        horse_name    TEXT,
        confidence_score INTEGER,
        ev            REAL,
        actual_rank   INTEGER,
        hit_flag      INTEGER,
        FOREIGN KEY(race_record_id) REFERENCES race_records(id)
    )""")

    # 第13波: auto_meta カラム追加（改善ループ用、後付け ALTER で互換性保持）
    try:
        c.execute("ALTER TABLE race_records ADD COLUMN auto_meta TEXT")
    except sqlite3.OperationalError:
        pass  # already exists

    conn.commit()
    conn.close()


# ============================================================
# レース予想を記録
# ============================================================

def save_race_prediction(
    race_id: str,
    race_name: str,
    race_date: str,
    venue: str,
    surface: str,
    distance: int,
    track_condition: str,
    eval_df: pd.DataFrame,
    bets: list[dict],
    bias_type: str,
    pace_predicted: str,
    budget: int,
    note: str = "",
    auto_meta: dict | None = None,
) -> int:
    """予想時点のデータを保存する。戻り値はrace_record_id。

    auto_meta: 改善ループ用の機械収集メタデータ（第13波）
      期待キー: data_unavailable_count, conformal_interval_width,
               conformal_skip_recommended, odds_alert_count,
               top_horse_popularity, ev_threshold_used, kelly_fraction,
               betting_mode (堅軸/爆穴), bet_count, intraday_bias_applied
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    top_score = int(eval_df["confidence_score"].max()) if not eval_df.empty and "confidence_score" in eval_df.columns else 0
    top_horse = eval_df.iloc[0]["horse_name"] if not eval_df.empty else ""

    c.execute("""
    INSERT INTO race_records
      (race_id, race_name, race_date, venue, surface, distance, track_condition,
       predicted_at, confidence_top_score, top_horse, bias_type, pace_predicted, budget, note, auto_meta)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (race_id, race_name, race_date, venue, surface, distance, track_condition,
          datetime.now().isoformat(), top_score, top_horse, bias_type, pace_predicted, budget, note,
          json.dumps(auto_meta or {}, ensure_ascii=False)))
    record_id = c.lastrowid

    # 買い目を保存
    for bet in bets:
        c.execute("""
        INSERT INTO bets (race_record_id, bet_type, horses, amount)
        VALUES (?,?,?,?)
        """, (record_id, bet.get("bet_type", ""),
              json.dumps(bet.get("horses", []), ensure_ascii=False),
              int(bet.get("amount", 0))))

    # ファクターログ
    if not eval_df.empty:
        for _, row in eval_df.iterrows():
            c.execute("""
            INSERT INTO factor_log (race_record_id, horse_name, confidence_score, ev)
            VALUES (?,?,?,?)
            """, (record_id, row.get("horse_name", ""),
                  int(row.get("confidence_score", 0)),
                  float(row.get("ev", 0.0)) if pd.notna(row.get("ev")) else 0.0))

    conn.commit()
    conn.close()
    return record_id


# ============================================================
# レース結果を自動取得
# ============================================================

# BUG-X5: TTL 統一 → レース固有 900秒
@st.cache_data(ttl=900)
def fetch_race_result_from_netkeiba(race_id: str) -> dict:
    """
    netkeibaのレース結果ページから着順と払い戻しを取得する。
    """
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    try:
        time.sleep(2)
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.content, "lxml")

        # 着順
        results = []
        for row in soup.select("table.RaceTable01 tr.HorseList"):
            cells = row.find_all("td")
            if len(cells) < 8:
                continue
            try:
                results.append({
                    "rank":       int(cells[0].get_text(strip=True)) if cells[0].get_text(strip=True).isdigit() else 99,
                    "gate":       cells[1].get_text(strip=True),
                    "horse_no":   cells[2].get_text(strip=True),
                    "horse_name": cells[3].get_text(strip=True),
                    "time_str":   cells[7].get_text(strip=True) if len(cells) > 7 else "",
                    "popularity": int(cells[-3].get_text(strip=True)) if cells[-3].get_text(strip=True).isdigit() else 0,
                    "odds":       float(cells[-4].get_text(strip=True)) if cells[-4].get_text(strip=True).replace(".", "").isdigit() else 0.0,
                })
            except Exception:
                continue

        # 払い戻し
        payouts = {}
        for table in soup.select("table.Payout_Detail_Table"):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    btype = cells[0].get_text(strip=True)
                    # 金額を抽出（カンマ除去、円記号除去）
                    amounts = re.findall(r"[\d,]+(?=円|$)", cells[-1].get_text())
                    combo_text = cells[1].get_text(strip=True) if len(cells) > 2 else ""
                    if btype and amounts:
                        # 第35波 (G1): 複勝・ワイドは1セルに複数配当が並ぶ/複数行になる
                        # → 旧実装は最初の1個 or 後勝ち上書きで配当が失われていた
                        _amts = [int(a.replace(",", "")) for a in amounts]
                        if btype in payouts:
                            payouts[btype]["amounts"].extend(_amts)
                            payouts[btype]["combination"] += "/" + combo_text
                        else:
                            payouts[btype] = {"combination": combo_text, "amounts": _amts}
                        # 後方互換: amount は平均値（複数配当の代表値）
                        payouts[btype]["amount"] = int(sum(payouts[btype]["amounts"]) / len(payouts[btype]["amounts"]))

        return {"results": results, "payouts": payouts, "fetched": True}

    except Exception as e:
        return {"results": [], "payouts": {}, "fetched": False, "error": str(e)}


def save_result_to_diary(race_record_id: int, fetched_data: dict, bets: list[dict]):
    """取得した結果をDBに保存し、買い目のヒット判定を行う。"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 着順を保存
    for r in fetched_data.get("results", []):
        c.execute("""
        INSERT INTO race_results (race_record_id, rank, horse_name, odds, popularity, time_str)
        VALUES (?,?,?,?,?,?)
        """, (race_record_id, r["rank"], r["horse_name"], r["odds"], r["popularity"], r["time_str"]))

    # 払い戻しを保存
    for btype, info in fetched_data.get("payouts", {}).items():
        c.execute("""
        INSERT INTO payouts (race_record_id, bet_type, combination, payout_yen)
        VALUES (?,?,?,?)
        """, (race_record_id, btype, info["combination"], info["amount"]))

    # 第25波修正: 旧実装は全券種を「3着内に全買い馬」で判定していた
    #   単勝→2,3着でも当たり扱い / 馬連→組・順序無視 / 払戻→最初の券種の金額を流用
    #   → 的中率・ROI が大幅に水増しされる二重バグだった
    _results = fetched_data.get("results", [])
    _rank_of = {r["horse_name"]: r["rank"] for r in _results}
    _name_at = {r["rank"]: r["horse_name"] for r in _results}
    top3 = {r["horse_name"] for r in _results if r["rank"] <= 3}
    top2 = {r["horse_name"] for r in _results if r["rank"] <= 2}
    _payouts = fetched_data.get("payouts", {})

    def _judge(bet_type: str, horses: list[str]) -> bool:
        hs = [h for h in horses if h != "残り全頭"]
        if not hs:
            return False
        bt = str(bet_type)
        # 流し馬券（"残り全頭"含む）は厳密判定不能 → 軸が条件を満たすかの緩判定
        if len(hs) < len(horses):
            return all(h in top3 for h in hs)
        if "単勝" in bt:
            return _rank_of.get(hs[0], 99) == 1
        if "複勝" in bt:
            return hs[0] in top3
        if "ワイド" in bt:
            return len(hs) >= 2 and all(h in top3 for h in hs[:2])
        if "馬単" in bt:
            return (len(hs) >= 2 and _name_at.get(1) == hs[0] and _name_at.get(2) == hs[1])
        if "馬連" in bt:
            return len(hs) >= 2 and set(hs[:2]) == top2 and len(top2) == 2
        if "3連単" in bt or "三連単" in bt:
            return (len(hs) >= 3 and _name_at.get(1) == hs[0]
                    and _name_at.get(2) == hs[1] and _name_at.get(3) == hs[2])
        if "3連複" in bt or "三連複" in bt:
            return len(hs) >= 3 and set(hs[:3]) == top3 and len(top3) == 3
        # 不明券種は従来の緩い判定にフォールバック
        return all(h in top3 for h in hs)

    c.execute("SELECT id, bet_type, horses, amount FROM bets WHERE race_record_id=?", (race_record_id,))
    for bet_id, bet_type, horses_json, amount in c.fetchall():
        horses = json.loads(horses_json)
        hit = _judge(bet_type, horses)
        payout = 0
        if hit:
            # 同一券種の払戻があればそれを使用（無ければ 0 = 過大計上しない）
            _po = next((v for k, v in _payouts.items() if str(bet_type) in k or k in str(bet_type)), None)
            if _po:
                payout = _po.get("amount", 0) * (amount // 100)
        c.execute("UPDATE bets SET is_hit=?, payout=? WHERE id=?", (int(hit), payout, bet_id))

    # factor_log の actual_rank を更新
    result_map = {r["horse_name"]: r["rank"] for r in fetched_data.get("results", [])}
    c.execute("SELECT id, horse_name FROM factor_log WHERE race_record_id=?", (race_record_id,))
    for log_id, horse_name in c.fetchall():
        actual_rank = result_map.get(horse_name, 99)
        hit = 1 if actual_rank <= 3 else 0
        c.execute("UPDATE factor_log SET actual_rank=?, hit_flag=? WHERE id=?",
                  (actual_rank, hit, log_id))

    conn.commit()
    conn.close()


# ============================================================
# レース後Claude自動分析
# ============================================================

def generate_post_race_analysis(
    race_record_id: int,
    fetched_data: dict,
    api_key: str,
) -> str:
    """
    レース結果とアプリの予測を比較してClaudeに「なぜ外れたか」を分析させる。
    返り値: 分析テキスト（日本語）
    """
    try:
        import anthropic
    except ImportError:
        return "anthropicパッケージが未インストールです"

    if not api_key:
        return "APIキーが設定されていません"

    init_db()
    conn = sqlite3.connect(DB_PATH)

    # 予測データ取得
    rr = pd.read_sql(
        "SELECT * FROM race_records WHERE id=?", conn, params=(race_record_id,)
    )
    bets = pd.read_sql(
        "SELECT * FROM bets WHERE race_record_id=?", conn, params=(race_record_id,)
    )
    factor_log = pd.read_sql(
        "SELECT * FROM factor_log WHERE race_record_id=?", conn, params=(race_record_id,)
    )
    conn.close()

    if rr.empty:
        return "レース記録が見つかりません"

    rec = rr.iloc[0]
    results = fetched_data.get("results", [])
    payouts = fetched_data.get("payouts", {})

    # 1〜3着馬
    top3 = [r for r in results if r.get("rank", 99) <= 3]
    top3_names = [r["horse_name"] for r in top3]

    # 予測していた馬（factor_log）
    predicted = factor_log[factor_log["confidence_score"] >= 60]["horse_name"].tolist() \
        if not factor_log.empty else []

    # 選ばなかった馬の中で好走した馬
    missed = [r for r in top3 if r["horse_name"] not in predicted]

    # 買い目のヒット状況
    hit_bets = bets[bets["is_hit"] == 1] if not bets.empty else pd.DataFrame()
    total_invest = int(bets["amount"].sum()) if not bets.empty else 0
    total_return = int(hit_bets["payout"].sum()) if not hit_bets.empty else 0

    prompt = f"""以下のレース予測と実際の結果を比較して、「なぜ外れたか」「何を見落としたか」「次回の教訓」を分析してください。

【レース情報】
{rec.get('race_name','')} / {rec.get('venue','')} {rec.get('surface','')} {rec.get('distance','')}m
展開予測: {rec.get('pace_predicted','')} / バイアス: {rec.get('bias_type','')}

【予測していた注目馬（スコア60点以上）】
{', '.join(predicted) if predicted else 'なし'}

【実際の1〜3着】
{' / '.join([f"{r['rank']}着 {r['horse_name']}({r.get('popularity','?')}番人気/{r.get('odds','?')}倍)" for r in top3])}

【選ばなかったが好走した馬】
{', '.join([f"{r['horse_name']}({r.get('popularity','?')}番人気)" for r in missed]) if missed else 'なし（予測通り）'}

【収支】
投資{total_invest:,}円 / 回収{total_return:,}円 / {'的中' if total_return > 0 else '外れ'}

以下の3点を簡潔に（各2〜3行で）分析してください：
1. なぜ外れたか（または当たったか）の主因
2. 見落としていた要因（特に「選ばなかった馬」が来た理由）
3. 次回のレース予想で活かすべき教訓
"""

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# ============================================================
# 統計・レポート
# ============================================================

def get_longshot_history_summary() -> dict:
    """
    ユーザーが過去に買った穴馬（7番人気以上）の的中パターンを集計する。

    Returns
    -------
    {
        "total_bets":     int,
        "total_hits":     int,
        "hit_rate":       float,
        "roi":            float,
        "by_popularity":  DataFrame,  人気帯×的中率
        "best_conditions": DataFrame, 的中が多い条件（venue, surface, distance）
        "miss_conditions": DataFrame, 外れが多い条件
    }
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)

    # 予想記録と買い目を結合
    query = """
    SELECT
        rr.race_date,
        rr.venue,
        rr.surface,
        rr.distance,
        rr.bias_type,
        rr.pace_predicted,
        fl.horse_name,
        fl.popularity,
        fl.confidence_score,
        fl.actual_rank,
        fl.hit_flag,
        b.amount,
        b.payout,
        b.is_hit
    FROM factor_log fl
    JOIN race_records rr ON fl.race_record_id = rr.id
    LEFT JOIN bets b ON b.race_record_id = rr.id
    WHERE fl.popularity >= 7
    ORDER BY rr.race_date DESC
    """
    try:
        df = pd.read_sql(query, conn)
    except Exception:
        conn.close()
        return {"total_bets": 0, "total_hits": 0, "hit_rate": 0.0, "roi": 0.0,
                "by_popularity": pd.DataFrame(), "best_conditions": pd.DataFrame(),
                "miss_conditions": pd.DataFrame()}
    conn.close()

    if df.empty:
        return {"total_bets": 0, "total_hits": 0, "hit_rate": 0.0, "roi": 0.0,
                "by_popularity": pd.DataFrame(), "best_conditions": pd.DataFrame(),
                "miss_conditions": pd.DataFrame()}

    # 重複排除（同一レース×同一馬）
    df = df.drop_duplicates(subset=["race_date", "horse_name"])

    total_bets = len(df)
    total_hits = int(df["hit_flag"].sum()) if "hit_flag" in df.columns else 0
    hit_rate   = total_hits / total_bets if total_bets > 0 else 0.0

    total_inv  = df["amount"].sum() if "amount" in df.columns else 0
    total_ret  = df["payout"].sum() if "payout" in df.columns else 0
    roi = total_ret / total_inv * 100 if total_inv > 0 else 0.0

    # 人気帯別集計
    df["pop_band"] = pd.cut(
        df["popularity"], bins=[6, 9, 12, 18],
        labels=["7〜9番人気", "10〜12番人気", "13番人気以上"]
    )
    by_pop = df.groupby("pop_band", observed=True).agg(
        bets=("hit_flag", "count"),
        hits=("hit_flag", "sum"),
    ).reset_index()
    by_pop["hit_rate%"] = (by_pop["hits"] / by_pop["bets"] * 100).round(1)

    # 的中が多い条件
    cond_cols = [c for c in ["venue", "surface", "distance"] if c in df.columns]
    if cond_cols and "hit_flag" in df.columns:
        best = df.groupby(cond_cols).agg(
            bets=("hit_flag", "count"),
            hits=("hit_flag", "sum"),
        ).reset_index()
        best = best[best["bets"] >= 3].copy()
        best["hit_rate%"] = (best["hits"] / best["bets"] * 100).round(1)
        best_conditions = best.nlargest(5, "hit_rate%")
        miss_conditions = best.nsmallest(5, "hit_rate%")
    else:
        best_conditions = pd.DataFrame()
        miss_conditions = pd.DataFrame()

    return {
        "total_bets":      total_bets,
        "total_hits":      total_hits,
        "hit_rate":        round(hit_rate, 3),
        "roi":             round(roi, 1),
        "by_popularity":   by_pop,
        "best_conditions": best_conditions,
        "miss_conditions": miss_conditions,
    }


def get_weekly_stats(weeks: int = 4) -> pd.DataFrame:
    """直近N週の週次ROI推移を返す。"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
    SELECT
        rr.race_date,
        b.amount,
        b.is_hit,
        b.payout
    FROM bets b
    JOIN race_records rr ON b.race_record_id = rr.id
    WHERE rr.race_date >= date('now', ?)
    """, conn, params=(f"-{weeks * 7} days",))
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["race_date"] = pd.to_datetime(df["race_date"])
    df["week"] = df["race_date"].dt.to_period("W").astype(str)
    weekly = df.groupby("week").agg(
        invested=("amount", "sum"),
        returned=("payout", "sum"),
        races=("amount", "count"),
        hits=("is_hit", "sum"),
    ).reset_index()
    weekly["roi"] = (weekly["returned"] / weekly["invested"] * 100).round(1)
    return weekly


def get_factor_accuracy() -> pd.DataFrame:
    """各ファクター（confidence_scoreの正確性）の精度集計を返す。"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
    SELECT
        confidence_score,
        actual_rank,
        hit_flag
    FROM factor_log
    WHERE actual_rank IS NOT NULL
    """, conn)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["score_bucket"] = pd.cut(df["confidence_score"],
                                bins=[0, 40, 55, 65, 75, 100],
                                labels=["0-40(✕)", "41-55(▲)", "56-65(△)", "66-75(○)", "76+(◎)"])
    stats = df.groupby("score_bucket", observed=True).agg(
        count=("hit_flag", "count"),
        hits=("hit_flag", "sum"),
    ).reset_index()
    stats["hit_rate"] = (stats["hits"] / stats["count"] * 100).round(1)
    return stats


def get_all_records() -> pd.DataFrame:
    """全レース記録を返す。"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
    SELECT
        rr.id, rr.race_date, rr.race_name, rr.venue, rr.surface, rr.distance,
        rr.confidence_top_score, rr.top_horse, rr.bias_type, rr.note,
        COALESCE(SUM(b.amount), 0) AS total_invested,
        COALESCE(SUM(b.payout), 0) AS total_returned,
        COALESCE(SUM(b.is_hit), 0) AS hits
    FROM race_records rr
    LEFT JOIN bets b ON b.race_record_id = rr.id
    GROUP BY rr.id
    ORDER BY rr.race_date DESC
    """, conn)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["roi"] = (df["total_returned"] / df["total_invested"].replace(0, np.nan) * 100).round(1)
    return df


# ============================================================
# 第13波: 改善ループ — 外れ要因の自動分類
# ============================================================

def get_failure_breakdown(since_days: int = 180) -> dict:
    """
    過去 N 日間の外れレースを 4 カテゴリに自動分類して集計を返す。

    分類ロジック:
      A. 当日要因   : winner_popularity >= 8 もしくは bias_type に「波乱」「不良」含む
      B. データ不足 : data_unavailable_count >= 1
      C. モデル崩れ : conformal_skip_recommended == True もしくは volatility_score >= 52
                     （荒れ場/読み違い — 第36波で interval_width 廃止・volatility 追加）
      D. 戦略ミス   : 上記いずれにも該当しない外れ（=本来当てるべきだったレース）

    Returns:
        {
          "total_races": int,
          "win_races": int,
          "lose_races": int,
          "categories": {"A": int, "B": int, "C": int, "D": int},
          "recommendations": [str, ...],
        }
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cutoff = (datetime.now() - pd.Timedelta(days=since_days)).strftime("%Y-%m-%d")
    df = pd.read_sql_query("""
        SELECT rr.id, rr.race_date, rr.bias_type, rr.auto_meta,
               COALESCE(SUM(b.amount), 0)  AS invested,
               COALESCE(SUM(b.payout), 0)  AS returned
        FROM race_records rr
        LEFT JOIN bets b ON b.race_record_id = rr.id
        WHERE rr.race_date >= ?
        GROUP BY rr.id
    """, conn, params=(cutoff,))
    conn.close()

    if df.empty:
        return {"total_races": 0, "win_races": 0, "lose_races": 0,
                "categories": {"A": 0, "B": 0, "C": 0, "D": 0},
                "recommendations": ["まだ予想レースの記録がありません。週末に1〜2レース予想して蓄積してください。"]}

    # 第20波 (U4): 買い目ゼロ（invested=0）のレースは勝敗集計から除外
    # （0 < 0 = False で「勝ち」扱いになり勝率が水増しされていた）
    df = df[df["invested"] > 0].copy()
    if df.empty:
        return {"total_races": 0, "win_races": 0, "lose_races": 0,
                "categories": {"A": 0, "B": 0, "C": 0, "D": 0},
                "recommendations": ["購入記録のあるレースがまだありません。"]}

    df["is_lose"] = df["returned"] < df["invested"]
    df["meta"] = df["auto_meta"].apply(lambda s: json.loads(s) if s else {})

    cat = {"A": 0, "B": 0, "C": 0, "D": 0}
    for _, row in df[df["is_lose"]].iterrows():
        m = row["meta"]
        bias = (row.get("bias_type") or "")
        win_pop = m.get("winner_popularity")
        if (win_pop is not None and win_pop >= 8) or "波乱" in bias or "不良" in bias:
            cat["A"] += 1
        elif m.get("data_unavailable_count", 0) >= 1:
            cat["B"] += 1
        elif m.get("conformal_skip_recommended") or (m.get("volatility_score") or 0) >= 52:
            # 第24波: interval_width 条件廃止（q_alpha=0.336 で常時True の死に条件だった）
            # 第36波: 「荒れ場（volatility>=52）で本命狙いして外した」も C（読み違い）に追加
            # — conformal の見送り推奨だけでは発火率が数%と低く C がほぼ死んでいたため
            cat["C"] += 1
        else:
            cat["D"] += 1

    total = len(df)
    win = int((~df["is_lose"]).sum())
    lose = total - win

    # 推奨アドバイス（カテゴリ最多に応じて）
    recs = []
    if lose >= 5:
        top_cat = max(cat, key=cat.get)
        share = cat[top_cat] / max(lose, 1) * 100
        msgs = {
            "A": f"外れの {share:.0f}% が当日要因（人気薄激走・馬場急変等）。これは構造的に困難なので、爆穴モードで EV>0.25 のみ厳選するか、堅軸モードに振り切るのが効果的です。",
            "B": f"外れの {share:.0f}% がデータ不足馬絡み。「データ不足」表示の馬は買わない、を徹底してください。",
            "C": f"外れの {share:.0f}% がモデル信頼度低の場面。Conformal 見送り推奨レースはスキップを徹底してください。",
            "D": f"外れの {share:.0f}% が「本来当てるべき」レース。EV閾値・Kelly倍率を見直すか、買い目を絞ってください。",
        }
        recs.append(msgs[top_cat])
    else:
        recs.append(f"記録 {total} レース（外れ {lose}）— サンプル蓄積中。30件超えると傾向が見えてきます。")

    return {
        "total_races": total,
        "win_races": win,
        "lose_races": lose,
        "categories": cat,
        "recommendations": recs,
    }


def update_auto_meta_with_result(race_record_id: int, winner_popularity: int | None):
    """
    レース結果取得後に、auto_meta に winner_popularity を追記する。
    save_result_to_diary から呼び出すフックポイント。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT auto_meta FROM race_records WHERE id=?", (race_record_id,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return
    meta = json.loads(row[0]) if row[0] else {}
    if winner_popularity is not None:
        meta["winner_popularity"] = int(winner_popularity)
    c.execute("UPDATE race_records SET auto_meta=? WHERE id=?",
              (json.dumps(meta, ensure_ascii=False), race_record_id))
    conn.commit()
    conn.close()
