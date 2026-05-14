"""
オッズ自動監視モジュール。

検出するシグナル:
1. 急落：短時間で20%以上のオッズ下落 → 大口資金の流入（情報馬の可能性）
2. 急騰：短時間で30%以上のオッズ上昇 → 陣営・関係者の見切り
3. 乖離：前日オッズと当日オッズの大きな差

データソース: netkeiba オッズページ（15分キャッシュ）
"""
import time
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import streamlit as st
from pathlib import Path
import json

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
ODDS_HISTORY_PATH = Path(__file__).parent / "odds_history.json"
MONITOR_INTERVAL = 300   # 5分ごとに取得
DROP_THRESHOLD   = 0.20  # 20%下落でアラート
SPIKE_THRESHOLD  = 0.30  # 30%上昇でアラート


# ============================================================
# オッズ取得
# ============================================================

def fetch_live_odds(race_id: str) -> dict[str, float]:
    """
    netkeiba の単勝オッズをリアルタイムで取得する。
    Returns: {horse_name: odds, ...}
    """
    url = f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={race_id}&type=1&action=update"
    try:
        time.sleep(2)
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = "utf-8"
        # APIレスポンスがJSONの場合
        if r.headers.get("content-type", "").startswith("application/json"):
            data = r.json()
            odds_map = {}
            for item in data.get("data", {}).get("odds", {}).get("1", []):
                horse_no = item.get("horse_num", "")
                odds_val = float(item.get("odds", 0))
                if odds_val > 0:
                    odds_map[horse_no] = odds_val
            return odds_map
    except Exception:
        pass

    # フォールバック: 出馬表ページのオッズ
    return fetch_odds_from_shutuba(race_id)


def fetch_odds_from_shutuba(race_id: str) -> dict[str, float]:
    """出馬表ページから単勝オッズを取得（フォールバック）"""
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    try:
        time.sleep(2)
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.content, "lxml")

        odds_map = {}
        for row in soup.select("tr.HorseList"):
            name_el = row.select_one(".HorseName a")
            odds_el = row.select_one(".Odds span.Odds")
            if not name_el:
                continue
            name = name_el.get_text(strip=True)
            if odds_el:
                try:
                    odds_map[name] = float(odds_el.get_text(strip=True))
                except Exception:
                    pass
        return odds_map
    except Exception:
        return {}


# ============================================================
# オッズ履歴の管理
# ============================================================

def load_odds_history() -> dict:
    if ODDS_HISTORY_PATH.exists():
        try:
            return json.loads(ODDS_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_odds_history(history: dict):
    ODDS_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def record_odds_snapshot(race_id: str, odds_map: dict[str, float]):
    """現在のオッズスナップショットを記録する。"""
    history = load_odds_history()
    if race_id not in history:
        history[race_id] = []

    history[race_id].append({
        "timestamp": datetime.now().isoformat(),
        "odds": odds_map,
    })
    # 直近20件のみ保持
    history[race_id] = history[race_id][-20:]
    save_odds_history(history)


# ============================================================
# シグナル検出
# ============================================================

def detect_odds_signals(race_id: str) -> list[dict]:
    """
    オッズ履歴から急変シグナルを検出する。
    Returns list of signal dicts.
    """
    history = load_odds_history()
    snapshots = history.get(race_id, [])
    if len(snapshots) < 2:
        return []

    latest = snapshots[-1]["odds"]
    prev   = snapshots[-2]["odds"]
    time_diff_min = _minutes_between(snapshots[-2]["timestamp"], snapshots[-1]["timestamp"])

    signals = []
    for name in latest:
        if name not in prev:
            continue
        curr_odds = latest[name]
        prev_odds = prev[name]
        if prev_odds <= 0:
            continue

        change_rate = (curr_odds - prev_odds) / prev_odds

        if change_rate <= -DROP_THRESHOLD:
            signals.append({
                "horse":      name,
                "type":       "急落",
                "emoji":      "⚡",
                "prev_odds":  prev_odds,
                "curr_odds":  curr_odds,
                "change_pct": round(change_rate * 100, 1),
                "time_min":   time_diff_min,
                "message":    f"**{name}** {prev_odds}→{curr_odds}倍（{change_rate*100:.0f}%）大口資金流入の可能性",
            })
        elif change_rate >= SPIKE_THRESHOLD:
            signals.append({
                "horse":      name,
                "type":       "急騰",
                "emoji":      "📈",
                "prev_odds":  prev_odds,
                "curr_odds":  curr_odds,
                "change_pct": round(change_rate * 100, 1),
                "time_min":   time_diff_min,
                "message":    f"**{name}** {prev_odds}→{curr_odds}倍（+{change_rate*100:.0f}%）関係者の見切り？",
            })

    return sorted(signals, key=lambda x: abs(x["change_pct"]), reverse=True)


def _minutes_between(ts1: str, ts2: str) -> float:
    try:
        t1 = datetime.fromisoformat(ts1)
        t2 = datetime.fromisoformat(ts2)
        return abs((t2 - t1).total_seconds() / 60)
    except Exception:
        return 0.0


def get_odds_change_table(race_id: str, horse_names: list[str]) -> pd.DataFrame:
    """
    監視中の全馬のオッズ推移テーブルを返す。
    """
    history = load_odds_history()
    snapshots = history.get(race_id, [])
    if not snapshots:
        return pd.DataFrame()

    rows = []
    latest = snapshots[-1]["odds"]
    first  = snapshots[0]["odds"]

    for name in horse_names:
        curr = latest.get(name)
        init = first.get(name)
        if curr is None:
            continue
        change = None
        if init and init > 0:
            change = round((curr - init) / init * 100, 1)
        rows.append({
            "馬名":         name,
            "現在オッズ":   curr,
            "初回オッズ":   init,
            "変化率(%)":    change,
            "スナップ数":   len(snapshots),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("変化率(%)")
    return df


# ============================================================
# Streamlit UI コンポーネント
# ============================================================

def render_odds_monitor_tab(race_id: str, horse_names: list[str]) -> None:
    """app.py から呼び出すオッズ監視タブ。"""
    st.subheader("📡 リアルタイム オッズ監視")
    st.caption("5分ごとに自動取得・急変を検知します（発走1〜2時間前から有効）")

    if not race_id:
        st.info("「レース詳細分析」タブでレースを選択してから使用してください。")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        auto_refresh = st.toggle("自動更新（5分ごと）", value=False, key="odds_auto_refresh")
    with col2:
        if st.button("🔄 今すぐ取得", key="fetch_odds_now"):
            _do_fetch_and_record(race_id)
            st.success("取得完了")

    if auto_refresh:
        st.info("自動更新ON: 5分後に再取得します")

    # Discord通知設定表示
    from discord_notify import render_discord_setup_section, _get_webhook_url
    discord_enabled = bool(_get_webhook_url())
    if discord_enabled:
        st.caption("🔔 Discord通知: ON（急落検出時に自動送信）")
    else:
        with st.expander("🔕 Discord通知を設定する"):
            render_discord_setup_section()

    # シグナル表示
    signals = detect_odds_signals(race_id)
    if signals:
        st.divider()
        st.subheader("🚨 オッズ急変アラート")
        for sig in signals:
            if sig["type"] == "急落":
                st.warning(f"{sig['emoji']} {sig['message']}")
            else:
                st.info(f"{sig['emoji']} {sig['message']}")
        # Discord に自動送信（未送信の場合のみ）
        sent_key = f"discord_sent_{race_id}_{len(signals)}"
        if discord_enabled and not st.session_state.get(sent_key):
            race_label = st.session_state.get("race_name_for_save", "")
            from discord_notify import notify_odds_alert
            if notify_odds_alert(signals, race_label):
                st.session_state[sent_key] = True
                st.toast("📡 Discord に通知を送信しました", icon="🔔")
    else:
        st.success("急変シグナルなし（正常）")

    # オッズ推移テーブル
    st.divider()
    st.subheader("📊 オッズ推移")
    change_df = get_odds_change_table(race_id, horse_names)
    if change_df.empty:
        st.info("まだデータがありません。「今すぐ取得」を押してください。")
    else:
        def color_change(val):
            try:
                v = float(val)
                if v <= -20: return "background-color: #c6efce; font-weight:bold"
                if v <= -10: return "background-color: #ffeb9c"
                if v >= 30:  return "background-color: #ffc7ce"
                return ""
            except Exception:
                return ""
        styled = change_df.style.map(color_change, subset=["変化率(%)"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # オッズ推移グラフ
        history = load_odds_history()
        snapshots = history.get(race_id, [])
        if len(snapshots) >= 2:
            import plotly.express as px
            plot_rows = []
            for snap in snapshots:
                ts = snap["timestamp"][:16]
                for h, o in snap["odds"].items():
                    if h in horse_names[:6]:  # 上位6頭のみ
                        plot_rows.append({"時刻": ts, "馬名": h, "オッズ": o})
            if plot_rows:
                import pandas as pd
                plot_df = pd.DataFrame(plot_rows)
                fig = px.line(plot_df, x="時刻", y="オッズ", color="馬名",
                              title="オッズ推移グラフ（上位6頭）")
                st.plotly_chart(fig, use_container_width=True)


def _do_fetch_and_record(race_id: str):
    with st.spinner("オッズを取得中..."):
        odds = fetch_live_odds(race_id)
    if odds:
        record_odds_snapshot(race_id, odds)
    else:
        st.warning("オッズ取得に失敗しました。時間をおいて再試行してください。")
