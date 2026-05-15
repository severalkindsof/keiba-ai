"""
競馬予想AI - 分析サポートツール v2
対象：JRA中央競馬 全レース（平場・重賞問わずEV優先）

起動方法: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from data_loader import (
    load_race_results,
    get_win_rate_table,
    get_sire_stats,
    get_jockey_stats,
    categorize_distance,
    merge_with_horse_cache,
    load_horse_cache,
    save_horse_cache,
)
from ev_calculator import evaluate_race
from horse_profiler import build_horse_profile
from bet_builder import build_tickets, format_tickets_for_display
from scraper import (
    fetch_today_races,
    fetch_race_entries,
    fetch_race_meta,
    fetch_horse_past_results,
    manual_entry_template,
)
from pace_analyzer import analyze_field_pace
from draw_bias import get_draw_bonus, get_draw_label, build_dynamic_draw_table
from jockey_change import get_jockey_change_for_field
from rotation_weight import analyze_rotation_for_field
from confluence import add_confluence_to_eval, get_race_quality_score
from race_selector import scan_weekend_races, get_this_weekend_dates, format_race_scan_display
from demo_data import generate_demo_race_results, get_demo_race_entries
from session_store import save_session, load_session, list_saved_sessions, save_scan_result, load_scan_result
# 新ファクターモジュール
from position_correction import apply_position_correction
from weight_handicap import apply_weight_handicap, eval_favorite_reliability
from nicks import build_nicks_table, apply_nicks
from season_climate import apply_season_climate
from race_level import build_race_level_table, apply_race_level_and_lap
from training_scraper import fetch_all_training, fetch_all_training_with_partner
from realtime_bias import apply_realtime_bias, render_bias_input_panel, BIAS_TYPES
from claude_chat import render_chat_tab
from race_diary import (
    init_db, save_race_prediction, fetch_race_result_from_netkeiba,
    save_result_to_diary, get_weekly_stats, get_factor_accuracy, get_all_records,
    generate_post_race_analysis,
)
from odds_monitor import render_odds_monitor_tab, _do_fetch_and_record
from knowledge_base import load_kb, save_kb, reload_kb
from discord_notify import (
    notify_analysis_complete, notify_bet_plan, notify_weekly_report,
    notify_post_race_analysis, render_discord_setup_section, _get_webhook_url,
)
from bankroll_manager import allocate_weekend_budget, render_bankroll_section
from longshot_evaluator import evaluate_all_longshots, get_top_structural_longshots
from bet_builder import get_bet_summary

# ---- ページ設定 ---- #
st.set_page_config(
    page_title="競馬予想AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏇 競馬予想AI v2 — 全レース対応・総合信頼スコア")
st.caption("JRA中央競馬専用 | 平場・重賞問わずEV期待値ベースで狙い目を分析")


# ============================================================
# バックテスト関数
# ============================================================

def _run_backtest(df: pd.DataFrame, wrt: pd.DataFrame) -> dict:
    if df.empty or wrt.empty:
        return {"ev_plus_rr": 0, "longshot_rr": 0, "ev_plus_pct": 0, "monthly_rr": pd.DataFrame()}
    required = ["odds", "win_flag", "popularity"]
    if not all(c in df.columns for c in required):
        return {"ev_plus_rr": 0, "longshot_rr": 0, "ev_plus_pct": 0, "monthly_rr": pd.DataFrame()}
    from ev_calculator import lookup_win_rate, calc_ev
    sample = df.dropna(subset=["odds", "win_flag", "popularity"]).copy()
    sample = sample[sample["odds"] > 1]
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=42)
    evs, returns = [], []
    for _, row in sample.iterrows():
        stats = lookup_win_rate(wrt, str(row.get("surface", "芝")),
                                int(row.get("distance", 2000)), int(row.get("popularity", 9)))
        ev = calc_ev(stats["win_rate"], float(row["odds"]))
        evs.append(-0.25 if np.isnan(ev) else ev)
        returns.append(float(row["odds"]) if int(row.get("win_flag", 0)) == 1 else -1.0)
    sample["ev_sim"] = evs
    sample["return"] = returns
    ev_plus = sample[sample["ev_sim"] > 0]
    longshots = sample[sample["popularity"] >= 10]
    ev_plus_rr = (ev_plus["return"].sum() / len(ev_plus) + 1) * 100 if len(ev_plus) > 0 else 0
    longshot_rr = (longshots["return"].sum() / len(longshots) + 1) * 100 if len(longshots) > 0 else 0
    ev_plus_pct = len(ev_plus) / len(sample) * 100 if len(sample) > 0 else 0
    monthly_rr = pd.DataFrame()
    if "date" in sample.columns:
        sample["month"] = pd.to_datetime(sample["date"], errors="coerce").dt.to_period("M").astype(str)
        ev_plus_m = sample[sample["ev_sim"] > 0]
        if not ev_plus_m.empty:
            monthly_rr = (
                ev_plus_m.groupby("month")
                .apply(lambda g: (g["return"].sum() / len(g) + 1) * 100)
                .reset_index(name="recovery_rate")
            )
    return {"ev_plus_rr": round(ev_plus_rr, 1), "longshot_rr": round(longshot_rr, 1),
            "ev_plus_pct": round(ev_plus_pct, 1), "monthly_rr": monthly_rr}


# ============================================================
# サイドバー
# ============================================================
with st.sidebar:
    st.header("⚙️ 設定")
    budget = st.number_input("1レースの予算（円）", min_value=500, max_value=10000, value=5000, step=500)
    target_date = st.date_input("対象日", value=date.today())
    date_str = target_date.strftime("%Y%m%d")

    # 金曜夜モード（枠順発表直後の分析フロー）
    from datetime import date as _date
    _today = _date.today()
    _is_friday = _today.weekday() == 4  # 4=金曜日
    friday_mode = st.toggle(
        "🌙 金曜夜モード（枠順発表後の分析）",
        value=_is_friday,
        help="金曜日の枠順発表後に使うモード。土日の全レースを枠順込みで一覧表示します。",
    )
    if friday_mode:
        st.info("🌙 金曜夜モード：翌土日のレースを先読みして予想の土台を作ります")

    st.divider()
    st.subheader("📂 過去データ")

    demo_mode = st.toggle("🎮 デモモード（Kaggleデータなしで動作確認）", value=False)
    if demo_mode:
        st.info("デモ用サンプルデータを使用中。実際の競馬データではありません。")
        with st.spinner("サンプルデータ生成中..."):
            df_hist = generate_demo_race_results(n_rows=3000)
        st.success(f"✅ デモデータ {len(df_hist):,}件")
    else:
        with st.spinner("過去データを読み込み中..."):
            df_hist = load_race_results()
        if df_hist.empty:
            st.error("data/ にCSVがありません。SETUP.md の STEP 3 を参照。または上の「デモモード」で動作確認できます。")
            st.stop()
        else:
            df_hist = merge_with_horse_cache(df_hist)
            st.success(f"✅ {len(df_hist):,}件のレース結果")

    with st.spinner("統計テーブルを構築中..."):
        win_rate_table  = get_win_rate_table(df_hist)
        sire_stats      = get_sire_stats(df_hist)
        jockey_stats    = get_jockey_stats(df_hist)
        draw_table      = build_dynamic_draw_table(df_hist)
        nicks_table     = build_nicks_table(df_hist)
        race_level_table = build_race_level_table(df_hist)

    st.divider()
    st.subheader("💾 前日セッション引継ぎ")
    saved_sessions = list_saved_sessions()
    if saved_sessions:
        session_options = {s["filename"]: s["path"] for s in saved_sessions}
        selected_session = st.selectbox(
            "保存済み分析を読込",
            ["（新規）"] + list(session_options.keys()),
        )
        if selected_session != "（新規）" and st.button("📂 読込"):
            loaded = load_session(session_options[selected_session])
            if loaded:
                st.session_state.update({
                    "eval_df": loaded["eval_df"],
                    "entries": loaded["entries"],
                    "surface": loaded["surface"],
                    "distance": loaded["distance"],
                    "venue": loaded["venue"],
                    "pace_info": loaded["pace_info"],
                })
                st.success(f"「{loaded['race_name']}」の分析を読み込みました")
    else:
        st.caption("保存済みセッションなし")

    st.divider()
    st.caption("📱 スマホURLをホーム画面に追加して使用")
    st.code("http://PCのIPアドレス:8501", language=None)

    st.divider()
    # Discord 通知設定（サイドバー）
    discord_ok = bool(_get_webhook_url())
    if discord_ok:
        st.success("🔔 Discord通知: ON")
    else:
        with st.expander("🔔 Discord通知を設定する"):
            render_discord_setup_section()


# ============================================================
# タブ構成
# ============================================================
init_db()  # DBを初期化（初回起動時のみテーブル作成）

(tab_home, tab_scan, tab_race, tab_horses, tab_bet,
 tab_chat, tab_odds, tab_diary, tab_backtest, tab_jockey, tab_kb) = st.tabs([
    "🏠 ホーム",
    "🔭 週末スキャン",
    "📋 レース分析",
    "🐎 馬プロファイル",
    "💰 馬券構成",
    "🤖 AI相談",
    "📡 オッズ監視",
    "📓 振り返り日記",
    "📊 バックテスト",
    "🏆 騎手・血統",
    "📖 メモ編集",
])


# ============================================================
# TAB 0: ホームページ（狙い目レース・資金配分）
# ============================================================
with tab_home:
    st.subheader("🏠 今日の競馬 ダッシュボード")
    st.caption("スコア65点以上の狙い目レースのみ表示 | オッズ急変・バイアスを一画面で把握")

    # ---- 上部バナー：アラートがあれば最優先表示 ---- #
    from odds_monitor import load_odds_history, detect_odds_signals as _det_sigs
    _all_odds_hist = load_odds_history()
    _urgent_signals = []
    for _rid, _ in _all_odds_hist.items():
        _sigs = _det_sigs(_rid)
        for _s in _sigs:
            if abs(_s["change_pct"]) >= 25:
                _urgent_signals.append(_s)
    if _urgent_signals:
        st.error(f"⚡ オッズ急変アラート {len(_urgent_signals)}件")
        for _s in _urgent_signals[:3]:
            st.warning(f"  {_s['emoji']} {_s['message']}")

    # ---- 週次ルールの設定と衝動買いカウンター ---- #
    with st.expander("📋 今週のマイルール設定（衝動買い防止）"):
        rule_max_races = st.number_input(
            "今週の最大購入レース数", min_value=1, max_value=10, value=3,
            key="rule_max_races",
        )
        rule_focus_day = st.selectbox(
            "集中する曜日", ["土日どちらも", "土曜のみ", "日曜のみ"],
            key="rule_focus_day",
        )
        rule_note = st.text_input(
            "今週の方針（一言）",
            placeholder="例：日曜の中山G2だけに集中。ハンデ戦は無視する。",
            key="rule_note",
        )
        if rule_note:
            st.success(f"マイルール設定済み：{rule_note}")

    purchased_count = len(st.session_state.get("purchased_races", []))
    if purchased_count > 0:
        st.info(f"📊 今週購入済み: {purchased_count}レース / ルール上限: {rule_max_races}レース")
        if purchased_count >= rule_max_races:
            st.error("⛔ 今週の購入上限に達しました。これ以上の購入はルール違反です。")

    st.divider()

    # ---- スキャンと狙い目レース表示 ---- #
    h_col1, h_col2, h_col3 = st.columns([2, 1, 1])
    with h_col1:
        weekly_budget_home = st.number_input(
            "週予算（円）", min_value=2000, max_value=50000, value=10000, step=1000,
            key="weekly_budget_home",
        )
    with h_col2:
        score_threshold = st.number_input(
            "表示閾値スコア", min_value=40, max_value=90, value=65,
            key="score_threshold",
            help="このスコア以上のレースのみ表示（65点推奨）",
        )
    with h_col3:
        max_races_home = st.number_input("スキャン上限", 5, 40, 20, key="home_max_races")

    weekend = get_this_weekend_dates()
    # 起動時にキャッシュから自動ロード
    if "home_scan_df" not in st.session_state:
        _cached_df, _cached_at, _cached_dates = load_scan_result()
        if not _cached_df.empty:
            st.session_state["home_scan_df"] = _cached_df
            st.session_state["home_scan_saved_at"] = _cached_at
            st.session_state["home_scan_dates"] = _cached_dates

    if st.button("🚀 今日の狙い目を更新", type="primary", use_container_width=True, key="home_scan"):
        _home_prog = st.empty()
        _home_err = st.container()
        home_scan_df = scan_weekend_races(
            weekend, win_rate_table, sire_stats, jockey_stats,
            max_races=max_races_home, progress_placeholder=_home_prog,
            error_container=_home_err,
        )
        st.session_state["home_scan_df"] = home_scan_df
        if not home_scan_df.empty:
            save_scan_result(home_scan_df, weekend)
            st.session_state["home_scan_saved_at"] = datetime.now().strftime("%m/%d %H:%M")
            allocs = allocate_weekend_budget(home_scan_df, weekly_budget_home)
            st.session_state["home_allocations"] = allocs

    scan_df_home = st.session_state.get("home_scan_df", pd.DataFrame())
    if not scan_df_home.empty:
        saved_at = st.session_state.get("home_scan_saved_at", "")
        if saved_at:
            st.caption(f"💾 最終スキャン: {saved_at}（ブラウザを閉じても保持されます）")
    if not scan_df_home.empty:
        # 閾値フィルター
        filtered = scan_df_home[scan_df_home["race_score"] >= score_threshold]
        if filtered.empty:
            st.info(f"スコア{score_threshold}点以上のレースがありません。上位3件を表示します。")
            filtered = scan_df_home.head(3)

        st.markdown(f"### 🎯 狙い目レース（スコア{score_threshold}点以上: {len(filtered)}件）")
        for i, (_, row) in enumerate(filtered.iterrows()):
            score = row.get("race_score", 0)
            ev_str = f"{row.get('top_ev', 0):+.3f}" if pd.notna(row.get("top_ev")) else "-"
            bg = "#f0fff0" if score >= 75 else "#fffbe6" if score >= 65 else "#fff5f5"

            with st.container():
                c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
                verdict_emoji = "◎" if score >= 75 else "○"
                c1.markdown(
                    f"**{verdict_emoji} {row.get('race_name','')}**  "
                    f"`{row.get('date_str','')}` {row.get('surface','')} {row.get('distance','')}m"
                )
                c2.metric("スコア", score)
                c3.metric("注目馬", row.get("top_horse", "-"))
                c4.metric("EV", ev_str)

                # 衝動買い危険ラベル
                rname = str(row.get("race_name", ""))
                if any(kw in rname for kw in ["ハンデ", "ハンデキャップ"]):
                    st.warning("⚠️ ハンデ戦：衝動買い注意レース")
                if st.button(f"このレースを詳細分析", key=f"home_detail_{i}"):
                    st.session_state["preselected_race_id"] = str(row.get("race_id", ""))
                    st.info("「レース分析」タブへ移動して分析スタートを押してください")

        # 資金配分
        allocs = st.session_state.get("home_allocations", [])
        if allocs:
            st.divider()
            st.markdown("### 💴 週予算の最適配分")
            render_bankroll_section(allocs, weekly_budget_home)
    else:
        st.info("「今日の狙い目を更新」を押すと、スコア65点以上のレースのみ絞り込んで表示します。")

    # ---- 金曜夜モード：週末レースの先読みビュー ---- #
    if friday_mode:
        st.divider()
        st.markdown("### 🌙 金曜夜モード：翌週末の先読み分析")
        st.caption("枠順が確定したら、土日全レースを枠順込みでスキャンして予想の土台を作ります")

        fri_col1, fri_col2 = st.columns(2)
        with fri_col1:
            friday_weekly_budget = st.number_input(
                "今週末の予算", 2000, 30000, 10000, 1000, key="friday_budget"
            )
        with fri_col2:
            friday_races_limit = st.number_input(
                "今週買うレース数の上限", 1, 5, 3, key="friday_race_limit"
            )

        if st.button("📋 枠順発表後スキャン開始", type="primary", key="friday_scan"):
            _fri_prog = st.empty()
            fri_scan_df = scan_weekend_races(
                get_this_weekend_dates(), win_rate_table, sire_stats,
                jockey_stats, max_races=24, progress_placeholder=_fri_prog
            )
            st.session_state["friday_scan_df"] = fri_scan_df
            if not fri_scan_df.empty:
                save_scan_result(fri_scan_df, get_this_weekend_dates())
                st.session_state["home_scan_df"] = fri_scan_df  # ホームタブにも反映
                st.session_state["home_scan_saved_at"] = datetime.now().strftime("%m/%d %H:%M")
                fri_allocs = allocate_weekend_budget(
                    fri_scan_df, friday_weekly_budget, max_races=friday_races_limit
                )
                st.session_state["friday_allocs"] = fri_allocs

        fri_df = st.session_state.get("friday_scan_df", pd.DataFrame())
        if not fri_df.empty:
            fri_allocs = st.session_state.get("friday_allocs", [])
            st.success(f"スキャン完了: {len(fri_df)}レース")

            # 注目レースカード
            st.markdown("#### 今週末の注目レース（上位5件）")
            for i, (_, row) in enumerate(fri_df.head(5).iterrows()):
                score = row.get("race_score", 0)
                with st.expander(
                    f"#{i+1} {row.get('race_name','')} — スコア {score}点  |  "
                    f"注目馬: {row.get('top_horse','-')}",
                    expanded=(i == 0),
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("日付", row.get("date_str", ""))
                    c2.metric("コース", f"{row.get('surface','')} {row.get('distance','')}m")
                    c3.metric("EV+馬数", row.get("ev_plus_count", 0))
                    if st.button(f"詳細分析へ", key=f"fri_detail_{i}"):
                        st.session_state["preselected_race_id"] = str(row.get("race_id", ""))
                        st.info("「レース分析」タブへ移動してください")

            # 資金配分
            if fri_allocs:
                st.markdown("#### 今週末の推奨資金配分")
                render_bankroll_section(fri_allocs, friday_weekly_budget)

            # 今週のメモ欄
            st.markdown("#### 📝 今週の予想メモ（金曜時点）")
            friday_memo = st.text_area(
                "気になる馬・展開予想・注意点など",
                height=100,
                placeholder="例：東京11Rの○○は叩き2走目でオッズが妙に高い。前走はハイペースで潰れただけ。枠は内寄りで良さそう。",
                key="friday_memo",
            )
            if friday_memo:
                st.session_state["friday_memo"] = friday_memo
                st.success("メモ保存済み（当日アプリを開いた時に表示されます）")

    # 金曜メモを当日に表示
    elif st.session_state.get("friday_memo"):
        st.divider()
        st.markdown("### 📝 前日（金曜）のメモ")
        st.info(st.session_state["friday_memo"])

    # ---- 掲示板テキスト解析（2ch/5ch/ハロン棒ch） ---- #
    st.divider()
    st.markdown("### 📋 掲示板・SNS情報の解析（ハロン棒ch / netkeiba掲示板）")
    st.caption("X・2ch・netkeibaから気になった投稿をコピペするとAIが馬場バイアスや展開予想を抽出します")

    board_text = st.text_area(
        "掲示板・SNS投稿をここに貼り付け",
        height=150,
        placeholder=(
            "例（ハロン棒chから）:\n"
            "「今日の東京は完全に内前有利。4コーナーで外に出すと届かない。\n"
            "先行馬が残りまくっている。展開は超スローになりそう。\n"
            "ディープ系の瞬発力型が合うコンディション。」"
        ),
        key="board_text_input",
    )

    if board_text and len(board_text.strip()) > 20:
        from realtime_bias import parse_bias_from_text
        from pace_analyzer import PACE_STYLE_BENEFIT

        parsed_bias = parse_bias_from_text(board_text)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if parsed_bias["confidence"] > 0:
                st.success(
                    f"🌿 **バイアス検出**: {parsed_bias.get('label', '')} "
                    f"（信頼度 {parsed_bias['confidence']}%）\n\n"
                    f"キーワード: {', '.join(parsed_bias.get('matched_keywords', [])[:5])}"
                )
                st.session_state["board_bias_type"] = parsed_bias["bias_type"]
            else:
                st.info("バイアスキーワードが検出されませんでした")

        with col_b2:
            # Claude APIで掲示板テキストをより詳細に解析
            if st.button("🤖 Claudeに詳しく解析させる", key="board_claude"):
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    st.warning("Claude APIキーを設定してください")
                else:
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)
                        with st.spinner("解析中..."):
                            resp = client.messages.create(
                                model="claude-haiku-4-5",
                                max_tokens=400,
                                messages=[{"role": "user", "content":
                                    f"以下の競馬掲示板の投稿から、馬場バイアス・展開予想・注目馬の情報を"
                                    f"箇条書き3点以内で簡潔に抽出してください：\n\n{board_text[:800]}"}],
                            )
                        st.info(f"📝 Claude解析結果:\n\n{resp.content[0].text}")
                    except Exception as e:
                        st.error(f"API呼び出しエラー: {e}")

# ============================================================
# TAB 1: 週末レーススキャン（詳細）
# ============================================================
with tab_scan:
    st.subheader("🔭 週末全レーススキャン")
    st.caption("JRA全レース（平場含む）をスキャンして期待値が高いレースをランキング表示します")

    weekend_dates = get_this_weekend_dates()
    st.info(f"スキャン対象：{' / '.join(weekend_dates)}")

    # 起動時・タブ切替時: ファイルキャッシュからスキャン結果を復元
    if "scan_df" not in st.session_state:
        _c_df, _c_at, _c_dates = load_scan_result()
        if not _c_df.empty:
            st.session_state["scan_df"] = _c_df
            st.session_state["scan_saved_at"] = _c_at

    scan_mode = st.radio(
        "スキャン範囲",
        ["🏆 重賞・9R以降のみ（速い）", "📋 全レース（遅い・36レース）"],
        horizontal=True,
    )

    col_l, col_r = st.columns([2, 1])
    with col_l:
        custom_dates = st.text_input(
            "日付を変更する場合（カンマ区切り YYYYMMDD）",
            value=",".join(weekend_dates),
        )
    with col_r:
        max_races = st.number_input("スキャン上限レース数", min_value=3, max_value=50,
                                    value=10 if "重賞" in scan_mode else 24)

    # 接続診断ボタン
    with st.expander("🔧 接続テスト（うまく動かない場合）"):
        if st.button("netkeibaへの接続を確認", key="diag_btn"):
            import requests as _req
            from bs4 import BeautifulSoup as _BS
            import re as _re
            _diag = st.empty()
            _test_date = weekend_dates[1]  # 日曜

            # ① レース一覧ページ
            try:
                _r1 = _req.get(
                    f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={_test_date}",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
                )
                _soup1 = _BS(_r1.content, "lxml")
                _links = _soup1.select("a[href*='shutuba.html'][href*='race_id']")
                _ids = list({_re.search(r"race_id=(\d{12})", a.get("href","")).group(1)
                             for a in _links if _re.search(r"race_id=(\d{12})", a.get("href",""))})
                st.success(f"✅ レース一覧：{len(_ids)}件取得")
            except Exception as _e:
                st.error(f"❌ レース一覧：失敗 → {_e}")
                _ids = []

            # ② 出馬表ページ（1レースだけ試す）
            if _ids:
                try:
                    import time as _t
                    _t.sleep(2)
                    _r2 = _req.get(
                        f"https://race.netkeiba.com/race/shutuba.html?race_id={_ids[0]}",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
                    )
                    _soup2 = _BS(_r2.content, "lxml")
                    _horses = _soup2.select("tr.HorseList")
                    st.success(f"✅ 出馬表（{_ids[0]}）：{len(_horses)}頭取得")
                except Exception as _e:
                    st.error(f"❌ 出馬表：失敗 → {_e}")

            # ③ オッズAPI
            if _ids:
                try:
                    _t.sleep(2)
                    _r3 = _req.get(
                        f"https://race.netkeiba.com/api/api_get_jra_odds.html?race_id={_ids[0]}&type=1&action=update",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
                    )
                    _d3 = _r3.json()
                    st.success(f"✅ オッズAPI：status={_d3.get('status')}")
                except Exception as _e:
                    st.error(f"❌ オッズAPI：失敗 → {_e}")

            # ④ fetch_race_entries（実際のスキャンで使う関数）を直接テスト
            if _ids:
                try:
                    fetch_race_entries.clear()
                    _entries = fetch_race_entries(_ids[0])
                    if _entries:
                        st.success(f"✅ fetch_race_entries：{len(_entries)}頭取得 (オッズ例: {_entries[0].get('odds')})")
                    else:
                        st.error(f"❌ fetch_race_entries：0件（ここが原因）")
                except Exception as _e:
                    st.error(f"❌ fetch_race_entries：例外発生 → {_e}")

    if st.button("🚀 スキャン開始", type="primary", use_container_width=True):
        scan_dates = [d.strip() for d in custom_dates.split(",") if d.strip()]
        fetch_today_races.clear()
        fetch_race_entries.clear()
        _scan_prog = st.empty()
        _scan_err = st.container()  # エラー詳細専用（上書きされない）

        # 重賞モードの場合は後半レース(9R以降)に相当するIDのみ対象
        _race_filter = None
        if "重賞" in scan_mode:
            # netkeibaのrace_idはYYYYVVRRNN形式。NN(レース番号)が09以上 = 9R以降
            def _race_filter(r):
                try:
                    return int(r["race_id"][-2:]) >= 9
                except Exception:
                    return True

        scan_df = scan_weekend_races(
            scan_dates, win_rate_table, sire_stats, jockey_stats,
            max_races=max_races, progress_placeholder=_scan_prog,
            race_filter=_race_filter, error_container=_scan_err,
        )
        st.session_state["scan_df"] = scan_df
        if not scan_df.empty:
            from datetime import datetime as _dt
            save_scan_result(scan_df, scan_dates)
            st.session_state["scan_saved_at"] = _dt.now().strftime("%m/%d %H:%M")

    if "scan_df" in st.session_state:
        scan_df = st.session_state["scan_df"]
        if scan_df.empty:
            st.warning("スキャン結果が空です。上の「接続テスト」でnetkeibaへの接続を確認してください。")
        else:
            _saved_at = st.session_state.get("scan_saved_at", "")
            if _saved_at:
                st.caption(f"💾 最終スキャン: {_saved_at}（ブラウザを閉じても保持）")
            st.success(f"{len(scan_df)}レースをスキャン完了")

            # 上位レース表示
            display_scan = format_race_scan_display(scan_df)
            def color_verdict(val):
                if "◎" in str(val):
                    return "background-color: #c6efce"
                elif "○" in str(val):
                    return "background-color: #ffeb9c"
                elif "✕" in str(val):
                    return "background-color: #ffc7ce"
                return ""
            styled_scan = display_scan.style.map(color_verdict, subset=["判定"])
            st.dataframe(styled_scan, use_container_width=True, hide_index=True)

            # クリックで詳細分析へ飛ぶ
            top_races = scan_df[scan_df["race_score"] >= 55]["race_id"].tolist()
            if top_races:
                selected_id = st.selectbox(
                    "詳細分析するレースを選ぶ",
                    options=scan_df["race_id"].tolist(),
                    format_func=lambda x: scan_df[scan_df["race_id"] == x]["race_name"].iloc[0]
                    + f" (スコア:{scan_df[scan_df['race_id']==x]['race_score'].iloc[0]})",
                )
                if st.button("このレースを詳細分析タブで開く"):
                    st.session_state["preselected_race_id"] = selected_id
                    st.info("「レース詳細分析」タブに移動して「自動取得」を押してください")

            # スコア分布グラフ
            fig = px.bar(
                scan_df.sort_values("race_score", ascending=True).tail(15),
                x="race_score", y="race_name",
                orientation="h",
                color="race_score",
                color_continuous_scale=["#ffc7ce", "#ffeb9c", "#c6efce"],
                title="レース別 狙い目スコア（上位15）",
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 1: レース詳細分析
# ============================================================
with tab_race:
    st.subheader("📋 レース詳細分析")

    fetch_mode = st.radio(
        "データ取得方法",
        ["🌐 netkeibaから自動取得", "✏️ 手動入力"],
        horizontal=True,
    )

    races = []
    if fetch_mode == "🌐 netkeibaから自動取得":
        # スキャンタブからの引き継ぎ
        preselected = st.session_state.get("preselected_race_id")

        # 今日のレースがなければ週末日付を順に試す
        _dates_to_try = list(dict.fromkeys([date_str] + get_this_weekend_dates()))
        with st.spinner("レース一覧を取得中..."):
            for _d in _dates_to_try:
                races = fetch_today_races(_d)
                if races:
                    break
        if not races:
            st.warning("レース情報を取得できませんでした。手動入力に切り替えてください。")
            fetch_mode = "✏️ 手動入力"

    entries, surface, distance, track_condition, venue = [], "芝", 2000, "良", "東京"

    if fetch_mode == "🌐 netkeibaから自動取得" and races:
        # 会場コード→会場名マッピング
        _VENUE_MAP = {
            "01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
            "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉",
        }
        # レースを会場別グループに分類
        _venue_groups: dict[str, list] = {}
        for r in races:
            _vc = r["race_id"][4:6]
            _vname = _VENUE_MAP.get(_vc, f"会場{_vc}")
            _venue_groups.setdefault(_vname, []).append(r)

        # 会場を選んでからレースを選ぶ2段階UI
        _venue_names = list(_venue_groups.keys())
        _default_venue_idx = 0
        if preselected:
            for vi, (vn, vr) in enumerate(_venue_groups.items()):
                if any(r["race_id"] == preselected for r in vr):
                    _default_venue_idx = vi
                    break
        selected_venue = st.selectbox("会場を選択", _venue_names, index=_default_venue_idx)
        _venue_races = _venue_groups[selected_venue]
        race_options = {f"{r['race_name']}": r["race_id"] for r in _venue_races}
        default_idx = 0
        if preselected and preselected in race_options.values():
            ids = list(race_options.values())
            default_idx = ids.index(preselected)
        selected_label = st.selectbox("レースを選択", list(race_options.keys()), index=default_idx)
        selected_race_id = race_options[selected_label]

        with st.spinner("出走馬・オッズを取得中..."):
            entries = fetch_race_entries(selected_race_id)
            meta = fetch_race_meta(selected_race_id)
        surface = meta.get("surface", "芝")
        distance = meta.get("distance", 2000)
        track_condition = meta.get("track_condition", "良")
        st.info(f"**{surface} {distance}m / 馬場: {track_condition}**")

        # Kaggleデータが欠落している馬の過去成績を自動取得
        # ① 完全にデータなし（2022年以降デビュー馬）
        # ② Kaggleにデータはあるが最終成績が2021年以前（現役馬の2022〜2026成績が欠落）
        _KAGGLE_CUTOFF = pd.Timestamp("2022-01-01")
        missing = []
        _has_horse_name = not df_hist.empty and "horse_name" in df_hist.columns
        _horses_with_id = [e for e in entries if e.get("horse_id")]
        st.caption(f"🔍 デバッグ: 出走馬{len(entries)}頭 / horse_id取得済み{len(_horses_with_id)}頭 / df_hist {len(df_hist)}行 / horse_name列: {_has_horse_name}")
        for _e in entries:
            if not _e.get("horse_id"):
                continue
            if not _has_horse_name:
                missing.append(_e)
                continue
            _hist_e = df_hist[df_hist["horse_name"] == _e["horse_name"]]
            if _hist_e.empty:
                missing.append(_e)
            elif "date" in _hist_e.columns:
                _last = _hist_e["date"].max()
                if pd.isna(_last) or pd.Timestamp(_last) < _KAGGLE_CUTOFF:
                    missing.append(_e)
        if missing:
            st.info(f"📡 {len(missing)}頭の過去成績をnetkeibaから取得中... (db.netkeiba.com)")
            _fetch_prog = st.empty()
            fetched_count = 0
            fail_count = 0
            for _fi, e in enumerate(missing):
                hid = e["horse_id"]
                hname = e["horse_name"]
                _fetch_prog.caption(f"取得中 {_fi+1}/{len(missing)}: {hname} (ID:{hid})")
                cached = load_horse_cache(hid)
                if cached is not None:
                    new_df = cached
                else:
                    new_df = fetch_horse_past_results(hid, hname)
                    if not new_df.empty:
                        save_horse_cache(hid, hname, new_df)
                    else:
                        fail_count += 1
                if new_df.empty:
                    continue
                    # Kaggleデータが既にある馬は2022年以降のみ追加（重複防止）
                    # new_df はnetkeibaの全成績（デビューから現在まで）を含むため
                    if "horse_name" in df_hist.columns:
                        has_kaggle = not df_hist[df_hist["horse_name"] == hname].empty
                    else:
                        has_kaggle = False
                    if has_kaggle and "date" in new_df.columns:
                        rows_to_add = new_df[new_df["date"] >= _KAGGLE_CUTOFF]
                    else:
                        rows_to_add = new_df
                    if not rows_to_add.empty:
                        df_hist = pd.concat([df_hist, rows_to_add], ignore_index=True)
                        fetched_count += 1
            _fetch_prog.empty()
            if fetched_count:
                st.success(f"✅ {fetched_count}頭の過去成績を取得しました（netkeiba）")
            if fail_count:
                st.warning(f"⚠️ {fail_count}頭の取得失敗（db.netkeiba.com への接続エラーの可能性）")
                # 取得データを反映した統計テーブルを再計算（df_hist が変わるのでキャッシュミスになる）
                win_rate_table = get_win_rate_table(df_hist)
                sire_stats     = get_sire_stats(df_hist)
                jockey_stats   = get_jockey_stats(df_hist)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            surface = st.selectbox("馬場", ["芝", "ダート"])
        with c2:
            distance = st.number_input("距離（m）", min_value=800, max_value=3600, value=2000, step=100)
        with c3:
            track_condition = st.selectbox("馬場状態", ["良", "稍重", "重", "不良"])
        with c4:
            venue = st.selectbox("会場", ["東京", "中山", "阪神", "京都", "中京", "新潟", "福島", "小倉", "函館", "札幌"])

        st.markdown("#### 出走馬リスト")
        n_horses = st.number_input("出走頭数", min_value=2, max_value=18, value=8)
        entries = []
        for i in range(int(n_horses)):
            with st.expander(f"馬{i+1}", expanded=(i < 2)):
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                name  = c1.text_input("馬名", key=f"name_{i}",  value=f"馬{i+1}")
                odds  = c2.number_input("単勝オッズ", key=f"odds_{i}", min_value=1.0, value=float((i+1)*5), step=0.1)
                pop   = c3.number_input("人気", key=f"pop_{i}",  min_value=1, max_value=18, value=i+1)
                gate  = c4.number_input("枠番", key=f"gate_{i}", min_value=1, max_value=18, value=i+1)
                jockey = c5.text_input("騎手", key=f"jockey_{i}")
                sire   = c6.text_input("父", key=f"sire_{i}")
                weight = c7.number_input("体重(kg)", key=f"weight_{i}", min_value=380, max_value=600, value=480)
                entries.append({
                    "horse_no": str(i+1), "horse_name": name,
                    "odds": odds, "popularity": pop, "gate": gate,
                    "jockey": jockey, "sire": sire, "horse_weight": weight,
                })

    # 共通フィールド補完
    current_date_str = target_date.strftime("%Y-%m-%d")
    dist_cat = categorize_distance(distance)
    total_horses = len(entries)
    for e in entries:
        e.setdefault("surface", surface)
        e.setdefault("distance", distance)
        e.setdefault("track_condition", track_condition)
        e.setdefault("venue", venue)
        e.setdefault("distance_cat", dist_cat)
        e.setdefault("race_date", current_date_str)

    # ---- 当日バイアス入力 ---- #
    selected_race_id_for_bias = st.session_state.get("preselected_race_id", "")
    bias_type = render_bias_input_panel(selected_race_id_for_bias or None)

    # ---- 分析実行 ---- #
    if st.button("🔍 分析スタート", type="primary", use_container_width=True):
        with st.spinner("全14ファクターを分析中..."):
            # 1. ペース・展開
            pace_info, entries_p = analyze_field_pace(entries, df_hist)
            predicted_pace = pace_info.get("predicted_pace", "ミドル")

            # 2. 騎手乗り替わり
            entries_jc = get_jockey_change_for_field(entries_p, df_hist, jockey_stats)

            # 3. ローテ・体重・クラス
            entries_rot = analyze_rotation_for_field(entries_jc, df_hist, current_date_str)

            # 4. 枠順バイアス
            for e in entries_rot:
                gate = int(e.get("gate", e.get("horse_no", 1)))
                e["draw_bonus"] = get_draw_bonus(
                    draw_table, e.get("venue", venue), surface, dist_cat, gate, total_horses)
                e["draw_label"] = get_draw_label(
                    e.get("venue", venue), surface, dist_cat, gate, total_horses)

            # 5. 前走位置取り補正（NEW）
            entries_pos = apply_position_correction(entries_rot, df_hist)

            # 6. 斤量馬体重比（NEW）
            entries_wh = apply_weight_handicap(entries_pos, distance)

            # 7. ニックス（NEW）
            entries_nicks = apply_nicks(entries_wh, nicks_table, surface, dist_cat)

            # 8. 季節・馬場状態適性（NEW）
            entries_season = apply_season_climate(
                entries_nicks, df_hist, surface, track_condition, target_date)

            # 9. 前走レースレベル・ラップ適性（NEW）
            entries_level = apply_race_level_and_lap(
                entries_season, df_hist, race_level_table, predicted_pace)

            # 10. リアルタイムバイアス（NEW）
            entries_full_bias = apply_realtime_bias(entries_level, bias_type)

            # 11. 上がり3F適性・同会場距離実績・時計ランク・馬体重トレンド（NEW）
            from horse_profiler import (calc_last3f_rank, calc_venue_distance_aptitude,
                                        calc_time_rank, get_weight_trend)
            from weight_handicap import analyze_handicap_weight_trend
            race_name_current = st.session_state.get("_race_label", "")
            for e in entries_full_bias:
                name = e.get("horse_name", "")
                # 上がり3F
                l3f = calc_last3f_rank(df_hist, name, surface, distance)
                e["last3f_label"]  = l3f["label"]
                e["last3f_bonus"]  = l3f["bonus"]
                e["last3f_avg"]    = l3f["avg_3f"]
                e["last3f_value"]  = l3f["last_3f"]
                # 会場×距離実績
                vda = calc_venue_distance_aptitude(df_hist, name, venue, surface, distance)
                e["venue_apt_score"]    = vda["score"]
                e["venue_apt_detail"]   = vda["detail"]
                e["venue_apt_is_exact"] = vda.get("is_venue_exact", False)
                # 時計ランク
                tr = calc_time_rank(df_hist, name, venue, surface, distance)
                e["time_rank"]       = tr["time_rank"]
                e["time_rank_bonus"] = tr["time_rank_bonus"]
                e["time_rank_label"] = tr["label"]
                # 馬体重トレンド
                wt = get_weight_trend(df_hist, name, n=10)
                e["weight_trend_label"] = wt["label"]
                e["weight_trend_bonus"] = wt["bonus"]
                e["weight_last_change"] = wt["last_change"]
                # ハンデ斤量
                wc = e.get("weight_carried")
                try:
                    wc_f = float(wc) if wc else None
                except Exception:
                    wc_f = None
                hc = analyze_handicap_weight_trend(df_hist, name, wc_f, race_name_current)
                e["handicap_trend_label"] = hc["label"]
                e["handicap_trend_bonus"] = hc["bonus"]

                # 右回り/左回り適性
                from horse_profiler import (calc_turn_aptitude,
                                            check_first_time_conditions,
                                            calc_stable_recent_form,
                                            check_field_size_change,
                                            check_closing_pace_fit)
                ta = calc_turn_aptitude(df_hist, name, venue, surface)
                e["turn_dir_bonus"]    = ta["bonus"]
                e["turn_dir_label"]    = ta["label"]
                e["turn_is_mismatch"]  = ta["is_mismatch"]
                e["turn_preferred"]    = ta["preferred"]

                # 初距離・初馬場
                ft = check_first_time_conditions(df_hist, name, surface, distance)
                e["first_time_bonus"]       = ft["bonus"]
                e["first_time_label"]       = ft["label"]
                e["is_first_surface"]       = ft["is_first_surface"]
                e["is_first_distance"]      = ft["is_first_distance"]

                # 厩舎近況
                sf = calc_stable_recent_form(df_hist, e.get("trainer", ""),
                                              race_date=str(target_date))
                e["stable_bonus"]  = sf["bonus"]
                e["stable_label"]  = sf["label"]
                e["stable_trend"]  = sf["trend"]

                # 頭数変化
                fc = check_field_size_change(df_hist, name, total_horses, surface)
                e["field_size_bonus"] = fc["bonus"]
                e["field_size_label"] = fc["label"]
                e["field_size_diff"]  = fc["diff"]

                # 前走上がり3F × ペース適合
                predicted_pace_str = pace_info.get("predicted_pace", "") if pace_info else ""
                cpf = check_closing_pace_fit(df_hist, name, predicted_pace_str, surface, distance)
                e["pace_fit_bonus"]  = cpf["bonus"]
                e["pace_fit_label"]  = cpf["label"]
                e["pace_shift"]      = cpf["pace_shift"]

                # 短期外国人騎手チェック
                from knowledge_base import get_short_term_foreign_jockey_bonus
                stf = get_short_term_foreign_jockey_bonus(
                    e.get("jockey", ""), e.get("race_class", "")
                )
                e["short_term_foreign_bonus"] = stf["bonus"]
                e["short_term_foreign_note"]  = stf["note"]
                e["is_short_term_foreign"]     = stf["is_short_term"]

            entries_full = entries_full_bias

            # EV評価 → 全補正値が horse dict に入っているのでそのまま渡す
            eval_df = evaluate_race(entries_full, win_rate_table, sire_stats, jockey_stats)

            # 有力馬撃破スコア + 近走巻き返しボーナスをパイプラインに付与
            from horse_profiler import (calc_beaten_strong_horses,
                                        analyze_recent_races,
                                        calc_resume_bonus_from_recent)
            for e in entries_full:
                name = e.get("horse_name", "")
                beaten = calc_beaten_strong_horses(df_hist, name, surface, distance)
                e["beat_count"]       = beaten["beat_count"]
                e["beat_bonus"]       = beaten["bonus"]
                e["beat_label"]       = beaten["label"]
                e["beat_best_victim"] = beaten["best_victim"]

                # 近走詳細 → 巻き返しボーナス
                recent = analyze_recent_races(
                    df_hist, name,
                    current_surface=surface,
                    current_distance=distance,
                    n=3,
                )
                rb = calc_resume_bonus_from_recent(recent)
                e["resume_bonus_total"]   = rb["total_bonus"]
                e["resume_summary"]       = rb["summary"]
                e["resume_top_excuse"]    = rb["top_excuse"]

                # VM格言チェック（前走距離を参照）
                from knowledge_base import get_proverb_bonus
                hist_e = df_hist[df_hist["horse_name"] == name].sort_values("date", ascending=False) \
                    if "date" in df_hist.columns else pd.DataFrame()
                prev_dist_e = int(pd.to_numeric(
                    hist_e.iloc[0].get("distance", 0), errors="coerce") or 0) \
                    if not hist_e.empty else 0
                prev_rank_e = int(pd.to_numeric(
                    hist_e.iloc[0].get("rank", 99), errors="coerce") or 99) \
                    if not hist_e.empty else 99
                prov = get_proverb_bonus(
                    race_name=st.session_state.get("_race_label", ""),
                    prev_distance=prev_dist_e,
                    prev_rank=prev_rank_e,
                )
                e["proverb_bonus"] = prov["bonus"]
                e["proverb_label"] = prov["label"]

            # 調教タイム + 併せ馬パートナー（race_idがある場合のみ試行）
            if selected_race_id_for_bias and not demo_mode:
                # 日曜レースなら前日土曜の日付を計算
                from datetime import timedelta
                _race_dt = target_date
                _sat_str = (_race_dt - timedelta(days=1)).strftime("%Y%m%d") \
                    if _race_dt.weekday() == 6 else ""  # 日曜=6

                spinner_msg = "調教タイム＋併せ馬を取得中（失敗しても分析は続きます）..."
                with st.spinner(spinner_msg):
                    training_results = fetch_all_training_with_partner(
                        selected_race_id_for_bias,
                        eval_df["horse_name"].tolist(),
                        saturday_date_str=_sat_str,
                    )
                    for _, row in eval_df.iterrows():
                        name = row["horse_name"]
                        if name in training_results:
                            tr = training_results[name]
                            eval_df.loc[eval_df["horse_name"] == name, "training_score"]   = tr.get("score", 50)
                            eval_df.loc[eval_df["horse_name"] == name, "training_label"]   = tr.get("label", "")
                            eval_df.loc[eval_df["horse_name"] == name, "training_bonus"]   = tr.get("bonus", 0.0)
                            eval_df.loc[eval_df["horse_name"] == name, "partner_name"]     = tr.get("partner_name", "")
                            eval_df.loc[eval_df["horse_name"] == name, "partner_won_sat"]  = tr.get("partner_won_sat", False)
                            eval_df.loc[eval_df["horse_name"] == name, "won_awase"]        = tr.get("won_awase")
                            eval_df.loc[eval_df["horse_name"] == name, "partner_message"]  = tr.get("partner_message", "")

            eval_df = add_confluence_to_eval(eval_df)

            # 頭数×本命信頼度
            fav_row = eval_df[eval_df["popularity"] == 1]
            fav_style = fav_row.iloc[0].get("running_style", "不明") if not fav_row.empty else "不明"
            fav_reliability = eval_favorite_reliability(total_horses, 1, fav_style)

        race_label_save = st.session_state.get("_race_label", "レース")
        st.session_state.update({
            "eval_df": eval_df,
            "entries": entries_full,
            "surface": surface,
            "distance": distance,
            "venue": venue,
            "pace_info": pace_info,
            "fav_reliability": fav_reliability,
            "bias_type": bias_type,
            "race_name_for_save": race_label_save,
        })

        # Discord: 分析完了通知（65点以上の馬がいる場合）
        if not eval_df.empty and "confidence_score" in eval_df.columns and _get_webhook_url():
            top_horses = [
                {
                    "name":  r["horse_name"],
                    "score": int(r.get("confidence_score", 0)),
                    "ev":    r.get("ev"),
                    "odds":  r.get("odds"),
                    "label": r.get("confidence_label", ""),
                }
                for _, r in eval_df.iterrows()
                if r.get("confidence_score", 0) >= 65
            ]
            if top_horses:
                notify_analysis_complete(
                    race_name=race_label_save,
                    venue=venue, surface=surface, distance=distance,
                    top_horses=top_horses, budget=budget,
                )
                st.toast("📡 Discord に分析結果を通知しました", icon="🔔")

    # ---- 結果表示 ---- #
    if "eval_df" in st.session_state:
        eval_df = st.session_state["eval_df"]
        pace_info = st.session_state.get("pace_info", {})

        st.divider()

        # ペース予測バナー
        if pace_info:
            pace_color = {"ハイペース": "🔴", "ミドル〜ハイ": "🟠", "ミドル": "🟡", "スローペース": "🟢"}
            emoji = pace_color.get(pace_info.get("predicted_pace", ""), "⚪")
            st.info(f"{emoji} **展開予測: {pace_info.get('predicted_pace', '?')}**  — {pace_info.get('summary', '')}")

        # 本命信頼度・荒れやすさバナー
        fav_rel = st.session_state.get("fav_reliability", {})
        if fav_rel:
            upset = fav_rel.get("upset_score", 50)
            if upset >= 70:
                st.warning(f"🎯 **荒れやすいレース！** 穴狙いチャンス（荒れスコア {upset}/100）— {fav_rel.get('message', '')}")
            elif upset <= 35:
                st.info(f"🔒 本命が信頼できるレース（荒れスコア {upset}/100）— {fav_rel.get('message', '')}")

        # 当日バイアス確認
        bias = st.session_state.get("bias_type", "unknown")
        if bias and bias != "unknown":
            bias_info = BIAS_TYPES.get(bias, {})
            st.success(f"🌿 **バイアス: {bias_info.get('label', bias)}** — {bias_info.get('desc', '')}")

        # 総合信頼スコアテーブル
        st.subheader("🎯 総合信頼スコア")
        score_cols = ["horse_name", "popularity", "odds", "confidence_score", "confidence_label",
                      "ev", "plus_factors", "running_style", "draw_label",
                      "jockey_change_signal", "rotation_signal", "weight_signal", "romance_danger"]
        score_cols = [c for c in score_cols if c in eval_df.columns]
        rename_score = {
            "horse_name": "馬名", "popularity": "人気", "odds": "単勝",
            "confidence_score": "総合スコア", "confidence_label": "判定",
            "ev": "EV", "plus_factors": "プラス数",
            "running_style": "脚質", "draw_label": "枠",
            "jockey_change_signal": "乗替",
            "rotation_signal": "ローテ", "weight_signal": "体重",
            "romance_danger": "ロマン危険度",
        }

        def color_score(val):
            try:
                v = int(val)
                if v >= 80: return "background-color: #c6efce; font-weight: bold"
                elif v >= 65: return "background-color: #c6efce"
                elif v >= 50: return "background-color: #ffeb9c"
                elif v >= 35: return ""
                else: return "background-color: #ffc7ce"
            except Exception:
                return ""

        show_score_df = eval_df[score_cols].rename(columns=rename_score)
        styled_score = show_score_df.style.map(color_score, subset=["総合スコア"])
        st.dataframe(styled_score, use_container_width=True, hide_index=True)

        # 総合スコアのゲージグラフ
        fig_score = px.bar(
            eval_df.sort_values("confidence_score", ascending=True),
            x="confidence_score", y="horse_name",
            orientation="h",
            color="confidence_score",
            color_continuous_scale=["#ffc7ce", "#ffeb9c", "#c6efce"],
            color_continuous_midpoint=50,
            title="総合信頼スコア（大きいほど買い推奨）",
            labels={"confidence_score": "スコア", "horse_name": "馬名"},
            range_x=[0, 100],
        )
        fig_score.add_vline(x=65, line_dash="dash", line_color="green", annotation_text="推奨ライン(65)")
        st.plotly_chart(fig_score, use_container_width=True)

        # スパイダーチャート（上位3頭）
        st.subheader("🕸️ ファクター内訳（上位3頭）")
        top3_rows = eval_df.head(3)
        if "factor_breakdown" in eval_df.columns:
            categories = ["EV期待値", "ペース展開", "枠順", "騎手", "ローテ",
                          "位置取り補正", "斤量比", "ニックス", "季節・馬場適性",
                          "前走レベル", "ラップ適性", "当日バイアス"]
            fig_spider = go.Figure()
            colors = ["#2196F3", "#FF5722", "#4CAF50"]
            for idx, (_, row) in enumerate(top3_rows.iterrows()):
                breakdown = row.get("factor_breakdown", {})
                if isinstance(breakdown, dict):
                    values = [breakdown.get(c, 50) for c in categories]
                    values.append(values[0])
                    fig_spider.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=row.get("horse_name", f"馬{idx+1}"),
                        line_color=colors[idx % len(colors)],
                        opacity=0.6,
                    ))
            fig_spider.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="ファクター別内訳スパイダーチャート（上位3頭）",
                showlegend=True,
            )
            st.plotly_chart(fig_spider, use_container_width=True)

        # 詳細ファクターテーブル（展開可能）
        with st.expander("📋 全ファクター詳細テーブル"):
            detail_cols = [
                "horse_name", "ev", "pace_benefit", "draw_bonus", "draw_label",
                "jockey_change_signal", "jockey_change_msg",
                "rotation_signal", "rotation_days", "tatakidai_flag",
                "class_signal", "weight_signal", "weight_message",
                "exhaustion_comeback", "exhaustion_message",
                "last3f_label", "last3f_avg", "venue_apt_score", "venue_apt_detail",
            ]
            detail_cols = [c for c in detail_cols if c in eval_df.columns]
            detail_rename = {
                "horse_name": "馬名", "ev": "EV",
                "pace_benefit": "ペース補正", "draw_bonus": "枠補正", "draw_label": "枠評価",
                "jockey_change_signal": "乗替シグナル", "jockey_change_msg": "乗替詳細",
                "rotation_signal": "ローテ", "rotation_days": "間隔(日)",
                "tatakidai_flag": "叩き台", "class_signal": "クラス変動",
                "weight_signal": "体重変化", "weight_message": "体重詳細",
                "exhaustion_comeback": "巻返し候補", "exhaustion_message": "巻返し理由",
                "last3f_label": "上がり3F評価", "last3f_avg": "上がり3F平均",
                "venue_apt_score": "当会場適性", "venue_apt_detail": "当会場実績",
            }
            st.dataframe(eval_df[detail_cols].rename(columns=detail_rename),
                         use_container_width=True, hide_index=True)

        # 上がり3F比較バー（末脚型穴馬アラート）
        if "last3f_avg" in eval_df.columns and eval_df["last3f_avg"].notna().any():
            st.subheader("⚡ 末脚型穴馬レーダー（上がり3F比較）")
            last3f_df = eval_df[["horse_name", "popularity", "last3f_avg", "last3f_label", "last3f_bonus"]].dropna(subset=["last3f_avg"])
            if not last3f_df.empty:
                last3f_df = last3f_df.sort_values("last3f_avg")
                fig_3f = px.bar(
                    last3f_df, x="last3f_avg", y="horse_name",
                    orientation="h",
                    color="last3f_bonus",
                    color_continuous_scale=["#ffc7ce", "#ffeb9c", "#c6efce"],
                    title="上がり3F平均（小さいほど末脚が速い）",
                    labels={"last3f_avg": "上がり3F（秒）", "horse_name": "馬名"},
                    text="last3f_label",
                )
                st.plotly_chart(fig_3f, use_container_width=True)
                # 末脚型穴馬アラート
                fast_longshots = last3f_df[(last3f_df["last3f_bonus"] > 0) & (last3f_df["popularity"] >= 7)]
                for _, r in fast_longshots.iterrows():
                    st.success(f"⚡ **末脚型穴馬候補: {r['horse_name']}** ({r['popularity']}番人気) — {r['last3f_label']}")

        # ---- 3. 出走間隔カレンダービュー ---- #
        if "rotation_days" in eval_df.columns:
            st.divider()
            st.subheader("📅 出走間隔ビュー")
            interval_data = []
            for _, row in eval_df.iterrows():
                days = row.get("rotation_days")
                name = row.get("horse_name", "")
                tataki = row.get("tatakidai_flag", False)
                sig = row.get("rotation_signal", "")
                try:
                    days_int = int(days) if days is not None else None
                except Exception:
                    days_int = None

                if days_int is None:
                    band = "不明"
                    color_css = "#eeeeee"
                elif tataki:
                    band = f"⛔ 叩き1走目 ({days_int}日)"
                    color_css = "#ffc7ce"
                elif days_int <= 21:
                    band = f"連闘・中2週 ({days_int}日)"
                    color_css = "#ffeb9c"
                elif days_int <= 56:
                    band = f"適正間隔 ({days_int}日)"
                    color_css = "#c6efce"
                elif days_int <= 84:
                    band = f"やや間隔空き ({days_int}日)"
                    color_css = "#ffeb9c"
                else:
                    band = f"⛔ 叩き1走目 長期休養明け ({days_int}日)"
                    color_css = "#ffc7ce"

                interval_data.append({
                    "馬名": name,
                    "人気": int(row.get("popularity", 99)),
                    "間隔": band,
                    "ローテ評価": sig,
                })
            int_df = pd.DataFrame(interval_data).sort_values("人気")

            def color_interval(val):
                if "叩き1走目" in str(val) or "⛔" in str(val):
                    return "background-color: #ffc7ce"
                elif "連闘" in str(val):
                    return "background-color: #ffeb9c"
                elif "適正" in str(val):
                    return "background-color: #c6efce"
                return ""

            st.dataframe(
                int_df.style.map(color_interval, subset=["間隔"]),
                use_container_width=True, hide_index=True,
            )

        # 注意馬・注目馬アラート
        comebacks        = eval_df[eval_df["exhaustion_comeback"] == True]   if "exhaustion_comeback"  in eval_df.columns else pd.DataFrame()
        tatakidai_horses = eval_df[eval_df["tatakidai_flag"] == True]        if "tatakidai_flag"       in eval_df.columns else pd.DataFrame()
        beaten_horses    = eval_df[eval_df["beat_count"] > 0]                if "beat_count"           in eval_df.columns else pd.DataFrame()
        partner_horses   = eval_df[eval_df["partner_won_sat"] == True]       if "partner_won_sat"      in eval_df.columns else pd.DataFrame()
        hurdle_horses    = eval_df[eval_df["hurdle_to_flat"] == True]        if "hurdle_to_flat"       in eval_df.columns else pd.DataFrame()
        mismatch_horses  = eval_df[eval_df["turn_is_mismatch"] == True]      if "turn_is_mismatch"     in eval_df.columns else pd.DataFrame()
        foreign_horses   = eval_df[eval_df["is_short_term_foreign"] == True] if "is_short_term_foreign" in eval_df.columns else pd.DataFrame()
        proverb_horses   = eval_df[eval_df["proverb_bonus"] > 0]             if "proverb_bonus"        in eval_df.columns else pd.DataFrame()
        first_time_horses = eval_df[eval_df["is_first_surface"] == True]     if "is_first_surface"     in eval_df.columns else pd.DataFrame()
        stable_hot        = eval_df[eval_df["stable_trend"].isin(["好調","絶好調"])] if "stable_trend" in eval_df.columns else pd.DataFrame()
        pace_fit_horses   = eval_df[eval_df.get("pace_shift", pd.Series("")) == "好転"] if "pace_shift" in eval_df.columns else pd.DataFrame()

        has_alerts = not all(df.empty for df in [
            comebacks, tatakidai_horses, beaten_horses, partner_horses,
            hurdle_horses, mismatch_horses, foreign_horses, proverb_horses,
            first_time_horses, stable_hot, pace_fit_horses,
        ])
        if has_alerts:
            st.divider()
            st.subheader("💡 注目馬アラート")

            # 格言一致（VMなど）
            for _, row in proverb_horses.iterrows():
                st.success(f"📜 **{row['horse_name']}** — {row.get('proverb_label', '')} ({row['popularity']}番人気/{row.get('odds','?')}倍)")

            # 土曜勝ち馬との併せ調教（最優先）
            for _, row in partner_horses.iterrows():
                st.success(f"🏋️ **{row['horse_name']}** — {row.get('partner_message', '')} ({row['popularity']}番人気/{row.get('odds','?')}倍)")

            # 短期外国人騎手の重賞
            for _, row in foreign_horses.iterrows():
                st.info(f"🌍 **{row['horse_name']}** — {row.get('short_term_foreign_note', '')} ({row['popularity']}番人気)")

            # 障害叩き後の平地
            for _, row in hurdle_horses.iterrows():
                st.success(f"🚧 **{row['horse_name']}** — {row.get('hurdle_to_flat_message', '')} ({row['popularity']}番人気)")

            # 有力馬撃破実績
            for _, row in beaten_horses.sort_values("beat_bonus", ascending=False).head(3).iterrows():
                st.info(f"⚔️ **{row['horse_name']}** — {row.get('beat_label', '')} ({row['popularity']}番人気/{row.get('odds','?')}倍)")

            # 回り方向ミスマッチ警告
            for _, row in mismatch_horses.iterrows():
                st.warning(f"↩️ **{row['horse_name']}** — {row.get('turn_dir_label', '')} ({row['popularity']}番人気)")

            # ペース×前走上がり好転
            for _, row in pace_fit_horses.iterrows():
                st.success(f"🌊 **{row['horse_name']}** — {row.get('pace_fit_label', '')} ({row['popularity']}番人気)")

            # 厩舎絶好調
            for _, row in stable_hot[stable_hot["stable_trend"]=="絶好調"].iterrows():
                st.success(f"🏠 **{row['horse_name']}** — {row.get('stable_label', '')} ({row['popularity']}番人気)")

            # 初馬場・初距離（穴の可能性）
            for _, row in first_time_horses.head(2).iterrows():
                st.info(f"🆕 **{row['horse_name']}** — {row.get('first_time_label', '')} ({row['popularity']}番人気)")

            for _, row in comebacks.iterrows():
                st.success(f"🔄 **{row['horse_name']}** — {row.get('exhaustion_message', '')}")
            for _, row in tatakidai_horses.iterrows():
                st.warning(f"⛔ **{row['horse_name']}** — {row.get('tatakidai_message', '')}")

        # ---- 11. 類似レース検索 ---- #
        with st.expander("🔍 過去の類似レース傾向を検索", expanded=False):
            from horse_profiler import find_similar_races
            if not df_hist.empty:
                sim = find_similar_races(
                    df_hist, venue=venue,
                    surface=surface, distance=distance,
                    n_horses=total_horses,
                )
                if sim["total_races"] > 0:
                    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
                    sim_c1.metric("類似レース数", f"{sim['total_races']}レース")
                    sim_c2.metric("1番人気勝率", f"{sim['fav_win_rate']*100:.0f}%")
                    sim_c3.metric("平均3着内人気", f"{sim['top3_pop_avg']:.1f}番人気")
                    sim_c4.metric("大穴率(10番人気以上3着内)", f"{sim['upset_rate']*100:.0f}%")
                    st.info(f"📊 {venue}×{surface}×{distance}m の傾向: {sim['pattern_summary']}")
                    if sim["fast_3f_wins"] > 0.3:
                        st.success(f"⚡ 上がり最速馬の勝率が{sim['fast_3f_wins']*100:.0f}%と高い → 末脚型を重視")
                    if sim["upset_rate"] > 0.2:
                        st.warning(f"🎯 このコース×距離は大穴が出やすい（{sim['upset_rate']*100:.0f}%）")
                else:
                    st.info("類似レースのデータが不足しています（過去データを増やすと改善）")
            else:
                st.info("過去データが読み込まれていません")

        # 前日分析の保存
        st.divider()
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            save_label = st.text_input("保存名（前日に保存して当日引継ぎ）",
                                        value=st.session_state.get("race_name_for_save", "レース分析"))
        with save_col2:
            st.write("")
            st.write("")
            if st.button("💾 分析を保存"):
                path = save_session(
                    race_name=save_label,
                    entries=st.session_state.get("entries", []),
                    eval_df=eval_df,
                    surface=st.session_state.get("surface", "芝"),
                    distance=st.session_state.get("distance", 2000),
                    venue=st.session_state.get("venue", "東京"),
                    pace_info=st.session_state.get("pace_info", {}),
                )
                st.success(f"保存完了。当日の朝にサイドバーから読み込めます。")


# ============================================================
# TAB 2: 馬プロファイル
# ============================================================
with tab_horses:
    st.subheader("🐎 馬別 詳細プロファイル")
    if "entries" not in st.session_state:
        st.info("先に「レース詳細分析」タブで分析を実行してください。")
    else:
        entries = st.session_state["entries"]
        surface = st.session_state.get("surface", "芝")
        distance = st.session_state.get("distance", 2000)
        eval_df = st.session_state.get("eval_df", pd.DataFrame())

        horse_names = [e["horse_name"] for e in entries]
        selected_horse = st.selectbox("馬を選択", horse_names)
        selected_entry = next((e for e in entries if e["horse_name"] == selected_horse), {})

        # スコアサマリー
        if not eval_df.empty and "horse_name" in eval_df.columns:
            horse_eval = eval_df[eval_df["horse_name"] == selected_horse]
            if not horse_eval.empty:
                row = horse_eval.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("総合信頼スコア", f"{row.get('confidence_score', '?')}/100")
                c2.metric("EV（単勝期待値）", f"{row.get('ev', '?'):+.3f}" if isinstance(row.get('ev'), float) else "?")
                c3.metric("判定", row.get("confidence_label", "?"))
                c4.metric("プラスファクター数", f"{row.get('plus_factors', '?')}/7")
                st.divider()

        profile = build_horse_profile(
            df_hist,
            horse_name=selected_horse,
            surface=surface,
            distance=distance,
            sire=selected_entry.get("sire", ""),
            jockey=selected_entry.get("jockey", ""),
            venue=st.session_state.get("venue", "東京"),
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("総合適性スコア", f"{profile['overall_score']}/100")
        col2.metric("近走フォームスコア", f"{profile['form_score']}/100")
        col3.metric("脚質", profile["running_style"])

        c1, c2, c3 = st.columns(3)
        with c1:
            dist_apt = profile["distance_aptitude"]
            st.markdown(f"**距離適性（{distance}m帯）**")
            st.progress(min(100, dist_apt.get("score", 50)) / 100)
            st.caption(dist_apt.get("detail", ""))
        with c2:
            surf_apt = profile["surface_aptitude"]
            st.markdown(f"**馬場適性（{surface}）**")
            st.progress(min(100, surf_apt.get("score", 50)) / 100)
            st.caption(surf_apt.get("detail", ""))
        with c3:
            venue_apt = profile.get("venue_distance_aptitude", {})
            venue_label = "◎" if venue_apt.get("is_venue_exact") and venue_apt.get("score", 0) >= 60 else ""
            st.markdown(f"**当会場×距離実績{venue_label}**")
            st.progress(min(100, venue_apt.get("score", 50)) / 100)
            st.caption(venue_apt.get("detail", "同会場実績なし"))

        # ---- 右回り/左回り適性 ---- #
        from horse_profiler import calc_turn_aptitude, VENUE_TURN_DIRECTION
        ta = calc_turn_aptitude(df_hist, selected_horse, st.session_state.get("venue", "東京"), surface)
        if ta["left_stats"]["races"] > 0 or ta["right_stats"]["races"] > 0:
            st.subheader("↩️ 右回り/左回り適性")
            turn_c1, turn_c2, turn_c3 = st.columns(3)
            turn_c1.metric(
                f"左回り（{ta['left_stats']['races']}走）",
                f"勝率{ta['left_stats']['win_rate']*100:.0f}%",
            )
            turn_c2.metric(
                f"右回り（{ta['right_stats']['races']}走）",
                f"勝率{ta['right_stats']['win_rate']*100:.0f}%",
            )
            current_dir = ta["current_direction"]
            turn_c3.metric("今回の回り方向", f"{current_dir}回り" if current_dir != "不明" else "不明")
            if ta["label"]:
                if ta["is_mismatch"]:
                    st.warning(f"↩️ {ta['label']}")
                elif ta["bonus"] > 0:
                    st.success(f"↩️ {ta['label']}")
                else:
                    st.caption(f"↩️ {ta['label']}")

        # 上がり3F情報
        l3f = profile.get("last3f_info", {})
        if l3f.get("last_3f"):
            l3f_col1, l3f_col2 = st.columns(2)
            with l3f_col1:
                delta_str = ""
                if l3f.get("race_avg_3f") and l3f.get("avg_3f"):
                    delta_val = l3f["avg_3f"] - l3f["race_avg_3f"]
                    delta_str = f"{delta_val:+.2f}秒（全体比）"
                st.metric("前走上がり3F", f"{l3f['last_3f']:.1f}秒",
                           delta=delta_str if delta_str else None,
                           delta_color="inverse")
            with l3f_col2:
                st.metric("自身の平均上がり3F", f"{l3f['avg_3f']:.2f}秒" if l3f.get("avg_3f") else "データなし")
            if l3f.get("label"):
                if l3f["bonus"] > 0:
                    st.success(f"⚡ {l3f['label']}")
                elif l3f["bonus"] < 0:
                    st.warning(f"⚡ {l3f['label']}")
                else:
                    st.caption(f"上がり3F: {l3f['label']}")

        # 騎手乗り替わり詳細（理由推定付き）
        if selected_entry.get("jockey_change_signal"):
            signal = selected_entry["jockey_change_signal"]
            msg    = selected_entry.get("jockey_change_msg", "")
            reason = selected_entry.get("jockey_change_reason", "")
            rnote  = selected_entry.get("jockey_change_reason_note", "")
            if signal == "鞍上強化":
                st.success(f"✅ 乗替（強化）: {msg}")
            elif signal == "鞍上弱化":
                st.error(f"⚠️ 乗替（弱化）: {msg}")
            elif signal == "手戻り":
                st.info(f"🔄 乗替（手戻り）: {msg}")
            else:
                st.caption(f"乗替: {msg}")
            if reason:
                st.caption(f"推定理由: **{reason}** — {rnote}")

        # ローテーション・体重
        col_r, col_w = st.columns(2)
        with col_r:
            rot_msg = selected_entry.get("rotation_message", "")
            if rot_msg:
                st.caption(f"⏱️ {rot_msg}")
        with col_w:
            w_msg = selected_entry.get("weight_message", "")
            if w_msg:
                st.caption(f"⚖️ {w_msg}")

        # ---- 新ファクター 4行表示 ---- #
        nf_c1, nf_c2 = st.columns(2)
        with nf_c1:
            # 初距離・初馬場
            ft_label = selected_entry.get("first_time_label", "")
            if ft_label:
                st.info(f"🆕 {ft_label}")
            # 厩舎近況
            s_label = selected_entry.get("stable_label", "")
            s_trend = selected_entry.get("stable_trend", "")
            if s_label:
                if s_trend in ("好調", "絶好調"):
                    st.success(f"🏠 {s_label}")
                elif s_trend == "不調":
                    st.warning(f"🏠 {s_label}")
                else:
                    st.caption(f"🏠 {s_label}")
        with nf_c2:
            # 頭数変化
            fc_label = selected_entry.get("field_size_label", "")
            if fc_label:
                st.caption(f"👥 {fc_label}")
            # ペース適合
            cpf_label = selected_entry.get("pace_fit_label", "")
            cpf_shift = selected_entry.get("pace_shift", "")
            if cpf_label:
                if cpf_shift == "好転":
                    st.success(f"🌊 {cpf_label}")
                elif cpf_shift == "悪化":
                    st.warning(f"🌊 {cpf_label}")
                else:
                    st.caption(f"🌊 {cpf_label}")

        st.subheader("直近5走の成績")
        hist = profile["history"]
        if hist.empty:
            st.info("過去データに該当する馬が見つかりません")
        else:
            show_cols = [c for c in ["date", "race_name", "rank", "odds", "popularity",
                                      "distance", "surface", "track_condition", "last_3f", "jockey"]
                         if c in hist.columns]
            rename = {"date": "日付", "race_name": "レース名", "rank": "着順",
                      "odds": "オッズ", "popularity": "人気", "distance": "距離",
                      "surface": "馬場", "track_condition": "馬場状態",
                      "last_3f": "上がり3F", "jockey": "騎手"}
            st.dataframe(hist[show_cols].rename(columns=rename),
                         use_container_width=True, hide_index=True)

        # ---- 近走詳細分析カード ---- #
        from horse_profiler import analyze_recent_races, calc_resume_bonus_from_recent
        st.subheader("🔬 近走詳細分析")
        st.caption("展開・不利・馬場ミスマッチを自動解析。カードが緑＝巻き返し期待、灰色＝特記なし")

        recent_analyses = analyze_recent_races(
            df_hist, selected_horse,
            current_surface=surface,
            current_distance=distance,
            n=5,
        )
        resume_info = calc_resume_bonus_from_recent(recent_analyses)

        if resume_info["total_bonus"] > 0.01:
            st.success(
                f"🔄 **近走言い訳あり → 巻き返し期待ボーナス +{resume_info['total_bonus']*100:.1f}pt**\n\n"
                f"{resume_info['summary']}"
            )

        if not recent_analyses:
            st.info("近走データが不足しています")
        else:
            for i, race in enumerate(recent_analyses):
                rank_str = f"{race['rank']}着" if race['rank'] else "?"
                pop_str  = f"{race['popularity']}人気" if race['popularity'] else ""
                surf_str = f"{race['surface']}{race['distance']}m" if race['distance'] else race['surface']
                track_str = race['track_condition']
                f3_str   = f"上がり{race['last_3f']}秒" if race['last_3f'] else ""

                # カラー判定
                has_excuse = bool(race["excuse"])
                has_plus   = bool(race["plus"])
                is_notable = race["resume_bonus"] >= 0.015

                # ヘッダー行
                header = f"**{i+1}走前** {race['date']} {race['race_name']} / {surf_str}({track_str}) / {rank_str} {pop_str} {f3_str}"
                if race["is_fastest_3f"]:
                    header += " ⚡上がり最速"

                with st.expander(header, expanded=(i == 0 or is_notable)):
                    # シグナルタグ
                    if race["signals"]:
                        for sig in race["signals"]:
                            if any(kw in sig for kw in ["最速", "僅差", "先着"]):
                                st.success(f"✅ {sig}")
                            elif any(kw in sig for kw in ["不利", "後退", "大敗", "不向き"]):
                                st.warning(f"⚠️ {sig}")
                            elif any(kw in sig for kw in ["格上", "変わり"]):
                                st.info(f"ℹ️ {sig}")
                            else:
                                st.caption(f"• {sig}")
                    else:
                        st.caption("特記シグナルなし")

                    # コーナー通過順
                    if race["corner_order"]:
                        st.caption(f"コーナー通過: {race['corner_order']}")

                    # 上がり3F順位
                    if race["fast_3f_rank"] is not None:
                        f3_rank_str = f"上がり{race['last_3f']}秒（同レース{race['fast_3f_rank']}位）"
                        if race["is_fastest_3f"]:
                            st.success(f"⚡ {f3_rank_str}")
                        else:
                            st.caption(f"⚡ {f3_rank_str}")

                    # まとめ
                    if race["excuse"]:
                        st.error(f"🚩 言い訳: {race['excuse']}")
                    if race["plus"]:
                        st.success(f"💡 強調材料: {race['plus']}")
                    if race["resume_bonus"] > 0:
                        st.caption(f"今回への巻き返し期待値: +{race['resume_bonus']*100:.1f}pt")

        # 血統
        if selected_entry.get("sire") and not sire_stats.empty:
            sire_name = selected_entry["sire"]
            dist_cat = categorize_distance(distance)
            sire_row = sire_stats[(sire_stats["sire"] == sire_name) & (sire_stats["distance_cat"] == dist_cat)]
            if not sire_row.empty:
                wr = float(sire_row["win_rate"].iloc[0])
                avg_wr = float(sire_stats[sire_stats["distance_cat"] == dist_cat]["win_rate"].mean())
                st.metric(f"血統: {sire_name} × {dist_cat}", f"勝率 {wr*100:.1f}%",
                           delta=f"平均比 {(wr-avg_wr)*100:+.1f}%")

        st.divider()

        # ---- 2. 馬体重トレンドグラフ ---- #
        from horse_profiler import get_weight_trend
        wt_info = get_weight_trend(df_hist, selected_horse, n=10)
        if wt_info["weights"]:
            st.subheader("⚖️ 馬体重トレンド")
            wt_df = pd.DataFrame({"出走順": range(1, len(wt_info["weights"]) + 1),
                                   "馬体重(kg)": wt_info["weights"]})
            fig_wt = px.line(wt_df, x="出走順", y="馬体重(kg)",
                               title=f"{selected_horse} 馬体重推移（直近{len(wt_info['weights'])}走）",
                               markers=True)
            fig_wt.add_hline(y=wt_info["weights"][-1], line_dash="dash",
                             line_color="green", annotation_text="最新")
            st.plotly_chart(fig_wt, use_container_width=True)
            if wt_info["bonus"] != 0:
                if wt_info["bonus"] > 0:
                    st.success(f"⚖️ {wt_info['label']}")
                else:
                    st.warning(f"⚖️ {wt_info['label']}")
            else:
                st.caption(f"⚖️ {wt_info['label']}")

        # ---- 5. コーナー通過順位適性 ---- #
        from horse_profiler import get_corner_position_stats
        corner_info = get_corner_position_stats(df_hist, selected_horse, surface, distance)
        if corner_info["avg_1st_corner"] is not None:
            st.subheader("🔄 コーナー通過位置分析")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("平均通過順位", f"{corner_info['avg_1st_corner']:.1f}番手")
            if corner_info["win_position"] is not None:
                cc2.metric("勝ち時の位置", f"{corner_info['win_position']:.1f}番手")
            if corner_info["lose_position"] is not None:
                cc3.metric("負け時の位置", f"{corner_info['lose_position']:.1f}番手")
            st.caption(f"脚質判定: {corner_info['label']}")

        # ---- 7. 斤量トレンド（ハンデ戦） ---- #
        from weight_handicap import analyze_handicap_weight_trend
        race_name_profile = st.session_state.get("race_name_for_save", "")
        wc = selected_entry.get("weight_carried")
        try:
            wc_float = float(wc) if wc else None
        except Exception:
            wc_float = None
        hc_info = analyze_handicap_weight_trend(df_hist, selected_horse, wc_float, race_name_profile)
        if hc_info["is_handicap"]:
            st.subheader("🎲 ハンデ戦斤量分析")
            hc1, hc2 = st.columns(2)
            hc1.metric("ハンデ戦勝率", f"{hc_info['handicap_win_rate']*100:.0f}%")
            if hc_info["current_vs_best"] is not None:
                delta_str = f"過去最軽量比{hc_info['current_vs_best']:+.0f}kg"
                hc2.metric("今回斤量評価", delta_str,
                           delta_color="inverse" if hc_info["current_vs_best"] > 0 else "normal")
            if hc_info["bonus"] > 0:
                st.success(f"🎲 {hc_info['label']}")
            elif hc_info["bonus"] < 0:
                st.warning(f"🎲 {hc_info['label']}")
            else:
                st.caption(f"🎲 {hc_info['label']}")

        # ---- 8. 時計ランク比較 ---- #
        from horse_profiler import calc_time_rank
        time_info = calc_time_rank(df_hist, selected_horse,
                                    st.session_state.get("venue", "東京"), surface, distance)
        if time_info["best_time_raw"] is not None:
            st.subheader("⏱️ 時計ランク（レースレベル補正済み）")
            tc1, tc2, tc3 = st.columns(3)
            m, s = divmod(time_info["best_time_raw"], 60)
            tc1.metric("自己最高タイム", f"{int(m)}:{s:04.1f}")
            tc2.metric("基準タイム比", f"{time_info['best_time_adj']:+.2f}秒")
            tc3.metric("時計ランク", time_info["time_rank"])
            if time_info["time_rank_bonus"] > 0:
                st.success(f"⏱️ {time_info['label']}")
            elif time_info["time_rank_bonus"] < 0:
                st.warning(f"⏱️ {time_info['label']}")
            else:
                st.caption(f"⏱️ {time_info['label']}")

        # ---- 有力馬撃破実績 ---- #
        from horse_profiler import calc_beaten_strong_horses
        beaten_info = calc_beaten_strong_horses(
            df_hist, selected_horse, surface, distance
        )
        if beaten_info["beat_count"] > 0:
            st.subheader("⚔️ 有力馬撃破実績")
            if beaten_info["bonus"] >= 0.02:
                st.success(f"⚔️ {beaten_info['label']}")
            else:
                st.info(f"⚔️ {beaten_info['label']}")
            beat_df = pd.DataFrame(beaten_info["beat_details"])
            if not beat_df.empty:
                st.dataframe(
                    beat_df[["date", "race_name", "beaten_popularity", "our_rank", "rival_rank"]].rename(columns={
                        "date": "日付", "race_name": "レース名",
                        "beaten_popularity": "下した相手人気",
                        "our_rank": "自馬着順", "rival_rank": "相手着順",
                    }),
                    use_container_width=True, hide_index=True,
                )

        # ---- 調教・併せ馬情報（eval_dfから取得） ---- #
        if not eval_df.empty:
            horse_row_p = eval_df[eval_df["horse_name"] == selected_horse]
            if not horse_row_p.empty:
                hr = horse_row_p.iloc[0]
                if hr.get("partner_name"):
                    st.subheader("🏋️ 最終追い切り（併せ馬）")
                    p_won = hr.get("partner_won_sat", False)
                    won_aw = hr.get("won_awase")
                    if p_won:
                        st.success(f"🏋️ {hr.get('partner_message', '')}")
                    elif won_aw is True:
                        st.info(f"🏋️ {hr.get('partner_message', '')}")
                    else:
                        st.caption(f"🏋️ {hr.get('partner_message', '')}")


# ============================================================
# TAB 3: 馬券構成
# ============================================================
with tab_bet:
    st.subheader("💰 馬券構成の自動提案")
    if "eval_df" not in st.session_state:
        st.info("先に「レース詳細分析」タブで分析を実行してください。")
    else:
        eval_df_bet = st.session_state["eval_df"]
        surface_bet = st.session_state.get("surface", "芝")
        distance_bet = st.session_state.get("distance", 2000)
        bias_bet = st.session_state.get("bias_type", "neutral")

        # ---- 穴馬判定（ロマン vs 構造的根拠） ---- #
        if "structural_count" not in eval_df_bet.columns:
            eval_df_bet = evaluate_all_longshots(eval_df_bet)
            st.session_state["eval_df"] = eval_df_bet

        longshots_df = eval_df_bet[eval_df_bet["popularity"] >= 7].sort_values(
            "structural_count", ascending=False
        )
        if not longshots_df.empty:
            st.subheader("🔍 穴馬判定結果")
            for _, row in longshots_df.head(4).iterrows():
                verdict_emoji = row.get("verdict_emoji", "⚪")
                verdict = row.get("verdict", "")
                summary = row.get("summary", "")
                score = row.get("structural_count", 0)
                if score >= 3:
                    st.success(f"{verdict_emoji} **{row['horse_name']}** ({row['popularity']}番人気/{row['odds']}倍) — {verdict}")
                elif score >= 2:
                    st.warning(f"{verdict_emoji} **{row['horse_name']}** ({row['popularity']}番人気) — {verdict}")
                else:
                    st.error(f"{verdict_emoji} **{row['horse_name']}** — {verdict}")
                if summary:
                    with st.expander(f"根拠詳細: {row['horse_name']}", expanded=False):
                        st.markdown(summary)
            st.divider()

        # ---- 馬券構成提案 ---- #
        result = build_tickets(
            eval_df_bet, budget=budget, surface=surface_bet,
            distance=distance_bet, bias_type=bias_bet,
        )

        if result["brain_warning"]:
            st.error(result["brain_warning"])

        axis = result.get("axis", {})
        if axis.get("longshot_axis"):
            st.info(
                f"🎯 **穴馬軸**: {axis['longshot_axis']} ({axis.get('ls_odds',0)}倍)  "
                f"**人気馬軸**: {axis['popular_axis']} ({axis.get('pop_odds',0)}倍)  "
                f"バイアス反映: {BIAS_TYPES.get(bias_bet,{}).get('label','不明')}"
            )

        st.markdown(f"### 推奨プラン（合計: **{result['total_cost']:,}円** / 予算 {budget:,}円）")
        if result["recommended"]:
            st.dataframe(format_tickets_for_display(result["recommended"]),
                         use_container_width=True, hide_index=True)
            st.caption(f"残り予算: {result['remaining_budget']:,}円")
        else:
            st.warning("EVプラスの馬がいないため、このレースは見送りを推奨します。")

        with st.expander("📌 従来スタイル（大穴×人気馬 3連複全流し）との比較"):
            if result["romance_plan"]:
                st.dataframe(format_tickets_for_display(result["romance_plan"]),
                             use_container_width=True, hide_index=True)
                st.metric("従来スタイル費用", f"{result['romance_cost']:,}円",
                           delta=f"{result['romance_cost'] - result['total_cost']:+,}円（推奨との差）")

        if result["recommended"]:
            bet_amounts = {t.bet_type: t.amount for t in result["recommended"]}
            fig = px.pie(values=list(bet_amounts.values()), names=list(bet_amounts.keys()),
                         title=f"馬券構成内訳（合計 {result['total_cost']:,}円）")
            st.plotly_chart(fig, use_container_width=True)

        # ---- 衝動買いブレーキ：確信ボタン ---- #
        st.divider()
        st.markdown("### ✋ 購入前の最終確認")
        race_name_bet = st.session_state.get("race_name_for_save", "このレース")

        # 今週のルールチェック
        purchased = st.session_state.get("purchased_races", [])
        rule_limit = st.session_state.get("rule_max_races", 3)
        if len(purchased) >= rule_limit:
            st.error(f"⛔ 今週の購入上限（{rule_limit}レース）に達しています。購入を中断することを強く推奨します。")

        # 衝動買いリスク判定
        impulse_risk = False
        if any(kw in race_name_bet for kw in ["ハンデ", "ハンデキャップ"]):
            st.warning("⚠️ このレースはハンデ戦です。衝動買い注意レースに指定されています。")
            impulse_risk = True

        conviction_reason = st.text_input(
            "🔑 この馬券を買う最大の根拠を一言で入力（確信があれば書ける）",
            placeholder="例：○○は前走でハイペースに巻き込まれた展開負けで、今回スロー想定の展開が向く",
            key="conviction_reason",
        )

        col_confirm1, col_confirm2 = st.columns(2)
        with col_confirm1:
            if st.button(
                "✅ 根拠あり・購入確定",
                type="primary",
                disabled=not conviction_reason,
                key="confirm_purchase",
            ):
                if not conviction_reason:
                    st.error("根拠を入力してください")
                else:
                    purchased.append({
                        "race": race_name_bet,
                        "reason": conviction_reason,
                        "amount": result["total_cost"],
                        "tickets": [{"type": t.bet_type, "horses": t.horses, "amount": t.amount}
                                    for t in result["recommended"]],
                    })
                    st.session_state["purchased_races"] = purchased
                    st.success(f"✅ 購入記録しました（今週 {len(purchased)}/{rule_limit} レース）\n根拠：{conviction_reason}")
                    # Discord 買い目通知
                    if _get_webhook_url():
                        ticket_dicts = [
                            {"bet_type": t.bet_type, "horses": t.horses,
                             "amount": t.amount, "ev": getattr(t, "ev", None)}
                            for t in result["recommended"]
                        ]
                        notify_bet_plan(race_name_bet, ticket_dicts, result["total_cost"])
                        st.toast("📡 Discord に買い目を送信しました", icon="🎫")
        with col_confirm2:
            if st.button("🚫 見送る", key="cancel_purchase"):
                st.info("賢明な判断です。見送り記録しました。")


# ============================================================
# TAB 5: AI相談（Claude API）
# ============================================================
with tab_chat:
    eval_df_chat  = st.session_state.get("eval_df", pd.DataFrame())
    pace_info_chat = st.session_state.get("pace_info", {})
    bias_chat     = st.session_state.get("bias_type", "unknown")
    surface_chat  = st.session_state.get("surface", "芝")
    dist_chat     = st.session_state.get("distance", 2000)
    venue_chat    = st.session_state.get("venue", "東京")
    render_chat_tab(
        eval_df=eval_df_chat,
        pace_info=pace_info_chat,
        bias_type=bias_chat,
        surface=surface_chat,
        distance=dist_chat,
        venue=venue_chat,
        budget=budget,
        race_name=st.session_state.get("race_name_for_save", ""),
    )

# ============================================================
# TAB 6: オッズ監視
# ============================================================
with tab_odds:
    current_race_id  = st.session_state.get("preselected_race_id", "")
    current_entries  = st.session_state.get("entries", [])
    horse_names_odds = [e["horse_name"] for e in current_entries]
    render_odds_monitor_tab(current_race_id, horse_names_odds)

# ============================================================
# TAB 7: 振り返り日記
# ============================================================
with tab_diary:
    st.subheader("📓 レース振り返り日記")
    diary_tab1, diary_tab2, diary_tab3 = st.tabs(["📝 予想を記録", "📥 結果を取得", "📊 週次レポート"])

    # --- 予想記録 ---
    with diary_tab1:
        st.markdown("#### 今日の予想を保存する")
        st.caption("ほぼ自動入力済み。確認して「保存」するだけです。")

        eval_df_diary = st.session_state.get("eval_df", pd.DataFrame())
        entries_diary = st.session_state.get("entries", [])

        if eval_df_diary.empty:
            st.info("先にレース詳細分析タブで分析を実行してください。")
        else:
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                diary_race_name = st.text_input("レース名", value=st.session_state.get("race_name_for_save", ""))
                diary_race_id   = st.text_input("レースID（任意）", value=st.session_state.get("preselected_race_id", ""))
            with d_col2:
                diary_note = st.text_area("一言メモ（任意）", height=80,
                                          placeholder="例：馬場が重かったので差し有利と判断。軸はAに決定。")

            # 推奨買い目の確認
            st.markdown("**買い目の確認**（自動入力、修正可）")
            rec_bets = []
            eval_res = st.session_state.get("eval_df", pd.DataFrame())
            if not eval_res.empty:
                from bet_builder import build_tickets
                bet_result = build_tickets(eval_res, budget=budget)
                for i, t in enumerate(bet_result.get("recommended", [])):
                    c1, c2, c3 = st.columns([2, 1, 1])
                    b_type = c1.text_input("券種", value=t.bet_type, key=f"diary_btype_{i}")
                    b_horses = c2.text_input("馬", value="/".join(t.horses), key=f"diary_bhorses_{i}")
                    b_amt = c3.number_input("金額(円)", value=t.amount, step=100, key=f"diary_bamt_{i}")
                    rec_bets.append({"bet_type": b_type,
                                     "horses": b_horses.split("/"),
                                     "amount": b_amt})

            if st.button("💾 予想を日記に保存", type="primary"):
                rid = save_race_prediction(
                    race_id=diary_race_id,
                    race_name=diary_race_name,
                    race_date=str(target_date),
                    venue=st.session_state.get("venue", ""),
                    surface=st.session_state.get("surface", "芝"),
                    distance=st.session_state.get("distance", 2000),
                    track_condition=st.session_state.get("entries", [{}])[0].get("track_condition", "良"),
                    eval_df=eval_df_diary,
                    bets=rec_bets,
                    bias_type=st.session_state.get("bias_type", ""),
                    pace_predicted=st.session_state.get("pace_info", {}).get("predicted_pace", ""),
                    budget=budget,
                    note=diary_note,
                )
                st.success(f"保存完了（記録ID: {rid}）")
                st.session_state["last_diary_id"] = rid

    # --- 結果取得 ---
    with diary_tab2:
        st.markdown("#### レース結果を自動取得（レース後）")
        all_records = get_all_records()

        if all_records.empty:
            st.info("まだ記録がありません。「予想を記録」タブで保存してください。")
        else:
            # 結果未取得のレースのみ表示
            pending = all_records[all_records["hits"] == 0]
            if pending.empty:
                st.success("全レースの結果取得済みです")
            else:
                options = {f"{row['race_date']} {row['race_name']} (ID:{row['id']})": row['id']
                           for _, row in pending.iterrows()}
                selected_label = st.selectbox("結果を取得するレース", list(options.keys()))
                selected_id = options[selected_label]
                selected_row = all_records[all_records["id"] == selected_id].iloc[0]
                race_id_for_result = selected_row.get("race_id") if "race_id" in selected_row else ""

                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    result_race_id = st.text_input("netkeiba レースID",
                                                    value=str(race_id_for_result) if race_id_for_result else "",
                                                    help="例: 202405050811")
                with r_col2:
                    st.write("")
                    st.write("")
                    fetch_btn = st.button("📥 結果を自動取得", type="primary")

                if fetch_btn and result_race_id:
                    with st.spinner("netkeibaから着順・払い戻しを取得中..."):
                        fetched = fetch_race_result_from_netkeiba(result_race_id)
                    if fetched["fetched"] and fetched["results"]:
                        save_result_to_diary(selected_id, fetched, [])
                        st.success(f"取得完了！ 1着: {fetched['results'][0]['horse_name']}")
                        # 結果表示
                        result_df = pd.DataFrame(fetched["results"][:5])
                        st.dataframe(result_df[["rank","horse_name","odds","popularity","time_str"]].rename(
                            columns={"rank":"着順","horse_name":"馬名","odds":"オッズ",
                                     "popularity":"人気","time_str":"タイム"}),
                            use_container_width=True, hide_index=True)
                        # 払い戻し
                        if fetched["payouts"]:
                            st.markdown("**払い戻し:**")
                            for btype, info in list(fetched["payouts"].items())[:6]:
                                st.write(f"  {btype}: {info['combination']} → **{info['amount']:,}円**")
                    else:
                        st.warning("結果の取得に失敗しました。レースIDを確認してください。")

                # Claude自動分析ボタン
                if st.session_state.get(f"fetched_{selected_id}") or True:
                    st.divider()
                    st.markdown("#### 🤖 Claudeによる振り返り分析")
                    st.caption("「なぜ外れたか」「選ばなかった馬の正体」「次回の教訓」を自動分析")
                    if st.button("🤖 Claude に振り返りを分析させる", key="claude_postrace"):
                        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                        if not api_key:
                            st.warning("APIキーを .streamlit/secrets.toml に設定してください")
                        else:
                            # 最新の結果を再取得して分析
                            if result_race_id:
                                with st.spinner("Claudeが振り返りを分析中..."):
                                    fetched_for_analysis = fetch_race_result_from_netkeiba(result_race_id)
                                    analysis_text = generate_post_race_analysis(
                                        selected_id, fetched_for_analysis, api_key
                                    )
                                st.success("分析完了")
                                st.markdown("---")
                                st.markdown(analysis_text)
                                # セッションに保存
                                st.session_state[f"postrace_analysis_{selected_id}"] = analysis_text
                                # Discord 振り返り通知
                                if _get_webhook_url():
                                    notify_post_race_analysis(
                                        selected_row.get("race_name", "レース"),
                                        analysis_text,
                                    )
                                    st.toast("📡 Discord に振り返りを送信しました", icon="📝")
                            else:
                                st.warning("レースIDを入力してから分析してください")

                # 保存済み分析の表示
                saved_analysis = st.session_state.get(f"postrace_analysis_{selected_id}")
                if saved_analysis:
                    with st.expander("📋 保存済み振り返り分析"):
                        st.markdown(saved_analysis)

    # --- 週次レポート ---
    with diary_tab3:
        st.markdown("#### 週次パフォーマンスレポート")
        weekly_df = get_weekly_stats(weeks=8)
        factor_df = get_factor_accuracy()

        if weekly_df.empty:
            st.info("まだデータが貯まっていません。数レース記録されると自動でグラフが表示されます。")
        else:
            # ROI推移グラフ
            import plotly.express as px
            fig_roi = px.bar(weekly_df, x="week", y="roi",
                             title="週別 回収率(%)",
                             labels={"week": "週", "roi": "回収率(%)"},
                             color="roi",
                             color_continuous_scale=["#ffc7ce", "#ffeb9c", "#c6efce"],
                             color_continuous_midpoint=100)
            fig_roi.add_hline(y=100, line_dash="dash", line_color="green")
            st.plotly_chart(fig_roi, use_container_width=True)

            # サマリー
            total_inv = weekly_df["invested"].sum()
            total_ret = weekly_df["returned"].sum()
            overall_roi = total_ret / total_inv * 100 if total_inv > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("累計投資", f"{int(total_inv):,}円")
            c2.metric("累計回収", f"{int(total_ret):,}円")
            c3.metric("通算回収率", f"{overall_roi:.1f}%",
                      delta=f"{overall_roi-100:.1f}%")

            # Discord週次レポート送信ボタン
            if _get_webhook_url():
                if st.button("📡 週次レポートをDiscordに送信", key="discord_weekly"):
                    ok = notify_weekly_report(
                        weekly_df, overall_roi,
                        int(total_inv), int(total_ret),
                    )
                    if ok:
                        st.success("✅ Discordに送信しました")
                    else:
                        st.error("送信失敗。Webhook URLを確認してください")
            else:
                st.caption("💡 Discord通知を設定するとここからレポートを送信できます")

        # ファクター精度
        if not factor_df.empty:
            st.divider()
            st.markdown("#### 📊 スコア別的中率（ファクター精度）")
            st.caption("スコアが高い馬が実際に3着以内に来る確率")
            import plotly.express as px
            fig_fac = px.bar(factor_df, x="score_bucket", y="hit_rate",
                             title="スコア帯別 複勝的中率(%)",
                             labels={"score_bucket": "スコア帯", "hit_rate": "的中率(%)"},
                             text="hit_rate")
            st.plotly_chart(fig_fac, use_container_width=True)

        # ---- 4. 穴馬履歴サマリー ---- #
        st.divider()
        st.markdown("#### 🎯 穴馬予想パターン分析（あなたの的中傾向）")
        from race_diary import get_longshot_history_summary
        ls_summary = get_longshot_history_summary()
        if ls_summary["total_bets"] > 0:
            ls_c1, ls_c2, ls_c3 = st.columns(3)
            ls_c1.metric("穴馬予想数", f"{ls_summary['total_bets']}頭")
            ls_c2.metric("的中数", f"{ls_summary['total_hits']}頭",
                         delta=f"的中率{ls_summary['hit_rate']*100:.1f}%")
            ls_c3.metric("穴馬ROI", f"{ls_summary['roi']:.1f}%",
                         delta=f"{ls_summary['roi']-100:.1f}%",
                         delta_color="normal" if ls_summary["roi"] >= 100 else "inverse")

            if not ls_summary["by_popularity"].empty:
                st.markdown("**人気帯別 的中率**")
                st.dataframe(ls_summary["by_popularity"].rename(columns={
                    "pop_band": "人気帯", "bets": "予想数", "hits": "的中", "hit_rate%": "的中率%"
                }), use_container_width=True, hide_index=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                if not ls_summary["best_conditions"].empty:
                    st.markdown("**✅ 得意な条件（的中率高）**")
                    st.dataframe(ls_summary["best_conditions"],
                                 use_container_width=True, hide_index=True)
            with bc2:
                if not ls_summary["miss_conditions"].empty:
                    st.markdown("**⚠️ 苦手な条件（的中率低）**")
                    st.dataframe(ls_summary["miss_conditions"],
                                 use_container_width=True, hide_index=True)
        else:
            st.info("穴馬予想の記録がまだありません。日記に記録が貯まると自動で分析されます。")

        # 全記録テーブル
        all_rec = get_all_records()
        if not all_rec.empty:
            st.divider()
            st.markdown("#### 📋 全レース記録")
            show_cols = [c for c in ["race_date","race_name","venue","surface","distance",
                                      "total_invested","total_returned","roi","hits","note"]
                         if c in all_rec.columns]
            st.dataframe(all_rec[show_cols].rename(columns={
                "race_date":"日付","race_name":"レース名","venue":"会場",
                "surface":"馬場","distance":"距離","total_invested":"投資",
                "total_returned":"回収","roi":"回収率%","hits":"的中","note":"メモ"
            }), use_container_width=True, hide_index=True)

# ============================================================
# TAB 8: バックテスト
# ============================================================
with tab_backtest:
    st.subheader("📊 バックテスト")
    st.caption("EVプラスの馬だけを買い続けた場合 vs 穴馬全流しスタイル")
    if win_rate_table.empty:
        st.warning("過去データを読み込んでください。")
    else:
        with st.spinner("バックテスト計算中..."):
            sim = _run_backtest(df_hist, win_rate_table)
        col1, col2, col3 = st.columns(3)
        col1.metric("EVプラス馬の単勝回収率", f"{sim['ev_plus_rr']:.1f}%",
                     delta=f"{sim['ev_plus_rr'] - 100:.1f}%")
        col2.metric("全穴馬（10人気以下）回収率", f"{sim['longshot_rr']:.1f}%",
                     delta=f"{sim['longshot_rr'] - 100:.1f}%")
        col3.metric("EVプラスの馬の割合", f"{sim['ev_plus_pct']:.1f}%")

        if "monthly_rr" in sim and not sim["monthly_rr"].empty:
            fig = px.line(sim["monthly_rr"], x="month", y="recovery_rate",
                          title="月別 単勝回収率（EVプラス馬のみ）",
                          labels={"month": "月", "recovery_rate": "回収率(%)"})
            fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="100%")
            fig.add_hline(y=75, line_dash="dot", line_color="orange", annotation_text="控除率ライン")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 5: 騎手・血統ランキング
# ============================================================
with tab_jockey:
    st.subheader("🏆 穴馬に強い騎手ランキング")
    if jockey_stats.empty:
        st.info("騎手データが生成できません。")
    else:
        top_j = jockey_stats.head(20).copy()
        top_j["place_rate_longshot"] = (top_j["place_rate_longshot"] * 100).round(1)
        top_j["win_rate_longshot"] = (top_j["win_rate_longshot"] * 100).round(1)
        st.dataframe(
            top_j[["jockey", "rides", "wins", "places", "place_rate_longshot", "win_rate_longshot"]]
            .rename(columns={"jockey": "騎手", "rides": "穴乗り数", "wins": "穴勝利",
                              "places": "穴複勝", "place_rate_longshot": "穴複勝率%",
                              "win_rate_longshot": "穴勝率%"}),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    st.subheader("🧬 血統×距離帯 適性ランキング")
    if sire_stats.empty:
        st.info("血統データが生成できません。")
    else:
        dist_filter = st.selectbox("距離帯フィルター", ["全て", "短距離", "マイル", "中距離", "長距離"])
        sire_show = sire_stats.copy()
        if dist_filter != "全て":
            sire_show = sire_show[sire_show["distance_cat"] == dist_filter]
        sire_show["win_rate"] = (sire_show["win_rate"] * 100).round(2)
        st.dataframe(
            sire_show.head(30)[["sire", "distance_cat", "races", "wins", "win_rate"]]
            .rename(columns={"sire": "父", "distance_cat": "距離帯",
                              "races": "出走数", "wins": "勝利数", "win_rate": "勝率%"}),
            use_container_width=True, hide_index=True,
        )

# ============================================================
# TAB 11: メモ編集（ナレッジベース）
# ============================================================
with tab_kb:
    st.subheader("📖 競馬メモ編集（ナレッジベース）")
    st.caption("騎手パターン・血統メモ・レース特性などを追加・削除できます。変更は即座に分析に反映されます。")

    kb_data = load_kb()

    kb_section = st.selectbox(
        "編集するセクション",
        ["jockey_patterns", "sire_patterns", "jockey_change_patterns",
         "trainer_jockey_combos", "race_specific_patterns",
         "upset_race_conditions", "special_signals", "avoid_conditions"],
        format_func=lambda x: {
            "jockey_patterns": "騎手パターン",
            "sire_patterns": "血統（種牡馬）パターン",
            "jockey_change_patterns": "乗り替わりパターン",
            "trainer_jockey_combos": "調教師×騎手コンボ",
            "race_specific_patterns": "レース固有パターン",
            "upset_race_conditions": "荒れやすいレース条件",
            "special_signals": "特殊シグナル",
            "avoid_conditions": "回避条件",
        }.get(x, x),
    )

    entries = kb_data.get(kb_section, [])

    st.write(f"**現在のエントリ数: {len(entries)}件**")

    # 既存エントリ一覧
    import json as _json
    delete_idx = None
    for i, entry in enumerate(entries):
        label = entry.get("jockey") or entry.get("sire") or entry.get("race_name") or entry.get("label") or f"エントリ{i+1}"
        note = entry.get("note", "")
        action = entry.get("action", "")
        action_tag = "✅ 買い" if action == "buy" else ("❌ 消し" if action == "avoid" else f"📌 {action}")
        with st.expander(f"{i+1}. {label}  {action_tag}  —  {note[:40]}{'...' if len(note)>40 else ''}"):
            st.json(entry)
            if st.button(f"このエントリを削除", key=f"del_kb_{kb_section}_{i}"):
                delete_idx = i

    if delete_idx is not None:
        entries.pop(delete_idx)
        kb_data[kb_section] = entries
        save_kb(kb_data)
        st.success("削除しました。")
        st.rerun()

    st.divider()
    st.write("**新しいエントリを追加（JSON形式）**")
    st.caption("例: {\"jockey\": \"○○\", \"conditions\": [{\"venue\": \"東京\"}], \"action\": \"buy\", \"bonus\": 0.02, \"note\": \"東京の○○は買い\"}")

    new_entry_json = st.text_area("JSONで入力", height=120, key="kb_new_entry")
    if st.button("追加する", key="kb_add_btn"):
        try:
            new_entry = _json.loads(new_entry_json)
            entries.append(new_entry)
            kb_data[kb_section] = entries
            save_kb(kb_data)
            st.success("追加しました！")
            st.rerun()
        except _json.JSONDecodeError as e:
            st.error(f"JSONの形式が正しくありません: {e}")

    st.divider()
    # 全データをダウンロード
    import io as _io
    kb_json_str = _json.dumps(kb_data, ensure_ascii=False, indent=2)
    st.download_button(
        "全データをダウンロード（バックアップ）",
        data=kb_json_str.encode("utf-8"),
        file_name="knowledge_base_backup.json",
        mime="application/json",
    )
    # アップロードして復元
    uploaded = st.file_uploader("バックアップから復元（JSONファイル）", type="json", key="kb_upload")
    if uploaded:
        try:
            restored = _json.loads(uploaded.read().decode("utf-8"))
            save_kb(restored)
            st.success("復元しました！")
            st.rerun()
        except Exception as e:
            st.error(f"復元エラー: {e}")
