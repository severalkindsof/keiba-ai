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
from datetime import date, timedelta, datetime
from pathlib import Path

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
from ev_calculator import evaluate_race, ev_label, get_model_type as _get_lgbm_model_type
from horse_profiler import build_horse_profile
from bet_builder import build_tickets, format_tickets_for_display
from scraper import (
    # 第45波: netkeiba依存削減 — 残すのは fetch_multi_odds, fetch_horse_weight, fetch_blinker*, fetch_weather のみ
    manual_entry_template,
)  # CLEAN: get_race_id_from_venue_date は未使用のため削除
from pace_analyzer import analyze_field_pace
from draw_bias import get_draw_bonus, get_draw_label, build_dynamic_draw_table
from jockey_change import get_jockey_change_for_field
from rotation_weight import analyze_rotation_for_field
from confluence import add_confluence_to_eval, get_race_quality_score
from race_selector import scan_weekend_races, get_this_weekend_dates, format_race_scan_display
# CLEAN: demo_data は未使用 (デモモード廃止済み)
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
from odds_monitor import render_odds_monitor_tab  # _do_fetch_and_record は未使用
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI テーマ・カスタムCSSをロード
from ui_theme import load_theme, banner as ui_banner, apply_chart_theme, COLOR, CHART_COLORS, pill, hero
load_theme()

# Y5: session_state ステイル検知 — 24時間以上未更新ならレース関連状態を破棄
_now_ts = datetime.now()
_stamp = st.session_state.get("_state_stamp")
if _stamp is None:
    st.session_state["_state_stamp"] = _now_ts
else:
    # _stamp が datetime オブジェクトでない場合（rerun越え型変換失敗）も考慮
    try:
        _stale_keys_24h = [
            "eval_df", "entries", "preselected_race_id", "_direct_race_id",
            "_tfjv_csv_path", "_auto_analyze", "wr_selected_venue",
            "wr_selected_race_no", "wr_selected_day", "wr_active",
            "race_selectbox", "venue_selectbox", "_pending_race_no",
            "pace_info", "fav_reliability", "bias_type",
        ]
        # 24時間以上経過なら掃除
        if hasattr(_stamp, "year") and (_now_ts - _stamp).total_seconds() > 86400:
            for _sk in _stale_keys_24h:
                st.session_state.pop(_sk, None)
    except (TypeError, ValueError):
        pass
    st.session_state["_state_stamp"] = _now_ts

hero("競馬予想AI", "JRA中央競馬専用　—　期待値（EV）ベースの分析サポート")


# ============================================================
# TFJV CSV から race_id 抽出
# ============================================================

@st.cache_resource
def _load_tfjv_full() -> "pd.DataFrame":
    """tfjv_all.parquet をシングルトンでロード（分析毎の800k行再読込を防ぐ）"""
    _tfjv_full_path = Path(__file__).parent / "data" / "tfjv_all.parquet"
    if not _tfjv_full_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(_tfjv_full_path)
    for c in ["rank", "popularity", "field_size", "speed_figure",
              "last_3f", "horse_weight", "finish_time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def extract_race_id_from_tfjv_csv(csv_path: str) -> str:
    """
    TFJV エクスポート CSV から対象レースの race_id を抽出する。
    第45波: 出馬表分析CSV（33列・新形式）にも対応。ファイル名で判定し、
    出馬表分析ならtfjv_entries経由で取得。それ以外は COLS52/45 形式として処理。
    """
    if not csv_path or not Path(csv_path).exists():
        return ""

    # 出馬表分析CSV は別経路（独自 14桁ID）
    if "出馬表分析" in Path(csv_path).name:
        try:
            from tfjv_entries import list_races as _tj_list
            _races = _tj_list(csv_path)
            if _races:
                return _races[0]["race_id"]
        except Exception:
            pass
        return ""
    df = None
    for enc in ("cp932", "utf-8"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, header=None, dtype=str)
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            return ""
    if df is None or df.empty:
        return ""

    RID_COL = 40  # COLS52/COLS45 共通
    if len(df.columns) > RID_COL:
        s = df.iloc[:, RID_COL].astype(str).str.strip()
        valid = s[s.str.match(r"^\d{12}$", na=False)]
        if not valid.empty:
            return valid.sort_values().iloc[-1]

    # フォールバック: 全列スキャン
    from collections import Counter
    cnt: Counter = Counter()
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        cnt.update(s[s.str.match(r"^\d{12}$", na=False)].tolist())
    if cnt:
        return cnt.most_common(1)[0][0]
    return ""


# ============================================================
# バックテスト関数
# ============================================================

def _run_backtest(df: pd.DataFrame, wrt: pd.DataFrame) -> dict:
    """第45波: TFJV に odds 列が無いため odds 依存メトリクスを廃止。
    代わりに過去データから「人気別的中率」「穴馬の複勝率」「クラス別」を集計。
    """
    empty = {"by_pop": pd.DataFrame(), "longshot_in3": 0, "fav_win": 0,
             "longshot_n": 0, "fav_n": 0, "monthly_long_in3": pd.DataFrame()}
    if df.empty:
        return empty
    if not all(c in df.columns for c in ("win_flag", "popularity")):
        return empty

    sample = df.dropna(subset=["popularity"]).copy()
    sample["popularity"] = pd.to_numeric(sample["popularity"], errors="coerce")
    sample["win_flag"] = pd.to_numeric(sample["win_flag"], errors="coerce").fillna(0).astype(int)
    if "rank" in sample.columns:
        sample["rank"] = pd.to_numeric(sample["rank"], errors="coerce")
        sample["place_flag"] = (sample["rank"] <= 3).astype(int)
    else:
        sample["place_flag"] = sample["win_flag"]
    sample = sample.dropna(subset=["popularity"])
    if sample.empty:
        return empty

    # 人気別: 1人気 / 2-3人気 / 4-5人気 / 6-9人気 / 10人気以下
    bins = [0, 1, 3, 5, 9, 99]
    labels = ["1番人気", "2-3番人気", "4-5番人気", "6-9番人気", "10番人気以下"]
    sample["pop_band"] = pd.cut(sample["popularity"], bins=bins, labels=labels)
    by_pop = (sample.groupby("pop_band", observed=True)
              .agg(races=("win_flag", "count"),
                   wins=("win_flag", "sum"),
                   in3=("place_flag", "sum"))
              .reset_index())
    by_pop["win_rate"]   = (by_pop["wins"] / by_pop["races"] * 100).round(1)
    by_pop["place_rate"] = (by_pop["in3"]  / by_pop["races"] * 100).round(1)

    fav = sample[sample["popularity"] == 1]
    longs = sample[sample["popularity"] >= 10]
    longshot_in3 = round(longs["place_flag"].mean() * 100, 1) if len(longs) > 0 else 0
    fav_win = round(fav["win_flag"].mean() * 100, 1) if len(fav) > 0 else 0

    # 月別 穴馬複勝率（直近24ヶ月）
    monthly = pd.DataFrame()
    if "date" in sample.columns and len(longs) > 0:
        longs = longs.copy()
        longs["month"] = pd.to_datetime(longs["date"], errors="coerce").dt.to_period("M").astype(str)
        monthly = (longs.groupby("month")
                   .agg(n=("place_flag", "count"), in3=("place_flag", "sum"))
                   .reset_index())
        monthly["in3_rate"] = (monthly["in3"] / monthly["n"] * 100).round(1)
        monthly = monthly.tail(24)

    return {"by_pop": by_pop, "longshot_in3": longshot_in3, "fav_win": fav_win,
            "longshot_n": len(longs), "fav_n": len(fav), "monthly_long_in3": monthly}


# ============================================================
# サイドバー
# ============================================================
with st.sidebar:
    st.header("基本設定")

    # 第13波: ベッティングモード（堅軸 / 爆穴）
    betting_mode = st.radio(
        "ベッティングモード",
        ["爆穴モード", "堅軸モード"],
        index=0,
        horizontal=True,
        help=(
            "堅軸: 1〜3人気中心、単・複・馬連、EV閾値ゆるめ（堅実に積み上げ）\n"
            "爆穴: 8人気以下含む、ワイド・三連複中心、EV閾値厳しめ（ロマン枠・予算は20%目安）"
        ),
        key="betting_mode_radio",
    )
    st.session_state["betting_mode"] = "堅軸" if "堅軸" in betting_mode else "爆穴"
    if st.session_state["betting_mode"] == "爆穴":
        st.caption("爆穴モード: Conformal 見送りレースを「むしろ戦場」として穴ヒモを拾います")
    else:
        st.caption("堅軸モード: 上位人気を軸に堅実な積み上げ重視")

    # 第19波 (V2): key 無しだと session_state["budget"] に入らず、
    # Portfolio Kelly の bankroll が常に 5000×20 固定になっていた
    budget = st.number_input("1レースの予算（円）", min_value=500, max_value=10000,
                             value=5000, step=500, key="budget")
    if st.session_state["betting_mode"] == "爆穴":
        # 爆穴モードは予算の 20% をデフォ（ロマン枠）
        budget = int(budget * 0.2)
        st.caption(f"爆穴モード適用: 実際の予算は {budget:,}円（元予算の20%）")
    target_date = st.date_input("対象日", value=date.today())
    date_str = target_date.strftime("%Y%m%d")

    # 金曜夜モード（枠順発表直後の分析フロー）
    from datetime import date as _date
    _today = _date.today()
    _is_friday = _today.weekday() == 4
    friday_mode = st.toggle(
        "金曜夜モード（枠順発表後）",
        value=_is_friday,
        help="金曜日の枠順発表後に使うモード。土日の全レースを枠順込みで一覧表示します。",
    )
    if friday_mode:
        st.caption("翌土日のレースを先読みして予想の土台を作ります")

    st.divider()

    # ============================================================
    # ① データロード（常時実行・状態のみ折りたたみ表示）
    # ============================================================
    demo_mode = False
    with st.spinner("過去データ読み込み中..."):
        df_hist = load_race_results()
    if not df_hist.empty:
        df_hist = merge_with_horse_cache(df_hist)
    with st.spinner("統計テーブル構築中..."):
        win_rate_table  = get_win_rate_table(df_hist)
        sire_stats      = get_sire_stats(df_hist)
        jockey_stats    = get_jockey_stats(df_hist)
        draw_table      = build_dynamic_draw_table(df_hist)
    with st.spinner("馬別プロファイル構築中..."):
        from horse_stats import build_horse_stats
        horse_stats_df  = build_horse_stats(df_hist)
        nicks_table     = build_nicks_table(df_hist)
        race_level_table = build_race_level_table(df_hist)

    with st.expander("データ状況", expanded=False):
        if df_hist.empty:
            st.warning("過去データなし。convert_tfjv.py を実行してください。")
        else:
            st.caption(f"レース結果 {len(df_hist):,}件 / 勝率テーブル {len(win_rate_table)}行")

    # ============================================================
    # ② 分析・ベッティング設定（Benter / ケリー / フィルター）
    # ============================================================
    with st.expander("分析・ベッティング設定", expanded=False):
        st.markdown("**Benter Odds Blending**")
        from ev_calculator import get_benter_weights as _get_bw
        _def_a, _def_b = _get_bw()
        st.caption("自モデル予測と市場確率を統合（α↑=自モデル重視 / β↑=市場重視）")
        st.session_state["benter_alpha"] = st.slider(
            "α（自モデル重み）", 0.0, 2.0, float(_def_a), 0.05, key="sld_benter_a")
        st.session_state["benter_beta"]  = st.slider(
            "β（市場重み）", 0.0, 2.0, float(_def_b), 0.05, key="sld_benter_b")
        st.session_state["benter_enabled"] = st.checkbox(
            "Benter ブレンド有効", value=True, key="cb_benter_en")

        st.markdown("---")
        st.markdown("**分数ケリー（破産防止）**")
        st.caption("Ziemba・Benter共通推奨は 1/2〜1/3 ケリー")
        st.session_state["kelly_fraction_ratio"] = st.select_slider(
            "ケリー倍率",
            options=[0.25, 0.33, 0.50, 0.75, 1.0],
            value=0.50,
            key="sld_kelly_ratio",
        )

        st.markdown("---")
        st.markdown("**見送り条件フィルター**")
        st.session_state["filter_ev_threshold"] = st.slider(
            "EV最低閾値", 0.0, 0.5, 0.10, 0.05,
            help="この値以下のEVは見送り")
        st.session_state["filter_odds_min"] = st.slider(
            "オッズ下限", 1.0, 10.0, 2.5, 0.5)
        st.session_state["filter_score_min"] = st.slider(
            "スコア下限", 0, 80, 50, 5)
        st.session_state["use_dynamic_ev"] = st.checkbox(
            "動的EV閾値（人気帯別）", value=True, key="cb_dyn_ev",
            help="人気1-3=+0.05 / 4-6=+0.10 / 7-9=+0.18 / 10-13=+0.25 / 14-=+0.35",
        )

        st.markdown("---")
        st.markdown("**勝率累乗フィルター**")
        st.caption("累乗→トップ2 差が閾値超で「買い」")
        st.session_state["power_filter_p"] = st.slider(
            "累乗指数 power", 1.0, 6.0, 4.0, 0.5, key="sld_pow_p")
        st.session_state["power_filter_gap"] = st.slider(
            "トップ2 EV差閾値", 0.0, 1.0, 0.4, 0.05, key="sld_pow_gap")

        st.markdown("---")
        st.markdown("**データ質フィルター**")
        st.session_state["min_past_races"] = st.slider(
            "最低過去戦数", 0, 10, 3, 1)

        st.markdown("---")
        st.markdown("**アンサンブル予測（NEW-6 / A-2）**")
        from ensemble import get_default_weights as _gdw
        _dw = _gdw()
        st.caption("LightGBM + 市場 + 一様分布 の加重平均で予測を安定化")
        # A-2: stacking モード（XGBoost + CatBoost と統合）
        try:
            from stacking_predictor import is_available as _stk_avail
            _stk_ok = _stk_avail()
        except Exception:
            _stk_ok = False
        st.session_state["stacking_enabled"] = st.checkbox(
            "Stacking 有効（LGBM + XGB + CB 統合 / 推奨）", value=_stk_ok, key="cb_stk_en",
            disabled=not _stk_ok,
            help="A-2: 3-way stacking。train_stacking.py 実行後に有効化されます。")
        st.session_state["ensemble_enabled"] = st.checkbox(
            "アンサンブル（市場+一様）有効", value=False, key="cb_ens_en")
        st.session_state["ens_w_lgbm"] = st.slider(
            "LGBM 重み w₁", 0.0, 1.0, _dw[0], 0.05, key="sld_ens_w1")
        st.session_state["ens_w_market"] = st.slider(
            "市場 重み w₂", 0.0, 1.0, _dw[1], 0.05, key="sld_ens_w2")
        st.session_state["ens_w_uniform"] = st.slider(
            "一様 重み w₃", 0.0, 0.5, _dw[2], 0.05, key="sld_ens_w3")

    # ============================================================
    # ③ セッション・通知
    # ============================================================
    with st.expander("セッション・通知", expanded=False):
        st.markdown("**前日セッション引継ぎ**")
        saved_sessions = list_saved_sessions()
        if saved_sessions:
            session_options = {s["filename"]: s["path"] for s in saved_sessions}
            selected_session = st.selectbox(
                "保存済み分析を読込",
                ["（新規）"] + list(session_options.keys()),
                label_visibility="collapsed",
            )
            if selected_session != "（新規）" and st.button("読込", key="btn_load_sess"):
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
                    st.success(f"「{loaded['race_name']}」を読み込みました")
        else:
            st.caption("保存済みセッションなし")

        st.markdown("---")
        st.markdown("**Discord通知**")
        discord_ok = bool(_get_webhook_url())
        if discord_ok:
            st.caption("ON")
        else:
            render_discord_setup_section()

        st.markdown("---")
        st.markdown("**スマホURL**")
        st.code("http://PCのIPアドレス:8501", language=None)


# ============================================================
# タブ構成
# ============================================================
init_db()  # DBを初期化（初回起動時のみテーブル作成）

(tab_home, tab_race, tab_horses, tab_bet,
 tab_chat, tab_odds, tab_diary, tab_backtest, tab_jockey, tab_kb, tab_win5) = st.tabs([
    "ホーム",
    "レース分析",
    "馬プロファイル",
    "馬券構成",
    "AI相談",
    "オッズ監視",
    "振り返り",
    "バックテスト",
    "騎手・血統",
    "メモ",
    "WIN5",
])


# ============================================================
# TAB 0: ホームページ（狙い目レース・資金配分）
# ============================================================
with tab_home:
    # ---- 第14波: ダークヒーロー ---- #
    _mode_hero = st.session_state.get("betting_mode", "爆穴")
    _wr_count = len(st.session_state.get("weekly_races", []))
    _purchased_n = len(st.session_state.get("purchased_races", []))
    from datetime import date as _hd
    _today_hero = _hd.today()
    _wd = ["月", "火", "水", "木", "金", "土", "日"][_today_hero.weekday()]
    _mode_disp = "爆穴 — 一発逆転の夜" if _mode_hero == "爆穴" else "堅軸 — 静かに積む夜"
    # 第18波: AUC ハードコード排除 — メトリクスファイルから読む（再学習で自動更新）
    def _read_metric(fname: str, fallback: str) -> str:
        try:
            import json as _mj
            with open(Path(__file__).parent / "data" / fname, encoding="utf-8") as _mf:
                return f"{_mj.load(_mf)['test_auc']:.3f}"
        except Exception:
            return fallback
    _auc_place = _read_metric("place_metrics.json", "0.809")
    _auc_win   = _read_metric("lgbm_metrics.json", "0.835")
    st.markdown(f"""
<div class="dash-hero">
  <div class="hero-eyebrow">KEIBA INTELLIGENCE — {_today_hero.strftime('%Y.%m.%d')} ({_wd})</div>
  <div class="hero-title">{_mode_disp}</div>
  <div class="hero-sub">複勝EVで穴を測り、Conformalで罠を避ける。データは嘘をつかない。</div>
  <div class="dash-stats">
    <div class="dash-stat accent"><div class="v">{_wr_count}<span class="unit">R</span></div><div class="k">今週の登録</div></div>
    <div class="dash-stat"><div class="v">{_purchased_n}<span class="unit">R</span></div><div class="k">購入済み</div></div>
    <div class="dash-stat gold"><div class="v">{_auc_place}<span class="unit">AUC</span></div><div class="k">馬券内モデル</div></div>
    <div class="dash-stat"><div class="v">{_auc_win}<span class="unit">AUC</span></div><div class="k">勝率モデル</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ---- オッズ急変：dash-alert スタイルで一行ずつ粋に ---- #
    from odds_monitor import load_odds_history, detect_odds_signals as _det_sigs
    _all_odds_hist = load_odds_history()
    _urgent_signals = []
    for _rid, _ in _all_odds_hist.items():
        _sigs = _det_sigs(_rid)
        for _s in _sigs:
            if abs(_s["change_pct"]) >= 25:
                _urgent_signals.append(_s)
    # 第17波 (Z5): 改善ループ用にアラート件数を保存
    st.session_state["odds_alert_count"] = len(_urgent_signals)
    # 第45波: MARKET SIGNALS 削除（ノイズ多く実用に耐えなかった）

    # ---- 週次ルールの設定と衝動買いカウンター ---- #
    with st.expander("今週のマイルール設定（衝動買い防止）"):
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
    # B-5: 連続外し時の自動ストップロス（race_diary から最近の結果を集計）
    try:
        from race_diary import get_all_records as _gar
        _recent = _gar()
        if not _recent.empty and "result_rank" in _recent.columns:
            _recent_sorted = _recent.sort_values("race_date", ascending=False).head(5)
            _losses = (_recent_sorted["result_rank"] > 3).sum()
            if _losses >= 5:
                ui_banner("avoid",
                    f"<b>5連敗中！</b> 直近5レース全て3着圏外。<b>今週の購入を控えることを推奨</b>します。")
            elif _losses >= 3:
                ui_banner("warn",
                    f"<b>3連敗中</b>。冷静に。次のレースは特に慎重な根拠を求めてください。")
    except Exception:
        pass

    if purchased_count > 0:
        st.info(f"今週購入済み: {purchased_count}レース / ルール上限: {rule_max_races}レース")
        if purchased_count >= rule_max_races:
            st.error("今週の購入上限に達しました。これ以上の購入はルール違反です。")

    st.divider()

    # ---- 第40波: 初ブリンカーアラート（手動チェック方式） ---- #
    st.markdown('<div class="dash-label">BLINKER ALERT — 初ブリンカー装着馬</div>', unsafe_allow_html=True)
    st.caption("装着馬は自動取得（赤B/黒BはHTMLで区別不可のため、初判定は新聞タブの赤丸を見てチェック）")
    with st.expander("ブリンカー馬を取得 / 初判定", expanded=False):
        _bc1, _bc2 = st.columns([3, 2])
        with _bc1:
            _bdate = st.text_input("対象日(YYYYMMDD)", value=date.today().strftime("%Y%m%d"), key="blinker_date")
        with _bc2:
            _bmin = st.number_input("対象レース番号以降", 1, 12, 1, key="blinker_minr")
        _bc3, _bc4 = st.columns([1, 1])
        with _bc3:
            if st.button("自動取得（装着馬リスト）", key="fetch_blinker"):
                try:
                    from scraper import fetch_blinkers_for_date as _fbd
                    with st.spinner("新聞タブから装着馬を取得中..."):
                        _bres = _fbd(_bdate.strip(), venue=None, min_race_no=int(_bmin))
                    st.session_state["blinker_auto"] = _bres
                    _n = sum(len(r["all"]) for r in _bres)
                    st.success(f"{len(_bres)}レースで計{_n}頭のブリンカー装着馬を取得")
                except Exception as _e_b:
                    st.error(f"取得失敗（金曜の出走確定後に再試行）: {type(_e_b).__name__}")
        with _bc4:
            _bman = st.text_area("手動補完（取得失敗時）", height=80, key="blinker_manual",
                                  placeholder="馬名を改行区切り")
            if _bman.strip():
                names = [n.strip() for n in _bman.splitlines() if n.strip()]
                st.session_state["blinker_manual_list"] = names

        # 取得済み装着馬：各馬に「初？」チェックボックス
        _auto = st.session_state.get("blinker_auto", [])
        _manual = st.session_state.get("blinker_manual_list", [])
        # 第45波: 馬名 → [(会場, レース番号), ...] のマップを構築
        from scraper import VENUE_CODES as _VC
        _code2venue = {v: k for k, v in _VC.items()}
        _horse_loc: dict[str, list[str]] = {}
        for r in _auto:
            rid = r.get("race_id", "")
            rno = r.get("race_no", "")
            venue = _code2venue.get(rid[4:6], "?") if len(rid) >= 6 else "?"
            tag = f"{venue}{rno}R"
            for nm in r.get("all", []):
                _horse_loc.setdefault(nm, []).append(tag)
        for nm in _manual:
            _horse_loc.setdefault(nm, [])
        _all_set = set(_horse_loc.keys())

        def _loc_str(nm: str) -> str:
            locs = _horse_loc.get(nm, [])
            return f"（{' / '.join(locs)}）" if locs else ""

        if _all_set:
            st.markdown("**装着馬リスト — 「初」を選択（新聞タブの赤丸B）**")
            _first_chosen = set()

            # 第45波: 会場+レース番号順にソート（race_no が連続でグルーピングしやすい）
            def _sort_key(nm: str):
                locs = _horse_loc.get(nm, [])
                if not locs:
                    return ("zzz", 999, nm)  # 手動入力は最後
                first = locs[0]  # 例: "函館10R"
                # 末尾 "数字R" を分離
                import re as _re
                m = _re.match(r"^(.+?)(\d+)R$", first)
                if m:
                    return (m.group(1), int(m.group(2)), nm)
                return (first, 0, nm)

            _sorted_names = sorted(_all_set, key=_sort_key)
            # 同じ「会場+レース番号」をグルーピング表示
            _current_loc = None
            for nm in _sorted_names:
                locs = _horse_loc.get(nm, [])
                loc_label = locs[0] if locs else "手動入力"
                if loc_label != _current_loc:
                    st.markdown(f"**📍 {loc_label}**")
                    _current_loc = loc_label
                    _cols = st.columns(2)
                    _i = 0
                with _cols[_i % 2]:
                    _key = f"blinker_first_{nm}"
                    # 複数レース出走時のみ追加表示
                    _extra = f"（他: {' / '.join(locs[1:])}）" if len(locs) > 1 else ""
                    if st.checkbox(f"⚡初 {nm}{_extra}", key=_key):
                        _first_chosen.add(nm)
                _i += 1

            st.session_state["blinker_all_set"] = _all_set
            st.session_state["blinker_first_set"] = _first_chosen
            st.session_state["blinker_horse_loc"] = _horse_loc
            st.caption(f"装着 {len(_all_set)}頭 / うち初 {len(_first_chosen)}頭")
        else:
            st.caption("（未取得 — 自動取得ボタンを押すか手動補完）")

    # 第45波: 重複していたフラット一覧（取得済みブリンカー馬のホーム要約）を削除
    # → expander 内のグルーピング表示があれば十分。「初」だけ要約表示する。
    _first_blinker = st.session_state.get("blinker_first_set", set())
    _blinker_loc = st.session_state.get("blinker_horse_loc", {})
    if _first_blinker:
        st.markdown('<div class="dash-label">⚡ 初ブリンカー装着馬</div>', unsafe_allow_html=True)
        for nm in sorted(_first_blinker):
            _locs = _blinker_loc.get(nm, [])
            _loc_html = f' <span style="opacity:.65;font-size:.85em">{" / ".join(_locs)}</span>' if _locs else ""
            st.markdown(f'<div class="dash-alert"><span class="tag">初B</span>'
                        f'<span class="body">{nm}{_loc_html}</span></div>', unsafe_allow_html=True)

    # 第45波: モード連動おすすめレース 削除（堅軸モードで爆穴判定が出るバグ含み）

    st.divider()

    # ---- 今週の分析レース（ショートカット登録）---- #
    st.subheader("今週の分析レース")
    st.caption("最大3レースを登録。レース分析タブでワンクリック選択できます。")

    # JSONファイルで永続化
    _weekly_path = Path(__file__).parent / "data" / "weekly_races.json"
    if "weekly_races" not in st.session_state:
        import json as _js
        try:
            if _weekly_path.exists():
                st.session_state["weekly_races"] = _js.loads(_weekly_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if "weekly_races" not in st.session_state:
        st.session_state["weekly_races"] = []

    _venues_list = ["東京", "中山", "阪神", "京都", "中京", "新潟", "小倉", "福島", "函館", "札幌"]
    _wr = st.session_state["weekly_races"]

    # 登録済みレース：分析ボタン（→レース分析タブへ）+ 削除ボタン
    _to_delete = None
    for _idx, _wr_item in enumerate(_wr):
      with st.container(border=True):
        _c1, _c2, _c3 = st.columns([3, 2, 1])
        with _c1:
            _day_str = _wr_item.get('day', '')
            _tfjv_str = _wr_item.get('tfjv_csv', '')
            if _tfjv_str:
                _src_pill = pill("TFJV", "accent")
            elif _wr_item.get('race_id'):
                _src_pill = pill("race_id", "buy")
            else:
                _src_pill = pill("auto", "mute")
            st.markdown(
                f"<div style='font-weight:500;font-size:15px;color:{COLOR['text']};margin-bottom:4px;'>"
                f"{_wr_item.get('label', '未設定')}</div>"
                f"<div style='color:{COLOR['muted']};font-size:13px;'>"
                f"{_day_str}曜 · {_wr_item.get('venue','')} {_wr_item.get('race_no','')}R &nbsp; {_src_pill}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with _c2:
            # EFF-1: お任せモード — 「1クリックで分析自動実行」フラグも立てる
            _bc1, _bc2 = st.columns(2)
            with _bc1:
                if st.button("設定のみ", key=f"home_go_race_{_idx}", use_container_width=True,
                             help="レース分析タブに移動して手動で「分析スタート」を押す"):
                    _tfjv_h = _wr_item.get("tfjv_csv", "")
                    _rid_h  = _wr_item.get("race_id", "")
                    if _tfjv_h and Path(_tfjv_h).exists():
                        st.session_state["_tfjv_csv_path"] = _tfjv_h
                        # 第45波: TFJV路線でも登録レースの race_id を session に渡す
                        if _rid_h:
                            st.session_state["preselected_race_id"] = _rid_h
                        st.session_state.pop("_direct_race_id", None)
                    elif _rid_h and len(_rid_h) == 12 and _rid_h.isdigit():
                        st.session_state["_direct_race_id"]     = _rid_h
                        st.session_state["preselected_race_id"] = _rid_h
                        st.session_state.pop("_tfjv_csv_path", None)
                    else:
                        st.session_state["wr_selected_venue"]   = _wr_item["venue"]
                        st.session_state["wr_selected_race_no"] = _wr_item["race_no"]
                        st.session_state["wr_selected_day"]     = _wr_item.get("day", "")
                        st.session_state.pop("_tfjv_csv_path", None)
                        st.session_state.pop("_direct_race_id", None)
                    st.session_state["wr_active"] = True
                    st.session_state.pop("_auto_analyze", None)
                    st.success("レース設定完了。画面上部の「レース分析」タブをクリックしてください。")
            with _bc2:
                if st.button("お任せ", key=f"home_auto_{_idx}", use_container_width=True, type="primary",
                             help="設定 + 分析スタートまで全自動で実行"):
                    _tfjv_h = _wr_item.get("tfjv_csv", "")
                    _rid_h  = _wr_item.get("race_id", "")
                    if _tfjv_h and Path(_tfjv_h).exists():
                        st.session_state["_tfjv_csv_path"] = _tfjv_h
                        if _rid_h:
                            st.session_state["preselected_race_id"] = _rid_h
                    elif _rid_h and len(_rid_h) == 12 and _rid_h.isdigit():
                        st.session_state["_direct_race_id"]     = _rid_h
                        st.session_state["preselected_race_id"] = _rid_h
                    st.session_state["_auto_analyze"] = True   # 分析タブ側で検知して自動実行
                    st.session_state["wr_active"] = True
                    st.success("お任せモード起動。「レース分析」タブに移動すると自動で分析が走ります。")
        with _c3:
            if st.button("削除", key=f"del_wr_{_idx}"):
                _to_delete = _idx
    if _to_delete is not None:
        _wr.pop(_to_delete)
        st.session_state["weekly_races"] = _wr
        _weekly_path.parent.mkdir(exist_ok=True)
        _weekly_path.write_text(__import__("json").dumps(_wr, ensure_ascii=False), encoding="utf-8")
        st.rerun()

    # 新規追加フォーム（第45波: 会場・R・曜日だけ入力。残りは自動取得）
    if len(_wr) < 3:
        with st.expander("＋ レースを追加", expanded=(len(_wr) == 0)):
            _nw_col1, _nw_col2, _nw_col3 = st.columns([2, 1, 1])
            with _nw_col1:
                _nw_venue = st.selectbox("会場", _venues_list, key="new_wr_venue")
            with _nw_col2:
                _nw_rno = st.number_input("R番号", 1, 12, 11, key="new_wr_rno")
            with _nw_col3:
                _nw_day = st.radio("曜日", ["土", "日"], horizontal=True, key="new_wr_day")
            st.caption("距離・頭数・芝ダ・race_id は登録時に自動取得（TFJV出馬表分析 > netkeiba の順）")

            if st.button("追加する", key="add_weekly_race", type="primary"):
                # 自動取得：まず TFJV出馬表分析CSVから検索、なければ netkeiba
                _auto_meta = {}
                # 1) TFJV出馬表分析
                _tj_path = ""
                try:
                    from pathlib import Path as _P
                    from tfjv_entries import list_races as _tj_list
                    _tj_files = sorted(_P("C:/TFJV/TXT").glob("出馬表分析*.CSV"),
                                       key=lambda p: p.stat().st_mtime, reverse=True)
                    for _tj in _tj_files:
                        _races = _tj_list(_tj)
                        for _r in _races:
                            if _r["venue"] == _nw_venue and int(_r["race_no"]) == int(_nw_rno):
                                _auto_meta = {
                                    "race_id":    _r["race_id"],
                                    "distance":   int(_r["distance"]),
                                    "field_size": int(_r["field_size"]),
                                    "surface":    _r["surface"],
                                    "source":     "TFJV",
                                }
                                _tj_path = str(_tj)
                                break
                        if _auto_meta:
                            break
                except Exception:
                    pass
                # 第45波: netkeiba フォールバック削除（TFJV出馬表分析CSVから取得できない場合はデフォルト値）

                if not _auto_meta:
                    st.warning("TFJV出馬表分析CSVから該当レースが見つかりませんでした。出馬表分析CSVをC:/TFJV/TXT/に配置してください。")
                    _auto_meta = {"race_id": "", "distance": 1800, "field_size": 16,
                                  "surface": "芝", "source": "未取得"}

                _new_entry = {
                    "venue":      _nw_venue,
                    "race_no":    int(_nw_rno),
                    "day":        _nw_day,
                    "label":      f"{_nw_venue}{_nw_rno}R",
                    "race_id":    _auto_meta.get("race_id", ""),
                    "tfjv_csv":   _tj_path,  # TFJVソースなら出馬表分析CSVパスを保存
                    "distance":   _auto_meta["distance"],
                    "field_size": _auto_meta["field_size"],
                    "surface":    _auto_meta["surface"],
                }
                _wr.append(_new_entry)
                st.session_state["weekly_races"] = _wr
                _weekly_path.parent.mkdir(exist_ok=True)
                _weekly_path.write_text(__import__("json").dumps(_wr, ensure_ascii=False), encoding="utf-8")
                st.success(
                    f"追加: {_nw_day}曜 {_new_entry['label']} "
                    f"({_auto_meta['surface']}{_auto_meta['distance']}m・{_auto_meta['field_size']}頭 / 取得元:{_auto_meta['source']})"
                )
                st.rerun()

    st.divider()


# ============================================================
# TAB 1: レース詳細分析
# ============================================================
with tab_race:
    st.subheader("レース詳細分析")

    # ---- 今週のレース ショートカット ---- #
    _wr_list = st.session_state.get("weekly_races", [])
    if _wr_list:
        st.markdown("**今週の分析レース**")
        _wr_cols = st.columns(len(_wr_list))
        for _wi, (_wrc, _witem) in enumerate(zip(_wr_cols, _wr_list)):
            with _wrc:
                _day_label = _witem.get("day", "")
                if st.button(
                    f"{_witem['label']}\n{_day_label}{'曜 ' if _day_label else ''}{_witem['venue']} {_witem['race_no']}R",
                    key=f"wr_btn_{_wi}",
                    use_container_width=True,
                ):
                    _tfjv_csv = _witem.get("tfjv_csv", "")
                    _rid_d    = _witem.get("race_id", "")
                    if _tfjv_csv and Path(_tfjv_csv).exists():
                        # TFJVモード: 次の rerun で CSV から race_id 抽出
                        st.session_state["_tfjv_csv_path"] = _tfjv_csv
                        st.session_state.pop("_direct_race_id", None)
                    elif _rid_d and len(_rid_d) == 12 and _rid_d.isdigit():
                        # race_id直接指定 — fetch_today_races 不要
                        st.session_state["_direct_race_id"]     = _rid_d
                        st.session_state["preselected_race_id"] = _rid_d
                        st.session_state.pop("_tfjv_csv_path", None)
                    else:
                        # 従来の venue マッチング
                        st.session_state["wr_selected_venue"]   = _witem["venue"]
                        st.session_state["wr_selected_race_no"] = _witem["race_no"]
                        st.session_state["wr_selected_day"]     = _witem.get("day", "")
                        st.session_state.pop("_tfjv_csv_path", None)
                        st.session_state.pop("_direct_race_id", None)
                    st.session_state["wr_active"] = True
                    st.rerun()
        st.divider()

    # 第45波: TFJV一本に集約。netkeibaは「ブリンカー・馬体重・馬場状態・リアルタイムオッズ」専用。
    # 手動race_id入力・TFJV CSVパス手入力・データ取得方法ラジオを全て削除。
    fetch_mode = "TFJV出馬表分析"

    races = []

    # 第45波: TFJV出馬表分析モード
    if fetch_mode == "TFJV出馬表分析":
        from pathlib import Path as _Path
        _tfjv_dir = _Path("C:/TFJV/TXT")
        _candidates = sorted(_tfjv_dir.glob("出馬表分析*.CSV"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not _candidates:
            st.warning("C:/TFJV/TXT に「出馬表分析*.CSV」が見つかりません。Targetから出力してください。")
        else:
            _tfjv_choice = st.selectbox(
                "出馬表分析CSV を選択",
                [str(p) for p in _candidates],
                key="tfjv_entries_csv",
            )
            try:
                from tfjv_entries import list_races as _tfjv_list
                _tfjv_races = _tfjv_list(_tfjv_choice)
                races = _tfjv_races
                st.success(f"{len(races)}レース読込（{_Path(_tfjv_choice).name}）")
            except Exception as _e_tj:
                st.error(f"TFJV取込エラー: {_e_tj}")

    # 第45波: 「今週の分析レース」が1件登録されているのに preselected_race_id が空 or 違う場合、
    # 自動的に先頭の登録レースを preselected にセット（ホーム経由クリックを省略可能に）
    _wr_for_auto = st.session_state.get("weekly_races", [])
    if (_wr_for_auto and not st.session_state.get("preselected_race_id")
            and not st.session_state.get("_tfjv_csv_path")):
        _first_wr = _wr_for_auto[0]
        _first_rid = _first_wr.get("race_id", "")
        _first_tfjv = _first_wr.get("tfjv_csv", "")
        if _first_rid and _first_tfjv and Path(_first_tfjv).exists():
            st.session_state["preselected_race_id"] = _first_rid
            st.session_state["tfjv_entries_csv_path"] = _first_tfjv
            # 古い selectbox state クリア
            st.session_state.pop("race_selectbox", None)
            st.session_state.pop("venue_selectbox", None)
            st.info(f"📌 登録済みレース「{_first_wr.get('label', '')}」を自動で選択します")
            st.rerun()

    # TFJVモード: CSVから race_id を抽出して _direct_race_id にセット → 次の rerun で直接取得経路へ
    _tfjv_path = st.session_state.get("_tfjv_csv_path", "")
    if _tfjv_path:
        if not Path(_tfjv_path).exists():
            st.error(f"TFJV CSVファイルが見つかりません: {_tfjv_path}")
            st.session_state.pop("_tfjv_csv_path", None)
            st.stop()
        # 第45波: ホームで登録済みの race_id があればそれを優先（特定のレースを選ぶ）
        _registered_rid = st.session_state.get("preselected_race_id", "")
        if _registered_rid:
            _rid_from_tfjv = _registered_rid
        else:
            _rid_from_tfjv = extract_race_id_from_tfjv_csv(_tfjv_path)
        if _rid_from_tfjv:
            # 第45波: 出馬表分析CSVの場合は CSVパスを session に残す + preselected_race_id だけ設定
            # → 下流のドロップダウンUIが preselected を見て自動選択する
            if "出馬表分析" in Path(_tfjv_path).name:
                st.session_state["tfjv_entries_csv_path"] = _tfjv_path
                st.session_state.pop("_direct_race_id", None)
            else:
                st.session_state["_direct_race_id"] = _rid_from_tfjv
            st.session_state["preselected_race_id"] = _rid_from_tfjv
            # 第45波: ホーム経由は「新しい指示」なので古い selectbox state を完全リセット
            st.session_state.pop("race_selectbox", None)
            st.session_state.pop("venue_selectbox", None)
            st.session_state.pop("_tfjv_csv_path", None)
            st.success(f"TFJVからrace_id={_rid_from_tfjv}を読込")
            st.rerun()
        else:
            st.error(f"CSVから race_id を抽出できませんでした: {_tfjv_path}")
            st.session_state.pop("_tfjv_csv_path", None)
            st.stop()

    # _direct_race_id がある場合は fetch_today_races を完全スキップして直接取得
    _direct_rid = st.session_state.get("_direct_race_id", "")

    # 第45波: netkeiba 12桁 race_id 経路を廃止（TFJV一本化）
    if fetch_mode in ("netkeibaから自動取得", "TFJV出馬表分析"):
        # 第45波: TFJVモードでも会場/レース選択UIを通せるよう条件を緩和
        # 今週末の日付リスト
        # 今週末 + サイドバーの対象日を合わせて試行（元の動作に戻す）
        _dates_to_try = list(dict.fromkeys(get_this_weekend_dates() + [date_str]))

        # 今週末・対象日以外の古い preselected_race_id を破棄
        _pre_raw = st.session_state.get("preselected_race_id", "")
        if _pre_raw and _pre_raw[:8] not in _dates_to_try:
            st.session_state.pop("preselected_race_id", None)
        preselected = st.session_state.get("preselected_race_id")

        # 第45波: netkeiba連携を全削除（races は TFJV出馬表分析CSVから既に populated）
        if not races:
            st.error("レース一覧を取得できませんでした。出馬表分析CSVを再選択してください。")
            fetch_mode = "手動入力"
        else:
            st.caption(f"{len(races)}レース利用可能")

    if not _direct_rid:
        # 直接取得モード以外のみ初期化（直接取得時は既に entries/meta がセット済み）
        entries, surface, distance, track_condition, venue = [], "芝", 2000, "良", "東京"

    # 第45波: TFJVモードでも下流が参照するので preselected を必ず初期化
    if "preselected" not in dir():
        preselected = st.session_state.get("preselected_race_id")

    if not _direct_rid and fetch_mode in ("netkeibaから自動取得", "TFJV出馬表分析") and races:
        # 会場コード→会場名マッピング
        _VENUE_MAP = {
            "01":"札幌","02":"函館","03":"福島","04":"新潟","05":"東京",
            "06":"中山","07":"中京","08":"京都","09":"阪神","10":"小倉",
        }
        # 第45波修正: race_id 先頭8桁は「年+会場+開催回+開催日コード」で日付ではない
        # → race["date_str"]（fetch_today_races がセット済み）から日付ラベルを作る
        _YOUBI = ["月","火","水","木","金","土","日"]
        _DATE_LABEL: dict[str, str] = {}
        for _r in races:
            _ds = str(_r.get("date_str", ""))
            if _ds and _ds not in _DATE_LABEL:
                try:
                    from datetime import datetime as _dt
                    _dobj = _dt.strptime(_ds, "%Y%m%d")
                    _DATE_LABEL[_ds] = f"{_dobj.month}/{_dobj.day}({_YOUBI[_dobj.weekday()]})"
                except Exception:
                    _DATE_LABEL[_ds] = _ds[4:]

        # 第45波: 日付フィルタは廃止。会場ラベル「6/14(日) 函館」だけで実質的に日付も決まる
        # レースを「日付+会場」グループに分類（日付昇順）
        # 第45波修正: TFJV race_id (14桁: YYYYMMDD+venue+race+race) と netkeiba (12桁: YYYY+venue+kaisai+day+R) で
        # venue 抽出位置が違う。tfjv_entries が返す r["venue"] を優先使用、なければ race_id 位置から推測
        _venue_groups: dict[str, list] = {}
        for r in sorted(races, key=lambda x: (x.get("date_str", ""), x["race_id"])):
            _rid = r["race_id"]
            _venue_from_record = r.get("venue", "")
            if _venue_from_record:
                _vname = _venue_from_record
            elif len(_rid) == 14:
                # TFJV: pos 8-10 が venue code
                _vname = _VENUE_MAP.get(_rid[8:10], f"会場{_rid[8:10]}")
            else:
                # netkeiba: pos 4-6 が venue code
                _vname = _VENUE_MAP.get(_rid[4:6], f"会場{_rid[4:6]}")
            _dlabel = _DATE_LABEL.get(str(r.get("date_str", "")), "")
            _key = f"{_dlabel} {_vname}" if _dlabel else _vname
            _venue_groups.setdefault(_key, []).append(r)

        # 会場を選んでからレースを選ぶ2段階UI
        _venue_names = list(_venue_groups.keys())
        # ショートカットボタンで選択された会場を優先
        _wr_venue   = st.session_state.pop("wr_selected_venue", None)
        _wr_race_no = st.session_state.pop("wr_selected_race_no", None)
        _wr_day     = st.session_state.pop("wr_selected_day", None)
        _venue_matched = False
        if _wr_venue:
            # 「5/25(日) 東京」のように曜日+会場名でマッチ
            _wr_candidates = [
                vn for vn in _venue_names
                if _wr_venue in vn and (_wr_day == "" or _wr_day is None or f"({_wr_day})" in vn)
            ]
            if _wr_candidates:
                st.session_state["venue_selectbox"] = _wr_candidates[0]
                st.session_state["_pending_race_no"] = _wr_race_no
                _venue_matched = True
        elif preselected:
            # 第45波: 自動選択は「未設定 or 古い値」の時のみ。ユーザー手動選択を尊重
            _venue_need_auto = (
                "venue_selectbox" not in st.session_state
                or st.session_state.get("venue_selectbox") not in _venue_names
            )
            if _venue_need_auto:
                for vn, vr in _venue_groups.items():
                    if any(r["race_id"] == preselected for r in vr):
                        st.session_state["venue_selectbox"] = vn
                        break

        # stale な venue_selectbox 値（レース再取得後に存在しなくなった会場）をクリア
        if st.session_state.get("venue_selectbox") not in _venue_names:
            st.session_state.pop("venue_selectbox", None)

        selected_venue = st.selectbox("会場を選択", _venue_names, key="venue_selectbox")
        _venue_races = _venue_groups[selected_venue]
        # 第45波: TFJVモードは全R表示。netkeibaモードのみ9R以降に絞る（重賞中心）
        if fetch_mode == "netkeibaから自動取得":
            _venue_races = [r for r in _venue_races if int(r["race_id"][-2:]) >= 9]
        if not _venue_races:
            st.warning("この会場にレースがありません。")
        race_options = {f"{r['race_name']}": r["race_id"] for r in _venue_races}

        # 第45波: 自動選択は「race_selectbox が未設定 or 別会場の値が残っている」時のみ実行
        # ユーザーが手動で選んだあとは preselected で毎回上書きしない
        _need_auto_select = (
            "race_selectbox" not in st.session_state
            or st.session_state.get("race_selectbox") not in list(race_options.keys())
        )
        _pending_rno = st.session_state.pop("_pending_race_no", None)
        if _need_auto_select:
            if _pending_rno:
                for _rname, _rid in race_options.items():
                    if int(_rid[-2:]) == _pending_rno:
                        st.session_state["race_selectbox"] = _rname
                        break
            elif preselected and preselected in race_options.values():
                _rname_pre = next((k for k, v in race_options.items() if v == preselected), None)
                if _rname_pre:
                    st.session_state["race_selectbox"] = _rname_pre

        selected_label = st.selectbox("レースを選択", list(race_options.keys()), key="race_selectbox")
        selected_race_id = race_options[selected_label]
        st.session_state["_race_label"] = selected_label
        # 第45波: preselected_race_id は「初期選択用」として残し、毎回上書きしない
        # （上書きすると次回 auto-select で同じ値に固定される不具合あり）
        # ただし下流コード（追い切り取得など）が参照するので、現在の選択を反映しておく
        st.session_state["preselected_race_id"] = selected_race_id

        with st.spinner("出走馬・オッズを取得中..."):
            # 第45波: TFJV一本に固定。netkeibaフォールバックは廃止
            from tfjv_entries import load_tfjv_entries as _tj_load
            _tj_csv = st.session_state.get("tfjv_entries_csv_path", "") \
                      or st.session_state.get("tfjv_entries_csv", "")
            if not _tj_csv:
                st.error("TFJV出馬表分析CSVが選択されていません")
                st.stop()
            _tj_data = _tj_load(_tj_csv)
            _race_data = _tj_data.get(selected_race_id, {})
            entries = _race_data.get("entries", [])
            meta = {
                "surface":         _race_data.get("surface", "芝"),
                "distance":        _race_data.get("distance", 2000),
                "track_condition": "良",
                "field_size":      _race_data.get("field_size", len(entries)),
                "race_name":       _race_data.get("race_class", selected_label),
                "venue":           _race_data.get("venue", ""),
            }
        # 第22波 (T2 修正): venue/surface/distance/track_condition は読出31箇所に対し
        # 書込ゼロの死にキー群だった（アプリ全体が常に東京・芝・1800m・良 前提で動作）
        # → メタ取得直後に session_state へ保存し全下流を実値化
        for _mk in ("venue", "surface", "distance", "track_condition"):
            if meta and meta.get(_mk):
                st.session_state[_mk] = meta[_mk]

        surface = meta.get("surface", "芝")
        distance = meta.get("distance", 2000)
        track_condition = meta.get("track_condition", "良")
        # 第45波: venue ローカル変数を meta から取得（既存バグ: 初期化値"東京"のまま下流に流れていた）
        venue = meta.get("venue", venue if "venue" in dir() else "東京")
        # session_state にも反映
        st.session_state["venue"] = venue
        race_name_display = meta.get("race_name", "")
        _title = f"**{race_name_display} | {venue}**　{surface} {distance}m / 馬場: {track_condition}" if race_name_display else f"**{venue}**　{surface} {distance}m / 馬場: {track_condition}"
        st.info(_title)

        # 第45波: netkeiba 過去成績取得を削除。過去成績は TFJV DSファイル経由で horse_latest_features に集約済み
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


    # ---- TFJV調教データ：キャッシュから自動読み込み ----
    # （手動入力不要。Claudeとのチャットで「エクスポートしました」と報告するだけ）
    if "tfjv_training_results" not in st.session_state:
        from tfjv_training import load_training_cache as _ltc
        _cached_tfjv = _ltc()
        if _cached_tfjv:
            st.session_state["tfjv_training_results"] = _cached_tfjv

    _existing_tfjv = st.session_state.get("tfjv_training_results", {})
    if _existing_tfjv:
        _got_tfjv = sum(1 for v in _existing_tfjv.values() if v.get("sessions_count", 0) > 0)
        # 第45波: 全馬数は誤解を生むので「DB全体」と明示
        st.caption(f"調教DB保有: {_got_tfjv}頭（今週末以外も含む全データ）。レース選択時に該当馬のみ評価対象。")
    else:
        st.caption("調教データ未取得（TFJVからエクスポート後、Claudeに報告してください）")

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

    # ---- 天気予報・馬場状態予測 ---- #
    from scraper import fetch_weather_forecast
    _wx_col1, _wx_col2 = st.columns([1, 2])
    with _wx_col1:
        # ISSUE-1: 直接モード / 自動取得モード共通で venue 変数を使う
        _wx_venue = venue
    _race_date_for_wx = current_date_str  # YYYY-MM-DD 形式
    if _wx_venue:
        _wx = fetch_weather_forecast(_wx_venue, _race_date_for_wx)
        if _wx.get("raw_code", -1) >= 0:
            _rain = _wx["precipitation_mm"]
            _cond = _wx["condition_forecast"]
            _icon = "晴" if _rain == 0 else ("雨" if _rain >= 5 else "小雨")
            st.info(
                f"**{_wx_venue} {_race_date_for_wx} 天気予報**  \n"
                f"[{_icon}] {_wx['weather']}　降水量: {_rain}mm  \n"
                f"馬場予測: **{_cond}**"
            )

    # ---- 当日バイアス入力 ---- #
    selected_race_id_for_bias = st.session_state.get("preselected_race_id", "")

    # TRAIN-3: 手動入力モード時のみ race_id を1行で入力（自動取得モードはTRAIN-1で自動セット済み）
    if not selected_race_id_for_bias and fetch_mode != "netkeibaから自動取得":
        _manual_rid = st.text_input(
            "netkeibaレースID（12桁）",
            placeholder="例: 202605250511　追い切りデータ取得に必要",
            key="manual_race_id_input",
        )
        if _manual_rid and _manual_rid.strip().isdigit() and len(_manual_rid.strip()) == 12:
            st.session_state["preselected_race_id"] = _manual_rid.strip()
            selected_race_id_for_bias = _manual_rid.strip()

    bias_type = render_bias_input_panel(selected_race_id_for_bias or None)

    # ---- 分析実行 ---- #
    # EFF-1: お任せモード — _auto_analyze フラグが立っていればボタンクリックなしで自動実行
    _auto_run = bool(st.session_state.pop("_auto_analyze", False))
    _btn_pressed = st.button("分析スタート", type="primary", use_container_width=True)
    if _auto_run:
        st.info("お任せモード：分析を自動実行中...")
    if _btn_pressed or _auto_run:
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

                # horse_stats ボーナス（距離適性・道悪・上がり・外国人騎手・コース特性）
                from horse_stats import get_horse_score_bonus
                hs_result = get_horse_score_bonus(
                    horse_name=name.strip(),
                    horse_stats=horse_stats_df,
                    distance_cat=dist_cat,
                    surface=surface,
                    track_condition=track_condition,
                    jockey=e.get("jockey", ""),
                    venue=venue,
                    running_style=e.get("running_style", "不明"),
                )
                e["horse_stats_bonus"]   = hs_result["bonus"]
                e["horse_stats_details"] = hs_result["details"]

                # 調教スコア（TFJVデータ優先 → netkeiba fallback）
                _tfjv_tr = st.session_state.get("tfjv_training_results", {})
                if name in _tfjv_tr:
                    _tev = _tfjv_tr[name]
                    e["training_bonus"]  = _tev["bonus"]
                    e["training_label"]  = _tev["label"]
                    e["training_detail"] = _tev["detail"]
                else:
                    # 第19波 (V3): "training_data" は架空キー（実キーは
                    # training_data_results、評価済み {name: {score,label,detail,bonus}} 形式）
                    # → netkeiba 調教フォールバックが常にスキップされていた
                    _tr_results = st.session_state.get("training_data_results", {})
                    if name in _tr_results:
                        _tev2 = _tr_results[name]
                        # 第28波: training_fetcher の score は ±0.5 級（勝率補正スケール）、
                        # confluence は ±5 点スケールを想定 → ×10 変換しないと
                        # 調教評価が 1/10 に薄まりほぼ無効だった
                        e["training_bonus"]  = float(_tev2.get("bonus", 0.0)) * 10.0
                        e["training_label"]  = _tev2.get("label", "調教データなし")
                        e["training_detail"] = _tev2.get("detail", "")
                    else:
                        # 第46波: Target坂路CSVは美浦・栗東のみ出力可能（ローカル開催滞在馬は対象外）
                        e["training_bonus"]  = 0.0
                        e["training_label"]  = "坂路対象外"
                        e["training_detail"] = "美浦・栗東坂路CSVに記録なし（ローカル滞在馬は取得不可）"

            entries_full = entries_full_bias

            # EV評価 → 全補正値が horse dict に入っているのでそのまま渡す
            eval_df = evaluate_race(entries_full, win_rate_table, sire_stats, jockey_stats)

            # ---- レース内正規化（ROOT-1: Isotonic Calibration後はsum-normalize） ----
            from ev_calculator import _CALIBRATOR as _calibrator_ref  # 1度だけimport（二重import防止）
            if "lgbm_win_rate" in eval_df.columns and eval_df["lgbm_win_rate"].notna().any():
                scores = eval_df["lgbm_win_rate"].fillna(0).values.astype(float)
                if _get_lgbm_model_type() == "ranker":
                    if _calibrator_ref is not None:
                        # キャリブレーション済み確率 → sum-normalize
                        _total = scores.sum()
                        eval_df["lgbm_norm_pct"] = (scores / _total * 100) if _total > 0 else 100.0 / len(eval_df)
                    else:
                        # フォールバック: 動的温度softmax
                        _score_range = scores.max() - scores.min()
                        _tau = max(_score_range / 5.0, 1.0)
                        _scaled = scores / _tau
                        _e = np.exp(_scaled - _scaled.max())
                        eval_df["lgbm_norm_pct"] = _e / _e.sum() * 100
                else:
                    # Classifier: 合計で割って正規化
                    _total = scores.sum()
                    eval_df["lgbm_norm_pct"] = (scores / _total * 100) if _total > 0 else 100.0 / len(eval_df)

                eval_df["market_pct"]  = 100.0 / eval_df["odds"].clip(lower=1.01)
                eval_df["value_ratio"] = eval_df["lgbm_norm_pct"] / eval_df["market_pct"].clip(lower=0.1)

                # SUPER-1: Benter Odds Blending — 自モデル × 市場（人気→実勝率）統合
                from ev_calculator import blend_with_market as _blend
                from market_prob import get_market_prob as _gmp
                _benter_on = st.session_state.get("benter_enabled", True)
                if _benter_on and "popularity" in eval_df.columns:
                    _alpha = float(st.session_state.get("benter_alpha", 0.7))
                    _beta  = float(st.session_state.get("benter_beta",  0.3))
                    # DEAD-1: 市場確率を 2源平均 — 人気順位ベース + 実オッズ補正ベース
                    from favorite_longshot import correct_from_odds as _cfo
                    _mkt_pop = eval_df["popularity"].apply(_gmp).values
                    _mkt_odds = eval_df["odds"].apply(
                        lambda o: _cfo(o) if pd.notna(o) and o > 1.0 else None
                    ).values
                    # 平均化（実オッズ補正が取れない馬は人気ベースのみ）
                    import numpy as _np
                    _mkt = _np.array([
                        0.5 * po + 0.5 * mo if mo is not None else po
                        for po, mo in zip(_mkt_pop, _mkt_odds)
                    ])
                    # 自モデル確率（softmax正規化済み）
                    _mdl = (eval_df["lgbm_norm_pct"] / 100).values
                    _blended = _blend(_mdl, _mkt, alpha=_alpha, beta=_beta)
                    eval_df["blended_pct"] = _blended * 100
                    eval_df["market_emp_pct"] = _mkt * 100
                    _p = pd.Series(_blended, index=eval_df.index)
                else:
                    eval_df["blended_pct"]   = eval_df["lgbm_norm_pct"]
                    eval_df["market_emp_pct"] = eval_df["market_pct"]
                    _p = eval_df["lgbm_norm_pct"] / 100

                # NEW-6: アンサンブル（任意・ON時のみ）
                if st.session_state.get("ensemble_enabled", False) and "blended_pct" in eval_df.columns:
                    from ensemble import ensemble_blend as _eblend
                    _ens_w = (
                        float(st.session_state.get("ens_w_lgbm",    0.7)),
                        float(st.session_state.get("ens_w_market",  0.2)),
                        float(st.session_state.get("ens_w_uniform", 0.1)),
                    )
                    _ens_mkt = eval_df["popularity"].apply(_gmp).values
                    _ens_arr = _eblend(_p.values, _ens_mkt, weights=_ens_w)
                    eval_df["ensemble_pct"] = _ens_arr * 100
                    _p = pd.Series(_ens_arr, index=eval_df.index)
                else:
                    eval_df["ensemble_pct"] = eval_df["blended_pct"]

                # A-2: Stacking モード（XGB + CB と統合）— 有効時は _p を上書き
                if st.session_state.get("stacking_enabled", False):
                    try:
                        from stacking_predictor import predict_xgb, predict_catboost, stack_blend_probs
                        _xgb_raw = []
                        _cb_raw = []
                        for _h in entries_full:
                            _xgb_raw.append(predict_xgb(_h) or 0.0)
                            _cb_raw.append(predict_catboost(_h) or 0.0)
                        _xgb_raw = np.asarray(_xgb_raw, dtype=float)
                        _cb_raw  = np.asarray(_cb_raw,  dtype=float)
                        # レース内 softmax
                        def _sm(x):
                            x = x - x.max()
                            e = np.exp(x); return e / e.sum() if e.sum() else x
                        _xgb_p = _sm(_xgb_raw)
                        _cb_p  = _sm(_cb_raw)
                        _stacked = stack_blend_probs(_p.values, _xgb_p, _cb_p)
                        eval_df["stacked_pct"] = _stacked * 100
                        # 第20波 (U6 修正): Harville タブ等は blended_pct を最終確率として
                        # 読むため、stacking 適用後はこちらも上書きしないと多券種EVだけ
                        # 古い確率で計算される不整合があった
                        eval_df["blended_pct"] = _stacked * 100
                        _p = pd.Series(_stacked, index=eval_df.index)
                    except Exception as _e_st:
                        print(f"[stacking] 推論失敗、LightGBM 単独にフォールバック: {_e_st}")

                # NEW-2: calibratorの有無に関わらず blended から常にEVを再計算
                eval_df["ev"] = _p * (eval_df["odds"] - 1) - (1 - _p)
                eval_df["ev"] = eval_df["ev"].fillna(-0.25)
                eval_df["ev_label"] = eval_df["ev"].apply(ev_label)

                # 第13波: B 馬券内特化サブモデルで place_prob 列を追加
                try:
                    from place_predictor import predict_place_prob as _ppp
                    _src = eval_df.copy()
                    for _col, _val in [("surface", st.session_state.get("surface", "芝")),
                                        ("venue",   st.session_state.get("venue",   "東京")),
                                        ("track_condition",
                                         st.session_state.get("track_condition", "良")),
                                        ("month",   pd.Timestamp.now().month),
                                        ("distance", st.session_state.get("distance", 1800))]:
                        if _col not in _src.columns:
                            _src[_col] = _val
                    eval_df["place_prob"] = _ppp(_src)
                    # 第20波 (U7): レース内正規化 — 3着内確率の合計は理論上 3.0。
                    # 実測で 2.7〜3.15 のレース間ばらつきがあり、正規化で複勝EVの
                    # 精度が上がる（sum が異常なレースは正規化しないガード付き）
                    _pp_sum = float(eval_df["place_prob"].sum())
                    if 1.5 <= _pp_sum <= 4.5 and len(eval_df) >= 8:
                        eval_df["place_prob"] = (eval_df["place_prob"] * 3.0 / _pp_sum).clip(0.005, 0.95)
                    eval_df["place_pct"]  = (eval_df["place_prob"] * 100).round(1)

                    # 第14波→第17波改良: 複勝EV（穴狙いの本丸）
                    # 複勝オッズ近似は人気帯別の逓減係数（単一係数 0.27 は大穴で
                    # 過大評価になり、複勝EVが膨らみ買いすぎる危険があった）
                    #   単勝 ~10倍: ×0.30 / 10~30倍: ×0.22 / 30倍~: ×0.15
                    _w = eval_df["odds"].astype(float)
                    # 第32波: 段差係数（10/30倍境界で不連続）が買い候補の人気構成を
                    # 歪めていた（8人気以降が急減）→ 線形補間の連続関数に
                    _coef = pd.Series(
                        np.clip(0.30 - 0.0075 * (_w - 10).clip(lower=0), 0.15, 0.30),
                        index=eval_df.index)
                    _place_odds_est = 1.0 + (_w - 1.0).clip(lower=0) * _coef
                    # 第30波: scraper が取得した複勝オッズ実値（min/max中間値）が
                    # あれば近似より優先（実データを取りながら近似で捨てていた）
                    if "place_odds" in eval_df.columns:
                        _po_real = pd.to_numeric(eval_df["place_odds"], errors="coerce")
                        _place_odds_est = _po_real.where(_po_real > 1.0, _place_odds_est)
                    eval_df["place_odds_est"] = _place_odds_est.round(2)
                    eval_df["place_ev"] = (
                        eval_df["place_prob"] * (_place_odds_est - 1.0)
                        - (1.0 - eval_df["place_prob"])
                    ).round(3)
                except Exception as _e_place:
                    print(f"[place_predictor] 失敗、複勝確率の付与スキップ: {_e_place}")

                # 第13波: C レアパターン（爆穴モードのみ confluence へ加算）
                try:
                    from rare_patterns import score_rare_bonus as _srb
                    if st.session_state.get("betting_mode") == "爆穴":
                        _ctx = {
                            "surface":  st.session_state.get("surface", "芝"),
                            "distance": st.session_state.get("distance", 1800),
                            "venue":    st.session_state.get("venue",   ""),
                            "track_condition":
                                st.session_state.get("track_condition", "良"),
                            "pace_predicted":
                                st.session_state.get("pace_info", {}).get("predicted_pace", ""),
                        }
                        def _apply_rare(row):
                            try:
                                pts, names = _srb(row.to_dict(), _ctx)
                                return pd.Series({"rare_bonus": pts,
                                                  "rare_labels": " / ".join(names)})
                            except Exception:
                                return pd.Series({"rare_bonus": 0.0, "rare_labels": ""})
                        _add = eval_df.apply(_apply_rare, axis=1)
                        eval_df["rare_bonus"] = _add["rare_bonus"]
                        eval_df["rare_labels"] = _add["rare_labels"]
                        if "confidence_score" in eval_df.columns:
                            eval_df["confidence_score"] = (
                                eval_df["confidence_score"] + eval_df["rare_bonus"]
                            ).clip(0, 100)
                except Exception as _e_rp:
                    print(f"[rare_patterns] 加算スキップ: {_e_rp}")

                # 第14波: 穴馬総合スコア（longshot_score 0-100）
                # 複勝EV(50%) + 馬券内確率(30%) + レアパターン(20%) の合成。人気薄のみ対象。
                try:
                    if "place_ev" in eval_df.columns:
                        _pop = pd.to_numeric(eval_df.get("popularity"), errors="coerce").fillna(99)
                        # 複勝EV: -1〜+1 想定 → 0〜100 へ
                        # 第33波: 実測の place_ev 分布 [-0.3, +0.4] に合わせて正規化
                        # （旧 [-0.5, 1.0] ではスコアが 13〜60 に圧縮され分離度が死んでいた）
                        _ev_part = ((eval_df["place_ev"].clip(-0.3, 0.4) + 0.3) / 0.7 * 100)
                        # 馬券内確率: 人気薄帯では 30% あれば優秀 → 0〜30% を 0〜100 へ
                        _pp_part = (eval_df["place_prob"].clip(0, 0.30) / 0.30 * 100)
                        _rb_part = (eval_df.get("rare_bonus", pd.Series(0.0, index=eval_df.index))
                                    .fillna(0.0) / 15.0 * 100)
                        eval_df["longshot_score"] = (
                            0.5 * _ev_part + 0.3 * _pp_part + 0.2 * _rb_part
                        ).round(1)
                        # 5人気以内は穴ではないので対象外（NaN）
                        eval_df.loc[_pop <= 5, "longshot_score"] = np.nan
                        # 第40波: 初ブリンカー馬は穴スコアに大幅加点（最大サイン）
                        _bset = st.session_state.get("blinker_all_set", set())
                        _bfirst = st.session_state.get("blinker_first_set", set())
                        if _bset:
                            _nm = eval_df["horse_name"].str.strip()
                            eval_df["blinker_flag"] = _nm.isin(_bset)
                            eval_df["blinker_first"] = _nm.isin(_bfirst)
                            _add = _nm.isin(_bfirst) * 15.0 + (_nm.isin(_bset) & ~_nm.isin(_bfirst)) * 6.0
                            eval_df["longshot_score"] = (eval_df["longshot_score"].fillna(0) + _add).where(
                                eval_df["longshot_score"].notna() | (_add > 0), np.nan)
                except Exception as _e_ls:
                    print(f"[longshot_score] 計算スキップ: {_e_ls}")

            # 有力馬撃破スコア + 近走巻き返しボーナスをパイプラインに付与
            # tfjv_all から対象馬の複数走履歴を取得（horse_latest_features は1行のみで不足）
            from horse_profiler import (calc_beaten_strong_horses,
                                        analyze_recent_races,
                                        calc_resume_bonus_from_recent)
            _tfjv_full_path = Path(__file__).parent / "data" / "tfjv_all.parquet"
            _df_full_hist = pd.DataFrame()
            if _tfjv_full_path.exists():
                try:
                    _names_set = set(
                        e.get("horse_name", "").strip()
                        for e in entries_full if e.get("horse_name")
                    )
                    # PERF: シングルトンキャッシュから取得（毎回 800k行再読込を回避）
                    _df_full_cache = _load_tfjv_full()
                    _df_full_hist = _df_full_cache[
                        _df_full_cache["horse_name"].str.strip().isin(_names_set)
                    ].copy()
                    # 数値変換（旧コードを残しているが、_load_tfjv_full で変換済の場合は no-op）
                    for _nc in ["rank", "popularity", "field_size", "speed_figure",
                                "last_3f", "horse_weight", "finish_time"]:
                        if _nc in _df_full_hist.columns:
                            _df_full_hist[_nc] = pd.to_numeric(
                                _df_full_hist[_nc], errors="coerce")
                except Exception as _e_full:
                    _df_full_hist = pd.DataFrame()

            # df_hist_for_profiler: 複数走履歴があればそちらを優先
            _df_prof = _df_full_hist if not _df_full_hist.empty else df_hist

            for e in entries_full:
                name = e.get("horse_name", "")
                beaten = calc_beaten_strong_horses(_df_prof, name, surface, distance)
                e["beat_count"]       = beaten["beat_count"]
                e["beat_bonus"]       = beaten["bonus"]
                e["beat_label"]       = beaten["label"]
                e["beat_best_victim"] = beaten["best_victim"]

                # 近走詳細 → 巻き返しボーナス
                recent = analyze_recent_races(
                    _df_prof, name,
                    current_surface=surface,
                    current_distance=distance,
                    n=5,  # 直近5走（3走から拡大）
                )
                rb = calc_resume_bonus_from_recent(recent)
                e["resume_bonus_total"]   = rb["total_bonus"]
                e["resume_summary"]       = rb["summary"]
                e["resume_top_excuse"]    = rb["top_excuse"]
                e["ippen_candidate"]      = rb.get("ippen_candidate", False)

                # NEW-2: PCI / RPCI ペースチェンジ指数
                from pci_calculator import get_horse_pci_stats as _gphps, pace_pci_match as _ppmatch
                _pci = _gphps(_df_prof, name, n=5)
                e["pci_avg"]      = _pci["pci_avg"]
                e["pci_latest"]   = _pci["pci_latest"]
                e["pci_label"]    = _pci["label"]
                # BOOST-3: ペース予測 × PCI マッチ判定でボーナス上書き
                _ppm = _ppmatch(predicted_pace, _pci["pci_avg"])
                if _ppm["match_bonus"] != 0:
                    e["pci_bonus"] = _pci["bonus"] + _ppm["match_bonus"]
                    e["pci_label"] = _ppm["match_label"]
                else:
                    e["pci_bonus"] = _pci["bonus"]

                # NEW-8: 厩舎×騎手ペア成績
                from trainer_jockey_matrix import get_pair_stats as _gps
                _pair = _gps(e.get("trainer", ""), e.get("jockey", ""))
                e["pair_label"] = _pair["label"]
                e["pair_lift"]  = _pair["lift"]
                e["pair_bonus"] = _pair["bonus"]
                e["pair_note"]  = _pair["note"]

                # NEW-7: タイム指数（スピードインデックス）
                from speed_index import get_horse_speed_stats as _ghss
                _sp = _ghss(_df_prof, name, n=5)
                e["speed_index_avg"]   = _sp["speed_avg"]
                e["speed_index_best"]  = _sp["speed_best"]
                e["speed_index_label"] = _sp["speed_label"]

                # B-3: コースバイアスの恒常推定（venue×surface×distance の構造的バイアス）
                try:
                    from course_bias import get_course_bias as _gcb
                    _cb = _gcb(venue, surface, distance)
                    _gate = int(e.get("gate", 4))
                    _field = int(e.get("field_size", 12) or 12)
                    # 自分の枠が「内寄り(<center)」「外寄り(>center)」を判定
                    _center = _field / 2
                    _is_inside = _gate < _center
                    _is_outside = _gate > _center
                    # 外枠有利コースで外枠の馬 → +0.02、内枠の馬 → -0.02
                    if _cb["gate_bias"] >= 0.15 and _is_outside:
                        e["course_bias_bonus"] = 0.02
                    elif _cb["gate_bias"] >= 0.15 and _is_inside:
                        e["course_bias_bonus"] = -0.02
                    elif _cb["gate_bias"] <= -0.15 and _is_inside:
                        e["course_bias_bonus"] = 0.02
                    elif _cb["gate_bias"] <= -0.15 and _is_outside:
                        e["course_bias_bonus"] = -0.02
                    else:
                        e["course_bias_bonus"] = 0.0
                    e["course_bias_label"] = _cb["label"]
                except Exception:
                    e["course_bias_bonus"] = 0.0
                    e["course_bias_label"] = ""
                e["excuse_flags"]         = rb.get("excuse_flags", [])

                # VM格言チェック（前走距離を参照）
                from knowledge_base import get_proverb_bonus
                hist_e = df_hist[df_hist["horse_name"] == name].sort_values("date", ascending=False) \
                    if "date" in df_hist.columns else pd.DataFrame()
                def _safe_int(val, default):
                    v = pd.to_numeric(val, errors="coerce")
                    return int(v) if pd.notna(v) else default
                prev_dist_e = _safe_int(hist_e.iloc[0].get("distance", 0), 0) \
                    if not hist_e.empty else 0
                prev_rank_e = _safe_int(hist_e.iloc[0].get("rank", 99), 99) \
                    if not hist_e.empty else 99
                prov = get_proverb_bonus(
                    race_name=st.session_state.get("_race_label", ""),
                    prev_distance=prev_dist_e,
                    prev_rank=prev_rank_e,
                )
                e["proverb_bonus"] = prov["bonus"]
                e["proverb_label"] = prov["label"]

            # 調教タイム + 併せ馬パートナー（TRAIN-5: キャッシュ優先）
            if selected_race_id_for_bias and not demo_mode:
                from datetime import timedelta
                from training_scraper import _load_training_cache

                _race_dt = target_date
                _sat_str = (_race_dt - timedelta(days=1)).strftime("%Y%m%d") \
                    if _race_dt.weekday() == 6 else ""

                # session_state のキャッシュ → ファイルキャッシュ → ネット取得 の順で優先
                training_results = (
                    st.session_state.get("training_data_results")
                    or _load_training_cache(selected_race_id_for_bias)
                    or None
                )
                if not training_results and not eval_df.empty and "horse_name" in eval_df.columns:
                    with st.spinner("調教タイム＋併せ馬を取得中（失敗しても分析は続きます）..."):
                        training_results = fetch_all_training_with_partner(
                            selected_race_id_for_bias,
                            eval_df["horse_name"].tolist(),
                            saturday_date_str=_sat_str,
                        )

                if training_results:
                    # PERF: iterrows + N×N loc → .map() でベクター化
                    _hn = eval_df["horse_name"]
                    eval_df["training_score"]   = _hn.map(lambda n: training_results.get(n, {}).get("score", 50))
                    eval_df["training_label"]   = _hn.map(lambda n: training_results.get(n, {}).get("label", ""))
                    eval_df["training_bonus"]   = _hn.map(lambda n: training_results.get(n, {}).get("bonus", 0.0))
                    eval_df["partner_name"]     = _hn.map(lambda n: training_results.get(n, {}).get("partner_name", ""))
                    eval_df["partner_won_sat"]  = _hn.map(lambda n: training_results.get(n, {}).get("partner_won_sat", False))
                    eval_df["won_awase"]        = _hn.map(lambda n: training_results.get(n, {}).get("won_awase"))
                    eval_df["partner_message"]  = _hn.map(lambda n: training_results.get(n, {}).get("partner_message", ""))

            # ---- エントリループで付与したボーナス値を eval_df にマージ ----
            # （evaluate_race() は事前情報がない状態で実行するため、後付けで追加）
            _bonus_cols = [
                "beat_bonus", "beat_count", "beat_label",
                "resume_bonus_total", "resume_summary", "resume_top_excuse",
                "ippen_candidate", "excuse_flags",
                "proverb_bonus", "proverb_label",
                "training_label", "training_score", "training_detail",  # ROOT-4
                # BUG-X1 修正: 第4-5波で追加した新ボーナスを伝搬させる
                # （これらが eval_df に乗らないと confluence で常に 0.0 扱いだった）
                "pair_bonus", "pair_label", "pair_lift", "pair_note",
                "pci_bonus", "pci_avg", "pci_label", "pci_latest",
                "elo", "elo_label", "elo_rank_in_race",
                "speed_index_avg", "speed_index_best", "speed_index_label",
                # 既存 _bonus 系も念のため再列挙（重複しても害なし）
                "last3f_label", "last3f_bonus", "last3f_avg", "last3f_value",
                "venue_apt_score", "venue_apt_detail", "venue_apt_is_exact",
                "time_rank", "time_rank_bonus", "time_rank_label",
                "weight_trend_label", "weight_trend_bonus", "weight_last_change",
                "handicap_trend_label", "handicap_trend_bonus",
                "turn_dir_bonus", "turn_dir_label", "turn_is_mismatch", "turn_preferred",
                "first_time_bonus", "first_time_label", "is_first_surface", "is_first_distance",
                "stable_bonus", "stable_label", "stable_trend",
                "field_size_bonus", "field_size_label", "field_size_diff",
                "pace_fit_bonus", "pace_fit_label", "pace_shift",
                "short_term_foreign_bonus", "short_term_foreign_note", "is_short_term_foreign",
                "horse_stats_bonus", "horse_stats_details",
                "training_bonus",
                # Y1 修正: apply_xxx 系の戻り値 bonus を伝搬補完
                "condition_apt_bonus", "surface_change_bonus", "hurdle_to_flat_bonus",
                "weight_ratio_bonus", "nicks_bonus", "season_bonus",
                "position_correction_bonus", "realtime_bias_bonus",
                "race_level_bonus", "lap_bonus",
                "won_awase", "partner_won_sat", "partner_name", "partner_message",
                # B-3: コースバイアス恒常推定
                "course_bias_bonus", "course_bias_label",
            
            # 第30波: longshot_evaluator（構造的穴馬判定）が読む表示・判定キー
            # （マージ漏れで position_mismatch 加点が無効 + 根拠表示が常に空だった）
            "nicks_label", "position_correction_msg", "position_mismatch_flag",
            "race_level_label",
        ]
            # list型を含む列はobject型で事前初期化しないとpandas 2.x以降で代入エラーになる
            _list_cols = {"excuse_flags", "resume_summary", "horse_stats_details"}
            for _lc in _list_cols:
                if _lc not in eval_df.columns:
                    eval_df[_lc] = None
                eval_df[_lc] = eval_df[_lc].astype(object)

            for _e in entries_full:
                _name = _e.get("horse_name", "")
                if not _name:
                    continue
                _idxs = eval_df.index[eval_df["horse_name"] == _name].tolist()
                if not _idxs:
                    continue
                _i = _idxs[0]
                for _col in _bonus_cols:
                    if _col in _e:
                        eval_df.at[_i, _col] = _e[_col]

            eval_df = add_confluence_to_eval(eval_df)

            # UI-2 + B-3: confidence_score 確定後に romance_danger を再計算（EV渡す）
            if "confidence_score" in eval_df.columns:
                import importlib, ev_calculator as _evc_mod
                importlib.reload(_evc_mod)  # Streamlitキャッシュ回避
                _rds = _evc_mod._romance_danger_score
                import inspect as _ins
                _rds_nargs = len(_ins.signature(_rds).parameters)
                eval_df["romance_danger"] = eval_df.apply(
                    lambda r: _rds(
                        int(r.get("popularity") or 9),
                        float(r.get("est_win_rate") or 5) / 100,
                        float(r.get("odds") or 10),
                        int(r.get("confidence_score") or 0),
                        float(r.get("ev") if r.get("ev") is not None else float("nan")),
                    ) if _rds_nargs >= 5 else _rds(
                        int(r.get("popularity") or 9),
                        float(r.get("est_win_rate") or 5) / 100,
                        float(r.get("odds") or 10),
                        int(r.get("confidence_score") or 0),
                    ), axis=1
                )

            # 頭数×本命信頼度（A-3: 実際のオッズを渡す）
            if "popularity" in eval_df.columns and not eval_df.empty:
                fav_row = eval_df[eval_df["popularity"] == 1]
                fav_style = fav_row.iloc[0].get("running_style", "不明") if not fav_row.empty else "不明"
            else:
                fav_style = "不明"
            _sorted_odds = sorted(eval_df["odds"].dropna().tolist()) if "odds" in eval_df.columns else []
            _fav_odds    = float(_sorted_odds[0]) if len(_sorted_odds) >= 1 else 0.0
            _second_odds = float(_sorted_odds[1]) if len(_sorted_odds) >= 2 else 0.0
            fav_reliability = eval_favorite_reliability(total_horses, 1, fav_style, fav_odds=_fav_odds, second_odds=_second_odds)

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

        # Y2: 監査用 eval_df 列ダンプ（tools/audit.py --section runtime で検証可能）
        try:
            import json as _aj
            _aj_path = Path(__file__).parent / "data" / "last_eval_df_columns.json"
            _aj_path.parent.mkdir(exist_ok=True)
            _aj_path.write_text(_aj.dumps({
                "columns": list(eval_df.columns),
                "n_rows": int(len(eval_df)),
                "race_name": race_label_save,
                "dumped_at": datetime.now().isoformat(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

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
                st.toast("Discord に分析結果を通知しました", icon="")

    # ---- 結果表示 ---- #
    if "eval_df" in st.session_state:
        eval_df = st.session_state["eval_df"]
        pace_info = st.session_state.get("pace_info", {})

        st.divider()

        # ペース予測バナー（単色トーン統一）
        if pace_info:
            _pace_name = pace_info.get("predicted_pace", "?")
            _pace_level = {"ハイペース": "avoid", "ミドル〜ハイ": "warn",
                           "ミドル": "info", "スローペース": "buy"}.get(_pace_name, "muted")
            ui_banner(_pace_level,
                f"<b>展開予測: {_pace_name}</b> &nbsp;—&nbsp; {pace_info.get('summary', '')}")

        # NEW-3: 予測確度バナー（レース単位の見送り判断指標）
        _prob_col_b = "blended_pct" if "blended_pct" in eval_df.columns else "lgbm_norm_pct"
        if _prob_col_b in eval_df.columns and eval_df[_prob_col_b].notna().any():
            _sorted_p = eval_df[_prob_col_b].sort_values(ascending=False).values
            _top1 = float(_sorted_p[0]) if len(_sorted_p) >= 1 else 0
            _top2 = float(_sorted_p[1]) if len(_sorted_p) >= 2 else 0
            _gap = _top1 - _top2
            _top3_std = float(pd.Series(_sorted_p[:3]).std()) if len(_sorted_p) >= 3 else 0
            if _gap >= 8 or _top1 >= 30:
                _label, _level = "高（決め打ち可）", "buy"
            elif _gap >= 4:
                _label, _level = "中（本命押し）", "info"
            elif _gap >= 2:
                _label, _level = "やや低（注意）", "warn"
            else:
                _label, _level = "低（混戦・見送り推奨）", "avoid"
            ui_banner(_level,
                f"<b>レース予測確度: {_label}</b> &nbsp;—&nbsp; "
                f"トップ <b>{_top1:.1f}%</b> / 2位 {_top2:.1f}% / 差 {_gap:.1f}pt / 上位3頭σ={_top3_std:.1f}"
            )

            # A-3: Conformal Prediction による信頼区間レース判定
            try:
                from conformal import race_confidence as _rc
                _cf = _rc((eval_df[_prob_col_b] / 100).tolist())
                # 第17波: 改善ループ(auto_meta)用に保存（Z2 修正）
                st.session_state["conformal_result"] = _cf
                if _cf["recommend_skip"]:
                    ui_banner("avoid",
                        f"<b>Conformal判定: {_cf['confidence_label']}</b> &nbsp;—&nbsp; "
                        f"平均信頼区間幅 {_cf['mean_interval_width']:.3f} / "
                        f"トップ下限 {_cf['top_lo']:.2f} vs 2位上限 {_cf['second_hi']:.2f} "
                        f"(重なり={_cf['overlap']})  <b>→ 見送り推奨</b>")
                else:
                    ui_banner("muted",
                        f"Conformal: {_cf['confidence_label']} (区間幅 {_cf['mean_interval_width']:.3f})")
            except Exception:
                pass

        # 第34波: 爆穴の主戦場検知 — 荒れ度（実出走データ版）× Conformal 不確実性の AND 結合
        # 分析時は実人気が手元にあるため field_avg_pop/std と1人気オッズを実値で渡せる
        # （ホームタブの事前推定 volatility より高精度）
        try:
            from race_volatility import compute_volatility as _cvol
            _ent_v = st.session_state.get("entries", [])
            _pops_v = pd.to_numeric(pd.Series([e.get("popularity") for e in _ent_v]), errors="coerce").dropna()
            _odds_v = pd.to_numeric(pd.Series([e.get("odds") for e in _ent_v]), errors="coerce").dropna()
            _meta_v = {
                "venue":           st.session_state.get("venue", ""),
                "surface":         st.session_state.get("surface", "芝"),
                "distance":        st.session_state.get("distance", 1800),
                "track_condition": st.session_state.get("track_condition", "良"),
                "race_name":       st.session_state.get("_race_label", ""),
                "field_size":      len(_ent_v) or 16,
                "field_avg_pop":   float(_pops_v.mean()) if len(_pops_v) else 8.5,
                "field_pop_std":   float(_pops_v.std()) if len(_pops_v) >= 2 else 4.5,
                "top_popularity_odds": float(_odds_v[_pops_v.idxmin()]) if len(_pops_v) and len(_odds_v) else None,
            }
            _vol_now = _cvol(_meta_v)
            st.session_state["volatility_result"] = _vol_now
            _cf_label = (st.session_state.get("conformal_result") or {}).get("confidence_label", "")
            _is_uncertain = ("低" in _cf_label)  # 「低」「やや低」両方
            _mode_v = st.session_state.get("betting_mode", "爆穴")
            if _vol_now["score"] >= 52 and _is_uncertain:  # 第34波: 実測分位に同期
                if _mode_v == "爆穴":
                    ui_banner("info",
                        f"<b>爆穴の主戦場</b> — 荒れ度 {_vol_now['score']:.0f}（{_vol_now['label']}）"
                        f"× 予測不確実（{_cf_label}）。複勝EVプラスの穴を厳選する好機です。")
                else:
                    ui_banner("warn",
                        f"<b>堅軸には不向きなレース</b> — 荒れ度 {_vol_now['score']:.0f} × {_cf_label}。"
                        f"爆穴モードへの切替か見送りを検討してください。")
            elif _vol_now["score"] <= 40 and "高" in _cf_label and _mode_v == "爆穴":  # 同期
                ui_banner("warn",
                    f"<b>爆穴には不向きなレース</b> — 荒れ度 {_vol_now['score']:.0f}（堅め）× 予測確信度高。"
                    f"穴の出にくい構造です。堅軸モードか見送りを検討してください。")
        except Exception as _e_vol:
            print(f"[volatility] 分析時計算スキップ: {_e_vol}")

        # ISSUE-4: 未確定オッズ警告
        _entries_now = st.session_state.get("entries", [])
        if _entries_now:
            _unconfirmed = [e for e in _entries_now if not e.get("odds_confirmed", True)]
            if len(_unconfirmed) > 0:
                ui_banner("warn",
                    f"<b>未確定オッズあり ({len(_unconfirmed)}頭)</b> — "
                    f"オッズが取得できなかった馬は暫定値（10.0倍）で計算されています。"
                    f"発走直前に「オッズ・出走馬を再取得」で更新してください。")

        # KPI カード（D10）
        _kpi_c1, _kpi_c2, _kpi_c3, _kpi_c4 = st.columns(4)
        with _kpi_c1:
            st.metric("出走頭数", f"{len(eval_df)}頭")
        with _kpi_c2:
            _ev_plus_n = int((eval_df["ev"] > 0).sum()) if "ev" in eval_df.columns else 0
            st.metric("EV+ 馬数", f"{_ev_plus_n}頭")
        with _kpi_c3:
            if "ev" in eval_df.columns and not eval_df.empty:
                _top_ev_row = eval_df.loc[eval_df["ev"].idxmax()]
                st.metric("EV最高", _top_ev_row.get("horse_name", "-"),
                          delta=f"{float(_top_ev_row.get('ev', 0)):+.3f}")
            else:
                st.metric("EV最高", "-")
        with _kpi_c4:
            # 第21波: この時点では buy_flag 未計算（フィルタは後段）のため
            # 常に 0 頭表示になっていた → プレースホルダにして後段で確定値を埋める
            _kpi_buy_ph = st.empty()
            _kpi_buy_ph.metric("買い推奨数", "計算中…")

        # 本命信頼度・荒れやすさバナー（単色トーン統一）
        fav_rel = st.session_state.get("fav_reliability", {})
        if fav_rel:
            upset = fav_rel.get("upset_score", 50)
            if upset >= 70:
                ui_banner("warn",
                    f"<b>荒れやすいレース</b>（荒れスコア {upset}/100）— {fav_rel.get('message', '')}")
            elif upset <= 35:
                ui_banner("info",
                    f"<b>本命が信頼できるレース</b>（荒れスコア {upset}/100）— {fav_rel.get('message', '')}")

        # 当日バイアス確認
        bias = st.session_state.get("bias_type", "unknown")
        if bias and bias != "unknown":
            bias_info = BIAS_TYPES.get(bias, {})
            ui_banner("muted",
                f"<b>バイアス: {bias_info.get('label', bias)}</b> — {bias_info.get('desc', '')}")

        # NEW-5: 直近トラックバイアス（過去7日の自動集計）
        try:
            from track_variant import get_recent_variant as _grv
            _tv = _grv(
                venue=st.session_state.get("venue", "東京"),
                surface=st.session_state.get("surface", "芝"),
                n_days=7,
            )
            if _tv.get("n_days", 0) > 0:
                _tv_msg = (
                    f"<b>直近{_tv['n_days']}日のトラックバイアス（{st.session_state.get('venue','')}{st.session_state.get('surface','')}）: "
                    f"{_tv['summary']}</b>"
                )
                _details = []
                if _tv.get("time_bias") is not None:
                    _details.append(f"時計傾向 {_tv['time_bias']:+.1f}")
                if _tv.get("gate_bias") is not None:
                    _details.append(f"枠順 {_tv['gate_bias']:+.2f}")
                if _tv.get("pace_bias") is not None:
                    _details.append(f"脚質 {_tv['pace_bias']:.2f}")
                _tv_msg += f" &nbsp;—&nbsp; {' / '.join(_details)}"
                ui_banner("muted", _tv_msg)
        except Exception as _e_tv:
            pass

        # C-5 (第12波): 当日 time_bias 動的補正
        try:
            from datetime import date as _date_today
            from scraper import fetch_today_finished_results as _ftfr
            from track_variant import compute_intraday_bias as _cib
            _venue = st.session_state.get("venue", "東京")
            _surface = st.session_state.get("surface", "芝")
            _today_str = _date_today.today().strftime("%Y%m%d")
            with st.expander("当日馬場補正（intraday time_bias）", expanded=False):
                _apply = st.checkbox(
                    "当日の終了済みレース結果から馬場を動的補正",
                    value=False,
                    key="intraday_bias_apply",
                    help="netkeibaから当日1R以降の結果を取得し、baseline (直近7日) との差分を計算します。",
                )
                if _apply:
                    with st.spinner("当日結果を取得中..."):
                        _today_res = _ftfr(_today_str, _venue)
                    _ib = _cib(_today_res, _venue, _surface, baseline_days=7)
                    if _ib["n_races"] > 0:
                        _delta = _ib.get("delta_time_bias")
                        _cf = _ib.get("correction_factor", 0.0)
                        st.success(
                            f"集計 {_ib['n_races']} R | 当日 time_bias={_ib['today_time_bias']} / "
                            f"baseline={_ib['baseline_time_bias']} / Δ={_delta} → "
                            f"補正係数 {_cf:+.2f}"
                        )
                        st.caption(_ib["summary"])
                        # 第21波: 旧実装（speed_index 一律減算）は
                        #   (1) confluence は speed_index_best/avg を読むため予想に届かない
                        #   (2) 全馬一律の加減算は序列不変で理論的にも無意味
                        # → 脚質×当日バイアスで confidence_score を直接調整する方式に変更
                        _delta_tb = _ib.get("delta_time_bias")
                        _pace_b = _ib.get("today_pace_bias")
                        if "running_style" in eval_df.columns and "confidence_score" in eval_df.columns:
                            _style = eval_df["running_style"].fillna("不明")
                            _front = _style.isin(["逃げ", "先行"])
                            _back  = _style.isin(["差し", "追込", "差し・追込"])
                            _adj = pd.Series(0.0, index=eval_df.index)
                            _notes = []
                            if _delta_tb is not None:
                                if _delta_tb > 5:    # 当日高速馬場 → 前残り傾向
                                    _adj[_front] += 3; _adj[_back] -= 2
                                    _notes.append("高速馬場→先行+3/差し-2")
                                elif _delta_tb < -5:  # 時計かかる → 差し台頭
                                    _adj[_back] += 3; _adj[_front] -= 2
                                    _notes.append("時計かかる→差し+3/先行-2")
                            if _pace_b is not None:
                                if _pace_b >= 0.6:
                                    _adj[_back] += 2
                                    _notes.append("当日差し決着傾向→差し+2")
                                elif _pace_b <= 0.35:
                                    _adj[_front] += 2
                                    _notes.append("当日前残り傾向→先行+2")
                            if _adj.abs().sum() > 0:
                                eval_df["confidence_score"] = (
                                    eval_df["confidence_score"] + _adj).clip(0, 100)
                                eval_df["intraday_adj"] = _adj
                                st.caption("→ 脚質補正適用: " + " / ".join(_notes))
                            else:
                                st.caption("→ 当日バイアスは中立圏のため補正なし")
                        st.session_state["intraday_bias_result"] = _ib
                    else:
                        st.info(_ib["summary"])
        except Exception as _e_ib:
            st.caption(f"（当日補正は利用不可: {type(_e_ib).__name__}）")

        # 総合信頼スコアテーブル
        st.subheader("実力スコアと市場評価")
        st.caption("**実力スコア** = 馬の強さ評価（EVは含まない）　|　**市場評価** = そのオッズが割安か割高か")

        # 凡例
        with st.expander("各列の見方", expanded=False):
            st.markdown("""
| 列名 | 意味 |
|------|------|
| **最終判定** | ◎軸 / ▲信頼軸（人気サイドの実力上位）/ ○妙味 / △穴ロマン / ・押さえ / ✕消し。**この1列だけ見ればOK** |
| **根拠** | 判定の理由1文（判定と必ず整合） |
| **実力スコア** | 0〜100点。馬の強さ評価のみ（EVは含まない） |
| **馬券内率%** | モデル推定の3着以内確率 |
| **Benter勝率%** | AIモデル予測と市場オッズをブレンドした勝率（精度最重視ならこれ） |
| **市場勝率%** | オッズから逆算した「市場が想定する勝率」（=1/オッズの正規化。人気の裏返し） |
| **EV** | 単勝期待値。+0.10 = 単勝を買い続けたら10%プラスの想定 |
| **複勝EV** | 複勝の期待値。穴の複勝勝負はここを見る |
| **ケリー%** | 推奨賭け比率。**EV+5%以上の馬のみ算出**（他は「－」表示）|
| **脚質/枠/乗替/ローテ** | 各ファクターの個別評価 |
""")

        # 並び順切り替え
        _sort_mode = st.radio("並び順", ["スコア順（推奨）", "枠番順"], horizontal=True, key="score_sort_mode")

        # NEW-5: 乗り替わり「不明」→空欄（意味のない表示を排除）
        if "jockey_change_signal" in eval_df.columns:
            eval_df["jockey_change_signal"] = eval_df["jockey_change_signal"].replace("不明", "")

        # "一変候補" 列を見やすく変換
        if "ippen_candidate" in eval_df.columns:
            eval_df["ippen_label"] = eval_df["ippen_candidate"].apply(
                lambda v: "一変" if v else "")
        if "excuse_flags" in eval_df.columns:
            eval_df["excuse_str"] = eval_df["excuse_flags"].apply(
                lambda v: " / ".join(v[:2]) if isinstance(v, list) and v else "")

        # Phase C: ケリー基準（推奨賭け比率） + SUPER-3: 分数ケリー倍率
        from bet_builder import kelly_fraction, apply_buy_filter
        _k_ratio = float(st.session_state.get("kelly_fraction_ratio", 0.50))
        # 勝率は Benter ブレンド後の blended_pct を優先（無ければ est_win_rate にフォールバック）
        _wr_col = "blended_pct" if "blended_pct" in eval_df.columns else "est_win_rate"
        if _wr_col in eval_df.columns and "odds" in eval_df.columns:
            eval_df["kelly_pct"] = eval_df.apply(
                lambda r: f"{kelly_fraction(r[_wr_col]/100, r['odds'], fraction=_k_ratio)*100:.1f}%"
                          if r.get("ev", -1) > 0 else "－",
                axis=1)

        # Phase F: 見送り条件フィルタ
        _ev_thr  = st.session_state.get("filter_ev_threshold",  0.10)
        _odds_mn = st.session_state.get("filter_odds_min",      2.5)
        _sc_mn   = st.session_state.get("filter_score_min",     55)
        _use_dyn = bool(st.session_state.get("use_dynamic_ev",  True))

        # 第13波→第14波改訂: 爆穴モードは複勝EVベースで判定
        # （単勝EVだと人気薄はほぼ常にマイナスになり候補全滅するため）
        _mode = st.session_state.get("betting_mode", "爆穴")
        # 第21波: 分析時点のモードを記録（後でモード切替された場合の stale 検知用）
        st.session_state["analyzed_with_mode"] = _mode
        if _mode == "爆穴":
            _odds_mn = max(_odds_mn, 8.0)   # 中穴〜大穴帯のみ
            _sc_mn   = min(_sc_mn, 45)
        st.session_state["ev_threshold_active"] = _ev_thr

        eval_df  = apply_buy_filter(eval_df, _ev_thr, _odds_mn, _sc_mn, use_dynamic=_use_dyn)

        # 第14波: 爆穴モードの buy_flag は複勝EVで上書き判定
        if _mode == "爆穴" and "place_ev" in eval_df.columns:
            _pop_f = pd.to_numeric(eval_df.get("popularity"), errors="coerce").fillna(99)
            _ana_mask = (
                (_pop_f >= 6)
                & (eval_df["place_ev"] >= 0.05)          # 複勝EVプラス圏
                & (eval_df["place_prob"] >= 0.10)        # 馬券内10%以上（紙くず回避）
            )
            if "buy_flag" in eval_df.columns:
                eval_df["buy_flag"] = _ana_mask
                eval_df.loc[_ana_mask, "buy_reason"] = eval_df.loc[_ana_mask].apply(
                    lambda r: f"複勝EV {r['place_ev']:+.2f} / 馬券内 {r.get('place_pct', 0):.0f}%", axis=1)

        # 第14波: 穴馬候補セクション（爆穴モード時のみ）
        if _mode == "爆穴" and "longshot_score" in eval_df.columns:
            _ana = eval_df[eval_df["longshot_score"].notna()].sort_values(
                "longshot_score", ascending=False)
            st.markdown("### 穴馬候補（穴スコア順）")
            st.caption("複勝EV 50% + 馬券内確率 30% + レアパターン 20% の合成。6人気以下のみ対象。")
            if _ana.empty:
                st.info("6人気以下の出走馬がいないため穴馬候補なし。")
            else:
                # 第46波: 上位3頭のみに厳選（「どれかしら当たる」散漫表示を排除）
                _show_n = min(len(_ana), 3)
                _hole_rows = []
                for _i, (_, _h) in enumerate(_ana.head(_show_n).iterrows()):
                    _ls = _h["longshot_score"]
                    _medal = ["①", "②", "③"][_i] if _i < 3 else "・"
                    _pev = _h.get("place_ev")
                    _pev_str = f"{_pev:+.2f}" if pd.notna(_pev) else "—"
                    _rare = _h.get("rare_labels", "") or ""
                    _pop_v = pd.to_numeric(pd.Series([_h.get("popularity")]), errors="coerce").iloc[0]
                    _pop_disp = f"{int(_pop_v)}" if pd.notna(_pop_v) else "?"
                    _odds_v = pd.to_numeric(pd.Series([_h.get("odds")]), errors="coerce").iloc[0]
                    _odds_disp = f"{_odds_v:.1f}" if pd.notna(_odds_v) else "?"
                    _hole_rows.append({
                        "順": _medal,
                        "馬名": _h.get("horse_name", ""),
                        "人気": _pop_disp,
                        "オッズ": _odds_disp,
                        "穴スコア": f"{_ls:.0f}",
                        "馬券内%": f"{_h.get('place_pct', 0):.0f}",
                        "複勝EV": _pev_str,
                        "レア": _rare or "—",
                    })
                _hole_df = pd.DataFrame(_hole_rows)
                st.dataframe(_hole_df, use_container_width=True, hide_index=True,
                             height=min(360, 40 + len(_hole_df) * 35))
                # 第45波: 警告閾値を緩和 — 馬券内%20%以上の馬がいれば「勝負可能」とみなす
                _has_strong_pct = (
                    (_ana["place_pct"] >= 20).any()
                    if "place_pct" in _ana.columns else False
                )
                _hits = int((_ana["place_ev"] >= 0.05).sum()) if "place_ev" in _ana.columns else 0
                if _hits == 0 and not _has_strong_pct:
                    st.warning("複勝EV プラス かつ 馬券内率20%以上の穴馬がいません — このレースは見送りも選択肢です。")

        # BUG-C: 未確定オッズの馬は EV / buy_flag を強制無効化
        if "horse_name" in eval_df.columns:
            _unconfirmed_set = {
                e.get("horse_name", "") for e in st.session_state.get("entries", [])
                if not e.get("odds_confirmed", True)
            }
            if _unconfirmed_set:
                _mask_uc = eval_df["horse_name"].isin(_unconfirmed_set)
                if "ev" in eval_df.columns:
                    eval_df.loc[_mask_uc, "ev"] = np.nan
                    eval_df.loc[_mask_uc, "ev_label"] = "? 未確定オッズ"
                if "buy_flag" in eval_df.columns:
                    eval_df.loc[_mask_uc, "buy_flag"] = False
                    eval_df.loc[_mask_uc, "buy_reason"] = "未確定オッズ（再取得待ち）"

        # SUPER-6: 馬 Elo レーティング（事前計算済 horse_elo.parquet がある場合）
        try:
            from horse_elo import get_elo_for_field, _ELO_PATH as _elo_path
            if _elo_path.exists() and "horse_name" in eval_df.columns:
                _elo_map = get_elo_for_field(eval_df["horse_name"].tolist())
                eval_df["elo"] = eval_df["horse_name"].map(lambda h: _elo_map.get(h, {}).get("elo", 1500))
                # 第25波 (M): confluence が読む _race_elo_avg が供給ゼロで、Elo ボーナスが
                # 常に固定基準1500との差で計算されていた（強メンバー戦で全馬過大評価）
                # → レース内平均を供給して相対評価に
                eval_df["_race_elo_avg"] = float(eval_df["elo"].mean())
                eval_df["elo_label"] = eval_df["horse_name"].map(lambda h: _elo_map.get(h, {}).get("label", ""))
                eval_df["elo_rank_in_race"] = eval_df["horse_name"].map(lambda h: _elo_map.get(h, {}).get("elo_rank_in_race", 0))
        except Exception as _ee:
            pass

        # 第33波 (P2): 道悪鬼パターンの配線 — wet_place_rate は horse_stats.parquet に
        # 存在するのに eval_df へのマージ経路がなく、道悪鬼判定が100%発火不能だった
        try:
            _hs_wet = pd.read_parquet(
                Path(__file__).parent / "data" / "horse_stats.parquet",
                columns=["horse_name", "wet_place_rate"])
            _wet_map = dict(zip(_hs_wet["horse_name"].str.strip(), _hs_wet["wet_place_rate"]))
            eval_df["wet_place_rate"] = eval_df["horse_name"].str.strip().map(_wet_map)
        except Exception:
            eval_df["wet_place_rate"] = float("nan")

        # 第40波: 会場・回り・距離適性タグ
        # 第41波: クラスの壁判定 + 隠れ実力馬 + 騎手×距離 + 騎手×重賞人気薄
        try:
            from course_aptitude import (build_aptitude_tag as _bat,
                build_class_wall_tag as _bcwt,
                jockey_distance_aptitude as _jda,
                jockey_venue_distance_aptitude as _jvd,
                jockey_grade_longshot_aptitude as _jgl,
                stable_grade_aptitude as _sga)
            _venue_a = st.session_state.get("venue", "東京")
            _surf_a = st.session_state.get("surface", "芝")
            _dist_a = st.session_state.get("distance", 1800)
            _race_a = st.session_state.get("_race_label", "")
            # クラス推定（race_name から）
            from course_aptitude import _class_lv as _cls_lv
            _cur_class = _cls_lv(_race_a)

            def _combine(row):
                nm = str(row.get("horse_name", "")).strip()
                jk = str(row.get("jockey", "")).strip()
                pop = int(pd.to_numeric(pd.Series([row.get("popularity")]), errors="coerce").fillna(9).iloc[0])
                t1 = _bat(nm, _venue_a, _surf_a, _dist_a)
                t2 = _bcwt(nm, _cur_class)
                t3 = _jda(jk, _dist_a) if jk else {"tag":"", "bonus":0.0, "detail":""}
                t3b = _jvd(jk, _venue_a, _dist_a) if jk else {"tag":"", "bonus":0.0, "detail":""}
                t4 = _jgl(jk, _race_a, pop) if jk else {"tag":"", "bonus":0.0, "detail":""}
                tr = str(row.get("trainer", "")).strip()
                t5 = _sga(tr, _race_a) if tr else {"tag":"", "bonus":0.0, "detail":""}
                tags = [x for x in t1.get("tags", []) + [t2.get("tag",""), t3.get("tag",""), t3b.get("tag",""), t4.get("tag",""), t5.get("tag","")] if x]
                bonus_total = (t1.get("bonus",0) + t2.get("bonus",0) +
                               t3.get("bonus",0) + t3b.get("bonus",0) + t4.get("bonus",0) + t5.get("bonus",0))
                details = [x for x in [", ".join(t1.get("tags",[]) or []) and t1.get("detail",""),
                                        t2.get("detail",""), t3.get("detail",""), t3b.get("detail",""), t4.get("detail",""), t5.get("detail","")] if x]
                return pd.Series({"aptitude_tags": " / ".join(tags),
                                   "aptitude_bonus": float(np.clip(bonus_total, -0.05, 0.05)),
                                   "aptitude_detail": " | ".join(details)})

            _apt = eval_df.apply(_combine, axis=1)
            eval_df["aptitude_tags"] = _apt["aptitude_tags"]
            eval_df["aptitude_bonus"] = _apt["aptitude_bonus"]
            eval_df["aptitude_detail"] = _apt["aptitude_detail"]
            if "confidence_score" in eval_df.columns:
                eval_df["confidence_score"] = (
                    eval_df["confidence_score"] + eval_df["aptitude_bonus"] * 100
                ).clip(0, 100)
        except Exception as _e_apt:
            print(f"[course_aptitude] スキップ: {_e_apt}")

        # NEW-1: 勝率累乗フィルター + IMPROVE-5: Portfolio Kelly
        from bet_builder import apply_power_filter, apply_data_quality_filter, apply_portfolio_kelly_to_df
        _pow      = float(st.session_state.get("power_filter_p",   4.0))
        _pow_gap  = float(st.session_state.get("power_filter_gap", 0.4))
        eval_df   = apply_power_filter(eval_df, power=_pow, gap_threshold=_pow_gap)

        # NEW-4: 過去N戦未満を見送り
        _min_past = int(st.session_state.get("min_past_races", 3))
        eval_df   = apply_data_quality_filter(eval_df, df_hist, min_past_races=_min_past)
        if "data_quality_ok" in eval_df.columns:
            # data不足の馬は buy_flag を強制 False
            eval_df.loc[~eval_df["data_quality_ok"], "buy_flag"] = False
            eval_df.loc[~eval_df["data_quality_ok"], "buy_reason"] = "データ不足（過去戦数不足）"

        if "buy_flag" in eval_df.columns:
            # 第45波: 3段階判定（買・様子見・消）— 閾値を実用的なレベルに緩和
            def _classify(row):
                if row.get("buy_flag"):
                    return "🟢買"
                try:
                    _ev = float(row.get("ev"))
                except (TypeError, ValueError):
                    _ev = float("nan")
                try:
                    _ev_place = float(row.get("ev_place"))
                except (TypeError, ValueError):
                    _ev_place = float("nan")
                try:
                    _cs = float(row.get("confidence_score", 0) or 0)
                except (TypeError, ValueError):
                    _cs = 0.0
                _rd_label = str(row.get("romance_danger", "") or "")
                _is_extreme_danger = "極高" in _rd_label  # 「高」だけでは消さない
                # 「消」基準を緩和: EV<-0.5（大幅マイナス）or 大穴危険度極高 かつ 実力低い
                if _is_extreme_danger and _cs < 45:
                    return "🔴消"
                if not pd.isna(_ev) and _ev < -0.5 and _cs < 50:
                    return "🔴消"
                # 「様子見」: 実力50以上 or 単勝/複勝のどちらかでEVプラス気味
                if _cs >= 50:
                    return "🟡様子見"
                if not pd.isna(_ev_place) and _ev_place >= 0:
                    return "🟡様子見"
                if not pd.isna(_ev) and _ev >= -0.2:
                    return "🟡様子見"
                return "🔴消"
            eval_df["buy_label"] = eval_df.apply(_classify, axis=1)
        # 第21波: KPI プレースホルダにフィルタ確定後の買い推奨数を反映
        try:
            _kpi_buy_ph.metric("買い推奨数", f"{int(eval_df['buy_flag'].sum())}頭")
        except Exception:
            pass

        # IMPROVE-6: 直前オッズ急変シグナル（odds_monitor 連携）
        _rid_now = st.session_state.get("preselected_race_id", "")
        if _rid_now and "horse_name" in eval_df.columns:
            try:
                from odds_monitor import detect_odds_signals as _dos
                _sigs = _dos(_rid_now)
                _signal_map = {}  # horse_name → label
                for _s in _sigs:
                    name = _s.get("horse", _s.get("name", ""))
                    chg = _s.get("change_pct", 0)
                    if chg <= -25:
                        _signal_map[name] = f"急下落{chg:.0f}%"
                    elif chg >= 25:
                        _signal_map[name] = f"急上昇+{chg:.0f}%"
                if _signal_map:
                    eval_df["odds_signal"] = eval_df["horse_name"].map(_signal_map).fillna("")
            except Exception:
                pass

        # IMPROVE-5: Portfolio Kelly で同レース内の EV+ 馬を正規化
        _bankroll = int(st.session_state.get("budget", 5000)) * 20  # 1レース予算 × 20 を総資金と仮定
        _k_ratio_pf = float(st.session_state.get("kelly_fraction_ratio", 0.50))
        eval_df = apply_portfolio_kelly_to_df(
            eval_df, bankroll=_bankroll, max_exposure=0.20,
            fraction=_k_ratio_pf, bet_unit=100,
        )

        # 第46波: 統一最終判定（全項目を一本の線でつなぐ）
        from unified_verdict import apply_unified_verdict, sort_by_verdict
        eval_df = apply_unified_verdict(eval_df, mode=st.session_state.get("betting_mode", "爆穴"))

        # 第46波: 矛盾していた判定列群（買い判定/判定/市場評価/推奨理由/非推奨理由）を
        # final_verdict + final_reason の2列に統合。詳細指標は後続列で確認可能。
        score_cols = ["final_verdict", "final_reason",
                      "gate", "horse_no", "horse_name", "jockey", "popularity", "odds",
                      "confidence_score",
                      "est_place_rate",
                      "blended_pct", "market_emp_pct",
                      "kelly_pct", "portfolio_amount", "power_buy", "odds_signal",
                      "ippen_label", "excuse_str",
                      "ev",
                      "place_odds", "ev_place",
                      "elo", "elo_label",
                      "speed_index_best", "speed_index_label",
                      "pci_avg", "pci_label",
                      "pair_label",
                      "running_style", "draw_label",
                      "jockey_change_signal", "rotation_signal",
                      "training_label"]
        score_cols = [c for c in score_cols if c in eval_df.columns]
        rename_score = {
            "final_verdict": "最終判定",
            "final_reason": "根拠",
            "gate": "枠番", "horse_no": "馬番", "horse_name": "馬名", "jockey": "騎手",
            "popularity": "人気", "odds": "単勝",
            "confidence_score": "実力スコア",
            "est_place_rate": "馬券内率%",
            "blended_pct": "Benter勝率%",
            "market_emp_pct": "市場勝率%",
            "kelly_pct": "ケリー%",
            "portfolio_amount": "Pf金額",
            "power_buy": "累乗買い",
            "odds_signal": "ｵｯｽﾞ急変",
            "elo": "Elo",
            "elo_label": "Eloランク",
            "pci_avg": "PCI",
            "pci_label": "ペース耐性",
            "pair_label": "厩×騎",
            "speed_index_best": "ﾀｲﾑ指数",
            "speed_index_label": "ﾀｲﾑ評価",
            "ippen_label": "一変",
            "excuse_str": "前走言い訳",
            "ev_label": "市場評価", "ev": "EV値",
            "place_odds": "複勝", "ev_place": "複勝EV", "ev_place_label": "複勝評価",
            "running_style": "脚質", "draw_label": "枠バイアス",
            "jockey_change_signal": "乗替",
            "rotation_signal": "ローテ",
            "romance_danger": "大穴危険度",
            "training_label": "追い切り",
        }

        # UI-1: 枠番ソート時は horse_no をサブキーに（同枠内も馬番順）
        if "枠番順" in _sort_mode and "gate" in eval_df.columns:
            _sort_keys = ["gate", "horse_no"] if "horse_no" in eval_df.columns else ["gate"]
            _display_df = eval_df.sort_values(_sort_keys)
        else:
            # 第46波: デフォルトは最終判定順（◎軸→○妙味→▲信頼軸→△穴ロマン→✕消し）
            _display_df = sort_by_verdict(eval_df)
        # ROOT-5: EV表示を小数2桁に丸め（-0.9999819... → -1.00）
        if "ev" in _display_df.columns:
            _display_df = _display_df.copy()
            _display_df["ev"] = _display_df["ev"].round(2)

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

        show_score_df = _display_df[score_cols].rename(columns=rename_score)

        # 「市場評価」「買い判定」「大穴危険度」もカラー化（統一トーン）
        def _color_market(val):
            s = str(val)
            if "◎" in s: return f"background-color: #E6F2EC; color: {COLOR['buy']}; font-weight:600"
            if "○" in s: return f"background-color: #F1F7EC; color: {COLOR['buy']}"
            if "▲" in s: return f"background-color: #FBF1DC; color: {COLOR['warn']}"
            if "✕" in s: return f"background-color: #F6E1DD; color: {COLOR['avoid']}"
            return ""
        def _color_buy(val):
            if "買" in str(val): return f"background-color: #E6F2EC; color: {COLOR['buy']}; font-weight:600"
            if "✗" in str(val): return f"color: {COLOR['avoid']}"
            return ""
        def _color_danger(val):
            s = str(val)
            if "極高" in s: return f"background-color: #F6E1DD; color: {COLOR['avoid']}; font-weight:600"
            if "高" in s:   return f"background-color: #FBF1DC; color: {COLOR['warn']}"
            return ""

        _styled = show_score_df.style
        if "実力スコア" in show_score_df.columns:
            _styled = _styled.map(color_score, subset=["実力スコア"])
        if "最終判定" in show_score_df.columns:
            _styled = _styled.map(_color_market, subset=["最終判定"])
        if "大穴危険度" in show_score_df.columns:
            _styled = _styled.map(_color_danger, subset=["大穴危険度"])
        # 罫線・ヘッダ統一
        _styled = _styled.set_table_styles([
            {"selector": "thead th",
             "props": f"background-color:{COLOR['sidebar_bg']}; color:{COLOR['text']}; "
                       f"border-bottom:1px solid {COLOR['border']}; font-weight:600;"},
            {"selector": "tbody td",
             "props": f"border-bottom:1px solid {COLOR['border']};"},
        ])
        st.dataframe(_styled, use_container_width=True, hide_index=True)
        # UI-7: ケリー%の説明
        st.caption("**ケリー%** = 総資金のうち推奨する賭け比率（例: 0.8% = 1万円なら80円）。EV+5%未満・スコア不足・低オッズの場合は「－」。")

        # 第46波: CSVエクスポート（最終判定+根拠で一本化）
        try:
            _export_cols_pref = [
                "final_verdict", "final_reason",
                "horse_name", "popularity", "odds", "ev",
                "confidence_score",
                "lgbm_win_rate", "blended_pct",
                "est_place_rate", "ev_place", "place_odds",
                "training_label",
            ]
            _export_cols = [c for c in _export_cols_pref if c in eval_df.columns]
            _exp_df = sort_by_verdict(eval_df)[_export_cols].copy()
            _exp_df = _exp_df.rename(columns={
                "final_verdict": "最終判定", "final_reason": "根拠",
                "horse_name": "馬名", "popularity": "人気", "odds": "オッズ",
                "ev": "EV", "confidence_score": "実力スコア",
                "lgbm_win_rate": "LGBM勝率%", "blended_pct": "Benter勝率%",
                "est_place_rate": "推定複勝率%", "ev_place": "複勝EV",
                "place_odds": "複勝オッズ", "training_label": "調教",
            })
            _race_lbl = st.session_state.get("_race_label", "race").replace(" ", "_").replace("/", "_")
            st.download_button(
                "📥 診断結果をCSVでダウンロード",
                _exp_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"診断_{_race_lbl}.csv",
                mime="text/csv",
                key="export_analysis_csv",
            )
        except Exception as _e_exp:
            st.caption(f"エクスポート生成失敗: {_e_exp}")

        # EFF-3: Claude 自然言語サマリ（上位3頭の短評）
        with st.expander("Claude AI による上位3頭の短評", expanded=False):
            try:
                from ai_summary import render_ai_summary
                _race_label = st.session_state.get("race_name_for_save", "")
                _rid_for_cache = st.session_state.get("preselected_race_id", "no_rid")
                render_ai_summary(
                    eval_df,
                    pace_info=st.session_state.get("pace_info", {}),
                    surface=st.session_state.get("surface", "芝"),
                    distance=int(st.session_state.get("distance", 2000)),
                    venue=st.session_state.get("venue", ""),
                    race_name=_race_label,
                    cache_key=str(_rid_for_cache),
                )
            except Exception as _e_ai:
                st.caption(f"AI サマリ生成エラー: {_e_ai}")

        # B-4: 重複ボタン削除（調教レポートHTML出力は「取得済み」直下のボタンに統一）

        # 第45波: 空evdf / カラム欠落時の早期 return
        if eval_df.empty or "confidence_score" not in eval_df.columns:
            st.warning("評価データを生成できませんでした。出馬表データを取得し直してください（horse_latest_features に該当馬の過去成績がない場合に発生します）。")
            st.stop()

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
        st.plotly_chart(apply_chart_theme(fig_score), use_container_width=True)

        # B-5: スパイダーチャート（全頭対応）
        st.subheader("ファクター内訳")
        if "factor_breakdown" in eval_df.columns:
            _spider_cats = ["EV期待値", "ペース展開", "枠順", "騎手", "ローテ",
                            "位置取り補正", "斤量比", "ニックス", "季節・馬場適性",
                            "前走レベル", "ラップ適性", "当日バイアス"]

            # 馬選択（selectbox）
            _horse_opts = eval_df["horse_name"].tolist()
            _sel_horse = st.selectbox("馬を選択してスパイダーチャートを表示", _horse_opts, key="spider_horse_select")
            _sel_row = eval_df[eval_df["horse_name"] == _sel_horse]
            if not _sel_row.empty:
                _bd = _sel_row.iloc[0].get("factor_breakdown", {})
                if isinstance(_bd, dict):
                    _vals = [_bd.get(c, 50) for c in _spider_cats]
                    _vals.append(_vals[0])
                    fig_spider1 = go.Figure(go.Scatterpolar(
                        r=_vals, theta=_spider_cats + [_spider_cats[0]],
                        fill="toself", name=_sel_horse, line_color="#2196F3", opacity=0.7,
                    ))
                    fig_spider1.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        title=f"ファクター内訳: {_sel_horse}",
                        showlegend=False, height=400,
                    )
                    st.plotly_chart(apply_chart_theme(fig_spider1), use_container_width=True)

            # 上位3頭比較チャートをexpanderに
            with st.expander("上位3頭 比較チャート"):
                _colors3 = ["#2196F3", "#FF5722", "#4CAF50"]
                fig_spider3 = go.Figure()
                for _i3, (_, _row3) in enumerate(eval_df.head(3).iterrows()):
                    _bd3 = _row3.get("factor_breakdown", {})
                    if isinstance(_bd3, dict):
                        _v3 = [_bd3.get(c, 50) for c in _spider_cats]
                        _v3.append(_v3[0])
                        fig_spider3.add_trace(go.Scatterpolar(
                            r=_v3, theta=_spider_cats + [_spider_cats[0]],
                            fill="toself", name=_row3.get("horse_name", f"馬{_i3+1}"),
                            line_color=_colors3[_i3 % 3], opacity=0.6,
                        ))
                fig_spider3.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    title="ファクター別内訳（上位3頭比較）", showlegend=True,
                )
                st.plotly_chart(apply_chart_theme(fig_spider3), use_container_width=True)

        # 詳細ファクターテーブル（展開可能）
        with st.expander("全ファクター詳細テーブル"):
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
            st.subheader("末脚型穴馬レーダー（上がり3F比較）")
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
                st.plotly_chart(apply_chart_theme(fig_3f), use_container_width=True)
                # 末脚型穴馬アラート（厳選: 平均比1.0秒以上速い + bonus上位3頭まで）
                fast_longshots = last3f_df[
                    (last3f_df["last3f_bonus"] >= 0.015)  # 強い末脚のみ（◎ or 「平均比1秒以上速い」相当）
                    & (last3f_df["popularity"] >= 7)
                ].sort_values("last3f_bonus", ascending=False).head(3)
                for _, r in fast_longshots.iterrows():
                    st.success(f"**末脚型穴馬候補: {r['horse_name']}** ({r['popularity']}番人気) — {r['last3f_label']}")

        # ---- 3. 出走間隔ビュー（rotation_signalをそのまま表示）---- #
        # 総合スコアに含まれているため、ここでは簡易確認用として表示
        if "rotation_signal" in eval_df.columns:
            with st.expander("出走間隔詳細", expanded=False):
                _int_cols = ["horse_name", "popularity", "rotation_signal", "rotation_days", "rotation_message"]
                _int_cols = [c for c in _int_cols if c in eval_df.columns]
                _int_df = eval_df[_int_cols].rename(columns={
                    "horse_name": "馬名", "popularity": "人気",
                    "rotation_signal": "ローテ判定", "rotation_days": "間隔(日)",
                    "rotation_message": "詳細",
                }).sort_values("人気")

                def _color_rot(val):
                    v = str(val)
                    if "休養明け" in v or "長期" in v: return "background-color: #ffc7ce"
                    if "連闘" in v or "中1週" in v:    return "background-color: #ffeb9c"
                    if "叩き2走目" in v or "標準" in v: return "background-color: #c6efce"
                    return ""

                st.dataframe(
                    _int_df.style.map(_color_rot, subset=["ローテ判定"]),
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
            st.subheader("注目馬アラート（カテゴリ別）")

            # 第45波: 馬ごとにアラートを集約して expander で表示。1行=1馬で見やすく整理
            _alert_per_horse: dict[str, list[tuple[str, str]]] = {}

            def _add(horse_name, category, message):
                _alert_per_horse.setdefault(horse_name, []).append((category, message))

            for _, row in proverb_horses.iterrows():
                _add(row['horse_name'], "🟢格言一致", row.get('proverb_label', ''))
            for _, row in partner_horses.iterrows():
                _add(row['horse_name'], "🟢併せ調教勝", row.get('partner_message', ''))
            for _, row in foreign_horses.iterrows():
                _add(row['horse_name'], "🔵短期外国人騎手", row.get('short_term_foreign_note', ''))
            for _, row in hurdle_horses.iterrows():
                _add(row['horse_name'], "🟢障害→平地", row.get('hurdle_to_flat_message', ''))
            if not beaten_horses.empty and len(beaten_horses.columns) > 0:
                _sort_col = "beat_bonus" if "beat_bonus" in beaten_horses.columns else beaten_horses.columns[0]
                for _, row in beaten_horses.sort_values(_sort_col, ascending=False).head(3).iterrows():
                    _add(row['horse_name'], "🔵有力馬撃破", row.get('beat_label', ''))
            for _, row in mismatch_horses.iterrows():
                _add(row['horse_name'], "🟡回り方向ミスマッチ", row.get('turn_dir_label', ''))
            for _, row in pace_fit_horses.iterrows():
                _add(row['horse_name'], "🟢ペース×上がり好転", row.get('pace_fit_label', ''))
            for _, row in stable_hot[stable_hot["stable_trend"]=="絶好調"].iterrows():
                _add(row['horse_name'], "🟢厩舎絶好調", row.get('stable_label', ''))
            for _, row in first_time_horses.head(4).iterrows():
                _add(row['horse_name'], "🔵初挑戦(穴期待)", row.get('first_time_label', ''))
            for _, row in comebacks.iterrows():
                _add(row['horse_name'], "🟢巻き返し候補", row.get('exhaustion_message', ''))
            for _, row in tatakidai_horses.iterrows():
                _add(row['horse_name'], "🟡叩き台", row.get('tatakidai_message', ''))

            # popularity / odds を補完
            _pop_map = dict(zip(eval_df["horse_name"], eval_df.get("popularity", []))) if "popularity" in eval_df.columns else {}
            _odds_map = dict(zip(eval_df["horse_name"], eval_df.get("odds", []))) if "odds" in eval_df.columns else {}

            # アラート数の多い馬 → 少ない馬の順
            _alert_sorted = sorted(_alert_per_horse.items(), key=lambda x: -len(x[1]))
            for _hn, _alerts in _alert_sorted:
                _pop = _pop_map.get(_hn, "?")
                _odds = _odds_map.get(_hn, "?")
                _cats = " ".join(set(c for c, _ in _alerts))
                with st.expander(f"**{_hn}** ({_pop}人気/{_odds}倍) — {_cats}", expanded=False):
                    for c, m in _alerts:
                        st.markdown(f"- {c} : {m}")

        # ---- 11. 類似レース検索 ---- #
        with st.expander("過去の類似レース傾向を検索", expanded=False):
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
                    st.info(f"{venue}×{surface}×{distance}m の傾向: {sim['pattern_summary']}")
                    if sim["fast_3f_wins"] > 0.3:
                        st.success(f"上がり最速馬の勝率が{sim['fast_3f_wins']*100:.0f}%と高い → 末脚型を重視")
                    if sim["upset_rate"] > 0.2:
                        st.warning(f"このコース×距離は大穴が出やすい（{sim['upset_rate']*100:.0f}%）")
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
            if st.button("分析を保存"):
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
    st.subheader("馬別 詳細プロファイル")
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
            st.subheader("↩右回り/左回り適性")
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
                    st.warning(f"↩{ta['label']}")
                elif ta["bonus"] > 0:
                    st.success(f"↩{ta['label']}")
                else:
                    st.caption(f"↩{ta['label']}")

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
                    st.success(f"{l3f['label']}")
                elif l3f["bonus"] < 0:
                    st.warning(f"{l3f['label']}")
                else:
                    st.caption(f"上がり3F: {l3f['label']}")

        # 騎手乗り替わり詳細（理由推定付き）
        if selected_entry.get("jockey_change_signal"):
            signal = selected_entry["jockey_change_signal"]
            msg    = selected_entry.get("jockey_change_msg", "")
            reason = selected_entry.get("jockey_change_reason", "")
            rnote  = selected_entry.get("jockey_change_reason_note", "")
            if signal == "鞍上強化":
                st.success(f"乗替（強化）: {msg}")
            elif signal == "鞍上弱化":
                st.error(f"乗替（弱化）: {msg}")
            elif signal == "手戻り":
                st.info(f"乗替（手戻り）: {msg}")
            else:
                st.caption(f"乗替: {msg}")
            if reason:
                st.caption(f"推定理由: **{reason}** — {rnote}")

        # ローテーション・体重
        col_r, col_w = st.columns(2)
        with col_r:
            rot_msg = selected_entry.get("rotation_message", "")
            if rot_msg:
                st.caption(f"{rot_msg}")
        with col_w:
            w_msg = selected_entry.get("weight_message", "")
            if w_msg:
                st.caption(f"{w_msg}")

        # ---- 新ファクター 4行表示 ---- #
        nf_c1, nf_c2 = st.columns(2)
        with nf_c1:
            # 初距離・初馬場
            ft_label = selected_entry.get("first_time_label", "")
            if ft_label:
                st.info(f"{ft_label}")
            # 厩舎近況
            s_label = selected_entry.get("stable_label", "")
            s_trend = selected_entry.get("stable_trend", "")
            if s_label:
                if s_trend in ("好調", "絶好調"):
                    st.success(f"{s_label}")
                elif s_trend == "不調":
                    st.warning(f"{s_label}")
                else:
                    st.caption(f"{s_label}")
        with nf_c2:
            # 頭数変化
            fc_label = selected_entry.get("field_size_label", "")
            if fc_label:
                st.caption(f"{fc_label}")
            # ペース適合
            cpf_label = selected_entry.get("pace_fit_label", "")
            cpf_shift = selected_entry.get("pace_shift", "")
            if cpf_label:
                if cpf_shift == "好転":
                    st.success(f"{cpf_label}")
                elif cpf_shift == "悪化":
                    st.warning(f"{cpf_label}")
                else:
                    st.caption(f"{cpf_label}")

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
        st.subheader("近走詳細分析")
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
                f"**近走言い訳あり → 巻き返し期待ボーナス +{resume_info['total_bonus']*100:.1f}pt**\n\n"
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
                    header += " 上がり最速"

                with st.expander(header, expanded=(i == 0 or is_notable)):
                    # シグナルタグ
                    if race["signals"]:
                        for sig in race["signals"]:
                            if any(kw in sig for kw in ["最速", "僅差", "先着"]):
                                st.success(f"{sig}")
                            elif any(kw in sig for kw in ["不利", "後退", "大敗", "不向き"]):
                                st.warning(f"{sig}")
                            elif any(kw in sig for kw in ["格上", "変わり"]):
                                st.info(f"ℹ{sig}")
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
                            st.success(f"{f3_rank_str}")
                        else:
                            st.caption(f"{f3_rank_str}")

                    # まとめ
                    if race["excuse"]:
                        st.error(f"言い訳: {race['excuse']}")
                    if race["plus"]:
                        st.success(f"強調材料: {race['plus']}")
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
            st.subheader("馬体重トレンド")
            wt_df = pd.DataFrame({"出走順": range(1, len(wt_info["weights"]) + 1),
                                   "馬体重(kg)": wt_info["weights"]})
            fig_wt = px.line(wt_df, x="出走順", y="馬体重(kg)",
                               title=f"{selected_horse} 馬体重推移（直近{len(wt_info['weights'])}走）",
                               markers=True)
            fig_wt.add_hline(y=wt_info["weights"][-1], line_dash="dash",
                             line_color="green", annotation_text="最新")
            st.plotly_chart(apply_chart_theme(fig_wt), use_container_width=True)
            if wt_info["bonus"] != 0:
                if wt_info["bonus"] > 0:
                    st.success(f"{wt_info['label']}")
                else:
                    st.warning(f"{wt_info['label']}")
            else:
                st.caption(f"{wt_info['label']}")

        # ---- 5. コーナー通過順位適性 ---- #
        from horse_profiler import get_corner_position_stats
        corner_info = get_corner_position_stats(df_hist, selected_horse, surface, distance)
        if corner_info["avg_1st_corner"] is not None:
            st.subheader("コーナー通過位置分析")
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
            st.subheader("ハンデ戦斤量分析")
            hc1, hc2 = st.columns(2)
            hc1.metric("ハンデ戦勝率", f"{hc_info['handicap_win_rate']*100:.0f}%")
            if hc_info["current_vs_best"] is not None:
                delta_str = f"過去最軽量比{hc_info['current_vs_best']:+.0f}kg"
                hc2.metric("今回斤量評価", delta_str,
                           delta_color="inverse" if hc_info["current_vs_best"] > 0 else "normal")
            if hc_info["bonus"] > 0:
                st.success(f"{hc_info['label']}")
            elif hc_info["bonus"] < 0:
                st.warning(f"{hc_info['label']}")
            else:
                st.caption(f"{hc_info['label']}")

        # ---- 8. 時計ランク比較 ---- #
        from horse_profiler import calc_time_rank
        time_info = calc_time_rank(df_hist, selected_horse,
                                    st.session_state.get("venue", "東京"), surface, distance)
        if time_info["best_time_raw"] is not None:
            st.subheader("時計ランク（レースレベル補正済み）")
            tc1, tc2, tc3 = st.columns(3)
            m, s = divmod(time_info["best_time_raw"], 60)
            tc1.metric("自己最高タイム", f"{int(m)}:{s:04.1f}")
            tc2.metric("基準タイム比", f"{time_info['best_time_adj']:+.2f}秒")
            tc3.metric("時計ランク", time_info["time_rank"])
            if time_info["time_rank_bonus"] > 0:
                st.success(f"{time_info['label']}")
            elif time_info["time_rank_bonus"] < 0:
                st.warning(f"{time_info['label']}")
            else:
                st.caption(f"{time_info['label']}")

        # ---- 有力馬撃破実績 ---- #
        from horse_profiler import calc_beaten_strong_horses
        beaten_info = calc_beaten_strong_horses(
            df_hist, selected_horse, surface, distance
        )
        if beaten_info["beat_count"] > 0:
            st.subheader("有力馬撃破実績")
            if beaten_info["bonus"] >= 0.02:
                st.success(f"{beaten_info['label']}")
            else:
                st.info(f"{beaten_info['label']}")
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
                    st.subheader("最終追い切り（併せ馬）")
                    p_won = hr.get("partner_won_sat", False)
                    won_aw = hr.get("won_awase")
                    if p_won:
                        st.success(f"{hr.get('partner_message', '')}")
                    elif won_aw is True:
                        st.info(f"{hr.get('partner_message', '')}")
                    else:
                        st.caption(f"{hr.get('partner_message', '')}")


# ============================================================
# TAB 3: 馬券構成
# ============================================================
with tab_bet:
    st.subheader("馬券構成")
    if "eval_df" not in st.session_state:
        st.info("先に「レース分析」タブで分析を実行してください。")
    else:
        eval_df_bet = st.session_state["eval_df"]
        surface_bet = st.session_state.get("surface", "芝")
        distance_bet = st.session_state.get("distance", 2000)
        bias_bet = st.session_state.get("bias_type", "neutral")

        # ---- 穴馬判定（自動提案サブタブで必要なので先に実行） ---- #
        if "structural_count" not in eval_df_bet.columns:
            eval_df_bet = evaluate_all_longshots(eval_df_bet)
            st.session_state["eval_df"] = eval_df_bet

        # ---- 馬券構成計算（全サブタブで共有） ---- #
        # 第21波: モード切替後の stale 検知 — 分析時と現在のモードが違うと
        # buy_flag が旧モード基準のままなので再分析を促す
        _mode_at_analysis = st.session_state.get("analyzed_with_mode")
        _mode_current = st.session_state.get("betting_mode", "爆穴")
        if _mode_at_analysis and _mode_at_analysis != _mode_current:
            st.warning(
                f"分析は「{_mode_at_analysis}モード」で実行されています。"
                f"現在は「{_mode_current}モード」のため、買い判定が古い基準のままです。"
                f"レース分析タブで再分析してください。"
            )
        # 第19波 (W3): 爆穴モードは複勝EVベースの専用構成に分岐
        if st.session_state.get("betting_mode") == "爆穴":
            from bet_builder import build_longshot_tickets
            result = build_longshot_tickets(eval_df_bet, budget=budget)
            if result.get("brain_warning"):
                st.warning(result["brain_warning"])
            else:
                st.caption("爆穴モード: 複勝EVベースの配分（複勝 Kelly 1/2 + ワイド）")
        else:
            result = build_tickets(
                eval_df_bet, budget=budget, surface=surface_bet,
                distance=distance_bet, bias_type=bias_bet,
            )

        _sub_auto, _sub_harv, _sub_buy = st.tabs(["自動提案", "Harville 多券種", "購入確認"])

        # ---- サブタブ①: 自動提案 ---- #
        with _sub_auto:
            longshots_df = eval_df_bet[eval_df_bet["popularity"] >= 7].sort_values(
                "structural_count", ascending=False
            )
            if not longshots_df.empty:
                st.markdown("**穴馬判定結果**")
                for _, row in longshots_df.head(4).iterrows():
                    verdict_emoji = row.get("verdict_emoji", "")
                    verdict = row.get("verdict", "")
                    summary = row.get("summary", "")
                    score = row.get("structural_count", 0)
                    _txt = f"<b>{row['horse_name']}</b> ({row['popularity']}番人気 / {row['odds']}倍) — {verdict}"
                    if score >= 3:
                        ui_banner("buy", _txt)
                    elif score >= 2:
                        ui_banner("warn", _txt)
                    else:
                        ui_banner("avoid", _txt)
                    if summary:
                        with st.expander(f"根拠詳細: {row['horse_name']}", expanded=False):
                            st.markdown(summary)
                st.divider()

            if result["brain_warning"]:
                ui_banner("avoid", result["brain_warning"])

            axis = result.get("axis", {})
            if axis.get("longshot_axis"):
                ui_banner("info",
                    f"<b>穴馬軸</b>: {axis['longshot_axis']} ({axis.get('ls_odds',0)}倍) &nbsp;|&nbsp; "
                    f"<b>人気馬軸</b>: {axis['popular_axis']} ({axis.get('pop_odds',0)}倍) &nbsp;|&nbsp; "
                    f"バイアス: {BIAS_TYPES.get(bias_bet,{}).get('label','不明')}"
                )

            st.markdown(f"#### 推奨プラン  <span style='color:#5F6B7A;font-weight:400;'>合計 {result['total_cost']:,}円 / 予算 {budget:,}円</span>", unsafe_allow_html=True)
            if result["recommended"]:
                st.dataframe(format_tickets_for_display(result["recommended"]),
                             use_container_width=True, hide_index=True)
                st.caption(f"残り予算: {result['remaining_budget']:,}円")
            else:
                ui_banner("warn", "EVプラスの馬がいないため、このレースは見送りを推奨します。")

            with st.expander("従来スタイル（大穴×人気馬 3連複全流し）との比較", expanded=False):
                if result["romance_plan"]:
                    st.dataframe(format_tickets_for_display(result["romance_plan"]),
                                 use_container_width=True, hide_index=True)
                    st.metric("従来スタイル費用", f"{result['romance_cost']:,}円",
                               delta=f"{result['romance_cost'] - result['total_cost']:+,}円（推奨との差）")

            if result["recommended"]:
                bet_amounts = {t.bet_type: t.amount for t in result["recommended"]}
                fig = px.pie(values=list(bet_amounts.values()), names=list(bet_amounts.keys()),
                             title=f"馬券構成内訳（合計 {result['total_cost']:,}円）")
                st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

            # A-6: 軸+押さえ自動構成
            st.divider()
            st.markdown("##### 軸 + 押さえ 自動構成（馬連 / 三連複）")
            from bet_builder import suggest_axis_and_partner as _saap
            _ap = _saap(eval_df_bet, budget=int(budget), max_partners=4)
            if _ap["axis"]:
                _ax = _ap["axis"]
                ui_banner("info",
                    f"<b>軸馬</b>: {_ax['horse_name']} (実力 {_ax['score']} / オッズ {_ax['odds']} / EV {_ax['ev']:+.2f})")
                if _ap["partners"]:
                    _p_strs = " · ".join(f"{p['horse_name']}({p['score']})" for p in _ap["partners"])
                    st.caption(f"押さえ: {_p_strs}")
                if _ap["umaren_bets"]:
                    st.markdown("**馬連**")
                    st.dataframe(pd.DataFrame(_ap["umaren_bets"])[["key", "prob_est", "amount"]]
                                 .rename(columns={"key": "組合せ", "prob_est": "推定確率", "amount": "金額"}),
                                 use_container_width=True, hide_index=True)
                if _ap["trio_bets"]:
                    st.markdown("**三連複**")
                    st.dataframe(pd.DataFrame(_ap["trio_bets"])[["key", "prob_est", "amount"]]
                                 .rename(columns={"key": "組合せ", "prob_est": "推定確率", "amount": "金額"}),
                                 use_container_width=True, hide_index=True)
                st.caption(f"合計 {_ap['total_amount']:,}円")

            # A-7: 複数券種で variance 分散（軸馬1頭に対し 単勝・複勝・ワイド配分）
            st.divider()
            st.markdown("##### 複数券種で variance 分散（軸馬1頭・自動配分）")
            from bet_builder import variance_diversify_bets as _vdb
            _axrow = None
            if "confidence_score" in eval_df_bet.columns and not eval_df_bet.empty:
                _axrow = eval_df_bet.sort_values("confidence_score", ascending=False).iloc[0]
            if _axrow is not None:
                _win_p = float(_axrow.get("blended_pct", 0) or 0) / 100
                _place_p = min(0.999, 1.0 - (1.0 - _win_p) ** 3)
                _win_o = float(_axrow.get("odds", 0) or 0)
                _place_o = float(_axrow.get("place_odds", 0) or 0)
                _vd = _vdb(_win_p, _place_p, _win_o, _place_o, None, None,
                           budget=int(budget * 0.5))   # 軸馬1頭への分散には予算の半分
                if _vd["bets"]:
                    st.dataframe(pd.DataFrame(_vd["bets"]),
                                 use_container_width=True, hide_index=True)
                    st.caption(f"合計 {_vd['total_amount']:,}円 — variance 分散で Sharpe 改善")
                else:
                    st.caption(_vd.get("note", "+EV な券種なし"))

        # ---- サブタブ②: Harville ---- #
        with _sub_harv:
            st.markdown("#### Harville 期待値計算（複数券種）")
            st.caption(
                "単勝勝率（Benter ブレンド済）から Harville (1973) で「i→j→k 着順確率」を導出。"
                "公平オッズ（控除率込み）を表示。実際の netkeiba オッズと比較して+EVな組合せを発見できます。"
            )

            _prob_col = "blended_pct" if "blended_pct" in eval_df_bet.columns else "lgbm_norm_pct"
            if _prob_col in eval_df_bet.columns and eval_df_bet[_prob_col].notna().any():
                from harville import top_n_combinations, TAKE_RATES, calc_ev_for_market_odds

                _h_col1, _h_col2, _h_col3 = st.columns([1, 1, 1])
                with _h_col1:
                    _h_ticket = st.selectbox(
                        "券種",
                        ["trio", "quinella", "exacta", "wide", "trifecta"],
                        format_func=lambda x: {
                            "trio": "三連複", "quinella": "馬連", "exacta": "馬単",
                            "wide": "ワイド", "trifecta": "三連単",
                        }[x],
                        key="harv_ticket",
                    )
                with _h_col2:
                    _h_topn = st.number_input("対象上位N頭", 3, 12, 6, key="harv_topn")
                with _h_col3:
                    _h_show = st.number_input("表示件数", 5, 50, 15, key="harv_show")

                _ed_sorted = eval_df_bet.sort_values("horse_no").reset_index(drop=True)
                _win_probs = (_ed_sorted[_prob_col].fillna(0) / 100).values
                _horse_nos = _ed_sorted["horse_no"].astype(int).tolist()
                _horse_names = dict(zip(_ed_sorted["horse_no"].astype(int), _ed_sorted["horse_name"]))

                _combos = top_n_combinations(
                    _win_probs,
                    ticket_type=_h_ticket,
                    top_n=int(_h_topn),
                    horse_nos=_horse_nos,
                    take_rate=TAKE_RATES.get(_h_ticket, 0.225),
                )

                # --- 実オッズ自動取得（race_id がある場合） ---
                _rid_for_odds = st.session_state.get("preselected_race_id", "")
                _auto_odds_on = st.toggle(
                    "netkeibaから実オッズ自動取得",
                    value=bool(_rid_for_odds), key="harv_auto_odds",
                    help="ON：レースの実オッズを取得して+EV組合せを自動判定",
                )
                _real_odds_map = {}
                if _auto_odds_on and _rid_for_odds:
                    from scraper import fetch_multi_odds as _fmo
                    try:
                        with st.spinner("実オッズ取得中..."):
                            _all_real = _fmo(_rid_for_odds)
                        _real_odds_map = _all_real.get(_h_ticket, {})
                        st.caption(f"実オッズ取得：{len(_real_odds_map)}件（{_h_ticket}）")
                    except Exception as _e_fmo:
                        st.warning(f"オッズ取得失敗: {_e_fmo}")

                # 組合せキーを netkeiba API 形式（zero-padded ハイフン区切り）に変換
                def _combo_to_key(horses):
                    return "-".join(str(int(h)).zfill(2) for h in horses)

                _rows = []
                for c in _combos[:int(_h_show)]:
                    names = " - ".join(_horse_names.get(n, str(n)) for n in c["horses"])
                    _real_odds = _real_odds_map.get(_combo_to_key(c["horses"]))
                    row = {
                        "組合せ": " - ".join(str(n) for n in c["horses"]),
                        "馬名": names,
                        "Harville確率%": round(c["prob"] * 100, 3),
                        "公平オッズ": round(c["fair_odds"], 1) if c["fair_odds"] != float("inf") else "∞",
                    }
                    if _real_odds is not None:
                        row["実オッズ"] = round(_real_odds, 1)
                        row["EV"] = round(calc_ev_for_market_odds(c["prob"], _real_odds), 3)
                    _rows.append(row)
                _df_harv = pd.DataFrame(_rows)

                # +EV 行をハイライト
                def _color_ev(val):
                    try:
                        v = float(val)
                        if v >= 0.1:  return f"background-color: #EEF3EC; color: {COLOR['buy']}; font-weight:600"
                        if v >= 0:    return f"color: {COLOR['buy']}"
                        return f"color: {COLOR['avoid']}"
                    except Exception:
                        return ""
                _styled_harv = _df_harv.style
                if "EV" in _df_harv.columns:
                    _styled_harv = _styled_harv.map(_color_ev, subset=["EV"])
                st.dataframe(_styled_harv, use_container_width=True, hide_index=True)

                # +EV サマリ
                if "EV" in _df_harv.columns:
                    _plus_df = _df_harv[_df_harv["EV"] > 0]
                    if not _plus_df.empty:
                        _top = _plus_df.iloc[0]
                        ui_banner("buy",
                            f"<b>+EV組合せ {len(_plus_df)}件発見</b>　最上位: {_top['組合せ']} "
                            f"(実オッズ {_top['実オッズ']}倍 / EV {_top['EV']:+.3f})")
                    else:
                        ui_banner("muted", "現時点で +EV な組合せはありません。")

                # BOOST-1: 自動買い目最適化（予算内・ハーフケリー）
                if _real_odds_map:
                    from bet_builder import optimize_multi_bet_harville as _omh
                    st.markdown("##### 自動買い目最適化")
                    _opt_budget = st.number_input(
                        "予算（円）", 100, 50000, int(budget), 100,
                        key="harv_opt_budget",
                    )
                    _opt_max = st.number_input("最大点数", 1, 20, 6, key="harv_opt_max")
                    _opt_bets = _omh(
                        combos=_combos,
                        real_odds_map=_real_odds_map,
                        budget=int(_opt_budget),
                        max_bets=int(_opt_max),
                        fraction=float(st.session_state.get("kelly_fraction_ratio", 0.50)),
                    )
                    if _opt_bets:
                        _opt_rows = []
                        _total_amt = 0
                        _total_exp = 0.0
                        for b in _opt_bets:
                            _names = " - ".join(_horse_names.get(int(n), str(n)) for n in b["horses"])
                            _opt_rows.append({
                                "組合せ": " - ".join(str(int(n)) for n in b["horses"]),
                                "馬名": _names,
                                "実オッズ": b["real_odds"],
                                "EV": round(b["ev"], 3),
                                "配分%": b["kelly_pct"],
                                "金額": b["amount"],
                                "期待払戻": int(b["prob"] * b["real_odds"] * b["amount"]),
                            })
                            _total_amt += b["amount"]
                            _total_exp += b["prob"] * b["real_odds"] * b["amount"]
                        _df_opt = pd.DataFrame(_opt_rows)
                        st.dataframe(_df_opt, use_container_width=True, hide_index=True)
                        ui_banner("buy",
                            f"合計 {_total_amt:,}円  /  期待払戻 {int(_total_exp):,}円  "
                            f"({int(_total_exp / max(_total_amt, 1) * 100)}%)")
                    else:
                        ui_banner("muted", "+EV な組合せが無いため自動買い目はありません。")

                st.caption(
                    "公平オッズ：Harville 確率から逆算した「控除前理論オッズ」。"
                    "実オッズ > 公平オッズ なら +EV の買い目。"
                )

                with st.expander("実オッズを入力してEV計算（任意）", expanded=False):
                    st.caption("半角カンマ区切りで実オッズを並べる（上から順）。例: `12.5, 8.3, 15.0`")
                    _odds_str = st.text_input("実オッズ列", key="harv_real_odds", placeholder="12.5, 8.3, 15.0")
                    if _odds_str.strip():
                        try:
                            _real_odds = [float(x.strip()) for x in _odds_str.split(",")]
                            _df_ev = _df_harv.head(len(_real_odds)).copy()
                            _df_ev["実オッズ"] = _real_odds
                            _df_ev["EV"] = [
                                round(calc_ev_for_market_odds(c["prob"], o), 3)
                                for c, o in zip(_combos[:len(_real_odds)], _real_odds)
                            ]
                            _df_ev["判定"] = _df_ev["EV"].apply(
                                lambda e: "◎ +EV" if e > 0.1 else ("○ EV+" if e > 0 else "✗ EV-")
                            )
                            st.dataframe(_df_ev, use_container_width=True, hide_index=True)
                            _plus = _df_ev[_df_ev["EV"] > 0]
                            if not _plus.empty:
                                ui_banner("buy",
                                    f"<b>+EV組合せ {len(_plus)}件発見</b>　上位: {_plus.iloc[0]['組合せ']} (EV={_plus.iloc[0]['EV']:+.3f})")
                        except ValueError:
                            ui_banner("avoid", "数値変換失敗。`12.5, 8.3` 形式で入力してください。")
            else:
                st.info(f"勝率列 ({_prob_col}) が見つかりません。分析を再実行してください。")

        # ---- サブタブ③: 購入確認 ---- #
        with _sub_buy:
            st.markdown("#### 購入前の最終確認")
            race_name_bet = st.session_state.get("race_name_for_save", "このレース")

            purchased = st.session_state.get("purchased_races", [])
            rule_limit = st.session_state.get("rule_max_races", 3)
            if len(purchased) >= rule_limit:
                ui_banner("avoid",
                    f"<b>今週の購入上限（{rule_limit}レース）に達しています</b> — 購入を中断することを強く推奨します。")

            impulse_risk = False
            if any(kw in race_name_bet for kw in ["ハンデ", "ハンデキャップ"]):
                ui_banner("warn", "このレースは<b>ハンデ戦</b>です。衝動買い注意レースに指定されています。")
                impulse_risk = True

            conviction_reason = st.text_input(
                "この馬券を買う最大の根拠を一言で入力（確信があれば書ける）",
                placeholder="例：○○は前走でハイペースに巻き込まれた展開負けで、今回スロー想定の展開が向く",
                key="conviction_reason",
            )

            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button(
                    "根拠あり・購入確定",
                    type="primary",
                    disabled=not conviction_reason,
                    key="confirm_purchase",
                    use_container_width=True,
                ):
                    if not conviction_reason:
                        ui_banner("avoid", "根拠を入力してください")
                    else:
                        purchased.append({
                            "race": race_name_bet,
                            "reason": conviction_reason,
                            "amount": result["total_cost"],
                            "tickets": [{"type": t.bet_type, "horses": t.horses, "amount": t.amount}
                                        for t in result["recommended"]],
                        })
                        st.session_state["purchased_races"] = purchased
                        ui_banner("buy",
                            f"<b>購入記録しました（今週 {len(purchased)}/{rule_limit} レース）</b><br>"
                            f"根拠：{conviction_reason}")
                        if _get_webhook_url():
                            ticket_dicts = [
                                {"bet_type": t.bet_type, "horses": t.horses,
                                 "amount": t.amount, "ev": getattr(t, "ev", None)}
                                for t in result["recommended"]
                            ]
                            notify_bet_plan(race_name_bet, ticket_dicts, result["total_cost"])
                            st.toast("Discord に買い目を送信しました", icon="")
            with col_confirm2:
                if st.button("見送る", key="cancel_purchase", use_container_width=True):
                    ui_banner("muted", "賢明な判断です。見送り記録しました。")


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
    st.subheader("レース振り返り日記")
    diary_tab1, diary_tab2, diary_tab3, diary_tab4 = st.tabs(
        ["予想を記録", "結果を取得", "週次レポート", "改善ループ"]
    )

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
                # 第19波 (W3): 日記の買い目自動入力もモード連動
                if st.session_state.get("betting_mode") == "爆穴":
                    from bet_builder import build_longshot_tickets
                    bet_result = build_longshot_tickets(eval_res, budget=budget)
                else:
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

            if st.button("予想を日記に保存", type="primary"):
                # 第13波: 改善ループ用の自動メタデータを収集
                # 第17波 (Z2-Z5 修正): 実在キー名へ全面修正
                #   conformal: 実キーは mean_interval_width / recommend_skip
                #   データ不足: 実列は data_quality_ok (True=OK)
                #   kelly: 実 widget key は sld_kelly_ratio
                #   オッズ急変: ホームタブで保存する odds_alert_count を読む
                _ib = st.session_state.get("intraday_bias_result", {})
                _conf = st.session_state.get("conformal_result", {})
                _dq_bad = 0
                if "data_quality_ok" in eval_df_diary.columns:
                    _dq_bad = int((~eval_df_diary["data_quality_ok"].fillna(True)).sum())
                _top_pop = None
                if not eval_df_diary.empty:
                    _tp_raw = pd.to_numeric(
                        pd.Series([eval_df_diary.iloc[0].get("popularity")]), errors="coerce").iloc[0]
                    _top_pop = int(_tp_raw) if pd.notna(_tp_raw) else None
                auto_meta = {
                    "betting_mode": st.session_state.get("betting_mode", "爆穴"),
                    "data_unavailable_count": _dq_bad,
                    "conformal_interval_width": _conf.get("mean_interval_width"),
                    "conformal_skip_recommended": bool(_conf.get("recommend_skip", False)),
                    "odds_alert_count": int(st.session_state.get("odds_alert_count", 0)),
                    "top_horse_popularity": _top_pop,
                    "ev_threshold_used": st.session_state.get("ev_threshold_active"),
                    "kelly_fraction": st.session_state.get("sld_kelly_ratio"),
                    "intraday_bias_applied": bool(st.session_state.get("intraday_bias_apply", False)),
                    "bet_count": len(rec_bets),
                    # 第34波: 主戦場判定の事後検証用
                    "volatility_score": (st.session_state.get("volatility_result") or {}).get("score"),
                }
                rid = save_race_prediction(
                    race_id=diary_race_id,
                    race_name=diary_race_name,
                    race_date=str(target_date),
                    venue=st.session_state.get("venue", ""),
                    surface=st.session_state.get("surface", "芝"),
                    distance=st.session_state.get("distance", 2000),
                    track_condition=st.session_state.get("track_condition", "良"),
                    eval_df=eval_df_diary,
                    bets=rec_bets,
                    bias_type=st.session_state.get("bias_type", ""),
                    pace_predicted=st.session_state.get("pace_info", {}).get("predicted_pace", ""),
                    budget=budget,
                    note=diary_note,
                    auto_meta=auto_meta,
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
                    fetch_btn = st.button("結果を自動取得", type="primary")

                if fetch_btn and result_race_id:
                    with st.spinner("netkeibaから着順・払い戻しを取得中..."):
                        fetched = fetch_race_result_from_netkeiba(result_race_id)
                    if fetched["fetched"] and fetched["results"]:
                        save_result_to_diary(selected_id, fetched, [])
                        # 第18波 (W1): 改善ループ用に勝ち馬人気を auto_meta へ追記
                        try:
                            from race_diary import update_auto_meta_with_result as _uamr
                            _win_pop = pd.to_numeric(
                                pd.Series([fetched["results"][0].get("popularity")]),
                                errors="coerce").iloc[0]
                            _uamr(selected_id, int(_win_pop) if pd.notna(_win_pop) else None)
                        except Exception as _e_uam:
                            print(f"[auto_meta] 勝ち馬人気の追記失敗: {_e_uam}")
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
                    st.markdown("#### Claudeによる振り返り分析")
                    st.caption("「なぜ外れたか」「選ばなかった馬の正体」「次回の教訓」を自動分析")
                    if st.button("Claude に振り返りを分析させる", key="claude_postrace"):
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
                                    st.toast("Discord に振り返りを送信しました", icon="")
                            else:
                                st.warning("レースIDを入力してから分析してください")

                # 保存済み分析の表示
                saved_analysis = st.session_state.get(f"postrace_analysis_{selected_id}")
                if saved_analysis:
                    with st.expander("保存済み振り返り分析"):
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
            st.plotly_chart(apply_chart_theme(fig_roi), use_container_width=True)

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
                if st.button("週次レポートをDiscordに送信", key="discord_weekly"):
                    ok = notify_weekly_report(
                        weekly_df, overall_roi,
                        int(total_inv), int(total_ret),
                    )
                    if ok:
                        st.success("Discordに送信しました")
                    else:
                        st.error("送信失敗。Webhook URLを確認してください")
            else:
                st.caption("Discord通知を設定するとここからレポートを送信できます")

        # ファクター精度
        if not factor_df.empty:
            st.divider()
            st.markdown("#### スコア別的中率（ファクター精度）")
            st.caption("スコアが高い馬が実際に3着以内に来る確率")
            import plotly.express as px
            fig_fac = px.bar(factor_df, x="score_bucket", y="hit_rate",
                             title="スコア帯別 複勝的中率(%)",
                             labels={"score_bucket": "スコア帯", "hit_rate": "的中率(%)"},
                             text="hit_rate")
            st.plotly_chart(apply_chart_theme(fig_fac), use_container_width=True)

        # ---- 4. 穴馬履歴サマリー ---- #
        st.divider()
        st.markdown("#### 穴馬予想パターン分析（あなたの的中傾向）")
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
                    st.markdown("**得意な条件（的中率高）**")
                    st.dataframe(ls_summary["best_conditions"],
                                 use_container_width=True, hide_index=True)
            with bc2:
                if not ls_summary["miss_conditions"].empty:
                    st.markdown("**苦手な条件（的中率低）**")
                    st.dataframe(ls_summary["miss_conditions"],
                                 use_container_width=True, hide_index=True)
        else:
            st.info("穴馬予想の記録がまだありません。日記に記録が貯まると自動で分析されます。")

        # 全記録テーブル
        all_rec = get_all_records()
        if not all_rec.empty:
            st.divider()
            st.markdown("#### 全レース記録")
            show_cols = [c for c in ["race_date","race_name","venue","surface","distance",
                                      "total_invested","total_returned","roi","hits","note"]
                         if c in all_rec.columns]
            st.dataframe(all_rec[show_cols].rename(columns={
                "race_date":"日付","race_name":"レース名","venue":"会場",
                "surface":"馬場","distance":"距離","total_invested":"投資",
                "total_returned":"回収","roi":"回収率%","hits":"的中","note":"メモ"
            }), use_container_width=True, hide_index=True)

    # --- 改善ループ（第13波）---
    with diary_tab4:
        st.markdown("#### 改善ループ — 外れ要因の自動分類")
        st.caption(
            "予想時点の自動メタデータから外れ要因を 4 カテゴリに分類し、最頻カテゴリに応じた改善アドバイスを表示します。"
        )
        from race_diary import get_failure_breakdown as _gfb
        _days = st.selectbox("集計期間", [30, 60, 90, 180, 365], index=3,
                              format_func=lambda d: f"過去 {d} 日")
        _br = _gfb(since_days=_days)

        col1, col2, col3 = st.columns(3)
        col1.metric("記録レース", _br["total_races"])
        col2.metric("当たり", _br["win_races"])
        col3.metric("外れ", _br["lose_races"])

        if _br["total_races"] == 0:
            st.info(_br["recommendations"][0])
        else:
            st.markdown("##### 外れ要因の内訳")
            cat_labels = {
                "A": "A 当日要因（人気薄激走・馬場急変等）",
                "B": "B データ不足馬の絡み",
                "C": "C モデル信頼度低の場面",
                "D": "D 戦略ミス（本来当てるべき）",
            }
            cat_df = pd.DataFrame([
                {"カテゴリ": cat_labels[k], "件数": v,
                 "割合": f"{v/max(_br['lose_races'],1)*100:.0f}%"}
                for k, v in _br["categories"].items()
            ])
            st.dataframe(cat_df, hide_index=True, use_container_width=True)

            st.markdown("##### 改善アドバイス")
            for msg in _br["recommendations"]:
                st.info(msg)

            with st.expander("各カテゴリの判定ロジック", expanded=False):
                st.markdown("""
| カテゴリ | 判定基準 | 対策 |
|---|---|---|
| **A 当日要因** | 勝ち馬が 8 人気以下 / バイアスに「波乱」「不良」含む | 構造的に困難。爆穴モードで EV>0.25 に厳選 |
| **B データ不足** | 出走馬中 1 頭以上が「データ不足」フラグ | 「データ不足」表示の馬は買わない |
| **C モデル崩れ** | Conformal 見送り推奨（混戦判定）だった | 見送り推奨レースはスキップ |
| **D 戦略ミス** | A/B/C いずれにも該当しない外れ | EV 閾値・Kelly 倍率を見直す |
                """)

# ============================================================
# TAB 8: バックテスト
# ============================================================
with tab_backtest:
    st.subheader("過去データ集計（人気別的中率）")
    st.caption("TFJV には単勝オッズが無いため、ベンチマークとして「人気別の勝率・複勝率」と「穴馬複勝率の時系列」を集計します。狙い目スタイルの仮説検証用。")

    if df_hist.empty:
        st.warning("過去データが読み込めていません。")
    else:
        with st.spinner("集計中..."):
            sim = _run_backtest(df_hist, win_rate_table)

        col1, col2, col3 = st.columns(3)
        col1.metric("1番人気の勝率", f"{sim['fav_win']:.1f}%",
                    help=f"サンプル {sim['fav_n']:,}走")
        col2.metric("10人気以下の複勝率", f"{sim['longshot_in3']:.1f}%",
                    help=f"サンプル {sim['longshot_n']:,}走")
        col3.metric("過去データ総走行数", f"{len(df_hist):,}")

        st.markdown("##### 人気帯別 勝率 / 複勝率")
        if not sim["by_pop"].empty:
            st.dataframe(
                sim["by_pop"][["pop_band", "races", "win_rate", "place_rate"]].rename(
                    columns={"pop_band": "人気帯", "races": "サンプル",
                             "win_rate": "勝率(%)", "place_rate": "複勝率(%)"}),
                use_container_width=True, hide_index=True,
            )

        if not sim["monthly_long_in3"].empty:
            fig = px.line(sim["monthly_long_in3"], x="month", y="in3_rate",
                          title="月別: 10人気以下の複勝率（直近24ヶ月）",
                          labels={"month": "月", "in3_rate": "複勝率(%)"})
            fig.add_hline(y=15, line_dash="dot", line_color="orange",
                          annotation_text="長期平均≒15%")
            st.plotly_chart(apply_chart_theme(fig), use_container_width=True)

        st.caption("💡 単勝回収率バックテストは odds 列が必要だが TFJV データには無いため停止中。"
                   "復活させるには別ソースからのオッズ取込が必要。")


# ============================================================
# TAB 5: 騎手・血統ランキング
# ============================================================
with tab_jockey:
    st.subheader("人気薄での好成績騎手")
    st.caption("対象：10番人気以下の馬での騎乗成績（最低騎乗20回）。条件別に4タブで切替可能。")

    if jockey_stats.empty:
        st.info("騎手データが生成できません。")
    else:
        # 短期外国人リスト取得
        try:
            from knowledge_base import load_kb as _load_kb_jky
            _kb = _load_kb_jky()
            _stf_list = [j.get("name", "") for j in _kb.get("short_term_foreign_jockeys", {}).get("notable", [])
                         if not j.get("resident", False)]
            _resident_foreign = [j.get("name", "") for j in _kb.get("short_term_foreign_jockeys", {}).get("notable", [])
                                  if j.get("resident", False)]
        except Exception:
            _stf_list = []
            _resident_foreign = []

        _all_foreign = set(_stf_list + _resident_foreign)

        def _fmt_jockey_table(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
            df = df.copy()
            if df.empty:
                return df
            df["place_rate_longshot"] = (df["place_rate_longshot"] * 100).round(1)
            df["win_rate_longshot"]   = (df["win_rate_longshot"]   * 100).round(1)
            return (df.head(n)[["jockey", "rides", "wins", "places", "place_rate_longshot", "win_rate_longshot"]]
                    .rename(columns={
                        "jockey": "騎手", "rides": "穴騎乗数", "wins": "穴勝利",
                        "places": "穴複勝", "place_rate_longshot": "穴複勝率%",
                        "win_rate_longshot": "穴勝率%",
                    }))

        _t_all, _t_local, _t_stf, _t_leader = st.tabs(
            ["全騎手", "国内ベース（騎乗200+）", "短期外国人", "リーディング順"]
        )

        with _t_all:
            st.dataframe(_fmt_jockey_table(jockey_stats, 30),
                         use_container_width=True, hide_index=True)
            st.caption("※外国人エース騎手（ルメール・モレイラ等）は人気薄でも質の高い騎乗で勝つため上位に来やすい。")

        with _t_local:
            _local = jockey_stats[~jockey_stats["jockey"].isin(_all_foreign)]
            _local = _local[_local["rides"] >= 200].copy()
            if _local.empty:
                st.info("該当騎手がいません。")
            else:
                st.dataframe(_fmt_jockey_table(_local, 30),
                             use_container_width=True, hide_index=True)
                st.caption("外国人騎手を除外、騎乗200回以上で絞り込んだ国内ベース騎手のランキング。")

        with _t_stf:
            _stf_df = jockey_stats[jockey_stats["jockey"].isin(_stf_list)].copy()
            if _stf_df.empty:
                st.info("短期外国人騎手の該当データがありません。")
            else:
                st.dataframe(_fmt_jockey_table(_stf_df, 30),
                             use_container_width=True, hide_index=True)
                st.caption("短期免許で来日経験のある外国人騎手（在留組除く）。")

        with _t_leader:
            _ld = jockey_stats.copy()
            _ld["leader_score"] = _ld["wins"] / _ld["rides"].clip(lower=1)
            _ld = _ld.sort_values("leader_score", ascending=False)
            st.dataframe(_fmt_jockey_table(_ld, 30),
                         use_container_width=True, hide_index=True)
            st.caption("穴勝率（wins / rides）で並び替えたランキング。")

    st.divider()
    st.subheader("血統×距離帯 適性ランキング")
    if sire_stats.empty:
        st.info("血統データが生成できません。")
    else:
        import numpy as _np
        _sc1, _sc2, _sc3, _sc4 = st.columns([1.5, 1.5, 1.5, 2])
        with _sc1:
            _sort_mode = st.radio(
                "並び替え",
                ["信頼度加重", "勝率順", "出走数順"],
                key="sire_sort",
                help="信頼度加重 = 勝率 × √出走数。少数派の偏ったデータを抑える",
            )
        with _sc2:
            _dist_filter = st.selectbox(
                "距離帯", ["全て", "短距離", "マイル", "中距離", "長距離"],
                key="sire_dist_filter",
            )
        with _sc3:
            _min_races = st.slider("最低出走数", 10, 500, 50, 10, key="sire_min_races")
        with _sc4:
            _search = st.text_input("種牡馬名検索", placeholder="例: Deep, Lord", key="sire_search")

        _sire_show = sire_stats.copy()
        if _dist_filter != "全て":
            _sire_show = _sire_show[_sire_show["distance_cat"] == _dist_filter]
        _sire_show = _sire_show[_sire_show["races"] >= _min_races]
        if _search.strip():
            _sire_show = _sire_show[_sire_show["sire"].str.contains(_search.strip(), case=False, na=False)]

        if _sort_mode == "勝率順":
            _sire_show = _sire_show.sort_values("win_rate", ascending=False)
        elif _sort_mode == "出走数順":
            _sire_show = _sire_show.sort_values("races", ascending=False)
        else:  # 信頼度加重
            _sire_show = _sire_show.copy()
            _sire_show["score"] = _sire_show["win_rate"] * _np.sqrt(_sire_show["races"].clip(lower=1))
            _sire_show = _sire_show.sort_values("score", ascending=False)

        _sire_show_disp = _sire_show.head(100).copy()
        _sire_show_disp["win_rate"] = (_sire_show_disp["win_rate"] * 100).round(2)
        _disp_cols = ["sire", "distance_cat", "races", "wins", "win_rate"]
        _rename = {"sire": "父", "distance_cat": "距離帯",
                   "races": "出走数", "wins": "勝利数", "win_rate": "勝率%"}
        st.dataframe(
            _sire_show_disp[_disp_cols].rename(columns=_rename),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"表示: {len(_sire_show_disp)} / 該当 {len(_sire_show)} / 全 {len(sire_stats)} 行")

        with st.expander("ピボット表示（sire × 距離帯 ヒートマップ）", expanded=False):
            _pivot = _sire_show.pivot_table(
                index="sire", columns="distance_cat", values="win_rate",
                aggfunc="mean",
            ).fillna(0) * 100
            _pivot = _pivot.round(2).head(50)
            st.dataframe(
                _pivot.style.background_gradient(cmap="YlGn", axis=None),
                use_container_width=True,
            )

# ============================================================
# TAB 11: メモ編集（ナレッジベース）
# ============================================================
with tab_kb:
    st.subheader("競馬メモ編集（ナレッジベース）")
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
        action_tag = "買い" if action == "buy" else ("消し" if action == "avoid" else f"{action}")
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


# ============================================================
# TAB 10: WIN5 3点組合せ
# ============================================================
with tab_win5:
    st.header("🎯 WIN5 組合せ提案")
    st.caption("対象5レースを取得 → 各レース1着確率を計算 → 指定点数内で的中確率最大の配分を自動探索。")

    _w5c1, _w5c2, _w5c3 = st.columns([1, 1, 1])
    with _w5c1:
        _w5_pts = st.slider("最大点数", 1, 12, 3, key="win5_max_pts",
                            help="100円×N点。混戦2レース×2頭軸ずつなら4点必要")
    with _w5c2:
        _w5_mode = st.radio("予測モード", ["オッズ簡易", "フル統合"],
                            key="win5_mode", horizontal=True,
                            help="フル統合 = LGBM + 調教 + Isotonic校正 + sum-normalize")
    with _w5c3:
        st.metric("投資額", f"{_w5_pts*100}円")

    with st.expander("💡 想定配当ってなに？", expanded=False):
        st.markdown("""
**期待値 = 的中確率 × 想定配当** で計算する。WIN5は配当のブレが大きい：

- **median(110万)** = 直近1年の中央値。**通常時はこれを使う**
- **mean(280万)** = 平均。1票で1億超えの特大配当が混じってる時の参考
- **p25(40万)** = 第1四分位。**堅く決まりそうな週**は控えめにこれ
- **p75(250万)** = 第3四分位。**混戦多い週**で大きい配当期待
- **キャリーオーバー額** は上の想定配当に**自動加算**される（CO 5億なら +5億）

判断目安：
- 全体が堅そう（堅レース3つ以上）→ **median or p25**
- 全体が混戦（混戦3つ以上）→ **p75 or mean**
        """)
    _w5p1, _w5p2 = st.columns([1, 1])
    with _w5p1:
        _w5_payout_mode = st.selectbox(
            "想定配当（CO=0時）",
            ["自動（難易度から推定）", "median(110万)", "mean(280万)", "p25(40万)", "p75(250万)", "カスタム"],
            index=0, key="win5_payout_mode")
    with _w5p2:
        _w5_custom_payout = st.number_input(
            "カスタム配当（円）", min_value=0, value=1_000_000, step=100_000,
            key="win5_custom_payout", disabled=(_w5_payout_mode != "カスタム"))

    if st.button("① WIN5対象レースを取得", key="win5_fetch_btn"):
        from win5_fetcher import fetch_win5_races
        with st.spinner("netkeibaから取得中..."):
            info = fetch_win5_races()
        if not info["race_ids"]:
            st.error("WIN5対象レースが取得できませんでした")
        else:
            st.session_state["win5_info"] = info
            st.success(f"取得完了：{len(info['race_ids'])}レース")

    if "win5_info" in st.session_state:
        info = st.session_state["win5_info"]
        st.write(f"**{info.get('title','WIN5')}**")
        if info.get("carryover_yen"):
            st.info(f"💰 キャリーオーバー: **{info['carryover_yen']:,}円**")

        st.markdown("### 対象5レース")
        from scraper import VENUE_CODES as _VC_W5, fetch_race_meta as _fmeta_w5
        _code2v = {v: k for k, v in _VC_W5.items()}
        # メタ情報キャッシュ
        if "win5_race_meta" not in st.session_state or st.session_state.get("win5_race_meta_key") != tuple(info["race_ids"]):
            _metas = []
            with st.spinner("レース詳細取得中..."):
                for _rid in info["race_ids"]:
                    try:
                        _m = _fmeta_w5(_rid) or {}
                    except Exception:
                        _m = {}
                    _metas.append(_m)
            st.session_state["win5_race_meta"] = _metas
            st.session_state["win5_race_meta_key"] = tuple(info["race_ids"])
        _metas = st.session_state["win5_race_meta"]
        for i, (rid, m) in enumerate(zip(info["race_ids"], _metas), 1):
            _venue = _code2v.get(rid[4:6], "?")
            _rno = int(rid[-2:])
            _rname = m.get("race_name", "") or ""
            _surface = m.get("surface", "")
            _dist = m.get("distance", "")
            _cond = m.get("track_condition", "")
            _meta_str = f" — {_surface}{_dist}m" if _surface else ""
            _cond_str = f" / 馬場:{_cond}" if _cond else ""
            st.text(f"R{i}: {_venue}{_rno}R 「{_rname}」{_meta_str}{_cond_str}")

        st.markdown("---")
        _w5cb1, _w5cb2 = st.columns([3, 1])
        with _w5cb1:
            _do_calc = st.button("② 組合せを計算", key="win5_calc_btn", use_container_width=True)
        with _w5cb2:
            if st.button("🗑️ キャッシュクリア", key="win5_clear_cache", use_container_width=True,
                         help="モード変更や予測再実行が必要なときに押す"):
                st.session_state.pop("win5_races_data", None)
                st.session_state.pop("win5_races_data_key", None)
                st.success("クリア済み")

        if _do_calc:
            from scraper import fetch_race_entries, fetch_multi_odds
            from win5_optimizer import optimize_win5
            from win5_predict import predict_race_win_probs, load_training_cache
            from win5_payout import (estimate_expected_payout, compute_ev,
                                       cross_race_confidence, PAYOUT_STATS,
                                       auto_select_payout_mode)

            # races_data セッションキャッシュ（race_ids + モードで鍵管理）
            _cache_key = (tuple(info["race_ids"]), _w5_mode)
            if st.session_state.get("win5_races_data_key") == _cache_key:
                races_data = st.session_state["win5_races_data"]
                st.caption(f"⚡ キャッシュヒット（{_w5_mode}・全{len(races_data)}レース）")
            else:
                _tr_cache = load_training_cache() if _w5_mode == "フル統合" else None
                races_data = []
                prog = st.progress(0.0)
                with st.spinner(f"各レース確率算出中（{_w5_mode}）..."):
                    for ridx, rid in enumerate(info["race_ids"]):
                        try:
                            pairs = []

                            if _w5_mode == "フル統合":
                                pairs = predict_race_win_probs(
                                    rid, win_rate_table, sire_stats, jockey_stats,
                                    training_results=_tr_cache, apply_calibration=True,
                                )

                            if not pairs:
                                # オッズ簡易 / フォールバック
                                entries = fetch_race_entries(rid)
                                odds_info = fetch_multi_odds(rid)
                                win_odds = (odds_info or {}).get("win", {})
                                no2name = {str(e.get("horse_no", "")): e.get("horse_name", "") for e in entries}
                                for no, od in win_odds.items():
                                    try:
                                        od_f = float(od)
                                        if od_f > 0:
                                            pairs.append((no2name.get(str(no), f"#{no}"), 1.0 / od_f))
                                    except Exception:
                                        pass
                                if not pairs:
                                    pairs = [(e.get("horse_name", f"#{i}"), 0.5 / (i + 1))
                                             for i, e in enumerate(entries)]
                                total = sum(p for _, p in pairs)
                                if total > 0:
                                    pairs = [(n, p / total) for n, p in pairs]

                            races_data.append(pairs)
                        except Exception as ex:
                            st.warning(f"R{ridx+1} 取得失敗 {rid}: {ex}")
                            races_data.append([])
                        prog.progress((ridx + 1) / 5)
                # キャッシュに保存
                st.session_state["win5_races_data"] = races_data
                st.session_state["win5_races_data_key"] = _cache_key

            if len(races_data) == 5 and all(races_data):
                result = optimize_win5(races_data, max_points=_w5_pts)
                st.markdown(f"### 結果: {result['ev_pattern']} = **{result['points']}点 / {result['cost_yen']}円**")

                # 横断スコアを先に計算（自動配当判定に必要）
                conf = cross_race_confidence(races_data)

                # 想定配当の決定
                _auto_reason = ""
                if _w5_payout_mode == "カスタム":
                    base_payout = int(_w5_custom_payout)
                    _mode_label = "カスタム"
                elif _w5_payout_mode.startswith("自動"):
                    _auto_mode, _auto_reason = auto_select_payout_mode(conf)
                    base_payout = PAYOUT_STATS.get(_auto_mode, PAYOUT_STATS["median"])
                    _mode_label = f"自動→{_auto_mode}（{_auto_reason}）"
                else:
                    _mode_key = _w5_payout_mode.split("(")[0]
                    base_payout = PAYOUT_STATS.get(_mode_key, PAYOUT_STATS["median"])
                    _mode_label = _w5_payout_mode
                co = info.get("carryover_yen") or 0
                expected_payout = base_payout + co
                ev = compute_ev(result["hit_prob"], expected_payout, result["cost_yen"])

                _cc1, _cc2, _cc3 = st.columns(3)
                with _cc1:
                    st.metric("理論的中確率", f"{result['hit_prob']*100:.4f}%")
                with _cc2:
                    st.metric("想定配当", f"{expected_payout:,}円",
                              delta=f"+CO {co:,}" if co else _mode_label,
                              help=f"配当根拠: {_mode_label}")
                with _cc3:
                    st.metric("期待値（手取り）", f"{ev['ev_net']:+,.0f}円",
                              delta=f"ROI {ev['roi']*100:+.1f}%")
                if _auto_reason:
                    st.caption(f"🤖 配当自動判定: {_auto_reason} → {_mode_label}")
                st.markdown("#### 5レース横断スコア")
                _sc1, _sc2, _sc3, _sc4 = st.columns(4)
                _sc1.metric("堅いレース", f"{conf['easy_count']}/5",
                            help="本命確率35%以上 or 平均+5pt以上のレース数")
                _sc2.metric("混戦レース", f"{conf['tough_count']}/5",
                            help="本命確率22%未満 or top2との差<3pt or 平均-5pt以下")
                _sc3.metric("最難レース1着率", f"{conf['top1_min']*100:.1f}%")
                _sc4.metric("難易度", f"{conf['complexity']*100:.0f}%",
                            help="0%=超堅い / 100%=超混戦")

                # 各レースのラベル
                _per = conf.get("per_race", [])
                if _per:
                    _label_str = " ".join(
                        f"R{i+1}:{r['label']}({r['top1']*100:.0f}%)"
                        for i, r in enumerate(_per)
                    )
                    st.caption(f"📊 {_label_str}")

                # 分散ガイド
                if conf["tough_count"] >= 2 and _w5_pts < 4:
                    st.warning(
                        f"⚠️ 混戦レースが{conf['tough_count']}本ありますが、現在の最大点数{_w5_pts}点では"
                        f"1レースに集中させる配分しか取れません。**点数を{conf['tough_count']*2}点に上げると複数混戦に分散できます**。"
                    )
                elif conf["easy_count"] == 5 and _w5_pts > 1:
                    st.info("✅ 全レースが堅い見込みです。1点（本命のみ）でも十分かも。")
                st.markdown("#### 軸馬")
                for i, horses in enumerate(result["selections"], 1):
                    suf = f"（{len(horses)}頭軸）" if len(horses) > 1 else ""
                    probs_dict = {n: p for n, p in races_data[i - 1]}
                    horse_str = " / ".join(
                        f"{h}({probs_dict.get(h, 0)*100:.1f}%)" for h in horses
                    )
                    st.write(f"**R{i}**{suf}: {horse_str}")

                with st.expander("各レース上位5頭の1着確率"):
                    for i, race in enumerate(races_data, 1):
                        top5 = sorted(race, key=lambda x: -x[1])[:5]
                        st.write(f"**R{i}**: " + " / ".join(f"{n}({p*100:.1f}%)" for n, p in top5))
            else:
                st.error("5レース全ての確率取得に失敗しました")
