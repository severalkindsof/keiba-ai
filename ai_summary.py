"""
EFF-3: Claude 自然言語サマリ

分析完了後、上位3頭について「なぜこの評価か」を3行で自動生成。
ユーザーは「数字の羅列」を読まずに済む。

使い方:
    from ai_summary import render_ai_summary
    render_ai_summary(eval_df, pace_info, surface, distance, venue, race_name)
"""
import pandas as pd
import streamlit as st


_SUMMARY_SYSTEM = """\
あなたは競馬予想の解説者です。データに基づいた客観的な短評を提供します。
ユーザーは数値羅列を読みたくないので、3行・各60字以内で簡潔に。
EV・スコア・人気・オッズに必ず触れ、買いの根拠 or 警戒理由を明示してください。
過剰な期待は避け、人気馬の不安要素もフェアに指摘します。"""


def _top_horses_data(eval_df: pd.DataFrame, n: int = 3) -> list[dict]:
    """confidence_score 上位 N 頭の情報を辞書リストで抽出"""
    cols_pref = [
        "horse_name", "popularity", "odds", "confidence_score", "confidence_label",
        "ev", "blended_pct", "draw_label", "running_style",
        "rotation_signal", "jockey_change_signal",
        "ippen_candidate", "excuse_str", "training_label",
        "elo", "pci_avg", "pci_label",
    ]
    avail = [c for c in cols_pref if c in eval_df.columns]
    sort_col = "confidence_score" if "confidence_score" in eval_df.columns else "ev"
    top = eval_df.sort_values(sort_col, ascending=False).head(n)[avail]
    return top.to_dict(orient="records")


def _build_user_prompt(
    horses: list[dict],
    pace_info: dict,
    surface: str,
    distance: int,
    venue: str,
    race_name: str,
) -> str:
    """Claude に渡す user メッセージを組み立てる"""
    pace_str = pace_info.get("predicted_pace", "?") if pace_info else "?"
    race_header = f"{race_name or '今回レース'}（{venue} {surface}{distance}m）／展開予測: {pace_str}"

    lines = [f"## レース：{race_header}", "", "## 上位3頭のデータ"]
    for i, h in enumerate(horses, 1):
        parts = [f"**{i}位 {h.get('horse_name','?')}**"]
        if "popularity" in h:        parts.append(f"{h['popularity']}番人気")
        if "odds" in h:              parts.append(f"単勝{h['odds']}倍")
        if "confidence_score" in h:  parts.append(f"実力{h['confidence_score']}")
        if "ev" in h and h["ev"] is not None: parts.append(f"EV{float(h['ev']):+.2f}")
        if "blended_pct" in h and h["blended_pct"] is not None:
            parts.append(f"Benter勝率{h['blended_pct']:.0f}%")
        if "elo" in h and h["elo"] is not None:
            parts.append(f"Elo{int(h['elo'])}")
        if "pci_label" in h and h["pci_label"] and h["pci_label"] != "データなし":
            parts.append(f"PCI:{h['pci_label']}")
        if "draw_label" in h and h["draw_label"]:
            parts.append(f"枠:{h['draw_label']}")
        if "running_style" in h and h["running_style"] and h["running_style"] != "不明":
            parts.append(f"脚質:{h['running_style']}")
        if "rotation_signal" in h and h["rotation_signal"]:
            parts.append(f"ローテ:{h['rotation_signal']}")
        if "jockey_change_signal" in h and h["jockey_change_signal"]:
            parts.append(f"乗替:{h['jockey_change_signal']}")
        if h.get("ippen_candidate"):
            parts.append("一変候補")
        if "training_label" in h and h["training_label"] and h["training_label"] != "調教未取得":
            parts.append(f"調教:{h['training_label']}")
        if "excuse_str" in h and h["excuse_str"]:
            parts.append(f"前走言い訳:{h['excuse_str']}")
        lines.append(" / ".join(parts))

    lines += [
        "",
        "## 求める出力フォーマット（厳守）",
        "各馬1ブロック、合計3ブロック。各ブロックは見出し1行 + 評価3行（各60字以内）：",
        "",
        "**1. 〇〇**（実力XX / EV±X.XX / X番人気）",
        "・買い根拠: 30〜60字",
        "・警戒点: 30〜60字",
        "・買い判断: 「軸◎」「ヒモ○」「様子見△」「消し✗」のいずれか + 一言",
    ]
    return "\n".join(lines)


def generate_summary(
    eval_df: pd.DataFrame,
    pace_info: dict,
    surface: str,
    distance: int,
    venue: str,
    race_name: str = "",
) -> str | None:
    """
    Claude で上位3頭のサマリを生成。
    APIキーがない場合は None を返す。
    """
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    horses = _top_horses_data(eval_df, n=3)
    if not horses:
        return None

    user_prompt = _build_user_prompt(horses, pace_info, surface, distance, venue, race_name)

    client = anthropic.Anthropic(api_key=api_key)
    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=800,
            thinking={"type": "adaptive"},
            system=_SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            final = stream.get_final_message()
        # final.content は ContentBlock のリスト。テキストブロックだけ抽出
        texts = [b.text for b in final.content if getattr(b, "type", "") == "text"]
        return "".join(texts).strip()
    except Exception as e:
        return f"Claude API エラー: {e}"


def render_ai_summary(
    eval_df: pd.DataFrame,
    pace_info: dict,
    surface: str,
    distance: int,
    venue: str,
    race_name: str = "",
    cache_key: str = "",
):
    """
    Streamlit UI 描画。
    cache_key を変えれば再生成、同じなら session_state からキャッシュ表示。
    """
    if eval_df is None or eval_df.empty:
        return

    state_key = f"_ai_summary_{cache_key}"
    cached = st.session_state.get(state_key)

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("#### Claude による上位3頭の短評")
    with col2:
        if st.button("再生成", key=f"ai_sum_regen_{cache_key}", use_container_width=True):
            cached = None
            st.session_state.pop(state_key, None)

    if cached:
        st.markdown(cached)
        st.caption("※ Claude による生成。最終判断はご自身で。")
        return

    if not st.secrets.get("ANTHROPIC_API_KEY", ""):
        st.info("ANTHROPIC_API_KEY が設定されていないためサマリを生成できません。")
        return

    with st.spinner("Claude が短評を生成中..."):
        summary = generate_summary(eval_df, pace_info, surface, distance, venue, race_name)
    if summary:
        st.session_state[state_key] = summary
        st.markdown(summary)
        st.caption("※ Claude による生成。最終判断はご自身で。")
