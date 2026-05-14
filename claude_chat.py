"""
Claude API 対話機能モジュール。

設計:
- 現在のレース分析データをシステムプロンプトに注入
- ユーザーが「この馬を軸にしたい」「この買い目どう思う？」と相談できる
- ブレイン役として、ロマン爆死リスクには冷静に警告する
- ストリーミングで返答（リアルタイムに文字が流れる）

必要な準備:
1. pip install anthropic
2. .streamlit/secrets.toml に ANTHROPIC_API_KEY = "sk-ant-..." を追加
"""
import pandas as pd
import numpy as np
import streamlit as st
from knowledge_base import get_claude_context_for_all_horses, get_race_pattern_info

SYSTEM_PROMPT_TEMPLATE = """
あなたは競馬予想の分析サポートAIです。
ユーザーの競馬スタイルと特性：
- 2年間で70〜80万円負け。穴馬狙いが多い。
- 穴馬の選定は合っていることが多いが、人気馬の相手選びで失敗することが多い。
- 叩き1走目（休養明け初戦）は買わない方針。
- 過剰な1番人気は嫌う傾向あり。
- 芝↔ダート変わりはむしろ面白いと思って買う。
- 3連複（穴馬軸×人気馬×流し）と馬連がメイン、3連単は2〜3点に絞る。
- 衝動買いしやすく、特にハンデ戦の土曜平場に注意が必要。

【役割】
- データに基づく客観的な意見を述べる
- ロマン爆死リスクには冷静に警告する（ただし根拠があれば許容する）
- 人気馬の相手選びを積極的にサポートする（ユーザーの最大の悩み）
- 最終判断はユーザーに委ねる
- 回答は日本語で簡潔かつ具体的に（スコア・EV・オッズを必ず引用）

【現在のレース情報】
{race_info}

【出走馬の総合信頼スコア一覧】
{horse_scores}

【展開・バイアス情報】
{context_info}

【ユーザーの競馬メモ（ナレッジベース）】
{kb_context}

【注意】
- 「絶対当たる」「確実に」という表現は使わない
- 期待値がマイナスの買い目には必ず数値付きで警告する
- 週予算{budget}円以内に収まるよう資金管理もアドバイスする
"""

INTRO_MESSAGE = (
    "こんにちは！現在のレース分析データを読み込みました。\n\n"
    "「馬Aを軸に3連複を買いたいんだけどどう思う？」\n"
    "「10番人気のBは買いに値する？」\n"
    "「おすすめの買い目を教えて」\n\n"
    "などを気軽に聞いてください。データに基づいて一緒に考えます。"
)


def build_system_prompt(
    eval_df: pd.DataFrame,
    pace_info: dict,
    bias_type: str,
    surface: str,
    distance: int,
    venue: str,
    budget: int,
    race_name: str = "",
) -> str:
    """レース分析データをシステムプロンプトに注入する。"""

    # レース基本情報
    race_info = f"{venue} {surface} {distance}m"
    if race_name:
        race_info = f"{race_name} / {race_info}"

    # 出走馬スコア（上位10頭）
    if not eval_df.empty:
        cols = [c for c in ['horse_name', 'popularity', 'odds', 'confidence_score',
                             'confidence_label', 'ev', 'running_style', 'romance_danger']
                if c in eval_df.columns]
        top_df = eval_df[cols].head(10)
        horse_scores = top_df.fillna('').to_string(index=False)
    else:
        horse_scores = "（分析データなし）"

    # 展開・バイアス
    predicted_pace = pace_info.get('predicted_pace', '不明') if pace_info else '不明'
    pace_summary = pace_info.get('summary', '') if pace_info else ''
    context_info = (
        f"展開予測: {predicted_pace} ({pace_summary})\n"
        f"馬場バイアス: {bias_type}"
    )

    # ナレッジベースコンテキスト
    kb_context = ""
    if not eval_df.empty:
        horses_list = eval_df.to_dict("records")
        for h in horses_list:
            h.setdefault("venue", venue)
            h.setdefault("surface", surface)
            h.setdefault("distance", distance)
            h.setdefault("race_name", race_name)
        try:
            kb_context = get_claude_context_for_all_horses(horses_list, race_name)
        except Exception:
            kb_context = ""

    if not kb_context:
        kb_context = "（この出走馬に関するメモなし）"

    return SYSTEM_PROMPT_TEMPLATE.format(
        race_info=race_info,
        horse_scores=horse_scores,
        context_info=context_info,
        budget=budget,
        kb_context=kb_context,
    )


def stream_chat_response(
    messages: list[dict],
    system_prompt: str,
) -> None:
    """
    Claude APIにストリーミングリクエストを送り、
    st.write_stream でリアルタイムに表示する。
    """
    try:
        import anthropic
    except ImportError:
        st.error("anthropic パッケージが未インストールです。`pip install anthropic` を実行してください。")
        return

    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error(
            "APIキーが設定されていません。\n\n"
            "`.streamlit/secrets.toml` に以下を追加してください:\n"
            "```\nANTHROPIC_API_KEY = 'sk-ant-...'\n```"
        )
        return

    client = anthropic.Anthropic(api_key=api_key)

    # anthropic の messages 形式に変換（roleは 'user' or 'assistant' のみ）
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]

    def _generator():
        with client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=1200,
            system=system_prompt,
            messages=api_messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    # Streamlit のストリーミング表示
    with st.chat_message("assistant"):
        response_text = st.write_stream(_generator())

    # メッセージ履歴に追加
    st.session_state["chat_messages"].append(
        {"role": "assistant", "content": response_text}
    )


def render_chat_tab(
    eval_df: pd.DataFrame,
    pace_info: dict,
    bias_type: str,
    surface: str,
    distance: int,
    venue: str,
    budget: int,
    race_name: str = "",
) -> None:
    """
    Streamlit のチャットタブをレンダリングする。
    app.py から呼び出す。
    """
    st.subheader("🤖 予想ブレインに相談")
    st.caption("レース分析データを踏まえてAIが一緒に考えます")

    # システムプロンプト構築
    system_prompt = build_system_prompt(
        eval_df, pace_info, bias_type, surface, distance, venue, budget, race_name
    )

    # メッセージ履歴の初期化
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "chat_initialized" not in st.session_state:
        st.session_state["chat_initialized"] = False

    # 初回メッセージ
    if not st.session_state["chat_initialized"] and not eval_df.empty:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": INTRO_MESSAGE}
        ]
        st.session_state["chat_initialized"] = True

    # 分析未実行の場合
    if eval_df.empty:
        st.info("先に「レース詳細分析」タブで分析を実行してから相談してください。")
        return

    # 会話履歴の表示
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # シーン別クイック質問
    st.write("**よく使う質問（タップで即送信）:**")

    scene = st.radio(
        "シーンを選ぶ",
        ["🎯 人気馬の相手選び", "💰 買い目の最終確認", "📊 レース後の振り返り"],
        horizontal=True,
        key="chat_scene",
    )

    if scene == "🎯 人気馬の相手選び":
        quick_questions = [
            "穴馬の軸は決まった。人気馬の相手として誰がいい？",
            "1〜3番人気の中で今回の展開に最も合うのは？",
            "人気馬をあえて嫌う理由はある？",
        ]
    elif scene == "💰 買い目の最終確認":
        quick_questions = [
            "今の推奨買い目で問題ない？ブレイン視点で意見を聞かせて",
            "予算5000円でこの馬券構成はどう思う？",
            "3連単は何点に絞るべき？",
        ]
    else:  # レース後
        quick_questions = [
            "今回なぜ外れたか一緒に考えて",
            "選ばなかったあの馬はなぜ来たの？",
            "次回に活かせる教訓を教えて",
        ]

    cols = st.columns(3)
    for i, (col, q) in enumerate(zip(cols, quick_questions)):
        if col.button(q, key=f"quick_{i}_{scene[:2]}"):
            st.session_state["chat_messages"].append({"role": "user", "content": q})
            st.rerun()

    # チャット入力
    if user_input := st.chat_input("例：馬Aを軸に3連複を買いたいんだけどどう思う？"):
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        stream_chat_response(st.session_state["chat_messages"], system_prompt)

    # 会話リセット
    if st.button("🔄 会話をリセット", key="chat_reset"):
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": INTRO_MESSAGE}
        ]
        st.rerun()
