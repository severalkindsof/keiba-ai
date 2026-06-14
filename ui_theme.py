"""
UIテーマ・共通コンポーネント
コンセプト (第14波〜): アングラ×小粋ダークテーマ
    ニアブラック基調 + 明朝（Noto Serif JP）+ 緋色アクセント + 金の差し色
    「夜の馬券師の書斎」— コーポレートの品格と裏路地の艶を両立
"""
from pathlib import Path
import streamlit as st


# ============================================================
# 色定数（ダーク・アンダーグラウンド）
# ============================================================

COLOR = {
    "primary":     "#E0633F",   # 緋色テラコッタ（ダーク上で映える輝度）
    "gold":        "#C9A35C",   # 勝負の金 — 第二アクセント
    "buy":         "#6FA77F",   # 夜の苔緑
    "avoid":       "#D9604F",   # 警告の朱
    "warn":        "#C9A35C",   # 金 = 注意
    "muted":       "#B8AFA0",   # 明るめ灰墨（ダーク上でも読める）
    "border":      "#423B32",   # 視認できる罫線
    "card_bg":     "#2B2622",   # カード（背景より一段浮く）
    "bg":          "#211D19",   # 暖色寄りニアブラック
    "sidebar_bg":  "#272320",
    "text":        "#F2EDE2",   # アイボリーホワイト
}

# Plotly / Altair で使う統一パレット（ダーク用に輝度調整）
CHART_COLORS = ["#E0633F", "#6FA77F", "#C9A35C", "#D9604F", "#B8AFA0", "#7A8CA3"]
CHART_SCALE_GOOD_TO_BAD = ["#6FA77F", "#C9A35C", "#D9604F"]

_CSS_PATH = Path(__file__).parent / "assets" / "style.css"


# ============================================================
# 初期化
# ============================================================

def load_theme():
    """
    Streamlit ページ設定の直後に1度だけ呼ぶ。
    .streamlit/config.toml がテーマ本体、本関数は微調整CSSを注入。
    """
    if _CSS_PATH.exists():
        st.markdown(
            f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


# ============================================================
# ヘルパー
# ============================================================

def banner(level: str, text: str):
    """
    単色トーン統一バナー。
        level = 'info'|'buy'|'avoid'|'warn'|'muted'
    左罫線3pxアクセント + 軽い枠。本文HTMLそのまま渡せる。
    """
    color_map = {
        "info":  COLOR["primary"],
        "buy":   COLOR["buy"],
        "avoid": COLOR["avoid"],
        "warn":  COLOR["warn"],
        "muted": COLOR["muted"],
    }
    color = color_map.get(level, COLOR["muted"])
    st.markdown(
        f'<div style="border-left:3px solid {color};'
        f'background:{COLOR["card_bg"]};padding:12px 16px;'
        f'border-radius:8px;margin:8px 0;'
        f'border:1px solid {COLOR["border"]};'
        f'color:{COLOR["text"]};font-size:14px;line-height:1.6;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def pill(text: str, color: str = "mute") -> str:
    """
    ピル型タグ（HTML文字列を返す。`st.markdown(..., unsafe_allow_html=True)` で表示）。
        color = 'accent'|'buy'|'warn'|'mute'|'avoid'
    """
    return f'<span class="pill pill-{color}">{text}</span>'


def hero(title: str, caption: str | None = None):
    """
    コーポレートサイト風ヒーロー: 大きなセリフ見出し + hairline + リード文。
    """
    hairline = f'<div style="width:48px;height:1px;background:{COLOR["primary"]};opacity:0.7;margin:0.4em 0 0.7em 0;"></div>'
    cap_html = f'<div style="color:{COLOR["muted"]};font-size:14px;line-height:1.6;">{caption}</div>' if caption else ""
    st.markdown(
        f'<div style="margin-top:0.2em;margin-bottom:1.2em;">'
        f'<h1 style="font-family:\'Noto Serif JP\',serif;font-weight:500;font-size:32px;'
        f'letter-spacing:-0.02em;color:{COLOR["text"]};margin:0;line-height:1.3;">'
        f'{title}</h1>'
        f'{hairline}{cap_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def apply_chart_theme(fig, *, title: str | None = None):
    """
    Plotly figure に共通テーマを適用。
    """
    layout = {
        "template": "simple_white",
        "font": {
            "family": "Noto Serif JP, Yu Mincho, serif",
            "color":  COLOR["text"],
            "size":   13,
        },
        "paper_bgcolor": COLOR["card_bg"],
        "plot_bgcolor":  COLOR["card_bg"],
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
        "colorway": CHART_COLORS,
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"size": 16, "color": COLOR["text"],
                     "family": "Noto Serif JP, serif"},
        }
    fig.update_layout(**layout)
    return fig
