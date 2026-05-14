"""
Discord Webhook 通知モジュール。

セットアップ手順（1回だけ）:
1. Discordでサーバー or DMのチャンネルを開く
2. チャンネル設定 → 連携サービス → ウェブフック → 新しいウェブフック
3. URLをコピーして .streamlit/secrets.toml に貼り付け:
   DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/..."

通知するイベント:
- オッズ急落・急騰アラート（20%超の変化）
- レース分析完了（65点以上の買い推奨馬が出た時）
- 週次ROIレポート（手動送信ボタン）
- 当日の推奨買い目まとめ
"""
import requests
import streamlit as st
from datetime import datetime


def _get_webhook_url() -> str:
    """secrets.toml から Webhook URL を取得する。"""
    try:
        return st.secrets.get("DISCORD_WEBHOOK_URL", "")
    except Exception:
        return ""


def send_discord(
    content: str = "",
    embeds: list[dict] | None = None,
    username: str = "競馬AI",
) -> bool:
    """
    Discord Webhook にメッセージを送信する。

    Parameters
    ----------
    content  : str           プレーンテキスト部分
    embeds   : list[dict]    Embed（カード）リスト（最大10件）
    username : str           Botの表示名

    Returns
    -------
    bool: 送信成功かどうか
    """
    url = _get_webhook_url()
    if not url:
        return False

    payload: dict = {"username": username}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds[:10]

    try:
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False


# ============================================================
# 通知テンプレート
# ============================================================

def notify_odds_alert(signals: list[dict], race_name: str = "") -> bool:
    """
    オッズ急変シグナルをDiscordに通知する。
    """
    if not signals:
        return False

    embeds = []
    for sig in signals[:5]:  # 最大5件
        color = 0xFF0000 if sig["type"] == "急落" else 0x00BFFF  # 赤/青
        embeds.append({
            "title": f"{sig['emoji']} オッズ{sig['type']}アラート",
            "description": sig["message"],
            "color": color,
            "fields": [
                {"name": "レース", "value": race_name or "不明", "inline": True},
                {"name": "変化率", "value": f"{sig['change_pct']:+.1f}%", "inline": True},
                {"name": "時刻", "value": datetime.now().strftime("%H:%M"), "inline": True},
            ],
            "footer": {"text": "競馬AI オッズモニター"},
        })

    content = f"🚨 **{race_name}** でオッズ急変を検出しました！"
    return send_discord(content=content, embeds=embeds)


def notify_analysis_complete(
    race_name: str,
    venue: str,
    surface: str,
    distance: int,
    top_horses: list[dict],  # [{"name": str, "score": int, "ev": float, "odds": float}, ...]
    budget: int = 5000,
) -> bool:
    """
    レース分析完了・推奨馬をDiscordに通知する。
    65点以上の馬がいる場合のみ送信。
    """
    buy_horses = [h for h in top_horses if h.get("score", 0) >= 65]
    if not buy_horses:
        return False

    lines = []
    for h in buy_horses[:5]:
        ev_str = f"EV{h['ev']:+.3f}" if h.get("ev") is not None else ""
        lines.append(
            f"**{h['name']}** — スコア{h['score']}点 / {h.get('odds', '?')}倍 / {ev_str} / {h.get('label', '')}"
        )

    embed = {
        "title": f"📋 分析完了: {race_name}",
        "description": "\n".join(lines),
        "color": 0x00C851,  # 緑
        "fields": [
            {"name": "コース", "value": f"{venue} {surface} {distance}m", "inline": True},
            {"name": "推奨馬数", "value": f"{len(buy_horses)}頭（65点以上）", "inline": True},
            {"name": "予算", "value": f"{budget:,}円", "inline": True},
        ],
        "footer": {"text": f"分析時刻: {datetime.now().strftime('%m/%d %H:%M')}"},
    }

    return send_discord(
        content=f"🐎 **{race_name}** の分析が完了しました。推奨馬 {len(buy_horses)}頭あり！",
        embeds=[embed],
    )


def notify_bet_plan(
    race_name: str,
    tickets: list[dict],  # [{"bet_type": str, "horses": list, "amount": int, "ev": float}, ...]
    total_cost: int,
) -> bool:
    """
    買い目プランをDiscordに通知する。
    """
    if not tickets:
        return False

    lines = []
    for t in tickets:
        horses_str = " - ".join(t.get("horses", []))
        ev_str = f"（EV{t['ev']:+.3f}）" if t.get("ev") is not None else ""
        lines.append(f"**{t['bet_type']}** {horses_str} {t['amount']:,}円 {ev_str}")

    embed = {
        "title": f"🎫 買い目プラン: {race_name}",
        "description": "\n".join(lines),
        "color": 0xFFAA00,  # オレンジ
        "fields": [
            {"name": "合計", "value": f"{total_cost:,}円", "inline": True},
        ],
        "footer": {"text": datetime.now().strftime("%m/%d %H:%M")},
    }

    return send_discord(
        content=f"💰 **{race_name}** の買い目を確定しました",
        embeds=[embed],
    )


def notify_weekly_report(
    weekly_df,  # pd.DataFrame: week, invested, returned, roi
    overall_roi: float,
    total_invested: int,
    total_returned: int,
) -> bool:
    """
    週次パフォーマンスレポートをDiscordに送信する。
    """
    import pandas as pd

    roi_emoji = "✅" if overall_roi >= 100 else "❌"
    roi_color = 0x00C851 if overall_roi >= 100 else 0xFF4444

    # 最近4週の行を文字列化
    lines = []
    if not weekly_df.empty:
        for _, row in weekly_df.tail(4).iterrows():
            roi_val = row.get("roi", 0)
            mark = "✅" if roi_val >= 100 else "❌"
            lines.append(
                f"{mark} **{row['week']}** — 投資{int(row['invested']):,}円 / 回収{int(row['returned']):,}円 / 回収率{roi_val:.0f}%"
            )

    embed = {
        "title": f"{roi_emoji} 週次ROIレポート",
        "description": "\n".join(lines) if lines else "データなし",
        "color": roi_color,
        "fields": [
            {"name": "累計投資", "value": f"{total_invested:,}円", "inline": True},
            {"name": "累計回収", "value": f"{total_returned:,}円", "inline": True},
            {"name": "通算回収率", "value": f"{overall_roi:.1f}%", "inline": True},
        ],
        "footer": {"text": f"送信: {datetime.now().strftime('%m/%d %H:%M')}"},
    }

    return send_discord(
        content=f"📊 週次レポート（通算回収率 **{overall_roi:.1f}%**）",
        embeds=[embed],
    )


def notify_post_race_analysis(
    race_name: str,
    analysis_text: str,
) -> bool:
    """
    Claude のレース後振り返り分析をDiscordに送信する。
    """
    # 長すぎる場合は先頭500文字に切る
    short_text = analysis_text[:500] + "…（続きはアプリで確認）" if len(analysis_text) > 500 else analysis_text

    embed = {
        "title": f"🤖 振り返り分析: {race_name}",
        "description": short_text,
        "color": 0x9C27B0,  # 紫
        "footer": {"text": datetime.now().strftime("%m/%d %H:%M")},
    }

    return send_discord(
        content=f"📝 **{race_name}** の振り返り分析が完了しました",
        embeds=[embed],
    )


# ============================================================
# セットアップUI（app.py から呼ぶ）
# ============================================================

def render_discord_setup_section() -> None:
    """
    サイドバーまたは設定画面で Discord Webhook を設定・テストするUI。
    """
    st.markdown("### 🔔 Discord通知設定")
    webhook_url = _get_webhook_url()

    if webhook_url:
        st.success("✅ Discord Webhook が設定されています")
        if st.button("📡 テスト通知を送信", key="discord_test"):
            ok = send_discord(
                content="🐎 競馬AI からテスト通知です！設定が正常に完了しています。",
                embeds=[{
                    "title": "✅ 通知テスト成功",
                    "description": "このメッセージが届いていれば、オッズ急落アラートや週次レポートが届くようになります。",
                    "color": 0x00C851,
                    "footer": {"text": datetime.now().strftime("%Y/%m/%d %H:%M")},
                }],
            )
            if ok:
                st.success("送信成功！Discordを確認してください")
            else:
                st.error("送信失敗。Webhook URLを確認してください")
    else:
        st.warning("Discord Webhook URLが未設定です")
        with st.expander("⚙️ 設定手順を見る"):
            st.markdown("""
**手順（3分で完了）:**

1. Discordで通知を受け取りたいサーバーの **チャンネル** を右クリック
   → 「チャンネルの編集」→「連携サービス」→「ウェブフックを作成」

   *スマホ1台だけの場合: 自分にDMを送る専用サーバーを作ると便利*

2. 「ウェブフックURLをコピー」をクリック

3. このアプリの `.streamlit/secrets.toml` に以下を追加して再起動:
```toml
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/xxxxxx/xxxxxx"
```

4. このページで「テスト通知を送信」を押して確認
            """)
