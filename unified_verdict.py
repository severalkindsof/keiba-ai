"""第46波: 統一最終判定 — 全評価項目を一本の線でつなぐ。

従来の問題:
- 判定(confidence_label)・市場評価(verdict)・買い判定(buy_label)・推奨理由・非推奨理由が
  それぞれ独立した絶対閾値で動き、矛盾だらけ（推奨なのに見送り等）
- 絶対閾値のため「全馬様子見」のような無差別状態が起きる

新方式（レース内相対評価 + 階層ロジック）:
1. 致命的ナレッジ（kb_avoids）あり → ✕消し
2. 実力レース内TOP3 かつ EV+        → ◎軸（強くて妙味あり）
3. 実力レース内TOP3 かつ EV-        → ▲信頼軸（強いが妙味薄 = 人気かぶり）
4. 複勝EV+ or 単勝EV+（実力中位以上）→ ○妙味（オッズが甘い）
5. 7人気以下 かつ 実力上位50%       → △穴ロマン（爆穴candidates）
6. それ以外                          → ✕消し

根拠は判定と必ず整合する1文のみ生成（推奨に非推奨理由を併記しない）。
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _safe_float(v, default=float("nan")) -> float:
    try:
        f = float(v)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def apply_unified_verdict(eval_df: pd.DataFrame, mode: str = "爆穴") -> pd.DataFrame:
    """eval_df に final_verdict / final_reason 列を追加して返す。

    final_verdict: ◎軸 / ▲信頼軸 / ○妙味 / △穴ロマン / ✕消し
    final_reason:  判定と整合する1文
    """
    if eval_df.empty:
        return eval_df
    df = eval_df.copy()

    n = len(df)
    # レース内ランク（実力スコア。同点は LGBM勝率 → EV でタイブレークし一意化）
    cs = pd.to_numeric(df.get("confidence_score"), errors="coerce").fillna(0)
    ev = pd.to_numeric(df.get("ev"), errors="coerce")
    evp = pd.to_numeric(df.get("place_ev", df.get("ev_place")), errors="coerce")
    pop = pd.to_numeric(df.get("popularity"), errors="coerce").fillna(99)
    lgbm = pd.to_numeric(df.get("lgbm_win_rate"), errors="coerce").fillna(0)
    _rank_key = cs + lgbm * 0.01 + ev.fillna(-1) * 0.001
    cs_rank = _rank_key.rank(method="first", ascending=False)  # 1=最強・一意

    verdicts, reasons = [], []
    for i in df.index:
        _cs = float(cs.loc[i])
        _rank = int(cs_rank.loc[i])
        _ev = ev.loc[i] if i in ev.index else float("nan")
        _evp = evp.loc[i] if (evp is not None and i in evp.index) else float("nan")
        _pop = int(pop.loc[i])
        _kb_avoids = df.at[i, "kb_avoids"] if "kb_avoids" in df.columns else []
        if isinstance(_kb_avoids, str):
            _kb_avoids = [_kb_avoids] if _kb_avoids else []
        _kb_notes = df.at[i, "kb_notes"] if "kb_notes" in df.columns else []
        if isinstance(_kb_notes, str):
            _kb_notes = [_kb_notes] if _kb_notes else []

        _ev_ok = not np.isnan(_ev) and _ev > 0.05
        _evp_ok = not np.isnan(_evp) and _evp > 0.03
        _top3 = _rank <= 3
        _upper_half = _rank <= max(3, int(n * 0.5))

        # 1. 致命的ナレッジ
        if _kb_avoids:
            verdicts.append("✕消し")
            reasons.append(f"致命傷: {_kb_avoids[0]}")
            continue

        _longshot = _pop >= 7  # 人気薄

        # 2. 実力TOP3 × 人気上位 → 信頼軸サイド
        if _top3 and not _longshot:
            if _ev_ok or _evp_ok:
                verdicts.append("◎軸")
                _which = "単勝EV+" if _ev_ok else "複勝EV+"
                _note = f" / {_kb_notes[0]}" if _kb_notes else ""
                reasons.append(f"実力{_rank}位×{_which}（強くて割安）{_note}")
            else:
                verdicts.append("▲信頼軸")
                reasons.append(f"実力{_rank}位の人気サイド（EV妙味は薄いが軸の信頼性は高い）")
            continue

        # 3. 実力TOP3 × 人気薄 → 爆穴の本丸（「信頼軸」とは呼ばない）
        if _top3 and _longshot:
            verdicts.append("△穴ロマン")
            _ev_str = "EV+も付く" if (_ev_ok or _evp_ok) else "EVは市場通り"
            _note = f" / {_kb_notes[0]}" if _kb_notes else ""
            reasons.append(f"{_pop}人気なのに実力{_rank}位/{n}頭（{_ev_str}）{_note}")
            continue

        # 4. 妙味（実力中位以上 × EV+）
        if (_ev_ok or _evp_ok) and _upper_half:
            verdicts.append("○妙味")
            _which = "単勝" if _ev_ok else "複勝"
            reasons.append(f"{_which}オッズが実力比で甘い（実力{_rank}位/{n}頭）")
            continue

        # 5. 穴ロマン: 人気薄 × (EV+ or 実力上位50%)
        if mode == "爆穴" and _longshot and (_ev_ok or _evp_ok):
            verdicts.append("△穴ロマン")
            _which = "単勝" if _ev_ok else "複勝"
            _note = f" / {_kb_notes[0]}" if _kb_notes else ""
            reasons.append(f"{_pop}人気で{_which}EV+（モデルは割安と判断）{_note}")
            continue
        if mode == "爆穴" and _longshot and _upper_half:
            verdicts.append("△穴ロマン")
            _note = f" / {_kb_notes[0]}" if _kb_notes else ""
            reasons.append(f"{_pop}人気だが実力{_rank}位/{n}頭（市場が見落とし疑い）{_note}")
            continue

        # 6. 押さえ: 実力上位50%（消すには惜しい中間層）
        if _upper_half:
            verdicts.append("・押さえ")
            reasons.append(f"実力{_rank}位/{n}頭・妙味は薄いが3連系の紐には残せる")
            continue

        # 7. 消し（下位50% × 妙味なし）
        verdicts.append("✕消し")
        if _pop <= 5:
            reasons.append(f"人気({_pop}人気)に実力({_rank}位/{n}頭)が見合わない")
        else:
            reasons.append(f"実力{_rank}位/{n}頭・妙味なし")

    df["final_verdict"] = verdicts
    df["final_reason"] = reasons
    return df


VERDICT_ORDER = {"◎軸": 0, "▲信頼軸": 1, "○妙味": 2, "△穴ロマン": 3, "・押さえ": 4, "✕消し": 5}


def sort_by_verdict(df: pd.DataFrame) -> pd.DataFrame:
    """final_verdict 優先 → 実力スコア降順でソート。"""
    if "final_verdict" not in df.columns:
        return df
    df = df.copy()
    df["_vo"] = df["final_verdict"].map(VERDICT_ORDER).fillna(9)
    cs = pd.to_numeric(df.get("confidence_score"), errors="coerce").fillna(0)
    df["_cs"] = cs
    out = df.sort_values(["_vo", "_cs"], ascending=[True, False]).drop(columns=["_vo", "_cs"])
    return out
