"""
TFJV調教データ（training_XXX.csv）の読み込みと評価モジュール。

CSVフォーマット（18列）:
  場所, 日付(YYYYMMDD), 曜日, 時刻, 馬名, (空), コース, 年齢,
  収得賞金, 騎乗者, Time1(4F全体), Time2(3F), Time3(2F), Time4(1F),
  Lap4, Lap3, Lap2, Lap1(最終1F)

評価方針:
  - タイムの絶対値は馬場・距離が違うと比較できないため「同馬の直近平均との乖離」で評価
  - レース前5日以内の最速セッション = 最終追い切り
  - Lap1（最終1F）の速さ = 末脚の鋭さ
  - 直近3本のTime1トレンド = 仕上がり上昇/下降
"""
import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


COLS = [
    "location", "date_str", "weekday", "time_str", "horse_name", "empty",
    "course", "age", "prize_money", "rider",
    "time1", "time2", "time3", "time4",
    "lap4", "lap3", "lap2", "lap1",
]


def load_tfjv_training(csv_path: str | Path) -> dict[str, list[dict]]:
    """
    TFJV調教CSVを読み込み、馬名→セッションリストのdictを返す。
    各セッション dict:
        date, time1, time4(=last1F), lap1, course, rider
    """
    path = Path(csv_path)
    if not path.exists():
        return {}

    sessions: dict[str, list[dict]] = {}
    with open(path, encoding="cp932") as f:
        for row in csv.reader(f):
            if len(row) < 18:
                continue
            # ヘッダ行スキップ
            if row[0].strip() in ("場所", "location", ""):
                continue
            try:
                horse = row[4].strip()
                if not horse:
                    continue
                date_str = row[1].strip()  # YYYYMMDD
                date = datetime.strptime(date_str, "%Y%m%d").date() if len(date_str) == 8 else None
                t1 = float(row[10]) if row[10].strip() else None   # 全体タイム
                t4 = float(row[13]) if row[13].strip() else None   # 最終1F
                l1 = float(row[17]) if row[17].strip() else None   # Lap1（=最終1F）
                if date is None or t1 is None:
                    continue
                sessions.setdefault(horse, []).append({
                    "date":     date,
                    "time1":    t1,
                    "last_1f":  l1 or t4,
                    "course":   row[6].strip(),
                    "rider":    row[9].strip(),
                })
            except (ValueError, IndexError):
                continue

    # 日付昇順ソート
    for horse in sessions:
        sessions[horse].sort(key=lambda x: x["date"])

    return sessions


def evaluate_training_tfjv(
    horse_name: str,
    sessions: list[dict],
    race_date_str: str = "",
) -> dict:
    """
    1頭分のセッションリストから調教評価を算出する。

    Parameters
    ----------
    horse_name  : 馬名
    sessions    : load_tfjv_training() が返す1頭分のリスト（日付昇順）
    race_date_str : レース日（YYYY-MM-DD）。省略時は最新セッションから7日後と仮定。

    Returns
    -------
    {
        "label"  : str   ("調教◎ 好仕上がり" / "調教○ 普通以上" / "調教△ 普通" / "調教▲ 低調"),
        "bonus"  : float (confluence_scoreへの加点: -3〜+5),
        "detail" : str   (詳細説明),
        "score"  : float (0〜1の内部スコア),
        "last_time1"  : float | None,
        "last_lap1"   : float | None,
        "trend"  : str   ("上昇" / "横ばい" / "下降"),
        "sessions_count": int,
    }
    """
    empty = {
        "label": "調教データなし", "bonus": 0.0, "detail": "",
        "score": 0.5, "last_time1": None, "last_lap1": None,
        "trend": "不明", "sessions_count": 0,
    }

    if not sessions:
        return empty

    # レース日付
    if race_date_str:
        try:
            race_date = datetime.strptime(race_date_str, "%Y-%m-%d").date()
        except ValueError:
            race_date = sessions[-1]["date"] + timedelta(days=7)
    else:
        race_date = sessions[-1]["date"] + timedelta(days=7)

    # レース7日前〜前日のセッション = 最終追い切り候補
    cutoff = race_date - timedelta(days=7)
    final_sessions = [s for s in sessions if s["date"] >= cutoff and s["date"] < race_date]
    all_recent = sessions[-min(10, len(sessions)):]  # 直近10本

    if not all_recent:
        return empty

    # ---- 1. 最終追い切りの速さ（同馬の直近平均との差）----
    all_time1 = [s["time1"] for s in all_recent if s["time1"] is not None]
    if not all_time1:
        return empty

    avg_t1 = sum(all_time1) / len(all_time1)
    std_t1 = (sum((x - avg_t1) ** 2 for x in all_time1) / len(all_time1)) ** 0.5 if len(all_time1) > 1 else 1.0

    # 最終追い切り = 最終週の最速セッション（なければ最新全体セッション）
    if final_sessions:
        best_final = min(final_sessions, key=lambda s: s["time1"] or 999)
    else:
        best_final = all_recent[-1]

    last_t1   = best_final["time1"]
    last_lap1 = best_final.get("last_1f")

    # z-score: マイナス = 速い（平均より速い）
    z = (last_t1 - avg_t1) / (std_t1 + 0.1)

    # ---- 2. ラップ加速度（最終1Fの速さ）----
    all_lap1 = [s["last_1f"] for s in all_recent if s.get("last_1f") is not None]
    avg_lap1 = sum(all_lap1) / len(all_lap1) if all_lap1 else None
    lap1_score = 0.0
    if avg_lap1 and last_lap1:
        lap1_diff = avg_lap1 - last_lap1  # プラス = 最終1Fが速くなった
        lap1_score = min(0.5, max(-0.5, lap1_diff / 1.0))

    # ---- 3. トレンド（直近3本 vs それ以前）----
    trend = "横ばい"
    trend_score = 0.0
    if len(all_recent) >= 4:
        recent3  = [s["time1"] for s in all_recent[-3:] if s["time1"]]
        older    = [s["time1"] for s in all_recent[:-3]  if s["time1"]]
        if recent3 and older:
            avg_recent = sum(recent3) / len(recent3)
            avg_older  = sum(older)   / len(older)
            delta = avg_older - avg_recent  # プラス = 直近が速くなった
            if delta > 1.5:
                trend, trend_score = "上昇", 0.3
            elif delta > 0.5:
                trend, trend_score = "やや上昇", 0.1
            elif delta < -1.5:
                trend, trend_score = "下降", -0.3
            elif delta < -0.5:
                trend, trend_score = "やや下降", -0.1

    # ---- 総合スコア（0〜1）----
    # z-score変換: z=-2(超速) → 1.0, z=0(平均) → 0.5, z=+2(遅い) → 0.0
    speed_score = max(0.0, min(1.0, 0.5 - z * 0.25))
    score = speed_score * 0.6 + (0.5 + lap1_score) * 0.25 + (0.5 + trend_score) * 0.15

    # ---- ラベルとボーナス ----
    # セッション数が少ない場合はニュートラルに（外れ値に引きずられやすい）
    if len(sessions) < 3:
        label, bonus = "調教△（普通）", 0
    elif score >= 0.75:
        label, bonus = "調教◎（好仕上がり）", +5
    elif score >= 0.60:
        label, bonus = "調教○（普通以上）", +2
    elif score >= 0.45:
        label, bonus = "調教△（普通）", 0
    elif score >= 0.35:
        label, bonus = "調教▲（やや低調）", -2
    else:
        label, bonus = "調教×（低調）", -3

    # ---- 詳細テキスト ----
    parts = []
    if last_t1:
        parts.append(f"最終4F={last_t1:.1f}秒")
    if last_lap1:
        parts.append(f"上がり1F={last_lap1:.1f}秒")
    if avg_t1:
        diff = avg_t1 - last_t1
        sign = "+" if diff > 0 else ""
        parts.append(f"自己平均比{sign}{diff:.1f}秒")
    if trend != "横ばい":
        parts.append(f"推移:{trend}")
    if final_sessions:
        parts.append(f"{best_final['date']}追切")

    detail = " / ".join(parts)

    return {
        "label":          label,
        "bonus":          float(bonus),
        "detail":         detail,
        "score":          round(score, 3),
        "last_time1":     last_t1,
        "last_lap1":      last_lap1,
        "trend":          trend,
        "sessions_count": len(sessions),
    }


_CACHE_PATH = Path(__file__).parent / "data" / "tfjv_training_cache.json"


def save_training_cache(results: dict, race_label: str = "") -> None:
    """評価結果をJSONキャッシュに保存（app.pyが自動読み込み）"""
    import json
    _CACHE_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "race_label": race_label,
        "saved_at":   datetime.now().isoformat(),
        "results":    {
            name: {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                   for k, v in data.items()}
            for name, data in results.items()
        },
    }
    _CACHE_PATH.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_training_cache() -> dict:
    """キャッシュから評価結果を読み込む"""
    import json
    if not _CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        return data.get("results", {})
    except Exception:
        return {}


def evaluate_all_horses_tfjv(
    csv_path: str | Path,
    horse_names: list[str],
    race_date_str: str = "",
) -> dict[str, dict]:
    """
    全出走馬の調教評価を一括で返す。

    Returns
    -------
    {馬名: evaluate_training_tfjv() の戻り値, ...}
    """
    sessions_map = load_tfjv_training(csv_path)
    result = {}
    for name in horse_names:
        # 部分一致でもマッチ（空白の違いに対応）
        matched_key = None
        for key in sessions_map:
            if key == name or key.strip() == name.strip():
                matched_key = key
                break
        if matched_key is None:
            # 部分一致フォールバック
            for key in sessions_map:
                if name.strip() in key or key in name.strip():
                    matched_key = key
                    break

        if matched_key:
            result[name] = evaluate_training_tfjv(name, sessions_map[matched_key], race_date_str)
        else:
            result[name] = {
                "label": "調教データなし", "bonus": 0.0, "detail": "",
                "score": 0.5, "last_time1": None, "last_lap1": None,
                "trend": "不明", "sessions_count": 0,
            }
    return result
