"""
前日分析→当日引継ぎ用のセッション保存/読込モジュール。
分析結果をJSONファイルに保存して、翌日のアプリ起動時に再ロードできる。
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

SAVE_DIR = Path(__file__).parent / "saved_sessions"
SAVE_DIR.mkdir(exist_ok=True)


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return super().default(obj)


def save_session(
    race_name: str,
    entries: list[dict],
    eval_df: pd.DataFrame,
    surface: str,
    distance: int,
    venue: str,
    pace_info: dict,
) -> str:
    """
    分析結果をJSONファイルに保存する。
    ファイル名: YYYYMMDD_HHMM_レース名.json
    Returns: 保存したファイルパス
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_name = race_name.replace("/", "").replace(" ", "_")[:20]
    filename = SAVE_DIR / f"{timestamp}_{safe_name}.json"

    # factor_breakdown など dict型の列を文字列に変換してからシリアライズ
    eval_records = []
    for _, row in eval_df.iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, dict):
                rec[k] = v  # dict はそのまま保持
            elif isinstance(v, (bool, np.bool_)):
                rec[k] = bool(v)
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                rec[k] = None
            else:
                rec[k] = v
        eval_records.append(rec)

    payload = {
        "saved_at": datetime.now().isoformat(),
        "race_name": race_name,
        "surface": surface,
        "distance": distance,
        "venue": venue,
        "pace_info": pace_info,
        "entries": entries,
        "eval_records": eval_records,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=_Encoder)

    return str(filename)


SCAN_CACHE_FILE = Path(__file__).parent / "saved_sessions" / "_scan_cache.json"


def save_scan_result(scan_df: pd.DataFrame, dates: list[str]) -> None:
    """週末スキャン結果を永続化する"""
    if scan_df.empty:
        return
    SAVE_DIR.mkdir(exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(),
        "dates": dates,
        "records": scan_df.to_dict(orient="records"),
    }
    with open(SCAN_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=_Encoder)


def load_scan_result() -> tuple[pd.DataFrame, str, list[str]]:
    """
    保存済みスキャン結果を読み込む。
    Returns: (DataFrame, 保存日時文字列, 対象日リスト)
    """
    if not SCAN_CACHE_FILE.exists():
        return pd.DataFrame(), "", []
    try:
        with open(SCAN_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data.get("records", []))
        saved_at = data.get("saved_at", "")[:16].replace("T", " ")
        dates = data.get("dates", [])
        return df, saved_at, dates
    except Exception:
        return pd.DataFrame(), "", []


def list_saved_sessions() -> list[dict]:
    """保存済みセッション一覧を返す（新しい順）"""
    files = sorted(SAVE_DIR.glob("*.json"), reverse=True)
    result = []
    for f in files[:20]:  # 最新20件
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            result.append({
                "path": str(f),
                "filename": f.name,
                "race_name": data.get("race_name", "?"),
                "saved_at": data.get("saved_at", "?"),
                "surface": data.get("surface", ""),
                "distance": data.get("distance", ""),
                "venue": data.get("venue", ""),
            })
        except Exception:
            continue
    return result


def load_session(filepath: str) -> dict | None:
    """保存済みセッションを読み込む"""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        # eval_records を DataFrame に復元
        eval_df = pd.DataFrame(data.get("eval_records", []))
        return {
            "race_name": data.get("race_name", ""),
            "surface": data.get("surface", "芝"),
            "distance": data.get("distance", 2000),
            "venue": data.get("venue", "東京"),
            "pace_info": data.get("pace_info", {}),
            "entries": data.get("entries", []),
            "eval_df": eval_df,
        }
    except Exception as e:
        return None
