"""
共通ヘルパー（CLEAN-3）

各モジュールで重複定義していた `_safe_int`, `_safe_float`, `_to_seconds` 等を集約。
新規モジュールは `from utils import safe_float, safe_int, to_seconds` を推奨。
"""
import numpy as np
import pandas as pd


def safe_int(val, default: int = 0) -> int:
    """NaN / None / 文字列を含む値を int に変換。失敗時は default。"""
    if val is None:
        return default
    try:
        v = pd.to_numeric(val, errors="coerce")
        if pd.isna(v):
            return default
        return int(v)
    except (ValueError, TypeError):
        return default


def safe_float(val, default: float = 0.0) -> float:
    """NaN / None / 文字列を含む値を float に変換。失敗時は default。"""
    if val is None:
        return default
    try:
        v = pd.to_numeric(val, errors="coerce")
        if pd.isna(v):
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


def safe_str(val, default: str = "") -> str:
    """None / NaN を空文字に。"""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def to_seconds(ft_raw) -> float | None:
    """
    TFJV finish_time（MSST 形式：1510 = 1分51秒0）を秒に変換。
    既に秒の場合（< 1000 = 1分未満ありえない値域）はそのまま返す。
    """
    if ft_raw is None:
        return None
    try:
        ft = float(pd.to_numeric(ft_raw, errors="coerce"))
        if pd.isna(ft) or ft <= 0:
            return None
        # 第29波修正: 旧閾値 1000 は 3桁 MSST（594 = 59.4秒 = 千直等の
        # 60秒未満レース）を「594秒」と誤解釈 → 1000m戦 14,123 行が10倍値で
        # 汚染されていた。実在の走破タイムは最長でも約285秒（AJ大障害）なので
        # 300 超は必ず MSST と断定できる。
        if ft > 300:
            m = int(ft // 1000)
            s = int((ft % 1000) // 10)
            t = int(ft % 10)
            return m * 60.0 + s + t / 10.0
        return ft
    except (ValueError, TypeError):
        return None


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """ゼロ除算を安全にハンドル。"""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        result = numerator / denominator
        if pd.isna(result) or np.isinf(result):
            return default
        return float(result)
    except (ValueError, TypeError, ZeroDivisionError):
        return default


def first_or_default(df: pd.DataFrame, default=None):
    """空チェック付き .iloc[0] — DataFrame でも Series でも安全に取れる。"""
    if df is None or len(df) == 0:
        return default
    return df.iloc[0]


def last_or_default(df: pd.DataFrame, default=None):
    """空チェック付き .iloc[-1]"""
    if df is None or len(df) == 0:
        return default
    return df.iloc[-1]
