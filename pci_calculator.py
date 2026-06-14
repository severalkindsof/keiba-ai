"""
NEW-2: PCI / RPCI（ペースチェンジ指数）

公式（train_lgbm.py の pci_raw と同じスケールに統一）:
    PCI = (前半部の3F換算タイム / 上り3F) × 100

意味:
    - PCI ≈ 100 → 均等ペース（前後半同じ）
    - PCI > 100 → 前半遅め＝後半が速い（差し有利ペース）
    - PCI < 100 → 前半速め＝前残りペース
    - 例: PCI=120（前半スロー→上り爆速）/ PCI=80（前半ハイ→上り失速）

用途:
    - 「前走 PCI が 50 から離れたのに着順が良い」 = ペース崩れの中で実力発揮 → 巻返し候補
    - 「前走 PCI 50 付近で上位入線」 = 純粋に強い

データ源:
    tfjv_all.parquet の finish_time, last_3f, distance から計算可能
"""
import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"


def calc_pci_row(finish_time: float, last_3f: float, distance: int) -> float | None:
    """
    1走の PCI を計算。
    finish_time: TFJV値（0.1秒単位、例: 1510 = 151.0秒）
                 ※ float の場合は自動判定：100以上なら0.1秒単位とみなす
    last_3f: 上り3F秒（例: 35.2）
    distance: 距離m（例: 1800）
    """
    if (finish_time is None or last_3f is None or distance is None
            or pd.isna(finish_time) or pd.isna(last_3f) or pd.isna(distance)
            or distance <= 600 or finish_time <= 0 or last_3f <= 0):
        return None
    try:
        ft_raw = float(finish_time)
        d_check = float(distance)
        # TFJV finish_time は MSST 形式（例: 1510 = 1分51秒0 = 111.0秒）
        # 検出: ft_raw / distance が 実走想定（~0.06秒/m）から大きく外れる場合
        if d_check > 0 and (ft_raw / d_check) > 0.15:
            # MSST 形式と判定: 分桁を分離して秒に変換
            minutes = int(ft_raw // 1000)
            sec_int = int((ft_raw % 1000) // 10)
            tenths  = int(ft_raw % 10)
            ft = minutes * 60.0 + sec_int + tenths / 10.0
        else:
            ft = ft_raw
        l3 = float(last_3f)
        d = float(distance)
        first_part_time = ft - l3      # 前半部のタイム（秒）
        first_part_dist = d - 600.0    # 前半部の距離（m）
        if first_part_dist <= 0 or first_part_time <= 0:
            return None
        # 前半部の3F (600m) 換算タイム
        first_3f_equiv = first_part_time * (600.0 / first_part_dist)
        # train_lgbm.py と同じスケール: 100基準（-50しない）、60-140にクリップ
        pci = (first_3f_equiv / l3) * 100.0
        return float(max(60.0, min(140.0, pci)))
    except Exception:
        return None


def add_pci_to_history(df: pd.DataFrame) -> pd.DataFrame:
    """過去レースDataFrameに pci 列を追加"""
    df = df.copy()
    df["pci"] = df.apply(
        lambda r: calc_pci_row(r.get("finish_time"), r.get("last_3f"), r.get("distance")),
        axis=1,
    )
    return df


def get_horse_pci_stats(df_hist: pd.DataFrame, horse_name: str, n: int = 5) -> dict:
    """
    指定馬の直近 n 走の PCI 統計を返す。

    Returns:
        {
            "pci_avg": 直近平均, "pci_std": ばらつき,
            "pci_latest": 最新走 PCI, "pci_latest_rank": 最新走着順,
            "pace_change_count": 50±15 から外れた走数,
            "label": "ペース変動耐性あり" 等の定性ラベル,
            "bonus": 0.0〜0.05 の補正値,
        }
    """
    if df_hist is None or df_hist.empty or "horse_name" not in df_hist.columns:
        return _empty_pci_result()
    sub = df_hist[df_hist["horse_name"].astype(str).str.strip() == str(horse_name).strip()]
    if sub.empty or "finish_time" not in sub.columns or "last_3f" not in sub.columns:
        return _empty_pci_result()
    # 日付ソート（最新を先頭）
    if "date" in sub.columns:
        sub = sub.sort_values("date", ascending=False).head(n)
    else:
        sub = sub.head(n)

    pcis = []
    ranks = []
    for _, r in sub.iterrows():
        p = calc_pci_row(r.get("finish_time"), r.get("last_3f"), r.get("distance"))
        if p is not None:
            pcis.append(p)
            rk = r.get("rank")
            try:
                ranks.append(int(rk) if rk is not None and not pd.isna(rk) else None)
            except (ValueError, TypeError):
                ranks.append(None)

    if not pcis:
        return _empty_pci_result()

    pci_avg = float(np.mean(pcis))
    pci_std = float(np.std(pcis))
    pci_latest = pcis[0]
    latest_rank = ranks[0] if ranks else None
    # 100 基準: 100±15 から外れたら「ペース乱高下」とみなす
    pace_change_count = sum(1 for p in pcis if abs(p - 100) >= 15)

    # 巻返し候補判定: 直近で PCI が 100 から大きく離れたのに着順が良かった
    is_comeback = False
    for p, rk in zip(pcis[:3], ranks[:3]):
        if abs(p - 100) >= 20 and rk is not None and rk <= 5:
            is_comeback = True
            break

    if is_comeback:
        label = "ペース乱高下耐性"
        bonus = 0.04
    elif pace_change_count >= 2 and pci_std >= 12:
        label = "ペース崩れに弱い可能性"
        bonus = -0.02
    elif abs(pci_avg - 100) <= 10:
        label = "安定ペース型"
        bonus = 0.01
    else:
        label = "中立"
        bonus = 0.0

    return {
        "pci_avg":           round(pci_avg, 1),
        "pci_std":           round(pci_std, 1),
        "pci_latest":        round(pci_latest, 1),
        "pci_latest_rank":   latest_rank,
        "pace_change_count": pace_change_count,
        "label":             label,
        "bonus":             bonus,
        "n_races":           len(pcis),
    }


def _empty_pci_result() -> dict:
    return {
        "pci_avg":         None,
        "pci_std":         None,
        "pci_latest":      None,
        "pci_latest_rank": None,
        "pace_change_count": 0,
        "label":           "データなし",
        "bonus":           0.0,
        "n_races":         0,
    }


def pace_pci_match(predicted_pace: str, pci_avg: float | None) -> dict:
    """
    BOOST-3: 予測ペースと馬のPCI傾向のマッチ度を判定。

    Args:
        predicted_pace: "ハイペース" / "ミドル〜ハイ" / "ミドル" / "スローペース"
        pci_avg:        過去5走PCI平均（100基準、>100=差し型、<100=前残り型）

    Returns:
        {"match_label": str, "match_bonus": float} bonus ∈ [-0.04, +0.04]
    """
    if pci_avg is None or (isinstance(pci_avg, float) and np.isnan(pci_avg)):
        return {"match_label": "", "match_bonus": 0.0}
    pace_str = str(predicted_pace or "")
    is_high = "ハイ" in pace_str
    is_slow = "スロー" in pace_str

    if is_high:
        if pci_avg >= 105:
            return {"match_label": "差し型×ハイペース◎", "match_bonus": 0.04}
        if pci_avg <= 95:
            return {"match_label": "前残り型×ハイペース✗", "match_bonus": -0.03}
    elif is_slow:
        if pci_avg <= 95:
            return {"match_label": "前残り型×スロー◎", "match_bonus": 0.04}
        if pci_avg >= 105:
            return {"match_label": "差し型×スロー✗", "match_bonus": -0.03}
    return {"match_label": "中立", "match_bonus": 0.0}


if __name__ == "__main__":
    # サンプル: 東京 1800m 良 仮想レース
    print("=== PCI 計算サンプル ===")
    samples = [
        ("均等ペース", 121.5, 35.0, 2000),    # 期待: ~50
        ("超スロー前半", 124.0, 33.0, 2000),  # 期待: >> 50
        ("超ハイ前半",   118.0, 37.0, 2000),  # 期待: << 50
    ]
    for label, ft, l3, d in samples:
        pci = calc_pci_row(ft, l3, d)
        print(f"  {label}: ft={ft} l3={l3} d={d} → PCI={pci:.1f}")

    # 実データテスト
    p = _DATA_DIR / "tfjv_all.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # 最頻馬で確認
        top_horses = df["horse_name"].value_counts().head(3).index.tolist()
        for h in top_horses:
            stats = get_horse_pci_stats(df, h)
            print(f"\n  {h}: {stats}")
