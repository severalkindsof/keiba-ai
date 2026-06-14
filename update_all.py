"""
REFINE-1: モデル・統計テーブル一括更新スクリプト

(2026-06-10) Windows cp932 でも安全になるよう stdout を utf-8 reconfigure。
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
"""

実行順:
    1. convert_tfjv.py    — TFJV CSV → tfjv_all.parquet (新データ反映)
    2. train_lgbm.py      — LightGBM 再学習 + Isotonic 校正 + Benter重み
    3. horse_elo.py       — 馬 Elo レーティング更新
    4. trainer_jockey_matrix.py — 厩舎×騎手マトリクス再構築
    5. speed_index.py     — スピード指数ベースライン更新
    6. track_variant.py   — トラックバイアス時系列更新
    7. market_prob.py     — 市場確率テーブル更新

使い方:
    python update_all.py            # 全部実行
    python update_all.py --skip-train  # train_lgbm のみスキップ
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


STEPS = [
    ("convert_tfjv.py",          "TFJV CSV → parquet 変換",        []),
    ("train_lgbm.py",            "LightGBM 再学習 + 校正",          []),
    ("horse_elo.py",             "馬 Elo レーティング",            []),
    ("trainer_jockey_matrix.py", "厩舎×騎手マトリクス",            []),
    ("speed_index.py",           "スピード指数ベースライン",        []),
    ("track_variant.py",         "トラックバイアス時系列",  ["2024-01-01"]),
    ("market_prob.py",           "市場確率テーブル",                []),
]


def run_step(script: str, label: str, args: list[str]) -> bool:
    script_path = ROOT / script
    if not script_path.exists():
        print(f"  {script} が見つかりません。スキップ。")
        return False
    print(f"\n{'='*70}")
    print(f"▶ {script}  [{label}]")
    print(f"{'='*70}")
    t0 = time.time()
    try:
        import os
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [sys.executable, str(script_path), *args],
            cwd=ROOT, check=False, env=env,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"  完了 ({elapsed:.1f}秒)")
            return True
        else:
            print(f"  失敗 (exit={result.returncode}, {elapsed:.1f}秒)")
            return False
    except Exception as e:
        print(f"  例外: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="train_lgbm.py をスキップ（既存モデル使用）")
    parser.add_argument("--skip-convert", action="store_true",
                        help="convert_tfjv.py をスキップ")
    parser.add_argument("--only", nargs="+", default=None,
                        help="指定スクリプトのみ実行")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"# 競馬AI: 全テーブル・モデル一括更新")
    print(f"# 推定所要時間: 15〜30分（train_lgbm 含む）")
    print(f"{'#'*70}")

    success = 0
    failed = 0
    t_total = time.time()

    for script, label, sargs in STEPS:
        if args.only and script not in args.only:
            continue
        if args.skip_train and script == "train_lgbm.py":
            print(f"\n{script} スキップ")
            continue
        if args.skip_convert and script == "convert_tfjv.py":
            print(f"\n{script} スキップ")
            continue
        ok = run_step(script, label, sargs)
        if ok: success += 1
        else:  failed += 1

    elapsed = time.time() - t_total
    print(f"\n{'#'*70}")
    print(f"# 完了: 成功 {success} / 失敗 {failed} / 合計 {elapsed/60:.1f}分")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
