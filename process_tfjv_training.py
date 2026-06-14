"""
TFJVからエクスポートした調教CSVを処理し、アプリが自動読み込みできる形式で保存する。
Claudeが「調教データをエクスポートしました」の報告を受けた際に実行するスクリプト。

使い方:
  python process_tfjv_training.py <csvファイルパス> [レース日 YYYY-MM-DD] [レース名]

例:
  python process_tfjv_training.py C:/TFJV/TXT/training_okus.csv 2026-05-25 オークス
  python process_tfjv_training.py C:/TFJV/TXT/heian_slopw.csv 2026-05-25 平安S
"""
import sys
from pathlib import Path
from tfjv_training import load_tfjv_training, evaluate_training_tfjv, save_training_cache


def main():
    if len(sys.argv) < 2:
        print("使い方: python process_tfjv_training.py <csvパス> [レース日] [レース名]")
        sys.exit(1)

    csv_path   = Path(sys.argv[1])
    race_date  = sys.argv[2] if len(sys.argv) > 2 else ""
    race_label = sys.argv[3] if len(sys.argv) > 3 else csv_path.stem

    if not csv_path.exists():
        print(f"ファイルが見つかりません: {csv_path}")
        sys.exit(1)

    print(f"読み込み中: {csv_path}")
    sessions_map = load_tfjv_training(csv_path)
    print(f"  馬数: {len(sessions_map)}頭")

    results = {}
    for name, sessions in sessions_map.items():
        results[name] = evaluate_training_tfjv(name, sessions, race_date)

    save_training_cache(results, race_label)
    print(f"\n[{race_label}] 調教評価完了")
    print(f"{'馬名':<22} {'ラベル':<22} {'bonus':>6} {'最終4F':>7} {'推移'}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["score"]):
        t1 = f"{r['last_time1']:.1f}s" if r["last_time1"] else "  -  "
        print(f"{name:<22} {r['label']:<22} {r['bonus']:>+6.0f} {t1:>7} {r['trend']}")

    print(f"\nキャッシュ保存: {Path('data/tfjv_training_cache.json').resolve()}")
    print("アプリを再起動すると自動的に反映されます")


if __name__ == "__main__":
    main()
