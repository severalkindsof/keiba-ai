"""
監査ツール（再発防止用）

出力 encoding は Windows cp932 でも安全になるよう sys.stdout を utf-8 にreconfigure。
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

"""

検出項目:
    1. evaluate_horse() の base dict キー
    2. app.py で e[...] 代入されるキー
    3. confluence horse.get() で参照されるキー
    4. 不整合（confluence期待 vs 実供給）
    5. div-by-var サイト一覧
    6. キャッシュ TTL 分布
    7. session_state キー重複度

使い方:
    python tools/audit.py            # 全項目
    python tools/audit.py --section keys  # 特定セクションのみ
"""
import argparse
import re
from pathlib import Path
from collections import Counter, defaultdict
import glob

ROOT = Path(__file__).resolve().parent.parent


def section_evaluate_horse_keys():
    print("=" * 60)
    print("[1] evaluate_horse() base dict キー")
    print("=" * 60)
    text = (ROOT / "ev_calculator.py").read_text(encoding="utf-8")
    m = re.search(r"base = \{(.+?)^\s{4}\}", text, re.MULTILINE | re.DOTALL)
    if m:
        body = m.group(1)
        # Y3: base dict の最左 "key": だけを抽出（horse.get("...") の中の文字列は無視）
        keys = sorted(set(re.findall(r'^\s+"(\w+)"\s*:', body, re.MULTILINE)))
        print(f"  {len(keys)} 件:")
        for k in keys:
            print(f"    {k}")
        return set(keys)
    return set()


def section_app_entries_keys():
    print("\n" + "=" * 60)
    print("[2] app.py で e[...] 代入されるキー")
    print("=" * 60)
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    keys = sorted(set(re.findall(r'e\[\"(\w+)\"\]\s*=', text)))
    print(f"  {len(keys)} 件:")
    for k in keys:
        print(f"    {k}")
    return set(keys)


def section_bonus_cols_keys():
    print("\n" + "=" * 60)
    print("[3] _bonus_cols（entries→eval_df マージ対象）")
    print("=" * 60)
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    m = re.search(r"_bonus_cols = \[(.+?)\]", text, re.DOTALL)
    if m:
        body = m.group(1)
        body = re.sub(r"#.*", "", body)
        keys = sorted(set(re.findall(r'\"(\w+)\"', body)))
        print(f"  {len(keys)} 件:")
        for k in keys:
            print(f"    {k}")
        return set(keys)
    return set()


def section_confluence_keys():
    print("\n" + "=" * 60)
    print("[4] confluence horse.get() 参照キー")
    print("=" * 60)
    text = (ROOT / "confluence.py").read_text(encoding="utf-8")
    keys = sorted(set(re.findall(r'horse\.get\(\"(\w+)\"', text)))
    print(f"  {len(keys)} 件:")
    for k in keys:
        print(f"    {k}")
    return set(keys)


def section_mismatch(base_keys, app_keys, bonus_cols, conf_keys):
    print("\n" + "=" * 60)
    print("[5] 不整合: confluence 期待 vs 実供給")
    print("=" * 60)
    supplied = base_keys | app_keys | bonus_cols
    missing = conf_keys - supplied
    if missing:
        print(f"  [!] confluence が期待するが、どこにも供給されていない {len(missing)} 件:")
        for k in sorted(missing):
            print(f"    {k}")
    else:
        print("  OK 全 confluence 期待キーが供給されている")


def section_cache_ttls():
    print("\n" + "=" * 60)
    print("[6] @st.cache_data TTL 分布")
    print("=" * 60)
    ttls = defaultdict(list)
    for f in glob.glob(str(ROOT / "*.py")):
        text = Path(f).read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, l in enumerate(lines):
            m = re.search(r"@st\.cache_data\(ttl=(\d+)\)", l)
            if m:
                # 直後の def を探す
                for j in range(i, min(i + 4, len(lines))):
                    fm = re.match(r"\s*def (\w+)", lines[j])
                    if fm:
                        ttls[int(m.group(1))].append(f"{Path(f).name}:{fm.group(1)}")
                        break
    for ttl in sorted(ttls.keys()):
        print(f"  TTL={ttl:>6}秒 ({len(ttls[ttl])}件):")
        for x in ttls[ttl]:
            print(f"    {x}")


def section_div_risks():
    print("\n" + "=" * 60)
    print("[7] div-by-var 候補（clip 未指定の除算）")
    print("=" * 60)
    SAFE = {"clip", "max", "min", "sum", "len", "round", "int", "float", "abs",
            "sqrt", "log", "exp", "self", "np", "pd", "math", "axis", "race_id",
            "race_key", "col", "copy", "f"}
    risky = []
    for f in glob.glob(str(ROOT / "*.py")):
        text = Path(f).read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, l in enumerate(lines, 1):
            if "/ " not in l or "clip" in l or "//" in l:
                continue
            m = re.search(r"/\s*([a-zA-Z_]\w*)\b(?!\.)", l)
            if m and m.group(1) not in SAFE and m.group(1).islower():
                risky.append((Path(f).name, i, l.strip()[:80]))
    print(f"  {len(risky)} 件:")
    for fn, ln, l in risky[:20]:
        print(f"    {fn}:{ln}  {l}")
    if len(risky) > 20:
        print(f"    ... 他 {len(risky)-20} 件")


def section_runtime():
    """Y2: 実走時の eval_df.columns ダンプを真実のソースとして使う"""
    print("\n" + "=" * 60)
    print("[9] Runtime audit (eval_df.columns dump)")
    print("=" * 60)
    dump_path = ROOT / "data" / "last_eval_df_columns.json"
    if not dump_path.exists():
        print(f"  {dump_path} が無い。app.py で分析実行→自動 dump を生成してください。")
        return
    import json
    with open(dump_path, encoding="utf-8") as f:
        info = json.load(f)
    cols = set(info.get("columns", []))
    print(f"  最終分析時の eval_df: {len(cols)} 列, 行数 {info.get('n_rows', '?')}")
    # confluence 期待キーとの照合
    text = (ROOT / "confluence.py").read_text(encoding="utf-8")
    conf = set(re.findall(r'horse\.get\(\"(\w+)\"', text))
    miss = conf - cols
    if miss:
        print(f"  [!] eval_df に欠落（confluence期待）: {len(miss)} 件")
        for k in sorted(miss):
            print(f"    {k}")
    else:
        print("  OK eval_df は confluence 期待を全て満たす")


def section_session_state_keys():
    print("\n" + "=" * 60)
    print("[8] session_state キー使用度")
    print("=" * 60)
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    keys = re.findall(r'session_state\[\"(\w+)\"\]', text)
    keys += re.findall(r'session_state\.get\(\"(\w+)\"', text)
    c = Counter(keys)
    print(f"  ユニーク {len(c)} / 総参照 {sum(c.values())}")
    print(f"  Top 15:")
    for k, n in c.most_common(15):
        print(f"    {k}: {n} 回")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="all",
                        choices=["all", "keys", "bonus", "confluence", "mismatch",
                                 "ttl", "div", "session", "runtime"])
    args = parser.parse_args()

    base = section_evaluate_horse_keys() if args.section in ("all", "keys") else set()
    app = section_app_entries_keys()       if args.section in ("all", "keys") else set()
    bc = section_bonus_cols_keys()         if args.section in ("all", "bonus") else set()
    cf = section_confluence_keys()         if args.section in ("all", "confluence") else set()
    if args.section in ("all", "mismatch"):
        section_mismatch(base, app, bc, cf)
    if args.section in ("all", "ttl"):
        section_cache_ttls()
    if args.section in ("all", "div"):
        section_div_risks()
    if args.section in ("all", "session"):
        section_session_state_keys()
    if args.section in ("all", "runtime"):
        section_runtime()


if __name__ == "__main__":
    main()
