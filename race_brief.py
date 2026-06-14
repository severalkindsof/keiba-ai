"""対話分析用 統合レースブリーフ

1コマンドで「出馬表 → LGBM評価 → 統一判定 → 相手強度 → 展開 → コース適性」を
まとめて返す。対話でのレース分析を高速化するためのエントリポイント。

使い方:
    python -X utf8 race_brief.py 20260613021111
    python -X utf8 race_brief.py 函館 11
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

_DIR = Path(__file__).parent
sys.path.insert(0, str(_DIR))


def _latest_entries_csv() -> str:
    files = sorted(Path("C:/TFJV/TXT").glob("出馬表分析*.CSV"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else ""


def analyze_race(race_id: str = "", venue: str = "", race_no: int = 0,
                 csv_path: str = "", mode: str = "爆穴", track_condition: str = "") -> dict:
    """統合分析を実行して dict を返す。"""
    from tfjv_entries import load_tfjv_entries
    from ev_calculator import evaluate_race
    from confluence import add_confluence_to_eval
    from unified_verdict import apply_unified_verdict, sort_by_verdict
    from opponent_strength import build_field_opponent_strength
    from running_style import predict_pace, pace_fit_bonus

    csv_path = csv_path or _latest_entries_csv()
    if not csv_path:
        return {"error": "出馬表分析CSVが見つかりません"}
    data = load_tfjv_entries(csv_path)

    # レース特定
    if not race_id:
        for rid, info in data.items():
            if info["venue"] == venue and int(info["race_no"]) == int(race_no):
                race_id = rid
                break
    info = data.get(race_id)
    if not info:
        return {"error": f"レースが見つかりません: {race_id} / {venue}{race_no}R"}

    entries = info["entries"]
    # 実馬場をnetkeibaから取得(ハードコード"良"を廃止。LGBMはtrack_condition特徴量を持つ)
    tc = track_condition
    if not tc:
        try:
            from netkeiba_odds import fetch_track_condition
            tc = fetch_track_condition(str(info.get("date_str", "")), info["venue"], int(info["race_no"])) or "良"
        except Exception:
            tc = "良"
    info["track_condition"] = tc
    for e in entries:
        e["surface"] = info["surface"]
        e["distance"] = info["distance"]
        e["track_condition"] = tc
    names = [e["horse_name"] for e in entries]

    # 1. LGBM評価 + 合流スコア
    wrt = pd.read_parquet(_DIR / "data/win_rate_table.parquet")
    sire = pd.read_parquet(_DIR / "data/sire_stats.parquet")
    jky = pd.read_parquet(_DIR / "data/jockey_stats.parquet")
    eval_df = evaluate_race(entries, wrt, sire, jky)
    eval_df = add_confluence_to_eval(eval_df)

    # ベース判定（LGBM + 合流スコア）。ボーナスはスコアに畳み込まず別レンズで併記
    eval_df = apply_unified_verdict(eval_df, mode=mode)
    eval_df = sort_by_verdict(eval_df)

    # 別レンズ: 相手強度 / 展開 / コース適性（過剰補正を避け、独立した判断材料として保持）
    opp = build_field_opponent_strength(names)
    pace = predict_pace(names)
    # 市場見限りエリート（爆穴検出。安田2026・VM2024で実証）
    pop_map = {e["horse_name"]: (int(e["popularity"]) if e.get("popularity") else None)
               for e in entries}
    try:
        from elite_neglect import build_elite_neglect
        elite = build_elite_neglect(names, pop_map)
    except Exception:
        elite = {}
    # 本命軸信頼度（鉄板/標準/罠。PIT検証済み）
    try:
        from honmei import build_honmei_reliability
        honmei = build_honmei_reliability(names, pop_map)
    except Exception:
        honmei = {}
    # 検証済み穴フラグ（第64波・大標本リフト実証）
    try:
        from anaba_score import anaba_flags
        anaba = {h: anaba_flags(h, info["venue"], info["surface"], info["distance"])
                 for h in names if (pop_map.get(h) or 0) >= 7}
    except Exception:
        anaba = {}
    try:
        from course_aptitude import build_aptitude_tag
    except Exception:
        build_aptitude_tag = None
    jmap = {e["horse_name"]: e.get("jockey", "") for e in entries}
    try:
        from knowledge_base import get_local_jockey_bonus
    except Exception:
        get_local_jockey_bonus = None
    try:
        from extra_factors import build_extra_factors, parse_class_level, sire_mud_aptitude
        cur_cls = parse_class_level(info["race_class"])
    except Exception:
        build_extra_factors = None
        sire_mud_aptitude = None
        cur_cls = None
    try:
        from draw_bias import get_draw_label
    except Exception:
        get_draw_label = None
    try:
        from jockey_anaba import lookup as jockey_anaba_lookup
    except Exception:
        jockey_anaba_lookup = None
    try:
        from blood_anaba import lookup as blood_anaba_lookup
    except Exception:
        blood_anaba_lookup = None
    # 格credential(第86波・検証済): 人気馬〜中位の重賞のみ。穴では格無効のため出さない。
    try:
        from grade_class import class_level as _grade_cls_level, grade_tag as _grade_tag
        _cred_df = pd.read_parquet('data/grade_credential.parquet')
        _today_cls = _grade_cls_level(info.get("race_class") or "")
    except Exception:
        _grade_tag = None
        _cred_df = None
        _today_cls = 3
    # 重賞枠ゾーン実測lift(第86波): 一般draw_biasが重賞特性と逆になるのを実測で上書き
    try:
        from grade_waku import waku_tag as _waku_tag
    except Exception:
        _waku_tag = None
    # 乗替→トップ/非トップ(第86波・検証1.26x/0.71x): 人気馬〜中位の重賞のみ
    try:
        from grade_class import jockey_change_tag as _jchg_tag
        _prev_jk_df = pd.read_parquet('data/prev_jockey.parquet')
    except Exception:
        _jchg_tag = None
        _prev_jk_df = None
    # 馬×騎手 個別コンビ実績(集計liftより優先・ユーザー指摘で追加)
    try:
        from grade_class import combo_tag as _combo_tag
        _combo_df = pd.read_parquet('data/horse_jockey_combo.parquet')
    except Exception:
        _combo_tag = None
        _combo_df = None
    # ★第88波 役割分担: 穴(7番↓)は独立実力モデル(popularity抜き)で買える穴/消し穴判定(バックテスト穴lift1.75x実証)。
    #   本命側は人気込みモデルに任せる。
    try:
        import independent_anaba as _ia
        _ia_scores = _ia.score_race(names, surface=info["surface"],
                                    track_condition=info.get("track_condition", "良"),
                                    venue=info["venue"], distance=int(float(info.get("distance") or 0)))
    except Exception:
        _ia = None
        _ia_scores = {}
    # 複合スコア(7要素)の特徴供給: 出走馬の最新Elo・前走情報→レース内Eloパーセンタイル
    comp_ctx, comp_elo_pct = {}, {}
    try:
        import anaba_composite as _ac
        comp_ctx = _ac.horse_context([e["horse_name"] for e in entries])
        _elos = {n: c["pre_elo"] for n, c in comp_ctx.items() if c.get("pre_elo")}
        if len(_elos) >= 3:
            ser = pd.Series(_elos).rank(pct=True)
            comp_elo_pct = ser.to_dict()
    except Exception:
        _ac = None

    def _dcat(d):
        d = int(d)
        if d <= 1400: return "短距離"
        if d <= 1800: return "マイル"
        if d <= 2200: return "中距離"
        return "長距離"

    dcat = _dcat(info["distance"])
    n_field = len(entries)
    hidmap = {e["horse_name"]: str(e.get("horse_id", "") or "") for e in entries}
    gatemap = {e["horse_name"]: e.get("horse_no", 0) for e in entries}
    siremap = {e["horse_name"]: e.get("sire", "") for e in entries}
    popmap = {e["horse_name"]: e.get("popularity") for e in entries}
    dammap = {e["horse_name"]: e.get("damsire", "") for e in entries}

    # 調教レンズ: 過去全週の training*.csv（履歴蓄積方式）をマージして出走馬を評価。
    # LGBM/ev_calculatorには調教が入っていないため、ここで後付けレンズとして配線する。
    # 同馬のセッションが複数週たまるほど「直近平均との乖離・トレンド」評価が効き始める。
    train_eval = {}
    try:
        import glob as _glob
        from tfjv_training import load_tfjv_training, evaluate_training_tfjv
        # training0611.csv のような数字日付ファイルのみ（training_okus等のゴミを除外）。全件＝履歴。
        tcsvs = sorted(_glob.glob("C:/TFJV/TXT/training[0-9]*.csv"))
        merged = {}
        for tp in tcsvs:
            for k, v in load_tfjv_training(tp).items():
                merged.setdefault(k.strip(), []).extend(v)
        for sess in merged.values():
            sess.sort(key=lambda s: s["date"])  # セッションのキーは date(datetime.date)
        ds = str(info.get("date_str", ""))
        race_date = f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}" if len(ds) == 8 else ""
        for h in names:
            key = h.strip()
            s = merged.get(key)
            if s is None:
                for k in merged:
                    if key and (key in k or k in key):
                        s = merged[k]; break
            if s:
                train_eval[h] = evaluate_training_tfjv(h, s, race_date)
    except Exception:
        train_eval = {}

    # 複合スコアはレース内相対で評価(絶対境界だと重賞で全馬軸になるため)。全馬分を先に計算
    comp_scores = {}
    if _ac is not None:
        for e in entries:
            h = e["horse_name"]; hn = h.strip()
            sr = str(pace.get("styles", {}).get(h, {}).get("style", ""))
            st = ("逃先" if sr in ("逃げ", "先行", "逃", "先") else
                  "差追" if sr in ("差し", "追込", "追い込み", "差", "追", "後") else
                  "中団" if sr in ("中団", "中") else None)
            ctx = comp_ctx.get(hn, {})
            try:
                comp_scores[h] = _ac.score(
                    elo_pct=comp_elo_pct.get(hn), style=st, jockey=e.get("jockey", ""),
                    sire=e.get("sire", ""), damsire=e.get("damsire", ""),
                    surface=info["surface"], track_condition=info.get("track_condition", "良"),
                    venue=info["venue"], distance_cat=dcat, distance=info["distance"],
                    horse_no=e.get("horse_no"), field_size=len(entries),
                    popularity=e.get("popularity"),
                    prev_rank=ctx.get("prev_rank"), prev_l3f_pct=ctx.get("prev_l3f_pct"))["score"]
            except Exception:
                comp_scores[h] = 0.0
    comp_pct = pd.Series(comp_scores).rank(pct=True).to_dict() if comp_scores else {}

    lenses = {}
    local_alerts = []
    for h in names:
        course_tags = ""
        if build_aptitude_tag:
            try:
                a = build_aptitude_tag(h, info["venue"], info["surface"], info["distance"])
                course_tags = " ".join(a.get("tags", []))
            except Exception:
                pass
        local_tag = ""
        if get_local_jockey_bonus:
            lj = get_local_jockey_bonus(jmap.get(h, ""))
            if lj["is_local"]:
                local_tag = "🚨地方騎手"
                local_alerts.append(f"{h}（{jmap.get(h,'')}）: {lj['note']}")
        extra_tags = ""
        if build_extra_factors:
            try:
                ef = build_extra_factors(h, cur_cls, info["venue"], info["surface"],
                                         info["distance"], info["date_str"])
                extra_tags = " ".join(ef.get("tags", []))
            except Exception:
                pass
        # 道悪血統（馬場が悪い時のみ）
        mud_tag = ""
        if sire_mud_aptitude:
            try:
                mud = sire_mud_aptitude(siremap.get(h, ""), info.get("track_condition", "良"))
                mud_tag = mud.get("tag", "")
            except Exception:
                pass
        # 枠バイアス
        draw_tag = ""
        if get_draw_label:
            try:
                dl = get_draw_label(info["venue"], info["surface"], dcat,
                                    int(gatemap.get(h, 5) or 5), n_field)
                if dl and ("有利" in dl or "不利" in dl):
                    draw_tag = dl
            except Exception:
                pass
        # 重賞枠ゾーン実測タグ。重賞では一般draw_bias(内有利前提)を抑制し実測を優先。
        gwaku_tag = ""
        if _waku_tag is not None and _today_cls >= 6:
            try:
                gwaku_tag = _waku_tag(info.get("race_class", ""), info["venue"], info["surface"],
                                      int(float(info.get("distance") or 0)),
                                      int(gatemap.get(h, 0) or 0), n_field)
            except Exception:
                pass
        # 重賞は一般draw_bias(内有利前提)が実測と逆になりがち→常に抑制し実測ゾーンliftのみ残す
        if _today_cls >= 6:
            draw_tag = ""
        # 単騎逃げ
        lone = "単騎逃げ" if pace.get("lone_leader") == h else ""
        # 市場見限りエリート
        elite_tag = elite.get(h, {}).get("tag", "") if elite else ""
        honmei_tag = honmei.get(h, {}).get("tag", "") if honmei else ""
        anaba_tag = ""
        if h in anaba and anaba[h]["n_flags"] >= 2:
            anaba_tag = f"穴{anaba[h]['n_flags']}フラグ({anaba[h]['tier']})"
        # 馬の脚質を逃先/中団/差追にマップ（穴は逃先で激走1.53x・差追は0.48xで消し）。穴騎手/血統で共用。
        _pop_h = popmap.get(h)
        _sr = str(pace.get("styles", {}).get(h, {}).get("style", ""))
        _st = ("逃先" if _sr in ("逃げ", "先行", "逃", "先") else
               "差追" if _sr in ("差し", "追込", "追い込み", "差", "追", "後") else
               "中団" if _sr in ("中団", "中") else None)
        # 穴騎手レンズ: その馬のオッズが穴寄り(7番人気以下)の時のみ発動（人気サイドは無意味）。
        janaba_tag = ""
        if jockey_anaba_lookup and _pop_h and int(_pop_h) >= 7:
            try:
                ja = jockey_anaba_lookup(jmap.get(h, ""), info["surface"], dcat, style=_st, venue=info["venue"])
                if ja["src"] == "foreign_prior":
                    janaba_tag = "🌍穴騎手(短期外国人1.52x)"
                elif ja["lb_lift"] >= 1.10:
                    cl = "".join(f"{v}{lf:.1f}x" for v, lf in ja.get("cond_lift", []))
                    janaba_tag = f"穴騎手({ja['lb_lift']:.2f}x {cl})"
            except Exception:
                pass
        # 調教レンズ: 好仕上がり(◎)・低調(▲)のみ注記（普通は省略）。bonusも併記
        training_tag = ""
        tr = train_eval.get(h)
        if tr and tr.get("sessions_count", 0) > 0:
            lab = tr.get("label", "")
            if lab.startswith("調教◎") or lab.startswith("調教▲"):
                training_tag = f"{lab}({tr.get('bonus', 0):+.1f})"
        # 穴血統レンズ: 穴馬(7番人気以下)の父・母父が買い/消し血統か（脚質・会場条件も加味）
        blood_tag = ""
        if blood_anaba_lookup and _pop_h and int(_pop_h) >= 7:
            try:
                bl = blood_anaba_lookup(siremap.get(h, ""), dammap.get(h, ""),
                                        surface=info["surface"], distance_cat=dcat,
                                        style=_st, venue=info["venue"])
                if bl["tags"]:
                    blood_tag = " ".join(bl["tags"])
            except Exception:
                pass
        # 複合スコア(7要素)レンズ: 穴馬(7番人気以下)にレース内相対で軸候補/消し判定
        comp_tag = ""
        if comp_scores and _pop_h and int(_pop_h) >= 7:
            p = comp_pct.get(h, 0.5)
            sc = comp_scores.get(h, 1.0)
            if p >= 0.6:      # レース内スコア上位40%＝この相手なら軸候補
                comp_tag = f"【複合◎軸候補 内{p*100:.0f}%ile {sc:.2f}】"
            elif p <= 0.3:    # 下位30%＝消し
                comp_tag = f"【複合✕消し 内{p*100:.0f}%ile {sc:.2f}】"
        # 格credentialタグ: 人気馬〜中位(pop<=6)の重賞(cls>=6)のみ。穴(7番人気↓)では格無効で出さない。
        grade_cred_tag = ""
        if _grade_tag is not None and _cred_df is not None and _today_cls >= 6 \
                and _pop_h and int(_pop_h) <= 6:
            try:
                grade_cred_tag = _grade_tag(h, _today_cls, _cred_df, horse_id=hidmap.get(h))
            except Exception:
                pass
        # 乗替タグ: 人気馬〜中位(pop<=6)×重賞のみ。穴では乗替も無効で出さない。
        jchg_tag = ""
        if _today_cls >= 6 and _pop_h and int(_pop_h) <= 6:
            cj = jmap.get(h, "")
            # 個別コンビ実績を最優先(集計liftはこの馬に当てはまらないことがある=レーン例)
            if _combo_tag is not None and _combo_df is not None:
                try:
                    jchg_tag = _combo_tag(h, cj, _combo_df)
                except Exception:
                    pass
            # 新コンビ(過去騎乗なし)のときだけ集計lift(乗替→トップ等)にフォールバック
            if not jchg_tag and _jchg_tag is not None and _prev_jk_df is not None:
                try:
                    jchg_tag = _jchg_tag(h, cj, _prev_jk_df)
                except Exception:
                    pass
        # 独立モデルの穴判定(7番人気↓のみ)
        ia_tag = ""
        if _ia_scores and _pop_h and int(_pop_h) >= 7 and h in _ia_scores:
            try:
                ia_tag = _ia.anaba_verdict_tag(_ia_scores[h]["pct"])
            except Exception:
                pass
        lenses[h] = {
            "grade": grade_cred_tag,
            "jchg": jchg_tag,
            "indep_anaba": ia_tag,
            "elite": elite_tag,
            "honmei": honmei_tag,
            "anaba": anaba_tag,
            "janaba": janaba_tag,
            "blood": blood_tag,
            "composite": comp_tag,
            "training": training_tag,
            "opp_tag": opp.get(h, {}).get("tag", ""),
            "tough_close": opp.get(h, {}).get("tough_close", 0),
            "class_drop": opp.get(h, {}).get("class_drop"),
            "style": pace["styles"].get(h, {}).get("style", "?"),
            "pace_fit": pace_fit_bonus_safe(h, pace).get("tag", ""),
            "course_tags": course_tags,
            "local_jockey": local_tag,
            "extra": extra_tags,
            "mud": mud_tag,
            "draw": draw_tag,
            "grade_waku": gwaku_tag,
            "lone": lone,
        }
    res_adj = lenses

    return {
        "local_alerts": local_alerts,
        "race_id": race_id,
        "venue": info["venue"], "race_no": info["race_no"],
        "race_class": info["race_class"],
        "surface": info["surface"], "distance": info["distance"],
        "track_condition": info.get("track_condition", "良"),
        "field_size": len(entries),
        "pace": pace,
        "eval_df": eval_df,
        "opp": opp,
        "adj": res_adj,
    }


def print_brief(res: dict):
    if "error" in res:
        print(res["error"]); return
    print(f"======== {res['venue']}{res['race_no']}R {res['race_class']} "
          f"{res['surface']}{res['distance']}m 馬場:{res.get('track_condition','良')} {res['field_size']}頭 ========")
    print(f"展開: {res['pace']['detail']}")
    # 重賞は過去の勝ち脚質傾向を併記(今年予想との対立を読み手が判断できるように)
    try:
        from grade_waku import style_tendency
        _st = style_tendency(res.get("race_class", ""))
        if _st:
            print(f"過去傾向: {_st}")
    except Exception:
        pass
    if res.get("local_alerts"):
        print("🚨 地方騎手中央騎乗（勝負気配）:")
        for a in res["local_alerts"]:
            print(f"   {a}")
    print()
    df = res["eval_df"]
    lenses = res["adj"]
    for _, r in df.iterrows():
        h = r["horse_name"]
        L = lenses.get(h, {})
        st = L.get("style", "?")
        _sty = res["pace"]["styles"].get(h, {})
        if _sty.get("unstable"):
            st += "⚠"   # 脚質不安定(先行↔後方を行き来・前残り期待薄)
        elif _sty.get("weak"):
            st += "薄"   # 標本不足
        pop = int(r["popularity"]) if pd.notna(r.get("popularity")) else 0
        ev = r.get("ev")
        ev_s = f"{ev:+.2f}" if pd.notna(ev) else "—"
        # 追加レンズの注記
        notes = []
        if L.get("grade"): notes.append(L["grade"])
        if L.get("jchg"): notes.append(L["jchg"])
        if L.get("indep_anaba"): notes.append(L["indep_anaba"])
        if L.get("composite"): notes.append(L["composite"])
        if L.get("honmei"): notes.append(L["honmei"])
        if L.get("elite"): notes.append(L["elite"])
        if L.get("anaba"): notes.append(L["anaba"])
        if L.get("janaba"): notes.append(L["janaba"])
        if L.get("blood"): notes.append(L["blood"])
        if L.get("training"): notes.append(L["training"])
        if L.get("local_jockey"): notes.append(L["local_jockey"])
        if L.get("lone"): notes.append(L["lone"])
        if L.get("opp_tag"): notes.append(L["opp_tag"])
        if L.get("course_tags"): notes.append(L["course_tags"])
        if L.get("extra"): notes.append(L["extra"])
        if L.get("mud"): notes.append(L["mud"])
        if L.get("draw"): notes.append(L["draw"])
        if L.get("grade_waku"): notes.append(L["grade_waku"])
        if L.get("pace_fit"): notes.append(L["pace_fit"])
        note_s = " ⟨" + " / ".join(notes) + "⟩" if notes else ""
        line = (f"{r['final_verdict']:6s} {h:13s} {pop:2d}人気 {st:3s} EV{ev_s} "
                f"| {r['final_reason'][:34]}{note_s}")
        print(line)


def pace_fit_bonus_safe(h, pace):
    from running_style import pace_fit_bonus
    try:
        return pace_fit_bonus(h, pace)
    except Exception:
        return {"bonus": 0.0, "tag": "", "style": ""}


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 1 and args[0].isdigit() and len(args[0]) >= 8:
        res = analyze_race(race_id=args[0])
    elif len(args) == 2:
        res = analyze_race(venue=args[0], race_no=int(args[1]))
    else:
        print("usage: race_brief.py <race_id> | <venue> <race_no>")
        sys.exit(1)
    print_brief(res)
