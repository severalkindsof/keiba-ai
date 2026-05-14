"""
ユーザーの競馬知識ベース（knowledge_base.json）のローダー＆クエリモジュール。

使い方:
    from knowledge_base import get_jockey_bonus, get_sire_bonus, get_race_pattern_info, get_claude_context_for_race

各関数はスコア補正値（bonus）とメモ文字列を返す。
ev_calculator.py の evaluate_horse() から呼び出して合計スコアに加算する。
"""
import json
import os
from pathlib import Path

# ---- JSONロード ---- #

_KB_PATH = Path(__file__).parent / "knowledge_base.json"
_kb_cache: dict | None = None


def load_kb() -> dict:
    """knowledge_base.json を読み込んでキャッシュする。"""
    global _kb_cache
    if _kb_cache is None:
        if not _KB_PATH.exists():
            _kb_cache = {}
        else:
            with open(_KB_PATH, encoding="utf-8") as f:
                _kb_cache = json.load(f)
    return _kb_cache


def reload_kb() -> dict:
    """キャッシュを破棄して再読み込み（編集UI用）。"""
    global _kb_cache
    _kb_cache = None
    return load_kb()


def save_kb(data: dict) -> None:
    """knowledge_base.json を上書き保存する（編集UI用）。"""
    with open(_KB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    reload_kb()


# ---- ヘルパ ---- #

def _venue_match(cond: dict, venue: str) -> bool:
    if "venue" in cond and cond["venue"] != venue:
        return False
    if "venue_list" in cond and venue not in cond["venue_list"]:
        return False
    return True


def _surface_match(cond: dict, surface: str) -> bool:
    if "surface" in cond and cond["surface"] != surface:
        return False
    return True


def _distance_match(cond: dict, distance: int) -> bool:
    if "distance" in cond and cond["distance"] != distance:
        return False
    if "distance_list" in cond and distance not in cond["distance_list"]:
        return False
    if "distance_min" in cond and distance < cond["distance_min"]:
        return False
    if "distance_max" in cond and distance > cond["distance_max"]:
        return False
    return True


def _gate_match(cond: dict, gate: int) -> bool:
    if "gate_number" in cond and gate != cond["gate_number"]:
        return False
    if cond.get("gate_outer") and gate < 7:
        return False
    if cond.get("gate_inner") and gate > 4:
        return False
    return True


def _popularity_match(cond: dict, popularity: int) -> bool:
    if "popularity_min" in cond and popularity < cond["popularity_min"]:
        return False
    return True


def _style_match(cond: dict, running_style: str) -> bool:
    if "running_style" in cond and cond["running_style"] not in (running_style or ""):
        return False
    return True


def _race_class_match(cond: dict, race_class: str) -> bool:
    if "race_class" in cond:
        allowed = [s.strip() for s in cond["race_class"].split(",")]
        if not any(a in (race_class or "") for a in allowed):
            return False
    return True


def _month_match(cond: dict, month: int) -> bool:
    if "month_min" in cond and month < cond["month_min"]:
        return False
    if "month_list" in cond and month not in cond["month_list"]:
        return False
    return True


def _sex_match(cond: dict, sex: str) -> bool:
    if "sex" in cond and cond["sex"] != sex:
        return False
    return True


def _cond_match_all(cond: dict, venue: str, surface: str, distance: int,
                    gate: int, popularity: int, running_style: str,
                    race_class: str, month: int, sex: str,
                    track_condition: str, distance_change: str) -> bool:
    """全条件のAND判定。"""
    if not _venue_match(cond, venue): return False
    if not _surface_match(cond, surface): return False
    if not _distance_match(cond, distance): return False
    if not _gate_match(cond, gate): return False
    if not _popularity_match(cond, popularity): return False
    if not _style_match(cond, running_style): return False
    if not _race_class_match(cond, race_class): return False
    if not _month_match(cond, month): return False
    if not _sex_match(cond, sex): return False
    if "track_condition_bad" in cond and cond["track_condition_bad"]:
        if track_condition not in ("重", "不良"):
            return False
    if "distance_change" in cond:
        if cond["distance_change"] != distance_change:
            return False
    if "surface_change" in cond:
        if cond["surface_change"] != distance_change:  # distance_change 欄に芝→ダート等も入れる
            return False
    return True


# ---- 騎手ボーナス ---- #

def get_jockey_bonus(
    jockey: str,
    venue: str = "",
    surface: str = "",
    distance: int = 0,
    gate: int = 5,
    popularity: int = 5,
    running_style: str = "",
    race_class: str = "",
    month: int = 1,
    sex: str = "",
    track_condition: str = "良",
    distance_change: str = "",
    prev_jockey: str = "",
    trainer: str = "",
) -> dict:
    """
    騎手パターン・乗り替わりパターン・調教師×騎手コンボを照合し、
    合算ボーナスとメモリストを返す。

    Returns
    -------
    {
        "bonus": float,
        "notes": list[str],   # 買い理由
        "avoids": list[str],  # 消し理由
    }
    """
    kb = load_kb()
    total_bonus = 0.0
    notes: list[str] = []
    avoids: list[str] = []

    # 騎手パターン
    for p in kb.get("jockey_patterns", []):
        if p.get("jockey") != jockey:
            continue
        for cond in p.get("conditions", [{}]):
            if _cond_match_all(cond, venue, surface, distance, gate,
                                popularity, running_style, race_class,
                                month, sex, track_condition, distance_change):
                b = p.get("bonus", 0.0)
                total_bonus += b
                if p.get("action") == "avoid":
                    avoids.append(p.get("note", ""))
                else:
                    notes.append(p.get("note", ""))
                break  # 同一騎手の同一エントリは1回だけ

    # 乗り替わりパターン
    for p in kb.get("jockey_change_patterns", []):
        # 前騎手からの乗り替わり
        if "from_jockeys" in p and prev_jockey in p["from_jockeys"]:
            to = p.get("to_jockey")
            if to and to != jockey:
                continue
            total_bonus += p.get("bonus", 0.0)
            notes.append(p.get("note", ""))
        # 特定騎手の2戦目
        if p.get("jockey") == jockey and p.get("condition") == "second_ride":
            if prev_jockey and prev_jockey != jockey:  # 乗り替わり直後
                total_bonus += p.get("bonus", 0.0)
                notes.append(p.get("note", ""))

    # 調教師×騎手コンボ
    for p in kb.get("trainer_jockey_combos", []):
        jockey_ok = (p.get("jockey", jockey) == jockey)
        trainer_ok = (p.get("trainer_contains", "") in trainer) if trainer else False
        if not (jockey_ok and trainer_ok):
            # trainer_contains のみのエントリ（特定騎手なし）
            if "jockey" not in p and trainer_ok:
                pass
            else:
                continue
        conds = p.get("conditions", [{}])
        matched = not conds
        for cond in conds:
            if _cond_match_all(cond, venue, surface, distance, gate,
                                popularity, running_style, race_class,
                                month, sex, track_condition, distance_change):
                matched = True
                break
        if matched:
            total_bonus += p.get("bonus", 0.0)
            notes.append(p.get("note", ""))

    return {"bonus": round(total_bonus, 4), "notes": notes, "avoids": avoids}


# ---- 種牡馬ボーナス ---- #

def get_sire_bonus(
    sire: str,
    venue: str = "",
    surface: str = "",
    distance: int = 0,
    sex: str = "",
    month: int = 1,
    track_condition: str = "良",
    distance_change: str = "",
    race_class: str = "",
    popularity: int = 5,
    horse_age: int = 4,
) -> dict:
    """
    種牡馬パターンと avoid_conditions の sire エントリを照合する。

    Returns
    -------
    {"bonus": float, "note": str}
    """
    kb = load_kb()
    total_bonus = 0.0
    matched_notes: list[str] = []

    # sire_patterns
    for p in kb.get("sire_patterns", []):
        if p.get("sire") != sire:
            continue
        for cond in p.get("conditions", [{}]):
            c = dict(cond)
            if "horse_age_max" in c and horse_age > c["horse_age_max"]:
                continue
            if _cond_match_all(c, venue, surface, distance, 5,
                                popularity, "", race_class,
                                month, sex, track_condition, distance_change):
                total_bonus += p.get("bonus", 0.0)
                matched_notes.append(p.get("note", ""))
                break

    # avoid_conditions にある sire エントリ
    for p in kb.get("avoid_conditions", []):
        if p.get("sire") != sire:
            continue
        v = p.get("venue", "")
        if v and v != venue:
            continue
        total_bonus += p.get("bonus", 0.0)
        matched_notes.append(p.get("note", ""))

    note = " / ".join(matched_notes) if matched_notes else ""
    return {"bonus": round(total_bonus, 4), "note": note}


# ---- 特殊シグナル（G1ブリンカー等） ---- #

def get_special_signal_bonus(
    is_g1_blinker_first: bool = False,
    g1_place_flag: bool = False,
    popularity: int = 5,
    is_foreign_horse: bool = False,
    track_condition: str = "良",
) -> dict:
    """
    特殊シグナルボーナス。

    Returns
    -------
    {"bonus": float, "notes": list[str]}
    """
    kb = load_kb()
    total_bonus = 0.0
    notes: list[str] = []

    for sig in kb.get("special_signals", []):
        sid = sig.get("id", "")
        b = sig.get("bonus", 0.0)
        if sid == "g1_blinker" and is_g1_blinker_first:
            total_bonus += b
            notes.append(sig.get("note", ""))
        elif sid == "g1_connected_cheapside" and g1_place_flag and popularity >= 7:
            total_bonus += b
            notes.append(sig.get("note", ""))
        elif sid == "muddy_foreign_horse" and is_foreign_horse and track_condition in ("重", "不良"):
            total_bonus += b
            notes.append(sig.get("note", ""))

    return {"bonus": round(total_bonus, 4), "notes": notes}


# ---- レース固有パターン ---- #

def get_race_pattern_info(race_name: str, venue: str = "", distance: int = 0) -> dict:
    """
    レース名でパターンを検索し、荒れ警告・チェックリストを返す。

    Returns
    -------
    {
        "is_upset_race": bool,
        "checklist": list[str],
        "notes": list[str],
        "upset_conditions": list[str],
    }
    """
    kb = load_kb()
    checklist: list[str] = []
    notes: list[str] = []
    is_upset = False

    # race_specific_patterns
    for p in kb.get("race_specific_patterns", []):
        rn = p.get("race_name", "")
        if rn and rn in race_name:
            notes.append(p.get("note", ""))
            checklist.extend(p.get("checklist", []))
            if p.get("action") in ("upset_warning", "conditional"):
                is_upset = True

    # upset_race_conditions
    upset_msgs: list[str] = []
    for u in kb.get("upset_race_conditions", []):
        race_match = not u.get("race_name") or u.get("race_name", "") in race_name
        venue_match = not u.get("venue") or u.get("venue") == venue
        dist_match = not u.get("distance") or u.get("distance") == distance
        if race_match and venue_match and dist_match:
            is_upset = True
            upset_msgs.append(u.get("note", ""))

    return {
        "is_upset_race": is_upset,
        "checklist": checklist,
        "notes": notes,
        "upset_conditions": upset_msgs,
    }


# ---- Claude チャット用コンテキスト生成 ---- #

def get_claude_context_for_race(
    horse: dict,
    race_name: str = "",
) -> str:
    """
    1頭分のナレッジベースの該当情報をテキストで返す。
    Claude のシステムプロンプトに埋め込んで使う。
    """
    lines: list[str] = []

    jockey = horse.get("jockey", "")
    sire = horse.get("sire", "")
    venue = horse.get("venue", "")
    surface = horse.get("surface", "")
    distance = int(horse.get("distance", 0) or 0)
    gate = int(horse.get("gate", 5) or 5)
    popularity = int(horse.get("popularity", 5) or 5)
    running_style = horse.get("running_style", "")
    race_class = horse.get("race_class", "")
    sex = horse.get("sex", "")
    month = int(horse.get("race_month", 1) or 1)
    track_condition = horse.get("track_condition", "良")
    distance_change = horse.get("distance_change", "")
    prev_jockey = horse.get("prev_jockey", "")
    trainer = horse.get("trainer", "")
    horse_name = horse.get("horse_name", "？")

    # 騎手
    if jockey:
        jr = get_jockey_bonus(jockey, venue, surface, distance, gate,
                               popularity, running_style, race_class,
                               month, sex, track_condition, distance_change,
                               prev_jockey, trainer)
        for n in jr["notes"]:
            lines.append(f"[KB騎手メモ] {horse_name}（{jockey}）: {n}")
        for a in jr["avoids"]:
            lines.append(f"[KB騎手消し] {horse_name}（{jockey}）: {a}")

    # 種牡馬
    if sire:
        sr = get_sire_bonus(sire, venue, surface, distance, sex, month,
                             track_condition, distance_change, race_class, popularity)
        if sr["note"]:
            lines.append(f"[KB血統メモ] {horse_name}（{sire}産駒）: {sr['note']}")

    # レース固有
    if race_name:
        rp = get_race_pattern_info(race_name, venue, distance)
        for n in rp["notes"]:
            lines.append(f"[KBレースメモ] {race_name}: {n}")
        if rp["checklist"]:
            lines.append(f"[KBチェックリスト] {race_name}: " + " / ".join(rp["checklist"]))
        for u in rp["upset_conditions"]:
            lines.append(f"[KB荒れ警告] {u}")

    return "\n".join(lines)


def get_claude_context_for_all_horses(
    horses: list[dict],
    race_name: str = "",
) -> str:
    """出走全頭分のKBコンテキストをまとめて返す。"""
    parts = []
    # レース固有は一度だけ
    if race_name and horses:
        h = horses[0]
        venue = h.get("venue", "")
        distance = int(h.get("distance", 0) or 0)
        rp = get_race_pattern_info(race_name, venue, distance)
        if rp["notes"]:
            parts.append("[KBレースパターン] " + " / ".join(rp["notes"]))
        if rp["checklist"]:
            parts.append("[KBチェックリスト] " + " / ".join(rp["checklist"]))
        for u in rp["upset_conditions"]:
            parts.append(f"[KB荒れ警告] {u}")

    for h in horses:
        ctx = get_claude_context_for_race(h, race_name="")  # レース固有は上で済ませた
        if ctx:
            parts.append(ctx)
    return "\n".join(parts)


# ---- 1頭への総合KBボーナス（ev_calculator から呼ぶ） ---- #

def apply_kb_to_horse(horse: dict, race_name: str = "") -> dict:
    """
    horse dict に kb_bonus / kb_notes / kb_avoids を付与して返す。
    ev_calculator.py の evaluate_horse() 内で使う。
    """
    h = dict(horse)
    jockey = h.get("jockey", "")
    sire = h.get("sire", "")
    venue = h.get("venue", "")
    surface = h.get("surface", "")
    distance = int(h.get("distance", 0) or 0)
    gate = int(h.get("gate", 5) or 5)
    popularity = int(h.get("popularity", 5) or 5)
    running_style = h.get("running_style", "")
    race_class = h.get("race_class", "")
    sex = h.get("sex", "")
    month = int(h.get("race_month", 1) or 1)
    track_condition = h.get("track_condition", "良")
    distance_change = h.get("distance_change", "")
    prev_jockey = h.get("prev_jockey", "")
    trainer = h.get("trainer", "")

    total_bonus = 0.0
    all_notes: list[str] = []
    all_avoids: list[str] = []

    # 騎手
    if jockey:
        jr = get_jockey_bonus(jockey, venue, surface, distance, gate,
                               popularity, running_style, race_class,
                               month, sex, track_condition, distance_change,
                               prev_jockey, trainer)
        total_bonus += jr["bonus"]
        all_notes.extend(jr["notes"])
        all_avoids.extend(jr["avoids"])

    # 種牡馬
    if sire:
        sr = get_sire_bonus(sire, venue, surface, distance, sex, month,
                             track_condition, distance_change, race_class, popularity)
        total_bonus += sr["bonus"]
        if sr["note"]:
            all_notes.append(sr["note"])

    # 特殊シグナル
    sig = get_special_signal_bonus(
        is_g1_blinker_first=h.get("blinker_first", False),
        g1_place_flag=h.get("g1_place_flag", False),
        popularity=popularity,
        is_foreign_horse=h.get("is_foreign_horse", False),
        track_condition=track_condition,
    )
    total_bonus += sig["bonus"]
    all_notes.extend(sig["notes"])

    h["kb_bonus"] = round(total_bonus, 4)
    h["kb_notes"] = all_notes
    h["kb_avoids"] = all_avoids
    return h


# ---- 短期外国人騎手チェック ---- #

def get_short_term_foreign_jockey_bonus(
    jockey: str,
    race_class: str = "",
) -> dict:
    """
    短期免許外国人騎手が重賞・G1〜G3に乗る場合のボーナスを返す。

    Returns
    -------
    {"bonus": float, "is_short_term": bool, "is_resident": bool, "note": str}
    """
    kb = load_kb()
    stf = kb.get("short_term_foreign_jockeys", {})
    notable = stf.get("notable", [])
    graded_bonus = stf.get("graded_race_bonus", 0.03)

    matched = next((j for j in notable if j.get("name", "") == jockey), None)
    if not matched:
        return {"bonus": 0.0, "is_short_term": False, "is_resident": False, "note": ""}

    is_resident = matched.get("resident", False)
    # 在留騎手（ルメール・デムーロ）は別途 jockey_patterns で評価されるため、ここでは加算しない
    if is_resident:
        return {"bonus": 0.0, "is_short_term": False, "is_resident": True,
                "note": matched.get("note", "")}

    # 重賞・G1〜G3 出走時のみボーナス
    is_graded = any(kw in (race_class or "") for kw in ["G1", "G2", "G3", "重賞", "GI", "GII", "GIII"])
    if not is_graded:
        return {"bonus": 0.0, "is_short_term": True, "is_resident": False,
                "note": matched.get("note", "")}

    note = f"短期外国人騎手（{jockey}）重賞乗り。{matched.get('note', '')} 市場が過小評価しがちで割安感あり。"
    return {
        "bonus":          graded_bonus,
        "is_short_term":  True,
        "is_resident":    False,
        "note":           note,
    }


# ---- レース格言チェック ---- #

def get_proverb_bonus(
    race_name: str,
    prev_distance: int = 0,
    prev_surface: str = "",
    is_fastest_3f_prev: bool = False,
    prev_rank: int = 99,
) -> dict:
    """
    ハロン棒chや格言に基づくレース固有のボーナスを返す。
    例: VMは前走1400mで強い競馬をした馬が有利。

    Returns
    -------
    {"bonus": float, "label": str, "checklist": list[str]}
    """
    kb = load_kb()
    proverbs = kb.get("race_proverbs", [])

    for p in proverbs:
        names = [p.get("race_name", "")] + p.get("alias", [])
        if not any(n in race_name for n in names if n):
            continue

        bonus = p.get("bonus", 0.0)
        label = p.get("proverb", "")
        checklist = p.get("checklist", [])

        # VM専用: 前走1400mチェック
        if "VM" in (p.get("race_name", "")) or "ヴィクトリアマイル" in race_name:
            if prev_distance == 1400:
                strength = ""
                if prev_rank <= 2:
                    strength = "前走1400m好走"
                elif is_fastest_3f_prev:
                    strength = "前走1400m上がり最速"
                if strength:
                    return {
                        "bonus":     bonus,
                        "label":     f"◎ VM格言一致（{strength}）",
                        "checklist": checklist,
                    }
            # 1400m実績なし → 小ボーナスなし
            return {"bonus": 0.0, "label": "", "checklist": checklist}

        # 汎用格言：チェックリスト表示のみ（ボーナスは基本与えない）
        return {"bonus": 0.0, "label": label, "checklist": checklist}

    return {"bonus": 0.0, "label": "", "checklist": []}
