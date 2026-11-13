"""The syllable tier, transcription systems and phoneme packs (idtap-contract 0.3.0).

PROP-7 ``Piece.system``, PROP-8 ``Phrase.syllables`` and PROP-9 phoneme packs.

This module is a port of the contract's reference rules
(``idtap-contract/conformance/syllables.py``). Like the reference, it works on
plain serialized JSON (camelCase dicts), not on model classes, so its behaviour
can be checked line by line against the spec. ``Piece`` builds a light JSON view
of itself and calls these functions (see ``Piece.heal_syllables``,
``Piece.project_syllables`` and friends).

Terms
  track       phraseGrid[k]: one phrase sequence (one singer).
  row0(ph)    ph["trajectoryGrid"][0], the melodic row and the only row on a
              vocal track. Syllables live on row 0 only.
  silence     a trajectory with id == 12.
  absent      a legacy field that is missing OR null. Null means absent
              (PROP-8, DIVERGENCES SYL-1).

Packs are bundled as data in ``idtap/packs/*.json``, copied verbatim from the
contract's ``packs/`` directory. Their tables are never copied into code.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

PACKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packs")

HINDUSTANI: Dict[str, Any] = {"pitch": "raga", "phonology": "hindi-urdu", "rhythm": "tala",
                              "display": ["sargam", "Deva"]}
LEGACY_PACK = "hindi-urdu"
# Instrumentation labels that count as a vocal track, pack-independent (the
# same list as idtap-platform and idtap.enums.Instrument.Vocal_M / Vocal_F).
VOCAL_INSTRUMENTS = ("Vocal (M)", "Vocal (F)")
# Invariant codes reported at warn level: shown, never refused.
WARNINGS = {"track-not-vocal"}
WIRE_FIELDS = ["vowel",
               "startConsonant", "startConsonantHindi", "startConsonantIpa", "startConsonantEngTrans",
               "endConsonant", "endConsonantHindi", "endConsonantIpa", "endConsonantEngTrans"]
# Fields the legacy projection can carry; anything else on a syllable makes the
# piece unsafe for a pre-0.3.0 writer.
LEGACY_HOLDABLE = {"anchor", "onset", "nucleus", "coda", "lyric"}

_PACKS: Dict[str, Dict[str, Any]] = {}


class UnknownPackError(ValueError):
    """``system.phonology`` names a pack this version of idtap does not bundle."""


def available_packs() -> List[str]:
    """Ids of the bundled phoneme packs."""
    return sorted(f[:-5] for f in os.listdir(PACKS_DIR) if f.endswith(".json"))


def load_pack(pid: str) -> Dict[str, Any]:
    """The bundled pack ``pid`` (cached; do not mutate the returned dict)."""
    if pid not in _PACKS:
        path = os.path.join(PACKS_DIR, pid + ".json")
        if not os.path.exists(path):
            raise UnknownPackError(
                f"phoneme pack {pid!r} is not bundled with this version of idtap "
                f"(bundled: {', '.join(available_packs())})")
        with open(path, encoding="utf-8") as f:
            _PACKS[pid] = json.load(f)
    return _PACKS[pid]


# --------------------------------------------------------------------- PROP-7
def resolve_system(piece: Dict[str, Any]) -> Dict[str, Any]:
    """Absent system = Hindustani; each missing component takes its default;
    unknown components are kept verbatim."""
    out = dict(HINDUSTANI)
    out.update(piece.get("system") or {})
    return out


def pack_of(piece: Dict[str, Any]) -> Dict[str, Any]:
    return load_pack(resolve_system(piece)["phonology"])


def codas_of(pack: Dict[str, Any]) -> Dict[str, Any]:
    return pack["consonants"] if pack["codas"] == "consonants" else pack["codas"]


# --------------------------------------------------------------------- helpers
def row0(ph: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid = ph.get("trajectoryGrid") or [[]]
    return grid[0] if grid else []


def silent(t: Dict[str, Any]) -> bool:
    return t.get("id") == 12


def val(t: Dict[str, Any], k: str) -> Any:
    v = t.get(k)
    return None if v is None else v


def has_phonetics(t: Dict[str, Any]) -> bool:
    return any(val(t, k) is not None for k in ("vowel", "startConsonant", "endConsonant"))


def uid_of(t: Dict[str, Any]) -> str:
    """A trajectory's uniqueId. Heal, projection and re-homing need one on every
    row-0 trajectory: loaders assign ids before healing (PROP-8 heal step 1)."""
    u = t.get("uniqueId")
    if not isinstance(u, str) or not u:
        raise ValueError("trajectory without uniqueId: assign ids before heal/project/rehome")
    return u


def track_flat(track: List[Dict[str, Any]]) -> List[Tuple[int, int, Dict[str, Any]]]:
    """[(phrase_idx, traj_idx, traj)] over row 0 of trajectoryGrid, in order."""
    return [(pi, ti, t) for pi, ph in enumerate(track) for ti, t in enumerate(row0(ph))]


# --------------------------------------------------------------------- heal
def first_traj_idxs(trajs: List[Dict[str, Any]]) -> List[int]:
    """Phrase.firstTrajIdxs, with null == absent."""
    idxs: List[int] = []
    ct, silent_trigger = 0, False
    last_vowel, end_trigger = None, False
    for i, t in enumerate(trajs):
        if not silent(t):
            c1 = ct == 0
            c3 = val(t, "startConsonant") is not None
            c5 = val(t, "vowel") != last_vowel
            if c1 or silent_trigger or c3 or end_trigger or c5:
                idxs.append(i)
            ct += 1
            end_trigger = val(t, "endConsonant") is not None
            last_vowel = val(t, "vowel")
        silent_trigger = silent(t)
    return idxs


def heal_phrase(ph: Dict[str, Any]) -> List[Dict[str, Any]]:
    trajs = row0(ph)
    idxs = first_traj_idxs(trajs)
    out = []
    for n, i in enumerate(idxs):
        stop = idxs[n + 1] if n + 1 < len(idxs) else len(trajs)
        span = []
        for t in trajs[i:stop]:
            if silent(t):
                break
            span.append(t)
        first, last = span[0], span[-1]
        syl: Dict[str, Any] = {"anchor": {"traj": uid_of(first), "t": 0}}
        if val(first, "startConsonant") is not None:
            syl["onset"] = first["startConsonant"]
        vowel = val(first, "vowel")
        if vowel is not None:
            syl["nucleus"] = vowel
        if val(last, "endConsonant") is not None:
            syl["coda"] = last["endConsonant"]
        syl["lyric"] = {"vowel": vowel, "source": "pack"}
        out.append(syl)
    return out


def track_is_phonetic(track: List[Dict[str, Any]]) -> bool:
    return any(has_phonetics(t) for _, _, t in track_flat(track))


def heal_piece(piece: Dict[str, Any]) -> Dict[str, Any]:
    """Normative heal. A phrase that already has `syllables` keeps them (even []).
    A phrase without the key is healed from its legacy fields; if its track has
    no phonetic data on any row-0 trajectory, it gets []. Returns a copy."""
    pc = copy.deepcopy(piece)
    for track in pc["phraseGrid"]:
        phonetic = track_is_phonetic(track)
        for ph in track:
            if "syllables" not in ph:
                ph["syllables"] = heal_phrase(ph) if phonetic else []
    return pc


# --------------------------------------------------------------------- projection
def _legacy_entry(pack: Dict[str, Any], table: str, key: Any) -> Optional[Tuple[Any, Any, Any, Any]]:
    """-> (iso, hindi, ipa, engTrans) written into the legacy fields for `key`,
    or None when the legacy fields cannot carry it."""
    if key is None or isinstance(key, list):
        return None  # clusters are reduced by the caller
    if pack["id"] == LEGACY_PACK:
        e = pack[table].get(key) if table != "codas" else codas_of(pack).get(key)
        leg = (e or {}).get("legacy")
        if e is None:
            # unknown stored string: today's code keeps it and its copies are undefined
            return (key, None, None, None)
        if leg is None:
            return None  # e.g. the new retroflex ṇa: no legacy form
        return (leg["iso"], leg.get("hindi"), leg.get("ipa"), leg.get("engTrans"))
    # Any other pack projects nothing (Jon, 2026-11-12): all platforms move to
    # the tier together; the write guard refuses pre-0.3.0 saves of such pieces.
    return None


def legacy_vowel_entry(pack: Dict[str, Any], key: Any) -> Optional[Tuple[Any, Any, Any, Any]]:
    """(iso, hindi, ipa, engTrans) the projection writes for a vowel key."""
    return _legacy_entry(pack, "vowels", key)


def _cluster_head(key: Any, onset: bool) -> Any:
    if isinstance(key, list):
        return key[0] if onset else key[-1]
    return key


def spans(track: List[Dict[str, Any]]):
    """[(syllable, [traj, ...])] in track order, and the list of uncovered
    non-silent trajs. A syllable runs from its anchor to the next anchor or a
    silence, along the track's row 0, so it can cross a phrase boundary."""
    anchors = {}
    for ph in track:
        for s in ph.get("syllables") or []:
            anchors[s["anchor"]["traj"]] = s
    out: List[Any] = []
    uncovered: List[Dict[str, Any]] = []
    cur = None
    for _, _, t in track_flat(track):
        if silent(t):
            cur = None
            continue
        uid = uid_of(t)
        if uid in anchors:
            cur = (anchors[uid], [])
            out.append(cur)
        if cur is None:
            uncovered.append(t)
        else:
            cur[1].append(t)
    return out, uncovered


def project_piece(piece: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normative projection syllables -> legacy fields. Returns
    {trajUniqueId: {wire field: value, ..., "art0": {...}|None, "art1": {...}|None}}
    for every row-0 trajectory; absent fields are omitted, "art0"/"art1"
    are the 'consonant' articulations at '0.00' / '1.00' (None = none)."""
    pack = pack_of(piece)
    res: Dict[str, Dict[str, Any]] = {}
    for track in piece["phraseGrid"]:
        sp, _uncovered = spans(track)
        for _, _, t in track_flat(track):
            res[uid_of(t)] = {"art0": None, "art1": None}
        for syl, trajs in sp:
            v = _legacy_entry(pack, "vowels", (syl.get("lyric") or {}).get("vowel"))
            on = _legacy_entry(pack, "consonants", _cluster_head(syl.get("onset"), True))
            co = _legacy_entry(pack, "codas", _cluster_head(syl.get("coda"), False))
            for t in trajs:
                if v:
                    res[t["uniqueId"]]["vowel"] = v[0]
            for c, where, pre, art in ((on, trajs[0], "startConsonant", "art0"),
                                       (co, trajs[-1], "endConsonant", "art1")):
                if not c:
                    continue
                r = res[where["uniqueId"]]
                for suffix, x in zip(("", "Hindi", "Ipa", "EngTrans"), c):
                    if x is not None:
                        r[pre + suffix] = x
                a = {"name": "consonant", "stroke": c[0]}
                for k, x in zip(("hindi", "ipa", "engTrans"), c[1:]):
                    if x is not None:
                        a[k] = x
                r[art] = a
    return res


def legacy_of_traj(t: Dict[str, Any]) -> Dict[str, Any]:
    """The legacy fields as stored on a trajectory, in project_piece's shape."""
    r = {k: t[k] for k in WIRE_FIELDS if t.get(k) is not None}
    arts = t.get("articulations") or {}
    for key, name in (("0.00", "art0"), ("1.00", "art1")):
        a = arts.get(key)
        r[name] = dict(a) if a and a.get("name") == "consonant" else None
    return r


def apply_projection(piece: Dict[str, Any]) -> Dict[str, Any]:
    """Write the projection onto a copy of the serialized piece (dual-write)."""
    pc = copy.deepcopy(piece)
    proj = project_piece(pc)
    for track in pc["phraseGrid"]:
        for _, _, t in track_flat(track):
            p = proj[uid_of(t)]
            for k in WIRE_FIELDS:
                t.pop(k, None)
                if k in p:
                    t[k] = p[k]
            had = t.get("articulations")
            arts = dict(had or {})
            for key, name in (("0.00", "art0"), ("1.00", "art1")):
                if (arts.get(key) or {}).get("name") == "consonant":
                    del arts[key]
                if p[name]:
                    arts[key] = p[name]
            if arts or isinstance(had, dict):
                t["articulations"] = arts
    return pc


# --------------------------------------------------------------------- derived
def final_key(syl: Dict[str, Any]) -> Optional[str]:
    """Key into pack["finals"] (nucleus + coda), or None when the coda is a
    cluster (no finals table lists clusters; check_invariants reports it)."""
    co = syl.get("coda")
    if isinstance(co, list):
        return None
    return (syl.get("nucleus") or "") + (co or "")


def _keys(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x] if x is not None else []


def derive_ipa(syl: Dict[str, Any], pack: Dict[str, Any]) -> Optional[str]:
    """Onset IPA + the final's realization where the pack has `finals`
    (Jyutping), else nucleus + coda IPA. IPA is derived, never stored.
    Returns None when any part is not in the pack: never raises, the
    violation is check_invariants' to report."""
    parts = []
    for k in _keys(syl.get("onset")):
        e = pack["consonants"].get(k)
        if e is None:
            return None
        parts.append(e["ipa"])
    if pack.get("finals") is not None:
        fk = final_key(syl)
        f = pack["finals"].get(fk) if fk is not None else None
        if f is None:
            return None
        parts.append(f["ipa"])
    else:
        if syl.get("nucleus") is not None:
            e = pack["vowels"].get(syl["nucleus"])
            if e is None:
                return None
            parts.append(e.get("ipa", ""))
        for k in _keys(syl.get("coda")):
            e = codas_of(pack).get(k)
            if e is None:
                return None
            parts.append(e["ipa"])
    return "".join(parts)


def derived_tone(syl: Dict[str, Any], pack: Dict[str, Any], system: str = "nine") -> Optional[str]:
    """A tone number in one of the pack's toneSystems (Jyutping 'nine'):
    the stored tone, mapped when the syllable is checked. None without a tone."""
    tone = syl.get("tone")
    ts = (pack.get("toneSystems") or {}).get(system)
    if tone is None or ts is None:
        return tone
    coda = syl.get("coda")
    if isinstance(coda, str) and coda in ts["checkedCodas"]:
        return ts["checked"].get(tone, tone)
    return tone


# --------------------------------------------------------------------- invariants
def check_invariants(piece: Dict[str, Any]) -> List[Tuple[str, str]]:
    """-> sorted list of (code, location). Codes in WARNINGS are warn level;
    every other code is an error. See split_violations."""
    pack = pack_of(piece)
    codas = codas_of(pack)
    v: List[Tuple[str, str]] = []
    for k, track in enumerate(piece["phraseGrid"]):
        any_syl = any(ph.get("syllables") for ph in track)
        anchored = set()
        for pi, ph in enumerate(track):
            idx = {}
            for i, t in enumerate(row0(ph)):
                u = t.get("uniqueId")
                if isinstance(u, str) and u:
                    idx[u] = (i, t)
                else:
                    v.append(("traj-uid-missing", f"track {k} phrase {pi} traj {i}"))
            last = -1
            for si, s in enumerate(ph.get("syllables") or []):
                where = f"track {k} phrase {pi} syllable {si}"
                a = s["anchor"]
                if a["traj"] not in idx:
                    v.append(("anchor-missing", where))
                    continue
                i, t = idx[a["traj"]]
                if silent(t):
                    v.append(("anchor-silence", where))
                if i <= last:
                    v.append(("anchor-order", where))
                last = i
                anchored.add(a["traj"])
                if a.get("t", 0) != 0:
                    v.append(("anchor-t", where))
                for f, table, flag in (("onset", pack["consonants"], "onsetClusters"),
                                       ("coda", codas, "codaClusters")):
                    x = s.get(f)
                    if isinstance(x, list) and not pack["flags"][flag]:
                        v.append(("cluster-not-allowed", where))
                    for key in (x if isinstance(x, list) else [x] if x is not None else []):
                        if key not in table:
                            v.append(("key-unknown", where))
                nuc = s.get("nucleus")
                if nuc is not None and nuc not in pack["vowels"]:
                    v.append(("key-unknown", where))
                # editors keep nucleus and lyric.vowel in sync; absent == null
                if nuc != (s.get("lyric") or {}).get("vowel"):
                    v.append(("nucleus-lyric-mismatch", where))
                for obs in ("lyric", "sung"):
                    ov = (s.get(obs) or {}).get("vowel")
                    if ov is not None and ov not in pack["vowels"]:
                        v.append(("key-unknown", where))
                if s.get("tone") is not None:
                    if not pack["flags"]["tones"]:
                        v.append(("tone-not-allowed", where))
                    elif s["tone"] not in pack["tones"]:
                        v.append(("key-unknown", where))
                if pack.get("finals") is not None:
                    fk = final_key(s)
                    f = pack["finals"].get(fk) if fk is not None else None
                    if fk is None:
                        pass  # coda cluster: already reported as cluster-not-allowed
                    elif f is None:
                        v.append(("final-unknown", where))
                    elif f.get("syllabic") and not pack["flags"]["syllabicNasals"]:
                        v.append(("syllabic-not-allowed", where))
                elif nuc is None and not pack["flags"]["nucleusOptional"]:
                    v.append(("nucleus-missing", where))
        instr = (piece.get("instrumentation") or [])
        if any_syl and k < len(instr) and instr[k] not in VOCAL_INSTRUMENTS:
            v.append(("track-not-vocal", f"track {k} ({instr[k]})"))
        if any_syl:
            prev_silent = True  # the track start counts as a silence
            for pi, ti, t in track_flat(track):
                if not silent(t) and prev_silent and (t.get("uniqueId") or None) not in anchored:
                    v.append(("unanchored-start", f"track {k} phrase {pi} traj {ti}"))
                prev_silent = silent(t)
    return sorted(v)


def split_violations(found: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """-> (error codes, warning codes), each sorted."""
    codes = [c for c, _ in found]
    return ([c for c in codes if c not in WARNINGS], [c for c in codes if c in WARNINGS])


# --------------------------------------------------------------------- fixture helpers
def syllable_grid(piece: Dict[str, Any]) -> List[List[Optional[List[Dict[str, Any]]]]]:
    """phraseGrid-shaped list of each phrase's syllables (None = key absent)."""
    return [[ph.get("syllables") for ph in tr] for tr in piece["phraseGrid"]]


def stored_legacy(piece: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{uid: legacy_of_traj(t)} for every row-0 trajectory, as stored."""
    return {uid_of(t): legacy_of_traj(t)
            for tr in piece["phraseGrid"] for _, _, t in track_flat(tr)}


# --------------------------------------------------------------------- write guard
def _comparable(syls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Syllables as the legacy fields can express them (lyric.source dropped)."""
    out = []
    for s in syls:
        c = {"traj": s["anchor"]["traj"], "t": s["anchor"].get("t", 0)}
        for f in ("onset", "nucleus", "coda"):
            if s.get(f) is not None:
                c[f] = s[f]
        c["vowel"] = (s.get("lyric") or {}).get("vowel")
        out.append(c)
    return out


def legacy_safe(piece: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """PROP-8 write guard, evaluated on the STORED piece. True when a pre-0.3.0
    client may save it: dropping its syllables loses nothing, because healing
    the stored legacy fields rebuilds exactly the stored syllables.
    -> (bool, sorted reason codes). The server enforces this; clients may use
    it to explain a refusal."""
    reasons = set()
    pack = pack_of(piece)
    if pack["id"] != LEGACY_PACK:
        reasons.add("pack")
    for track in piece["phraseGrid"]:
        for ph in track:
            for s in ph.get("syllables") or []:
                for f in s:
                    if f not in LEGACY_HOLDABLE:
                        reasons.add(f)
                if s["anchor"].get("t", 0) != 0:
                    reasons.add("anchorT")
                for f in ("onset", "coda"):
                    if isinstance(s.get(f), list):
                        reasons.add("cluster")
                if pack["id"] == LEGACY_PACK:
                    for k, table in ((s.get("onset"), pack["consonants"]),
                                     (s.get("coda"), codas_of(pack)),
                                     (s.get("nucleus"), pack["vowels"]),
                                     ((s.get("lyric") or {}).get("vowel"), pack["vowels"])):
                        if isinstance(k, str) and k in table and "legacy" not in table[k]:
                            reasons.add("key")
    if not reasons:
        rebuilt = heal_piece(strip_syllables(piece))
        for ta, tb in zip(piece["phraseGrid"], rebuilt["phraseGrid"]):
            for pa, pb in zip(ta, tb):
                # A phrase with no stored tier (absent or null) is not compared:
                # a pre-0.3.0 save keeps its legacy fields and drops nothing.
                if pa.get("syllables") is None:
                    continue
                if _comparable(pa["syllables"]) != _comparable(pb["syllables"]):
                    reasons.add("boundary")
    return (not reasons, sorted(reasons))


def strip_syllables(piece: Dict[str, Any]) -> Dict[str, Any]:
    """What a pre-0.3.0 client's save leaves: the syllable tier dropped."""
    pc = copy.deepcopy(piece)
    for track in pc["phraseGrid"]:
        for ph in track:
            ph.pop("syllables", None)
    return pc


# --------------------------------------------------------------------- re-homing
def rehome(before: Dict[str, Any], after: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Run after every phrase split, merge or moved division. `before` is the
    piece as it was before the edit, `after` the piece the edit produced
    (whatever its phrases' syllable lists now hold is ignored). Every syllable
    of a track before the edit moves to the phrase of that track that now
    holds its anchor trajectory, in trajectory order. Syllables whose
    trajectory no longer exists are dropped. -> (piece copy, dropped anchor uids)."""
    pc = copy.deepcopy(after)
    dropped: List[str] = []
    for k, track in enumerate(pc["phraseGrid"]):
        where = {uid_of(t): (pi, ti) for pi, ti, t in track_flat(track)}
        old = before["phraseGrid"][k] if k < len(before["phraseGrid"]) else []
        pool = [s for ph in old for s in ph.get("syllables") or []]
        buckets: List[List[Any]] = [[] for _ in track]
        for s in pool:
            loc = where.get(s["anchor"]["traj"])
            if loc is None:
                dropped.append(s["anchor"]["traj"])
                continue
            buckets[loc[0]].append((loc[1], s))
        for ph, b in zip(track, buckets):
            ph["syllables"] = [s for _, s in sorted(b, key=lambda x: x[0])]
    return pc, dropped
