"""Syllable tier behaviour beyond the contract fixtures (idtap-contract 0.3.0,
PROP-7/8/9): the feature flag ("preserve, never create"), null == absent,
uniqueId stability, re-homing after Python's own trajectory edits, edits made
through the legacy fields of a tiered phrase, unknown packs, and the dry-run
harness detecting a real difference."""
import copy
import json
import os
import subprocess
import sys
import warnings

import pytest

from idtap import syllables as SYL
from idtap.classes.piece import Piece
from idtap.classes.trajectory import Trajectory

PITCH = {"swara": 0, "raised": True, "oct": 0, "logOffset": 0.0}
RAGA = {"name": "Yaman", "fundamental": 246.0,
        "ratios": [1, 1.122462048309373, 1.2599210498948732, 1.4142135623730951,
                   1.4983070768766815, 1.681792830507429, 1.8877486253633868]}


def traj(uid, tid=0, **legacy):
    t = {"id": tid, "pitches": [PITCH], "durTot": 0.5, "durArray": [1.0], "uniqueId": uid,
         "articulations": {}}
    t.update(legacy)
    if "startConsonant" in legacy and legacy["startConsonant"] is not None:
        t["articulations"]["0.00"] = {"name": "consonant", "stroke": legacy["startConsonant"]}
    if "endConsonant" in legacy and legacy["endConsonant"] is not None:
        t["articulations"]["1.00"] = {"name": "consonant", "stroke": legacy["endConsonant"]}
    return t


def phrase(uid, trajs, syllables=None, start=0.0):
    ph = {"durTot": 0.5 * len(trajs), "durArray": [1 / len(trajs)] * len(trajs), "startTime": start,
          "trajectoryGrid": [trajs], "instrumentation": ["Vocal (F)"], "uniqueId": uid,
          "groupsGrid": [[]], "categorizationGrid": [], "adHocCategorizationGrid": []}
    if syllables is not None:
        ph["syllables"] = syllables
    return ph


def piece(phrases, **top):
    pj = {"raga": RAGA, "instrumentation": ["Vocal (F)"], "phraseGrid": [phrases],
          "title": "t", "_id": "p1"}
    pj.update(top)
    return pj


def load(pj):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Piece.from_json(copy.deepcopy(pj))


def save(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.to_json()


def legacy_doc():
    return piece([phrase("ph0", [traj("a1", vowel="ā", startConsonant="ba"),
                                 traj("a2", vowel="ā"),
                                 traj("a3", vowel="u", startConsonant="ga", endConsonant="la")])])


SYL_A1 = {"anchor": {"traj": "a1", "t": 0}, "onset": "ba", "nucleus": "ā",
          "lyric": {"vowel": "ā", "source": "human"}, "futureSyllableKey": {"camelInner": 1}}


# --------------------------------------------------------------------------- flag
def test_flag_off_heals_in_memory_but_never_writes_the_tier():
    p = load(legacy_doc())
    assert not p.syllable_tier_enabled
    assert [s["anchor"]["traj"] for s in p.phrase_grid[0][0].syllables] == ["a1", "a3"]
    assert not p.phrase_grid[0][0].has_stored_syllables
    out = save(p)
    assert "syllables" not in out["phraseGrid"][0][0]
    assert "system" not in out


def test_flag_on_writes_the_healed_tier():
    p = load(piece(legacy_doc()["phraseGrid"][0], system={"phonology": "hindi-urdu"}))
    assert p.syllable_tier_enabled
    out = save(p)
    assert [s["anchor"]["traj"] for s in out["phraseGrid"][0][0]["syllables"]] == ["a1", "a3"]


def test_stored_tier_is_kept_byte_identical_with_the_flag_off():
    stored = [SYL_A1, {"anchor": {"traj": "a3", "t": 0}, "onset": "ga", "nucleus": "u", "coda": "la",
                       "lyric": {"vowel": "u", "source": "pack"}}]
    doc = legacy_doc()
    doc["phraseGrid"][0][0]["syllables"] = copy.deepcopy(stored)
    out = save(load(doc))
    assert out["phraseGrid"][0][0]["syllables"] == stored  # unknown syllable key too
    assert json.dumps(out["phraseGrid"][0][0]["syllables"], ensure_ascii=False) == \
        json.dumps(stored, ensure_ascii=False)


def test_null_syllables_are_absent():
    doc = legacy_doc()
    doc["phraseGrid"][0][0]["syllables"] = None
    p = load(doc)
    assert not p.phrase_grid[0][0].has_stored_syllables
    assert "syllables" not in save(p)["phraseGrid"][0][0]


def test_null_legacy_fields_are_absent_in_the_heal():
    """SYL-1: a stored null consonant does not start a syllable."""
    doc = piece([phrase("ph0", [traj("a1", vowel="ā", startConsonant="ba"),
                                traj("a2", vowel="ā", startConsonant=None, endConsonant=None)])])
    p = load(doc)
    assert [s["anchor"]["traj"] for s in p.phrase_grid[0][0].syllables] == ["a1"]


def test_assigning_syllables_makes_them_part_of_the_document():
    p = load(legacy_doc())
    p.phrase_grid[0][0].syllables = [copy.deepcopy(SYL_A1)]
    assert p.phrase_grid[0][0].has_stored_syllables
    assert save(p)["phraseGrid"][0][0]["syllables"] == [SYL_A1]


def test_adopt_writes_and_projects_the_healed_tier():
    p = load(legacy_doc())
    p.adopt_healed_syllables()
    out = save(p)
    assert len(out["phraseGrid"][0][0]["syllables"]) == 2
    assert SYL.stored_legacy(out) == SYL.stored_legacy(save(load(legacy_doc())))


def test_empty_list_is_a_healed_tier_with_no_syllables():
    doc = legacy_doc()
    doc["phraseGrid"][0][0]["syllables"] = []
    out = save(load(doc))
    assert out["phraseGrid"][0][0]["syllables"] == []
    # the tier is the source of truth: no syllables, so no legacy fields
    assert all(v == {"art0": None, "art1": None} for v in SYL.stored_legacy(out).values())


# --------------------------------------------------------------------------- projection
def test_projection_updates_the_in_memory_trajectories():
    doc = legacy_doc()
    doc["phraseGrid"][0][0]["syllables"] = [
        {"anchor": {"traj": "a1", "t": 0}, "onset": "ma", "nucleus": "ī", "lyric": {"vowel": "ī", "source": "human"}}]
    p = load(doc)
    t1, t2, t3 = p.phrase_grid[0][0].trajectories
    assert (t1.vowel, t1.start_consonant, t1.start_consonant_hindi) == ("ī", "ma", "म")
    assert t1.vowel_hindi == "ई"
    assert t1.articulations["0.00"].stroke == "ma"
    assert (t3.vowel, t3.start_consonant, t3.end_consonant) == ("ī", None, None)
    assert "0.00" not in t3.articulations and "1.00" not in t3.articulations


def test_editing_legacy_fields_of_a_tiered_phrase_warns():
    doc = legacy_doc()
    doc["phraseGrid"][0][0]["syllables"] = [copy.deepcopy(SYL_A1)]
    p = load(doc)
    p.phrase_grid[0][0].trajectories[1].update_vowel("ō")
    with pytest.warns(UserWarning, match="edit phrase.syllables instead|Edit phrase.syllables"):
        out = p.to_json()
    assert out["phraseGrid"][0][0]["trajectoryGrid"][0][1]["vowel"] == "ā"


def test_editing_a_syllable_changes_the_saved_legacy_fields():
    doc = legacy_doc()
    doc["phraseGrid"][0][0]["syllables"] = [copy.deepcopy(SYL_A1)]
    p = load(doc)
    s = p.phrase_grid[0][0].syllables[0]
    s["nucleus"] = "ō"
    s["lyric"] = {"vowel": "ō", "source": "human"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "edited legacy field" warning
        out = p.to_json()
    assert [t.get("vowel") for t in out["phraseGrid"][0][0]["trajectoryGrid"][0]] == ["ō"] * 3


def test_non_hindi_pack_projects_nothing():
    doc = legacy_doc()
    doc["system"] = {"phonology": "kurmanji"}
    doc["phraseGrid"][0][0]["syllables"] = [
        {"anchor": {"traj": "a1", "t": 0}, "onset": "b", "nucleus": "a", "lyric": {"vowel": "a", "source": "human"}}]
    out = save(load(doc))
    assert all(v == {"art0": None, "art1": None} for v in SYL.stored_legacy(out).values())


def test_unknown_pack_is_kept_and_not_projected():
    doc = legacy_doc()
    doc["system"] = {"phonology": "klingon", "futureComponent": "x"}
    doc["phraseGrid"][0][0]["syllables"] = [copy.deepcopy(SYL_A1)]
    with pytest.warns(UserWarning, match="not bundled"):
        p = Piece.from_json(copy.deepcopy(doc))
    out = save(p)
    assert out["system"] == doc["system"]
    assert out["phraseGrid"][0][0]["syllables"] == [SYL_A1]


# --------------------------------------------------------------------------- uids
def test_unique_ids_are_stable_across_save_and_load():
    doc = legacy_doc()
    del doc["phraseGrid"][0][0]["trajectoryGrid"][0][1]["uniqueId"]
    p = load(doc)
    assert ("traj-uid-missing", "track 0 phrase 0 traj 1") in p.syllable_violations()
    j1 = save(p)
    j2 = save(load(j1))
    uids = lambda j: [t["uniqueId"] for t in j["phraseGrid"][0][0]["trajectoryGrid"][0]]
    assert uids(j1) == uids(j2)
    assert all(uids(j1))


# --------------------------------------------------------------------------- re-homing
def test_deleted_trajectory_drops_its_syllable():
    doc = legacy_doc()
    doc["phraseGrid"][0][0]["syllables"] = [
        copy.deepcopy(SYL_A1),
        {"anchor": {"traj": "a3", "t": 0}, "onset": "ga", "nucleus": "u", "coda": "la",
         "lyric": {"vowel": "u", "source": "pack"}}]
    p = load(doc)
    ph = p.phrase_grid[0][0]
    ph.trajectory_grid[0] = ph.trajectories[:2]
    assert p.rehome_syllables() == ["a3"]
    assert [s["anchor"]["traj"] for s in ph.syllables] == ["a1"]
    assert ph.has_stored_syllables


def test_trajectory_moved_between_phrases_takes_its_syllable():
    doc = piece([phrase("ph0", [traj("a1", vowel="ā"), traj("a2", vowel="ī")], syllables=[
        {"anchor": {"traj": "a1", "t": 0}, "nucleus": "ā", "lyric": {"vowel": "ā", "source": "pack"}},
        {"anchor": {"traj": "a2", "t": 0}, "nucleus": "ī", "lyric": {"vowel": "ī", "source": "pack"}}]),
        phrase("ph1", [traj("b1", vowel="ō")], start=1.0, syllables=[
            {"anchor": {"traj": "b1", "t": 0}, "nucleus": "ō", "lyric": {"vowel": "ō", "source": "pack"}}])])
    p = load(doc)
    before = save(p)
    ph0, ph1 = p.phrase_grid[0]
    moved = ph0.trajectory_grid[0].pop()
    ph1.trajectory_grid[0].insert(0, moved)
    p.rehome_syllables(before)
    assert [s["anchor"]["traj"] for s in ph0.syllables] == ["a1"]
    assert [s["anchor"]["traj"] for s in ph1.syllables] == ["a2", "b1"]


def test_add_trajectory_into_a_tiered_track_warns_when_uncovered():
    doc = piece([phrase("ph0", [traj("a1", vowel="ā"), traj("s1", tid=12)], syllables=[
        {"anchor": {"traj": "a1", "t": 0}, "nucleus": "ā", "lyric": {"vowel": "ā", "source": "pack"}}])])
    p = load(doc)
    new = Trajectory({"id": 0, "pitches": [], "dur_tot": 0.5, "vowel": "ī"})
    with pytest.warns(UserWarning, match="not what the syllable tier"):
        assert p.add_trajectory(new, 0, 0.5)


# --------------------------------------------------------------------------- packs
def test_bundled_packs():
    assert SYL.available_packs() == ["hindi-urdu", "jyutping", "kurmanji"]
    with pytest.raises(SYL.UnknownPackError):
        SYL.load_pack("klingon")


def test_jyutping_nine_tone_derivation():
    pack = SYL.load_pack("jyutping")
    sik6 = {"onset": "s", "nucleus": "i", "coda": "k", "tone": "6"}
    si1 = {"onset": "s", "nucleus": "i", "tone": "1"}
    sik2 = dict(sik6, tone="2")  # a changed tone on a checked syllable keeps its number
    assert [SYL.derived_tone(s, pack) for s in (sik6, si1, sik2)] == ["9", "1", "2"]
    assert SYL.derived_tone({"onset": "s"}, pack) is None


# --------------------------------------------------------------------------- dry run
def _origin_main_available():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run(["git", "-C", here, "rev-parse", "--verify", "-q", "origin/main"],
                       capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _origin_main_available(), reason="needs git ref origin/main")
def test_dry_run_harness_passes_legacy_and_catches_a_display_copy_diff(tmp_path):
    script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "syllable_dry_run.py")
    src = tmp_path / "docs"
    src.mkdir()
    (src / "plain.json").write_text(json.dumps(legacy_doc(), ensure_ascii=False), encoding="utf-8")
    run = lambda: subprocess.run([sys.executable, script, str(src), "--tmp", str(tmp_path),
                                  "--report", str(tmp_path / "r.json")],
                                 capture_output=True, text=True)
    ok = run()
    assert ok.returncode == 0, ok.stdout + ok.stderr
    # a stored display copy that differs from the table: today's save keeps it,
    # the projection rewrites it (PROP-8 "stored display copies win today")
    odd = legacy_doc()
    odd["phraseGrid"][0][0]["trajectoryGrid"][0][0]["startConsonantHindi"] = "बा"
    (src / "odd.json").write_text(json.dumps(odd, ensure_ascii=False), encoding="utf-8")
    bad = run()
    assert bad.returncode == 1, bad.stdout + bad.stderr
    rep = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rep["gate"]["diffPieces"] == 1
    assert {d["field"] for d in rep["gate"]["examples"][0]["diffs"]} == {"art0", "startConsonantHindi"}
