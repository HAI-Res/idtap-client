"""Conformance tests against the idtap-contract golden fixtures.

These load the language-neutral fixtures from the sibling `idtap-contract` repo
and assert the Python model reproduces the expected values. They encode the
canonical (post-strip) serialization contract shared with the TypeScript app.

The `stripped-*` fixtures FAIL until `from_json` threads the raga context
(ratios/fundamental) down the chain — that failure IS the Yaman bug.

Fixture location: sibling `../idtap-contract/fixtures` by default; override with
the IDTAP_CONTRACT_DIR env var.
"""
import os
import glob
import json
import math

import pytest

from idtap.classes.pitch import Pitch
from idtap.classes.trajectory import Trajectory
from idtap.classes.phrase import Phrase
from idtap.classes.piece import Piece


def _contract_dir():
    env = os.environ.get("IDTAP_CONTRACT_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    # Python-API/idtap/tests -> ../../../idtap-contract
    return os.path.abspath(os.path.join(here, "..", "..", "..", "idtap-contract"))


def _fixtures(entity):
    d = os.path.join(_contract_dir(), "fixtures", entity)
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    return [f for f in files if not f.endswith("index.json")]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _rel(a, b, rel):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def test_contract_fixtures_available():
    """When CI sets REQUIRE_CONTRACT=1, the idtap-contract fixtures MUST be present —
    fail loudly instead of silently skipping the parametrized conformance cases (an
    empty parameter set is not a test failure on its own)."""
    if os.environ.get("REQUIRE_CONTRACT") != "1":
        pytest.skip("REQUIRE_CONTRACT not set — conformance fixtures are optional locally")
    assert PITCH_FIXTURES, f"no idtap-contract fixtures found under {_contract_dir()}"


# --------------------------------------------------------------------------- pitch
PITCH_FIXTURES = _fixtures("pitch")


@pytest.mark.parametrize("path", PITCH_FIXTURES, ids=lambda p: os.path.basename(p))
def test_pitch_conformance(path):
    fx = _load(path)
    ctx = fx.get("context") or {}
    # Contract rule: threaded context wins; embedded (legacy) values are fallback.
    p = Pitch.from_json(
        fx["pitchJson"],
        ratios=ctx.get("ratios"),
        fundamental=ctx.get("fundamental"),
    )
    exp = fx["expected"]
    rel = fx.get("tolerance", {}).get("rel", 1e-9)
    assert _rel(p.frequency, exp["frequency"], rel), (
        f"{fx['name']}: frequency {p.frequency} != expected {exp['frequency']}"
    )
    assert _rel(p.log_freq, exp["logFreq"], rel)
    assert p.numbered_pitch == exp["numberedPitch"]
    assert p.chroma == exp["chroma"]
    assert p.sargam_letter == exp["sargamLetter"]


def test_pitch_to_json_is_stripped():
    """to_json() must emit the canonical stripped shape (no ratios/fundamental)."""
    p = Pitch({"swara": "re", "raised": True, "oct": 0})
    keys = set(p.to_json().keys())
    assert keys == {"swara", "raised", "oct", "logOffset"}, (
        f"to_json emitted {keys}; expected the stripped 4-key shape"
    )


# --------------------------------------------------------------------------- raga
RAGA_FIXTURES = _fixtures("raga")


def _flatten(x):
    for e in x:
        if isinstance(e, list):
            yield from e
        else:
            yield e


@pytest.mark.parametrize("path", RAGA_FIXTURES, ids=lambda p: os.path.basename(p))
def test_raga_conformance(path):
    """PROP-1: with ruleSet serialized, Raga.from_json must reconstruct the correct
    stratifiedRatios + fundamental for NON-Yaman ragas (Bhairav/Todi/both-ni) too —
    it reads the serialized ruleSet rather than defaulting to Yaman."""
    from idtap.classes.raga import Raga
    fx = _load(path)
    rg = Raga.from_json(fx["ragaJson"])
    rel = fx.get("tolerance", {}).get("rel", 1e-9)
    got = list(_flatten(rg.stratified_ratios))
    want = list(_flatten(fx["expected"]["stratifiedRatios"]))
    assert len(got) == len(want), f"{fx['name']}: {len(got)} ratios vs {len(want)}"
    for g, w in zip(got, want):
        assert _rel(g, w, rel), f"{fx['name']}: ratio {g} != {w}"
    assert rg.fundamental == fx["expected"]["fundamental"]


def test_raga_to_json_includes_rule_set():
    """PROP-1: to_json must emit ruleSet so the piece's raga is self-contained."""
    from idtap.classes.raga import Raga
    j = Raga().to_json()
    assert "ruleSet" in j, "Raga.to_json must serialize ruleSet (PROP-1)"


# --------------------------------------------------------------------------- trajectory
TRAJECTORY_FIXTURES = _fixtures("trajectory")


@pytest.mark.parametrize("path", TRAJECTORY_FIXTURES, ids=lambda p: os.path.basename(p))
def test_trajectory_conformance(path):
    fx = _load(path)
    ctx = fx.get("context") or {}
    t = Trajectory.from_json(
        fx["trajectoryJson"],
        ratios=ctx.get("ratios"),
        fundamental=ctx.get("fundamental"),
    )
    exp = fx["expected"]
    rel = fx.get("tolerance", {}).get("rel", 1e-9)
    got = [p.frequency for p in t.pitches]
    assert len(got) == len(exp["pitchFrequencies"])
    for g, w in zip(got, exp["pitchFrequencies"]):
        assert _rel(g, w, rel), f"{fx['name']}: pitch freq {g} != {w}"
    assert t.id == exp["id"]
    # Derived/default checks apply to the stripped form. Legacy fixtures embed
    # real name/tags, which must be PRESERVED (backward compat), not defaulted.
    if fx.get("scenario") == "stripped":
        assert t.name == exp["derivedName"]   # name derived from id
        assert t.tags == exp["tagsDefault"]   # tags default []
    # E: sub-objects (articulations / automation) must round-trip when present
    tj = fx["trajectoryJson"]
    for k, art in (tj.get("articulations") or {}).items():
        assert k in t.articulations, f"articulation key {k} lost"
        assert t.articulations[k].name == art["name"]
    if tj.get("automation"):
        assert t.automation is not None, "automation lost"
        assert len(t.automation.values) == len(tj["automation"]["values"])


def test_trajectory_to_json_strips_fields():
    """to_json() must NOT include name / instrumentation / tags."""
    t = Trajectory({"id": 0, "pitches": [Pitch()], "durTot": 1})
    keys = set(t.to_json().keys())
    for stripped in ("name", "instrumentation", "tags"):
        assert stripped not in keys, f"to_json still emits {stripped}"


# --------------------------------------------------------------------------- phrase
PHRASE_FIXTURES = _fixtures("phrase")


@pytest.mark.parametrize("path", PHRASE_FIXTURES, ids=lambda p: os.path.basename(p))
def test_phrase_conformance(path):
    fx = _load(path)
    ctx = fx.get("context") or {}
    ph = Phrase.from_json(
        fx["phraseJson"],
        ratios=ctx.get("ratios"),
        fundamental=ctx.get("fundamental"),
    )
    rel = fx.get("tolerance", {}).get("rel", 1e-9)
    got = [p.frequency for row in ph.trajectory_grid for t in row for p in t.pitches]
    want = fx["expected"]["pitchFrequencies"]
    assert len(got) == len(want), f"{fx['name']}: {len(got)} pitches vs {len(want)}"
    for g, w in zip(got, want):
        assert _rel(g, w, rel), f"{fx['name']}: pitch freq {g} != {w}"


def test_piece_to_json_prop2():
    """PROP-2: to_json must omit `collections` (server-managed metadata) and emit
    ISO-8601 UTC dates ending in 'Z' (never naive, never {$date})."""
    from idtap.classes.piece import Piece
    j = Piece().to_json()
    assert "collections" not in j, "to_json must not emit collections (PROP-2)"
    for k in ("dateCreated", "dateModified"):
        assert k in j and isinstance(j[k], str) and j[k].endswith("Z"), (
            f"{k} must be an ISO-8601 UTC string ending in Z, got {j.get(k)!r}"
        )


def test_phrase_to_json_strips_raga():
    """to_json() must NOT include raga (inherited from the piece)."""
    from idtap.classes.raga import Raga
    ph = Phrase({"trajectories": [Trajectory({"id": 0, "pitches": [Pitch()], "durTot": 1})],
                 "raga": Raga()})
    assert "raga" not in ph.to_json()


# --------------------------------------------------------------------------- piece
PIECE_FIXTURES = _fixtures("piece")


@pytest.mark.parametrize("path", PIECE_FIXTURES, ids=lambda p: os.path.basename(p))
def test_piece_conformance(path):
    fx = _load(path)
    # Piece is the ROOT: from_json extracts (ratios, fundamental) from its raga
    # and threads them down the whole chain. No context arg.
    piece = Piece.from_json(fx["pieceJson"])
    rel = fx.get("tolerance", {}).get("rel", 1e-9)
    # Melodic pitches only — exclude id-12 (Silent) trajectories so the frequency
    # check is robust to string-sync (TS synthesizes a silent 2nd string; Python
    # does not — see STRING-SYNC in DIVERGENCES.md).
    got = [p.frequency
           for phrases in piece.phrase_grid
           for ph in phrases
           for string in ph.trajectory_grid   # ALL strings (polyphonic)
           for t in string if t.id != 12
           for p in t.pitches]
    want = fx["expected"]["allPitchFrequencies"]
    assert len(got) == len(want), f"{fx['name']}: {len(got)} pitches vs {len(want)}"
    for g, w in zip(got, want):
        assert _rel(g, w, rel), f"{fx['name']}: pitch freq {g} != {w}"
    # F: title (incl. Unicode) must round-trip unchanged
    if fx["pieceJson"].get("title") is not None:
        assert piece.title == fx["pieceJson"]["title"]


# --------------------------------------------------------------------------- idempotence (C)
@pytest.mark.parametrize("path", PIECE_FIXTURES, ids=lambda p: os.path.basename(p))
def test_piece_roundtrip_idempotence(path):
    """to_json -> from_json -> to_json must be STABLE. The first save strips
    fields, so cycle-1 vs cycle-2 output must be byte-identical (idempotent).
    Volatile fields (generated timestamps) are populated on first load and then
    preserved, so comparing the two post-first-roundtrip outputs is deterministic."""
    fx = _load(path)
    j1 = Piece.from_json(fx["pieceJson"]).to_json()
    j2 = Piece.from_json(j1).to_json()
    assert j1 == j2, f"{fx['name']}: serialization not idempotent"


# --------------------------------------------------------------------------- meter
METER_FIXTURES = _fixtures("meter")


@pytest.mark.parametrize("path", METER_FIXTURES, ids=lambda p: os.path.basename(p))
def test_meter_conformance(path):
    from idtap.classes.meter import Meter
    fx = _load(path)
    if fx.get("scenario") == "behavioral":
        # round-trip: expressive proportional offsets must survive to_json (METER-1)
        m = Meter.from_json(fx["meterJson"])
        j = m.to_json()
        got = [ps["offsets"] for layer in j["pulseStructures"] for ps in layer]
        want = fx["expected"]["pulseStructureOffsets"]
        rel = fx.get("tolerance", {}).get("rel", 1e-9)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert len(g) == len(w), f"{fx['name']}: offsets length"
            for a, b in zip(g, w):
                assert _rel(a, b, rel), f"{fx['name']}: offset {a} != {b}"
    else:  # structural
        inst = fx["instance"]
        for k in fx["expected"]["requiredKeys"]:
            assert k in inst
