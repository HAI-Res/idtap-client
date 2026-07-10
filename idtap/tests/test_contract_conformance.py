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
    got = [p.frequency
           for phrases in piece.phrase_grid
           for ph in phrases
           for t in ph.trajectory_grid[0]
           for p in t.pitches]
    want = fx["expected"]["allPitchFrequencies"]
    assert len(got) == len(want), f"{fx['name']}: {len(got)} pitches vs {len(want)}"
    for g, w in zip(got, want):
        assert _rel(g, w, rel), f"{fx['name']}: pitch freq {g} != {w}"
