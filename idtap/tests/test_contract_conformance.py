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
