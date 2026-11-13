"""Readers must accept and round-trip keys they do not know (idtap-contract
0.3.0, PROP-8 "things not to forget"): idtap 0.1.58 raised on `syllables` and
refused `system` / `lyricsText`, so it could not load any document a 0.3.0
client wrote. Fixture-driven: fixtures/unknown_keys_roundtrip.json."""
import copy
import json
import os
import warnings

import pytest

from idtap.classes.articulation import Articulation
from idtap.classes.phrase import Phrase
from idtap.classes.piece import Piece
from idtap.classes.trajectory import Trajectory

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "unknown_keys_roundtrip.json")


def _fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _at(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def _roundtrip(pj):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Piece.from_json(copy.deepcopy(pj)).to_json()


@pytest.mark.parametrize("path", _fixture()["expected"]["kept"], ids=lambda p: "/".join(map(str, p)))
def test_unknown_key_survives_load_and_save(path):
    fx = _fixture()
    out = _roundtrip(fx["pieceJson"])
    assert _at(out, path) == _at(fx["pieceJson"], path)


def test_second_roundtrip_is_stable():
    fx = _fixture()
    j1 = _roundtrip(fx["pieceJson"])
    j2 = _roundtrip(j1)
    assert j1 == j2


def test_load_emits_no_unrecognized_field_warning():
    fx = _fixture()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Piece.from_json(copy.deepcopy(fx["pieceJson"]))
    assert not [w for w in rec if "unrecognized" in str(w.message)]


def test_phrase_from_json_accepts_syllables():
    """The exact 0.1.58 failure: ValueError: Invalid parameter: 'syllables'."""
    ph = _fixture()["pieceJson"]["phraseGrid"][0][0]
    phrase = Phrase.from_json(copy.deepcopy(ph))
    assert phrase.syllables == ph["syllables"]
    assert phrase.to_json()["syllables"] == ph["syllables"]  # wordStart stays camelCase


def test_constructors_stay_strict_for_typos():
    """Hand-built objects still get the helpful errors; only from_json is lenient."""
    with pytest.raises(ValueError):
        Phrase({"trajs": []})
    with pytest.raises(ValueError):
        Trajectory({"durTotal": 1})
    with pytest.raises(ValueError):
        Articulation({"english": "x"})
    with pytest.raises(ValueError):
        Piece({"phrase_grid": []})
