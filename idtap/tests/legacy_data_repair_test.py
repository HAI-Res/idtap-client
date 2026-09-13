"""Tests for legacy server-data repair in from_json paths.

Old server records contain data the web editor never validated: null entries
in 'collections'/'sectionStarts' (left behind by deleted collections) and
negative durations written by early editor bugs. from_json repairs these with
a warning instead of rejecting the piece; direct construction stays strict.
"""
import warnings

import pytest

from idtap.classes.piece import Piece
from idtap.classes.phrase import Phrase
from idtap.classes.trajectory import Trajectory


def make_piece_json(**overrides):
    obj = {
        'title': 'Legacy Test',
        'phraseGrid': [[]],
        'instrumentation': ['Vocal (M)'],
    }
    obj.update(overrides)
    return obj


def test_piece_drops_null_collections():
    obj = make_piece_json(collections=[None, 'abc123', None])
    with pytest.warns(UserWarning, match='collection'):
        piece = Piece.from_json(obj)
    assert piece.collections == ['abc123']


def test_piece_drops_null_section_starts():
    obj = make_piece_json(sectionStarts=[0, None])
    with pytest.warns(UserWarning, match='sectionStarts'):
        piece = Piece.from_json(obj)


def test_piece_drops_null_section_starts_grid_entries():
    # section_starts_grid is derived from per-phrase is_section_start flags, so
    # the piece needs real phrases for the repaired starts to be observable.
    phrase = {'trajectories': [{'id': 0, 'durTot': 1.0, 'durArray': [1]}],
              'durTot': 1.0, 'durArray': [1]}
    obj = make_piece_json(phraseGrid=[[dict(phrase), dict(phrase), dict(phrase)]],
                          sectionStartsGrid=[[0, None, 2]])
    with pytest.warns(UserWarning, match=r"dropped 1 null .* 'sectionStartsGrid'"):
        piece = Piece.from_json(obj)
    assert piece.section_starts_grid == [[0, 2]]


def test_piece_valid_collections_unchanged():
    obj = make_piece_json(collections=['abc123'])
    piece = Piece.from_json(obj)
    assert piece.collections == ['abc123']


def test_trajectory_repairs_negative_dur_tot():
    obj = {'id': 0, 'durTot': -1.28, 'durArray': [1]}
    with pytest.warns(UserWarning, match='durTot'):
        traj = Trajectory.from_json(obj)
    assert traj.dur_tot == pytest.approx(1.28)


def test_trajectory_repairs_negative_dur_array():
    obj = {'id': 6, 'durTot': 0.5, 'durArray': [-0.25, 0.75, 0.5]}
    with pytest.warns(UserWarning, match='durArray'):
        traj = Trajectory.from_json(obj)
    assert all(d >= 0 for d in traj.dur_array)
    assert sum(traj.dur_array) == pytest.approx(1.0)


def test_phrase_repairs_negative_durations():
    obj = {
        'trajectories': [
            {'id': 0, 'durTot': 1.0, 'durArray': [1]},
            {'id': 0, 'durTot': -0.5, 'durArray': [1]},
        ],
        'durTot': 1.5,
        'durArray': [2 / 3, -1 / 3],
    }
    with pytest.warns(UserWarning, match='durArray'):
        phrase = Phrase.from_json(obj)
    assert all(d >= 0 for d in phrase.dur_array)


@pytest.mark.parametrize('dur_array', [[0.0, 0.0, 0.0], [-0.0, 0.0]])
def test_trajectory_all_zero_dur_array_still_rejected(dur_array):
    """Repair flips signs; it cannot invent durations. An all-zero durArray is a
    genuinely degenerate record: no repair fires (-0.0 is not < 0, and abs() of
    any real negative is > 0, so the `total > 0` guard is purely defensive) and
    the constructor's all-zero check still raises."""
    obj = {'id': 6, 'durTot': 0.5, 'durArray': dur_array}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='all zero'):
            Trajectory.from_json(obj)


def test_direct_construction_stays_strict():
    with pytest.raises(ValueError):
        Trajectory({'id': 0, 'dur_tot': -1.0})
    with pytest.raises(ValueError):
        Trajectory({'id': 6, 'dur_array': [-0.25, 0.75, 0.5]})
    with pytest.raises(TypeError):
        Piece({'collections': [None]})
    with pytest.raises(TypeError):
        Piece({'sectionStarts': [0, None]})
