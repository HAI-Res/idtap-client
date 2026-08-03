import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest

from idtap.classes.piece import Piece
from idtap.classes.phrase import Phrase
from idtap.classes.trajectory import Trajectory
from idtap.classes.pitch import Pitch
from idtap.classes.raga import Raga
from idtap.enums import Instrument

# Mirror of the TS regression tests for the polyphonic timeline-stretch bug
# (idtap#34): a second string whose trajectories outlast the phrase must not
# inflate the phrase/piece dur_tot; the main string is authoritative.


def silent(dur_tot):
    return Trajectory({'id': 12, 'dur_tot': dur_tot, 'pitches': [Pitch()]})


def sounding(dur_tot):
    return Trajectory({'id': 0, 'dur_tot': dur_tot,
                       'pitches': [Pitch(), Pitch()]})


def string_sum(trajs):
    return sum(t.dur_tot for t in trajs)


def make_piece(phrase):
    return Piece({
        'phrases': [phrase],
        'raga': Raga({'fundamental': 240}),
        'instrumentation': [Instrument.Sarangi],
    })


def test_phrase_dur_tot_ignores_overlong_second_string():
    phrase = Phrase({
        'trajectory_grid': [[silent(10)], [silent(4), sounding(2), silent(10)]],
    })
    assert phrase.dur_tot == pytest.approx(10, abs=1e-9)


def test_sync_trims_trailing_second_string_silence():
    phrase = Phrase({
        'trajectory_grid': [[silent(10)], [silent(4), sounding(2), silent(10)]],
    })
    piece = make_piece(phrase)
    s2 = piece.phrase_grid[0][0].trajectory_grid[1]
    assert string_sum(s2) == pytest.approx(10, abs=1e-6)
    assert any(t.id == 0 and abs(t.dur_tot - 2) < 1e-9 for t in s2)
    assert piece.dur_tot == pytest.approx(10, abs=1e-6)


def test_sync_pads_short_second_string_with_silence():
    phrase = Phrase({
        'trajectory_grid': [[silent(10)], [silent(2), sounding(3)]],
    })
    piece = make_piece(phrase)
    s2 = piece.phrase_grid[0][0].trajectory_grid[1]
    assert string_sum(s2) == pytest.approx(10, abs=1e-6)
    assert s2[-1].id == 12


def test_sounding_overhang_preserved_and_warned():
    phrase = Phrase({
        'trajectory_grid': [[silent(10)], [silent(2), sounding(14)]],
    })
    with pytest.warns(UserWarning):
        piece = make_piece(phrase)
    s2 = piece.phrase_grid[0][0].trajectory_grid[1]
    assert any(t.id == 0 and abs(t.dur_tot - 14) < 1e-9 for t in s2)
    assert piece.phrase_grid[0][0].dur_tot == pytest.approx(10, abs=1e-9)
    assert piece.dur_tot == pytest.approx(10, abs=1e-6)


def test_healthy_dual_string_phrase_untouched():
    phrase = Phrase({
        'trajectory_grid': [
            [silent(4), sounding(6)],
            [silent(1), sounding(2), silent(7)],
        ],
    })
    piece = make_piece(phrase)
    s2 = piece.phrase_grid[0][0].trajectory_grid[1]
    assert len(s2) == 3
    assert string_sum(s2) == pytest.approx(10, abs=1e-9)
    assert piece.dur_tot == pytest.approx(10, abs=1e-9)
