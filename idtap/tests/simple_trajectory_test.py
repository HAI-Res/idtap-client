import os
import sys
sys.path.insert(0, os.path.abspath("."))

import math

import pytest

from idtap.classes.simple_trajectory import (
    TYPE_IDS,
    OrientationDot,
    SimpleTrajectory,
    decompose_trajectory,
)
from idtap.classes.piece import Piece
from idtap.classes.phrase import Phrase
from idtap.classes.trajectory import Trajectory
from idtap.classes.pitch import Pitch
from idtap.classes.raga import Raga
from idtap.enums import Instrument


def three_pitches():
    return [Pitch(), Pitch({'swara': 'r', 'raised': False}), Pitch({'swara': 'g'})]


def chunk_freq_at(chunk: SimpleTrajectory, abs_time: float) -> float:
    x = (abs_time - chunk.start.time) / chunk.dur_tot
    return chunk.compute(x)


def find_chunk(chunks, abs_time):
    for c in chunks:
        if c.start.time <= abs_time < c.end.time:
            return c
    return chunks[-1]


def assert_matches_compute(traj: Trajectory, t0: float = 0.0, samples: int = 200):
    """Chunks, evaluated piecewise, must reproduce traj.compute exactly."""
    chunks = decompose_trajectory(traj, t0)
    assert chunks[0].start.time == t0
    assert chunks[-1].end.time == pytest.approx(t0 + traj.dur_tot)
    for k in range(1, samples):
        x = k / samples
        abs_time = t0 + x * traj.dur_tot
        chunk = find_chunk(chunks, abs_time)
        assert chunk_freq_at(chunk, abs_time) == pytest.approx(
            traj.compute(x), rel=1e-9
        ), f"mismatch at x={x} for id {traj.id}"


def test_fixed_single_chunk():
    traj = Trajectory({'id': 0, 'pitches': [Pitch()], 'dur_tot': 2.0})
    chunks = decompose_trajectory(traj, 5.0)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.type == 'fixed'
    assert c.type_id == 0
    assert not c.continuation
    assert c.start == OrientationDot(5.0, traj.log_freqs[0])
    assert c.end == OrientationDot(7.0, traj.log_freqs[0])
    assert_matches_compute(traj, 5.0)


@pytest.mark.parametrize('traj_id,expected_type', [
    (1, 'cosine'), (2, 'sloped-start'), (3, 'sloped-end'),
])
def test_simple_bends_single_chunk(traj_id, expected_type):
    traj = Trajectory({
        'id': traj_id, 'pitches': three_pitches()[:2], 'dur_tot': 1.5,
        'slope': 3.0,
    })
    chunks = decompose_trajectory(traj, 1.0)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.type == expected_type
    assert c.start.log_freq == traj.log_freqs[0]
    assert c.end.log_freq == traj.log_freqs[1]
    if traj_id in (2, 3):
        assert c.slope == 3.0
    assert_matches_compute(traj, 1.0)


def test_ladle_decomposes_to_sloped_start_then_cosine():
    traj = Trajectory({
        'id': 4, 'pitches': three_pitches(), 'dur_tot': 3.0, 'slope': 2.5,
    })
    chunks = decompose_trajectory(traj, 10.0)
    assert [c.type for c in chunks] == ['sloped-start', 'cosine']
    first, second = chunks
    assert not first.continuation
    assert second.continuation
    # boundary at dur_array[0] of the total duration, shared orientation dot
    assert first.end.time == pytest.approx(10.0 + 3.0 * traj.dur_array[0])
    assert second.start == first.end
    assert first.slope == 2.5
    assert second.slope == 2.0
    assert_matches_compute(traj, 10.0)


def test_reverse_ladle_decomposes_to_cosine_then_sloped_end():
    traj = Trajectory({
        'id': 5, 'pitches': three_pitches(), 'dur_tot': 2.0, 'slope': 1.7,
    })
    chunks = decompose_trajectory(traj)
    assert [c.type for c in chunks] == ['cosine', 'sloped-end']
    assert [c.continuation for c in chunks] == [False, True]
    assert chunks[1].slope == 1.7
    assert_matches_compute(traj)


def test_yoyo_decomposes_to_cosine_per_segment():
    pitches = three_pitches() + [Pitch({'swara': 'm'})]
    traj = Trajectory({'id': 6, 'pitches': pitches, 'dur_tot': 2.0})
    chunks = decompose_trajectory(traj)
    assert len(chunks) == 3
    assert all(c.type == 'cosine' for c in chunks)
    assert [c.continuation for c in chunks] == [False, True, True]
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start == prev.end
    assert_matches_compute(traj)


def test_krintin_decomposes_to_fixed_steps():
    traj = Trajectory({'id': 7, 'pitches': three_pitches()[:2], 'dur_tot': 1.0})
    chunks = decompose_trajectory(traj)
    assert [c.type for c in chunks] == ['fixed', 'fixed']
    # continuation marks same-gesture membership even across the pitch jump
    assert [c.continuation for c in chunks] == [False, True]
    assert chunks[0].end.time == pytest.approx(traj.dur_array[0])
    assert chunks[0].start.log_freq == traj.log_freqs[0]
    assert chunks[1].start.log_freq == traj.log_freqs[1]
    assert_matches_compute(traj)


def test_step_traj_all_chunks_after_first_are_continuations():
    p = [Pitch(), Pitch({'swara': 'r'}), Pitch({'swara': 'r'}), Pitch()]
    traj = Trajectory({'id': 9, 'pitches': p, 'dur_tot': 1.0})
    chunks = decompose_trajectory(traj)
    assert len(chunks) == 4
    assert [c.continuation for c in chunks] == [False, True, True, True]
    assert_matches_compute(traj)


def test_slide_decomposes_to_two_fixed_steps():
    traj = Trajectory({'id': 11, 'pitches': three_pitches()[:2], 'dur_tot': 1.0})
    chunks = decompose_trajectory(traj)
    assert [c.type for c in chunks] == ['fixed', 'fixed']
    assert_matches_compute(traj)


def test_silent_chunk_has_no_log_freqs():
    traj = Trajectory({'id': 12, 'pitches': [Pitch()], 'dur_tot': 2.5,
                       'fund_id12': 261.63})
    chunks = decompose_trajectory(traj, 4.0)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.type == 'silent'
    assert c.type_id == 4
    assert (c.start.time, c.end.time) == (4.0, 6.5)
    assert c.start.log_freq is None and c.end.log_freq is None
    assert c.compute(0.5) is None
    assert not c.continuation


def test_vibrato_decomposes_to_half_period_cosines():
    traj = Trajectory({'id': 13, 'pitches': [Pitch()], 'dur_tot': 1.0,
                       'vib_obj': {'periods': 4, 'init_up': True,
                                   'extent': 0.06, 'vert_offset': 0.0}})
    chunks = decompose_trajectory(traj)
    assert len(chunks) == 8  # 2 chunks per period
    assert all(c.type == 'cosine' for c in chunks)
    assert [c.continuation for c in chunks] == [False] + [True] * 7
    # boundary dots sit on the actual vibrato curve
    for c in chunks:
        assert chunk_freq_at(c, c.start.time) == pytest.approx(
            traj.compute(c.start.time), rel=1e-9)
    # interior extremes span the vibrato extent
    interior = [c.start.log_freq for c in chunks[2:-1]]
    assert max(interior) - min(interior) == pytest.approx(0.06)
    assert_matches_compute(traj)


def test_piece_simplified_trajectories_absolute_times():
    raga = Raga()
    t1 = Trajectory({'id': 0, 'pitches': [Pitch()], 'dur_tot': 1.0})
    t2 = Trajectory({'id': 4, 'pitches': three_pitches(), 'dur_tot': 2.0})
    t3 = Trajectory({'id': 12, 'pitches': [Pitch()], 'dur_tot': 1.0})
    p1 = Phrase({'trajectories': [t1, t2], 'raga': raga})
    p2 = Phrase({'trajectories': [t3], 'raga': raga})
    piece = Piece({'phrases': [p1, p2], 'raga': raga,
                   'instrumentation': [Instrument.Sitar]})
    chunks = piece.simplified_trajectories()
    assert [c.type for c in chunks] == ['fixed', 'sloped-start', 'cosine', 'silent']
    # continuous, absolute timeline across phrase boundaries
    assert chunks[0].start.time == 0.0
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start.time == pytest.approx(prev.end.time)
    assert chunks[-1].end.time == pytest.approx(4.0)
    # continuation never crosses a source-trajectory boundary
    assert [c.continuation for c in chunks] == [False, False, True, False]


def test_trajectory_to_simple_delegates():
    traj = Trajectory({'id': 5, 'pitches': three_pitches(), 'dur_tot': 1.0})
    assert traj.to_simple(2.0) == decompose_trajectory(traj, 2.0)


def test_json_round_trip():
    traj = Trajectory({'id': 4, 'pitches': three_pitches(), 'dur_tot': 2.0,
                       'slope': 2.5})
    chunks = decompose_trajectory(traj, 1.5)
    for c in chunks:
        data = c.to_json()
        assert data['typeId'] == TYPE_IDS[c.type]
        assert len(data['dots']) == 2
        assert SimpleTrajectory.from_json(data) == c
    silent = decompose_trajectory(
        Trajectory({'id': 12, 'pitches': [Pitch()], 'dur_tot': 1.0}))[0]
    data = silent.to_json()
    assert 'logFreq' not in data['dots'][0]
    assert SimpleTrajectory.from_json(data) == silent


def test_invalid_type_rejected():
    with pytest.raises(ValueError):
        SimpleTrajectory('wiggly', OrientationDot(0.0), OrientationDot(1.0))
