import os
import sys
sys.path.insert(0, os.path.abspath("."))

import math

import pytest

from idtap.classes.simple_trajectory import (
    TYPE_IDS,
    VIBRATO_FIELDS,
    OrientationDot,
    SimpleTrajectory,
    decompose_trajectory,
    vibrato_cosine_segments,
    vibrato_log_freq,
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


def assert_matches_compute(traj: Trajectory, t0: float = 0.0, samples: int = 200,
                           *, vibrato_as_cosines: bool = False):
    """Chunks, evaluated piecewise, must reproduce traj.compute exactly."""
    chunks = decompose_trajectory(traj, t0, vibrato_as_cosines=vibrato_as_cosines)
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


def test_type_ids_are_stable():
    # stored data and the consuming model depend on these codes; 'vibrato' is
    # appended, never renumbered into the existing five
    assert TYPE_IDS == {'fixed': 0, 'cosine': 1, 'sloped-start': 2,
                        'sloped-end': 3, 'silent': 4, 'vibrato': 5}
    assert VIBRATO_FIELDS == ('rate', 'extent_start', 'extent_end', 'phase')


VIB_CASES = [
    # (vib_obj, dur_tot): constant extents, ramps, odd phases, P < 1, short
    ({'rate': 5.5, 'extent_start': 0.05, 'extent_end': 0.05, 'vert_offset': 0.0, 'phase': math.pi}, 2.0),
    ({'rate': 3.7, 'extent_start': 0.06, 'extent_end': 0.02, 'vert_offset': 0.0, 'phase': 1.3}, 0.55),
    ({'rate': 4.7, 'extent_start': 0.0, 'extent_end': 0.08, 'vert_offset': 0.0, 'phase': 0.0}, 1.3),
    ({'rate': 1.0, 'extent_start': 0.05, 'extent_end': 0.05, 'vert_offset': 0.0, 'phase': 0.2 * math.pi}, 1.0),
    ({'rate': 0.6, 'extent_start': 0.03, 'extent_end': 0.05, 'vert_offset': 0.0, 'phase': math.pi / 2}, 0.3),
    ({'rate': 8.9, 'extent_start': 0.1, 'extent_end': 0.01, 'vert_offset': 0.0, 'phase': -2.7}, 3.1),
]


def _vib13(vib, dur_tot):
    return Trajectory({'id': 13, 'pitches': [Pitch({'swara': 'g'})],
                       'dur_tot': dur_tot, 'vib_obj': dict(vib)})


@pytest.mark.parametrize('vib, dur_tot', VIB_CASES)
def test_vibrato_decomposes_to_one_chunk_carrying_the_vib_obj(vib, dur_tot):
    traj = _vib13(vib, dur_tot)
    chunks = decompose_trajectory(traj, 2.0)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.type == 'vibrato'
    assert c.type_id == 5
    assert not c.continuation
    # both dots on the centre, the notated pitch
    assert c.start == OrientationDot(2.0, traj.log_freqs[0])
    assert c.end == OrientationDot(2.0 + dur_tot, traj.log_freqs[0])
    # a rename, not a fit
    assert (c.rate, c.extent_start, c.extent_end, c.phase) == (
        vib['rate'], vib['extent_start'], vib['extent_end'], vib['phase'])
    assert_matches_compute(traj, 2.0)


@pytest.mark.parametrize('vib, dur_tot', VIB_CASES)
def test_vibrato_chunk_compute_matches_id13_to_1e12(vib, dur_tot):
    traj = _vib13(vib, dur_tot)
    c = decompose_trajectory(traj, 0.0)[0]
    for k in range(0, 1001):
        x = k / 1000
        expected = traj.id13(x)
        assert abs(c.compute(x) - expected) <= 1e-12 * abs(expected), (x, vib)
        assert abs(vibrato_log_freq(
            x, dur_tot, traj.log_freqs[0], vib['rate'], vib['extent_start'],
            vib['extent_end'], vib['phase']) - math.log2(expected)) <= 1e-12


def test_vibrato_log_freq_with_offset_matches_id13():
    vib = {'rate': 4.0, 'extent_start': 0.0, 'extent_end': 0.08,
           'vert_offset': 0.01, 'phase': math.pi}
    traj = _vib13(vib, 2.0)
    for k in range(0, 1001):
        x = k / 1000
        assert vibrato_log_freq(
            x, 2.0, traj.log_freqs[0], vib['rate'], vib['extent_start'],
            vib['extent_end'], vib['phase'], vib['vert_offset'],
        ) == pytest.approx(math.log2(traj.id13(x)), abs=1e-12)


def test_vibrato_with_vert_offset_falls_back_to_cosine_chain():
    # the chunk has no offset field; the chain is what still reproduces the
    # curve, and the ends stay on the notated pitch
    traj = _vib13({'rate': 5.5, 'extent_start': 0.05, 'extent_end': 0.05,
                   'vert_offset': 0.01, 'phase': math.pi}, 2.0)
    chunks = decompose_trajectory(traj, 1.0)
    assert all(c.type == 'cosine' for c in chunks)
    assert len(chunks) == len(vibrato_cosine_segments(traj))
    assert chunks[0].start.log_freq == traj.log_freqs[0]
    assert chunks[-1].end.log_freq == pytest.approx(traj.log_freqs[0])
    assert_matches_compute(traj, 1.0)


def test_vibrato_as_cosines_flag_selects_the_chain():
    traj = _vib13(*VIB_CASES[0])
    chain = decompose_trajectory(traj, 1.0, vibrato_as_cosines=True)
    assert len(chain) > 1 and all(c.type == 'cosine' for c in chain)
    assert [c.continuation for c in chain] == [False] + [True] * (len(chain) - 1)
    assert_matches_compute(traj, 1.0, vibrato_as_cosines=True)
    assert traj.to_simple(1.0, vibrato_as_cosines=True) == chain
    assert traj.to_simple(1.0) == decompose_trajectory(traj, 1.0)


def test_vibrato_json_round_trip_and_keys():
    traj = _vib13(*VIB_CASES[1])
    c = decompose_trajectory(traj, 1.5)[0]
    data = c.to_json()
    assert data['typeId'] == 5
    assert (data['rate'], data['extentStart'], data['extentEnd'], data['phase']) == (
        c.rate, c.extent_start, c.extent_end, c.phase)
    assert SimpleTrajectory.from_json(data) == c
    # non-vibrato chunks never carry the vibrato keys ...
    fixed = decompose_trajectory(
        Trajectory({'id': 0, 'pitches': [Pitch()], 'dur_tot': 1.0}))[0].to_json()
    assert not {'rate', 'extentStart', 'extentEnd', 'phase'} & set(fixed)
    # ... and a chunk serialized before the type existed still loads
    old = {'type': 'cosine', 'typeId': 1,
           'dots': [{'time': 0.0, 'logFreq': 7.0}, {'time': 1.0, 'logFreq': 7.1}],
           'slope': 2.0, 'continuation': True}
    loaded = SimpleTrajectory.from_json(old)
    assert loaded.type == 'cosine'
    assert (loaded.rate, loaded.extent_start, loaded.extent_end, loaded.phase) == (
        None, None, None, None)


def test_vibrato_fields_default_and_validate():
    dots = (OrientationDot(0.0, 7.0), OrientationDot(1.0, 7.0))
    c = SimpleTrajectory('vibrato', *dots, rate=5.0, extent_start=0.04)
    assert (c.extent_end, c.phase) == (0.04, 0.0)
    assert c.compute(0.0) == pytest.approx(2 ** 7.0)
    assert c.compute(1.0) == pytest.approx(2 ** 7.0)
    with pytest.raises(ValueError, match="requires rate"):
        SimpleTrajectory('vibrato', *dots, extent_start=0.04)
    with pytest.raises(ValueError, match="requires extent_start"):
        SimpleTrajectory('vibrato', *dots, rate=5.0)
    with pytest.raises(ValueError, match="rate must be > 0"):
        SimpleTrajectory('vibrato', *dots, rate=0.0, extent_start=0.04)
    with pytest.raises(ValueError, match="extent_end must be >= 0"):
        SimpleTrajectory('vibrato', *dots, rate=5.0, extent_start=0.04,
                         extent_end=-0.01)
    with pytest.raises(TypeError, match="must be a number"):
        SimpleTrajectory('vibrato', *dots, rate='5', extent_start=0.04)
    # the format requires only rate > 0 and extents >= 0: a slow human-made
    # vibrato and a zero-extent one are both valid chunks
    SimpleTrajectory('vibrato', *dots, rate=1.5, extent_start=0.0)


@pytest.mark.parametrize('typ', ['fixed', 'cosine', 'sloped-start', 'sloped-end', 'silent'])
def test_vibrato_fields_rejected_on_other_types(typ):
    lf = None if typ == 'silent' else 7.0
    dots = (OrientationDot(0.0, lf), OrientationDot(1.0, lf))
    with pytest.raises(ValueError, match="vibrato-only"):
        SimpleTrajectory(typ, *dots, rate=5.0)
    with pytest.raises(ValueError, match="vibrato-only"):
        SimpleTrajectory(typ, *dots, phase=0.0)


def test_vibrato_decomposes_to_half_period_cosines():
    # the cosine-chain view of a v1 input healed to v2 on load (PROP-6);
    # v2-specific shapes are covered in vibrato_v2_test.py
    traj = Trajectory({'id': 13, 'pitches': [Pitch()], 'dur_tot': 1.0,
                       'vib_obj': {'periods': 4, 'init_up': True,
                                   'extent': 0.06, 'vert_offset': 0.0}})
    chunks = decompose_trajectory(traj, vibrato_as_cosines=True)
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
    assert_matches_compute(traj, vibrato_as_cosines=True)


def test_piece_simplified_trajectories_absolute_times():
    raga = Raga()
    t1 = Trajectory({'id': 0, 'pitches': [Pitch()], 'dur_tot': 1.0})
    t2 = Trajectory({'id': 4, 'pitches': three_pitches(), 'dur_tot': 2.0})
    t3 = Trajectory({'id': 12, 'pitches': [Pitch()], 'dur_tot': 1.0})
    t4 = Trajectory({'id': 13, 'pitches': [Pitch()], 'dur_tot': 1.0})
    p1 = Phrase({'trajectories': [t1, t2], 'raga': raga})
    p2 = Phrase({'trajectories': [t3, t4], 'raga': raga})
    piece = Piece({'phrases': [p1, p2], 'raga': raga,
                   'instrumentation': [Instrument.Sitar]})
    chunks = piece.simplified_trajectories()
    assert [c.type for c in chunks] == [
        'fixed', 'sloped-start', 'cosine', 'silent', 'vibrato']
    # continuous, absolute timeline across phrase boundaries
    assert chunks[0].start.time == 0.0
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start.time == pytest.approx(prev.end.time)
    assert chunks[-1].end.time == pytest.approx(5.0)
    # continuation never crosses a source-trajectory boundary
    assert [c.continuation for c in chunks] == [False, False, True, False, False]
    # the flag threads through to the cosine-chain view
    chain = piece.simplified_trajectories(vibrato_as_cosines=True)
    assert chain[:4] == chunks[:4]
    assert len(chain) > 5 and all(c.type == 'cosine' for c in chain[4:])


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
