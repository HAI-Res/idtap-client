"""Chained dot arrays to chunks: the shape a model emits."""
import math

import pytest

from idtap import (
    OrientationDot,
    SimpleTrajectory,
    decompose_trajectory,
    simple_trajectories_from_dots,
)
from idtap.classes.pitch import Pitch
from idtap.classes.trajectory import Trajectory


def test_shares_dots_between_consecutive_chunks():
    """n dots describe n-1 chunks, and each chunk ends where the next begins."""
    chunks = simple_trajectories_from_dots(
        times=[0.0, 1.0, 2.5],
        log_freqs=[7.0, 7.5, 7.5],
        types=['cosine', 'fixed'],
    )
    assert len(chunks) == 2
    assert chunks[0].end.time == chunks[1].start.time == 1.0
    assert chunks[0].end.log_freq == chunks[1].start.log_freq == 7.5
    assert chunks[1].dur_tot == 1.5


def test_accepts_type_ids_as_well_as_names():
    by_name = simple_trajectories_from_dots([0.0, 1.0], [7.0, 7.2], ['cosine'])
    by_id = simple_trajectories_from_dots([0.0, 1.0], [7.0, 7.2], [1])
    assert by_name[0].type == by_id[0].type == 'cosine'


def test_slopes_default_and_apply_to_the_right_chunk():
    chunks = simple_trajectories_from_dots(
        times=[0.0, 1.0, 2.0],
        log_freqs=[7.0, 7.2, 7.4],
        types=['sloped-start', 'sloped-end'],
        slopes=[3.0, 0.5],
    )
    assert [c.slope for c in chunks] == [3.0, 0.5]
    assert simple_trajectories_from_dots(
        [0.0, 1.0], [7.0, 7.2], ['cosine'])[0].slope == 2.0


def test_continuation_marks_all_but_the_first():
    off = simple_trajectories_from_dots([0.0, 1.0, 2.0], [7.0, 7.1, 7.2],
                                        ['cosine', 'cosine'])
    on = simple_trajectories_from_dots([0.0, 1.0, 2.0], [7.0, 7.1, 7.2],
                                       ['cosine', 'cosine'], continuation=True)
    assert [c.continuation for c in off] == [False, False]
    assert [c.continuation for c in on] == [False, True]


def test_silent_chunks_carry_no_pitch():
    chunks = simple_trajectories_from_dots(
        times=[0.0, 1.0], log_freqs=[None, None], types=['silent'])
    assert chunks[0].compute(0.5) is None


@pytest.mark.parametrize("times, log_freqs, types, slopes, message", [
    ([0.0], [7.0], [], None, "at least two dots"),
    ([0.0, 1.0], [7.0], ['cosine'], None, "log_freqs"),
    ([0.0, 1.0, 2.0], [7.0, 7.1, 7.2], ['cosine'], None, "types"),
    ([0.0, 1.0], [7.0, 7.1], ['cosine'], [2.0, 2.0], "slopes"),
    ([0.0, 0.0], [7.0, 7.1], ['cosine'], None, "must increase"),
    ([0.0, 1.0], [7.0, 7.1], ['wobble'], None, "invalid simple trajectory"),
    ([0.0, 1.0], [7.0, 7.1], [99], None, "invalid simple trajectory type id"),
])
def test_rejects_malformed_input(times, log_freqs, types, slopes, message):
    with pytest.raises(ValueError, match=message):
        simple_trajectories_from_dots(times, log_freqs, types, slopes)


def test_round_trips_with_decompose_trajectory():
    """Chunks out of a real trajectory, flattened to arrays, and back."""
    traj = Trajectory({
        'id': 6,
        'pitches': [Pitch({'swara': 0}), Pitch({'swara': 2}), Pitch({'swara': 1})],
        'dur_tot': 3.0,
        'dur_array': [0.4, 0.6],
    })
    original = decompose_trajectory(traj, start_time=1.0)

    times = [c.start.time for c in original] + [original[-1].end.time]
    log_freqs = [c.start.log_freq for c in original] + [original[-1].end.log_freq]
    rebuilt = simple_trajectories_from_dots(
        times, log_freqs, [c.type for c in original],
        [c.slope for c in original], continuation=True)

    assert len(rebuilt) == len(original)
    for a, b in zip(original, rebuilt):
        assert a.type == b.type
        assert a.continuation == b.continuation
        assert a.start.time == pytest.approx(b.start.time)
        assert a.end.time == pytest.approx(b.end.time)
        assert a.start.log_freq == pytest.approx(b.start.log_freq)
        assert a.end.log_freq == pytest.approx(b.end.log_freq)


# ------------------------------------------------------------------ vibrato

VIB = {'rate': 5.5, 'extent_start': 0.05, 'extent_end': 0.03, 'phase': 1.0}


def test_vibrato_chunks_take_their_numbers_as_mapping_or_sequence():
    by_map = simple_trajectories_from_dots(
        [0.0, 1.0, 2.0], [7.0, 7.0, 7.2], ['vibrato', 'cosine'],
        vibratos=[VIB, None])
    by_seq = simple_trajectories_from_dots(
        [0.0, 1.0, 2.0], [7.0, 7.0, 7.2], [5, 1],
        vibratos=[(5.5, 0.05, 0.03, 1.0), None])
    assert by_map == by_seq
    v = by_map[0]
    assert v.type == 'vibrato'
    assert (v.rate, v.extent_start, v.extent_end, v.phase, v.vert_offset) == (
        5.5, 0.05, 0.03, 1.0, 0.0)
    assert by_map[1].rate is None
    # a 5-sequence carries the lean too
    leaning = simple_trajectories_from_dots(
        [0.0, 1.0], [7.0, 7.0], ['vibrato'],
        vibratos=[(5.5, 0.05, 0.03, 1.0, 0.01)])
    assert leaning[0].vert_offset == 0.01


def test_vibrato_mapping_may_omit_the_defaulted_numbers():
    chunks = simple_trajectories_from_dots(
        [0.0, 1.0], [7.0, 7.0], ['vibrato'],
        vibratos=[{'rate': 4.0, 'extent_start': 0.05}])
    assert (chunks[0].extent_end, chunks[0].phase) == (0.05, 0.0)


@pytest.mark.parametrize("types, vibratos, message", [
    (['vibrato'], None, "gives no numbers"),
    (['vibrato'], [None], "gives no numbers"),
    (['fixed'], [VIB], "not 'vibrato'"),
    (['vibrato'], [VIB, VIB], "vibratos"),
    (['vibrato'], [(5.5, 0.05)], "4- or 5-sequence"),
    (['vibrato'], [(5.5, 0.05, 0.03, 1.0, 0.0, 9.0)], "4- or 5-sequence"),
    (['vibrato'], [{'rate': 5.5, 'extent_start': 0.05, 'wobble': 1}], "unknown keys"),
    (['vibrato'], [{'rate': -1.0, 'extent_start': 0.05}], "rate must be > 0"),
])
def test_rejects_malformed_vibratos(types, vibratos, message):
    with pytest.raises(ValueError, match=message):
        simple_trajectories_from_dots([0.0, 1.0], [7.0, 7.0], types,
                                      vibratos=vibratos)


def test_vibrato_round_trips_with_decompose_trajectory():
    traj = Trajectory({
        'id': 13, 'pitches': [Pitch({'swara': 2})], 'dur_tot': 1.7,
        'vib_obj': {'rate': 6.1, 'extent_start': 0.04, 'extent_end': 0.06,
                    'vert_offset': 0.0, 'phase': 2.2},
    })
    original = decompose_trajectory(traj, start_time=1.0)
    assert [c.type for c in original] == ['vibrato']
    times = [original[0].start.time, original[0].end.time]
    log_freqs = [original[0].start.log_freq, original[0].end.log_freq]
    rebuilt = simple_trajectories_from_dots(
        times, log_freqs, ['vibrato'],
        vibratos=[[getattr(original[0], f) for f in
                   ('rate', 'extent_start', 'extent_end', 'phase')]])
    assert rebuilt == original
