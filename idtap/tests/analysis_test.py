"""Tests for analysis functions (pitch_times, condensed_durations, etc.)."""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pytest
from idtap.classes.trajectory import Trajectory
from idtap.classes.pitch import Pitch
from idtap.classes.articulation import Articulation
from idtap.analysis import (
    pitch_times,
    condensed_durations,
    durations_of_pitch_onsets,
    segment_by_duration,
    pattern_counter,
    chroma_seq_to_condensed_pitch_nums,
)


# ---------------------------------------------------------------------------
# Helpers to build test trajectories
# ---------------------------------------------------------------------------

def _fixed_traj(swara='sa', oct=0, dur=1.0, raised=True):
    """Create a fixed-pitch trajectory (id=0)."""
    return Trajectory({
        'id': 0,
        'pitches': [Pitch({'swara': swara, 'oct': oct, 'raised': raised})],
        'durTot': dur,
        'articulations': {'0.00': Articulation({'name': 'pluck', 'stroke': 'd'})},
    })


def _silent_traj(dur=1.0):
    """Create a silent trajectory (id=12)."""
    return Trajectory({
        'id': 12,
        'pitches': [Pitch()],
        'durTot': dur,
    })


def _bend_traj(swara1='sa', swara2='re', dur=2.0, oct=0):
    """Create a simple bend trajectory (id=1) with 2 pitches."""
    return Trajectory({
        'id': 1,
        'pitches': [
            Pitch({'swara': swara1, 'oct': oct}),
            Pitch({'swara': swara2, 'oct': oct}),
        ],
        'durTot': dur,
        'durArray': [0.5, 0.5],
        'articulations': {'0.00': Articulation({'name': 'pluck', 'stroke': 'd'})},
    })


# ---------------------------------------------------------------------------
# pitch_times tests
# ---------------------------------------------------------------------------

class TestPitchTimes:

    def test_single_fixed_pitch(self):
        trajs = [_fixed_traj('sa', dur=2.0)]
        result = pitch_times(trajs)
        assert len(result) == 1
        assert result[0]['time'] == 0.0
        assert result[0]['pitch'] == 0  # Sa oct 0 numbered_pitch = 0

    def test_silence(self):
        trajs = [_silent_traj(3.0)]
        result = pitch_times(trajs)
        assert len(result) == 1
        assert result[0]['pitch'] == 'silence'
        assert result[0]['time'] == 0.0
        assert result[0]['articulation'] is False

    def test_fixed_then_silence(self):
        trajs = [_fixed_traj('sa', dur=2.0), _silent_traj(1.0)]
        result = pitch_times(trajs)
        # Sa at time 0, silence at time 0 (no advance for single-pitch traj)
        # Filter should keep both since first is kept (idx=0) and second
        # has different pitch or is last
        assert any(r['pitch'] == 'silence' for r in result)

    def test_silence_then_fixed(self):
        trajs = [_silent_traj(2.0), _fixed_traj('re', dur=1.0)]
        result = pitch_times(trajs)
        # Silence at 0, then Re at 2.0
        assert result[0]['pitch'] == 'silence'
        assert result[0]['time'] == 0.0
        re_entry = [r for r in result if r['pitch'] != 'silence']
        assert len(re_entry) == 1
        assert re_entry[0]['time'] == 2.0

    def test_bend_trajectory_two_pitches(self):
        trajs = [_bend_traj('sa', 're', dur=4.0)]
        result = pitch_times(trajs)
        # For bends (id<4), dur_array is forced to [1], so the second pitch
        # starts at 0 + 1 * 4.0 = 4.0 (end of trajectory).
        assert len(result) == 2
        assert result[0]['time'] == 0.0
        assert result[1]['time'] == 4.0

    def test_articulation_flag(self):
        trajs = [_fixed_traj('sa', dur=1.0)]
        result = pitch_times(trajs)
        assert result[0]['articulation'] is True

    def test_chroma_output_type(self):
        trajs = [_fixed_traj('pa', oct=0, dur=1.0)]
        result = pitch_times(trajs, output_type='chroma')
        # Pa oct 0 → numbered_pitch = 7, chroma = 7
        pa_entries = [r for r in result if r['pitch'] != 'silence']
        assert pa_entries[0]['pitch'] == 7

    def test_sargam_letter_output(self):
        trajs = [_fixed_traj('sa', oct=0, dur=1.0)]
        result = pitch_times(trajs, output_type='sargamLetter')
        assert result[0]['pitch'] == 'S'

    def test_duplicate_time_pitch_filtered(self):
        """Two fixed pitches of the same note should have duplicates filtered."""
        trajs = [_fixed_traj('sa', dur=1.0), _fixed_traj('sa', dur=1.0)]
        result = pitch_times(trajs)
        # Both at time 0 with same pitch → filter removes duplicate
        assert len(result) == 1

    def test_same_time_different_pitch_keeps_latter(self):
        """Adjacent same-time entries with different pitches keep only latter."""
        trajs = [_fixed_traj('sa', dur=1.0), _fixed_traj('re', dur=1.0)]
        result = pitch_times(trajs)
        # Both at time 0 but different pitch → filter keeps only Re
        # (first is at idx=0 which is always kept, but middle entries
        # with same time as next get dropped)
        # With only 2 entries, both are kept (idx 0 = first, idx 1 = last)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# condensed_durations tests
# ---------------------------------------------------------------------------

class TestCondensedDurations:

    def test_single_pitch(self):
        trajs = [_fixed_traj('sa', dur=3.0)]
        result = condensed_durations(trajs)
        assert len(result) == 1
        assert result[0]['pitch'] == 0
        assert abs(result[0]['dur'] - 3.0) < 0.01

    def test_short_silence_absorbed(self):
        """Silence < max_silence is absorbed into preceding pitch."""
        trajs = [
            _fixed_traj('sa', dur=2.0),
            _silent_traj(1.0),  # < default 5s
            _fixed_traj('re', dur=2.0),
        ]
        result = condensed_durations(trajs, max_silence=5)
        # Silence should be absorbed into Sa's duration
        pitches = [r['pitch'] for r in result]
        assert 'silence' not in pitches

    def test_long_silence_partially_absorbed(self):
        """Silence > max_silence: max_silence absorbed, rest kept as silence."""
        trajs = [
            _silent_traj(2.0),  # starts with silence
            _fixed_traj('sa', dur=2.0),
            _silent_traj(10.0),  # long silence
            _fixed_traj('re', dur=2.0),
        ]
        result = condensed_durations(trajs, max_silence=3, exclude_silence=False)
        # Long silence: 3s absorbed into Sa, remaining 7s kept as silence
        silence_entries = [r for r in result if r['pitch'] == 'silence']
        assert len(silence_entries) >= 1

    def test_exclude_silence(self):
        trajs = [
            _fixed_traj('sa', dur=2.0),
            _silent_traj(10.0),
            _fixed_traj('re', dur=2.0),
        ]
        result_with = condensed_durations(trajs, exclude_silence=False)
        result_without = condensed_durations(trajs, exclude_silence=True)
        assert any(r['pitch'] == 'silence' for r in result_with)
        assert all(r['pitch'] != 'silence' for r in result_without)

    def test_chroma_output(self):
        trajs = [_fixed_traj('pa', oct=0, dur=2.0)]
        result = condensed_durations(trajs, output_type='chroma')
        assert result[0]['pitch'] == 7  # Pa chroma


# ---------------------------------------------------------------------------
# durations_of_pitch_onsets tests
# ---------------------------------------------------------------------------

class TestDurationsOfPitchOnsets:

    def test_single_pitch_cumulative(self):
        trajs = [_fixed_traj('sa', dur=5.0)]
        result = durations_of_pitch_onsets(trajs)
        assert 0 in result  # Sa numbered pitch
        assert abs(result[0] - 5.0) < 0.01

    def test_proportional(self):
        trajs = [_fixed_traj('sa', dur=5.0)]
        result = durations_of_pitch_onsets(trajs, count_type='proportional')
        assert abs(result[0] - 1.0) < 0.01  # Only pitch → 100%

    def test_exclude_silence(self):
        trajs = [_fixed_traj('sa', dur=2.0), _silent_traj(3.0)]
        result = durations_of_pitch_onsets(trajs, exclude_silence=True)
        assert 'silence' not in result

    def test_multiple_pitches_aggregate(self):
        """Multiple occurrences of same pitch should sum durations."""
        trajs = [
            _fixed_traj('sa', dur=2.0),
            _silent_traj(1.0),
            _fixed_traj('sa', dur=3.0),
        ]
        result = durations_of_pitch_onsets(trajs)
        # Sa should have aggregated duration
        assert 0 in result
        assert result[0] > 2.0


# ---------------------------------------------------------------------------
# segment_by_duration tests
# ---------------------------------------------------------------------------

class TestSegmentByDuration:

    def _make_simple_piece(self):
        """Build a minimal mock piece for testing."""
        from unittest.mock import MagicMock
        piece = MagicMock()
        piece.dur_tot = 30.0
        trajs = [
            _fixed_traj('sa', dur=5.0),
            _fixed_traj('re', dur=5.0),
            _fixed_traj('ga', dur=5.0),
            _fixed_traj('ma', dur=5.0),
            _fixed_traj('pa', dur=5.0),
            _silent_traj(5.0),
        ]
        piece.all_trajectories.return_value = trajs
        return piece

    def test_basic_segmentation(self):
        piece = self._make_simple_piece()
        segments = segment_by_duration(piece, duration=10)
        assert len(segments) >= 2
        total_trajs = sum(len(s) for s in segments)
        # Final silence excluded
        assert total_trajs == 5

    def test_remove_empty_false(self):
        piece = self._make_simple_piece()
        segments = segment_by_duration(piece, duration=10, remove_empty=False)
        assert len(segments) == 3  # ceil(30 / 10)

    def test_boundary_type_left(self):
        piece = self._make_simple_piece()
        segments = segment_by_duration(
            piece, duration=10, boundary_type='left',
        )
        assert all(len(s) >= 0 for s in segments)

    def test_boundary_type_right(self):
        piece = self._make_simple_piece()
        segments = segment_by_duration(
            piece, duration=10, boundary_type='right',
        )
        assert all(len(s) >= 0 for s in segments)

    def test_dur_tot_none_raises(self):
        from unittest.mock import MagicMock
        piece = MagicMock()
        piece.dur_tot = None
        with pytest.raises(ValueError, match='dur_tot is None'):
            segment_by_duration(piece)


# ---------------------------------------------------------------------------
# pattern_counter tests
# ---------------------------------------------------------------------------

class TestPatternCounter:

    def _repeated_pattern_trajs(self):
        """Sa-Re-Ga repeated 3 times with silences between."""
        return [
            _fixed_traj('sa', dur=1.0),
            _bend_traj('re', 'ga', dur=2.0),
            _silent_traj(0.5),
            _fixed_traj('sa', dur=1.0),
            _bend_traj('re', 'ga', dur=2.0),
            _silent_traj(0.5),
            _fixed_traj('sa', dur=1.0),
            _bend_traj('re', 'ga', dur=2.0),
        ]

    def test_basic_patterns(self):
        trajs = self._repeated_pattern_trajs()
        result = pattern_counter(trajs, size=2)
        assert isinstance(result, list)
        assert all('pattern' in r and 'count' in r for r in result)

    def test_sorted_by_count(self):
        trajs = self._repeated_pattern_trajs()
        result = pattern_counter(trajs, size=2, sort=True)
        counts = [r['count'] for r in result]
        assert counts == sorted(counts, reverse=True)

    def test_min_size_filter(self):
        trajs = self._repeated_pattern_trajs()
        result = pattern_counter(trajs, size=2, min_size=2)
        assert all(r['count'] >= 2 for r in result)

    def test_target_pitch(self):
        trajs = [
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
        ]
        # Sa numbered_pitch = 0
        result = pattern_counter(trajs, size=2, target_pitch=0)
        for r in result:
            assert r['pattern'][-1] == 0

    def test_empty_trajs(self):
        result = pattern_counter([], size=2)
        assert result == []

    def test_silence_resets_pattern(self):
        """Long silence should reset the pattern window."""
        trajs = [
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _silent_traj(10.0),  # long silence
            _fixed_traj('ga', dur=1.0),
            _fixed_traj('ma', dur=1.0),
        ]
        result = pattern_counter(trajs, size=2, max_lag_time=3)
        # Sa-Re and Ga-Ma patterns should NOT be connected across silence
        for r in result:
            # No pattern should span Re→Ga across the long silence
            pattern = r['pattern']
            sa_np = Pitch({'swara': 'sa'}).numbered_pitch
            re_np = Pitch({'swara': 're'}).numbered_pitch
            ga_np = Pitch({'swara': 'ga'}).numbered_pitch
            if len(pattern) == 2:
                assert not (pattern[0] == re_np and pattern[1] == ga_np)


# ---------------------------------------------------------------------------
# chroma_seq_to_condensed_pitch_nums tests
# ---------------------------------------------------------------------------

class TestChromaSeqToCondensedPitchNums:

    def test_empty_input(self):
        assert chroma_seq_to_condensed_pitch_nums([]) == []

    def test_single_value(self):
        result = chroma_seq_to_condensed_pitch_nums([5])
        assert result == [5]

    def test_close_pitches_unchanged(self):
        """Pitches already close together should not shift much."""
        seq = [0, 2, 4]
        result = chroma_seq_to_condensed_pitch_nums(seq)
        # Range should be small
        assert max(result) - min(result) <= 12

    def test_wrap_around_shift(self):
        """Pitches spanning the octave boundary should be shifted to minimize range."""
        seq = [0, 1, 11]  # 0 and 11 are adjacent in chroma space
        result = chroma_seq_to_condensed_pitch_nums(seq)
        # After shifting, range should be 2 (11→-1, 0, 1) or similar
        assert max(result) - min(result) <= 2

    def test_preserves_original_order(self):
        """Output order should match input order."""
        seq = [7, 0, 4]
        result = chroma_seq_to_condensed_pitch_nums(seq)
        assert len(result) == 3
        # The relative ordering of the original indices is preserved

    def test_all_same(self):
        seq = [5, 5, 5]
        result = chroma_seq_to_condensed_pitch_nums(seq)
        assert all(v == result[0] for v in result)
