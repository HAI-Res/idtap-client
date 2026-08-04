"""Tests for visualization functions (plot_melodic_contour, etc.)."""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import pytest
from unittest.mock import MagicMock
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for tests
import matplotlib.pyplot as plt

from idtap.classes.trajectory import Trajectory
from idtap.classes.pitch import Pitch
from idtap.classes.phrase import Phrase
from idtap.classes.articulation import Articulation
from idtap.visualization import (
    plot_melodic_contour,
    plot_pitch_prevalence,
    plot_pitch_patterns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fixed_traj(swara='sa', oct=0, dur=1.0, raised=True):
    return Trajectory({
        'id': 0,
        'pitches': [Pitch({'swara': swara, 'oct': oct, 'raised': raised})],
        'durTot': dur,
        'articulations': {'0.00': Articulation({'name': 'pluck', 'stroke': 'd'})},
    })


def _silent_traj(dur=1.0):
    return Trajectory({
        'id': 12,
        'pitches': [Pitch()],
        'durTot': dur,
    })


def _bend_traj(swara1='sa', swara2='re', dur=2.0, oct=0):
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


def _sample_trajs():
    return [
        _fixed_traj('sa', dur=2.0),
        _bend_traj('re', 'ga', dur=3.0),
        _silent_traj(1.0),
        _fixed_traj('pa', dur=2.0),
    ]


def _mock_piece(trajs=None, dur_tot=None):
    """Build a minimal mock Piece."""
    if trajs is None:
        trajs = _sample_trajs()
    piece = MagicMock()
    piece.dur_tot = dur_tot or sum(t.dur_tot for t in trajs)

    # Build phrases with trajectory grids
    phrase = MagicMock()
    phrase.trajectory_grid = [trajs]
    phrase.is_section_start = True
    piece.phrase_grid = [[phrase]]
    piece.section_starts_grid = [[0]]
    piece.all_trajectories.return_value = trajs

    return piece


# ---------------------------------------------------------------------------
# plot_melodic_contour tests
# ---------------------------------------------------------------------------

class TestPlotMelodicContour:

    def test_returns_figure(self):
        trajs = _sample_trajs()
        fig = plot_melodic_contour(trajs)
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_with_existing_axes(self):
        fig, ax = plt.subplots()
        trajs = _sample_trajs()
        returned_fig = plot_melodic_contour(trajs, ax=ax)
        assert returned_fig is fig
        plt.close(fig)

    def test_empty_trajectories(self):
        fig = plot_melodic_contour([])
        assert fig is not None
        plt.close(fig)

    def test_show_consonants_draws_diamonds(self):
        traj = _fixed_traj('sa', dur=2.0)
        traj.add_consonant('ka')
        traj.add_consonant('ga', start=False)
        trajs = [traj, _fixed_traj('pa', dur=1.0)]

        fig, ax = plt.subplots()
        plot_melodic_contour(trajs, ax=ax, show_consonants=True)
        # One PathCollection per diamond: start + end of the first trajectory.
        from matplotlib.collections import PathCollection
        diamonds = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(diamonds) == 2
        starts = diamonds[0].get_offsets()
        ends = diamonds[1].get_offsets()
        assert starts[0][0] == pytest.approx(0.0)
        assert ends[0][0] == pytest.approx(2.0)
        plt.close(fig)

    def test_show_consonants_off_by_default(self):
        traj = _fixed_traj('sa', dur=2.0)
        traj.add_consonant('ka')
        fig, ax = plt.subplots()
        plot_melodic_contour([traj], ax=ax)
        from matplotlib.collections import PathCollection
        assert not [c for c in ax.collections if isinstance(c, PathCollection)]
        plt.close(fig)

    def test_only_silence(self):
        trajs = [_silent_traj(5.0)]
        fig = plot_melodic_contour(trajs)
        assert fig is not None
        plt.close(fig)

    def test_with_title(self):
        trajs = _sample_trajs()
        fig = plot_melodic_contour(trajs, title='Test Contour')
        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(self):
        trajs = _sample_trajs()
        fig = plot_melodic_contour(trajs, figsize=(8, 3))
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_pitch_prevalence tests
# ---------------------------------------------------------------------------

class TestPlotPitchPrevalence:

    def test_returns_figure_section(self):
        piece = _mock_piece()
        fig = plot_pitch_prevalence(piece, segmentation='section')
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_returns_figure_phrase(self):
        piece = _mock_piece()
        fig = plot_pitch_prevalence(piece, segmentation='phrase')
        assert fig is not None
        plt.close(fig)

    def test_returns_figure_duration(self):
        piece = _mock_piece()
        fig = plot_pitch_prevalence(piece, segmentation='duration')
        assert fig is not None
        plt.close(fig)

    def test_chroma_output(self):
        piece = _mock_piece()
        fig = plot_pitch_prevalence(piece, output_type='chroma')
        assert fig is not None
        plt.close(fig)

    def test_unknown_segmentation_raises(self):
        piece = _mock_piece()
        with pytest.raises(ValueError, match='Unknown segmentation'):
            plot_pitch_prevalence(piece, segmentation='unknown')

    def test_section_types_filter(self):
        trajs = _sample_trajs()
        phrase = MagicMock()
        phrase.trajectory_grid = [trajs]

        piece = MagicMock()
        piece.dur_tot = 16.0
        sections = []
        for sec_type in ('Improvisation', 'Composition'):
            sec = MagicMock()
            sec.phrases = [phrase]
            sec.categorization = {'Top Level': sec_type}
            sections.append(sec)
        piece.sections_grid = [sections]
        piece.section_starts_grid = [[0, 1]]
        piece.dur_starts.return_value = [0.0, 8.0]

        fig = plot_pitch_prevalence(piece, segmentation='section',
                                    section_types=['Improvisation'])
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert 'Improvisation' in texts
        assert 'Composition' not in texts
        plt.close(fig)

    def test_raga_only_pitch_labels(self):
        from idtap.classes.raga import Raga
        piece = _mock_piece()   # trajs span pitch numbers 0..7
        piece.raga = Raga()     # all-raised rule set: 0, 2, 4, 6, 7 in range
        piece.title = 'Test'
        fig = plot_pitch_prevalence(piece, segmentation='duration')
        texts = [t.get_text() for t in fig.axes[0].texts]
        for raga_letter in ('S', 'R', 'G', 'M', 'P'):
            assert raga_letter in texts
        for non_raga_letter in ('r', 'g', 'm'):   # chromatic rows 1, 3, 5
            assert non_raga_letter not in texts
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_pitch_patterns tests
# ---------------------------------------------------------------------------

class TestPlotPitchPatterns:

    def test_returns_figure_from_trajs(self):
        trajs = [
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
        ]
        fig = plot_pitch_patterns(trajs, pattern_size=2)
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_returns_figure_from_piece(self):
        trajs = [
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
        ]
        piece = _mock_piece(trajs)
        fig = plot_pitch_patterns(piece, pattern_size=2)
        assert fig is not None
        plt.close(fig)

    def test_empty_patterns(self):
        """Single trajectory can't form a pattern."""
        trajs = [_fixed_traj('sa', dur=1.0)]
        fig = plot_pitch_patterns(trajs, pattern_size=3)
        assert fig is not None
        plt.close(fig)

    def test_multiple_sizes(self):
        trajs = [
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
        ]
        fig = plot_pitch_patterns(trajs, pattern_sizes=[2, 3])
        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_with_contour_plot(self):
        trajs = [
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
        ]
        fig = plot_pitch_patterns(trajs, pattern_size=2, plot=True)
        assert fig is not None
        plt.close(fig)

    def test_max_patterns_limit(self):
        trajs = [
            _fixed_traj('sa', dur=1.0),
            _fixed_traj('re', dur=1.0),
            _fixed_traj('ga', dur=1.0),
        ] * 10
        fig = plot_pitch_patterns(trajs, pattern_size=2, max_patterns=5)
        assert fig is not None
        plt.close(fig)
