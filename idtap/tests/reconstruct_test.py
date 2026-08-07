import math

import pytest

from idtap.classes.piece import Piece
from idtap.classes.phrase import Phrase
from idtap.classes.raga import Raga
from idtap.classes.reconstruct import reconstruct_piece
from idtap.classes.simple_trajectory import OrientationDot, SimpleTrajectory
from idtap.classes.trajectory import Trajectory
from idtap.enums import Instrument


def vocal_pitches(raga, n=4):
    pitches = raga.get_pitches(low=200, high=500)
    return pitches[:n]


def make_vocal_piece():
    raga = Raga()
    p = vocal_pitches(raga)
    trajs = [
        Trajectory({'id': 0, 'pitches': [p[0]], 'dur_tot': 1.0,
                    'instrumentation': Instrument.Vocal_M}),
        Trajectory({'id': 1, 'pitches': [p[0], p[1]], 'dur_tot': 0.5,
                    'instrumentation': Instrument.Vocal_M}),
        Trajectory({'id': 2, 'pitches': [p[1], p[2]], 'dur_tot': 0.75,
                    'slope': 3.0, 'instrumentation': Instrument.Vocal_M}),
        Trajectory({'id': 4, 'pitches': [p[0], p[1], p[2]], 'dur_tot': 1.5,
                    'instrumentation': Instrument.Vocal_M}),
        Trajectory({'id': 6, 'pitches': [p[0], p[2], p[1], p[3]], 'dur_tot': 2.0,
                    'instrumentation': Instrument.Vocal_M}),
        Trajectory({'id': 12, 'pitches': [p[0]], 'dur_tot': 1.0,
                    'fund_id12': raga.fundamental,
                    'instrumentation': Instrument.Vocal_M}),
        Trajectory({'id': 5, 'pitches': [p[2], p[1], p[0]], 'dur_tot': 1.25,
                    'instrumentation': Instrument.Vocal_M}),
    ]
    phrase = Phrase({'trajectories': trajs, 'raga': raga,
                     'instrumentation': [Instrument.Vocal_M.value]})
    piece = Piece({'phrases': [phrase], 'raga': raga,
                   'instrumentation': [Instrument.Vocal_M]})
    return piece, raga


# ---------------------------------------------------------------- round trip

def test_round_trip_ids_and_durations():
    piece, raga = make_vocal_piece()
    chunks = piece.simplified_trajectories()
    rec = reconstruct_piece(chunks, raga, Instrument.Vocal_M, synthetic=True)

    assert len(rec.phrases) == 1
    orig = piece.all_trajectories()
    out = rec.all_trajectories()
    assert [t.id for t in out] == [t.id for t in orig]
    for a, b in zip(orig, out):
        assert b.dur_tot == pytest.approx(a.dur_tot)
        if a.dur_array and b.dur_array:
            assert b.dur_array == pytest.approx(a.dur_array)
    assert rec.dur_tot == pytest.approx(piece.dur_tot)


def test_round_trip_pitches_exact():
    piece, raga = make_vocal_piece()
    chunks = piece.simplified_trajectories()
    rec = reconstruct_piece(chunks, raga, Instrument.Vocal_M, synthetic=True)
    for a, b in zip(piece.all_trajectories(), rec.all_trajectories()):
        if a.id == 12:
            continue
        assert b.log_freqs == pytest.approx(a.log_freqs, abs=1e-9)


def test_round_trip_preserves_log_offset():
    raga = Raga()
    base = vocal_pitches(raga, 1)[0]
    offset = raga.pitch_from_log_freq(math.log2(base.frequency) + 0.013)
    traj = Trajectory({'id': 0, 'pitches': [offset], 'dur_tot': 1.0,
                       'instrumentation': Instrument.Vocal_M})
    phrase = Phrase({'trajectories': [traj], 'raga': raga,
                     'instrumentation': [Instrument.Vocal_M.value]})
    piece = Piece({'phrases': [phrase], 'raga': raga,
                   'instrumentation': [Instrument.Vocal_M]})
    rec = reconstruct_piece(piece.simplified_trajectories(), raga,
                            Instrument.Vocal_M, synthetic=True)
    assert rec.all_trajectories()[0].log_freqs[0] == pytest.approx(
        traj.log_freqs[0], abs=1e-12)


def test_round_trip_slopes():
    piece, raga = make_vocal_piece()
    chunks = piece.simplified_trajectories()
    rec = reconstruct_piece(chunks, raga, Instrument.Vocal_M, synthetic=True)
    orig = piece.all_trajectories()
    out = rec.all_trajectories()
    for a, b in zip(orig, out):
        if a.id in (2, 3, 4, 5):
            assert b.slope == pytest.approx(a.slope)


def test_vibrato_reconstructs_as_yoyo_with_same_curve_at_boundaries():
    raga = Raga()
    p = vocal_pitches(raga, 1)
    traj = Trajectory({'id': 13, 'pitches': p, 'dur_tot': 1.0,
                       'instrumentation': Instrument.Vocal_M})
    phrase = Phrase({'trajectories': [traj], 'raga': raga,
                     'instrumentation': [Instrument.Vocal_M.value]})
    piece = Piece({'phrases': [phrase], 'raga': raga,
                   'instrumentation': [Instrument.Vocal_M]})
    rec = reconstruct_piece(piece.simplified_trajectories(), raga,
                            Instrument.Vocal_M, synthetic=True)
    out = rec.all_trajectories()
    assert len(out) == 1
    assert out[0].id == 6
    n = 2 * traj.vib_obj['periods']
    for k in range(n + 1):
        assert out[0].compute(k / n) == pytest.approx(
            traj.compute(k / n), rel=1e-6)


def test_reconstructed_curve_matches_chunks():
    piece, raga = make_vocal_piece()
    chunks = piece.simplified_trajectories()
    rec = reconstruct_piece(chunks, raga, Instrument.Vocal_M, synthetic=True)
    starts = rec.traj_start_times()
    trajs = rec.all_trajectories()
    for chunk in chunks:
        if chunk.type == 'silent':
            continue
        for x in (0.1, 0.5, 0.9):
            t = chunk.start.time + x * chunk.dur_tot
            for traj, start in zip(trajs, starts):
                if start <= t <= start + traj.dur_tot and traj.id != 12:
                    rel = (t - start) / traj.dur_tot
                    assert traj.compute(rel) == pytest.approx(
                        chunk.compute(x), rel=1e-6)
                    break


# ------------------------------------------------------------- robustness

def chunk(type_, t0, t1, lf0=8.0, lf1=8.0, continuation=False, slope=2.0):
    return SimpleTrajectory(
        type=type_,
        start=OrientationDot(t0, None if type_ == 'silent' else lf0),
        end=OrientationDot(t1, None if type_ == 'silent' else lf1),
        slope=slope,
        continuation=continuation,
    )


def test_unmatched_group_splits_into_primitives():
    chunks = [
        chunk('fixed', 0.0, 1.0),
        chunk('cosine', 1.0, 2.0, 8.0, 8.2, continuation=True),
    ]
    rec = reconstruct_piece(chunks, Raga(), Instrument.Vocal_M, synthetic=True)
    assert [t.id for t in rec.all_trajectories()] == [0, 1]


def test_gap_between_groups_becomes_silence():
    chunks = [
        chunk('fixed', 0.0, 1.0),
        chunk('fixed', 1.5, 2.5),
    ]
    rec = reconstruct_piece(chunks, Raga(), Instrument.Vocal_M, synthetic=True)
    trajs = rec.all_trajectories()
    assert [t.id for t in trajs] == [0, 12, 0]
    assert trajs[1].dur_tot == pytest.approx(0.5)


def test_tiny_gap_is_absorbed():
    chunks = [
        chunk('fixed', 0.0, 1.0),
        chunk('fixed', 1.0000001, 2.0),
    ]
    rec = reconstruct_piece(chunks, Raga(), Instrument.Vocal_M, synthetic=True)
    assert [t.id for t in rec.all_trajectories()] == [0, 0]


def test_overlapping_chunks_raise():
    chunks = [
        chunk('fixed', 0.0, 1.0),
        chunk('fixed', 0.8, 2.0),
    ]
    with pytest.raises(ValueError, match="[Oo]verlap"):
        reconstruct_piece(chunks, Raga(), Instrument.Vocal_M, synthetic=True)


def test_zero_duration_chunks_are_dropped():
    chunks = [
        chunk('fixed', 0.0, 1.0),
        chunk('fixed', 1.0, 1.0),
        chunk('cosine', 1.0, 2.0, 8.0, 8.2),
    ]
    rec = reconstruct_piece(chunks, Raga(), Instrument.Vocal_M, synthetic=True)
    assert [t.id for t in rec.all_trajectories()] == [0, 1]


# ------------------------------------------------------------------- gates

def test_sitar_is_rejected():
    chunks = [chunk('fixed', 0.0, 1.0)]
    with pytest.raises(ValueError, match="vocal"):
        reconstruct_piece(chunks, Raga(), Instrument.Sitar, synthetic=True)


def test_sarangi_is_rejected():
    chunks = [chunk('fixed', 0.0, 1.0)]
    with pytest.raises(ValueError, match="sarangi"):
        reconstruct_piece(chunks, Raga(), Instrument.Sarangi, synthetic=True)


def test_instrument_accepts_string():
    chunks = [chunk('fixed', 0.0, 1.0)]
    rec = reconstruct_piece(chunks, Raga(), 'Vocal (F)', synthetic=True)
    assert rec.instrumentation == [Instrument.Vocal_F]


def test_provenance_must_be_declared():
    chunks = [chunk('fixed', 0.0, 1.0)]
    with pytest.raises(ValueError, match="provenance"):
        reconstruct_piece(chunks, Raga(), Instrument.Vocal_M)


def test_provenance_must_be_single():
    chunks = [chunk('fixed', 0.0, 1.0)]
    with pytest.raises(ValueError, match="provenance"):
        reconstruct_piece(chunks, Raga(), Instrument.Vocal_M,
                          audio_id='abc123', synthetic=True)


def test_audio_id_without_client_is_trusted():
    chunks = [chunk('fixed', 0.0, 1.0)]
    rec = reconstruct_piece(chunks, Raga(), Instrument.Vocal_M,
                            audio_id='abc123')
    assert rec.audio_id == 'abc123'
    assert rec.synthetic is False


def test_synthetic_piece_has_no_audio():
    chunks = [chunk('fixed', 0.0, 1.0)]
    rec = reconstruct_piece(chunks, Raga(), Instrument.Vocal_M, synthetic=True)
    assert rec.audio_id is None
    assert rec.synthetic is True


def test_audio_path_requires_client_and_metadata():
    chunks = [chunk('fixed', 0.0, 1.0)]
    with pytest.raises(ValueError, match="audio_path requires"):
        reconstruct_piece(chunks, Raga(), Instrument.Vocal_M,
                          audio_path='/tmp/x.wav')


def test_empty_chunks_rejected():
    with pytest.raises(ValueError, match="empty"):
        reconstruct_piece([], Raga(), Instrument.Vocal_M, synthetic=True)


def test_raga_by_name_without_client_falls_back():
    chunks = [chunk('fixed', 0.0, 1.0)]
    rec = reconstruct_piece(chunks, 'Yaman', Instrument.Vocal_M,
                            synthetic=True, fundamental=240.0)
    assert rec.raga.name == 'Yaman'
    assert rec.raga.fundamental == pytest.approx(240.0)


# ------------------------------------------------------- real transcription

def test_real_piece_vocal_track_round_trip():
    import json
    import os
    fixture = os.path.join(os.path.dirname(__file__), 'fixtures',
                           'serialization_test.json')
    with open(fixture) as f:
        piece = Piece.from_json(json.load(f))

    orig = piece.all_trajectories(0, 0)
    chunks = piece.simplified_trajectories(0, 0)
    rec = reconstruct_piece(chunks, piece.raga, piece.instrumentation[0],
                            synthetic=True)
    out = rec.all_trajectories()

    assert len(rec.phrases) == 1
    assert [t.id for t in out] == [6 if t.id == 13 else t.id for t in orig]
    assert rec.dur_tot == pytest.approx(sum(t.dur_tot for t in orig))
    for a, b in zip(orig, out):
        assert b.dur_tot == pytest.approx(a.dur_tot)

    starts = rec.traj_start_times()
    for c in chunks:
        if c.type == 'silent':
            continue
        for x in (0.25, 0.5, 0.75):
            t = c.start.time + x * c.dur_tot
            for traj, s in zip(out, starts):
                if traj.id != 12 and s <= t <= s + traj.dur_tot:
                    rel = (t - s) / traj.dur_tot
                    assert traj.compute(rel) == pytest.approx(
                        c.compute(x), rel=1e-9)
                    break

    # reconstructed piece must survive serialization
    rt = Piece.from_json(rec.to_json())
    assert len(rt.all_trajectories()) == len(out)
