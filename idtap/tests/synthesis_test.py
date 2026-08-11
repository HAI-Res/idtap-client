import os
import sys
sys.path.insert(0, os.path.abspath("."))

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from idtap.classes.piece import Piece
from idtap.classes.phrase import Phrase
from idtap.classes.trajectory import Trajectory
from idtap.classes.pitch import Pitch
from idtap.classes.raga import Raga
from idtap.classes.articulation import Articulation
from idtap.enums import Instrument
from idtap.synthesis import (synthesize_piece, extract_track_control,
                             render_track, write_wav)
from idtap.synthesis import kernels
from idtap.synthesis.control import (envelope_from_spans,
                                     cutoff_curve_with_dampens)

SR = 44100.0
CONTROL_RATE = 200.0
HOP = SR / CONTROL_RATE


# ---------------------------------------------------------------------------
# kernel-level tests
# ---------------------------------------------------------------------------

def _spectral_peak(seg: np.ndarray, sr: float, fmin: float = 0.0,
                   fmax: float = 1e9) -> float:
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    mask = (freqs >= fmin) & (freqs <= fmax)
    idx = np.where(mask)[0]
    p = idx[np.argmax(spec[mask])]
    if 0 < p < len(spec) - 1:
        a, b, c = spec[p - 1], spec[p], spec[p + 1]
        denom = a - 2 * b + c
        if denom != 0:
            p = p + 0.5 * (a - c) / denom
    return float(p * sr / len(seg))


def _cents(f: float, ref: float) -> float:
    return 1200.0 * math.log2(f / ref)


def test_pink_burst_deterministic():
    b1 = kernels.pink_burst(441, 100, 1.0, 42)
    b2 = kernels.pink_burst(441, 100, 1.0, 42)
    b3 = kernels.pink_burst(441, 100, 1.0, 43)
    assert b1.shape == (441,)
    assert np.array_equal(b1, b2)
    assert not np.array_equal(b1, b3)
    assert b1[0] == 0.0  # attack ramp starts at zero
    assert np.max(np.abs(b1)) > 0


@pytest.mark.parametrize("target", [220.0, 440.0, 880.0])
def test_ks_string_tuning(target):
    n = int(SR * 2)
    nf = int(n / HOP) + 2
    f0 = np.full(nf, target)
    cutoff = np.full(nf, 0.5)
    exc = np.zeros(n)
    exc[:441] = kernels.pink_burst(441, 100, 1.0, 42)
    y = kernels.ks_string(f0, cutoff, exc, SR, HOP, 8.0)
    y = kernels.dc_blocker(y, SR)
    seg = y[int(SR * 0.25):int(SR * 1.95)]
    peak = _spectral_peak(seg, SR, target * 0.8, target * 1.2)
    assert abs(_cents(peak, target)) < 3.0


def test_sarangi_string_tuning():
    n = int(SR * 4)
    nf = int(n / HOP) + 2
    f0 = np.full(nf, 220.0)
    bow = np.full(nf, 0.5)
    gain = np.full(nf, 1.0)
    y = kernels.sarangi_string(f0, bow, gain, n, SR, HOP, 7)
    seg = y[int(SR * 1.0):]
    peak = _spectral_peak(seg, SR, 220 * 0.85, 220 * 1.15)
    # noise-excited resonance wanders a little; ±12 cents is a real failure
    assert abs(_cents(peak, 220.0)) < 12.0
    assert np.all(np.isfinite(y))


def test_klatt_voice_pitch_and_formant():
    n = int(SR * 2)
    nf = int(n / HOP) + 2
    f0 = np.full(nf, 200.0)
    gain = np.full(nf, 1.0)
    formants = np.zeros((6, nf))
    bws = np.zeros((6, nf))
    for j, (f, b) in enumerate(zip([620, 1220, 2550, 3168, 4135, 5020],
                                   [80, 50, 140, 102, 816, 596])):
        formants[j, :] = f
        bws[j, :] = b
    y = kernels.klatt_voice(f0, gain, formants, bws, n, SR, HOP,
                            0.0, 0.7, -25.0, 0.0, -25.0, 0.5, 3, 0.0)
    seg = y[int(SR * 0.5):int(SR * 1.5)]
    # pitch via autocorrelation (flutter disabled above)
    ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
    lo, hi = int(SR / 400), int(SR / 100)
    lag = np.argmax(ac[lo:hi]) + lo
    assert abs(_cents(SR / lag, 200.0)) < 15.0
    # F1 energy: spectrum should peak near 620 Hz (strongest formant)
    peak = _spectral_peak(seg, SR, 300, 1000)
    # peak lands on a harmonic of 200 near F1 (600 or 800)
    assert 400 < peak < 900
    assert np.all(np.isfinite(y))


def test_klatt_silence_is_silent():
    n = int(SR * 0.5)
    nf = int(n / HOP) + 2
    f0 = np.zeros(nf)
    gain = np.zeros(nf)
    formants = np.tile(np.array([520., 1006., 2831., 3168., 4135., 5020.]),
                       (nf, 1)).T.copy()
    bws = np.tile(np.array([76., 102., 72., 102., 816., 596.]),
                  (nf, 1)).T.copy()
    y = kernels.klatt_voice(f0, gain, formants, bws, n, SR, HOP,
                            0.15, 0.7, -25.0, 0.0, -25.0, 0.5, 3, 0.0)
    assert np.max(np.abs(y)) == 0.0


# ---------------------------------------------------------------------------
# control extraction
# ---------------------------------------------------------------------------

def _build_piece(instrument, vowel=None, with_pluck=False):
    raga = Raga()  # fundamental 261.63
    arts = {}
    if with_pluck:
        arts['0.00'] = Articulation({'name': 'pluck'})
    t1 = Trajectory({'id': 0, 'pitches': [Pitch()], 'dur_tot': 1.0,
                     'articulations': arts,
                     **({'vowel': vowel} if vowel else {})})
    t2 = Trajectory({'id': 12, 'pitches': [Pitch()], 'dur_tot': 1.0})
    p1 = Phrase({'trajectories': [t1], 'dur_tot': 1.0, 'raga': raga})
    p2 = Phrase({'trajectories': [t2], 'dur_tot': 1.0, 'raga': raga})
    return Piece({'phrases': [p1, p2], 'raga': raga,
                  'instrumentation': [instrument]})


def test_extract_track_control_basic():
    piece = _build_piece(Instrument.Sitar, with_pluck=True)
    ctrl = extract_track_control(piece, 0, 0, CONTROL_RATE)
    assert ctrl.dur_tot == pytest.approx(2.0)
    # first second active at the raga fundamental, second second silent
    k_mid = int(0.5 * CONTROL_RATE)
    k_late = int(1.5 * CONTROL_RATE)
    assert ctrl.active[k_mid]
    assert not ctrl.active[k_late]
    assert ctrl.f0[k_mid] == pytest.approx(261.63, rel=1e-3)
    assert ctrl.f0[k_late] == 0.0
    # one pluck at t=0
    assert len(ctrl.bursts) == 1
    assert ctrl.bursts[0].time == pytest.approx(0.0)
    # one span, silence on both sides
    assert len(ctrl.spans) == 1
    span = ctrl.spans[0]
    assert span.from_sil and span.to_sil
    assert span.start == pytest.approx(0.0)
    assert span.end == pytest.approx(1.0)


def test_envelope_from_spans_ramps():
    piece = _build_piece(Instrument.Sarangi)
    ctrl = extract_track_control(piece, 0, 0, CONTROL_RATE)
    env = envelope_from_spans(ctrl, ramp=0.01, level=0.5)
    k_mid = int(0.5 * CONTROL_RATE)
    assert env[k_mid] == pytest.approx(0.5)
    assert env[int(1.5 * CONTROL_RATE)] == 0.0
    # ramps at the boundaries stay within [0, level]
    assert np.all(env >= 0.0) and np.all(env <= 0.5 + 1e-9)


def test_cutoff_curve_with_dampens():
    piece = _build_piece(Instrument.Sitar)
    ctrl = extract_track_control(piece, 0, 0, CONTROL_RATE)
    ctrl.dampen_times.append(0.5)
    cur = cutoff_curve_with_dampens(ctrl, 0.5)
    assert cur[0] == pytest.approx(0.5)
    # fully damped between 25ms and 50ms after the event
    k = int((0.5 + 0.04) * CONTROL_RATE)
    assert cur[k] == 0.0
    # recovered afterwards
    assert cur[int((0.5 + 0.2) * CONTROL_RATE)] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# end-to-end rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("instrument", [Instrument.Sitar, Instrument.Sarangi,
                                        Instrument.Vocal_M])
def test_render_track_end_to_end(instrument):
    vowel = 'a' if instrument == Instrument.Vocal_M else None
    piece = _build_piece(instrument, vowel=vowel, with_pluck=True)
    sig = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE)
    assert sig is not None
    assert sig.shape[0] == pytest.approx(2.0 * SR, abs=2)
    assert np.all(np.isfinite(sig))
    # active half should carry energy; some instruments ring into the
    # silent half (sitar sustain), so only assert presence, not absence
    first_half = sig[:int(1.0 * SR)]
    assert np.sqrt(np.mean(first_half ** 2)) > 1e-5


def test_synthesize_piece_writes_wav(tmp_path: Path):
    piece = _build_piece(Instrument.Sitar, with_pluck=True)
    out = tmp_path / 'render.wav'
    mix = piece.synthesize(out=str(out), sr=int(SR))
    assert out.exists()
    assert np.max(np.abs(mix)) <= 1.0
    with wave.open(str(out), 'rb') as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == int(SR)
        assert wf.getnframes() == mix.shape[0]


def test_synthesize_fixture_piece():
    fixture = Path(__file__).parent / 'fixtures' / 'serialization_test.json'
    data = json.loads(fixture.read_text())
    piece = Piece.from_json(data)
    mix = synthesize_piece(piece, sr=22050)
    assert np.all(np.isfinite(mix))
    assert mix.shape[0] > 22050  # at least a second of audio
    assert np.max(np.abs(mix)) > 0.01
