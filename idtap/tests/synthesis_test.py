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


def _band_energy(seg: np.ndarray, sr: float, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    return float(spec[(freqs >= lo) & (freqs <= hi)].sum())


def _klatt_ctrl(nf, freqs, bws_):
    """Formant/bandwidth control arrays for the kernel tests."""
    formants = np.zeros((6, nf))
    bandwidths = np.zeros((6, nf))
    for j, (f, b) in enumerate(zip(freqs, bws_)):
        formants[j, :] = f
        bandwidths[j, :] = b
    return formants, bandwidths


VOWEL_F = [620., 1220., 2550., 3168., 4135., 5020.]
VOWEL_B = [80., 50., 140., 102., 816., 596.]


def test_pink_burst_deterministic():
    b1 = kernels.pink_burst(441, 100, 1.0, 42)
    b2 = kernels.pink_burst(441, 100, 1.0, 42)
    b3 = kernels.pink_burst(441, 100, 1.0, 43)
    assert b1.shape == (441,)
    assert np.array_equal(b1, b2)
    assert not np.array_equal(b1, b3)
    assert np.max(np.abs(b1)) > 0
    # the attack still ramps in: the opening is far quieter than the body
    assert np.abs(b1[:20]).mean() < 0.4 * np.abs(b1).max()


def test_pink_burst_has_no_dc():
    """A pluck cannot displace the string on average, and the string loop
    integrates DC forever, so the excitation must carry none."""
    for seed in (1, 42, 1234):
        b = kernels.pink_burst(1000, 100, 1.0, seed)
        assert abs(b.mean()) < 1e-12 * max(np.abs(b).max(), 1.0)


def _t60(f0, ctrl_val, dur=12.0):
    n = int(SR * dur)
    nf = int(n / HOP) + 2
    exc = np.zeros(n)
    exc[:441] = kernels.pink_burst(441, 100, 1.0, 42)
    y = kernels.ks_string(np.full(nf, f0), np.full(nf, ctrl_val),
                          exc, SR, HOP, 8.0)
    y = kernels.dc_blocker(y, SR)
    w = int(0.05 * SR)
    m = len(y) // w * w
    env = np.sqrt((y[:m].reshape(-1, w) ** 2).mean(axis=1))
    below = np.where(env < env.max() / 1000.0)[0]
    return below[0] * 0.05 if len(below) else dur


def test_string_decay_follows_its_control():
    """The damping control sets decay time, and damping actually damps.

    The worklet drove its loop filter coefficient to zero for a 'dampen',
    which freezes the filter's state rather than damping the string.
    """
    free = _t60(146.8, 1.0)
    normal = _t60(146.8, 0.5)
    damped = _t60(146.8, 0.0)
    assert free > normal > damped
    assert damped < 0.5              # a dampen stops the note

    # A sitar rings, it does not plink — but how long is a fitted value,
    # so this checks the control does what it says rather than pinning a
    # number that a refit will move. Half damping should land near half
    # the free ring time; the measurement reads a little short because it
    # bins energy in 50 ms windows.
    expected = kernels.KS_T60_MIN + (kernels.KS_T60_MAX
                                     - kernels.KS_T60_MIN) * 0.5
    assert 0.7 * expected < normal < 1.2 * expected
    assert normal > 2.0


def test_long_plucked_render_stays_bounded():
    """Repeated plucks must not accumulate: the loop filter has unity DC
    gain, so a biased excitation grows without limit over a long render."""
    n = int(SR * 30)
    nf = int(n / HOP) + 2
    f0 = np.full(nf, 220.0)
    cutoff = np.full(nf, 0.5)
    exc = np.zeros(n)
    # a pluck every half second for 30 s
    for k in range(60):
        start = int(k * 0.5 * SR)
        burst = kernels.pink_burst(441, 100, 1.0, 100 + k)
        exc[start:start + 441] += burst
    y = kernels.ks_string(f0, cutoff, exc, SR, HOP, 8.0)
    assert np.all(np.isfinite(y))
    first = np.abs(y[:int(SR * 5)]).max()
    last = np.abs(y[-int(SR * 5):]).max()
    # the end must not be wildly louder than the beginning
    assert last < 5 * first


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


def test_sarangi_is_bowed_not_hissed():
    """The bow must drive the string into Helmholtz motion.

    The worklet this replaced excited a resonator with bandpassed noise,
    which cannot produce stick-slip motion however it is tuned. A bowed
    string locks into a steady harmonic series; noise does not.
    """
    n = int(SR * 3)
    nf = int(n / HOP) + 2
    y = kernels.sarangi_string(np.full(nf, 220.0), np.full(nf, 0.5),
                               np.full(nf, 1.0), n, SR, HOP, 7)
    assert np.all(np.isfinite(y))
    seg = y[int(SR * 1.2):]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    harmonic = 0.0
    for k in range(1, 13):
        m = (freqs > 220.0 * k * 0.985) & (freqs < 220.0 * k * 1.015)
        harmonic += float(spec[m].sum())
    assert harmonic / float(spec.sum()) > 0.8   # a tone, not a hiss
    # and the level holds steady, as a bowed note does
    w = int(0.05 * SR)
    m = len(seg) // w * w
    rms = np.sqrt((seg[:m].reshape(-1, w) ** 2).mean(axis=1))
    assert rms.std() / rms.mean() < 0.15


def test_no_bow_no_sound():
    """Lifting the bow stops the note; the old model hissed regardless."""
    n = int(SR * 1.5)
    nf = int(n / HOP) + 2
    y = kernels.sarangi_string(np.full(nf, 220.0), np.zeros(nf),
                               np.full(nf, 1.0), n, SR, HOP, 7)
    assert np.max(np.abs(y)) == 0.0


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
    formants, bws = _klatt_ctrl(nf, VOWEL_F, VOWEL_B)
    extra = kernels.default_extra_ctrl(nf)
    par_db = kernels.default_par_formant_db_ctrl(nf)
    y = kernels.klatt_voice(f0, gain, formants, bws, extra, par_db, n, SR,
                            HOP, 0.0, 0.7, -25.0, 3, 0.0)
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
    extra = kernels.default_extra_ctrl(nf)
    par_db = kernels.default_par_formant_db_ctrl(nf)
    y = kernels.klatt_voice(f0, gain, formants, bws, extra, par_db, n, SR,
                            HOP, 0.15, 0.7, -25.0, 3, 0.0)
    assert np.max(np.abs(y)) == 0.0


# Reference RMS of the cascade-only kernel (measured before the parallel
# branch and nasal filters were added). With the neutral defaults the noise
# draw order is unchanged, so the output is in fact bit-identical; the loose
# tolerance below only guards against a real change of character.
KLATT_NEUTRAL_REF_RMS = 0.9851024455231268


def test_klatt_neutral_defaults_match_cascade_only_reference():
    """Neutral (parallel off, no nasals) controls reproduce the old kernel."""
    n = int(SR * 1.0)
    nf = int(n / HOP) + 2
    f0 = np.full(nf, 200.0)
    gain = np.full(nf, 1.0)
    formants, bws = _klatt_ctrl(nf, VOWEL_F, VOWEL_B)
    extra = kernels.default_extra_ctrl(nf)
    par_db = kernels.default_par_formant_db_ctrl(nf)
    y = kernels.klatt_voice(f0, gain, formants, bws, extra, par_db, n, SR,
                            HOP, 0.0, 0.7, -25.0, 3, 0.0)
    assert np.all(np.isfinite(y))
    rms = float(np.sqrt(np.mean(y ** 2)))
    assert rms == pytest.approx(KLATT_NEUTRAL_REF_RMS, rel=0.1)
    # pitch is still the target f0
    seg = y[int(SR * 0.25):]
    ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
    lo, hi = int(SR / 400), int(SR / 100)
    lag = np.argmax(ac[lo:hi]) + lo
    assert abs(_cents(SR / lag, 200.0)) < 15.0


def test_klatt_parallel_frication_formant():
    """Frication-only segment: noise energy concentrates at parallel F2."""
    n = int(SR * 1.0)
    nf = int(n / HOP) + 2
    f0 = np.full(nf, 200.0)
    gain = np.full(nf, 1.0)
    fric_f = 2500.0
    formants, bws = _klatt_ctrl(
        nf, [500., fric_f, 3000., 3500., 4500., 5500.],
        [100., 200., 200., 200., 400., 400.])
    extra = kernels.default_extra_ctrl(nf)
    # voicing fully off in both branches
    extra[kernels.ROW_CASC_VOICING_DB, :] = kernels.OFF_DB
    extra[kernels.ROW_CASC_ASPIRATION_DB, :] = kernels.OFF_DB
    extra[kernels.ROW_FRICATION_DB, :] = 0.0
    extra[kernels.ROW_FRICATION_MOD, :] = 0.5
    par_db = kernels.default_par_formant_db_ctrl(nf)
    par_db[1, :] = 0.0  # only parallel F2 is unmuted
    y = kernels.klatt_voice(f0, gain, formants, bws, extra, par_db, n, SR,
                            HOP, 0.0, 0.7, kernels.OFF_DB, 5, 0.0)
    assert np.all(np.isfinite(y))
    seg = y[int(SR * 0.25):]
    assert np.sqrt(np.mean(seg ** 2)) > 1e-3
    peak = _spectral_peak(seg, SR, 1000.0, 5000.0)
    assert abs(peak - fric_f) < 250.0
    # energy really is concentrated in the F2 band, compared with equally
    # wide bands below and above it
    f2_band = _band_energy(seg, SR, fric_f - 400, fric_f + 400)
    assert f2_band > 3.0 * _band_energy(seg, SR, 700.0, 1500.0)
    assert f2_band > 3.0 * _band_energy(seg, SR, 3200.0, 4000.0)
    # per-period amplitude modulation: the second half of each F0 period is
    # attenuated by (1 - fricationMod)
    period = int(round(SR / 200.0))
    idx = np.arange(seg.shape[0]) % period
    first = seg[idx < period // 2]
    second = seg[idx >= period // 2]
    assert np.sqrt(np.mean(first ** 2)) > 1.3 * np.sqrt(np.mean(second ** 2))


def test_klatt_nasal_murmur():
    """Nasal formant + antiformant put a spectral dip at the antiformant."""
    n = int(SR * 1.0)
    nf = int(n / HOP) + 2
    f0 = np.full(nf, 125.0)
    gain = np.full(nf, 1.0)
    formants, bws = _klatt_ctrl(nf, VOWEL_F, VOWEL_B)
    par_db = kernels.default_par_formant_db_ctrl(nf)

    plain = kernels.default_extra_ctrl(nf)
    nasal = kernels.default_extra_ctrl(nf)
    nasal[kernels.ROW_NASAL_FORMANT_FREQ, :] = 270.0
    nasal[kernels.ROW_NASAL_FORMANT_BW, :] = 100.0
    nasal[kernels.ROW_NASAL_ANTIFORMANT_FREQ, :] = 800.0
    nasal[kernels.ROW_NASAL_ANTIFORMANT_BW, :] = 150.0

    def _run(extra):
        return kernels.klatt_voice(f0, gain, formants, bws, extra, par_db, n,
                                   SR, HOP, 0.0, 0.7, -25.0, 9, 0.0)

    y_plain = _run(plain)
    y_nasal = _run(nasal)
    assert np.all(np.isfinite(y_nasal))
    assert not np.allclose(y_plain, y_nasal)

    s_plain = y_plain[int(SR * 0.25):]
    s_nasal = y_nasal[int(SR * 0.25):]
    # relative energy in a band around the antiformant must drop sharply
    r_plain = (_band_energy(s_plain, SR, 700, 900)
               / _band_energy(s_plain, SR, 100, 3000))
    r_nasal = (_band_energy(s_nasal, SR, 700, 900)
               / _band_energy(s_nasal, SR, 100, 3000))
    assert r_nasal < 0.5 * r_plain


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


# ---------------------------------------------------------------------------
# consonant gestures
# ---------------------------------------------------------------------------

def _consonant_piece(ipa: str, vowel: str = 'a', lead_silence: float = 0.5,
                     dur: float = 1.0):
    """A vocal piece: silence, then one note carrying an onset consonant."""
    raga = Raga()
    sil = Trajectory({'id': 12, 'pitches': [Pitch()],
                      'dur_tot': lead_silence})
    note = Trajectory({'id': 0, 'pitches': [Pitch()], 'dur_tot': dur,
                       'vowel': vowel, 'start_consonant': ipa,
                       'start_consonant_ipa': ipa})
    phrases = [Phrase({'trajectories': [sil], 'dur_tot': lead_silence,
                       'raga': raga}),
               Phrase({'trajectories': [note], 'dur_tot': dur,
                       'raga': raga})]
    return Piece({'phrases': phrases, 'raga': raga,
                  'instrumentation': [Instrument.Vocal_M]})


def _phases(ipa: str, lead: float = 0.5):
    """Render an onset consonant; return (closure, vowel) segments."""
    piece = _consonant_piece(ipa, lead_silence=lead)
    sig = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE)
    rel = int(lead * SR)
    closure = sig[rel - int(0.04 * SR):rel]
    vowel = sig[rel + int(0.35 * SR):rel + int(0.85 * SR)]
    return sig, closure, vowel


def test_consonant_annotations_reach_spans():
    piece = _consonant_piece('m')
    ctrl = extract_track_control(piece, 0, 0, CONTROL_RATE)
    spans = [s for s in ctrl.spans if s.start_consonant]
    assert len(spans) == 1
    assert spans[0].start_consonant == 'm'


@pytest.mark.parametrize("ipa", ['k', 'kʰ', 'ʈ'])
def test_voiceless_stop_closure_is_silent(ipa):
    """A voiceless closure must be true silence, not just a gain dip."""
    _, closure, vowel = _phases(ipa)
    assert np.max(np.abs(closure)) == 0.0
    assert np.sqrt(np.mean(vowel ** 2)) > 0.1  # the vowel still sounds


def test_nasal_murmur_is_voiced_low_and_below_vowel():
    _, closure, vowel = _phases('m')
    c_rms = np.sqrt(np.mean(closure ** 2))
    v_rms = np.sqrt(np.mean(vowel ** 2))
    assert c_rms > 0.01                      # murmur is audible
    assert 0.15 < c_rms / v_rms < 0.75       # but clearly below the vowel
    # murmur energy is overwhelmingly low-frequency
    lo = _band_energy(closure, SR, 100, 600)
    hi = _band_energy(closure, SR, 2500, 9000)
    assert hi < 0.05 * lo


def test_voiced_stop_has_quiet_voice_bar():
    _, closure, vowel = _phases('g')
    c_rms = np.sqrt(np.mean(closure ** 2))
    v_rms = np.sqrt(np.mean(vowel ** 2))
    assert 0.0 < c_rms < 0.2 * v_rms


def test_fricative_constriction_is_high_frequency():
    _, closure, _ = _phases('ʃ')
    lo = _band_energy(closure, SR, 100, 600)
    hi = _band_energy(closure, SR, 2500, 9000)
    assert hi > lo


def test_consonants_flag_disables_gestures():
    piece = _consonant_piece('k')
    on = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE,
                      consonants=True)
    off = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE,
                       consonants=False)
    assert not np.allclose(on, off)
    # without gestures there is no closure carved out before the note
    rel = int(0.5 * SR)
    assert np.max(np.abs(on[rel - int(0.04 * SR):rel])) == 0.0


# ---------------------------------------------------------------------------
# measured vowel spaces
# ---------------------------------------------------------------------------

def test_female_voice_renders_with_higher_formants_by_default():
    """With no measured space, a female track must not get male formants."""
    male = render_track(_build_piece(Instrument.Vocal_M, vowel='a'), 0,
                        sr=int(SR), control_rate=CONTROL_RATE)
    female = render_track(_build_piece(Instrument.Vocal_F, vowel='a'), 0,
                          sr=int(SR), control_rate=CONTROL_RATE)
    assert not np.array_equal(male, female)

    def peak(x, lo, hi):
        seg = x[int(0.2 * SR):int(0.9 * SR)]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1 / SR)
        m = (freqs >= lo) & (freqs <= hi)
        return float(freqs[m][np.argmax(spec[m])])

    assert peak(female, 900, 1900) > peak(male, 900, 1900)


def test_measured_space_supersedes_voice_scaling():
    from idtap.synthesis.formants import VowelSpace
    piece = _build_piece(Instrument.Vocal_F, vowel='a')
    space = VowelSpace(targets={'a': (620.0, 1220.0, 2550.0, 80., 50., 140.)},
                       counts={'a': 99}, voice='female', measured=('a',))
    with_space = render_track(piece, 0, sr=int(SR),
                              control_rate=CONTROL_RATE, vowel_space=space)
    default = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE)
    # the measured targets are the unscaled male values, so the render must
    # differ from the female-scaled default
    assert not np.allclose(with_space, default)


def test_delta_f_recovers_uniform_tube_length():
    """A textbook uniform tube must give back its own length."""
    from idtap.synthesis.formants import (_delta_f, vocal_tract_length_cm)
    # 17.5 cm tube: F1..F3 = 500, 1500, 2500 Hz
    delta = _delta_f([(500.0, 1500.0, 2500.0)])
    assert delta == pytest.approx(1000.0, rel=0.01)
    assert vocal_tract_length_cm(delta) == pytest.approx(17.5, rel=0.02)


def test_implausible_vtl_falls_back_to_voice_default(tmp_path: Path):
    """An estimate outside physiology must be refused, not applied.

    On real Hindustani recordings this estimator returns male-length tracts
    for female singers, because pooled-formant estimation assumes varied
    vowels and this repertoire sustains one.
    """
    from idtap.synthesis import formants as F
    lo, hi = F.PLAUSIBLE_VTL_CM['female']
    assert lo < hi < F.PLAUSIBLE_VTL_CM['male'][1]
    # a female voice measuring an 18 cm tract is a failure, not a singer
    delta_for_18cm = F.SOUND_SPEED_CMS / (2.0 * 18.0)
    assert not (lo <= F.vocal_tract_length_cm(delta_for_18cm) <= hi)


def test_scaled_default_space_raises_female_formants():
    from idtap.synthesis.formants import scaled_default_space
    male = scaled_default_space('male')
    female = scaled_default_space('female')
    m, f = male.get('a'), female.get('a')
    assert m is not None and f is not None
    # a shorter vocal tract puts every formant higher
    for j in range(3):
        assert f[j] > m[j]
    # bandwidths are not scaled
    assert f[3] == m[3]


def test_vowel_space_round_trip(tmp_path: Path):
    from idtap.synthesis.formants import VowelSpace, scaled_default_space
    space = scaled_default_space('female')
    space.source = 'test'
    p = tmp_path / 'space.json'
    space.save(str(p))
    back = VowelSpace.load(str(p))
    assert back.voice == 'female'
    assert back.source == 'test'
    assert back.get('a') == pytest.approx(space.get('a'))
    assert 'vowel space' in back.summary()


def test_vowel_space_overrides_formants_in_render():
    from idtap.synthesis.formants import VowelSpace
    piece = _build_piece(Instrument.Vocal_F, vowel='a')
    plain = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE)
    # an extreme, unmistakable vowel target
    space = VowelSpace(targets={'a': (300.0, 2400.0, 3200.0, 80., 90., 120.)},
                       counts={'a': 99}, voice='female')
    shifted = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE,
                           vowel_space=space)
    assert not np.allclose(plain, shifted)
    seg = shifted[int(0.2 * SR):int(0.9 * SR)]
    # energy should now sit around the imposed F2 rather than the default
    near_f2 = _band_energy(seg, SR, 2200, 2600)
    default_f2 = _band_energy(seg, SR, 1100, 1400)
    assert near_f2 > default_f2


def test_unknown_vowel_falls_back_to_table():
    from idtap.synthesis.formants import VowelSpace
    piece = _build_piece(Instrument.Vocal_M, vowel='a')
    space = VowelSpace(targets={'ō': (400., 800., 2400., 80., 90., 120.)},
                       counts={'ō': 5}, voice='male')
    sig = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE,
                       vowel_space=space)
    assert np.all(np.isfinite(sig))
    assert np.max(np.abs(sig)) > 0


def test_more_instruments_than_phrase_tracks():
    """Legacy transcriptions can declare instruments they have no phrase
    grid for; those tracks are skipped instead of raising IndexError."""
    raga = Raga()
    note = Trajectory({'id': 0, 'pitches': [Pitch()], 'dur_tot': 1.0})
    phrase = Phrase({'trajectories': [note], 'dur_tot': 1.0, 'raga': raga})
    piece = Piece({'phrases': [phrase], 'raga': raga,
                   'instrumentation': [Instrument.Vocal_F,
                                       Instrument.Sarangi]})
    assert len(piece.phrase_grid) < len(piece.instrumentation)
    # the absent track yields empty control signals rather than raising
    ctrl = extract_track_control(piece, 1, 0, CONTROL_RATE)
    assert ctrl.spans == []
    mix = synthesize_piece(piece, sr=22050)
    assert np.all(np.isfinite(mix))
    assert np.max(np.abs(mix)) > 0.0


def test_unknown_consonant_symbol_is_ignored():
    piece = _consonant_piece('not-a-phone')
    sig = render_track(piece, 0, sr=int(SR), control_rate=CONTROL_RATE)
    assert np.all(np.isfinite(sig))
    assert np.max(np.abs(sig)) > 0.0
