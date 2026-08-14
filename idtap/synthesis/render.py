"""Offline rendering of Piece transcriptions to audio.

Per-instrument renderers that combine the control-signal extraction
(control.py) with the DSP kernels (kernels.py), plus track mixing and WAV
output. Entry point: synthesize_piece() / Piece.synthesize().
"""
from __future__ import annotations

import wave
from typing import List, Optional, Sequence

import numpy as np

from ..enums import Instrument
from . import kernels
from ._nb import warn_if_slow
from .control import (TrackControl, extract_track_control,
                      envelope_from_spans, cutoff_curve_with_dampens)
from .vowels import (DEFAULT_B, DEFAULT_F, FEMALE_FORMANT_SCALE,
                     SHWAH_TIME, vowel_targets)

DEFAULT_SR = 44100
DEFAULT_CONTROL_RATE = 200.0

SITAR_DAMPEN = 0.5           # loop-filter cutoff (web control default)
# Chikari ring shorter than the main string: they are thin high strings,
# struck as punctuation. At the web app's value they rang for 16.8 s —
# longer than the main string — so 419 strums smeared into a wash.
CHIKARI_CUTOFF = 0.2
# How far apart the rake reaches successive strings. A chikari strum is a
# flick of the little finger, not a strummed chord — the strings sound
# almost together. Varying it per strum is most of what stops hundreds of
# them sounding mechanical.
CHIKARI_STRUM_MIN = 0.003
CHIKARI_STRUM_MAX = 0.009
KS_AMP = 8.0
OUT_GAIN_COMP = 0.707        # -3 dB compensation (web outGain scaling)
KLATT_FLUTTER = 0.15         # Synths.vue sets 0.15 at playback
KLATT_OPEN_PHASE_RATIO = 0.7
KLATT_BREATHINESS_DB = -25.0
KLATT_CASCADE_VOICING_DB = 0.0
KLATT_CASCADE_ASPIRATION_DB = -25.0
KLATT_CASCADE_ASPIRATION_MOD = 0.5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _forward_fill_f0(f0: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Hold the last active f0 through silent regions (AudioParams hold their
    value in the web app, keeping strings ringing at the last frequency)."""
    out = f0.copy()
    last = 0.0
    for i in range(out.shape[0]):
        if active[i] and out[i] > 0:
            last = out[i]
        else:
            out[i] = last
    if last == 0.0:
        return out
    # back-fill leading zeros with first valid value
    first = 0.0
    for i in range(out.shape[0]):
        if out[i] > 0:
            first = out[i]
            break
    for i in range(out.shape[0]):
        if out[i] == 0.0:
            out[i] = first
        else:
            break
    return out


def _build_excitation(n_samples: int, sr: float, events,
                      seed_base: int) -> np.ndarray:
    """Sum pink-noise bursts into a sample buffer (sendBurst equivalent;
    amplitudes are doubled here as sendBurst did)."""
    exc = np.zeros(n_samples, dtype=np.float64)
    for e_idx, ev in enumerate(events):
        start = int(round(ev.time * sr))
        if start >= n_samples or start < 0:
            continue
        dur_n = max(int(round(ev.dur * sr)), 1)
        atk_n = max(int(round(ev.atk * sr)), 1)
        burst = kernels.pink_burst(dur_n, atk_n, ev.amp * 2.0,
                                   (seed_base + e_idx * 7919) % (2 ** 31))
        stop = min(start + dur_n, n_samples)
        exc[start:stop] += burst[:stop - start]
    return exc


def _n_samples(ctrl: TrackControl, sr: float) -> int:
    return int(round(ctrl.dur_tot * sr)) + 1


def _hop(sr: float, control_rate: float) -> float:
    return sr / control_rate


# ---------------------------------------------------------------------------
# per-instrument renderers
# ---------------------------------------------------------------------------

def render_sitar(piece, inst_idx: int, sr: float = DEFAULT_SR,
                 control_rate: float = DEFAULT_CONTROL_RATE,
                 params: Optional[dict] = None,
                 chikari_freqs: Optional[Sequence[float]] = None
                 ) -> np.ndarray:
    """Render a sitar track.

    ``params`` overrides the instrument's tone controls (see
    synthesis.fitting.SITAR_PARAMS); ``chikari_freqs`` overrides the
    chikari tuning, which is better measured from the recording than
    derived from the raga — see idtap-platform#40.
    """
    P = params or {}
    ctrl = extract_track_control(piece, inst_idx, 0, control_rate)
    n = _n_samples(ctrl, sr)
    hop = _hop(sr, control_rate)

    bright = P.get('brightness', kernels.KS_BRIGHTNESS)
    stiff = P.get('stiffness', kernels.KS_STIFFNESS)
    jaw = P.get('jawari', kernels.JAWARI_DEPTH)
    jaw_thr = P.get('jawari_threshold', kernels.JAWARI_THRESHOLD)

    # main string
    f0 = _forward_fill_f0(ctrl.f0, ctrl.active)
    cutoff = cutoff_curve_with_dampens(ctrl, SITAR_DAMPEN)
    if 't60_max' in P:
        # the damping control is a fraction of the ring time, so rescale it
        scale = P['t60_max'] / kernels.KS_T60_MAX
        cutoff = np.clip(cutoff * scale, 0.0, 1.0)
    exc = _build_excitation(n, sr, ctrl.bursts, seed_base=inst_idx * 1000 + 1)
    main = kernels.ks_string(f0, cutoff, exc, float(sr), hop, KS_AMP,
                             stiff, bright, jaw, jaw_thr)
    main = kernels.dc_blocker(main, float(sr))
    main = kernels.tracking_lowpass(main, f0 * 8.0, float(sr), hop)

    out = main

    # jor string (string_idx 1), bypasses the filters — as in the web app
    jor_ctrl = extract_track_control(piece, inst_idx, 1, control_rate)
    if any(s for s in jor_ctrl.spans):
        jor_f0 = _forward_fill_f0(jor_ctrl.f0, jor_ctrl.active)
        jor_cut = cutoff_curve_with_dampens(jor_ctrl, SITAR_DAMPEN)
        jor_exc = _build_excitation(n, sr, jor_ctrl.bursts,
                                    seed_base=inst_idx * 1000 + 101)
        jor = kernels.ks_string(jor_f0, jor_cut, jor_exc, float(sr), hop,
                                KS_AMP, stiff, bright, jaw, jaw_thr)
        out = out + jor[:out.shape[0]]

    # chikari strings: fixed-frequency KS loops strummed at chikari events
    # Chikari tuning: what the transcription says this instrument was
    # tuned to, falling back to the raga only if it says nothing.
    per_event = [e for e in ctrl.chikari_events if e.freqs]
    if chikari_freqs is not None:
        chik_freqs = sorted((f for f in chikari_freqs if f and f > 0),
                            reverse=True)
    elif per_event:
        chik_freqs = list(per_event[0].freqs)
    else:
        try:
            chik_freqs = sorted((f for f in piece.chikari_freqs(inst_idx)
                                 if f and f > 0), reverse=True)
        except Exception:
            chik_freqs = []
    if chik_freqs and ctrl.chikari_events:
        rng = np.random.default_rng(inst_idx * 7717 + 5)
        n_str = len(chik_freqs)
        # one excitation buffer per string, built from all the strums
        per_string = [[] for _ in range(n_str)]
        for ev in ctrl.chikari_events:
            freqs = list(ev.freqs) if ev.freqs else chik_freqs
            # How many strings the finger caught: usually all of them,
            # sometimes just the top pair. Never only one — a single
            # chikari does not read as a strum.
            n_hit = int(rng.choice([2, n_str], p=[0.3, 0.7])) \
                if n_str > 2 else n_str
            n_hit = min(n_hit, len(freqs))
            spread = float(rng.uniform(CHIKARI_STRUM_MIN,
                                       CHIKARI_STRUM_MAX))
            level = float(rng.uniform(0.75, 1.25))
            for rank in range(n_hit):
                f = freqs[rank]
                try:
                    idx = chik_freqs.index(f)
                except ValueError:
                    idx = min(rank, n_str - 1)
                # the rake reaches each string a little later, and the
                # finger loses energy as it crosses
                per_string[idx].append(type(ev)(
                    time=ev.time + rank * spread,
                    amp=ev.amp * level * (1.0 - 0.12 * rank),
                    atk=ev.atk, dur=ev.dur))
        chik_sum = np.zeros(n, dtype=np.float64)
        for s_idx, freq in enumerate(chik_freqs):
            if not per_string[s_idx]:
                continue
            f0_arr = np.full(ctrl.n_frames, freq, dtype=np.float64)
            chik_t60 = P.get('chikari_t60')
            cut_val = (CHIKARI_CUTOFF if chik_t60 is None
                       else min(max((chik_t60 - kernels.KS_T60_MIN)
                                    / (kernels.KS_T60_MAX
                                       - kernels.KS_T60_MIN), 0.0), 1.0))
            cut_arr = np.full(ctrl.n_frames, cut_val, dtype=np.float64)
            s_exc = _build_excitation(
                n, sr, per_string[s_idx],
                seed_base=inst_idx * 1000 + 200 + s_idx)
            chik = kernels.ks_string(
                f0_arr, cut_arr, s_exc, float(sr), hop,
                KS_AMP * P.get('chikari_level', 0.5),
                stiff, bright, jaw, jaw_thr)
            chik_sum += chik
        chik_sum = kernels.dc_blocker(chik_sum, float(sr))
        out = out + chik_sum

    # Sympathetic strings. A sitar carries eleven to thirteen taraf, tuned
    # to the raga being played — a sitarist retunes them when changing raga
    # — and they are never plucked, only set ringing through the bridge.
    # The raga supplies the tuning, so this follows the transcription.
    taraf_mix = P.get('taraf_mix', kernels.SITAR_TARAF_MIX)
    if taraf_mix > 0:
        try:
            sa = piece.raga.fundamental
            freqs = sorted(piece.raga.get_frequencies(low=sa * 0.95,
                                                      high=sa * 4.1))[:13]
        except Exception:
            freqs = []
        if freqs:
            symp = kernels.sympathetic_bank(
                out, np.asarray(freqs, dtype=np.float64),
                P.get('taraf_t60', kernels.SITAR_TARAF_T60),
                P.get('taraf_drive', kernels.SITAR_TARAF_DRIVE), float(sr))
            out = out + taraf_mix * symp

    # the gourd and soundboard
    body_mix = P.get('body_mix', kernels.SITAR_BODY_MIX)
    if body_mix > 0:
        freqs = kernels.SITAR_BODY_FREQS.copy()
        for j, key in enumerate(('body_f1', 'body_f2', 'body_f3',
                                 'body_f4', 'body_f5', 'body_f6')):
            if key in P:
                freqs[j] = P[key]
        out = kernels.body_resonators(out, freqs, kernels.SITAR_BODY_QS,
                                      body_mix, float(sr))

    return out * OUT_GAIN_COMP


def render_sarangi(piece, inst_idx: int, sr: float = DEFAULT_SR,
                   control_rate: float = DEFAULT_CONTROL_RATE,
                   params: Optional[dict] = None) -> np.ndarray:
    """Render a sarangi track; ``params`` overrides its tone controls
    (see synthesis.fitting.SARANGI_PARAMS)."""
    ctrl = extract_track_control(piece, inst_idx, 0, control_rate)
    n = _n_samples(ctrl, sr)
    hop = _hop(sr, control_rate)

    def _one_string(c: TrackControl, seed: int) -> np.ndarray:
        f0 = _forward_fill_f0(c.f0, c.active)
        bow = envelope_from_spans(c, ramp=0.01, level=0.5)
        gain = c.gain
        P = params or {}
        return kernels.sarangi_string(
            f0, bow, gain, n, float(sr), hop, seed,
            P.get('bow_position', kernels.SARANGI_BOW_POSITION),
            kernels.SARANGI_N_TARAF,
            P.get('bridge_damp', kernels.SARANGI_BRIDGE_DAMP),
            P.get('body_mix', kernels.SARANGI_BODY_MIX),
            P.get('bow_noise', kernels.SARANGI_BOW_NOISE))

    out = _one_string(ctrl, seed=inst_idx * 1000 + 11)

    second = extract_track_control(piece, inst_idx, 1, control_rate)
    if any(s for s in second.spans):
        out = out + _one_string(second, seed=inst_idx * 1000 + 12)
    return out


def render_vocal(piece, inst_idx: int, sr: float = DEFAULT_SR,
                 control_rate: float = DEFAULT_CONTROL_RATE,
                 uniform_vowel: bool = False,
                 consonants: bool = True,
                 vowel_space=None) -> np.ndarray:
    ctrl = extract_track_control(piece, inst_idx, 0, control_rate)
    n = _n_samples(ctrl, sr)
    hop = _hop(sr, control_rate)
    nf = ctrl.n_frames

    f0 = _forward_fill_f0(ctrl.f0, ctrl.active)

    # per-sample gain: automation curve * silence-boundary envelope
    env = envelope_from_spans(ctrl, ramp=0.01, level=1.0)
    gain = ctrl.gain * env

    # formant control arrays: defaults, overwritten from each span onward
    # (AudioParams hold their values between trajectories)
    formants = np.zeros((6, nf), dtype=np.float64)
    bws = np.zeros((6, nf), dtype=np.float64)
    for j in range(6):
        formants[j, :] = DEFAULT_F[j]
        bws[j, :] = DEFAULT_B[j]

    # Without a measured vowel space, scale the built-in (adult-male) table
    # to the voice type, so a female singer is not rendered with male
    # formants — which is what happened to every vocal track before this.
    formant_scale = 1.0
    if vowel_space is None:
        try:
            if piece.instrumentation[inst_idx] == Instrument.Vocal_F:
                formant_scale = FEMALE_FORMANT_SCALE
        except (AttributeError, IndexError):
            pass

    rate = ctrl.control_rate
    spans = sorted(ctrl.spans, key=lambda s: s.start)
    for s_idx, span in enumerate(spans):
        k0 = max(int(np.ceil(span.start * rate - 1e-9)), 0)
        if s_idx + 1 < len(spans):
            k_end = min(int(np.ceil(spans[s_idx + 1].start * rate)), nf)
        else:
            k_end = nf
        if k0 >= k_end:
            continue
        vowel = 'a' if uniform_vowel else span.vowel
        s0, s1 = vowel_targets(vowel)
        if formant_scale != 1.0:
            s0 = [v * formant_scale if j < 3 else v for j, v in enumerate(s0)]
            s1 = [v * formant_scale if j < 3 else v for j, v in enumerate(s1)]
        if vowel_space is not None:
            measured = vowel_space.get(vowel)
            if measured is not None:
                # a measured vowel is a single steady target: the built-in
                # table's two-point glide is a stand-in for a real one
                s0 = list(measured)
                s1 = list(measured)
        for k in range(k0, k_end):
            t = k / rate - span.start
            frac = t / SHWAH_TIME
            if frac > 1.0:
                frac = 1.0
            elif frac < 0.0:
                frac = 0.0
            for j in range(3):
                formants[j, k] = s0[j] + (s1[j] - s0[j]) * frac
                bws[j, k] = s0[3 + j] + (s1[3 + j] - s0[3 + j]) * frac

    # source-level / nasal / parallel-branch controls: neutral (cascade-only,
    # no nasals, parallel branch off) defaults, which reproduce the web
    # worklet exactly, then overwritten within consonant gesture windows.
    extra = kernels.default_extra_ctrl(
        nf,
        cascade_voicing_db=KLATT_CASCADE_VOICING_DB,
        cascade_aspiration_db=KLATT_CASCADE_ASPIRATION_DB,
        cascade_aspiration_mod=KLATT_CASCADE_ASPIRATION_MOD)
    par_formant_db = kernels.default_par_formant_db_ctrl(nf)

    # Silence the voice source outside sounding trajectories. The web app
    # relied on output gain alone, which leaves the synth ringing internally;
    # that is inaudible there but would leak into any window a consonant
    # gesture opens the gain for (e.g. a stop closure before a note).
    silent = ~ctrl.active
    if silent.any():
        extra[kernels.ROW_CASC_VOICING_DB, silent] = kernels.OFF_DB
        extra[kernels.ROW_CASC_ASPIRATION_DB, silent] = kernels.OFF_DB

    if consonants:
        from .gestures import apply_consonant_gestures
        apply_consonant_gestures(ctrl, formants, bws, extra, par_formant_db,
                                 gain, f0, float(sr))

    seed = inst_idx * 1000 + 21
    flutter_offset = float((seed * 2654435761) % 1000)
    return kernels.klatt_voice(
        f0, gain, formants, bws, extra, par_formant_db, n, float(sr), hop,
        KLATT_FLUTTER, KLATT_OPEN_PHASE_RATIO, KLATT_BREATHINESS_DB,
        seed, flutter_offset)


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------

def render_track(piece, inst_idx: int, sr: float = DEFAULT_SR,
                 control_rate: float = DEFAULT_CONTROL_RATE,
                 uniform_vowel: bool = False,
                 consonants: bool = True,
                 vowel_space=None) -> Optional[np.ndarray]:
    inst = piece.instrumentation[inst_idx]
    if inst == Instrument.Sitar:
        return render_sitar(piece, inst_idx, sr, control_rate)
    if inst == Instrument.Sarangi:
        return render_sarangi(piece, inst_idx, sr, control_rate)
    if inst in (Instrument.Vocal_M, Instrument.Vocal_F):
        return render_vocal(piece, inst_idx, sr, control_rate,
                            uniform_vowel=uniform_vowel,
                            consonants=consonants,
                            vowel_space=vowel_space)
    return None


def synthesize_piece(piece, out: Optional[str] = None,
                     tracks: Optional[Sequence[int]] = None,
                     sr: int = DEFAULT_SR,
                     control_rate: float = DEFAULT_CONTROL_RATE,
                     uniform_vowel: bool = False,
                     track_gains: Optional[Sequence[float]] = None,
                     consonants: bool = True,
                     vowel_space=None
                     ) -> np.ndarray:
    """Render a Piece to audio.

    Args:
        piece: the Piece to render.
        out: optional path; if given, a 16-bit PCM WAV file is written.
        tracks: instrument track indices to include (default: all).
        sr: output sample rate.
        control_rate: control-signal rate in Hz (default 200 = 5 ms hop,
            finer than the web app's 20 ms).
        uniform_vowel: render all vocal trajectories with the vowel 'a'.
        track_gains: optional per-selected-track linear gains applied after
            per-track peak normalization.
        consonants: render consonant gestures (closures, bursts, aspiration,
            nasal murmurs) from the trajectories' consonant annotations.
        vowel_space: optional VowelSpace of formants measured from a
            singer's own recording (see synthesis.formants); overrides the
            generic built-in vowel table.

    Returns:
        float64 numpy array of mono samples in [-1, 1].
    """
    warn_if_slow()
    if tracks is None:
        # legacy transcriptions can declare more instruments than they have
        # phrase-grid tracks; only render the tracks that carry music
        n_tracks = len(piece.instrumentation)
        grid = getattr(piece, 'phrase_grid', None)
        if grid is not None:
            n_tracks = min(n_tracks, len(grid))
        tracks = list(range(n_tracks))
    rendered: List[np.ndarray] = []
    for i, idx in enumerate(tracks):
        sig = render_track(piece, idx, sr, control_rate,
                           uniform_vowel=uniform_vowel,
                           consonants=consonants,
                           vowel_space=vowel_space)
        if sig is None:
            continue
        peak = float(np.max(np.abs(sig))) if sig.size else 0.0
        if peak > 0:
            sig = sig / peak * 0.5
        if track_gains is not None and i < len(track_gains):
            sig = sig * track_gains[i]
        rendered.append(sig)

    if not rendered:
        return np.zeros(1, dtype=np.float64)

    length = max(s.shape[0] for s in rendered)
    mix = np.zeros(length, dtype=np.float64)
    for sig in rendered:
        mix[:sig.shape[0]] += sig
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * 0.891  # ~ -1 dBFS
    if out is not None:
        write_wav(out, mix, sr)
    return mix


def write_wav(path: str, samples: np.ndarray, sr: int) -> None:
    """Write mono float samples in [-1, 1] as 16-bit PCM WAV."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())
