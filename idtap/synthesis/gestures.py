"""Assembly of consonant gestures into Klatt control signals.

Turns the consonant annotations on a trajectory (``start_consonant_ipa`` /
``end_consonant_ipa``, surfaced as :class:`~idtap.synthesis.control.Span`
fields) into time-varying control data for the Klatt kernel, following
classic rule-based formant synthesis (Klatt 1980): each consonant is a
sequence of phases —

    closure  ->  release burst  ->  aspiration (VOT)  ->  formant transition

Formant transitions use locus theory: at the moment of release the oral
formants sit at the consonant's place-specific locus, then glide to the
following vowel's targets over ``transition_dur``. Nasals substitute a
nasal murmur (nasal formant + antiformant) for the closure silence, and
fricatives run the parallel branch's frication source instead of a burst.

Onset consonants straddle the trajectory boundary: the closure occupies
the time *before* the trajectory start so that the vowel lands on time,
which is how consonants are sung against a beat. The closure is never
allowed to consume more than half of the preceding trajectory.

Frication and burst resonances are voiced through parallel-branch formant
slot ``FRICATION_SLOT`` (the top slot, F6). The cascade shares the formant
frequency arrays, so this slot is restored to its default outside gesture
windows; during voiceless phases the cascade is silent anyway.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from . import kernels
from .consonants import ConsonantSpec, get_consonant
from .control import Span, TrackControl
from .vowels import DEFAULT_B, DEFAULT_F

# Oral tract closed: F1 drops to a very low, narrow resonance.
CLOSURE_F1 = 180.0
CLOSURE_F1_BW = 60.0

# Voicing level during a voiced stop's closure (the "voice bar").
VOICE_BAR_DB = -14.0
# Voicing level during a nasal murmur. Nasal murmurs sit roughly 6 dB below
# an adjacent vowel; the cascade's nasal formant adds considerable gain near
# f0, so the source is attenuated well past that difference to compensate.
NASAL_MURMUR_DB = -11.0
# Voicing level during a voiced fricative's constriction.
VOICED_FRICATIVE_DB = -8.0
# Extra breathiness for the breathy-voiced (ʱ) release.
BREATHY_ASPIRATION_DB = -12.0

# --- glottal /h/ -----------------------------------------------------------
# Voicing is reduced, not cut: a fully devoiced /h/ breaks the sung line.
H_VOICING_DB = -8.0
# F1 bandwidth during /h/. The open glottis couples the subglottal system and
# heavily damps F1 — Klatt (1980) widens B1 to about this for aspiration.
H_B1_HZ = 300.0
# Duration of the breathy onset before, and after, the vowel onset.
H_PRE_DUR = 0.040
H_POST_DUR = 0.045
# Aspiration level the voice rests at outside a gesture (render.py default).
KLATT_ASPIRATION_REST_DB = -25.0

# Parallel formant slot used for burst / frication resonance.
FRICATION_SLOT = 5

# Stop-release bursts are extremely short, so the parallel branch's peak-gain
# resonators make them read far louder than their nominal level suggests.
# This trim keeps a burst at or below the level of the following vowel; it is
# applied to release bursts only, never to sustained fricative constrictions.
BURST_TRIM_DB = -10.0

# Reference frequency for frication-source tilt compensation. Mid-band, so
# palato-alveolar frication (which Klatt's levels were calibrated against)
# is unchanged and only resonances well above it are lifted.
NOISE_REF_HZ = 3000.0
# Cap on that compensation, so a very high resonance cannot run away.
MAX_NOISE_COMP_DB = 18.0


def _noise_source_mag(freq: float, sr: float) -> float:
    """Magnitude response of klatt-syn's LP-filtered noise source at ``freq``.

    The frication and aspiration sources are low-pass filtered (a one-pole
    filter matched to Klatt's original 10 kHz / b=0.75 design), so the noise
    reaching a high resonance is far weaker than at a low one. Reproduced
    here from Klatt.ts's LpNoiseSource / LpFilter1.set.
    """
    old_b = 0.75
    g = (1.0 - old_b) / math.sqrt(
        1.0 - 2.0 * old_b * math.cos(2.0 * math.pi * 1000.0 / 10000.0)
        + old_b ** 2)
    w_ref = 2.0 * math.pi * NOISE_REF_HZ / sr
    q = (1.0 - g ** 2 * math.cos(w_ref)) / (1.0 - g ** 2)
    b = q - math.sqrt(q ** 2 - 1.0)
    a = 1.0 - b
    w = 2.0 * math.pi * freq / sr
    return a / math.sqrt(1.0 - 2.0 * b * math.cos(w) + b ** 2)


def _noise_tilt_comp_db(freq: float, sr: float) -> float:
    """dB to add so a frication level means the same audible strength
    wherever the resonance sits (Klatt did this per consonant by hand)."""
    if freq <= 0:
        return 0.0
    ref = _noise_source_mag(NOISE_REF_HZ, sr)
    here = _noise_source_mag(freq, sr)
    if here <= 0 or ref <= 0:
        return 0.0
    comp = 20.0 * math.log10(ref / here)
    return min(max(comp, 0.0), MAX_NOISE_COMP_DB)

# Fraction of a preceding trajectory an onset closure may consume.
MAX_PREV_ENCROACH = 0.5

# Shortest window the formants may take to approach a consonant's locus.
# The articulators move toward a constriction over a longer span than the
# constriction itself lasts — a tap's closure is ~20 ms, only four control
# frames, and gliding the formants that fast is heard as a click rather than
# as a consonant. The approach may therefore start before the closure does.
MIN_APPROACH_DUR = 0.045

# Time the velum takes to close after a nasal, over which nasal coupling
# is faded out rather than switched off.
DENASAL_DUR = 0.035

# Depth of the amplitude dip at a tap's contact.
TAP_DIP_DB = -13.0


def _frames(t0: float, t1: float, rate: float, nf: int) -> Tuple[int, int]:
    """Half-open control-frame range covering [t0, t1)."""
    k0 = max(int(np.ceil(t0 * rate - 1e-9)), 0)
    k1 = min(int(np.ceil(t1 * rate - 1e-9)), nf)
    return k0, k1


def _fill(arr: np.ndarray, k0: int, k1: int, value: float) -> None:
    if k1 > k0:
        arr[k0:k1] = value


def _silence_sources(extra: np.ndarray, k0: int, k1: int) -> None:
    """Turn every source off across a frame range (a voiceless closure)."""
    if k1 <= k0:
        return
    extra[kernels.ROW_CASC_VOICING_DB, k0:k1] = kernels.OFF_DB
    extra[kernels.ROW_CASC_ASPIRATION_DB, k0:k1] = kernels.OFF_DB
    extra[kernels.ROW_PAR_VOICING_DB, k0:k1] = kernels.OFF_DB
    extra[kernels.ROW_PAR_ASPIRATION_DB, k0:k1] = kernels.OFF_DB
    extra[kernels.ROW_FRICATION_DB, k0:k1] = kernels.OFF_DB


def _set_nasal(extra: np.ndarray, k0: int, k1: int,
               murmur: Tuple[float, float, float, float]) -> None:
    """Enable the nasal formant + antiformant across a frame range."""
    if k1 <= k0:
        return
    nf_freq, nf_bw, af_freq, af_bw = murmur
    extra[kernels.ROW_NASAL_FORMANT_FREQ, k0:k1] = nf_freq
    extra[kernels.ROW_NASAL_FORMANT_BW, k0:k1] = nf_bw
    extra[kernels.ROW_NASAL_ANTIFORMANT_FREQ, k0:k1] = af_freq
    extra[kernels.ROW_NASAL_ANTIFORMANT_BW, k0:k1] = af_bw


def _set_frication(extra: np.ndarray, par_db: np.ndarray,
                   formants: np.ndarray, bws: np.ndarray,
                   k0: int, k1: int, spec: ConsonantSpec,
                   sr: float, trim_db: float = 0.0) -> None:
    """Run the parallel branch's frication source across a frame range."""
    if k1 <= k0 or spec.burst_formant is None:
        return
    f, bw = spec.burst_formant
    comp = _noise_tilt_comp_db(f, sr)
    formants[FRICATION_SLOT, k0:k1] = f
    bws[FRICATION_SLOT, k0:k1] = bw
    par_db[FRICATION_SLOT, k0:k1] = 0.0
    extra[kernels.ROW_FRICATION_DB, k0:k1] = spec.frication_db + comp + trim_db
    extra[kernels.ROW_FRICATION_MOD, k0:k1] = 0.0
    # the bypass path carries the un-resonated source, so it is not tilted
    extra[kernels.ROW_PAR_BYPASS_DB, k0:k1] = spec.bypass_db + trim_db


def _glide_range(arr: np.ndarray, k0: int, k1: int, start: float,
                 end: float) -> None:
    """Linearly ramp ``arr`` from ``start`` at k0 to ``end`` at k1."""
    if k1 <= k0:
        return
    span = k1 - k0
    for k in range(k0, k1):
        arr[k] = start + (end - start) * ((k - k0 + 1) / span)


def _approach_locus(formants: np.ndarray, k0: int, k1: int,
                    locus: Tuple[float, float, float]) -> None:
    """Glide F1-F3 from wherever they were into the consonant's locus.

    Stepping straight to the locus puts a discontinuity in the formant
    tracks; when the closure is voiced (an approximant, a nasal murmur, a
    voiced stop's voice bar) that step is audible as a click. Approaching
    the locus over the closure is also what the articulators actually do.
    """
    if k1 <= k0 or k0 <= 0:
        if k1 > k0:
            for j in range(3):
                _fill(formants[j], k0, k1, locus[j])
        return
    for j in range(3):
        _glide_range(formants[j], k0, k1, formants[j, k0 - 1], locus[j])


def _glide_formants(formants: np.ndarray, bws: np.ndarray,
                    k0: int, k1: int,
                    locus: Tuple[float, float, float],
                    target_k: int) -> None:
    """Glide F1-F3 from ``locus`` at k0 to whatever the arrays already hold
    at ``target_k`` (the vowel target written by the caller)."""
    if k1 <= k0:
        return
    span = max(target_k - k0, 1)
    for j in range(3):
        end_f = formants[j, min(target_k, formants.shape[1] - 1)]
        end_bw = bws[j, min(target_k, bws.shape[1] - 1)]
        start_f = locus[j]
        for k in range(k0, k1):
            frac = min(max((k - k0) / span, 0.0), 1.0)
            formants[j, k] = start_f + (end_f - start_f) * frac
            bws[j, k] = end_bw


def _apply_onset(spec: ConsonantSpec, span: Span, prev: Optional[Span],
                 formants: np.ndarray, bws: np.ndarray, extra: np.ndarray,
                 par_db: np.ndarray, gain: np.ndarray, f0: np.ndarray,
                 rate: float, nf: int, sr: float) -> None:
    """Write an onset (pre-vocalic) consonant gesture."""
    release = span.start
    span_len = span.end - span.start

    # /h/ is a glottal fricative: no oral constriction (no locus) and no
    # noise resonance of its own (no burst formant) — it borrows the
    # following vowel's tract shape entirely.
    is_glottal_h = (spec.manner == 'fricative' and spec.locus is None
                    and spec.burst_formant is None)

    # The closure sits before the release. If the previous trajectory runs
    # right up to this one, the closure interrupts it (as a consonant does
    # in speech) — but never consume more than MAX_PREV_ENCROACH of it.
    closure_dur = spec.closure_dur
    asp_dur = spec.aspiration_dur
    if is_glottal_h:
        # no silent gap to hide behind, so a long /h/ just sounds like a
        # patch of noise stuck onto the note; keep the whole gesture brief
        # and weighted before the vowel onset
        closure_dur = min(closure_dur, H_PRE_DUR)
        asp_dur = min(asp_dur, H_POST_DUR)
    # The formants approach the locus over a window that may start earlier
    # than the closure, so short constrictions still glide rather than jump.
    approach_dur = max(closure_dur, spec.transition_dur, MIN_APPROACH_DUR)
    if prev is not None and prev.end > release - closure_dur:
        available = MAX_PREV_ENCROACH * max(release - prev.start, 0.0)
        closure_dur = min(closure_dur, available)
        approach_dur = min(approach_dur, available)
    closure_start = max(release - closure_dur, 0.0)
    approach_start = max(release - approach_dur, 0.0)

    # Post-release phases are clamped so they never exceed the trajectory.
    burst_dur = spec.burst_dur
    # the formants leave the locus no faster than they approached it
    trans_dur = max(spec.transition_dur, MIN_APPROACH_DUR)
    post = burst_dur + asp_dur + trans_dur
    if post > span_len * 0.9 and post > 0:
        scale = (span_len * 0.9) / post
        burst_dur *= scale
        asp_dur *= scale
        trans_dur *= scale

    t_burst_end = release + burst_dur
    t_asp_end = t_burst_end + asp_dur
    t_trans_end = t_asp_end + trans_dur

    k_clo, k_rel = _frames(closure_start, release, rate, nf)
    k_app, _ = _frames(approach_start, release, rate, nf)
    _, k_burst = _frames(release, t_burst_end, rate, nf)
    _, k_asp = _frames(t_burst_end, t_asp_end, rate, nf)
    _, k_trans = _frames(t_asp_end, t_trans_end, rate, nf)

    # The pre-span closure needs the output gain open and the pitch of the
    # upcoming note (a sung nasal onset is already on pitch).
    if k_rel > k_clo:
        k_ref = min(k_rel, nf - 1)
        gain_at_start = gain[k_ref]
        f0_at_start = f0[k_ref]
        for k in range(k_clo, k_rel):
            gain[k] = max(gain[k], gain_at_start)
            if f0_at_start > 0:
                f0[k] = f0_at_start

    # ---- closure ---------------------------------------------------------
    if is_glottal_h:
        # /h/ has no closure at all: the tract already stands in the
        # following vowel's shape and only the glottal source changes.
        # Klatt's recipe is to keep the vowel's formants, widen B1 (the open
        # glottis couples the subglottal system and damps F1), and replace
        # voicing with aspiration. Sung /h/ keeps some voicing rather than
        # devoicing completely, which reads as breathy rather than as a gap.
        # ramp in as well as out: switching the breath on in one frame is
        # itself an audible edge
        _glide_range(extra[kernels.ROW_CASC_ASPIRATION_DB], k_clo, k_rel,
                     KLATT_ASPIRATION_REST_DB, spec.aspiration_db)
        _fill(extra[kernels.ROW_CASC_ASPIRATION_MOD], k_clo, k_rel, 0.0)
        _glide_range(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_rel,
                     0.0, H_VOICING_DB)
        _glide_range(bws[0], k_clo, k_rel,
                     bws[0, k_clo - 1] if k_clo > 0 else H_B1_HZ, H_B1_HZ)
    else:
        _silence_sources(extra, k_clo, k_rel)
        if spec.locus is not None:
            _approach_locus(formants, k_app, k_rel, spec.locus)
        if spec.manner == 'nasal' and spec.nasal_murmur is not None:
            _set_nasal(extra, k_clo, k_rel, spec.nasal_murmur)
            _fill(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_rel,
                  NASAL_MURMUR_DB)
        elif spec.manner in ('fricative', 'affricate') and spec.burst_dur == 0:
            # a true fricative: constriction noise runs through the closure
            _set_frication(extra, par_db, formants, bws, k_clo, k_rel, spec, sr)
            if spec.voiced:
                _fill(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_rel,
                      VOICED_FRICATIVE_DB)
        elif spec.manner in ('stop', 'affricate'):
            # F1 falls to the closed-tract resonance; glide it for a voiced
            # closure so the voice bar does not start with a step
            if spec.voice_bar and k_clo > 0:
                _glide_range(formants[0], k_clo, k_rel,
                             formants[0, k_clo - 1], CLOSURE_F1)
            else:
                _fill(formants[0], k_clo, k_rel, CLOSURE_F1)
            _fill(bws[0], k_clo, k_rel, CLOSURE_F1_BW)
            if spec.voice_bar:
                _fill(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_rel,
                      VOICE_BAR_DB)
        elif spec.manner == 'trill':
            # Hindi /r/ is a tap: the tongue briefly touches the ridge, so
            # the note dips in amplitude. Without that dip the gesture is
            # only a fast formant swing, which reads as a click rather than
            # as a consonant. Ramp down and back up around the contact.
            mid = (k_clo + k_rel) // 2
            _glide_range(extra[kernels.ROW_CASC_VOICING_DB], k_clo, mid,
                         0.0, TAP_DIP_DB)
            _glide_range(extra[kernels.ROW_CASC_VOICING_DB], mid, k_rel,
                         TAP_DIP_DB, 0.0)
        else:
            # approximants / laterals: voiced continuant at the locus
            _fill(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_rel, 0.0)

    # ---- release burst ---------------------------------------------------
    if k_burst > k_rel:
        _set_frication(extra, par_db, formants, bws, k_rel, k_burst, spec,
                       sr, trim_db=BURST_TRIM_DB)
        if not spec.voiced:
            _fill(extra[kernels.ROW_CASC_VOICING_DB], k_rel, k_burst,
                  kernels.OFF_DB)
        if spec.locus is not None:
            for j in range(3):
                _fill(formants[j], k_rel, k_burst, spec.locus[j])

    # ---- aspiration / VOT ------------------------------------------------
    if k_asp > k_burst:
        _fill(extra[kernels.ROW_CASC_ASPIRATION_DB], k_burst, k_asp,
              spec.aspiration_db)
        _fill(extra[kernels.ROW_CASC_ASPIRATION_MOD], k_burst, k_asp, 0.0)
        if is_glottal_h:
            # stay breathy rather than devoicing, and keep F1 damped
            _fill(extra[kernels.ROW_CASC_VOICING_DB], k_burst, k_asp,
                  H_VOICING_DB)
            _fill(bws[0], k_burst, k_asp, H_B1_HZ)
        elif not spec.voiced:
            _fill(extra[kernels.ROW_CASC_VOICING_DB], k_burst, k_asp,
                  kernels.OFF_DB)
        elif spec.breathy_offset:
            _fill(extra[kernels.ROW_CASC_ASPIRATION_DB], k_burst, k_asp,
                  max(spec.aspiration_db, BREATHY_ASPIRATION_DB))

    # ---- formant transition to the vowel ---------------------------------
    if spec.locus is not None and k_trans > k_burst:
        _glide_formants(formants, bws, k_burst, k_trans, spec.locus, k_trans)
    # Denasalize gradually. The velum closes over a few tens of ms, so
    # cutting the nasal filters off in one frame leaves a spectral step at
    # the release. Sliding the antiformant onto the nasal formant cancels
    # the pair smoothly, after which both can be switched off.
    if (spec.manner == 'nasal' and spec.nasal_murmur is not None
            and k_trans > k_rel):
        nf_freq, nf_bw, af_freq, af_bw = spec.nasal_murmur
        k_den = min(k_rel + max(int(DENASAL_DUR * rate), 1), k_trans)
        _fill(extra[kernels.ROW_NASAL_FORMANT_FREQ], k_rel, k_den, nf_freq)
        _fill(extra[kernels.ROW_NASAL_FORMANT_BW], k_rel, k_den, nf_bw)
        _fill(extra[kernels.ROW_NASAL_ANTIFORMANT_BW], k_rel, k_den, af_bw)
        _glide_range(extra[kernels.ROW_NASAL_ANTIFORMANT_FREQ], k_rel, k_den,
                     af_freq, nf_freq)

    if is_glottal_h and k_trans > k_asp:
        # breathiness and the damped F1 resolve gradually into the vowel;
        # switching them off in one frame is what makes /h/ sound stuck on
        _glide_range(extra[kernels.ROW_CASC_ASPIRATION_DB], k_asp, k_trans,
                     spec.aspiration_db, KLATT_ASPIRATION_REST_DB)
        _glide_range(extra[kernels.ROW_CASC_VOICING_DB], k_asp, k_trans,
                     H_VOICING_DB, 0.0)
        _glide_range(bws[0], k_asp, k_trans, H_B1_HZ,
                     bws[0, min(k_trans, bws.shape[1] - 1)])
    if spec.breathy_offset and k_trans > k_asp:
        # breathy voice continues a little into the vowel
        _fill(extra[kernels.ROW_CASC_ASPIRATION_DB], k_asp, k_trans,
              BREATHY_ASPIRATION_DB)


def _apply_coda(spec: ConsonantSpec, span: Span,
                formants: np.ndarray, bws: np.ndarray, extra: np.ndarray,
                par_db: np.ndarray, rate: float, nf: int,
                sr: float) -> None:
    """Write a coda (post-vocalic) consonant gesture: the vowel's formants
    move to the consonant locus and the constriction closes. Codas are left
    unreleased (no burst), as they typically are in sung Hindi."""
    end = span.end
    span_len = span.end - span.start
    closure_dur = min(spec.closure_dur, span_len * 0.3)
    trans_dur = min(spec.transition_dur, span_len * 0.3)

    t_trans_start = end - closure_dur - trans_dur
    t_closure_start = end - closure_dur
    k_trans, k_clo = _frames(t_trans_start, t_closure_start, rate, nf)
    _, k_end = _frames(t_closure_start, end, rate, nf)

    # vowel formants glide toward the locus
    if spec.locus is not None and k_clo > k_trans:
        span_k = max(k_clo - k_trans, 1)
        for j in range(3):
            start_f = formants[j, k_trans]
            for k in range(k_trans, k_clo):
                frac = (k - k_trans) / span_k
                formants[j, k] = start_f + (spec.locus[j] - start_f) * frac

    if k_end <= k_clo:
        return
    if spec.locus is not None:
        for j in range(3):
            _fill(formants[j], k_clo, k_end, spec.locus[j])
    if spec.manner == 'nasal' and spec.nasal_murmur is not None:
        _set_nasal(extra, k_clo, k_end, spec.nasal_murmur)
        _fill(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_end,
              NASAL_MURMUR_DB)
    elif spec.manner in ('fricative', 'affricate'):
        _set_frication(extra, par_db, formants, bws, k_clo, k_end, spec, sr)
        if not spec.voiced:
            _fill(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_end,
                  kernels.OFF_DB)
    else:
        _fill(formants[0], k_clo, k_end, CLOSURE_F1)
        _fill(bws[0], k_clo, k_end, CLOSURE_F1_BW)
        if not spec.voice_bar:
            _silence_sources(extra, k_clo, k_end)
        else:
            _fill(extra[kernels.ROW_CASC_VOICING_DB], k_clo, k_end,
                  VOICE_BAR_DB)


def apply_consonant_gestures(ctrl: TrackControl, formants: np.ndarray,
                             bws: np.ndarray, extra: np.ndarray,
                             par_db: np.ndarray, gain: np.ndarray,
                             f0: np.ndarray, sr: float) -> int:
    """Write every consonant gesture of a track into its control arrays.

    The vowel formant targets must already be written into ``formants`` /
    ``bws``; gesture windows overwrite them and glide into whatever the
    arrays hold at the end of each transition.

    Returns the number of consonant gestures applied.
    """
    rate = ctrl.control_rate
    nf = ctrl.n_frames
    spans = sorted(ctrl.spans, key=lambda s: s.start)
    count = 0
    for idx, span in enumerate(spans):
        if span.start_consonant:
            spec = get_consonant(span.start_consonant)
            if spec is not None:
                prev = spans[idx - 1] if idx > 0 else None
                _apply_onset(spec, span, prev, formants, bws, extra,
                             par_db, gain, f0, rate, nf, sr)
                count += 1
        if span.end_consonant:
            spec = get_consonant(span.end_consonant)
            if spec is not None:
                _apply_coda(spec, span, formants, bws, extra, par_db,
                            rate, nf, sr)
                count += 1
    return count
