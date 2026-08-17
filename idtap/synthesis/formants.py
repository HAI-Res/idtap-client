"""Measuring a singer's own vowel space from their recordings.

The synthesis default is one generic formant table (see vowels.py), so
every voice comes out the same. This module replaces it with formants
measured from the singer's actual audio: the transcription already says
which vowel is being sung and exactly when, so those labels segment the
recording and Praat's Burg LPC tracker (via Parselmouth) reads the
formants out of each labelled vowel.

Measure the *isolated vocal stem*, not the raw recording — a tanpura drone
or sarangi in the signal will pull the tracker badly — and separate it with
a general-purpose vocal model rather than a karaoke one. See
``idtap.synthesis.formants.SEPARATION_NOTE``, and check
``VowelSpace.coverage`` on every result.

Typical use::

    space = measure_vowel_space('vocals.wav', piece, inst_idx=0)
    space.save('kesarbai.json')
    piece.synthesize(out='render.wav', vowel_space=space)

Requires parselmouth (``pip install praat-parselmouth``), which is an
optional dependency: only this module needs it.
"""
from __future__ import annotations

import json
import statistics
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..enums import Instrument
from .control import extract_track_control
from .vowels import DEFAULT_B, DEFAULT_F, VOWELS, vowel_targets

SEPARATION_NOTE = (
    "Formant measurement assumes a vocal stem with the accompaniment "
    "removed: run the recording through a denoiser and a vocal separator "
    "first. Prefer a general-purpose vocal model over a karaoke/lead-vocal "
    "one here. Karaoke models are built to send sustained drones to the "
    "accompaniment stem, which is right for a tanpura but can take a "
    "Hindustani vocalist's held alap tones with it — on one recording "
    "tested that left only 29% of the singing in the vocal stem, against "
    "98% from a general model. Always check VowelSpace.coverage."
)

# Praat Burg tracker settings. The formant ceiling has to suit the voice:
# too low and F3 is missed, too high and spurious poles appear.
MAX_FORMANT_HZ = {'male': 5000.0, 'female': 5500.0}
TIME_STEP = 0.01
N_FORMANTS = 5.0
WINDOW_LENGTH = 0.025
PRE_EMPHASIS = 50.0

# Only measure the stable interior of a vowel: the edges are transitions
# into and out of neighbouring consonants.
INTERIOR_FRACTION = 0.5
MIN_SPAN_DUR = 0.12          # shorter spans have no steady portion
# Frames (at TIME_STEP) a vowel needs before its median is trustworthy.
# Sung vowels are hard to track — a high f0 undersamples the spectral
# envelope — so a handful of frames is noise, not a measurement. 50 frames
# is half a second of steady vowel spread across the recording.
MIN_INSTANCES = 50
# Largest interquartile spread, relative to the median, at which a formant
# estimate is still taken as a measurement rather than tracker noise.
MAX_REL_IQR = 0.22
# Below this fraction of the transcription's vowel time being audible in the
# stem, the separation is assumed to have failed rather than the singer
# having been quiet — the transcription says they were singing.
MIN_COVERAGE = 0.6
# How far below a stem's loud end (95th percentile intensity) a frame may
# sit and still count as the singer rather than as residue.
SIGNAL_RANGE_DB = 25.0
# Plausibility gates, so octave errors and spurious poles are discarded.
FORMANT_RANGES = {
    'male': ((200.0, 900.0), (700.0, 2500.0), (1800.0, 3500.0)),
    'female': ((250.0, 1100.0), (800.0, 3000.0), (2000.0, 4000.0)),
}


def voice_type(instrument: Instrument) -> str:
    return 'female' if instrument == Instrument.Vocal_F else 'male'


@dataclass
class VowelSpace:
    """Measured formant targets for one singer, keyed by vowel label."""
    targets: Dict[str, Tuple[float, float, float, float, float, float]] = \
        field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    voice: str = 'male'
    source: str = ''
    # vowels that were measured, as opposed to filled from the default table
    measured: Tuple[str, ...] = ()
    # fraction of the transcription's own vowel time that carried usable
    # signal in the stem; low values mean the separation lost the singer
    coverage: float = 1.0

    def get(self, vowel: Optional[str]):
        """(f1,f2,f3,b1,b2,b3) for a vowel, or None if not held."""
        if vowel is None:
            return None
        return self.targets.get(vowel)

    def is_measured(self, vowel: str) -> bool:
        return vowel in self.measured

    def to_json(self) -> Dict:
        return {'voice': self.voice, 'source': self.source,
                'counts': self.counts, 'measured': list(self.measured),
                'coverage': self.coverage,
                'targets': {k: list(v) for k, v in self.targets.items()}}

    def save(self, path: str) -> None:
        with open(path, 'w') as fh:
            json.dump(self.to_json(), fh, indent=2)

    @staticmethod
    def load(path: str) -> 'VowelSpace':
        with open(path) as fh:
            obj = json.load(fh)
        return VowelSpace(
            targets={k: tuple(v) for k, v in obj.get('targets', {}).items()},
            counts=obj.get('counts', {}),
            voice=obj.get('voice', 'male'),
            source=obj.get('source', ''),
            measured=tuple(obj.get('measured', ())),
            coverage=obj.get('coverage', 1.0))

    def summary(self) -> str:
        rows = [f"vowel space ({self.voice}) from {self.source or 'unknown'}",
                f"  stem coverage of transcribed vowel time: {self.coverage:.0%}"]
        for v, t in sorted(self.targets.items(),
                           key=lambda kv: -self.counts.get(kv[0], 0)):
            tag = 'measured' if self.is_measured(v) else 'default '
            rows.append(f"  {v:>3}  {tag}  n={self.counts.get(v,0):5d}  "
                        f"F1={t[0]:6.0f} F2={t[1]:6.0f} F3={t[2]:6.0f}")
        return '\n'.join(rows)


def scaled_default_space(voice: str) -> VowelSpace:
    """Fallback when nothing has been measured.

    The built-in table is an adult-male one. Female formants run roughly a
    sixth higher for the same vowel (a shorter vocal tract), so scaling it
    is much closer than using it unchanged — but it is a stopgap, not a
    substitute for measuring the actual singer.
    """
    scale = 1.17 if voice == 'female' else 1.0
    targets = {}
    for v in VOWELS:
        s0, _ = vowel_targets(v)
        targets[v] = (s0[0] * scale, s0[1] * scale, s0[2] * scale,
                      s0[3], s0[4], s0[5])
    return VowelSpace(targets=targets, counts={v: 0 for v in VOWELS},
                      voice=voice, source='scaled default table')


# Speed of sound in the vocal tract, cm/s — the constant relating formant
# spacing to vocal tract length.
SOUND_SPEED_CMS = 35000.0

# Physiological range of adult vocal tract length, cm. An estimate outside
# this is a measurement failure, not an unusual singer.
PLAUSIBLE_VTL_CM = {'male': (15.0, 19.5), 'female': (12.5, 16.5)}


def _delta_f(samples: Sequence[Sequence[float]]) -> Optional[float]:
    """Mean formant spacing ΔF from pooled (F1,F2,F3) samples.

    For a uniform tube closed at the glottis, Fi = (2i-1)·ΔF/2, so
    regressing every observed formant against (2i-1) through the origin
    recovers ΔF without needing to know which vowel was being sung. This is
    the Reby-style estimator described by Anikin, Barreda & Reby (2023),
    Behavior Research Methods 56:5588, and it is deliberately label-free:
    it pools all vowels rather than treating them separately, so it holds up
    under the extreme vowel imbalance of this repertoire, where almost
    everything is sung on one vowel.
    """
    num = 0.0
    den = 0.0
    for row in samples:
        for i, f in enumerate(row):
            if f and f > 0:
                k = 2 * (i + 1) - 1
                num += k * f
                den += k * k
    if den <= 0:
        return None
    slope = num / den          # slope = ΔF / 2
    delta = 2.0 * slope
    return delta if delta > 0 else None


def vocal_tract_length_cm(delta_f: float) -> float:
    """Vocal tract length implied by a formant spacing."""
    return SOUND_SPEED_CMS / (2.0 * delta_f)


def _template_delta_f() -> float:
    """ΔF of the built-in (adult-male) vowel table, measured the same way,
    so a singer's ΔF can be expressed as a ratio against it."""
    rows = []
    for v in VOWELS:
        s0, _ = vowel_targets(v)
        rows.append(s0[:3])
    return _delta_f(rows) or 1000.0


def _vowel_intervals(piece, inst_idx: int,
                     control_rate: float = 200.0
                     ) -> List[Tuple[str, float, float]]:
    """(vowel, start, end) for the steady interior of each labelled vowel."""
    ctrl = extract_track_control(piece, inst_idx, 0, control_rate)
    out: List[Tuple[str, float, float]] = []
    for span in ctrl.spans:
        if not span.vowel:
            continue
        dur = span.end - span.start
        if dur < MIN_SPAN_DUR:
            continue
        # skip the onset consonant's transition and the final release
        margin = dur * (1.0 - INTERIOR_FRACTION) / 2.0
        out.append((span.vowel, span.start + margin, span.end - margin))
    return out


def measure_vowel_space(audio_path: str, piece, inst_idx: int = 0,
                        control_rate: float = 200.0,
                        voice: Optional[str] = None,
                        min_instances: int = MIN_INSTANCES,
                        verbose: bool = False) -> VowelSpace:
    """Measure a singer's vowel formants from an isolated vocal stem.

    Args:
        audio_path: vocal stem (accompaniment removed — see SEPARATION_NOTE).
        piece: the Piece whose vowel labels segment the recording.
        inst_idx: which vocal track to measure.
        voice: 'male'/'female'; inferred from the instrumentation if omitted.
        min_instances: vowels measured fewer times than this are dropped.

    Returns:
        VowelSpace holding the median formants of each vowel.
    """
    try:
        import parselmouth
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "measuring formants needs parselmouth: "
            "pip install praat-parselmouth") from exc

    if voice is None:
        try:
            voice = voice_type(piece.instrumentation[inst_idx])
        except (AttributeError, IndexError):
            voice = 'male'

    sound = parselmouth.Sound(audio_path)
    formant = sound.to_formant_burg(
        time_step=TIME_STEP,
        max_number_of_formants=N_FORMANTS,
        maximum_formant=MAX_FORMANT_HZ[voice],
        window_length=WINDOW_LENGTH,
        pre_emphasis_from=PRE_EMPHASIS)
    intensity = sound.to_intensity(minimum_pitch=75.0)

    # Ignore frames that are essentially silent in the stem. The reference
    # has to be the stem's loud end, not its median: a separation that lost
    # the singer produces a mostly-silent stem whose median is itself
    # silence, so a median-relative floor would pass that silence as signal
    # and report full coverage of a stem containing almost nothing.
    try:
        levels = [intensity.get_value(t) for t in intensity.xs()]
        levels = sorted(x for x in levels if x is not None and x == x)
        if levels:
            loud = levels[int(0.95 * (len(levels) - 1))]
            floor = loud - SIGNAL_RANGE_DB
        else:
            floor = 0.0
    except Exception:
        floor = 0.0

    lo, mid, hi = FORMANT_RANGES[voice]
    samples: Dict[str, List[Tuple[float, float, float]]] = {}
    expected = 0
    audible = 0
    for vowel, t0, t1 in _vowel_intervals(piece, inst_idx, control_rate):
        t = t0
        while t < t1:
            if t > sound.xmax:
                break
            expected += 1
            try:
                level = intensity.get_value(t)
            except Exception:
                level = None
            if level is not None and level == level and level >= floor:
                audible += 1
                f1 = formant.get_value_at_time(1, t)
                f2 = formant.get_value_at_time(2, t)
                f3 = formant.get_value_at_time(3, t)
                if (f1 == f1 and f2 == f2 and f3 == f3
                        and lo[0] <= f1 <= lo[1]
                        and mid[0] <= f2 <= mid[1]
                        and hi[0] <= f3 <= hi[1]
                        and f1 < f2 < f3):
                    samples.setdefault(vowel, []).append((f1, f2, f3))
            t += TIME_STEP

    # Start from a voice-appropriate table so vowels this recording never
    # sings often enough to measure are still not rendered with another
    # voice type's formants, then overwrite what was measured well.
    space = scaled_default_space(voice)
    space.source = audio_path
    counts: Dict[str, int] = {}
    measured: List[str] = []
    for vowel, vals in samples.items():
        counts[vowel] = len(vals)
        if len(vals) < min_instances:
            continue
        # Medians are robust to the tracker's occasional wild frame, but a
        # median is only meaningful if the frames agree. Accept each formant
        # separately, keeping the table's value where the tracker disagreed
        # with itself: F3 in particular is often unresolvable in singing,
        # and a confidently wrong F3 is worse than a generic one.
        s0, _ = vowel_targets(vowel)
        base = list(space.targets.get(vowel, tuple(s0)))
        accepted = []
        for j in range(3):
            series = sorted(v[j] for v in vals)
            med = statistics.median(series)
            q1 = series[len(series) // 4]
            q3 = series[(3 * len(series)) // 4]
            if med > 0 and (q3 - q1) / med <= MAX_REL_IQR:
                base[j] = med
                accepted.append(j)
        if not accepted:
            continue
        space.targets[vowel] = tuple(base)
        measured.append(vowel)
        if verbose:
            taken = ','.join(f'F{j + 1}' for j in accepted)
            print(f"  {vowel:>3}: n={len(vals):5d} accepted={taken:<8} "
                  f"F1={base[0]:6.0f} F2={base[1]:6.0f} F3={base[2]:6.0f}")
    space.counts = counts
    space.measured = tuple(measured)
    space.coverage = (audible / expected) if expected else 0.0
    if space.coverage < MIN_COVERAGE:
        warnings.warn(
            f"only {space.coverage:.0%} of this track's transcribed vowel "
            f"time is audible in {audio_path}. The transcription says the "
            f"singer was singing, so the separation has probably routed "
            f"them into the accompaniment stem — a karaoke/lead-vocal model "
            f"can do this to sustained singing, mistaking it for a drone. "
            f"Try a general-purpose vocal model before trusting these "
            f"formants.", RuntimeWarning, stacklevel=2)
    return space



def vowel_space_from_audio(audio_path: str, voice: str = 'male',
                           f0_percentile: float = 50.0,
                           verbose: bool = False) -> VowelSpace:
    """Characterize a singer from audio alone — no transcription needed.

    Instead of measuring each vowel separately (which needs labels saying
    which vowel is where), this pools formants over all voiced frames and
    estimates the singer's formant spacing dF, hence vocal tract length.
    The built-in vowel template is then warped by that ratio.

    That replaces ``scaled_default_space``'s guessed 1.17 female constant
    with a measurement of the actual singer, while keeping every vowel's
    relative position from the template — the right trade for this
    repertoire, where alap, tan and akar are sung on a single vowel by
    definition, so there is little varied-vowel material to measure even
    when labels do exist.

    Only the lower part of the singer's range is used: all-pole formant
    estimation is biased toward the nearest harmonic as f0 rises (JASA
    134(2):1295), and that bias is measurable in this material.

    Args:
        audio_path: isolated vocal stem (see SEPARATION_NOTE).
        voice: 'male' or 'female' — sets the formant ceiling only.
        f0_percentile: use frames at or below this percentile of the
            singer's f0 range; lower is less biased but has fewer frames.
    """
    try:
        import parselmouth
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "measuring formants needs parselmouth: "
            "pip install praat-parselmouth") from exc

    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    formant = sound.to_formant_burg(
        time_step=TIME_STEP, max_number_of_formants=N_FORMANTS,
        maximum_formant=MAX_FORMANT_HZ[voice],
        window_length=WINDOW_LENGTH, pre_emphasis_from=PRE_EMPHASIS)
    intensity = sound.to_intensity(minimum_pitch=75.0)

    levels = sorted(x for x in (intensity.get_value(t)
                                for t in intensity.xs())
                    if x is not None and x == x)
    floor = (levels[int(0.95 * (len(levels) - 1))] - SIGNAL_RANGE_DB
             if levels else 0.0)

    voiced = [(t, pitch.get_value_at_time(t)) for t in pitch.xs()]
    f0s = sorted(f for _, f in voiced if f is not None and f == f and f > 0)
    if not f0s:
        return scaled_default_space(voice)
    f0_cut = f0s[int((f0_percentile / 100.0) * (len(f0s) - 1))]

    lo, mid, hi = FORMANT_RANGES[voice]
    rows: List[Tuple[float, float, float]] = []
    for t, f0 in voiced:
        if f0 is None or f0 != f0 or f0 <= 0 or f0 > f0_cut:
            continue
        level = intensity.get_value(t) if t <= intensity.xmax else None
        if level is None or level != level or level < floor:
            continue
        f1 = formant.get_value_at_time(1, t)
        f2 = formant.get_value_at_time(2, t)
        f3 = formant.get_value_at_time(3, t)
        if (f1 == f1 and f2 == f2 and f3 == f3
                and lo[0] <= f1 <= lo[1] and mid[0] <= f2 <= mid[1]
                and hi[0] <= f3 <= hi[1] and f1 < f2 < f3):
            rows.append((f1, f2, f3))

    space = scaled_default_space(voice)
    space.source = audio_path
    space.counts = {'_voiced_frames': len(rows)}
    if len(rows) < MIN_INSTANCES:
        if verbose:
            print(f"  only {len(rows)} usable voiced frames; keeping default")
        return space

    delta = _delta_f(rows)
    if delta is None:
        return space
    vtl = vocal_tract_length_cm(delta)
    lo, hi = PLAUSIBLE_VTL_CM[voice]
    if not (lo <= vtl <= hi):
        # Measured on four Hindustani recordings, every estimate landed
        # outside the physiological range and male and female did not
        # separate at all — the pooled-formant estimator assumes varied
        # vowels to average over, and this repertoire sings one vowel for
        # most of its duration, on top of the harmonic bias in the formant
        # estimates themselves. Refuse rather than warp the template by a
        # number known to be wrong.
        warnings.warn(
            f"vocal tract length estimated at {vtl:.1f} cm for a {voice} "
            f"voice, outside the plausible {lo:.0f}-{hi:.0f} cm range. "
            f"Falling back to the voice-type default. Pooled-formant "
            f"estimation needs varied vowels, which sustained-vowel "
            f"singing does not provide.", RuntimeWarning, stacklevel=2)
        if verbose:
            print(f"  dF={delta:.0f} Hz -> VTL {vtl:.1f} cm: implausible, "
                  f"using {voice} default")
        return space
    ratio = delta / _template_delta_f()
    for v in list(space.targets):
        s0, _ = vowel_targets(v)
        space.targets[v] = (s0[0] * ratio, s0[1] * ratio, s0[2] * ratio,
                            s0[3], s0[4], s0[5])
    space.measured = ('_vtl',)
    space.coverage = 1.0
    if verbose:
        vtl = vocal_tract_length_cm(delta)
        print(f"  {len(rows)} frames (f0 <= {f0_cut:.0f} Hz), "
              f"dF={delta:.0f} Hz, VTL={vtl:.1f} cm, scale={ratio:.3f}")
    return space
