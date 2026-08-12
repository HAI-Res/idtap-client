"""Measuring a singer's own vowel space from their recordings.

The synthesis default is one generic formant table (see vowels.py), so
every voice comes out the same. This module replaces it with formants
measured from the singer's actual audio: the transcription already says
which vowel is being sung and exactly when, so those labels segment the
recording and Praat's Burg LPC tracker (via Parselmouth) reads the
formants out of each labelled vowel.

Measure the *isolated vocal stem*, not the raw recording — a tanpura drone
or sarangi in the signal will pull the tracker badly. See
``idtap.synthesis.formants.SEPARATION_NOTE``.

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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..enums import Instrument
from .control import extract_track_control
from .vowels import DEFAULT_B, DEFAULT_F, VOWELS, vowel_targets

SEPARATION_NOTE = (
    "Formant measurement assumes a vocal stem with the accompaniment "
    "removed. Run the recording through a vocal separator first (denoise, "
    "then a karaoke-style separator, which routes sustained drones to the "
    "accompaniment stem instead of leaking them into the vocal)."
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
            measured=tuple(obj.get('measured', ())))

    def summary(self) -> str:
        rows = [f"vowel space ({self.voice}) from {self.source or 'unknown'}"]
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

    # ignore frames that are essentially silent in the stem
    try:
        levels = [intensity.get_value(t) for t in intensity.xs()]
        levels = [x for x in levels if x is not None and x == x]
        floor = (statistics.median(levels) - 12.0) if levels else 0.0
    except Exception:
        floor = 0.0

    lo, mid, hi = FORMANT_RANGES[voice]
    samples: Dict[str, List[Tuple[float, float, float]]] = {}
    for vowel, t0, t1 in _vowel_intervals(piece, inst_idx, control_rate):
        t = t0
        while t < t1:
            if t > sound.xmax:
                break
            try:
                level = intensity.get_value(t)
            except Exception:
                level = None
            if level is not None and level == level and level >= floor:
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
            print(f"  {vowel:>3}: accepted formants "
                  f"{[f'F{j+1}' for j in accepted]}")
        if verbose:
            print(f"  {vowel:>3}: n={len(vals):5d} "
                  f"F1={f1:6.0f} F2={f2:6.0f} F3={f3:6.0f}")
    space.counts = counts
    space.measured = tuple(measured)
    return space
