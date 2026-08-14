"""Fitting synthesis parameters to a recording.

The synthesizer's parameters are physical quantities — how bright the
string's damping is, how far the bridge contact travels, how long the note
rings — and their right values differ per instrument and per recording.
This module scores a candidate set of them against an actual recording so
they can be searched for rather than guessed.

What is being fitted is the physical model itself, not a neural network:
about a dozen numbers, searched to minimize the difference between the
render of a transcription and the recording that transcription was made
from.

Two things constrain how the comparison must be done.

**Alignment.** The render follows the transcription, not the recording, so
the two are not sample-aligned and never can be — transcribed timing is
approximate and the synthesized phase is arbitrary. The loss therefore
compares spectral *statistics* over long windows rather than waveforms.

**Bandwidth.** Historical recordings are band-limited by the equipment of
their day: one 1960 transfer tested here sits at its noise floor above
3 kHz. Scoring that band would teach the optimizer to imitate a worn tape
machine — the cheapest way to lose high-frequency energy is to damp the
string, which is precisely the defect this synthesis started with. The
band is therefore restricted to where the recording carries signal, which
``usable_band`` estimates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Third-octave-ish bands: fine enough to describe an instrument's spectral
# shape, coarse enough to be insensitive to which harmonic lands where.
DEFAULT_BAND_EDGES = [
    (80, 125), (125, 200), (200, 315), (315, 500), (500, 800),
    (800, 1250), (1250, 2000), (2000, 3150), (3150, 5000),
    (5000, 8000), (8000, 12500),
]
ANALYSIS_SR = 22050
NFFT = 4096
# A band is only worth scoring if the recording has signal there rather
# than noise floor.
NOISE_FLOOR_DB = -50.0


@dataclass
class FitTarget:
    """The recording a parameter set is being scored against."""
    band_db: np.ndarray                  # level per band, peak-normalized
    edges: List[Tuple[float, float]]
    weights: np.ndarray                  # 0 for bands outside the usable range
    source: str = ''
    limit_s: Optional[float] = None      # only audio up to here was used

    def describe(self) -> str:
        rows = [f"target: {self.source}"]
        if self.limit_s:
            rows.append(f"  using audio up to {self.limit_s:.0f}s "
                        f"(where the transcription ends)")
        for (lo, hi), db, w in zip(self.edges, self.band_db, self.weights):
            mark = '' if w > 0 else '   (excluded: at the noise floor)'
            rows.append(f"  {lo:6.0f}-{hi:<6.0f} Hz {db:7.1f} dB{mark}")
        return '\n'.join(rows)


def band_levels(x: np.ndarray, sr: float,
                edges: Sequence[Tuple[float, float]] = DEFAULT_BAND_EDGES
                ) -> np.ndarray:
    """Average level per band over frames that contain signal."""
    if sr != ANALYSIS_SR:
        m = int(len(x) * ANALYSIS_SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x)
    hop = NFFT // 2
    win = np.hanning(NFFT)
    acc = None
    count = 0
    for i in range((len(x) - NFFT) // hop):
        f = x[i * hop:i * hop + NFFT]
        if np.sqrt((f ** 2).mean()) < 1e-4:
            continue
        p = np.abs(np.fft.rfft(f * win)) ** 2
        acc = p if acc is None else acc + p
        count += 1
    if acc is None:
        return np.full(len(edges), -120.0)
    acc /= count
    freqs = np.fft.rfftfreq(NFFT, 1 / ANALYSIS_SR)
    out = []
    for lo, hi in edges:
        m = (freqs >= lo) & (freqs < hi)
        out.append(10 * np.log10(acc[m].mean() + 1e-20) if m.any() else -120.0)
    out = np.array(out)
    return out - out.max()


def usable_band(band_db: np.ndarray,
                floor_db: float = NOISE_FLOOR_DB) -> np.ndarray:
    """Which bands carry signal rather than the recording's noise floor.

    Everything from the loudest band outward is kept until the level falls
    below ``floor_db``; beyond that the recording is not describing the
    instrument, and fitting there fits the recording chain instead.
    """
    w = (band_db > floor_db).astype(np.float64)
    # keep the usable region contiguous around the peak
    peak = int(np.argmax(band_db))
    for i in range(peak, len(w)):
        if w[i] == 0:
            w[i:] = 0
            break
    for i in range(peak, -1, -1):
        if w[i] == 0:
            w[:i + 1] = 0
            break
    return w


def make_target(audio: np.ndarray, sr: float, source: str = '',
                limit_s: Optional[float] = None,
                edges: Sequence[Tuple[float, float]] = DEFAULT_BAND_EDGES
                ) -> FitTarget:
    """Summarize a recording for fitting, using only its usable bands."""
    if limit_s is not None:
        audio = audio[:int(limit_s * sr)]
    db = band_levels(audio, sr, edges)
    return FitTarget(band_db=db, edges=list(edges),
                     weights=usable_band(db), source=source,
                     limit_s=limit_s)


def spectral_loss(render: np.ndarray, sr: float, target: FitTarget) -> float:
    """Mean absolute band-level error, in dB, over the usable bands.

    Both sides are peak-normalized, so this measures spectral *shape* and
    is indifferent to overall gain — which is right, since the recording's
    level says nothing about the instrument.
    """
    db = band_levels(render, sr, target.edges)
    err = np.abs(db - target.band_db) * target.weights
    denom = target.weights.sum()
    return float(err.sum() / denom) if denom else float('inf')


# ---------------------------------------------------------------------------
# the parameters being searched
# ---------------------------------------------------------------------------

@dataclass
class Param:
    name: str
    low: float
    high: float
    default: float
    note: str = ''

    def clip(self, v: float) -> float:
        return float(min(max(v, self.low), self.high))


SITAR_PARAMS = [
    Param('brightness', 0.005, 0.30, 0.05,
          'loop damping; compounds once per round trip, so small values '
          'still darken a note substantially'),
    Param('stiffness', 0.0, 0.5, 0.22, 'dispersion allpass'),
    Param('jawari', 0.0, 0.5, 0.06, 'how far the bridge contact travels'),
    Param('jawari_threshold', 0.002, 0.08, 0.02,
          'displacement at which the string first touches the bridge'),
    Param('t60_max', 4.0, 30.0, 24.0, 'ring time with the string undamped'),
    Param('chikari_level', 0.1, 2.0, 0.5, 'chikari against the main string'),
    Param('chikari_t60', 1.0, 12.0, 4.9, 'chikari ring time'),
    # Body. Without these the model can only tilt its spectrum, not shape
    # it — which is what the first fit ran aground on, pinning five of
    # seven parameters at their bounds trying to match a tilt it could
    # only reach by damping everything.
    Param('body_mix', 0.0, 0.7, 0.25, 'gourd and soundboard in the output'),
    Param('body_f1', 80.0, 200.0, 120.0, 'gourd resonance'),
    Param('body_f2', 150.0, 350.0, 220.0, 'body resonance'),
    Param('body_f3', 300.0, 600.0, 400.0, 'body resonance'),
    Param('body_f4', 550.0, 1100.0, 750.0, 'soundboard resonance'),
    Param('body_f5', 1000.0, 2000.0, 1400.0, 'soundboard resonance'),
    Param('body_f6', 2000.0, 4000.0, 2600.0, 'soundboard resonance'),
    # Sympathetic strings, tuned by the raga rather than fitted.
    Param('taraf_mix', 0.0, 1.0, 0.35, 'sympathetic strings in the output'),
    Param('taraf_drive', 0.0, 0.2, 0.05, 'how hard the bridge drives them'),
    Param('taraf_t60', 0.5, 8.0, 3.0, 'how long they ring'),
]

SARANGI_PARAMS = [
    Param('bridge_damp', 0.0, 0.7, 0.25, 'darkness of the bridge reflection'),
    Param('body_mix', 0.0, 0.6, 0.15, 'parchment belly in the output'),
    Param('bow_noise', 0.0, 0.15, 0.02, 'bow-hair noise'),
    Param('bow_position', 0.04, 0.25, 0.13, 'bow distance from the bridge'),
    Param('taraf_mix', 0.0, 1.2, 0.5, 'sympathetic strings'),
    Param('taraf_t60', 0.5, 8.0, 2.5, 'how long the sympathetics ring'),
]


def defaults(params: Sequence[Param]) -> Dict[str, float]:
    return {p.name: p.default for p in params}


def to_vector(values: Dict[str, float],
              params: Sequence[Param]) -> np.ndarray:
    """Parameters as a vector in [0, 1], which is what searchers want."""
    return np.array([(values[p.name] - p.low) / (p.high - p.low)
                     for p in params])


def from_vector(v: Sequence[float],
                params: Sequence[Param]) -> Dict[str, float]:
    return {p.name: p.clip(p.low + float(x) * (p.high - p.low))
            for p, x in zip(params, v)}
