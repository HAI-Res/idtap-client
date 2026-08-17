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
# a less blind loss
# ---------------------------------------------------------------------------

def sixth_octave_edges(low: float = 80.0, high: float = 8000.0):
    """Sixth-octave bands: finer than third-octave, still coarse enough to
    be insensitive to exactly which harmonic lands where."""
    edges = []
    f = low
    ratio = 2.0 ** (1.0 / 6.0)
    while f * ratio <= high:
        edges.append((f, f * ratio))
        f *= ratio
    return edges


FINE_BAND_EDGES = sixth_octave_edges()
# Octaves over which to compare tonal-vs-noisy character.
FLATNESS_EDGES = [(200, 400), (400, 800), (800, 1600),
                  (1600, 3200), (3200, 6400)]


def spectral_flatness(x, sr, edges=FLATNESS_EDGES):
    """Geometric over arithmetic mean of the spectrum, per band.

    Near 0 for a clean harmonic tone, near 1 for noise. A jawari makes a
    string measurably buzzier, and a band-level average cannot see that at
    all — which is why both fits switched the bridge off.
    """
    if sr != ANALYSIS_SR:
        m = int(len(x) * ANALYSIS_SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x)
    hop = NFFT // 2
    win = np.hanning(NFFT)
    freqs = np.fft.rfftfreq(NFFT, 1 / ANALYSIS_SR)
    acc = np.zeros(len(edges))
    count = 0
    for i in range((len(x) - NFFT) // hop):
        f = x[i * hop:i * hop + NFFT]
        if np.sqrt((f ** 2).mean()) < 1e-4:
            continue
        p = np.abs(np.fft.rfft(f * win)) ** 2 + 1e-20
        for j, (lo, hi) in enumerate(edges):
            m = (freqs >= lo) & (freqs < hi)
            if m.any():
                seg = p[m]
                acc[j] += float(np.exp(np.log(seg).mean()) / seg.mean())
        count += 1
    return acc / max(count, 1)


def decay_profile(x, sr, n_steps: int = 8, step: float = 0.25):
    """How the level falls after the loudest moments.

    Ring time is a property of the envelope, and nothing in a time-averaged
    spectrum reports it. This takes the loudest short windows, follows the
    level for a couple of seconds after each, and averages — giving a decay
    curve in dB relative to the peak.
    """
    if sr != ANALYSIS_SR:
        m = int(len(x) * ANALYSIS_SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, m), np.arange(len(x)), x)
    w = int(0.05 * ANALYSIS_SR)
    m = len(x) // w * w
    if m < w * 4:
        return np.zeros(n_steps)
    env = np.sqrt((x[:m].reshape(-1, w) ** 2).mean(axis=1))
    per = int(step / 0.05)
    need = n_steps * per
    # onsets: a window much louder than the one before it
    cand = []
    for i in range(1, len(env) - need):
        if env[i] > 3.0 * env[i - 1] and env[i] > 0.05 * env.max():
            cand.append(i)
    if len(cand) < 5:
        cand = list(np.argsort(env[:len(env) - need])[-40:])
    rows = []
    for i in cand[:200]:
        peak = env[i]
        if peak <= 0:
            continue
        rows.append([20 * np.log10(max(env[i + k * per], 1e-9) / peak)
                     for k in range(n_steps)])
    return np.mean(rows, axis=0) if rows else np.zeros(n_steps)


@dataclass
class RichTarget:
    """A recording summarized by shape, texture and decay together."""
    bands: np.ndarray
    weights: np.ndarray
    flatness: np.ndarray
    decay: np.ndarray
    edges: List[Tuple[float, float]]
    source: str = ''

    def describe(self) -> str:
        used = int((self.weights > 0).sum())
        return (f"target {self.source}: {used}/{len(self.edges)} bands used, "
                f"flatness {np.round(self.flatness, 3).tolist()}, "
                f"decay {np.round(self.decay, 1).tolist()} dB")


def make_rich_target(audio, sr, source: str = '', limit_s=None) -> RichTarget:
    if limit_s is not None:
        audio = audio[:int(limit_s * sr)]
    bands = band_levels(audio, sr, FINE_BAND_EDGES)
    return RichTarget(bands=bands, weights=usable_band(bands),
                      flatness=spectral_flatness(audio, sr),
                      decay=decay_profile(audio, sr),
                      edges=list(FINE_BAND_EDGES), source=source)


# How much each term counts. Shape dominates; texture and decay are there
# to make visible what a spectral average cannot see.
W_SHAPE = 1.0
W_FLATNESS = 6.0
W_DECAY = 0.35


def rich_loss(render, sr, target: RichTarget, detail: bool = False):
    """Spectral shape, tonal character and decay, combined."""
    b = band_levels(render, sr, target.edges)
    shape = float((np.abs(b - target.bands) * target.weights).sum()
                  / max(target.weights.sum(), 1e-9))
    fl = spectral_flatness(render, sr)
    flat = float(np.abs(fl - target.flatness).mean())
    dc = decay_profile(render, sr)
    decay = float(np.abs(dc - target.decay).mean())
    total = W_SHAPE * shape + W_FLATNESS * flat + W_DECAY * decay
    if detail:
        return total, {'shape_db': shape, 'flatness': flat,
                       'decay_db': decay}
    return total


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
    Param('t60_max', 8.0, 30.0, 20.0,
          'ring time with the string undamped; a sitar sustains, and letting this fall to a few seconds lets the fit substitute sympathetic ring for the played note'),
    Param('chikari_level', 0.05, 0.6, 0.3,
          'chikari against the main string; they punctuate rather than sing, so they stay below it'),
    Param('chikari_t60', 1.0, 8.0, 4.9, 'chikari ring time'),
    # Body. Without these the model can only tilt its spectrum, not shape
    # it — which is what the first fit ran aground on, pinning five of
    # seven parameters at their bounds trying to match a tilt it could
    # only reach by damping everything.
    Param('body_mix', 0.0, 1.5, 0.25, 'gourd and soundboard in the output'),
    Param('body_f1', 60.0, 200.0, 120.0, 'gourd resonance'),
    Param('body_f2', 150.0, 600.0, 220.0, 'body resonance'),
    Param('body_f3', 300.0, 600.0, 400.0, 'body resonance'),
    Param('body_f4', 550.0, 1100.0, 750.0, 'soundboard resonance'),
    Param('body_f5', 1000.0, 3000.0, 1400.0, 'soundboard resonance'),
    Param('body_f6', 2000.0, 4000.0, 2600.0, 'soundboard resonance'),
    # Sympathetic strings, tuned by the raga rather than fitted.
    Param('taraf_mix', 0.0, 1.2, 0.4, 'sympathetic strings in the output'),
    Param('taraf_drive', 0.0, 0.25, 0.08,
          'how hard the bridge drives them'),
    Param('taraf_t60', 1.0, 6.0, 3.0, 'how long they ring'),
    Param('taraf_damp', 0.15, 0.8, 0.45,
          'how dark the sympathetics are; thin wire over a bridge loses its highs quickly'),
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


def load_preset(name: str) -> Dict[str, float]:
    """Parameters from a fitted preset in ``synthesis/presets``.

    See the README there before trusting one wholesale: the loss cannot
    tell played-string energy from sympathetic energy, so a preset's
    chikari and taraf levels are an upper bound rather than a finding.
    """
    import json
    from pathlib import Path

    path = Path(__file__).parent / 'presets' / f'{name}.json'
    if not path.exists():
        available = sorted(p.stem for p in path.parent.glob('*.json'))
        raise ValueError(f'no preset {name!r}; have {available}')
    return json.loads(path.read_text())['params']


def to_vector(values: Dict[str, float],
              params: Sequence[Param]) -> np.ndarray:
    """Parameters as a vector in [0, 1], which is what searchers want."""
    return np.array([(values[p.name] - p.low) / (p.high - p.low)
                     for p in params])


def from_vector(v: Sequence[float],
                params: Sequence[Param]) -> Dict[str, float]:
    return {p.name: p.clip(p.low + float(x) * (p.high - p.low))
            for p, x in zip(params, v)}
