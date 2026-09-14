"""Simplified trajectory representation for AI training pipelines.

Every IDTAP trajectory can be expressed as a sequence of "simple" trajectories
drawn from six primitive types:

- ``silent``       (source id 12)
- ``fixed``        (source id 0) — constant log-frequency
- ``cosine``       (source id 1) — cosine interpolation between two log-freqs
- ``sloped-start`` (source id 2) — steep start, easing into the end pitch
- ``sloped-end``   (source id 3) — easing out of the start pitch, steep end
- ``vibrato``      (source id 13) — the PROP-6 vibrato curve around a centre
  log-frequency, carried by four extra numbers (``rate`` in Hz,
  ``extent_start`` / ``extent_end`` in log2 peak to peak, ``phase`` in
  radians); both dots hold the centre, as a ``fixed`` does

Composite trajectory types are broken into runs of these primitives:

- id 4 (ladle)         -> sloped-start + cosine
- id 5 (reverse ladle) -> cosine + sloped-end
- id 6 (yoyo)          -> one cosine per dur_array segment
- ids 7-11 (krintin family, slide) -> one fixed chunk per plateau

An id 13 vibrato maps one-to-one onto a single ``vibrato`` chunk (the
``vib_obj`` is renamed, not fitted). The older lossy view — one cosine per
half period between consecutive extremes of the curve — is still available via
``decompose_trajectory(..., vibrato_as_cosines=True)``; it is what earlier
corpora were measured with, and it is used automatically for the one id 13
shape a vibrato chunk cannot carry, a non-zero ``vert_offset``.

The simplified format is discrete, so ``vibrato`` is a type here rather than
a modifier on ``fixed`` as it is in a differentiable model's vocabulary (Jon,
2026-09-14): the mapping to id 13 is one-to-one in both directions, and the
floor between "a fixed" and "a small vibrato" lives in exactly one place, the
producer's decoder, rather than in every consumer.

Each simple trajectory is defined by two orientation dots (time in seconds,
log2(frequency)), a slope (only meaningful for sloped-start / sloped-end;
defaults to 2.0 elsewhere), and a ``continuation`` flag. ``continuation`` is
True for every chunk after the first within a single source trajectory: it
marks that the chunk continues the same gesture and should be attached to the
previous chunk. The first chunk of every source trajectory is always
``continuation=False``, and the flag never crosses source-trajectory
boundaries. Note that for step-wise chunks (krintin hammer-ons/offs) a
continuation chunk's start pitch may still differ from the previous chunk's
end pitch — continuation describes gesture membership, not pitch continuity.

For ``silent`` chunks only the orientation-dot times are meaningful; their
``log_freq`` values are ``None``. The four vibrato numbers ride only on
``vibrato`` chunks; on every other type they are ``None`` and never serialized.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover
    from .trajectory import Trajectory

DEFAULT_SLOPE = 2.0

# Integer codes for categorical encoding in training data. 0-3 match the
# source trajectory ids for the non-composite types; silent gets 4 and vibrato
# 5. Existing codes are never renumbered: stored data depends on them.
TYPE_IDS: Dict[str, int] = {
    'fixed': 0,
    'cosine': 1,
    'sloped-start': 2,
    'sloped-end': 3,
    'silent': 4,
    'vibrato': 5,
}

# The vibrato numbers, in the order a 4-sequence is read by
# ``simple_trajectories_from_dots`` (matches the consuming model's VIB_*
# column order) and the JSON keys they serialize to.
VIBRATO_FIELDS = ('rate', 'extent_start', 'extent_end', 'phase')
_VIBRATO_JSON_KEYS = {
    'rate': 'rate',
    'extent_start': 'extentStart',
    'extent_end': 'extentEnd',
    'phase': 'phase',
}


def vibrato_log_freq(
    x: float,
    dur_tot: float,
    lf0: float,
    rate: float,
    extent_start: float,
    extent_end: float,
    phase: float,
    vert_offset: float = 0.0,
) -> float:
    """log2-frequency of the PROP-6 vibrato curve at normalised position ``x``.

    Term for term ``Trajectory.id13`` / ``Trajectory._vib_curve`` (which mirror
    the TypeScript reference), with ``dur_tot`` standing in for the
    trajectory's duration and ``lf0`` for ``log_freqs[0]``:

        P       = rate * dur_tot, or 1 if that is under one cycle
        A(x)    = (extent_start + (extent_end - extent_start) * x) / 2
        core(x) = lf0 + clamp(vert_offset, +-A(x)) + A(x) * cos(2 pi P x + phase)

    The curve attaches at an extreme so the note starts and ends on ``lf0``:
    ``x1`` is the first extreme at least a quarter period in, ``x2`` the last
    at least a quarter period before the end (``x2 = x1`` when the span is too
    short to hold both), and raised cosines carry ``lf0`` out to ``core(x1)``
    and ``core(x2)`` back home. Kept as a free function so the simplified
    format has no dependency on ``Trajectory``; ``simple_trajectory_test``
    pins it to ``Trajectory.id13`` at 1e-12 relative.
    """
    P = rate * dur_tot
    if not P >= 1:
        P = 1.0

    def core(xx: float) -> float:
        A = (extent_start + (extent_end - extent_start) * xx) / 2
        vo = vert_offset
        if abs(vo) > A:
            vo = math.copysign(A, vo)
        return lf0 + vo + A * math.cos(2 * math.pi * P * xx + phase)

    ph = phase / math.pi
    k1 = math.ceil(0.5 + ph)
    k2 = math.floor(2 * P - 0.5 + ph)
    x1 = (k1 - ph) / (2 * P)
    x2 = (k2 - ph) / (2 * P)
    if x2 < x1:
        x2 = x1

    if x <= x1:
        end = core(x1)
        return lf0 + (end - lf0) * (1 - math.cos(math.pi * x / x1)) / 2
    if x >= x2:
        start = core(x2)
        return start + (lf0 - start) * (1 - math.cos(math.pi * (x - x2) / (1 - x2))) / 2
    return core(x)


@dataclass
class OrientationDot:
    time: float
    log_freq: Optional[float] = None

    def to_json(self) -> Dict:
        data: Dict = {'time': self.time}
        if self.log_freq is not None:
            data['logFreq'] = self.log_freq
        return data

    @staticmethod
    def from_json(obj: Dict) -> 'OrientationDot':
        return OrientationDot(time=obj['time'], log_freq=obj.get('logFreq'))


@dataclass
class SimpleTrajectory:
    type: str
    start: OrientationDot
    end: OrientationDot
    slope: float = DEFAULT_SLOPE
    continuation: bool = False
    # vibrato only (PROP-6 numbers; the chunk's dot is the centre, so there is
    # no vert_offset): None on every other type.
    rate: Optional[float] = None            # Hz, > 0
    extent_start: Optional[float] = None    # log2, peak to peak, >= 0
    extent_end: Optional[float] = None      # log2, peak to peak, >= 0
    phase: Optional[float] = None           # radians at the chunk's start

    def __post_init__(self) -> None:
        if self.type not in TYPE_IDS:
            raise ValueError(
                f"invalid simple trajectory type: {self.type!r}. "
                f"Allowed types: {sorted(TYPE_IDS)}"
            )
        if self.type == 'vibrato':
            self._validate_vibrato()
        else:
            stray = [f for f in VIBRATO_FIELDS if getattr(self, f) is not None]
            if stray:
                raise ValueError(
                    f"{stray} are vibrato-only fields but the chunk type is "
                    f"{self.type!r}; the vibrato numbers ride only on "
                    f"'vibrato' chunks"
                )

    def _validate_vibrato(self) -> None:
        """``rate`` and ``extent_start`` are required; ``extent_end`` defaults
        to ``extent_start`` (constant extent) and ``phase`` to 0."""
        if self.rate is None:
            raise ValueError("a vibrato chunk requires rate (Hz)")
        if self.extent_start is None:
            raise ValueError(
                "a vibrato chunk requires extent_start (log2, peak to peak)")
        if self.extent_end is None:
            self.extent_end = self.extent_start
        if self.phase is None:
            self.phase = 0.0
        for f in VIBRATO_FIELDS:
            v = getattr(self, f)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"vibrato {f} must be a number, got {v!r}")
            if not math.isfinite(v):
                raise ValueError(f"vibrato {f} must be finite, got {v!r}")
            setattr(self, f, float(v))
        if self.rate <= 0:
            raise ValueError(f"vibrato rate must be > 0 Hz, got {self.rate}")
        for f in ('extent_start', 'extent_end'):
            if getattr(self, f) < 0:
                raise ValueError(
                    f"vibrato {f} must be >= 0, got {getattr(self, f)}")

    @property
    def type_id(self) -> int:
        return TYPE_IDS[self.type]

    @property
    def dur_tot(self) -> float:
        return self.end.time - self.start.time

    def compute(self, x: float) -> Optional[float]:
        """Frequency (Hz) at normalized position x in [0, 1]; None if silent."""
        if self.type == 'silent':
            return None
        lf0 = self.start.log_freq
        lf1 = self.end.log_freq
        if self.type == 'fixed':
            out = lf0
        elif self.type == 'vibrato':
            # centre is the dot; the chunk carries no vert_offset
            out = vibrato_log_freq(
                x, self.dur_tot, lf0, self.rate, self.extent_start,
                self.extent_end, self.phase,
            )
        elif self.type == 'cosine':
            pi_x = (math.cos(math.pi * (x + 1)) / 2) + 0.5
            out = pi_x * (lf1 - lf0) + lf0
        elif self.type == 'sloped-start':
            out = (lf0 - lf1) * ((1 - x) ** self.slope) + lf1
        else:  # sloped-end
            out = (lf1 - lf0) * (x ** self.slope) + lf0
        return 2 ** out

    def to_json(self) -> Dict:
        data: Dict = {
            'type': self.type,
            'typeId': self.type_id,
            'dots': [self.start.to_json(), self.end.to_json()],
            'slope': self.slope,
            'continuation': self.continuation,
        }
        if self.type == 'vibrato':
            for field, key in _VIBRATO_JSON_KEYS.items():
                data[key] = getattr(self, field)
        return data

    @staticmethod
    def from_json(obj: Dict) -> 'SimpleTrajectory':
        dots = obj['dots']
        return SimpleTrajectory(
            type=obj['type'],
            start=OrientationDot.from_json(dots[0]),
            end=OrientationDot.from_json(dots[1]),
            slope=obj.get('slope', DEFAULT_SLOPE),
            continuation=obj.get('continuation', False),
            **{field: obj.get(key) for field, key in _VIBRATO_JSON_KEYS.items()},
        )


def _lf(log_freqs: List[float], idx: int) -> float:
    """Log-freq at idx, clamped for malformed old transcriptions."""
    return log_freqs[min(idx, len(log_freqs) - 1)]


def decompose_trajectory(
    traj: 'Trajectory',
    start_time: Optional[float] = None,
    *,
    vibrato_as_cosines: bool = False,
) -> List[SimpleTrajectory]:
    """Break a Trajectory into its simple-trajectory chunks.

    ``start_time`` is the absolute start of the trajectory in seconds; it
    defaults to ``traj.start_time`` (which, on trajectories inside a piece, is
    phrase-relative — pass an absolute time for piece-level output, as
    ``Piece.simplified_trajectories`` does).

    An id 13 vibrato becomes one ``vibrato`` chunk carrying its v2 ``vib_obj``
    numbers. ``vibrato_as_cosines=True`` selects the older lossy view instead
    — one cosine per half period between consecutive extremes of the curve
    (see ``vibrato_cosine_segments``). That view is also used, regardless of
    the flag, for an id 13 with a non-zero ``vert_offset``: the chunk has no
    offset field, and the cosine chain is what still reproduces the curve.
    """
    t0 = start_time if start_time is not None else (traj.start_time or 0.0)
    d = traj.dur_tot

    if traj.id == 12:
        return [SimpleTrajectory(
            'silent', OrientationDot(t0), OrientationDot(t0 + d)
        )]

    if traj.id == 13:
        v = traj.vib_obj
        if not vibrato_as_cosines and v['vert_offset'] == 0:
            lf0 = traj.log_freqs[0]
            return [SimpleTrajectory(
                'vibrato',
                OrientationDot(t0, lf0), OrientationDot(t0 + d, lf0),
                rate=v['rate'],
                extent_start=v['extent_start'],
                extent_end=v['extent_end'],
                phase=v['phase'],
            )]

    lfs = traj.log_freqs
    da = traj.dur_array or [1.0]
    # segments: (type, fraction of dur_tot, start log_freq, end log_freq, slope)
    if traj.id == 0:
        segs = [('fixed', 1.0, lfs[0], lfs[0], DEFAULT_SLOPE)]
    elif traj.id == 1:
        segs = [('cosine', 1.0, lfs[0], _lf(lfs, 1), DEFAULT_SLOPE)]
    elif traj.id == 2:
        segs = [('sloped-start', 1.0, lfs[0], _lf(lfs, 1), traj.slope)]
    elif traj.id == 3:
        segs = [('sloped-end', 1.0, lfs[0], _lf(lfs, 1), traj.slope)]
    elif traj.id == 4:
        segs = [
            ('sloped-start', da[0], lfs[0], _lf(lfs, 1), traj.slope),
            ('cosine', da[1], _lf(lfs, 1), _lf(lfs, 2), DEFAULT_SLOPE),
        ]
    elif traj.id == 5:
        segs = [
            ('cosine', da[0], lfs[0], _lf(lfs, 1), DEFAULT_SLOPE),
            ('sloped-end', da[1], _lf(lfs, 1), _lf(lfs, 2), traj.slope),
        ]
    elif traj.id == 6:
        segs = [
            ('cosine', da[i], _lf(lfs, i), _lf(lfs, i + 1), DEFAULT_SLOPE)
            for i in range(len(da))
        ]
    elif traj.id in (7, 11):
        # id7's compute (also used by id11) only ever plays pitches 0 and 1,
        # switching at dur_array[0].
        segs = [
            ('fixed', da[0], lfs[0], lfs[0], DEFAULT_SLOPE),
            ('fixed', 1.0 - da[0], _lf(lfs, 1), _lf(lfs, 1), DEFAULT_SLOPE),
        ]
    elif traj.id in (8, 9, 10):
        segs = [
            ('fixed', da[i], _lf(lfs, i), _lf(lfs, i), DEFAULT_SLOPE)
            for i in range(len(da))
        ]
    elif traj.id == 13:
        segs = vibrato_cosine_segments(traj)
    else:
        raise ValueError(f"cannot decompose trajectory with id {traj.id}")

    chunks: List[SimpleTrajectory] = []
    elapsed = 0.0
    for i, (typ, frac, lf_start, lf_end, slope) in enumerate(segs):
        chunk_start = t0 + d * elapsed
        elapsed += frac
        # last chunk lands exactly on the trajectory's end time
        chunk_end = t0 + d if i == len(segs) - 1 else t0 + d * elapsed
        chunks.append(SimpleTrajectory(
            type=typ,
            start=OrientationDot(chunk_start, lf_start),
            end=OrientationDot(chunk_end, lf_end),
            slope=slope,
            continuation=i > 0,
        ))
    return chunks


def vibrato_cosine_segments(traj: 'Trajectory') -> List[tuple]:
    """The cosine-chain view of an id 13 vibrato, as ``decompose_trajectory``
    segments ``(type, fraction of dur_tot, start log_freq, end log_freq, slope)``.

    One cosine chunk per interval between consecutive extremes of the actual
    curve (idtap-contract PROP-6). The first and last intervals are the
    raised-cosine tapers from/to ``log_freqs[0]``, which are exactly a
    'cosine' chunk; between two interior extremes the curve is a half cosine,
    exact when the extent is constant (every healed v1 vibrato) and a close
    fit when it ramps (the chunk ends still sit on the curve; only the
    interior differs, by at most the ramp increment over one half period).

    This was the only decomposition of id 13 before the ``vibrato`` type
    existed, so it is how earlier corpora were measured; it remains the
    fallback for a poor vibrato fit on the producer side and for an id 13
    with a non-zero ``vert_offset``.
    """
    xs = traj.vib_breakpoints()
    bounds = [math.log2(traj.id13(x)) for x in xs]
    return [
        ('cosine', xs[k + 1] - xs[k], bounds[k], bounds[k + 1], DEFAULT_SLOPE)
        for k in range(len(xs) - 1)
    ]


VibratoSpec = Union[Mapping[str, float], Sequence[float]]


def _vibrato_kwargs(spec: Optional[VibratoSpec], i: int) -> Dict[str, float]:
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        unknown = set(spec) - set(VIBRATO_FIELDS)
        if unknown:
            raise ValueError(
                f"vibratos[{i}] has unknown keys {sorted(unknown)}; "
                f"allowed: {list(VIBRATO_FIELDS)}")
        return {k: float(v) for k, v in spec.items() if v is not None}
    if len(spec) != len(VIBRATO_FIELDS):
        raise ValueError(
            f"vibratos[{i}] must be a mapping or a {len(VIBRATO_FIELDS)}-"
            f"sequence {VIBRATO_FIELDS}, got {len(spec)} values")
    return {k: float(v) for k, v in zip(VIBRATO_FIELDS, spec)}


def simple_trajectories_from_dots(
    times: Sequence[float],
    log_freqs: Sequence[Optional[float]],
    types: Sequence[Union[str, int]],
    slopes: Optional[Sequence[float]] = None,
    vibratos: Optional[Sequence[Optional[VibratoSpec]]] = None,
    *,
    continuation: bool = False,
) -> List[SimpleTrajectory]:
    """Build chunks from a chained dot sequence, the inverse entry point.

    `decompose_trajectory` turns one trajectory into chunks. This turns the
    other common shape into chunks: parallel arrays, where consecutive chunks
    *share* a dot, so n dots describe n-1 chunks. That is how the
    representation is stored and how a model that predicts it emits it -- a
    curve broken at breakpoints rather than a list of independent segments --
    and building `SimpleTrajectory` objects by hand from that shape means
    re-deriving the sharing convention at every call site.

        times      (n,)    seconds, strictly increasing
        log_freqs  (n,)    log2(Hz); None for a dot bounding silence
        types      (n-1,)  'fixed' | 'cosine' | 'sloped-start' | 'sloped-end'
                           | 'silent' | 'vibrato', or the matching TYPE_IDS
                           integer
        slopes     (n-1,)  only the sloped types read it; defaults to 2.0
        vibratos   (n-1,)  per chunk: None, or the PROP-6 numbers for a
                           'vibrato' chunk as a mapping over
                           rate / extent_start / extent_end / phase or a
                           4-sequence in that order (VIBRATO_FIELDS). Required
                           on every vibrato chunk, forbidden on every other.

    `continuation` marks every chunk after the first as continuing the one
    before, which is what `reconstruct_piece` groups on when it rebuilds
    composite trajectories. Leave it False when the chunks are independent
    observations, as they are coming from a model that has no source
    trajectory to have been decomposed from.

    Round-trips with `decompose_trajectory`: chunks in, arrays out, chunks
    back.
    """
    n = len(times)
    if n < 2:
        raise ValueError(f"need at least two dots to make a chunk, got {n}")
    if len(log_freqs) != n:
        raise ValueError(
            f"log_freqs has {len(log_freqs)} entries for {n} dots")
    if len(types) != n - 1:
        raise ValueError(
            f"{n} dots describe {n - 1} chunks, but got {len(types)} types")
    if slopes is not None and len(slopes) != n - 1:
        raise ValueError(
            f"{n - 1} chunks, but got {len(slopes)} slopes")
    if vibratos is not None and len(vibratos) != n - 1:
        raise ValueError(
            f"{n - 1} chunks, but got {len(vibratos)} vibratos")

    by_id = {v: k for k, v in TYPE_IDS.items()}
    out: List[SimpleTrajectory] = []
    for i in range(n - 1):
        if times[i + 1] <= times[i]:
            raise ValueError(
                f"times must increase: dot {i} at {times[i]} is not before "
                f"dot {i + 1} at {times[i + 1]}")
        typ = types[i]
        if not isinstance(typ, str):
            if typ not in by_id:
                raise ValueError(
                    f"invalid simple trajectory type id: {typ!r}. "
                    f"Allowed ids: {sorted(by_id)}")
            typ = by_id[typ]
        vib = _vibrato_kwargs(None if vibratos is None else vibratos[i], i)
        if typ == 'vibrato' and not vib:
            raise ValueError(
                f"chunk {i} is a vibrato but vibratos[{i}] gives no numbers")
        if typ != 'vibrato' and vib:
            raise ValueError(
                f"vibratos[{i}] is set but chunk {i} is {typ!r}, not 'vibrato'")
        out.append(SimpleTrajectory(
            type=typ,
            start=OrientationDot(float(times[i]), log_freqs[i]),
            end=OrientationDot(float(times[i + 1]), log_freqs[i + 1]),
            slope=DEFAULT_SLOPE if slopes is None else float(slopes[i]),
            continuation=continuation and i > 0,
            **vib,
        ))
    return out
