"""Simplified trajectory representation for AI training pipelines.

Every IDTAP trajectory can be expressed as a sequence of "simple" trajectories
drawn from five primitive types:

- ``silent``       (source id 12)
- ``fixed``        (source id 0) — constant log-frequency
- ``cosine``       (source id 1) — cosine interpolation between two log-freqs
- ``sloped-start`` (source id 2) — steep start, easing into the end pitch
- ``sloped-end``   (source id 3) — easing out of the start pitch, steep end

Composite trajectory types are broken into runs of these primitives:

- id 4 (ladle)         -> sloped-start + cosine
- id 5 (reverse ladle) -> cosine + sloped-end
- id 6 (yoyo)          -> one cosine per dur_array segment
- ids 7-11 (krintin family, slide) -> one fixed chunk per plateau
- id 13 (vibrato)      -> one cosine per half period, between the actual
  extreme values of the vibrato curve

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
``log_freq`` values are ``None``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:  # pragma: no cover
    from .trajectory import Trajectory

DEFAULT_SLOPE = 2.0

# Integer codes for categorical encoding in training data. 0-3 match the
# source trajectory ids for the non-composite types; silent gets 4.
TYPE_IDS: Dict[str, int] = {
    'fixed': 0,
    'cosine': 1,
    'sloped-start': 2,
    'sloped-end': 3,
    'silent': 4,
}


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

    def __post_init__(self) -> None:
        if self.type not in TYPE_IDS:
            raise ValueError(
                f"invalid simple trajectory type: {self.type!r}. "
                f"Allowed types: {sorted(TYPE_IDS)}"
            )

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
        elif self.type == 'cosine':
            pi_x = (math.cos(math.pi * (x + 1)) / 2) + 0.5
            out = pi_x * (lf1 - lf0) + lf0
        elif self.type == 'sloped-start':
            out = (lf0 - lf1) * ((1 - x) ** self.slope) + lf1
        else:  # sloped-end
            out = (lf1 - lf0) * (x ** self.slope) + lf0
        return 2 ** out

    def to_json(self) -> Dict:
        return {
            'type': self.type,
            'typeId': self.type_id,
            'dots': [self.start.to_json(), self.end.to_json()],
            'slope': self.slope,
            'continuation': self.continuation,
        }

    @staticmethod
    def from_json(obj: Dict) -> 'SimpleTrajectory':
        dots = obj['dots']
        return SimpleTrajectory(
            type=obj['type'],
            start=OrientationDot.from_json(dots[0]),
            end=OrientationDot.from_json(dots[1]),
            slope=obj.get('slope', DEFAULT_SLOPE),
            continuation=obj.get('continuation', False),
        )


def _lf(log_freqs: List[float], idx: int) -> float:
    """Log-freq at idx, clamped for malformed old transcriptions."""
    return log_freqs[min(idx, len(log_freqs) - 1)]


def decompose_trajectory(
    traj: 'Trajectory', start_time: Optional[float] = None
) -> List[SimpleTrajectory]:
    """Break a Trajectory into its simple-trajectory chunks.

    ``start_time`` is the absolute start of the trajectory in seconds; it
    defaults to ``traj.start_time`` (which, on trajectories inside a piece, is
    phrase-relative — pass an absolute time for piece-level output, as
    ``Piece.simplified_trajectories`` does).
    """
    t0 = start_time if start_time is not None else (traj.start_time or 0.0)
    d = traj.dur_tot

    if traj.id == 12:
        return [SimpleTrajectory(
            'silent', OrientationDot(t0), OrientationDot(t0 + d)
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
        periods = traj.vib_obj['periods']
        n = 2 * periods
        # Boundary values at each half-period are the vibrato curve's actual
        # extremes; between adjacent extremes the curve is a half cosine.
        bounds = [math.log2(traj.id13(k / n)) for k in range(n + 1)]
        segs = [
            ('cosine', 1.0 / n, bounds[k], bounds[k + 1], DEFAULT_SLOPE)
            for k in range(n)
        ]
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


def simple_trajectories_from_dots(
    times: Sequence[float],
    log_freqs: Sequence[Optional[float]],
    types: Sequence[Union[str, int]],
    slopes: Optional[Sequence[float]] = None,
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
                           | 'silent', or the matching TYPE_IDS integer
        slopes     (n-1,)  only the sloped types read it; defaults to 2.0

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
        out.append(SimpleTrajectory(
            type=typ,
            start=OrientationDot(float(times[i]), log_freqs[i]),
            end=OrientationDot(float(times[i + 1]), log_freqs[i + 1]),
            slope=DEFAULT_SLOPE if slopes is None else float(slopes[i]),
            continuation=continuation and i > 0,
        ))
    return out
