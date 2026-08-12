"""Extraction of control signals from a Piece for offline synthesis.

Produces, per instrument track, the same control data the web app feeds its
AudioWorklet synths: a dense f0 curve (from Trajectory.compute), gain
envelopes (from trajectory automation), and discrete event lists
(plucks/dampens, chikari strums, vowel spans with silence-boundary flags).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

SILENCE_ID = 12

# burst amplitudes from Synths.vue playSitarArticulations (sendBurst doubles
# option.amp internally, so the final amplitude is 2x these values)
ARTICULATION_BURST_AMPS = {
    'pluck': 0.5,
    'hammer-off': 0.5,
    'hammer-on': 0.3,
    'slide': 0.1,
}
BURST_DUR = 0.01     # seconds (sendBurst default)
BURST_ATK = 0.05     # seconds (sendBurst default; ramp is clipped to dur)
CHIKARI_BURST_AMP = 0.2
CHIKARI_BURST_ATK = 0.025
CHIKARI_STRUM_DELAY = 0.0025


@dataclass
class BurstEvent:
    """A noise-burst excitation event (pluck / hammer / slide / chikari)."""
    time: float
    amp: float          # pre-doubling amplitude, as passed to sendBurst
    atk: float = BURST_ATK
    dur: float = BURST_DUR


@dataclass
class Span:
    """A contiguous non-silent trajectory, with silence-boundary flags.

    ``start_consonant`` / ``end_consonant`` are IPA strings (from the
    trajectory's ``*_consonant_ipa`` fields) naming a consonant articulated
    at the onset / offset of this trajectory; see synthesis.consonants.
    """
    start: float
    end: float
    from_sil: bool
    to_sil: bool
    vowel: Optional[str] = None
    start_consonant: Optional[str] = None
    end_consonant: Optional[str] = None


@dataclass
class TrackControl:
    """Control-rate signals + events for one string of one track."""
    control_rate: float
    dur_tot: float
    f0: np.ndarray                  # Hz; 0 during silence
    gain: np.ndarray                # trajectory automation value; 0 in silence
    active: np.ndarray              # bool mask per control frame
    spans: List[Span] = field(default_factory=list)
    bursts: List[BurstEvent] = field(default_factory=list)
    dampen_times: List[float] = field(default_factory=list)
    chikari_events: List[BurstEvent] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.f0.shape[0])


def extract_track_control(piece, inst_idx: int, string_idx: int = 0,
                          control_rate: float = 200.0) -> TrackControl:
    """Build control signals for one instrument track / string index."""
    trajs = piece.all_trajectories(inst_idx, string_idx)
    starts = piece.traj_start_times(inst_idx, string_idx)
    dur_tot = float(piece.dur_tot or 0.0)
    if trajs and starts:
        dur_tot = max(dur_tot, starts[-1] + trajs[-1].dur_tot)
    hop = 1.0 / control_rate
    n_frames = max(int(math_ceil(dur_tot * control_rate)) + 1, 2)

    f0 = np.zeros(n_frames, dtype=np.float64)
    gain = np.zeros(n_frames, dtype=np.float64)
    active = np.zeros(n_frames, dtype=np.bool_)

    ctrl = TrackControl(control_rate=control_rate, dur_tot=dur_tot,
                        f0=f0, gain=gain, active=active)

    for t_idx, (traj, start) in enumerate(zip(trajs, starts)):
        if traj.id == SILENCE_ID:
            continue
        dur = traj.dur_tot
        end = start + dur
        if dur <= 0:
            continue

        # silence-boundary flags (mirrors fromSil/toSil in Synths.vue)
        prev_traj = trajs[t_idx - 1] if t_idx > 0 else None
        next_traj = trajs[t_idx + 1] if t_idx + 1 < len(trajs) else None
        from_sil = prev_traj is None or prev_traj.id == SILENCE_ID
        to_sil = next_traj is None or next_traj.id == SILENCE_ID
        ctrl.spans.append(Span(
            start=start, end=end, from_sil=from_sil, to_sil=to_sil,
            vowel=getattr(traj, 'vowel', None),
            start_consonant=getattr(traj, 'start_consonant_ipa', None),
            end_consonant=getattr(traj, 'end_consonant_ipa', None)))

        # control frames covered by this trajectory
        k0 = int(np.ceil(start / hop - 1e-9))
        k1 = min(int(np.floor(end / hop + 1e-9)), n_frames - 1)
        automation = getattr(traj, 'automation', None)
        for k in range(max(k0, 0), k1 + 1):
            x = (k * hop - start) / dur
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0
            f0[k] = traj.compute(x)
            gain[k] = automation.value_at_x(x) if automation else 1.0
            active[k] = True

        # articulation events
        arts = getattr(traj, 'articulations', None) or {}
        for key, art in arts.items():
            name = getattr(art, 'name', None)
            try:
                norm_t = float(key)
            except (TypeError, ValueError):
                continue
            when = start + norm_t * dur
            if name in ARTICULATION_BURST_AMPS:
                ctrl.bursts.append(
                    BurstEvent(time=when, amp=ARTICULATION_BURST_AMPS[name]))
            elif name == 'dampen':
                ctrl.dampen_times.append(when)

    # chikari strums (phrase-level, main string only)
    if string_idx == 0:
        try:
            phrases = piece.phrase_grid[inst_idx]
        except (AttributeError, IndexError):
            phrases = []
        for phrase in phrases:
            p_start = phrase.start_time or 0.0
            try:
                chikaris = phrase.chikaris
            except AttributeError:
                chikaris = {}
            for key in chikaris.keys():
                try:
                    when = p_start + float(key)
                except (TypeError, ValueError):
                    continue
                ctrl.chikari_events.append(
                    BurstEvent(time=when, amp=CHIKARI_BURST_AMP,
                               atk=CHIKARI_BURST_ATK))
    return ctrl


def math_ceil(x: float) -> int:
    return int(np.ceil(x))


def envelope_from_spans(ctrl: TrackControl, ramp: float = 0.01,
                        level: float = 1.0) -> np.ndarray:
    """Build a control-rate envelope that ramps level up over `ramp` seconds
    at each span start that comes from silence, and down at each span end
    that goes to silence (the web app's fromSil/toSil bowGain / envGain
    ramps)."""
    n = ctrl.n_frames
    hop = 1.0 / ctrl.control_rate
    env = np.zeros(n, dtype=np.float64)
    for span in ctrl.spans:
        k0 = max(int(np.ceil(span.start / hop - 1e-9)), 0)
        k1 = min(int(np.floor(span.end / hop + 1e-9)), n - 1)
        if k1 < k0:
            continue
        env[k0:k1 + 1] = level
        ramp_frames = max(int(round(ramp * ctrl.control_rate)), 1)
        if span.from_sil:
            for j in range(ramp_frames):
                k = k0 + j
                if k > k1:
                    break
                env[k] = level * (j + 1) / ramp_frames
        if span.to_sil:
            for j in range(ramp_frames):
                k = k1 - j
                if k < k0:
                    break
                env[k] = min(env[k], level * (j + 1) / ramp_frames)
    return env


def cutoff_curve_with_dampens(ctrl: TrackControl,
                              base_cutoff: float) -> np.ndarray:
    """Sitar loop-filter cutoff curve: constant, with dips to 0 at dampen
    articulations (Synths.vue ramps cutoff to 0 over 25 ms, holds 25 ms,
    then back up over 5 ms)."""
    n = ctrl.n_frames
    rate = ctrl.control_rate
    cur = np.full(n, base_cutoff, dtype=np.float64)
    for when in ctrl.dampen_times:
        t0 = when
        t1 = when + 0.025   # ramp down complete
        t2 = when + 0.050   # hold at 0
        t3 = when + 0.055   # ramp back up complete
        k0 = int(t0 * rate)
        k1 = int(t1 * rate)
        k2 = int(t2 * rate)
        k3 = int(t3 * rate)
        for k in range(max(k0, 0), min(k1, n - 1) + 1):
            frac = (k / rate - t0) / (t1 - t0)
            cur[k] = min(cur[k], base_cutoff * (1.0 - frac))
        for k in range(max(k1, 0), min(k2, n - 1) + 1):
            cur[k] = 0.0
        for k in range(max(k2, 0), min(k3, n - 1) + 1):
            frac = (k / rate - t2) / (t3 - t2)
            cur[k] = min(cur[k], base_cutoff * frac)
    return cur
