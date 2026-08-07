"""Reconstruct full transcriptions from simplified trajectory chunks.

Inverse of ``idtap.classes.simple_trajectory``: takes a sequence of
SimpleTrajectory chunks — typically the output of a generative model, so the
chunks are the ground truth rather than a lossy view of an existing
transcription — and builds a viewable ``Piece``.

Design constraints (see project notes):

- The simplified format stays minimal: no source ids, no metadata header.
  All reconstruction context (raga, instrumentation, audio provenance) is
  passed in as parameters.
- Vocal only. Sitar is rejected by design (this process isn't meant for it);
  sarangi may be added later.
- Chunks are grouped by their ``continuation`` flag; each group maps to one
  trajectory. For the vocal trajectory palette the mapping from a group's
  chunk-type sequence to a trajectory id is deterministic:

      [fixed]                  -> 0
      [cosine]                 -> 1
      [sloped-start]           -> 2
      [sloped-end]             -> 3
      [sloped-start, cosine]   -> 4 (ladle)
      [cosine, sloped-end]     -> 5 (reverse ladle)
      [cosine, ... , cosine]   -> 6 (yoyo; vibrato-shaped runs also land here)
      [silent]                 -> 12

- Generated output should always reconstruct to *something* viewable: a group
  whose type sequence matches no composite is split into one primitive
  trajectory per chunk, and where a composite needs a shared boundary pitch
  but adjacent chunks disagree, the earlier chunk's value wins.
- Chunks are consumed in the order given (continuation refers to the previous
  chunk in sequence). Trajectories are placed sequentially; gaps between
  groups larger than ``GAP_EPSILON`` become silent trajectories, and the piece
  starts at the first chunk's start time (any leading offset is dropped).
  Overlapping chunk times are the one malformation that raises instead of
  being absorbed: an overlap is ambiguous (which gesture owns the span?), and
  resolving it is a manual decision, never a silent one.
- Everything goes into a single phrase. Phrase segmentation is an analyst's
  job for now.
- Audio provenance must be declared explicitly: exactly one of ``audio_id``
  (extant recording), ``audio_path`` (uploaded via the client), or
  ``synthetic=True`` (no recording exists). The synthetic declaration is not
  yet persisted in the saved document; it is enforced at reconstruction time.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from ..enums import Instrument
from .phrase import Phrase
from .piece import Piece
from .pitch import Pitch
from .raga import Raga
from .simple_trajectory import SimpleTrajectory
from .trajectory import Trajectory

# Gaps between chunk groups shorter than this (seconds) are treated as float
# noise and absorbed; longer gaps become silent trajectories.
GAP_EPSILON = 1e-3

VOCAL_INSTRUMENTS = (Instrument.Vocal_M, Instrument.Vocal_F)

_PRIMITIVE_IDS = {
    'fixed': 0,
    'cosine': 1,
    'sloped-start': 2,
    'sloped-end': 3,
    'silent': 12,
}


def _group_chunks(chunks: Sequence[SimpleTrajectory]) -> List[List[SimpleTrajectory]]:
    groups: List[List[SimpleTrajectory]] = []
    for chunk in chunks:
        if chunk.continuation and groups:
            groups[-1].append(chunk)
        else:
            groups.append([chunk])
    return groups


def _silent_traj(dur: float, raga: Raga, inst: Instrument) -> Trajectory:
    return Trajectory({
        'id': 12,
        'pitches': [Pitch()],
        'dur_tot': dur,
        'fund_id12': raga.fundamental,
        'instrumentation': inst,
    })


def _primitive_traj(chunk: SimpleTrajectory, raga: Raga, inst: Instrument) -> Trajectory:
    if chunk.type == 'silent':
        return _silent_traj(chunk.dur_tot, raga, inst)
    start = raga.pitch_from_log_freq(chunk.start.log_freq)
    if chunk.type == 'fixed':
        pitches = [start]
    else:
        pitches = [start, raga.pitch_from_log_freq(chunk.end.log_freq)]
    return Trajectory({
        'id': _PRIMITIVE_IDS[chunk.type],
        'pitches': pitches,
        'dur_tot': chunk.dur_tot,
        'slope': chunk.slope,
        'instrumentation': inst,
    })


def _trajs_from_group(
    group: List[SimpleTrajectory], raga: Raga, inst: Instrument
) -> List[Trajectory]:
    """One or more trajectories for a continuation group.

    Composite groups whose type sequence matches no vocal composite are split
    into primitives rather than erroring; boundary-pitch disagreements resolve
    to the earlier chunk's value.
    """
    group = [c for c in group if c.dur_tot > 0]
    if not group:
        return []
    if len(group) == 1:
        return [_primitive_traj(group[0], raga, inst)]

    types = [c.type for c in group]

    if types == ['sloped-start', 'cosine']:
        traj_id, slope = 4, group[0].slope
    elif types == ['cosine', 'sloped-end']:
        traj_id, slope = 5, group[1].slope
    elif all(t == 'cosine' for t in types):
        traj_id, slope = 6, None
    else:
        # relative timing survives the split: each primitive carries its own
        # dur_tot, so no group-level dur_array is needed
        return [_primitive_traj(c, raga, inst) for c in group]

    dur_tot = sum(c.dur_tot for c in group)
    log_freqs = [group[0].start.log_freq] + [c.end.log_freq for c in group]
    options = {
        'id': traj_id,
        'pitches': [raga.pitch_from_log_freq(lf) for lf in log_freqs],
        'dur_tot': dur_tot,
        'dur_array': [c.dur_tot / dur_tot for c in group],
        'instrumentation': inst,
    }
    if slope is not None:
        options['slope'] = slope
    return [Trajectory(options)]


def _resolve_instrument(instrumentation: Union[Instrument, str]) -> Instrument:
    if isinstance(instrumentation, str):
        try:
            instrumentation = Instrument(instrumentation)
        except ValueError:
            raise ValueError(
                f"unknown instrumentation: {instrumentation!r}. "
                f"Valid values: {[i.value for i in Instrument]}"
            )
    if instrumentation not in VOCAL_INSTRUMENTS:
        raise ValueError(
            f"simplified-trajectory reconstruction supports vocal "
            f"instrumentation only (got {instrumentation.value!r}). Sitar is "
            f"not supported by design; sarangi may be added later."
        )
    return instrumentation


def reconstruct_piece(
    chunks: Sequence[SimpleTrajectory],
    raga: Union[Raga, str],
    instrumentation: Union[Instrument, str] = Instrument.Vocal_M,
    *,
    title: str = 'Reconstructed from simplified trajectories',
    audio_id: Optional[str] = None,
    audio_path: Optional[Union[str, Path]] = None,
    synthetic: bool = False,
    client=None,
    fundamental: Optional[float] = None,
    audio_metadata=None,
    audio_event=None,
) -> Piece:
    """Build a Piece from simplified trajectory chunks.

    ``raga`` may be a Raga object or a raga name; a name requires ``client``
    to fetch the rule set (``fundamental`` optionally sets the tonic).

    Exactly one audio-provenance argument is required:

    - ``audio_id``: ObjectId of an extant recording. Verified against the
      server when ``client`` is provided; otherwise taken on trust.
    - ``audio_path``: local audio file, uploaded via ``client.upload_audio``
      (requires ``client`` and ``audio_metadata``) and then linked.
    - ``synthetic=True``: explicit declaration that no recording exists.
    """
    declared = sum([audio_id is not None, audio_path is not None, bool(synthetic)])
    if declared != 1:
        raise ValueError(
            "audio provenance must be declared: pass exactly one of audio_id= "
            "(extant recording), audio_path= (file to upload), or "
            "synthetic=True (no recording exists)."
        )
    if not chunks:
        raise ValueError("cannot reconstruct a piece from an empty chunk sequence")

    inst = _resolve_instrument(instrumentation)

    if audio_path is not None:
        if client is None or audio_metadata is None:
            raise ValueError(
                "audio_path requires client and audio_metadata to upload the "
                "recording"
            )
        result = client.upload_audio(
            str(audio_path), audio_metadata, audio_event=audio_event
        )
        audio_id = result.audio_id
    elif audio_id is not None and client is not None:
        try:
            client.get_audio_recording(audio_id)
        except Exception as exc:
            raise ValueError(
                f"audio_id {audio_id!r} could not be verified against the "
                f"archive (missing recording, or a network/auth failure — "
                f"see chained exception): {exc}"
            ) from exc

    if isinstance(raga, str):
        options = {'name': raga}
        if fundamental is not None:
            options['fundamental'] = fundamental
        raga = Raga(options, client=client)

    trajs: List[Trajectory] = []
    prev_end: Optional[float] = None
    for group in _group_chunks(chunks):
        if prev_end is not None:
            gap = group[0].start.time - prev_end
            if gap < -GAP_EPSILON:
                raise ValueError(
                    f"overlapping chunks: a chunk starting at "
                    f"{group[0].start.time}s begins {-gap:.6f}s before the "
                    f"previous chunk ends ({prev_end}s). Overlaps cannot be "
                    f"reconstructed automatically — fix the chunk times and "
                    f"decide which gesture owns the overlapping span."
                )
            if gap > GAP_EPSILON:
                trajs.append(_silent_traj(gap, raga, inst))
        trajs.extend(_trajs_from_group(group, raga, inst))
        prev_end = group[-1].end.time

    phrase = Phrase({
        'trajectories': trajs,
        'raga': raga,
        'instrumentation': [inst.value],
    })
    piece = Piece({
        'phrases': [phrase],
        'raga': raga,
        'instrumentation': [inst],
        'title': title,
        'audioID': audio_id,
    })
    piece.synthetic = bool(synthetic)
    return piece
