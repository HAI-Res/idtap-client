"""Analysis functions for musical transcription data.

Ports of TypeScript analysis.ts functions for pitch analysis,
duration computation, segmentation, and pattern detection.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

from .classes.trajectory import Trajectory
from .classes.pitch import Pitch
from .classes.automation import get_starts


def pitch_times(
    trajs: List[Trajectory],
    output_type: str = 'pitchNumber',
) -> List[Dict]:
    """List all pitches with cumulative start times across a trajectory sequence.

    Port of TypeScript ``PitchTimes``.

    Args:
        trajs: Sequence of trajectories to analyse.
        output_type: Pitch representation — ``'pitchNumber'``, ``'chroma'``,
            ``'scaleDegree'``, ``'sargamLetter'``, or ``'octavedSargamLetter'``.

    Returns:
        List of dicts ``{'time': float, 'pitch': int|str|'silence',
        'articulation': bool}``.
    """
    result: List[Dict] = []
    start_time = 0.0

    for traj in trajs:
        art = traj.articulations.get('0') or traj.articulations.get('0.00')
        if traj.id == 12:
            result.append({
                'time': start_time,
                'pitch': 'silence',
                'articulation': False,
            })
            start_time += traj.dur_tot
        else:
            for p_idx, pitch in enumerate(traj.pitches):
                pitch_rep: Union[int, str]
                if output_type == 'pitchNumber':
                    pitch_rep = pitch.numbered_pitch
                elif output_type == 'chroma':
                    pitch_rep = pitch.numbered_pitch  # convert later
                elif output_type == 'sargamLetter':
                    pitch_rep = pitch.sargam_letter
                elif output_type == 'octavedSargamLetter':
                    pitch_rep = pitch.octaved_sargam_letter
                elif output_type == 'scaleDegree':
                    pitch_rep = pitch.scale_degree
                else:
                    raise ValueError(f'output_type not recognized: {output_type}')

                result.append({
                    'time': start_time,
                    'pitch': pitch_rep,
                    'articulation': p_idx == 0 and art is not None,
                })

                if p_idx < len(traj.pitches) - 1:
                    if traj.dur_array is None:
                        raise ValueError('traj.dur_array is None')
                    start_time += traj.dur_array[p_idx] * traj.dur_tot

    # Filter duplicate (same time + pitch)
    filtered: List[Dict] = []
    for i, obj in enumerate(result):
        if i < len(result) - 1:
            nxt = result[i + 1]
            if obj['pitch'] == nxt['pitch'] and obj['time'] == nxt['time']:
                continue
        filtered.append(obj)
    result = filtered

    # If adjacent entries share the same time but different pitch, keep only
    # the latter.
    filtered2: List[Dict] = []
    for i, obj in enumerate(result):
        if i == 0 or i == len(result) - 1:
            filtered2.append(obj)
        else:
            if obj['time'] != result[i + 1]['time']:
                filtered2.append(obj)
    result = filtered2

    # Convert to chroma after filtering if requested
    if output_type == 'chroma':
        for pt in result:
            if pt['pitch'] != 'silence':
                pt['pitch'] = Pitch.pitch_number_to_chroma(int(pt['pitch']))

    return result


def condensed_durations(
    trajs: List[Trajectory],
    output_type: str = 'pitchNumber',
    max_silence: float = 5,
    exclude_silence: bool = True,
) -> List[Dict]:
    """Merge adjacent identical pitches into single entries with summed durations.

    Port of TypeScript ``condensedDurations``.

    Args:
        trajs: Sequence of trajectories.
        output_type: Pitch representation (see :func:`pitch_times`).
        max_silence: Maximum silence duration to absorb into preceding pitch.
        exclude_silence: Whether to remove silence entries from the result.

    Returns:
        List of ``{'dur': float, 'pitch': int|str}``.
    """
    pt = pitch_times(trajs, output_type='pitchNumber')

    # Apply chroma conversion if needed
    if output_type == 'chroma':
        for entry in pt:
            if entry['pitch'] != 'silence':
                entry['pitch'] = Pitch.pitch_number_to_chroma(int(entry['pitch']))

    # Remove latter of adjacent items where pitch is equal
    deduped: List[Dict] = []
    for i, obj in enumerate(pt):
        if i == 0 or obj['pitch'] != pt[i - 1]['pitch']:
            deduped.append(obj)

    # Compute durations from time gaps
    end_time = sum(t.dur_tot for t in trajs)
    ends = [obj['time'] for obj in deduped[1:]] + [end_time]
    durations = [
        {'dur': ends[i] - obj['time'], 'pitch': obj['pitch']}
        for i, obj in enumerate(deduped)
    ]

    # Condense silences
    condensed: List[Dict] = []
    for i, obj in enumerate(durations):
        if obj['pitch'] == 'silence':
            if i == 0:
                condensed.append(dict(obj))
            elif obj['dur'] < max_silence:
                condensed[-1]['dur'] += obj['dur']
            else:
                condensed[-1]['dur'] += max_silence
                remainder = obj['dur'] - max_silence
                condensed.append({'dur': remainder, 'pitch': 'silence'})
        else:
            condensed.append(dict(obj))

    if exclude_silence:
        condensed = [obj for obj in condensed if obj['pitch'] != 'silence']

    return condensed


def durations_of_pitch_onsets(
    trajs: List[Trajectory],
    output_type: str = 'pitchNumber',
    count_type: str = 'cumulative',
    max_silence: float = 5,
    exclude_silence: bool = True,
) -> Dict:
    """Aggregate pitch durations into a pitch→duration mapping.

    Port of TypeScript ``durationsOfPitchOnsets``.

    Args:
        trajs: Sequence of trajectories.
        output_type: Pitch representation.
        count_type: ``'cumulative'`` for raw durations, ``'proportional'``
            for normalised (summing to 1.0).
        max_silence: Maximum silence duration to absorb.
        exclude_silence: Whether to exclude silence from results.

    Returns:
        Dict mapping pitch to total duration.
    """
    pt = pitch_times(trajs, output_type='pitchNumber')

    if output_type == 'chroma':
        for entry in pt:
            if entry['pitch'] != 'silence':
                entry['pitch'] = Pitch.pitch_number_to_chroma(int(entry['pitch']))

    # Remove adjacent duplicates
    deduped: List[Dict] = []
    for i, obj in enumerate(pt):
        if i == 0 or obj['pitch'] != pt[i - 1]['pitch']:
            deduped.append(obj)

    end_time = sum(t.dur_tot for t in trajs)
    ends = [obj['time'] for obj in deduped[1:]] + [end_time]
    durations = [
        {'dur': ends[i] - obj['time'], 'pitch': obj['pitch']}
        for i, obj in enumerate(deduped)
    ]

    # Condense silences
    condensed: List[Dict] = []
    for i, obj in enumerate(durations):
        if obj['pitch'] == 'silence':
            if i == 0:
                condensed.append(dict(obj))
            elif obj['dur'] < max_silence:
                condensed[-1]['dur'] += obj['dur']
            else:
                condensed[-1]['dur'] += max_silence
                remainder = obj['dur'] - max_silence
                condensed.append({'dur': remainder, 'pitch': 'silence'})
        else:
            condensed.append(dict(obj))

    # Aggregate by pitch
    pitch_durations: Dict = {}
    for obj in condensed:
        key = obj['pitch']
        pitch_durations[key] = pitch_durations.get(key, 0) + obj['dur']

    if exclude_silence:
        pitch_durations.pop('silence', None)
        condensed = [obj for obj in condensed if obj['pitch'] != 'silence']

    if count_type == 'proportional':
        total = sum(obj['dur'] for obj in condensed)
        if total > 0:
            for key in pitch_durations:
                pitch_durations[key] /= total

    return pitch_durations


def segment_by_duration(
    piece,
    duration: float = 10,
    boundary_type: str = 'rounded',
    inst: int = 0,
    remove_empty: bool = True,
) -> List[List[Trajectory]]:
    """Split a piece into fixed-duration time windows of trajectories.

    Port of TypeScript ``segmentByDuration``.

    Args:
        piece: A ``Piece`` instance.
        duration: Length of each segment in seconds.
        boundary_type: How to assign border trajectories —
            ``'left'``, ``'right'``, or ``'rounded'``.
        inst: Instrument index.
        remove_empty: Whether to drop empty segments.

    Returns:
        List of trajectory lists, one per segment.
    """
    if piece.dur_tot is None:
        raise ValueError('piece.dur_tot is None')

    num_segments = math.ceil(piece.dur_tot / duration)
    segments: List[List[Trajectory]] = [[] for _ in range(num_segments)]

    trajs = piece.all_trajectories(inst)
    traj_durs = [t.dur_tot for t in trajs]
    starts = get_starts(traj_durs)

    for i, traj in enumerate(trajs):
        start = starts[i]
        end = start + traj.dur_tot

        if math.floor(start / duration) == math.floor(end / duration):
            segment_idx = math.floor(start / duration)
        elif boundary_type == 'left':
            segment_idx = math.floor(start / duration)
        elif boundary_type == 'right':
            segment_idx = math.floor(end / duration)
        else:  # rounded
            center = (start + end) / 2
            segment_idx = math.floor(center / duration)

        # Clamp to valid range
        segment_idx = min(segment_idx, num_segments - 1)

        # Exclude final trajectory if it's a silence
        if not (i == len(trajs) - 1 and traj.id == 12):
            segments[segment_idx].append(traj)

    if remove_empty:
        segments = [seg for seg in segments if len(seg) > 0]

    return segments


def pattern_counter(
    trajs: List[Trajectory],
    size: int = 2,
    max_lag_time: float = 3,
    sort: bool = True,
    output_type: str = 'pitchNumber',
    target_pitch: Optional[Union[int, str]] = None,
    min_size: int = 1,
) -> List[Dict]:
    """Find N-gram pitch patterns and their occurrence counts.

    Port of TypeScript ``patternCounter``.

    Args:
        trajs: Sequence of trajectories.
        size: Length of each pattern (N-gram size).
        max_lag_time: Silences shorter than this are absorbed; longer reset
            the pattern window.
        sort: Whether to sort results by count descending.
        output_type: Pitch representation.
        target_pitch: If provided, only return patterns ending with this pitch.
        min_size: Minimum occurrence count to include.

    Returns:
        List of ``{'pattern': [p1, p2, ...], 'count': int}``.
    """
    pt = pitch_times(trajs, output_type=output_type)

    # Remove adjacent duplicates unless articulation is True on the second
    i = 1
    while i < len(pt):
        if pt[i]['pitch'] == pt[i - 1]['pitch']:
            if pt[i].get('articulation') is True:
                i += 1
            else:
                pt.pop(i)
        else:
            i += 1

    # Compute durations
    end_time = sum(t.dur_tot for t in trajs)
    ends = [obj['time'] for obj in pt[1:]] + [end_time]
    durations = [
        {
            'dur': ends[idx] - obj['time'],
            'pitch': obj['pitch'],
            'articulation': obj.get('articulation', False),
        }
        for idx, obj in enumerate(pt)
    ]

    # Condense silences
    condensed: List[Dict] = []
    for idx, obj in enumerate(durations):
        if obj['pitch'] == 'silence':
            if idx == 0:
                condensed.append(dict(obj))
            elif obj['dur'] < max_lag_time:
                condensed[-1]['dur'] += obj['dur']
            else:
                condensed[-1]['dur'] += max_lag_time
                remainder = obj['dur'] - max_lag_time
                condensed.append({'dur': remainder, 'pitch': 'silence'})
        else:
            condensed.append(dict(obj))

    # Build nested pattern tree
    patterns: Dict = {}
    sub_pattern: list = []

    for obj in condensed:
        if obj['pitch'] == 'silence':
            sub_pattern = []
        elif len(sub_pattern) < size:
            sub_pattern.append(obj['pitch'])
        else:
            sub_pattern.pop(0)
            sub_pattern.append(obj['pitch'])

        if obj['pitch'] != 'silence' and len(sub_pattern) == size:
            sel = patterns
            for p_idx, p in enumerate(sub_pattern):
                if p not in sel:
                    sel[p] = 0 if p_idx == len(sub_pattern) - 1 else {}
                if p_idx == len(sub_pattern) - 1:
                    if isinstance(sel[p], (int, float)):
                        sel[p] += 1
                    else:
                        raise ValueError('Unexpected non-numeric value in pattern tree')
                else:
                    if isinstance(sel[p], dict):
                        sel = sel[p]

    # Flatten tree to list
    out: List[Dict] = []

    def _recurse(obj: Dict, pattern: list):
        for key in obj:
            if isinstance(obj[key], (int, float)):
                out.append({
                    'pattern': pattern + [key],
                    'count': int(obj[key]),
                })
            else:
                _recurse(obj[key], pattern + [key])

    _recurse(patterns, [])

    if sort:
        out.sort(key=lambda x: x['count'], reverse=True)

    if target_pitch is not None:
        out = [o for o in out if o['pattern'][-1] == target_pitch]

    out = [o for o in out if o['count'] >= min_size]

    return out


def chroma_seq_to_condensed_pitch_nums(chroma_seq: List[int]) -> List[int]:
    """Octave-shift chroma pitches to minimise the range.

    Port of TypeScript ``chromaSeqToCondensedPitchNums``.

    Given an array of chroma pitches, shift some up or down by an octave such
    that the maximum difference between any two elements is minimised.

    Args:
        chroma_seq: List of chroma pitch values (0–11).

    Returns:
        List of adjusted pitch values in the original order.
    """
    if not chroma_seq:
        return []

    n = len(chroma_seq)

    # argSort: indices that would sort the array
    sort_idxs = sorted(range(n), key=lambda i: chroma_seq[i])
    # inverse permutation
    dbl_arg_sort = sorted(range(n), key=lambda i: sort_idxs[i])
    sorted_vals = [chroma_seq[i] for i in sort_idxs]

    # Differences between adjacent sorted values (with wrap-around)
    difs = []
    for i in range(len(sorted_vals)):
        if i == len(sorted_vals) - 1:
            difs.append(sorted_vals[0] - (sorted_vals[i] - 12))
        else:
            difs.append(sorted_vals[i + 1] - sorted_vals[i])

    max_dif_idx = difs.index(max(difs))

    # Everything after max gap gets shifted down by an octave
    out = []
    for i, val in enumerate(sorted_vals):
        if i > max_dif_idx:
            out.append(val - 12)
        else:
            out.append(val)

    # Unsort back to original order
    unsorted = [out[i] for i in dbl_arg_sort]
    return unsorted
