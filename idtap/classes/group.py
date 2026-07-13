from __future__ import annotations
from typing import List, Optional, Dict
import uuid

from .trajectory import Trajectory
from .pitch import Pitch

class Group:
    def __init__(self, options: Optional[Dict] = None) -> None:
        opts = options or {}
        
        # Parameter validation
        self._validate_parameters(opts)
        self.trajectories: List[Trajectory] = opts.get('trajectories', [])
        self.id: str = str(opts.get('id', uuid.uuid4()))
        # sort and validate
        self.trajectories.sort(key=lambda t: self._require_num(t))
        if len(self.trajectories) < 2:
            raise ValueError('Group must have at least 2 trajectories')
        if not self.test_for_adjacency():
            raise ValueError('Trajectories are not adjacent')
        for traj in self.trajectories:
            traj.group_id = self.id

    def _validate_parameters(self, opts: dict) -> None:
        """Validate constructor parameters and provide helpful error messages."""
        if not opts:
            return
            
        # Define allowed parameter names
        allowed_keys = {'trajectories', 'id'}
        provided_keys = set(opts.keys())
        invalid_keys = provided_keys - allowed_keys
        
        # Check for invalid parameter names
        if invalid_keys:
            error_messages = []
            
            for key in invalid_keys:
                if key == 'trajectory_list' or key == 'trajs':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'trajectories'?")
                elif key == 'group_id' or key == 'unique_id':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'id'?")
                else:
                    error_messages.append(f"Invalid parameter: '{key}'")
            
            error_msg = "; ".join(error_messages)
            error_msg += f". Allowed parameters: {sorted(allowed_keys)}"
            raise ValueError(error_msg)
        
        # Validate parameter types
        if 'trajectories' in opts:
            if not isinstance(opts['trajectories'], list):
                raise TypeError(f"Parameter 'trajectories' must be a list, got {type(opts['trajectories']).__name__}")
        
        if 'id' in opts and opts['id'] is not None:
            if not isinstance(opts['id'], str):
                # Allow UUID objects to be converted
                opts['id'] = str(opts['id'])

    def _require_num(self, traj: Trajectory) -> int:
        if traj.num is None:
            raise ValueError('Trajectory must have a num')
        return traj.num

    @property
    def min_freq(self) -> float:
        return min(t.min_freq for t in self.trajectories)

    @property
    def max_freq(self) -> float:
        return max(t.max_freq for t in self.trajectories)

    def all_pitches(self, repetition: bool = True) -> List[Pitch]:
        pitches: List[Pitch] = []
        for t in self.trajectories:
            if t.id != 12:
                pitches.extend(t.pitches)
        if not repetition:
            out: List[Pitch] = []
            for i, p in enumerate(pitches):
                if i == 0:
                    out.append(p)
                else:
                    prev = out[-1]
                    if not (p.swara == prev.swara and p.oct == prev.oct and p.raised == prev.raised):
                        out.append(p)
            return out
        return pitches

    def test_for_adjacency(self) -> bool:
        phrase_idxs = {t.phrase_idx for t in self.trajectories}
        if len(phrase_idxs) != 1:
            return False
        nums = [self._require_num(t) for t in self.trajectories]
        nums.sort()
        diffs = [nums[i+1]-nums[i] for i in range(len(nums)-1)]
        return all(d == 1 for d in diffs)

    def add_traj(self, traj: Trajectory) -> None:
        self.trajectories.append(traj)
        self.trajectories.sort(key=lambda t: self._require_num(t))
        if not self.test_for_adjacency():
            raise ValueError('Trajectories are not adjacent')
        traj.group_id = self.id

    def to_json(self) -> Dict:
        return {
            'trajectories': [t.to_json() if hasattr(t, 'to_json') else t for t in self.trajectories],
            'id': self.id,
        }

    @staticmethod
    def from_json(obj: Dict) -> 'Group':
        trajs = [t if isinstance(t, Trajectory) else Trajectory.from_json(t)
                 for t in obj.get('trajectories', [])]
        return Group({'trajectories': trajs, 'id': obj.get('id')})
