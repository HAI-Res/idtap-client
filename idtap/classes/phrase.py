from __future__ import annotations
from typing import List, Dict, Optional, Any
import uuid

import humps
from ..utils import selective_decamelize

from .raga import Raga
from .trajectory import Trajectory
from .chikari import Chikari
from .group import Group
from .pitch import Pitch
from .automation import get_starts
from .note_view_phrase import NoteViewPhrase


PhraseCatType = Dict[str, Dict[str, bool]]


def init_phrase_categorization() -> PhraseCatType:
    return {
        "Phrase": {
            "Mohra": False,
            "Mukra": False,
            "Asthai": False,
            "Antara": False,
            "Manjha": False,
            "Abhog": False,
            "Sanchari": False,
            "Jhala": False,
        },
        "Elaboration": {
            "Vistar": False,
            "Barhat": False,
            "Prastar": False,
            "Bol Banao": False,
            "Bol Alap": False,
            "Bol Bandt": False,
            "Behlava": False,
            "Gat-kari": False,
            "Tan (Sapat)": False,
            "Tan (Gamak)": False,
            "Laykari": False,
            "Tihai": False,
            "Chakradar": False,
        },
        "Vocal Articulation": {
            "Bol": False,
            "Non-Tom": False,
            "Tarana": False,
            "Aakar": False,
            "Sargam": False,
        },
        "Instrumental Articulation": {
            "Bol": False,
            "Non-Bol": False,
        },
        "Incidental": {
            "Talk/Conversation": False,
            "Praise ('Vah')": False,
            "Tuning": False,
            "Pause": False,
        },
    }


class Phrase:
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        opts = selective_decamelize(options or {})
        
        # Parameter validation
        self._validate_parameters(opts)
        trajectories_in = opts.get('trajectories', [])
        self.start_time: Optional[float] = opts.get('start_time')
        self.raga: Optional[Raga] = opts.get('raga')
        instrumentation = opts.get('instrumentation', ['Sitar'])
        self.instrumentation: List[str] = instrumentation
        trajectory_grid_opt = opts.get('trajectory_grid')
        chikari_grid_opt = opts.get('chikari_grid')
        chikaris_in = opts.get('chikaris', {})
        groups_grid = opts.get('groups_grid')
        categorization_grid = opts.get('categorization_grid')
        unique_id = opts.get('unique_id')
        self.piece_idx = opts.get('piece_idx')
        ad_hoc_cat = opts.get('ad_hoc_categorization_grid')

        # Initialize is_section_start (optional boolean)
        self.is_section_start = opts.get('is_section_start')

        trajs: List[Trajectory] = []
        for t in trajectories_in:
            if not isinstance(t, Trajectory):
                t = Trajectory(t)  # type: ignore
            trajs.append(t)

        if trajectory_grid_opt is not None:
            self.trajectory_grid = trajectory_grid_opt
            for _ in range(len(self.trajectory_grid), len(instrumentation)):
                self.trajectory_grid.append([])
        else:
            self.trajectory_grid = [trajs]
            for _ in range(1, len(instrumentation)):
                self.trajectory_grid.append([])

        chikaris_dict: Dict[str, Chikari] = {}
        for k, v in chikaris_in.items():
            if not isinstance(v, Chikari):
                chikaris_dict[str(k)] = Chikari(v)  # type: ignore
            else:
                chikaris_dict[str(k)] = v

        if chikari_grid_opt is not None:
            self.chikari_grid = chikari_grid_opt
            for _ in range(len(self.chikari_grid), len(instrumentation)):
                self.chikari_grid.append({})
        else:
            self.chikari_grid = [chikaris_dict]
            for _ in range(1, len(instrumentation)):
                self.chikari_grid.append({})

        if len(self.trajectories) == 0:
            self.dur_tot = opts.get('dur_tot', 1)
            self.dur_array = opts.get('dur_array', [])
        else:
            self.dur_tot_from_trajectories()
            self.dur_array_from_trajectories()
            dur_tot = opts.get('dur_tot')
            if dur_tot is not None and dur_tot != self.dur_tot:
                for t in self.trajectories:
                    t.dur_tot = t.dur_tot * dur_tot / self.dur_tot
                self.dur_tot = dur_tot
            dur_array = opts.get('dur_array')
            if dur_array is not None and dur_array != self.dur_array:
                for i, t in enumerate(self.trajectories):
                    t.dur_tot = t.dur_tot * dur_array[i] / self.dur_array[i]
                self.dur_array = dur_array
                self.dur_tot_from_trajectories()

        self.assign_start_times()
        self.assign_traj_nums()

        if groups_grid is not None:
            self.groups_grid: List[List[Group]] = groups_grid
        else:
            self.groups_grid = [ [] for _ in instrumentation ]

        self.categorization_grid: List[PhraseCatType] = categorization_grid or []
        if len(self.categorization_grid) == 0:
            for _ in range(len(self.trajectory_grid)):
                self.categorization_grid.append(init_phrase_categorization())
        if self.categorization_grid[0]['Elaboration'].get('Bol Alap') is None:
            for cat in self.categorization_grid:
                cat['Elaboration']['Bol Alap'] = False

        self.ad_hoc_categorization_grid: List[str] = ad_hoc_cat or []
        self.unique_id = str(unique_id or uuid.uuid4())

    def _validate_parameters(self, opts: Dict[str, Any]) -> None:
        """Validate constructor parameters and provide helpful error messages."""
        if not opts:
            return
            
        # Define allowed parameter names
        allowed_keys = {
            'trajectories', 'start_time', 'raga', 'instrumentation', 'trajectory_grid',
            'chikari_grid', 'chikaris', 'groups_grid', 'categorization_grid',
            'unique_id', 'piece_idx', 'ad_hoc_categorization_grid', 'dur_tot', 'dur_array',
            'is_section_start'
        }
        provided_keys = set(opts.keys())
        invalid_keys = provided_keys - allowed_keys
        
        # Check for invalid parameter names with helpful suggestions
        if invalid_keys:
            error_messages = []
            
            for key in invalid_keys:
                if key == 'duration_total' or key == 'duration':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'dur_tot'?")
                elif key == 'duration_array':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'dur_array'?")
                elif key == 'start':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'start_time'?")
                elif key == 'trajectory_list' or key == 'trajs':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'trajectories'?")
                elif key == 'instruments':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'instrumentation'?")
                else:
                    error_messages.append(f"Invalid parameter: '{key}'")
            
            error_msg = "; ".join(error_messages)
            error_msg += f". Allowed parameters: {sorted(allowed_keys)}"
            raise ValueError(error_msg)
        
        # Validate parameter types and values
        self._validate_parameter_types(opts)
        self._validate_parameter_values(opts)
    
    def _validate_parameter_types(self, opts: Dict[str, Any]) -> None:
        """Validate that all parameters have correct types."""
        if 'trajectories' in opts:
            if not isinstance(opts['trajectories'], list):
                raise TypeError(f"Parameter 'trajectories' must be a list, got {type(opts['trajectories']).__name__}")
        
        if 'start_time' in opts and opts['start_time'] is not None:
            if not isinstance(opts['start_time'], (int, float)):
                raise TypeError(f"Parameter 'start_time' must be a number, got {type(opts['start_time']).__name__}")
        
        if 'raga' in opts and opts['raga'] is not None:
            if not isinstance(opts['raga'], (Raga, dict)):
                raise TypeError(f"Parameter 'raga' must be a Raga object or dict, got {type(opts['raga']).__name__}")
        
        if 'instrumentation' in opts:
            if not isinstance(opts['instrumentation'], list):
                raise TypeError(f"Parameter 'instrumentation' must be a list, got {type(opts['instrumentation']).__name__}")
            if not all(isinstance(item, str) for item in opts['instrumentation']):
                raise TypeError("All items in 'instrumentation' must be strings")
        
        if 'trajectory_grid' in opts and opts['trajectory_grid'] is not None:
            if not isinstance(opts['trajectory_grid'], list):
                raise TypeError(f"Parameter 'trajectory_grid' must be a list, got {type(opts['trajectory_grid']).__name__}")
            if not all(isinstance(row, list) for row in opts['trajectory_grid']):
                raise TypeError("All items in 'trajectory_grid' must be lists")
        
        if 'chikari_grid' in opts and opts['chikari_grid'] is not None:
            if not isinstance(opts['chikari_grid'], list):
                raise TypeError(f"Parameter 'chikari_grid' must be a list, got {type(opts['chikari_grid']).__name__}")
            if not all(isinstance(row, dict) for row in opts['chikari_grid']):
                raise TypeError("All items in 'chikari_grid' must be dictionaries")
        
        if 'chikaris' in opts and opts['chikaris'] is not None:
            if not isinstance(opts['chikaris'], dict):
                raise TypeError(f"Parameter 'chikaris' must be a dict, got {type(opts['chikaris']).__name__}")
        
        if 'groups_grid' in opts and opts['groups_grid'] is not None:
            if not isinstance(opts['groups_grid'], list):
                raise TypeError(f"Parameter 'groups_grid' must be a list, got {type(opts['groups_grid']).__name__}")
            if not all(isinstance(row, list) for row in opts['groups_grid']):
                raise TypeError("All items in 'groups_grid' must be lists")
        
        if 'categorization_grid' in opts and opts['categorization_grid'] is not None:
            if not isinstance(opts['categorization_grid'], list):
                raise TypeError(f"Parameter 'categorization_grid' must be a list, got {type(opts['categorization_grid']).__name__}")
        
        if 'unique_id' in opts and opts['unique_id'] is not None:
            if not isinstance(opts['unique_id'], str):
                raise TypeError(f"Parameter 'unique_id' must be a string, got {type(opts['unique_id']).__name__}")
        
        if 'piece_idx' in opts and opts['piece_idx'] is not None:
            if not isinstance(opts['piece_idx'], int):
                raise TypeError(f"Parameter 'piece_idx' must be an integer, got {type(opts['piece_idx']).__name__}")
        
        if 'ad_hoc_categorization_grid' in opts and opts['ad_hoc_categorization_grid'] is not None:
            if not isinstance(opts['ad_hoc_categorization_grid'], list):
                raise TypeError(f"Parameter 'ad_hoc_categorization_grid' must be a list, got {type(opts['ad_hoc_categorization_grid']).__name__}")
            if not all(isinstance(item, str) for item in opts['ad_hoc_categorization_grid']):
                raise TypeError("All items in 'ad_hoc_categorization_grid' must be strings")
        
        if 'dur_tot' in opts and opts['dur_tot'] is not None:
            if not isinstance(opts['dur_tot'], (int, float)):
                raise TypeError(f"Parameter 'dur_tot' must be a number, got {type(opts['dur_tot']).__name__}")
        
        if 'dur_array' in opts and opts['dur_array'] is not None:
            if not isinstance(opts['dur_array'], list):
                raise TypeError(f"Parameter 'dur_array' must be a list, got {type(opts['dur_array']).__name__}")
            if not all(isinstance(item, (int, float)) for item in opts['dur_array']):
                raise TypeError("All items in 'dur_array' must be numbers")

        # Validate is_section_start
        if 'is_section_start' in opts and opts['is_section_start'] is not None:
            if not isinstance(opts['is_section_start'], bool):
                raise TypeError(f"Parameter 'is_section_start' must be a boolean, got {type(opts['is_section_start']).__name__}")

    def _validate_parameter_values(self, opts: Dict[str, Any]) -> None:
        """Validate that parameter values are in valid ranges."""
        if 'start_time' in opts and opts['start_time'] is not None:
            if opts['start_time'] < 0:
                raise ValueError(f"Parameter 'start_time' must be non-negative, got {opts['start_time']}")
        
        if 'piece_idx' in opts and opts['piece_idx'] is not None:
            if opts['piece_idx'] < 0:
                raise ValueError(f"Parameter 'piece_idx' must be non-negative, got {opts['piece_idx']}")
        
        if 'dur_tot' in opts and opts['dur_tot'] is not None:
            if opts['dur_tot'] <= 0:
                raise ValueError(f"Parameter 'dur_tot' must be positive, got {opts['dur_tot']}")
        
        if 'dur_array' in opts and opts['dur_array'] is not None:
            dur_array = opts['dur_array']
            if any(d < 0 for d in dur_array):
                raise ValueError("All values in 'dur_array' must be non-negative")
            if len(dur_array) > 0 and sum(dur_array) == 0:
                raise ValueError("'dur_array' cannot have all zero values")
        
        # Validate categorization grid structure
        if 'categorization_grid' in opts and opts['categorization_grid'] is not None:
            for i, cat in enumerate(opts['categorization_grid']):
                if not isinstance(cat, dict):
                    raise TypeError(f"categorization_grid[{i}] must be a dictionary")
                required_categories = ['Phrase', 'Elaboration', 'Vocal Articulation', 'Instrumental Articulation', 'Incidental']
                for req_cat in required_categories:
                    if req_cat not in cat:
                        import warnings
                        warnings.warn(f"categorization_grid[{i}] missing category '{req_cat}'. "
                                     "This may cause issues with phrase categorization.", UserWarning)

    # ------------------------------------------------------------------
    def update_fundamental(self, fundamental: float) -> None:
        for traj in self.trajectories:
            traj.update_fundamental(fundamental)

    def get_groups(self, idx: int = 0) -> List[Group]:
        if idx < len(self.groups_grid) and self.groups_grid[idx] is not None:
            return self.groups_grid[idx]
        raise Exception('No groups for this index')

    def get_group_from_id(self, gid: str) -> Optional[Group]:
        for g_list in self.groups_grid:
            for g in g_list:
                if g.id == gid:
                    return g
        return None

    def assign_phrase_idx(self) -> None:
        for traj in self.trajectories:
            traj.phrase_idx = self.piece_idx

    def assign_traj_nums(self) -> None:
        for i, traj in enumerate(self.trajectories):
            traj.num = i

    def dur_tot_from_trajectories(self) -> None:
        self.dur_tot = sum(t.dur_tot for t in self.trajectories)

    def dur_array_from_trajectories(self) -> None:
        self.dur_tot_from_trajectories()
        if self.dur_tot == 0:
            self.dur_array = [0 for _ in self.trajectories]
        else:
            self.dur_array = [t.dur_tot / self.dur_tot for t in self.trajectories]

    def compute(self, x: float, log_scale: bool = False):
        if self.dur_array is None:
            raise Exception('durArray is undefined')
        if len(self.dur_array) == 0:
            return None
        starts = get_starts(self.dur_array)
        idx = 0
        for i, s in enumerate(starts):
            if x >= s:
                idx = i
            else:
                break
        inner_x = (x - starts[idx]) / self.dur_array[idx]
        traj = self.trajectories[idx]
        return traj.compute(inner_x, log_scale)

    def realign_pitches(self) -> None:
        if not self.raga:
            return
        ratios = self.raga.stratified_ratios
        for traj in self.trajectories:
            new_pitches = []
            for p in traj.pitches:
                opts = p.to_json()
                opts['ratios'] = ratios
                new_pitches.append(Pitch(opts))
            traj.pitches = new_pitches

    def assign_start_times(self) -> None:
        if self.dur_array is None:
            raise Exception('durArray is undefined')
        if self.dur_tot is None:
            raise Exception('durTot is undefined')
        starts = [s * self.dur_tot for s in get_starts(self.dur_array)]
        for traj, st in zip(self.trajectories, starts):
            traj.start_time = st

    def get_range(self) -> Dict[str, Dict[str, Any]]:
        all_pitches = [p for t in self.trajectories for p in t.pitches]
        all_pitches.sort(key=lambda p: p.frequency)
        low = all_pitches[0]
        high = all_pitches[-1]
        low_obj = {
            'frequency': low.frequency,
            'swara': low.swara,
            'oct': low.oct,
            'raised': low.raised,
            'numberedPitch': low.numbered_pitch,
        }
        high_obj = {
            'frequency': high.frequency,
            'swara': high.swara,
            'oct': high.oct,
            'raised': high.raised,
            'numberedPitch': high.numbered_pitch,
        }
        return {'min': low_obj, 'max': high_obj}

    def consolidate_silent_trajs(self) -> None:
        chain = False
        start: Optional[int] = None
        del_idxs: List[int] = []
        for i, traj in enumerate(self.trajectories):
            if traj.id == 12:
                if not chain:
                    start = i
                    chain = True
                if i == len(self.trajectories) - 1:
                    if start is None:
                        raise Exception('start is undefined')
                    extra = sum(t.dur_tot for t in self.trajectories[start+1:])
                    self.trajectories[start].dur_tot += extra
                    del_idxs.extend(range(start+1, len(self.trajectories)))
            else:
                if chain:
                    if start is None:
                        raise Exception('start is undefined')
                    extra = sum(t.dur_tot for t in self.trajectories[start+1:i])
                    self.trajectories[start].dur_tot += extra
                    del_idxs.extend(range(start+1, i))
                    chain = False
                    start = None
        new_ts: List[Trajectory] = []
        for traj in self.trajectories:
            if traj.num is None:
                raise Exception('traj.num is undefined')
            if traj.num not in del_idxs:
                new_ts.append(traj)
        self.trajectory_grid[0] = new_ts
        self.dur_array_from_trajectories()
        self.assign_start_times()
        self.assign_traj_nums()
        self.assign_phrase_idx()

    def chikaris_during_traj(self, traj: Trajectory, track: int):
        start = traj.start_time
        if start is None:
            return []
        dur = traj.dur_tot
        end = start + dur
        chikaris = self.chikari_grid[0]
        out = []
        for k, c in chikaris.items():
            time = float(k)
            if time >= start and time <= end:
                real_time = time + (self.start_time or 0)
                out.append({
                    'time': real_time,
                    'phraseTimeKey': k,
                    'phraseIdx': self.piece_idx,
                    'track': track,
                    'chikari': c,
                    'uId': c.unique_id,
                })
        return out

    # ---------------------------- properties ---------------------------
    @property
    def trajectories(self) -> List[Trajectory]:
        return self.trajectory_grid[0]

    @property
    def chikaris(self) -> Dict[str, Chikari]:
        return self.chikari_grid[0]

    @chikaris.setter
    def chikaris(self, val: Dict[str, Chikari]) -> None:
        self.chikari_grid[0] = val

    @property
    def swara(self) -> List[Dict[str, Any]]:
        swara = []
        if self.start_time is None:
            raise Exception('startTime is undefined')
        for traj in self.trajectories:
            if traj.id != 12:
                if traj.dur_array is None:
                    raise Exception('traj.durArray is undefined')
                if traj.start_time is None:
                    raise Exception('traj.startTime is undefined')
                if len(traj.dur_array) == len(traj.pitches) - 1:
                    pitches = traj.pitches[:-1]
                else:
                    pitches = traj.pitches
                for i, pitch in enumerate(pitches):
                    st = self.start_time + traj.start_time
                    time = st + get_starts(traj.dur_array)[i] * traj.dur_tot
                    swara.append({'pitch': pitch, 'time': time})
        return swara

    def all_pitches(self, repetition: bool = True) -> List[Pitch]:
        pitches: List[Pitch] = []
        for traj in self.trajectories:
            if traj.id != 12:
                pitches.extend(traj.pitches)
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

    def first_traj_idxs(self) -> List[int]:
        idxs: List[int] = []
        ct = 0
        silent_trigger = False
        last_vowel: Optional[str] = None
        end_consonant_trigger: Optional[bool] = None
        for t_idx, traj in enumerate(self.trajectories):
            if traj.id != 12:
                c1 = ct == 0
                c2 = silent_trigger
                c3 = traj.start_consonant is not None
                c4 = end_consonant_trigger
                c5 = traj.vowel != last_vowel
                if c1 or c2 or c3 or c4 or c5:
                    idxs.append(t_idx)
                ct += 1
                end_consonant_trigger = traj.end_consonant is not None
                last_vowel = traj.vowel
            silent_trigger = traj.id == 12
        return idxs

    def traj_idx_from_time(self, time: float) -> int:
        phrase_time = time - (self.start_time or 0)
        small_offset = 1e-10
        matches = [
            traj for traj in self.trajectories
            if traj.start_time is not None and
               phrase_time >= traj.start_time - small_offset and
               phrase_time < traj.start_time + traj.dur_tot
        ]
        if not matches:
            raise Exception('No trajectory found')
        return matches[0].num  # type: ignore

    def to_json(self) -> Dict[str, Any]:
        return {
            'durTot': self.dur_tot,
            'durArray': self.dur_array,
            'chikaris': {k: c.to_json() for k, c in self.chikaris.items()},
            # raga: stripped — inherited from the piece (idtap-contract PHRASE-1)
            'startTime': self.start_time,
            'trajectoryGrid': [[t.to_json() for t in row] for row in self.trajectory_grid],
            'instrumentation': self.instrumentation,
            'groupsGrid': self.groups_grid,
            'categorizationGrid': self.categorization_grid,
            'uniqueId': self.unique_id,
            'adHocCategorizationGrid': self.ad_hoc_categorization_grid,
            'isSectionStart': self.is_section_start,
        }

    @staticmethod
    def from_json(obj: Dict[str, Any], ratios=None, fundamental=None) -> 'Phrase':
        opts = selective_decamelize(obj)
        # Thread raga context to trajectories; if none passed, fall back to the
        # phrase's own embedded raga (legacy data) — idtap-contract PHRASE-2.
        phrase_raga = opts.get('raga')
        if phrase_raga is not None and not isinstance(phrase_raga, Raga):
            phrase_raga = Raga.from_json(phrase_raga)
            opts['raga'] = phrase_raga
        r = ratios if ratios is not None else (phrase_raga.stratified_ratios if phrase_raga else None)
        f = fundamental if fundamental is not None else (phrase_raga.fundamental if phrase_raga else None)
        trajectory_grid = opts.get('trajectory_grid')
        if trajectory_grid is not None:
            tg = []
            for row in trajectory_grid:
                tg.append([Trajectory.from_json(t, ratios=r, fundamental=f) for t in row])
            opts['trajectory_grid'] = tg
        trajectories = opts.get('trajectories')
        if trajectories is not None:
            opts['trajectories'] = [Trajectory.from_json(t, ratios=r, fundamental=f) for t in trajectories]
        chikaris = opts.get('chikaris')
        if chikaris is not None:
            new_c = {}
            for k, v in chikaris.items():
                new_c[str(k)] = Chikari.from_json(v)
            opts['chikaris'] = new_c
        chikari_grid = opts.get('chikari_grid')
        if chikari_grid is not None:
            new_grid = []
            for cg in chikari_grid:
                new_obj = {}
                for k, v in cg.items():
                    new_obj[str(k)] = Chikari.from_json(v)
                new_grid.append(new_obj)
            opts['chikari_grid'] = new_grid
        # (raga already parsed above and threaded as context)
        return Phrase(opts)

    def to_note_view_phrase(self) -> 'NoteViewPhrase':
        pitches: List[Pitch] = []
        for traj in self.trajectories:
            if traj.id != 0:
                pitches.extend(traj.pitches)
            elif len(traj.articulations) > 0:
                pitches.extend(traj.pitches)
        return NoteViewPhrase({
            'pitches': pitches,
            'dur_tot': self.dur_tot,
            'raga': self.raga,
            'start_time': self.start_time,
        })

    def reset(self) -> None:
        self.dur_array_from_trajectories()
        self.assign_start_times()
        self.assign_phrase_idx()
        self.assign_traj_nums()


