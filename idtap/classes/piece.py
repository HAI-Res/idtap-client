from __future__ import annotations
from typing import List, Optional, Dict, Union, Any
from datetime import datetime, timezone
import warnings

from .phrase import Phrase
from .trajectory import Trajectory
from .pitch import Pitch
from .raga import Raga
from .meter import Meter
from ..enums import Instrument
from .chikari import Chikari
from .group import Group
from .automation import get_starts, get_ends
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .section import Section

SecCatType = Dict[str, Union[Dict[str, bool], str]]


def init_sec_categorization() -> SecCatType:
    return {
        "Pre-Chiz Alap": {"Pre-Chiz Alap": False},
        "Alap": {"Alap": False, "Jor": False, "Alap-Jhala": False},
        "Composition Type": {
            "Dhrupad": False,
            "Bandish": False,
            "Thumri": False,
            "Ghazal": False,
            "Qawwali": False,
            "Dhun": False,
            "Tappa": False,
            "Bhajan": False,
            "Kirtan": False,
            "Kriti": False,
            "Masitkhani Gat": False,
            "Razakhani Gat": False,
            "Ferozkhani Gat": False,
        },
        "Comp.-section/Tempo": {
            "Ati Vilambit": False,
            "Vilambit": False,
            "Madhya": False,
            "Drut": False,
            "Ati Drut": False,
            "Jhala": False,
        },
        "Tala": {"Ektal": False, "Tintal": False, "Rupak": False},
        "Improvisation": {"Improvisation": False},
        "Other": {"Other": False},
        "Top Level": "None",
    }


# ----------------------------------------------------------------------
# Helper used outside the class
# ----------------------------------------------------------------------
def durations_of_fixed_pitches(
    trajs: List[Trajectory],
    output_type: str = "pitchNumber",
    count_type: str = "cumulative",
) -> Dict:
    pitch_durs: Dict[Any, float] = {}
    for traj in trajs:
        traj_pitch_durs = traj.durations_of_fixed_pitches({"output_type": output_type})
        if not isinstance(traj_pitch_durs, dict):
            raise SyntaxError(
                "invalid trajPitchDurs type, must be object: " + str(traj_pitch_durs)
            )
        for k, v in traj_pitch_durs.items():
            pitch_durs[k] = pitch_durs.get(k, 0.0) + float(v)

    if count_type == "proportional":
        total = sum(pitch_durs.values()) or 1.0
        for k in pitch_durs:
            pitch_durs[k] /= total
    return pitch_durs


def _iso_utc(dt: datetime) -> str:
    """PROP-2: serialize a datetime as ISO-8601 UTC with an explicit 'Z'.
    Naive datetimes are assumed to be UTC (legacy values were stored naive)."""
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.isoformat().replace('+00:00', 'Z')


def _parse_utc(value) -> datetime:
    """PROP-2: parse a wire date into a tz-aware UTC datetime. Accepts an ISO
    string (with 'Z' or offset), a legacy Mongo {"$date": ...} wrapper, or an
    existing datetime. Naive results are treated as UTC."""
    if isinstance(value, datetime):
        dt = value
    else:
        if isinstance(value, dict) and "$date" in value:
            value = value["$date"]
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


class Piece:
    def __init__(self, options: Optional[dict] = None) -> None:
        opts = options or {}
        
        # Parameter validation
        self._validate_parameters(opts)
        raga_opt = opts.get("raga")
        if raga_opt is not None and not isinstance(raga_opt, Raga):
            raga_opt = Raga.from_json(raga_opt)
        self.raga: Raga = raga_opt or Raga()

        instrumentation = opts.get("instrumentation", [Instrument.Sitar])
        inst_list: List[Instrument] = []
        for i in instrumentation:
            if isinstance(i, Instrument):
                inst_list.append(i)
            else:
                try:
                    inst_list.append(Instrument(i))
                except ValueError:
                    inst_list.append(i)
        self.instrumentation = inst_list

        # Initialize trackTitles (after instrumentation is set)
        track_titles = opts.get('trackTitles')
        if track_titles is not None:
            self.track_titles = list(track_titles)  # Create a copy
        else:
            # Create empty strings for each instrument track
            self.track_titles = [''] * len(self.instrumentation)

        # Ensure trackTitles array matches instrumentation length
        while len(self.track_titles) < len(self.instrumentation):
            self.track_titles.append('')
        while len(self.track_titles) > len(self.instrumentation):
            self.track_titles.pop()

        self.possible_trajs: Dict[Instrument, List[int]] = {
            Instrument.Sitar: list(range(14)),
            Instrument.Vocal_M: [0, 1, 2, 3, 4, 5, 6, 12, 13],
            Instrument.Vocal_F: [0, 1, 2, 3, 4, 5, 6, 12, 13],
            Instrument.Sarangi: list(range(14)),
        }

        first_inst = self.instrumentation[0]
        self.traj_idxs = self.possible_trajs.get(first_inst, [])
        self.traj_idxs_grid = [self.possible_trajs.get(i, []) for i in self.instrumentation]

        phrase_grid = opts.get("phraseGrid")
        if phrase_grid is None:
            phrases = opts.get("phrases", [])
            phrase_grid = [phrases]
        grid: List[List[Phrase]] = []
        for row in phrase_grid:
            new_row = []
            for p in row:
                if not isinstance(p, Phrase):
                    new_row.append(Phrase.from_json(p))
                else:
                    new_row.append(p)
            grid.append(new_row)
        self.phrase_grid = grid

        self.title: str = opts.get("title", "untitled")
        self.date_created: datetime = opts.get("dateCreated", datetime.now(timezone.utc))
        self.date_modified: datetime = opts.get("dateModified", datetime.now(timezone.utc))
        self.location: str = opts.get("location", "Santa Cruz")
        self._id: Optional[str] = opts.get("_id")
        self.audio_id: Optional[str] = opts.get("audioID")
        self.audio_db_id: Optional[str] = opts.get("audio_DB_ID")
        self.user_id: Optional[str] = opts.get("userID")
        self.name: Optional[str] = opts.get("name")
        self.family_name: Optional[str] = opts.get("family_name")
        self.given_name: Optional[str] = opts.get("given_name")
        self.permissions: Optional[str] = opts.get("permissions")
        self.soloist: Optional[str] = opts.get("soloist")
        self.solo_instrument: Optional[str] = opts.get("soloInstrument")
        self.explicit_permissions: Dict[str, Any] = opts.get(
            "explicitPermissions",
            {"edit": [], "view": [], "publicView": True},
        )
        # Optional list of collection names this piece belongs to
        self.collections: List[str] = opts.get("collections", [])

        self.meters: List[Meter] = []
        for m in opts.get("meters", []):
            if isinstance(m, Meter):
                self.meters.append(m)
            else:
                self.meters.append(Meter.from_json(m))

        # Parse section starts into a local variable, then apply to phrases
        ss_grid = opts.get("sectionStartsGrid")
        if ss_grid is None:
            ss = opts.get("sectionStarts", [0])
            ss_grid = [ss]
        for _ in range(len(ss_grid), len(self.instrumentation)):
            ss_grid.append([0])
        ss_grid = [sorted(list(s)) for s in ss_grid]

        # Apply section starts to phrase-level is_section_start flags
        if ss_grid and self.phrase_grid:
            for inst_idx, phrases in enumerate(self.phrase_grid):
                if inst_idx < len(ss_grid):
                    starts = ss_grid[inst_idx]
                    for phrase_idx, phrase in enumerate(phrases):
                        if phrase.is_section_start is None:
                            phrase.is_section_start = phrase_idx in [int(s) for s in starts]
                # Ensure every phrase has a boolean is_section_start
                for phrase in phrases:
                    if phrase.is_section_start is None:
                        phrase.is_section_start = False

        sc_grid = opts.get("sectionCatGrid")
        if sc_grid is None:
            section_cat = opts.get("sectionCategorization")
            sc_grid = []
            for i, ss in enumerate(ss_grid):
                if i == 0:
                    if section_cat is not None:
                        for c in section_cat:
                            self.clean_up_section_categorization(c)
                        row = section_cat
                    else:
                        row = [init_sec_categorization() for _ in ss]
                else:
                    row = [init_sec_categorization() for _ in ss]
                sc_grid.append(row)
        self.section_cat_grid: List[List[SecCatType]] = sc_grid
        for i, ss in enumerate(ss_grid):
            while len(self.section_cat_grid) <= i:
                self.section_cat_grid.append([init_sec_categorization() for _ in ss])
            if len(self.section_cat_grid[i]) < len(ss):
                diff = len(ss) - len(self.section_cat_grid[i])
                for _ in range(diff):
                    self.section_cat_grid[i].append(init_sec_categorization())

        ad_hoc = opts.get("adHocSectionCatGrid")
        if ad_hoc is None:
            self.ad_hoc_section_cat_grid = [[[] for _ in row] for row in self.section_cat_grid]
        else:
            self.ad_hoc_section_cat_grid = [
                [ [f for f in fields if f != ""] for fields in track ]
                for track in ad_hoc
            ]
        while len(self.ad_hoc_section_cat_grid) < len(ss_grid):
            self.ad_hoc_section_cat_grid.append([[] for _ in self.ad_hoc_section_cat_grid[0]])

        self.excerpt_range = opts.get("excerptRange")
        self.assemblage_descriptors = opts.get("assemblageDescriptors", [])

        self.dur_tot: Optional[float] = opts.get("durTot")
        self.dur_array_grid: Optional[List[List[float]]] = opts.get("durArrayGrid")
        if self.dur_tot is None or self.dur_array_grid is None:
            self.dur_array_from_phrases()
        else:
            self.update_start_times()

        # Mirror the TS constructor: keep polyphonic second strings synchronized
        # (idempotent; from_json also calls this on the load path)
        self.ensure_string_synchronization()

    def _validate_parameters(self, opts: dict) -> None:
        """Validate constructor parameters and provide helpful error messages."""
        if not opts:
            return
            
        # Define allowed parameter names
        allowed_keys = {
            'raga', 'instrumentation', 'phraseGrid', 'phrases', 'title', 'dateCreated',
            'dateModified', 'location', '_id', 'audioID', 'audio_DB_ID', 'userID', 'name',
            'family_name', 'given_name', 'permissions', 'soloist', 'soloInstrument',
            'explicitPermissions', 'meters', 'sectionStartsGrid', 'sectionStarts',
            'sectionCatGrid', 'sectionCategorization', 'adHocSectionCatGrid', 'excerptRange',
            'assemblageDescriptors', 'collections', 'durTot', 'durArrayGrid', 'durArray',
            'trackTitles'
        }
        provided_keys = set(opts.keys())
        invalid_keys = provided_keys - allowed_keys
        
        # Check for invalid parameter names with helpful suggestions
        if invalid_keys:
            error_messages = []
            
            for key in invalid_keys:
                if key == 'phrases_grid' or key == 'phrase_grid':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'phraseGrid'?")
                elif key == 'audio_id':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'audioID'?")
                elif key == 'user_id':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'userID'?")
                elif key == 'duration_total' or key == 'dur_total':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'durTot'?")
                elif key == 'duration_array_grid':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'durArrayGrid'?")
                elif key == 'section_starts_grid':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'sectionStartsGrid'?")
                elif key == 'explicit_permissions':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'explicitPermissions'?")
                elif key == 'solo_instrument':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'soloInstrument'?")
                else:
                    error_messages.append(f"Invalid parameter: '{key}'")
            
            error_msg = "; ".join(error_messages)
            error_msg += f". Allowed parameters: {sorted(allowed_keys)}"
            raise ValueError(error_msg)
        
        # Validate parameter types and values
        self._validate_parameter_types(opts)
        self._validate_parameter_values(opts)
    
    def _validate_parameter_types(self, opts: dict) -> None:
        """Validate that all parameters have correct types."""
        if 'raga' in opts and opts['raga'] is not None and not isinstance(opts['raga'], (Raga, dict)):
            raise TypeError(f"Parameter 'raga' must be a Raga object or dict, got {type(opts['raga']).__name__}")
        
        if 'instrumentation' in opts:
            if not isinstance(opts['instrumentation'], list):
                raise TypeError(f"Parameter 'instrumentation' must be a list, got {type(opts['instrumentation']).__name__}")
        
        if 'phraseGrid' in opts:
            if not isinstance(opts['phraseGrid'], list):
                raise TypeError(f"Parameter 'phraseGrid' must be a list, got {type(opts['phraseGrid']).__name__}")
            if not all(isinstance(row, list) for row in opts['phraseGrid']):
                raise TypeError("All items in 'phraseGrid' must be lists")
        
        if 'phrases' in opts:
            if not isinstance(opts['phrases'], list):
                raise TypeError(f"Parameter 'phrases' must be a list, got {type(opts['phrases']).__name__}")
        
        if 'title' in opts and not isinstance(opts['title'], str):
            raise TypeError(f"Parameter 'title' must be a string, got {type(opts['title']).__name__}")
        
        if 'meters' in opts:
            if not isinstance(opts['meters'], list):
                raise TypeError(f"Parameter 'meters' must be a list, got {type(opts['meters']).__name__}")
        
        if 'sectionStartsGrid' in opts:
            if not isinstance(opts['sectionStartsGrid'], list):
                raise TypeError(f"Parameter 'sectionStartsGrid' must be a list, got {type(opts['sectionStartsGrid']).__name__}")
            if not all(isinstance(row, list) for row in opts['sectionStartsGrid']):
                raise TypeError("All items in 'sectionStartsGrid' must be lists")
            if not all(all(isinstance(item, (int, float)) for item in row) for row in opts['sectionStartsGrid']):
                raise TypeError("All items in 'sectionStartsGrid' sublists must be numbers")
        
        if 'sectionStarts' in opts:
            if not isinstance(opts['sectionStarts'], list):
                raise TypeError(f"Parameter 'sectionStarts' must be a list, got {type(opts['sectionStarts']).__name__}")
            if not all(isinstance(item, (int, float)) for item in opts['sectionStarts']):
                raise TypeError("All items in 'sectionStarts' must be numbers")
        
        if 'sectionCatGrid' in opts and opts['sectionCatGrid'] is not None:
            if not isinstance(opts['sectionCatGrid'], list):
                raise TypeError(f"Parameter 'sectionCatGrid' must be a list, got {type(opts['sectionCatGrid']).__name__}")
            if not all(isinstance(row, list) for row in opts['sectionCatGrid']):
                raise TypeError("All items in 'sectionCatGrid' must be lists")
        
        if 'explicitPermissions' in opts and not isinstance(opts['explicitPermissions'], dict):
            raise TypeError(f"Parameter 'explicitPermissions' must be a dict, got {type(opts['explicitPermissions']).__name__}")
        
        if 'durTot' in opts and opts['durTot'] is not None:
            if not isinstance(opts['durTot'], (int, float)):
                raise TypeError(f"Parameter 'durTot' must be a number, got {type(opts['durTot']).__name__}")
        
        if 'durArrayGrid' in opts and opts['durArrayGrid'] is not None:
            if not isinstance(opts['durArrayGrid'], list):
                raise TypeError(f"Parameter 'durArrayGrid' must be a list, got {type(opts['durArrayGrid']).__name__}")
            if not all(isinstance(row, list) for row in opts['durArrayGrid']):
                raise TypeError("All items in 'durArrayGrid' must be lists")
            if not all(all(isinstance(item, (int, float)) for item in row) for row in opts['durArrayGrid']):
                raise TypeError("All items in 'durArrayGrid' sublists must be numbers")
        
        # Validate string parameters
        string_params = ['location', 'audioID', 'audio_DB_ID', 'userID', 'name', 
                        'family_name', 'given_name', 'permissions', 'soloist', 'soloInstrument']
        
        for param in string_params:
            if param in opts and opts[param] is not None and not isinstance(opts[param], str):
                raise TypeError(f"Parameter '{param}' must be a string, got {type(opts[param]).__name__}")
        
        # Handle _id specially since it can be string or MongoDB-style dict
        if '_id' in opts and opts['_id'] is not None:
            if not isinstance(opts['_id'], (str, dict)):
                raise TypeError(f"Parameter '_id' must be a string or dict, got {type(opts['_id']).__name__}")
        
        # Validate list parameters
        list_params = ['assemblageDescriptors', 'collections']
        for param in list_params:
            if param in opts and opts[param] is not None and not isinstance(opts[param], list):
                raise TypeError(f"Parameter '{param}' must be a list, got {type(opts[param]).__name__}")
        
        # If collections provided, ensure it is a list of strings
        if 'collections' in opts and opts['collections'] is not None:
            if not isinstance(opts['collections'], list):
                raise TypeError(f"Parameter 'collections' must be a list, got {type(opts['collections']).__name__}")
            if not all(isinstance(item, str) for item in opts['collections']):
                raise TypeError("All items in 'collections' must be strings")

        # Validate trackTitles
        if 'trackTitles' in opts and opts['trackTitles'] is not None:
            if not isinstance(opts['trackTitles'], list):
                raise TypeError(f"Parameter 'trackTitles' must be a list, got {type(opts['trackTitles']).__name__}")
            if not all(isinstance(title, str) for title in opts['trackTitles']):
                raise TypeError("All items in 'trackTitles' must be strings")

        # Handle excerptRange specially since it can be dict or list
        if 'excerptRange' in opts and opts['excerptRange'] is not None:
            if not isinstance(opts['excerptRange'], (list, dict)):
                raise TypeError(f"Parameter 'excerptRange' must be a list or dict, got {type(opts['excerptRange']).__name__}")
    
    def _validate_parameter_values(self, opts: dict) -> None:
        """Validate that parameter values are in valid ranges."""
        if 'title' in opts and opts['title'] == "":
            raise ValueError("Parameter 'title' cannot be empty")
        
        if 'durTot' in opts and opts['durTot'] is not None:
            if opts['durTot'] < 0:
                raise ValueError(f"Parameter 'durTot' must be non-negative, got {opts['durTot']}")
        
        if 'sectionStarts' in opts:
            section_starts = opts['sectionStarts']
            if any(s < 0 for s in section_starts):
                raise ValueError("All values in 'sectionStarts' must be non-negative")
        
        if 'sectionStartsGrid' in opts:
            for i, row in enumerate(opts['sectionStartsGrid']):
                if any(s < 0 for s in row):
                    raise ValueError(f"All values in 'sectionStartsGrid[{i}]' must be non-negative")
        
        if 'durArrayGrid' in opts and opts['durArrayGrid'] is not None:
            for i, row in enumerate(opts['durArrayGrid']):
                if any(d < 0 for d in row):
                    raise ValueError(f"All values in 'durArrayGrid[{i}]' must be non-negative")
                if len(row) > 0 and sum(row) == 0:
                    raise ValueError(f"'durArrayGrid[{i}]' cannot have all zero values")
        
        # Validate instrumentation contains valid instruments
        if 'instrumentation' in opts:
            for i, inst in enumerate(opts['instrumentation']):
                if not isinstance(inst, Instrument) and not isinstance(inst, str):
                    raise TypeError(f"instrumentation[{i}] must be an Instrument enum or string")
                if isinstance(inst, str):
                    try:
                        Instrument(inst)
                    except ValueError:
                        warnings.warn(f"Unknown instrument name: '{inst}'. This may cause issues with instrument-specific features.", UserWarning)
        
        # Validate grid structure consistency
        if 'phraseGrid' in opts and 'instrumentation' in opts:
            phrase_grid = opts['phraseGrid']
            instrumentation = opts['instrumentation']
            if len(phrase_grid) != len(instrumentation):
                warnings.warn(f"phraseGrid has {len(phrase_grid)} tracks but instrumentation has {len(instrumentation)} instruments. "
                             "These should typically match.", UserWarning)
        
        # Validate explicit permissions structure
        if 'explicitPermissions' in opts:
            perms = opts['explicitPermissions']
            required_keys = {'edit', 'view', 'publicView'}
            if not required_keys.issubset(perms.keys()):
                missing = required_keys - perms.keys()
                raise ValueError(f"explicitPermissions missing required keys: {sorted(missing)}")
            
            if not isinstance(perms.get('edit'), list):
                raise TypeError("explicitPermissions['edit'] must be a list")
            if not isinstance(perms.get('view'), list):
                raise TypeError("explicitPermissions['view'] must be a list")
            if not isinstance(perms.get('publicView'), bool):
                raise TypeError("explicitPermissions['publicView'] must be a boolean")

    # ------------------------------------------------------------------
    @property
    def phrases(self) -> List[Phrase]:
        return self.phrase_grid[0]

    @phrases.setter
    def phrases(self, arr: List[Phrase]) -> None:
        self.phrase_grid[0] = arr

    @property
    def dur_array(self) -> List[float]:
        return self.dur_array_grid[0] if self.dur_array_grid else []

    @dur_array.setter
    def dur_array(self, arr: List[float]) -> None:
        if self.dur_array_grid is None:
            self.dur_array_grid = [arr]
        else:
            self.dur_array_grid[0] = arr

    @property
    def section_starts_grid(self) -> List[List[int]]:
        """Compute section starts from phrase-level is_section_start flags."""
        return [[idx for idx, p in enumerate(phrases) if p.is_section_start]
                for phrases in self.phrase_grid]

    @section_starts_grid.setter
    def section_starts_grid(self, value: List[List[int]]) -> None:
        """Apply section starts to phrase-level is_section_start flags."""
        for inst_idx, starts in enumerate(value):
            if inst_idx < len(self.phrase_grid):
                for p_idx, phrase in enumerate(self.phrase_grid[inst_idx]):
                    phrase.is_section_start = p_idx in [int(s) for s in starts]

    @property
    def section_starts(self) -> List[int]:
        return self.section_starts_grid[0]

    @section_starts.setter
    def section_starts(self, arr: List[int]) -> None:
        self.section_starts_grid = [arr] + self.section_starts_grid[1:]

    @property
    def section_categorization(self) -> List[SecCatType]:
        return self.section_cat_grid[0]

    @section_categorization.setter
    def section_categorization(self, arr: List[SecCatType]) -> None:
        self.section_cat_grid[0] = arr

    @property
    def assemblages(self) -> List["Assemblage"]:
        from .assemblage import Assemblage
        flat_phrases = [p for row in self.phrase_grid for p in row]
        return [Assemblage.from_descriptor(d, flat_phrases) for d in self.assemblage_descriptors]

    # ------------------------------------------------------------------
    def update_start_times(self) -> None:
        if not self.dur_array_grid or self.dur_tot is None:
            return
        for track, phrases in enumerate(self.phrase_grid):
            starts = [s * self.dur_tot for s in get_starts(self.dur_array_grid[track])]
            for p, st in zip(phrases, starts):
                p.start_time = st
                p.piece_idx = phrases.index(p)

    # ------------------------------------------------------------------
    def dur_tot_from_phrases(self) -> None:
        """Set ``durTot`` from contained phrases and pad shorter tracks."""
        totals = [sum(p.dur_tot for p in row) for row in self.phrase_grid]
        max_dur = max(totals) if totals else 0.0
        self.dur_tot = max_dur
        for i, total in enumerate(totals):
            if total != max_dur:
                extra = max_dur - total
                phrases = self.phrase_grid[i]
                silent = Trajectory({"id": 12, "dur_tot": extra})
                if phrases:
                    phrases[-1].trajectory_grid[0].append(silent)
                    phrases[-1].reset()
                else:
                    p = Phrase({"trajectories": [silent], "dur_tot": extra, "raga": self.raga})
                    phrases.append(p)
                    p.reset()

    def dur_array_from_phrases(self) -> None:
        """Recompute ``durArrayGrid`` removing NaN trajectories."""
        self.dur_tot_from_phrases()
        self.dur_array_grid = []
        if self.dur_tot == 0:
            self.dur_array_grid.append([])
            return
        for row in self.phrase_grid:
            arr = []
            for p in row:
                if p.dur_tot is None:
                    raise Exception("p.durTot is undefined")
                if math.isnan(p.dur_tot):
                    p.trajectory_grid[0] = [t for t in p.trajectories if not math.isnan(t.dur_tot)]
                    p.dur_tot_from_trajectories()
                arr.append(p.dur_tot / self.dur_tot)
            self.dur_array_grid.append(arr)
        self.update_start_times()

    # ------------------------------------------------------------------
    def set_dur_tot(self, dur_tot: float) -> None:
        if self.dur_tot is None:
            self.dur_tot_from_phrases()
        if self.dur_tot is None:
            self.dur_tot = dur_tot
            return
        if dur_tot < self.dur_tot:
            raise ValueError("cannot shorten duration")
        extra = dur_tot - self.dur_tot
        if extra > 0 and self.phrases:
            last_phrase = self.phrases[-1]
            if last_phrase.trajectories and last_phrase.trajectories[-1].id == 12:
                last_phrase.trajectories[-1].dur_tot += extra
            else:
                silent = Trajectory({"id": 12, "dur_tot": extra})
                last_phrase.trajectory_grid[0].append(silent)
            last_phrase.reset()
        self.dur_tot = dur_tot
        self.dur_array_from_phrases()

    def fill_remaining_duration(self, target_duration: float, track: int = 0) -> None:
        """Add a silent trajectory to fill the remaining duration to reach target_duration.
        
        Args:
            target_duration: The desired total duration for the piece
            track: Which instrument track to add the silence to (default: 0)
        """
        if self.dur_tot is None:
            self.dur_tot_from_phrases()
        
        current_duration = sum(p.dur_tot for p in self.phrase_grid[track])
        
        if target_duration <= current_duration:
            return  # Already at or exceeding target duration
            
        remaining = target_duration - current_duration
        
        # Create silent trajectory to fill the remaining duration
        silent_traj = Trajectory({
            'id': 12,
            'dur_tot': remaining,
            'fundID12': self.raga.fundamental
        })
        
        # Add as a new phrase or append to existing phrase
        if self.phrase_grid[track]:
            # Add to last phrase
            last_phrase = self.phrase_grid[track][-1]
            last_phrase.trajectory_grid[0].append(silent_traj)
            last_phrase.reset()
        else:
            # Create new phrase with silent trajectory
            silent_phrase = Phrase({
                'trajectories': [silent_traj],
                'dur_tot': remaining,
                'raga': self.raga
            })
            self.phrase_grid[track].append(silent_phrase)
        
        # Update piece-level durations
        self.dur_array_from_phrases()

    def add_trajectory(
        self,
        trajectory_data: Union[Trajectory, Dict[str, Any]], 
        inst_track: int,
        start_time: float
    ) -> bool:
        """Add a trajectory to the piece by replacing part of a silent trajectory.
        
        Args:
            trajectory_data: Either a Trajectory object or dict with trajectory parameters
            inst_track: Index of the instrument track to add the trajectory to
            start_time: Start time in the piece (in seconds)
            
        Returns:
            bool: True if trajectory was successfully added, False otherwise
        """
        # Validate inputs
        if not self._validate_trajectory_addition_inputs(trajectory_data, inst_track, start_time):
            return False
        
        # Prepare trajectory with piece context
        trajectory = self._prepare_trajectory_for_addition(trajectory_data, inst_track)
        
        # Validate all requirements
        if not self._validate_trajectory_addition_requirements(trajectory, inst_track, start_time):
            return False
        
        # Find target phrase and silent trajectory
        phrase, silent_traj_idx = self._find_target_location(inst_track, start_time)
        if phrase is None or silent_traj_idx is None:
            return False
        
        # Apply replacement logic
        self._replace_silent_trajectory(phrase, silent_traj_idx, trajectory, start_time)
        
        # Update piece state
        phrase.reset()
        return True

    def _validate_trajectory_addition_inputs(
        self, 
        trajectory_data: Union[Trajectory, Dict[str, Any]], 
        inst_track: int, 
        start_time: float
    ) -> bool:
        """Validate basic inputs for trajectory addition."""
        if not isinstance(inst_track, int) or inst_track < 0:
            return False
        if not isinstance(start_time, (int, float)) or start_time < 0:
            return False
        if inst_track >= len(self.phrase_grid):
            return False
        return True

    def _prepare_trajectory_for_addition(
        self, 
        trajectory_data: Union[Trajectory, Dict[str, Any]], 
        inst_track: int
    ) -> Trajectory:
        """Convert input to Trajectory object with piece context."""
        if isinstance(trajectory_data, Trajectory):
            # Even for existing Trajectory objects, recreate with piece's raga/context
            traj_dict = trajectory_data.to_json()
        else:
            traj_dict = dict(trajectory_data)
        
        # Ensure trajectory has piece context
        traj_dict['instrumentation'] = self.instrumentation[inst_track]
        
        # Update pitches to use piece's raga fundamental  
        if 'pitches' in traj_dict:
            updated_pitches = []
            for pitch_data in traj_dict['pitches']:
                if isinstance(pitch_data, dict):
                    pitch_data['fundamental'] = self.raga.fundamental
                    pitch_data['raga'] = self.raga.to_json()
                    updated_pitches.append(Pitch(pitch_data))
                else:
                    updated_pitches.append(pitch_data)
            traj_dict['pitches'] = updated_pitches
        
        # Set fundID12 for silent trajectories
        if traj_dict.get('id') == 12:
            traj_dict['fundID12'] = self.raga.fundamental
        
        trajectory = Trajectory(traj_dict)
        return trajectory

    def _validate_trajectory_addition_requirements(
        self, 
        trajectory: Trajectory, 
        inst_track: int, 
        start_time: float
    ) -> bool:
        """Validate all 5 requirements for trajectory addition."""
        end_time = start_time + trajectory.dur_tot
        
        # 1. Time range must be within piece duration
        if self.dur_tot is None:
            self.dur_tot_from_phrases()
        if end_time > self.dur_tot:
            return False
        
        # 2. Track must be valid (already validated in inputs)
        
        # 3. Time range must be within a single phrase
        phrase_idx = self.phrase_idx_from_time(start_time, inst_track)
        phrase = self.phrase_grid[inst_track][phrase_idx]
        phrase_start = phrase.start_time or 0
        phrase_end = phrase_start + phrase.dur_tot
        
        if end_time > phrase_end:
            return False
        
        # 4. Time range must be within a single trajectory
        target_traj = self.traj_from_time(start_time, inst_track)
        if target_traj is None:
            return False
        
        # Find trajectory's absolute start time
        traj_abs_start = phrase_start + (target_traj.start_time or 0)
        traj_abs_end = traj_abs_start + target_traj.dur_tot
        
        if start_time < traj_abs_start or end_time > traj_abs_end:
            return False
        
        # 5. Target trajectory must be silent (id == 12)
        if target_traj.id != 12:
            return False
        
        return True

    def _find_target_location(self, inst_track: int, start_time: float) -> tuple[Optional[Phrase], Optional[int]]:
        """Find the phrase and trajectory index for the target location."""
        phrase_idx = self.phrase_idx_from_time(start_time, inst_track)
        phrase = self.phrase_grid[inst_track][phrase_idx]
        
        # Find the trajectory index within the phrase
        phrase_start = phrase.start_time or 0
        relative_start_time = start_time - phrase_start
        
        current_time = 0.0
        for i, traj in enumerate(phrase.trajectories):
            traj_start = current_time
            traj_end = current_time + traj.dur_tot
            
            if traj_start <= relative_start_time < traj_end:
                return phrase, i
            
            current_time = traj_end
        
        return None, None

    def _replace_silent_trajectory(
        self, 
        phrase: Phrase, 
        silent_traj_idx: int, 
        new_trajectory: Trajectory, 
        start_time: float
    ) -> None:
        """Replace part or all of a silent trajectory with the new trajectory."""
        silent_traj = phrase.trajectories[silent_traj_idx]
        phrase_start = phrase.start_time or 0
        silent_abs_start = phrase_start + (silent_traj.start_time or 0)
        
        # Calculate relative times within the silent trajectory
        rel_start = start_time - silent_abs_start
        rel_end = rel_start + new_trajectory.dur_tot
        
        trajs = phrase.trajectories
        
        # Determine which case we're in (from NewTrajEmit logic)
        starts_equal = abs(rel_start) < 1e-10  # essentially zero
        ends_equal = abs(rel_end - silent_traj.dur_tot) < 1e-10
        
        if starts_equal and ends_equal:
            # Case 1: Replace entire silent trajectory
            trajs[silent_traj_idx] = new_trajectory
        elif starts_equal:
            # Case 2: Replace left side of silent trajectory
            silent_traj.dur_tot = silent_traj.dur_tot - new_trajectory.dur_tot
            trajs.insert(silent_traj_idx, new_trajectory)
        elif ends_equal:
            # Case 3: Replace right side of silent trajectory
            silent_traj.dur_tot = silent_traj.dur_tot - new_trajectory.dur_tot
            trajs.insert(silent_traj_idx + 1, new_trajectory)
        else:
            # Case 4: Replace internal portion of silent trajectory
            first_dur = rel_start
            last_dur = silent_traj.dur_tot - rel_end
            
            # Shorten the original silent trajectory to the first part
            silent_traj.dur_tot = first_dur
            
            # Create new silent trajectory for the remaining part
            last_silent_traj = Trajectory({
                'id': 12,
                'pitches': [],
                'dur_tot': last_dur,
                'fundID12': self.raga.fundamental,
                'instrumentation': self.instrumentation[self.phrase_grid.index(next(row for row in self.phrase_grid if phrase in row))]
            })
            
            # Insert new trajectory and remaining silent trajectory
            trajs.insert(silent_traj_idx + 1, new_trajectory)
            trajs.insert(silent_traj_idx + 2, last_silent_traj)

    def realign_pitches(self) -> None:
        for phrases in self.phrase_grid:
            for p in phrases:
                p.realign_pitches()

    def update_fundamental(self, fundamental: float) -> None:
        self.raga.fundamental = fundamental
        for phrases in self.phrase_grid:
            for p in phrases:
                p.update_fundamental(fundamental)

    def put_raga_in_phrase(self) -> None:
        for phrases in self.phrase_grid:
            for p in phrases:
                p.raga = self.raga

    # ------------------------------------------------------------------
    @property
    def sections_grid(self) -> List[List["Section"]]:
        from .section import Section
        grid: List[List["Section"]] = []
        for i, starts in enumerate(self.section_starts_grid):
            sections: List["Section"] = []
            for j, s in enumerate(starts):
                if j == len(starts) - 1:
                    slice_phrases = self.phrase_grid[i][s:]
                else:
                    slice_phrases = self.phrase_grid[i][s:starts[j + 1]]
                sections.append(
                    Section(
                        {
                            "phrases": slice_phrases,
                            "categorization": self.section_cat_grid[i][j],
                            "ad_hoc_categorization": self.ad_hoc_section_cat_grid[i][j],
                        }
                    )
                )
            grid.append(sections)
        return grid

    @property
    def sections(self) -> List["Section"]:
        return self.sections_grid[0]

    def add_meter(self, meter: Meter) -> None:
        for m in self.meters:
            dur_m = m.cycle_dur
            dur_new = meter.cycle_dur
            c1 = m.start_time <= meter.start_time < m.start_time + dur_m
            c2 = m.start_time < meter.start_time + dur_new <= m.start_time + dur_m
            c3 = meter.start_time <= m.start_time and meter.start_time + dur_new >= m.start_time + dur_m
            if c1 or c2 or c3:
                raise ValueError("meters overlap")
        self.meters.append(meter)

    def remove_meter(self, meter: Meter) -> None:
        if meter in self.meters:
            self.meters.remove(meter)

    # ------------------------------------------------------------------
    def all_trajectories(self, inst: int = 0, string_idx: int = 0) -> List[Trajectory]:
        """Get all trajectories for a given instrument track and string index.

        Args:
            inst: Instrument track index (default 0).
            string_idx: String index within the instrument (default 0).
                For Sitar/Sarangi, string 0 is main, string 1 is jor/second.
        """
        trajs: List[Trajectory] = []
        for p in self.phrase_grid[inst]:
            if string_idx < len(p.trajectory_grid):
                trajs.extend(p.trajectory_grid[string_idx])
        return trajs

    # ------------------------------------------------------------------
    def track_from_traj(self, traj: Trajectory) -> int:
        for i in range(len(self.instrumentation)):
            if traj in self.all_trajectories(i):
                return i
        raise ValueError("Trajectory not found")

    def track_from_traj_uid(self, traj_uid: str) -> int:
        for i in range(len(self.instrumentation)):
            for t in self.all_trajectories(i):
                if t.unique_id == traj_uid:
                    return i
        raise ValueError("Trajectory not found")

    def phrase_from_uid(self, uid: str) -> Phrase:
        for track in self.phrase_grid:
            for p in track:
                if p.unique_id == uid:
                    return p
        raise ValueError("Phrase not found")

    def track_from_phrase_uid(self, uid: str) -> int:
        for i, track in enumerate(self.phrase_grid):
            if any(p.unique_id == uid for p in track):
                return i
        raise ValueError("Phrase not found")

    def string_from_traj(self, traj: Trajectory) -> int:
        """Determine which string index contains a given trajectory.

        Searches all phrases across all strings by unique_id.
        Returns the string index (0 or 1). Raises ValueError if not found.
        """
        for phrases in self.phrase_grid:
            for phrase in phrases:
                for string_idx, string_trajs in enumerate(phrase.trajectory_grid):
                    for t in string_trajs:
                        if t.unique_id == traj.unique_id:
                            return string_idx
        raise ValueError("Trajectory not found in any string")

    def traj_from_uid(self, uid: str, track: int = 0) -> Trajectory:
        for t in self.all_trajectories(track):
            if t.unique_id == uid:
                return t
        raise ValueError("Trajectory not found")

    def traj_from_time(self, time: float, track: int = 0) -> Optional[Trajectory]:
        trajs = self.all_trajectories(track)
        if not trajs:
            return None
        starts = self.traj_start_times(track)
        end_times = [s + t.dur_tot for s, t in zip(starts, trajs)]
        idx = -1
        for i, s in enumerate(starts):
            if time >= s:
                idx = i
        if idx == -1:
            return trajs[0]
        if time < end_times[idx]:
            return trajs[idx]
        if idx + 1 < len(trajs):
            return trajs[idx + 1]
        return None

    def phrase_from_time(self, time: float, track: int = 0) -> Phrase:
        starts = self.dur_starts(track)
        idx = 0
        for i, s in enumerate(starts):
            if time >= s:
                idx = i
        return self.phrase_grid[track][idx]

    def phrase_idx_from_time(self, time: float, track: int = 0) -> int:
        starts = self.dur_starts(track)
        idx = 0
        for i, s in enumerate(starts):
            if time >= s:
                idx = i
        return idx

    def all_groups(self, instrument_idx: int = 0) -> List["Group"]:
        groups: List["Group"] = []
        for p in self.phrase_grid[instrument_idx]:
            for g_list in p.groups_grid:
                groups.extend(g_list)
        return groups

    def p_idx_from_group(self, g: "Group") -> int:
        for i, p in enumerate(self.phrase_grid[0]):
            for group_list in p.groups_grid:
                if g in group_list:
                    return i
        return -1

    def all_display_vowels(self, inst: int = 0) -> List[Any]:
        vocal = [Instrument.Vocal_M, Instrument.Vocal_F]
        if self.instrumentation[inst] not in vocal:
            raise Exception("instrumentation is not vocal")
        # placeholder minimal return
        return []

    def s_idx_from_p_idx(self, p_idx: int, inst: int = 0) -> int:
        ss = self.section_starts_grid[inst]
        s_idx = len(ss) - 1
        for i, s in enumerate(ss):
            if p_idx < s:
                s_idx = i - 1
                break
        if s_idx < 0:
            s_idx = 0
        return s_idx

    def durations_of_fixed_pitches(
        self, inst: int = 0, output_type: str = "pitchNumber"
    ) -> Dict:
        trajs = self.all_trajectories(inst)
        return durations_of_fixed_pitches(trajs, output_type)

    def proportions_of_fixed_pitches(
        self, inst: int = 0, output_type: str = "pitchNumber"
    ) -> Dict:
        return durations_of_fixed_pitches(trajs=self.all_trajectories(inst), output_type=output_type, count_type="proportional")

    # ------------------------------------------------------------------
    def ensure_string_synchronization(self) -> None:
        """Mirror of the TS Piece.ensureStringSynchronization (STRING-SYNC).

        For Sitar/Sarangi tracks, guarantee a second string (trajectory_grid[1])
        exists holding a single silent (id-12) trajectory spanning the phrase whenever
        it has no non-silent content — keeping the polyphonic 2nd string temporally
        aligned. This makes a Python load->save round-trip structurally identical to
        the TS client. Called from from_json (the load path).

        Idempotent by design: an existing silent trajectory is PRESERVED (its uniqueId
        kept), and a fresh one is synthesized only when the second string is empty — so
        to_json -> from_json -> to_json is byte-stable.

        Second strings holding non-silent content are reconciled against the phrase
        duration (the main string is authoritative — TS parity, idtap#34): trailing
        silence is trimmed, shortfalls are padded with silence, and sounding overhang
        is warned about but never destroyed. The reconciliation is deterministic, so
        round-trips remain stable after the first normalization.
        """
        for track_idx, track_phrases in enumerate(self.phrase_grid):
            if track_idx >= len(self.instrumentation):
                continue
            instrument = self.instrumentation[track_idx]
            inst_val = getattr(instrument, 'value', instrument)
            if inst_val not in (Instrument.Sitar.value, Instrument.Sarangi.value):
                continue
            for phrase in track_phrases:
                while len(phrase.trajectory_grid) < 2:
                    phrase.trajectory_grid.append([])
                second = phrase.trajectory_grid[1]
                has_non_silent = any(t.id != 12 for t in second)
                if second and has_non_silent:
                    # Reconcile a drifted second string with the phrase duration.
                    # Historic phrase divisions could leave string-2 trajectories
                    # spanning the pre-division range, stretching the timeline
                    # against the audio (idtap#34).
                    eps = 1e-6
                    target = phrase.dur_tot or 0
                    total = sum(t.dur_tot for t in second)
                    if total > target + eps:
                        # Trim trailing silence; never cut sounding trajectories
                        i = len(second) - 1
                        while i >= 0 and total > target + eps and second[i].id == 12:
                            cut = min(total - target, second[i].dur_tot)
                            second[i].dur_tot -= cut
                            total -= cut
                            if second[i].dur_tot < eps:
                                second.pop(i)
                            i -= 1
                        if total > target + eps:
                                warnings.warn(
                                f"ensure_string_synchronization: track {track_idx}, "
                                f"phrase {phrase.piece_idx}: second-string trajectories "
                                f"span {total}s but the phrase is {target}s; sounding "
                                f"content extends past the phrase and was left "
                                f"untouched.",
                                UserWarning,
                            )
                        phrase.reset()
                    elif total < target - eps:
                        second.append(Trajectory({
                            'id': 12,
                            'dur_tot': target - total,
                            'fund_id12': self.raga.fundamental,
                            'start_time': total,
                        }))
                        phrase.reset()
                    continue
                # Empty or all-silent: collapse to a single silent trajectory. Reuse the
                # first existing silent traj (preserve its uniqueId) for idempotence.
                if second:
                    keep = second[0]
                    keep.dur_tot = phrase.dur_tot
                    keep.start_time = 0
                    keep.fund_id12 = self.raga.fundamental
                else:
                    keep = Trajectory({
                        'id': 12,
                        'dur_tot': phrase.dur_tot,
                        'fund_id12': self.raga.fundamental,
                        'start_time': 0,
                    })
                phrase.trajectory_grid[1] = [keep]
                phrase.reset()

    def chikari_freqs(self, inst_idx: int = 0) -> List[float]:
        """Return 4 chikari frequencies derived from the raga.

        Returns 0.0 for strings that are silent (None pitch).
        """
        return [p.frequency if p is not None else 0.0 for p in self.raga.chikari_pitches]

    # ------------------------------------------------------------------
    def dur_starts(self, track: int = 0) -> List[float]:
        if self.dur_array_grid is None:
            raise Exception("durArray is undefined")
        if self.dur_tot is None:
            raise Exception("durTot is undefined")
        return get_starts([d * self.dur_tot for d in self.dur_array_grid[track]])

    def traj_start_times(self, inst: int = 0, string_idx: int = 0) -> List[float]:
        """Get start times for all trajectories in a given string.

        For string 0: cumulative duration (standard sequential timing).
        For string > 0: phrase-boundary based (phrase.start_time + traj.start_time).
        """
        if string_idx == 0:
            trajs = self.all_trajectories(inst, 0)
            times = [0.0]
            for t in trajs[:-1]:
                times.append(times[-1] + t.dur_tot)
            return times
        else:
            times: List[float] = []
            for p in self.phrase_grid[inst]:
                phrase_start = p.start_time or 0.0
                if string_idx < len(p.trajectory_grid):
                    for traj in p.trajectory_grid[string_idx]:
                        times.append(phrase_start + (traj.start_time or 0.0))
            return times

    def all_pitches(self, repetition: bool = True, pitch_number: bool = False, track: int = 0) -> List[Any]:
        pitches: List[Any] = []
        for p in self.phrase_grid[track]:
            pitches.extend(p.all_pitches(True))
        if not repetition:
            out: List[Any] = []
            for i, pitch in enumerate(pitches):
                if isinstance(pitch, (int, float)):
                    raise ValueError("pitch is a number")
                if i == 0:
                    out.append(pitch)
                else:
                    prev = out[-1]
                    if isinstance(prev, (int, float)):
                        raise ValueError("lastP is a number")
                    if not (pitch.swara == prev.swara and pitch.oct == prev.oct and pitch.raised == prev.raised):
                        out.append(pitch)
            pitches = out
        if pitch_number:
            nums: List[Any] = []
            for p in pitches:
                if isinstance(p, (int, float)):
                    nums.append(p)
                else:
                    nums.append(p.numbered_pitch)
            return nums
        return pitches

    @property
    def highest_pitch_number(self) -> float:
        return max(self.all_pitches(pitch_number=True))

    @property
    def lowest_pitch_number(self) -> float:
        return min(self.all_pitches(pitch_number=True))

    def most_recent_traj(self, time: float, inst: int = 0) -> Trajectory:
        trajs = self.all_trajectories(inst)
        end_times: List[float] = []
        for t in trajs:
            phrase = next((p for p in self.phrase_grid[inst] if t in p.trajectories), None)
            if phrase is None:
                continue
            end_times.append((phrase.start_time or 0) + (t.start_time or 0) + t.dur_tot)
        latest = max([et for et in end_times if et <= time], default=-float("inf"))
        idx = end_times.index(latest)
        return trajs[idx]

    # ------------------------------------------------------------------
    def chunked_trajs(self, inst: int = 0, duration: float = 30) -> List[List[Trajectory]]:
        trajs = self.all_trajectories(inst)
        durs = [t.dur_tot for t in trajs]
        starts = get_starts(durs)
        ends = get_ends(durs)
        chunks: List[List[Trajectory]] = []
        if self.dur_tot is None:
            self.dur_tot_from_phrases()
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            def f1(st: float) -> bool:
                return st >= i and st < i + duration

            def f2(et: float) -> bool:
                return et > i and et <= i + duration

            def f3(st: float, et: float) -> bool:
                return st < i and et > i + duration

            chunk = [trajs[j] for j in range(len(trajs)) if f1(starts[j]) or f2(ends[j]) or f3(starts[j], ends[j])]
            chunks.append(chunk)
            i += duration
        return chunks

    def all_display_bols(self, inst: int = 0) -> List[Dict[str, Any]]:
        trajs = self.all_trajectories(inst)
        starts = self.traj_start_times(inst)
        idxs: List[int] = []
        bol_trajs: List[Trajectory] = []
        for idx, t in enumerate(trajs):
            art = t.articulations.get("0.00")
            if art and art.name == "pluck":
                idxs.append(idx)
                bol_trajs.append(t)
        bols = []
        for i, t in enumerate(bol_trajs):
            art = t.articulations["0.00"]
            bols.append({
                "time": starts[idxs[i]],
                "bol": getattr(art, "stroke_nickname", None),
                "uId": t.unique_id,
                "logFreq": t.log_freqs[0],
                "track": inst,
            })
        return bols

    def all_display_sargam(self, inst: int = 0) -> List[Dict[str, Any]]:
        trajs = self.all_trajectories(inst)
        starts = self.traj_start_times(inst)
        sargams: List[Dict[str, Any]] = []
        last_pitch: Dict[str, Optional[float]] = {"logFreq": None, "time": None}
        for i, t in enumerate(trajs):
            if t.id != 12:
                sub_durs = [d * t.dur_tot for d in (t.dur_array or [])]
                time_pts = get_starts(sub_durs)
                time_pts.append(t.dur_tot)
                time_pts = [tp + starts[i] for tp in time_pts]
                for tp_idx, tp in enumerate(time_pts):
                    log_freq = t.log_freqs[tp_idx] if tp_idx < len(t.log_freqs) else t.log_freqs[tp_idx - 1]
                    c_lf = last_pitch["logFreq"] == log_freq
                    c_t = last_pitch["time"] == tp
                    if not (c_lf or (c_lf and c_t)):
                        pitch = t.pitches[min(tp_idx, len(t.pitches)-1)]
                        sargams.append({
                            "logFreq": log_freq,
                            "sargam": pitch.sargam_letter,
                            "time": tp,
                            "uId": t.unique_id,
                            "track": inst,
                            "solfege": pitch.solfege_letter,
                            "pitchClass": str(pitch.chroma),
                            "westernPitch": pitch.western_pitch,
                        })
                    last_pitch = {"logFreq": log_freq, "time": tp}
        phrase_divs = [p.start_time + p.dur_tot for p in self.phrase_grid[inst]]
        pwr = 10 ** 5
        rounded_pds = [round(pd * pwr) / pwr for pd in phrase_divs]
        for s_idx, s in enumerate(sargams):
            pos = 1
            last_higher = True
            next_higher = True
            if s_idx != 0 and s_idx != len(sargams) - 1:
                last_s = sargams[s_idx - 1]
                next_s = sargams[s_idx + 1]
                last_higher = last_s["logFreq"] > s["logFreq"]
                next_higher = next_s["logFreq"] > s["logFreq"]
            if last_higher and next_higher:
                pos = 0
            elif not last_higher and not next_higher:
                pos = 1
            elif last_higher and not next_higher:
                pos = 3
            elif not last_higher and next_higher:
                pos = 2
            if round(s["time"] * pwr) / pwr in rounded_pds:
                pos = 5 if next_higher else 4
            s["pos"] = pos
        return sargams

    def all_phrase_divs(self, inst: int = 0) -> List[Dict[str, Any]]:
        objs: List[Dict[str, Any]] = []
        for p_idx, p in enumerate(self.phrase_grid[inst]):
            if p_idx != 0:
                objs.append({
                    "time": p.start_time,
                    "type": "section" if p_idx in self.section_starts_grid[inst] else "phrase",
                    "idx": p_idx,
                    "track": inst,
                    "uId": p.unique_id,
                })
        return objs

    def all_display_vowels(self, inst: int = 0) -> List[Dict[str, Any]]:
        vocal = [Instrument.Vocal_M, Instrument.Vocal_F]
        if self.instrumentation[inst] not in vocal:
            raise Exception("instrumentation is not vocal")
        display_vowels: List[Dict[str, Any]] = []
        for phrase in self.phrase_grid[inst]:
            first_idxs = phrase.first_traj_idxs()
            phrase_start = phrase.start_time or 0
            for t_idx in first_idxs:
                traj = phrase.trajectories[t_idx]
                time = phrase_start + (traj.start_time or 0)
                log_freq = traj.log_freqs[0]
                with_c = traj.start_consonant is not None
                art = traj.articulations.get("0.00") if with_c else None
                ipa_text = (art.ipa if art else "") + (traj.vowel_ipa or "")
                devanagari_text = (art.hindi if art else "") + (traj.vowel_hindi or "")
                english_text = (art.eng_trans if art else "") + (traj.vowel_eng_trans or "")
                display_vowels.append({
                    "time": time,
                    "logFreq": log_freq,
                    "ipaText": ipa_text,
                    "devanagariText": devanagari_text,
                    "englishText": english_text,
                    "uId": traj.unique_id,
                })
        return display_vowels

    def all_display_ending_consonants(self, inst: int = 0) -> List[Dict[str, Any]]:
        display: List[Dict[str, Any]] = []
        trajs = self.all_trajectories(inst)
        for t in trajs:
            if t.end_consonant is not None:
                phrase = next((p for p in self.phrase_grid[inst] if t in p.trajectories), None)
                phrase_start = phrase.start_time if phrase else 0
                time = phrase_start + (t.start_time or 0) + t.dur_tot
                log_freq = t.log_freqs[-1]
                art = t.articulations.get("1.00")
                display.append({
                    "time": time,
                    "logFreq": log_freq,
                    "ipaText": art.ipa if art else None,
                    "devanagariText": art.hindi if art else None,
                    "englishText": art.eng_trans if art else None,
                    "uId": t.unique_id,
                })
        return display

    def all_display_consonant_symbols(self, inst: int = 0) -> List[Dict[str, Any]]:
        """Consonant articulation points, as the web app's transcription
        renderer marks them with diamonds (renderConsonantSymbols() in
        TranscriptionLayer.vue).

        A trajectory whose articulation at '0.00' is a consonant yields a
        symbol at its start point on the curve; one at '1.00' yields a symbol
        at its end point. ``position`` is ``'start'`` or ``'end'``; ``time``
        and ``logFreq`` locate the symbol; the text fields carry the
        consonant in each script.
        """
        display: List[Dict[str, Any]] = []
        for phrase in self.phrase_grid[inst]:
            phrase_start = phrase.start_time or 0
            for t in phrase.trajectories:
                for key, norm_x, position in (
                    ("0.00", 0.0, "start"),
                    ("1.00", 1.0, "end"),
                ):
                    art = t.articulations.get(key)
                    if art is None or art.name != "consonant":
                        continue
                    display.append({
                        "time": (phrase_start + (t.start_time or 0)
                                 + norm_x * t.dur_tot),
                        "logFreq": t.compute(norm_x, log_scale=True),
                        "position": position,
                        "ipaText": art.ipa,
                        "devanagariText": art.hindi,
                        "englishText": art.eng_trans,
                        "uId": t.unique_id,
                    })
        return display

    def all_display_chikaris(self, inst: int = 0) -> List[Dict[str, Any]]:
        display: List[Dict[str, Any]] = []
        for p in self.phrase_grid[inst]:
            for k, chikari in p.chikaris.items():
                time = p.start_time + float(k)
                display.append({
                    "time": time,
                    "phraseTimeKey": k,
                    "phraseIdx": p.piece_idx,
                    "track": inst,
                    "chikari": chikari,
                    "uId": chikari.unique_id,
                })
        return display

    def chunked_display_chikaris(self, inst: int = 0, duration: float = 30) -> List[List[Dict[str, Any]]]:
        display = self.all_display_chikaris(inst)
        chunks: List[List[Dict[str, Any]]] = []
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            chunk = [c for c in display if c["time"] >= i and c["time"] < i + duration]
            chunks.append(chunk)
            i += duration
        return chunks

    def chunked_display_consonants(self, inst: int = 0, duration: float = 30) -> List[List[Dict[str, Any]]]:
        display = self.all_display_ending_consonants(inst)
        chunks: List[List[Dict[str, Any]]] = []
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            chunk = [c for c in display if c["time"] >= i and c["time"] < i + duration]
            chunks.append(chunk)
            i += duration
        return chunks

    def chunked_display_vowels(self, inst: int = 0, duration: float = 30) -> List[List[Dict[str, Any]]]:
        display = self.all_display_vowels(inst)
        chunks: List[List[Dict[str, Any]]] = []
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            chunk = [v for v in display if v["time"] >= i and v["time"] < i + duration]
            chunks.append(chunk)
            i += duration
        return chunks

    def chunked_display_sargam(self, inst: int = 0, duration: float = 30) -> List[List[Dict[str, Any]]]:
        display = self.all_display_sargam(inst)
        chunks: List[List[Dict[str, Any]]] = []
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            chunk = [s for s in display if s["time"] >= i and s["time"] < i + duration]
            chunks.append(chunk)
            i += duration
        return chunks

    def chunked_display_bols(self, inst: int = 0, duration: float = 30) -> List[List[Dict[str, Any]]]:
        display = self.all_display_bols(inst)
        chunks: List[List[Dict[str, Any]]] = []
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            chunk = [b for b in display if b["time"] >= i and b["time"] < i + duration]
            chunks.append(chunk)
            i += duration
        return chunks

    def chunked_phrase_divs(self, inst: int = 0, duration: float = 30) -> List[List[Dict[str, Any]]]:
        phrase_divs = self.all_phrase_divs(inst)
        chunks: List[List[Dict[str, Any]]] = []
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            chunk = [pd for pd in phrase_divs if pd["time"] >= i and pd["time"] < i + duration]
            chunks.append(chunk)
            i += duration
        return chunks

    def chunked_meters(self, duration: float = 30) -> List[List[Meter]]:
        chunks: List[List[Meter]] = []
        dur_tot = self.dur_tot or 0.0
        i = 0.0
        while i < dur_tot:
            chunk = [m for m in self.meters if m.start_time >= i and m.start_time < i + duration]
            chunks.append(chunk)
            i += duration
        return chunks

    def pulse_from_id(self, id: str):
        all_pulses = [p for m in self.meters for p in m.all_pulses]
        for pulse in all_pulses:
            if pulse.unique_id == id:
                return pulse
        return None

    # ------------------------------------------------------------------
    def clean_up_section_categorization(self, c: SecCatType) -> None:
        if "Improvisation" not in c:
            c["Improvisation"] = {"Improvisation": False}
        if "Other" not in c:
            c["Other"] = {"Other": False}
        if "Comp.-section/Tempo" not in c and "Composition-section/Tempo" in c:
            c["Comp.-section/Tempo"] = c["Composition-section/Tempo"]
            del c["Composition-section/Tempo"]
        top_level = c.get("Top Level")
        if not top_level or top_level == "None":
            if any(c["Pre-Chiz Alap"].values()):
                c["Top Level"] = "Pre-Chiz Alap"
            elif any(c["Alap"].values()):
                c["Top Level"] = "Alap"
            elif any(c["Composition Type"].values()):
                c["Top Level"] = "Composition"
            elif any(c["Comp.-section/Tempo"].values()):
                c["Top Level"] = "Composition"
            elif any(c["Improvisation"].values()):
                c["Top Level"] = "Improvisation"
            elif any(c["Other"].values()):
                c["Top Level"] = "Other"
            else:
                c["Top Level"] = "None"

    # ------------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "raga": self.raga.to_json(),
            "phraseGrid": [[p.to_json() for p in row] for row in self.phrase_grid],
            "instrumentation": [i.value if isinstance(i, Instrument) else i for i in self.instrumentation],
            "trackTitles": self.track_titles,
            "durTot": self.dur_tot,
            "durArrayGrid": self.dur_array_grid,
            "meters": [m.to_json() for m in self.meters],
            "title": self.title,
            # PROP-2: emit ISO-8601 UTC with an explicit 'Z', never naive/{$date}.
            "dateCreated": _iso_utc(self.date_created),
            "dateModified": _iso_utc(self.date_modified),
            "location": self.location,
            "_id": self._id,
            "audioID": self.audio_id,
            "userID": self.user_id,
            "permissions": self.permissions,
            "name": self.name,
            "family_name": self.family_name,
            "given_name": self.given_name,
            "sectionCatGrid": self.section_cat_grid,
            "explicitPermissions": self.explicit_permissions,
            "soloist": self.soloist,
            "soloInstrument": self.solo_instrument,
            "excerptRange": self.excerpt_range,
            "adHocSectionCatGrid": self.ad_hoc_section_cat_grid,
            "assemblageDescriptors": self.assemblage_descriptors,
            # PROP-2: `collections` is transcription-document metadata maintained only by
            # the add/remove-to-collection endpoints (server uses $set, so omitting it is
            # non-destructive). Including it risks clobbering server-managed membership.
        }
        # drop None values so they serialize as undefined (omitted) rather than null
        return {k: v for k, v in data.items() if v is not None}

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "Piece":
        new_obj = dict(obj)
        raga = None
        if "raga" in new_obj:
            raga = Raga.from_json(new_obj["raga"])
            new_obj["raga"] = raga
        # ROOT of the context chain: extract (ratios, fundamental) from the raga
        # and thread down to every phrase -> trajectory -> pitch. Without this,
        # stripped pieces load at the default 261.63 Hz (idtap-contract PIECE-1).
        ratios = raga.stratified_ratios if raga else None
        fundamental = raga.fundamental if raga else None
        if "phraseGrid" in new_obj:
            pg = []
            for row in new_obj["phraseGrid"]:
                phrase_row = [Phrase.from_json(p, ratios=ratios, fundamental=fundamental) for p in row]
                pg.append(phrase_row)
            # reconstruct groups so they reference existing trajectories
            for row in pg:
                for phrase in row:
                    new_groups: List[List[Group]] = []
                    for g_list in phrase.groups_grid:
                        rebuilt: List[Group] = []
                        for g in g_list:
                            data = g if isinstance(g, dict) else g.to_json()
                            trajs = []
                            for t in data.get("trajectories", []):
                                traj = t if isinstance(t, Trajectory) else Trajectory.from_json(t)
                                num = traj.num
                                if num is None or num >= len(phrase.trajectory_grid[0]):
                                    continue
                                real_traj = phrase.trajectory_grid[0][num]
                                art = real_traj.articulations.get("0.00") or real_traj.articulations.get("0")
                                if art and art.name == "slide":
                                    art.name = "pluck"
                                trajs.append(real_traj)
                            rebuilt.append(Group({"trajectories": trajs, "id": data.get("id")}))
                        new_groups.append(rebuilt)
                    phrase.groups_grid = new_groups
            new_obj["phraseGrid"] = pg
        if "meters" in new_obj:
            new_obj["meters"] = [Meter.from_json(m) for m in new_obj["meters"]]
        if "dateCreated" in new_obj:
            new_obj["dateCreated"] = _parse_utc(new_obj["dateCreated"])
        if "dateModified" in new_obj:
            new_obj["dateModified"] = _parse_utc(new_obj["dateModified"])

        piece = Piece(new_obj)

        # reconnect groups to actual trajectories
        for phrases in piece.phrase_grid:
            for phrase in phrases:
                new_group_grid: List[List[Group]] = []
                for group_list in phrase.groups_grid:
                    rebuilt: List[Group] = []
                    for g in group_list:
                        if not isinstance(g, Group):
                            g = Group.from_json(g)
                        new_trajs: List[Trajectory] = []
                        for t in g.trajectories:
                            if t.num is None:
                                raise Exception("traj.num is undefined")
                            new_trajs.append(phrase.trajectory_grid[0][t.num])
                        rebuilt.append(Group({"trajectories": new_trajs, "id": g.id}))
                    new_group_grid.append(rebuilt)
                phrase.groups_grid = new_group_grid

                for traj in phrase.trajectories:
                    art = traj.articulations.get("0.00")
                    if art and art.name == "slide":
                        art.name = "pluck"
                phrase.consolidate_silent_trajs()

        piece.dur_array_from_phrases()
        piece.section_starts_grid = [sorted(set(arr)) for arr in piece.section_starts_grid]

        # STRING-SYNC runs in Piece.__init__ (matching the TS constructor) — no
        # second call here, or overhang warnings would fire twice per load.

        return piece
