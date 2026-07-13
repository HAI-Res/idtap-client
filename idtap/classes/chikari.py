from __future__ import annotations
from typing import List, Optional, Dict, TypedDict
import uuid

import humps

from .pitch import Pitch
from ..constants import MIN_FUNDAMENTAL_HZ, MAX_FUNDAMENTAL_HZ


class ChikariOptionsType(TypedDict, total=False):
    pitches: List[Pitch] | List[Dict]
    fundamental: float
    unique_id: str


class Chikari:
    def __init__(self, options: Optional[ChikariOptionsType] = None) -> None:
        opts = humps.decamelize(options or {})
        
        # Parameter validation
        self._validate_parameters(opts)

        default_pitches = [
            Pitch({'swara': 's', 'oct': 2}),
            Pitch({'swara': 's', 'oct': 1}),
            Pitch({'swara': 'p', 'oct': 0}),
            Pitch({'swara': 'g', 'oct': 0}),
        ]
        pitches_in = opts.get('pitches', default_pitches)
        fundamental = opts.get('fundamental', Pitch().fundamental)
        unique_id = opts.get('unique_id')

        self.unique_id: str = str(unique_id) if unique_id is not None else str(uuid.uuid4())
        self.fundamental: float = fundamental

        self.pitches: List[Pitch] = []
        for p in pitches_in:
            if not isinstance(p, Pitch):
                p = Pitch(p)  # type: ignore[arg-type]
            p.fundamental = self.fundamental
            self.pitches.append(p)

    def _validate_parameters(self, opts: dict) -> None:
        """Validate constructor parameters and provide helpful error messages."""
        if not opts:
            return
            
        # Define allowed parameter names
        allowed_keys = {'pitches', 'fundamental', 'unique_id'}
        provided_keys = set(opts.keys())
        invalid_keys = provided_keys - allowed_keys
        
        # Check for invalid parameter names
        if invalid_keys:
            error_messages = []
            
            for key in invalid_keys:
                if key == 'fundamental_freq':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'fundamental'?")
                elif key == 'pitch_list':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'pitches'?")
                else:
                    error_messages.append(f"Invalid parameter: '{key}'")
            
            error_msg = "; ".join(error_messages)
            error_msg += f". Allowed parameters: {sorted(allowed_keys)}"
            raise ValueError(error_msg)
        
        # Validate parameter types and values
        if 'pitches' in opts and opts['pitches'] is not None:
            if not isinstance(opts['pitches'], list):
                raise TypeError(f"Parameter 'pitches' must be a list, got {type(opts['pitches']).__name__}")
        
        if 'fundamental' in opts and opts['fundamental'] is not None:
            if not isinstance(opts['fundamental'], (int, float)):
                raise TypeError(f"Parameter 'fundamental' must be a number, got {type(opts['fundamental']).__name__}")
            if opts['fundamental'] <= 0:
                raise ValueError(f"Parameter 'fundamental' must be positive, got {opts['fundamental']}")
            if opts['fundamental'] < MIN_FUNDAMENTAL_HZ or opts['fundamental'] > MAX_FUNDAMENTAL_HZ:
                import warnings
                warnings.warn(
                    f"Fundamental frequency {opts['fundamental']}Hz is outside typical range ({MIN_FUNDAMENTAL_HZ}-{MAX_FUNDAMENTAL_HZ}Hz)",
                    UserWarning
                )
        
        if 'unique_id' in opts and opts['unique_id'] is not None:
            if not isinstance(opts['unique_id'], str):
                raise TypeError(f"Parameter 'unique_id' must be a string, got {type(opts['unique_id']).__name__}")

    # ------------------------------------------------------------------
    def to_json(self) -> Dict:
        return {
            'fundamental': self.fundamental,
            'uniqueId': self.unique_id,
        }

    @staticmethod
    def from_json(obj: Dict) -> 'Chikari':
        opts = humps.decamelize(obj)
        # Handle old format (with pitches) for backward compatibility
        pitches_data = opts.get('pitches')
        if pitches_data:
            pitches = [Pitch.from_json(p) for p in pitches_data]
            opts['pitches'] = pitches
        else:
            # PROP-3: the canonical form ({fundamental, uniqueId}) carries no pitches.
            # Do NOT synthesize 12-TET default pitches here — that produced a footgun
            # where a Python consumer reading chikari.pitches[i].frequency got 12-TET
            # for a non-12-TET raga. Drone tuning is owned by the raga
            # (Piece.chikari_freqs -> raga.chikari_pitches); leave pitches empty on
            # canonical load, matching the TS client's behavior.
            opts['pitches'] = []
        return Chikari(opts)  # type: ignore[arg-type]
