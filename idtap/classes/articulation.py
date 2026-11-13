from __future__ import annotations
from typing import Optional, TypedDict, Dict

import humps
from ..utils import split_unknown_keys, merge_extra_keys

class ArticulationOptions(TypedDict, total=False):
    name: str
    stroke: str
    hindi: str
    ipa: str
    eng_trans: str
    stroke_nickname: str

class Articulation:
    _KNOWN_KEYS = {'name', 'stroke', 'hindi', 'ipa', 'eng_trans', 'stroke_nickname'}

    def __init__(self, options: Optional[ArticulationOptions] = None) -> None:
        opts = humps.decamelize(options or {})
        
        # Parameter validation
        self._validate_parameters(opts)
        self.name: str = opts.get('name', 'pluck')
        stroke = opts.get('stroke')
        hindi = opts.get('hindi')
        ipa = opts.get('ipa')
        eng_trans = opts.get('eng_trans')
        stroke_nickname = opts.get('stroke_nickname')

        if stroke is not None:
            self.stroke = stroke
        if hindi is not None:
            self.hindi = hindi
        if ipa is not None:
            self.ipa = ipa
        if eng_trans is not None:
            self.eng_trans = eng_trans
        if stroke_nickname is not None:
            self.stroke_nickname = stroke_nickname

        if getattr(self, 'stroke', None) == 'd' and not getattr(self, 'stroke_nickname', None):
            self.stroke_nickname = 'da'
        elif getattr(self, 'stroke', None) == 'r' and not getattr(self, 'stroke_nickname', None):
            self.stroke_nickname = 'ra'

    def _validate_parameters(self, opts: dict) -> None:
        """Validate constructor parameters and provide helpful error messages."""
        if not opts:
            return
            
        # Define allowed parameter names
        allowed_keys = Articulation._KNOWN_KEYS
        provided_keys = set(opts.keys())
        invalid_keys = provided_keys - allowed_keys
        
        # Check for invalid parameter names
        if invalid_keys:
            error_messages = []
            
            for key in invalid_keys:
                if key == 'english_trans' or key == 'english':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'eng_trans'?")
                elif key == 'articulation_name':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'name'?")
                else:
                    error_messages.append(f"Invalid parameter: '{key}'")
            
            error_msg = "; ".join(error_messages)
            error_msg += f". Allowed parameters: {sorted(allowed_keys)}"
            raise ValueError(error_msg)
        
        # Validate parameter types
        # Note: stroke can be various types in some contexts, so we're less strict
        string_params = ['name', 'hindi', 'ipa', 'eng_trans', 'stroke_nickname']
        for param in string_params:
            if param in opts and opts[param] is not None and not isinstance(opts[param], str):
                raise TypeError(f"Parameter '{param}' must be a string, got {type(opts[param]).__name__}")
        
        # Validate that name is not empty if provided
        if 'name' in opts and opts['name'] == "":
            raise ValueError("Parameter 'name' cannot be empty")

    @staticmethod
    def from_json(obj: Dict) -> 'Articulation':
        # Unknown wire keys are kept verbatim and re-emitted (PROP-8).
        obj, extra = split_unknown_keys(obj, Articulation._KNOWN_KEYS)
        art = Articulation(obj)
        art._extra = extra
        return art

    def to_json(self) -> Dict:
        out = {}
        if hasattr(self, 'name'):
            out['name'] = self.name
        if hasattr(self, 'stroke'):
            out['stroke'] = self.stroke
        if hasattr(self, 'hindi'):
            out['hindi'] = self.hindi
        if hasattr(self, 'ipa'):
            out['ipa'] = self.ipa
        if hasattr(self, 'eng_trans'):
            out['engTrans'] = self.eng_trans
        if hasattr(self, 'stroke_nickname'):
            out['strokeNickname'] = self.stroke_nickname
        return merge_extra_keys(out, getattr(self, '_extra', None))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Articulation):
            return False
        return self.to_json() == other.to_json()
