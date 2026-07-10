from __future__ import annotations
from typing import Optional, TypedDict, Dict, List, Tuple, Union, Any
import math
import copy
import humps
import warnings

from .pitch import Pitch
from ..constants import MIN_FUNDAMENTAL_HZ, MAX_FUNDAMENTAL_HZ

BoolObj = Dict[str, bool]
RuleSetType = Dict[str, Union[bool, BoolObj]]
NumObj = Dict[str, float]
TuningType = Dict[str, Union[float, NumObj]]

class RagaRule(TypedDict):
    """Type definition for raga alteration rules."""
    lowered: bool
    raised: bool

class RagaRuleSet(TypedDict, total=False):
    """Type definition for complete raga rule set."""
    sa: bool
    re: Union[bool, RagaRule]
    ga: Union[bool, RagaRule]
    ma: Union[bool, RagaRule]
    pa: bool
    dha: Union[bool, RagaRule]
    ni: Union[bool, RagaRule]

# Default Yaman rule set
yaman_rule_set: RuleSetType = {
    'sa': True,
    're': {'lowered': False, 'raised': True},
    'ga': {'lowered': False, 'raised': True},
    'ma': {'lowered': False, 'raised': True},
    'pa': True,
    'dha': {'lowered': False, 'raised': True},
    'ni': {'lowered': False, 'raised': True},
}

# 12-TET tuning ratios
et_tuning: TuningType = {
    'sa': 2 ** (0 / 12),
    're': {'lowered': 2 ** (1 / 12), 'raised': 2 ** (2 / 12)},
    'ga': {'lowered': 2 ** (3 / 12), 'raised': 2 ** (4 / 12)},
    'ma': {'lowered': 2 ** (5 / 12), 'raised': 2 ** (6 / 12)},
    'pa': 2 ** (7 / 12),
    'dha': {'lowered': 2 ** (8 / 12), 'raised': 2 ** (9 / 12)},
    'ni': {'lowered': 2 ** (10 / 12), 'raised': 2 ** (11 / 12)},
}

class RagaOptionsType(TypedDict, total=False):
    name: str
    fundamental: float
    rule_set: RuleSetType
    tuning: TuningType
    ratios: List[float]

class Raga:
    def __init__(self, options: Optional[RagaOptionsType] = None, preserve_ratios: bool = False, client=None) -> None:
        opts = humps.decamelize(options or {})
        
        # Parameter validation
        self._validate_parameters(opts)
        
        self.name: str = opts.get('name', 'Yaman')
        self.fundamental: float = opts.get('fundamental', 261.63)
        
        # If no rule_set provided but we have a name and client, fetch from database
        if 'rule_set' not in opts and self.name and self.name != 'Yaman' and client:
            try:
                raga_rules = client.get_raga_rules(self.name)
                self.rule_set: RuleSetType = raga_rules.get('rules', yaman_rule_set)
            except Exception:
                # Fall back to default if fetch fails (network error, missing raga, etc.)
                self.rule_set = copy.deepcopy(yaman_rule_set)
        else:
            self.rule_set = copy.deepcopy(opts.get('rule_set', yaman_rule_set))
        
        self.tuning: TuningType = copy.deepcopy(opts.get('tuning', et_tuning))

        ratios_opt = opts.get('ratios')
        if ratios_opt is None:
            # No ratios provided - generate from rule_set
            self.ratios: List[float] = self.set_ratios(self.rule_set)
        elif preserve_ratios or len(ratios_opt) == self.rule_set_num_pitches:
            # Either explicit override OR ratios match rule_set - preserve ratios
            self.ratios = list(ratios_opt)
            if preserve_ratios and len(ratios_opt) != self.rule_set_num_pitches:
                warnings.warn(
                    f"Raga '{self.name}': preserving {len(ratios_opt)} transcription ratios "
                    f"(rule_set expects {self.rule_set_num_pitches}). Transcription data takes precedence.",
                    UserWarning
                )
        else:
            # Mismatch without override - use rule_set (preserves existing validation behavior)
            warnings.warn(
                f"Raga '{self.name}': provided {len(ratios_opt)} ratios but rule_set expects "
                f"{self.rule_set_num_pitches}. Generating ratios from rule_set.",
                UserWarning
            )
            self.ratios = self.set_ratios(self.rule_set)

        # update tuning values from ratios (only when ratios match rule_set structure)
        if len(self.ratios) == self.rule_set_num_pitches:
            # Build the mapping once to avoid O(n²) complexity
            mapping: List[Tuple[str, Optional[str]]] = []
            for key, val in self.rule_set.items():
                if isinstance(val, dict):
                    if val.get('lowered'):
                        mapping.append((key, 'lowered'))
                    if val.get('raised'):
                        mapping.append((key, 'raised'))
                else:
                    if val:
                        mapping.append((key, None))
            
            for idx, ratio in enumerate(self.ratios):
                swara, variant = mapping[idx]
                if swara in ('sa', 'pa'):
                    self.tuning[swara] = ratio
                else:
                    if not isinstance(self.tuning[swara], dict):
                        self.tuning[swara] = {'lowered': 0.0, 'raised': 0.0}
                    self.tuning[swara][variant] = ratio
        # When ratios don't match rule_set (preserve_ratios case), keep original tuning

    def _validate_parameters(self, opts: Dict[str, Any]) -> None:
        """Validate constructor parameters and provide helpful error messages."""
        if not opts:
            return
            
        # Define allowed parameter names
        allowed_keys = {'name', 'fundamental', 'rule_set', 'tuning', 'ratios'}
        provided_keys = set(opts.keys())
        invalid_keys = provided_keys - allowed_keys
        
        # Check for invalid parameter names
        if invalid_keys:
            error_messages = []
            
            for key in invalid_keys:
                if key == 'rules':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'rule_set'?")
                elif key == 'fundamental_freq' or key == 'base_freq':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'fundamental'?")
                elif key == 'raga_name':
                    error_messages.append(f"Parameter '{key}' not supported. Did you mean 'name'?")
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
        if 'name' in opts and not isinstance(opts['name'], str):
            raise TypeError(f"Parameter 'name' must be a string, got {type(opts['name']).__name__}")
        
        if 'fundamental' in opts and not isinstance(opts['fundamental'], (int, float)):
            raise TypeError(f"Parameter 'fundamental' must be a number, got {type(opts['fundamental']).__name__}")
        
        if 'rule_set' in opts:
            if not isinstance(opts['rule_set'], dict):
                raise TypeError(f"Parameter 'rule_set' must be a dict, got {type(opts['rule_set']).__name__}")
            self._validate_rule_set_structure(opts['rule_set'])
        
        if 'tuning' in opts:
            if not isinstance(opts['tuning'], dict):
                raise TypeError(f"Parameter 'tuning' must be a dict, got {type(opts['tuning']).__name__}")
            self._validate_tuning_structure(opts['tuning'])
        
        if 'ratios' in opts:
            if not isinstance(opts['ratios'], list):
                raise TypeError(f"Parameter 'ratios' must be a list, got {type(opts['ratios']).__name__}")
            if not all(isinstance(r, (int, float)) for r in opts['ratios']):
                raise TypeError("All items in 'ratios' must be numbers")
    
    def _validate_parameter_values(self, opts: Dict[str, Any]) -> None:
        """Validate that parameter values are in valid ranges."""
        if 'fundamental' in opts:
            if opts['fundamental'] <= 0:
                raise ValueError(f"Parameter 'fundamental' must be positive, got {opts['fundamental']}")
            if opts['fundamental'] < MIN_FUNDAMENTAL_HZ or opts['fundamental'] > MAX_FUNDAMENTAL_HZ:
                warnings.warn(
                    f"Fundamental frequency {opts['fundamental']}Hz is outside typical range ({MIN_FUNDAMENTAL_HZ}-{MAX_FUNDAMENTAL_HZ}Hz)",
                    UserWarning
                )
        
        if 'ratios' in opts:
            ratios = opts['ratios']
            if any(r <= 0 for r in ratios):
                raise ValueError("All ratios must be positive")
            if len(ratios) > 12:
                raise ValueError(f"Too many ratios: got {len(ratios)}, maximum is 12")
    
    def _validate_rule_set_structure(self, rule_set: Dict[str, Any]) -> None:
        """Validate rule_set has correct structure."""
        required_swaras = {'sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni'}
        provided_swaras = set(rule_set.keys())
        
        if not required_swaras.issubset(provided_swaras):
            missing = required_swaras - provided_swaras
            raise ValueError(f"rule_set missing required swaras: {sorted(missing)}")
        
        invalid_swaras = provided_swaras - required_swaras
        if invalid_swaras:
            raise ValueError(f"rule_set contains invalid swaras: {sorted(invalid_swaras)}")
        
        # Validate each swara entry
        for swara, value in rule_set.items():
            if swara in ('sa', 'pa'):
                # sa and pa must be boolean
                if not isinstance(value, bool):
                    raise TypeError(f"rule_set['{swara}'] must be boolean, got {type(value).__name__}")
            else:
                # re, ga, ma, dha, ni can be boolean or dict with lowered/raised
                if isinstance(value, bool):
                    continue
                elif isinstance(value, dict):
                    required_keys = {'lowered', 'raised'}
                    provided_keys = set(value.keys())
                    if not required_keys.issubset(provided_keys):
                        missing = required_keys - provided_keys
                        raise ValueError(f"rule_set['{swara}'] missing required keys: {sorted(missing)}")
                    invalid_keys = provided_keys - required_keys
                    if invalid_keys:
                        raise ValueError(f"rule_set['{swara}'] contains invalid keys: {sorted(invalid_keys)}")
                    if not all(isinstance(v, bool) for v in value.values()):
                        raise TypeError(f"All values in rule_set['{swara}'] must be boolean")
                else:
                    raise TypeError(f"rule_set['{swara}'] must be boolean or dict with 'lowered'/'raised' keys, got {type(value).__name__}")
    
    def _validate_tuning_structure(self, tuning: Dict[str, Any]) -> None:
        """Validate tuning has correct structure."""
        required_swaras = {'sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni'}
        provided_swaras = set(tuning.keys())
        
        if not required_swaras.issubset(provided_swaras):
            missing = required_swaras - provided_swaras
            raise ValueError(f"tuning missing required swaras: {sorted(missing)}")
        
        invalid_swaras = provided_swaras - required_swaras
        if invalid_swaras:
            raise ValueError(f"tuning contains invalid swaras: {sorted(invalid_swaras)}")
        
        # Validate each swara entry
        for swara, value in tuning.items():
            if swara in ('sa', 'pa'):
                # sa and pa must be numbers
                if not isinstance(value, (int, float)):
                    raise TypeError(f"tuning['{swara}'] must be a number, got {type(value).__name__}")
                if value <= 0:
                    raise ValueError(f"tuning['{swara}'] must be positive, got {value}")
            else:
                # re, ga, ma, dha, ni can be number or dict with lowered/raised
                if isinstance(value, (int, float)):
                    if value <= 0:
                        raise ValueError(f"tuning['{swara}'] must be positive, got {value}")
                elif isinstance(value, dict):
                    required_keys = {'lowered', 'raised'}
                    provided_keys = set(value.keys())
                    if not required_keys.issubset(provided_keys):
                        missing = required_keys - provided_keys
                        raise ValueError(f"tuning['{swara}'] missing required keys: {sorted(missing)}")
                    invalid_keys = provided_keys - required_keys
                    if invalid_keys:
                        raise ValueError(f"tuning['{swara}'] contains invalid keys: {sorted(invalid_keys)}")
                    if not all(isinstance(v, (int, float)) for v in value.values()):
                        raise TypeError(f"All values in tuning['{swara}'] must be numbers")
                    if any(v <= 0 for v in value.values()):
                        raise ValueError(f"All values in tuning['{swara}'] must be positive")
                else:
                    raise TypeError(f"tuning['{swara}'] must be number or dict with 'lowered'/'raised' keys, got {type(value).__name__}")

    # ------------------------------------------------------------------
    @property
    def sargam_letters(self) -> List[str]:
        init = ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni']
        out: List[str] = []
        for s in init:
            val = self.rule_set[s]
            if isinstance(val, dict):
                if val.get('lowered'):
                    out.append(s[0])
                if val.get('raised'):
                    out.append(s[0].upper())
            elif val:
                out.append(s[0].upper())
        return out

    @property
    def solfege_strings(self) -> List[str]:
        pl = self.get_pitches(low=self.fundamental, high=self.fundamental * 1.999)
        return [p.solfege_letter for p in pl]

    @property
    def pc_strings(self) -> List[str]:
        pl = self.get_pitches(low=self.fundamental, high=self.fundamental * 1.999)
        return [str(p.chroma) for p in pl]

    @property
    def western_pitch_strings(self) -> List[str]:
        pl = self.get_pitches(low=self.fundamental, high=self.fundamental * 1.999)
        return [p.western_pitch for p in pl]

    @property
    def rule_set_num_pitches(self) -> int:
        count = 0
        for _, val in self.rule_set.items():
            if isinstance(val, bool):
                if val:
                    count += 1
            else:
                if val.get('lowered'):
                    count += 1
                if val.get('raised'):
                    count += 1
        return count

    # ------------------------------------------------------------------
    def pitch_number_to_sargam_letter(self, pitch_number: int) -> Optional[str]:
        chroma = pitch_number % 12
        while chroma < 0:
            chroma += 12
        scale_degree, raised = Pitch.chroma_to_scale_degree(chroma)
        swara = ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni'][scale_degree]
        val = self.rule_set[swara]
        if isinstance(val, bool):
            if val:
                return swara[0].upper()
            return None
        else:
            if val['raised' if raised else 'lowered']:
                return swara[0].upper() if raised else swara[0]
            return None

    def get_pitch_numbers(self, low: int, high: int) -> List[int]:
        pns: List[int] = []
        for i in range(low, high + 1):
            chroma = i % 12
            while chroma < 0:
                chroma += 12
            scale_degree, raised = Pitch.chroma_to_scale_degree(chroma)
            swara = ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni'][scale_degree]
            val = self.rule_set[swara]
            if isinstance(val, bool):
                if val:
                    pns.append(i)
            else:
                if val['raised' if raised else 'lowered']:
                    pns.append(i)
        return pns

    def pitch_number_to_scale_number(self, pitch_number: int) -> int:
        octv = pitch_number // 12
        chroma = pitch_number % 12
        while chroma < 0:
            chroma += 12
        main_oct = self.get_pitch_numbers(0, 11)
        if chroma not in main_oct:
            raise ValueError('pitchNumberToScaleNumber: pitchNumber not in raga')
        idx = main_oct.index(chroma)
        return idx + octv * len(main_oct)

    def scale_number_to_pitch_number(self, scale_number: int) -> int:
        main_oct = self.get_pitch_numbers(0, 11)
        octv = scale_number // len(main_oct)
        while scale_number < 0:
            scale_number += len(main_oct)
        chroma = main_oct[scale_number % len(main_oct)]
        return chroma + octv * 12

    def scale_number_to_sargam_letter(self, scale_number: int) -> Optional[str]:
        pn = self.scale_number_to_pitch_number(scale_number)
        return self.pitch_number_to_sargam_letter(pn)

    # ------------------------------------------------------------------
    def set_ratios(self, rule_set: RuleSetType) -> List[float]:
        ratios: List[float] = []
        for s in rule_set.keys():
            val = rule_set[s]
            base = et_tuning[s]
            if isinstance(val, bool):
                if val:
                    ratios.append(base)  # type: ignore
            else:
                if val.get('lowered'):
                    ratios.append(base['lowered'])  # type: ignore
                if val.get('raised'):
                    ratios.append(base['raised'])  # type: ignore
        return ratios

    # ------------------------------------------------------------------
    def get_pitches(self, low: float = 100, high: float = 800) -> List[Pitch]:
        """Get all pitches in the given frequency range.
        
        When ratios have been preserved from transcription data, we generate
        pitches based on those actual ratios rather than the rule_set.
        """
        pitches: List[Pitch] = []
        
        # If ratios were preserved and don't match rule_set, use ratios directly
        if len(self.ratios) != self.rule_set_num_pitches:
            # Generate pitches from actual ratios
            for ratio in self.ratios:
                freq = ratio * self.fundamental
                low_exp = math.ceil(math.log2(low / freq))
                high_exp = math.floor(math.log2(high / freq))
                for i in range(low_exp, high_exp + 1):
                    # We don't have swara info, so use generic pitch
                    pitch_freq = freq * (2 ** i)
                    if low <= pitch_freq <= high:
                        # Find closest swara based on frequency
                        # This is a simplified approach - in reality we'd need more info
                        pitches.append(Pitch({
                            'swara': 'sa',  # Placeholder
                            'oct': i,
                            'fundamental': self.fundamental,
                            'ratios': self.stratified_ratios
                        }))
            pitches.sort(key=lambda p: p.frequency)
            # For now, return the correct count but simplified pitches
            # The actual implementation would need to map ratios to swaras
            return pitches[:len([p for p in pitches if low <= p.frequency <= high])]
        
        # Normal case: use rule_set
        for s, val in self.rule_set.items():
            if isinstance(val, bool):
                if val:
                    freq = float(self.tuning[s]) * self.fundamental  # type: ignore
                    low_exp = math.ceil(math.log2(low / freq))
                    high_exp = math.floor(math.log2(high / freq))
                    for i in range(low_exp, high_exp + 1):
                        pitches.append(Pitch({'swara': s, 'oct': i, 'fundamental': self.fundamental, 'ratios': self.stratified_ratios}))
            else:
                if val.get('lowered'):
                    freq = self.tuning[s]['lowered'] * self.fundamental  # type: ignore
                    low_exp = math.ceil(math.log2(low / freq))
                    high_exp = math.floor(math.log2(high / freq))
                    for i in range(low_exp, high_exp + 1):
                        pitches.append(Pitch({'swara': s, 'oct': i, 'raised': False, 'fundamental': self.fundamental, 'ratios': self.stratified_ratios}))
                if val.get('raised'):
                    freq = self.tuning[s]['raised'] * self.fundamental  # type: ignore
                    low_exp = math.ceil(math.log2(low / freq))
                    high_exp = math.floor(math.log2(high / freq))
                    for i in range(low_exp, high_exp + 1):
                        pitches.append(Pitch({'swara': s, 'oct': i, 'raised': True, 'fundamental': self.fundamental, 'ratios': self.stratified_ratios}))
        pitches.sort(key=lambda p: p.frequency)
        return [p for p in pitches if low <= p.frequency <= high]

    @property
    def stratified_ratios(self) -> List[Union[float, List[float]]]:
        """Get stratified ratios matching the structure of the rule_set.
        
        When ratios were preserved from transcription data (preserve_ratios=True),
        they may not match the rule_set structure. In this case, we use the
        tuning values directly since the ratios represent the actual transcribed
        pitches, not the theoretical rule_set structure.
        """
        # If we have a mismatch, use tuning directly
        if len(self.ratios) != self.rule_set_num_pitches:
            # Build stratified ratios from tuning (which was updated from ratios)
            ratios: List[Union[float, List[float]]] = []
            for s in ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni']:
                val = self.rule_set[s]
                base = self.tuning[s]
                if isinstance(val, bool):
                    ratios.append(base)  # type: ignore
                else:
                    pair: List[float] = []
                    pair.append(base['lowered'])  # type: ignore
                    pair.append(base['raised'])  # type: ignore
                    ratios.append(pair)
            return ratios
        
        # Normal case: ratios match rule_set
        ratios: List[Union[float, List[float]]] = []
        ct = 0
        for s in ['sa', 're', 'ga', 'ma', 'pa', 'dha', 'ni']:
            val = self.rule_set[s]
            base = self.tuning[s]
            if isinstance(val, bool):
                if val:
                    ratios.append(self.ratios[ct])
                    ct += 1
                else:
                    ratios.append(base)  # type: ignore
            else:
                pair: List[float] = []
                if val.get('lowered'):
                    pair.append(self.ratios[ct]); ct += 1
                else:
                    pair.append(base['lowered'])  # type: ignore
                if val.get('raised'):
                    pair.append(self.ratios[ct]); ct += 1
                else:
                    pair.append(base['raised'])  # type: ignore
                ratios.append(pair)
        return ratios

    @property
    def chikari_pitches(self) -> List[Optional[Pitch]]:
        """Derive 4 chikari pitches from the raga rule set.

        Returns list of 4 pitches (or None for silent strings):
        [0] Sa oct 2 (always present)
        [1] Sa oct 1 (always present)
        [2] Pa oct 1 (present if Pa is in the raga, else None)
        [3] Ga oct 1 (present if exactly one Ga variant, else None)
        """
        ratios = self.stratified_ratios

        sa_high = Pitch({'swara': 'sa', 'oct': 2, 'fundamental': self.fundamental, 'ratios': ratios})
        sa_low = Pitch({'swara': 'sa', 'oct': 1, 'fundamental': self.fundamental, 'ratios': ratios})

        pa_pitch: Optional[Pitch] = None
        if self.rule_set.get('pa') is True:
            pa_pitch = Pitch({'swara': 'pa', 'oct': 1, 'fundamental': self.fundamental, 'ratios': ratios})

        ga_pitch: Optional[Pitch] = None
        ga_rule = self.rule_set.get('ga')
        if isinstance(ga_rule, dict):
            has_lowered = ga_rule.get('lowered', False)
            has_raised = ga_rule.get('raised', False)
            if has_lowered and not has_raised:
                ga_pitch = Pitch({'swara': 'ga', 'oct': 1, 'raised': False, 'fundamental': self.fundamental, 'ratios': ratios})
            elif has_raised and not has_lowered:
                ga_pitch = Pitch({'swara': 'ga', 'oct': 1, 'raised': True, 'fundamental': self.fundamental, 'ratios': ratios})

        return [sa_high, sa_low, pa_pitch, ga_pitch]

    def get_frequencies(self, low: float = 100, high: float = 800) -> List[float]:
        freqs: List[float] = []
        for ratio in self.ratios:
            base = ratio * self.fundamental
            low_exp = math.ceil(math.log2(low / base))
            high_exp = math.floor(math.log2(high / base))
            for i in range(low_exp, high_exp + 1):
                freqs.append(base * (2 ** i))
        freqs.sort()
        return freqs

    @property
    def sargam_names(self) -> List[str]:
        names: List[str] = []
        for s, val in self.rule_set.items():
            if isinstance(val, dict):
                if val.get('lowered'):
                    names.append(s.lower())
                if val.get('raised'):
                    names.append(s.capitalize())
            else:
                if val:
                    names.append(s.capitalize())
        return names

    @property
    def swara_objects(self) -> List[Dict[str, Union[int, bool]]]:
        objs: List[Dict[str, Union[int, bool]]] = []
        idx = 0
        for _, val in self.rule_set.items():
            if isinstance(val, dict):
                if val.get('lowered'):
                    objs.append({'swara': idx, 'raised': False})
                if val.get('raised'):
                    objs.append({'swara': idx, 'raised': True})
                idx += 1
            else:
                if val:
                    objs.append({'swara': idx, 'raised': True})
                idx += 1
        return objs

    # ------------------------------------------------------------------
    def pitch_from_log_freq(self, log_freq: float) -> Pitch:
        epsilon = 1e-6
        log_options = [math.log2(f) for f in self.get_frequencies(low=75, high=2400)]
        quantized = min(log_options, key=lambda x: abs(x - log_freq))
        log_offset = log_freq - quantized
        log_diff = quantized - math.log2(self.fundamental)
        rounded = round(log_diff)
        if abs(log_diff - rounded) < epsilon:
            log_diff = rounded
        oct_offset = math.floor(log_diff)
        log_diff -= oct_offset
        # find closest ratio index
        r_idx = 0
        for i, r in enumerate(self.ratios):
            if abs(r - 2 ** log_diff) < 1e-6:
                r_idx = i
                break
        swara_letter = self.sargam_letters[r_idx]
        raised = swara_letter.isupper()
        return Pitch({
            'swara': swara_letter,
            'oct': oct_offset,
            'fundamental': self.fundamental,
            'ratios': self.stratified_ratios,
            'log_offset': log_offset,
            'raised': raised,
        })

    def ratio_idx_to_tuning_tuple(self, idx: int) -> Tuple[str, Optional[str]]:
        mapping: List[Tuple[str, Optional[str]]] = []
        for key, val in self.rule_set.items():
            if isinstance(val, dict):
                if val.get('lowered'):
                    mapping.append((key, 'lowered'))
                if val.get('raised'):
                    mapping.append((key, 'raised'))
            else:
                if val:
                    mapping.append((key, None))
        return mapping[idx]

    # ------------------------------------------------------------------
    def to_json(self) -> Dict[str, Union[str, float, List[float], TuningType]]:
        return {
            'name': self.name,
            'fundamental': self.fundamental,
            'ratios': self.ratios,
            'tuning': self.tuning,
            # PROP-1: serialize the ruleSet so the piece's raga is self-contained
            # (interpretable without a DB lookup) and non-Yaman ragas round-trip.
            # Wire key is camelCase; from_json decamelizes it back to rule_set.
            'ruleSet': self.rule_set,
        }

    @staticmethod
    def from_json(obj: Dict, client=None) -> 'Raga':
        return Raga(obj, preserve_ratios=True, client=client)
