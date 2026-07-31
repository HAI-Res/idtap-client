import os
import sys
import math
import pytest

sys.path.insert(0, os.path.abspath('.'))

from idtap.classes.raga import Raga, yaman_rule_set, et_tuning
from idtap.classes.pitch import Pitch

base_tuning = et_tuning

base_ratios = [
    base_tuning['sa'],
    base_tuning['re']['raised'],
    base_tuning['ga']['raised'],
    base_tuning['ma']['raised'],
    base_tuning['pa'],
    base_tuning['dha']['raised'],
    base_tuning['ni']['raised'],
]

custom_rule_set = {
    'sa': True,
    're': {'lowered': True, 'raised': False},
    'ga': {'lowered': False, 'raised': True},
    'ma': {'lowered': True, 'raised': True},
    'pa': True,
    'dha': {'lowered': True, 'raised': False},
    'ni': {'lowered': False, 'raised': True},
}

ratio_mapping = [
    ('sa', True),
    ('re', False),
    ('ga', True),
    ('ma', False),
    ('ma', True),
    ('pa', True),
    ('dha', False),
    ('ni', True),
]

def compute_expected_pitches(r: Raga, low=100, high=800):
    pitches: list[Pitch] = []
    for swara, raised in ratio_mapping:
        ratio = (r.tuning[swara]['raised' if raised else 'lowered']
                 if isinstance(r.tuning[swara], dict)
                 else r.tuning[swara])
        freq = ratio * r.fundamental
        low_exp = math.ceil(math.log2(low / freq))
        high_exp = math.floor(math.log2(high / freq))
        for i in range(low_exp, high_exp + 1):
            pitches.append(Pitch({'swara': swara, 'oct': i, 'raised': raised,
                                  'fundamental': r.fundamental, 'ratios': r.stratified_ratios}))
    pitches.sort(key=lambda p: p.frequency)
    return [p for p in pitches if low <= p.frequency <= high]


def test_default_raga():
    r = Raga()
    assert isinstance(r, Raga)
    assert r.name == 'Yaman'
    assert r.fundamental == 261.63
    assert r.rule_set == yaman_rule_set
    assert r.tuning == base_tuning
    assert r.ratios == base_ratios
    assert r.sargam_letters == ['S', 'R', 'G', 'M', 'P', 'D', 'N']
    assert r.rule_set_num_pitches == 7

    pitch_nums = list(range(12))
    sargam_letters = ['S', None, 'R', None, 'G', None, 'M', 'P', None, 'D', None, 'N']
    for pn in pitch_nums:
        assert r.pitch_number_to_sargam_letter(pn) == sargam_letters[pn]

    single_oct_pns = [0, 2, 4, 6, 7, 9, 11, 12]
    assert r.get_pitch_numbers(0, 12) == single_oct_pns

    pns = [
        -12, -10, -8, -6, -5, -3, -1,
        0, 2, 4, 6, 7, 9, 11,
        12, 14, 16, 18, 19, 21, 23, 24,
    ]
    assert r.get_pitch_numbers(-12, 24) == pns

    sns = [
        -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12, 13, 14,
    ]
    throw_pns = [
        -11, -9, -7, -4, -2,
        1, 3, 5, 8, 10,
        13, 15, 17, 20, 22,
    ]

    for idx, pn in enumerate(pns):
        assert r.pitch_number_to_scale_number(pn) == sns[idx]

    for pn in throw_pns:
        with pytest.raises(ValueError):
            r.pitch_number_to_scale_number(pn)

    for idx, sn in enumerate(sns):
        assert r.scale_number_to_pitch_number(sn) == pns[idx]

    s_letters = ['S', 'R', 'G', 'M', 'P', 'D', 'N'] * 3 + ['S']
    for idx, sn in enumerate(sns):
        assert r.scale_number_to_sargam_letter(sn) == s_letters[idx]

    p_swaras = [
        5, 6,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4,
    ]
    p_octs = [
        -2, -2,
        -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
    ]
    pitches = [Pitch({'swara': s, 'oct': p_octs[idx]}) for idx, s in enumerate(p_swaras)]
    assert r.get_pitches() == pitches

    s_ratios = [
        2 ** 0,
        [2 ** (1 / 12), 2 ** (2 / 12)],
        [2 ** (3 / 12), 2 ** (4 / 12)],
        [2 ** (5 / 12), 2 ** (6 / 12)],
        2 ** (7 / 12),
        [2 ** (8 / 12), 2 ** (9 / 12)],
        [2 ** (10 / 12), 2 ** (11 / 12)],
    ]
    assert r.stratified_ratios == s_ratios
    # Yaman: Pa=True, only raised Ga → all 4 chikari pitches present
    cp = r.chikari_pitches
    assert len(cp) == 4
    assert cp[0].swara == 0 and cp[0].oct == 2  # Sa high
    assert cp[1].swara == 0 and cp[1].oct == 1  # Sa low
    assert cp[2] is not None and cp[2].swara == 4 and cp[2].oct == 1  # Pa
    assert cp[3] is not None and cp[3].swara == 2 and cp[3].raised is True  # Ga raised

    hard_coded_freqs = [
        110.00186456141468, 123.47291821345574,
        130.815, 146.83487284959062,
        164.81657214199782, 185.00034716183643,
        196.0010402616231, 220.00372912282936,
        246.94583642691148, 261.63,
        293.66974569918125, 329.63314428399565,
        370.00069432367286, 392.0020805232462,
        440.0074582456587, 493.89167285382297,
        523.26, 587.3394913983625,
        659.2662885679913, 740.0013886473457,
        784.0041610464924,
    ]
    for idx, freq in enumerate(r.get_frequencies()):
        assert math.isclose(pitches[idx].frequency, freq, abs_tol=1e-6)
        assert math.isclose(hard_coded_freqs[idx], freq, abs_tol=1e-6)

    s_names = ['Sa', 'Re', 'Ga', 'Ma', 'Pa', 'Dha', 'Ni']
    assert r.sargam_names == s_names

    json_obj = {
        'name': 'Yaman',
        'fundamental': 261.63,
        'ratios': base_ratios,
        'tuning': base_tuning,
        'ruleSet': yaman_rule_set,  # PROP-1: ruleSet now serialized
    }
    assert r.to_json() == json_obj


def test_pitch_from_log_freq():
    r = Raga()
    offset = 0.03
    base_log = math.log2(r.fundamental * 2)
    p = r.pitch_from_log_freq(base_log + offset)
    assert isinstance(p, Pitch)
    assert math.isclose(p.frequency, 2 ** (base_log + offset), abs_tol=1e-6)


def test_pitch_string_getters():
    r = Raga()
    pl = r.get_pitches(low=r.fundamental, high=r.fundamental * 1.999)
    solfege = [p.solfege_letter for p in pl]
    pcs = [str(p.chroma) for p in pl]
    western = [p.western_pitch for p in pl]
    assert r.solfege_strings == solfege
    assert r.pc_strings == pcs
    assert r.western_pitch_strings == western


def test_swara_objects():
    r = Raga()
    objs = [
        {'swara': 0, 'raised': True},
        {'swara': 1, 'raised': True},
        {'swara': 2, 'raised': True},
        {'swara': 3, 'raised': True},
        {'swara': 4, 'raised': True},
        {'swara': 5, 'raised': True},
        {'swara': 6, 'raised': True},
    ]
    assert r.swara_objects == objs


def test_ratio_idx_to_tuning_tuple():
    r = Raga()
    mapping = [
        ('sa', None),
        ('re', 'raised'),
        ('ga', 'raised'),
        ('ma', 'raised'),
        ('pa', None),
        ('dha', 'raised'),
        ('ni', 'raised'),
    ]
    for idx, tup in enumerate(mapping):
        assert r.ratio_idx_to_tuning_tuple(idx) == tup


def test_custom_rule_set_pitch_numbers_and_invalid_conversions():
    r = Raga({'rule_set': custom_rule_set, 'fundamental': 200})
    expected = [0, 1, 4, 5, 6, 7, 8, 11]
    assert r.rule_set_num_pitches == len(expected)
    assert r.get_pitch_numbers(0, 11) == expected
    for idx, pn in enumerate(expected):
        assert r.pitch_number_to_scale_number(pn) == idx
    disallowed = [2, 3, 9, 10]
    for pn in disallowed:
        with pytest.raises(ValueError):
            r.pitch_number_to_scale_number(pn)


def test_get_pitches_with_lowered_and_raised_notes():
    r = Raga({'rule_set': custom_rule_set, 'fundamental': 200})
    expected = compute_expected_pitches(r)
    result = r.get_pitches()
    assert len(result) == len(expected)
    for idx, p in enumerate(result):
        assert math.isclose(p.frequency, expected[idx].frequency, abs_tol=1e-6)
        assert p.swara == expected[idx].swara
        assert p.oct == expected[idx].oct
        assert p.raised == expected[idx].raised


def test_pitch_from_log_freq_octave_rounding():
    r = Raga()
    base = math.log2(r.fundamental)
    near = base + 1 - 5e-7
    p1 = r.pitch_from_log_freq(near)
    assert p1.sargam_letter == 'S'
    assert p1.oct == 1
    p2 = r.pitch_from_log_freq(base + 1 + 5e-7)
    assert p2.sargam_letter == 'S'
    assert p2.oct == 1


def test_pitch_from_log_freq_near_exact_octave_offset():
    fundamental = 128.5 - 1e-12
    r = Raga({'fundamental': fundamental})
    log_freq = math.log2(fundamental * 2)
    p = r.pitch_from_log_freq(log_freq)
    assert p.sargam_letter == 'S'
    assert p.oct == 1


def test_ratio_idx_to_tuning_tuple_mixed_rule_set():
    r = Raga({'rule_set': custom_rule_set})
    mapping = [
        ('sa', None),
        ('re', 'lowered'),
        ('ga', 'raised'),
        ('ma', 'lowered'),
        ('ma', 'raised'),
        ('pa', None),
        ('dha', 'lowered'),
        ('ni', 'raised'),
    ]
    for idx, tup in enumerate(mapping):
        assert r.ratio_idx_to_tuning_tuple(idx) == tup


def test_json_round_trip_with_custom_tuning():
    tuning = {
        'sa': 1.01,
        're': {'lowered': 1.02, 'raised': 1.035},
        'ga': {'lowered': 1.04, 'raised': 1.05},
        'ma': {'lowered': 1.06, 'raised': 1.07},
        'pa': 1.08,
        'dha': {'lowered': 1.09, 'raised': 1.1},
        'ni': {'lowered': 1.11, 'raised': 1.12},
    }
    ratios = [
        tuning['sa'],
        tuning['re']['lowered'],
        tuning['ga']['raised'],
        tuning['ma']['lowered'],
        tuning['ma']['raised'],
        tuning['pa'],
        tuning['dha']['lowered'],
        tuning['ni']['raised'],
    ]
    r = Raga({'name': 'Custom', 'fundamental': 300, 'rule_set': custom_rule_set, 'tuning': tuning, 'ratios': ratios})
    json_obj = r.to_json()
    round_trip = Raga.from_json({**json_obj, 'rule_set': custom_rule_set})
    assert round_trip.to_json() == json_obj

# Additional raga utilities
additional_rule_set = {
    'sa': True,
    're': {'lowered': True, 'raised': False},
    'ga': {'lowered': False, 'raised': True},
    'ma': {'lowered': True, 'raised': True},
    'pa': True,
    'dha': {'lowered': False, 'raised': True},
    'ni': {'lowered': True, 'raised': False},
}

fundamental = 200

def et(n: int):
    return 2 ** (n / 12)

expected_ratios = [
    et(0),
    et(1),
    et(4),
    et(5),
    et(6),
    et(7),
    et(9),
    et(10),
]

expected_stratified = [
    et(0),
    [et(1), et(2)],
    [et(3), et(4)],
    [et(5), et(6)],
    et(7),
    [et(8), et(9)],
    [et(10), et(11)],
]

def compute_freqs(r: Raga, low=100, high=800):
    freqs: list[float] = []
    for ratio in expected_ratios:
        base = ratio * r.fundamental
        low_exp = math.ceil(math.log2(low / base))
        high_exp = math.floor(math.log2(high / base))
        for i in range(low_exp, high_exp + 1):
            freqs.append(base * 2 ** i)
    freqs.sort()
    return freqs

mapping_additional = [
    ('sa', None),
    ('re', 'lowered'),
    ('ga', 'raised'),
    ('ma', 'lowered'),
    ('ma', 'raised'),
    ('pa', None),
    ('dha', 'raised'),
    ('ni', 'lowered'),
]

def test_set_ratios_and_stratified_ratios_with_custom_rules():
    r = Raga({'rule_set': additional_rule_set, 'fundamental': fundamental})
    assert r.set_ratios(additional_rule_set) == expected_ratios
    assert len(r.ratios) == len(expected_ratios)
    for idx, ratio in enumerate(expected_ratios):
        assert math.isclose(r.ratios[idx], ratio, abs_tol=1e-6)
    assert len(r.stratified_ratios) == len(expected_stratified)
    for idx, ratio in enumerate(expected_stratified):
        if isinstance(ratio, list):
            assert r.stratified_ratios[idx] == ratio
        else:
            assert math.isclose(r.stratified_ratios[idx], ratio, abs_tol=1e-6)


def test_from_json():
    """Test creating a Raga from JSON data."""
    json_data = {
        "name": "Bhairav",
        "fundamental": 440.0,
        "ruleSet": {
            "sa": True,
            "re": {"lowered": True, "raised": False},
            "ga": {"lowered": False, "raised": True},
            "ma": {"lowered": False, "raised": True},
            "pa": True,
            "dha": {"lowered": True, "raised": False},
            "ni": {"lowered": False, "raised": True}
        }
    }
    
    raga = Raga.from_json(json_data)
    
    assert raga.name == "Bhairav"
    assert raga.fundamental == 440.0
    assert raga.rule_set_num_pitches == 7


def test_rageshree_issue_17():
    """Test for Issue #17: Raga class incorrectly transforms stored ratios.
    
    Rageshree has 6 pitches: S R G m D n (komal Ma, komal Ni, no Pa).
    The stored ratios should be preserved exactly when using from_json().
    
    This test uses a static snapshot of the data structure to ensure
    the Raga class handles it correctly without needing API access.
    """
    # Actual stored data structure from transcription ID: 68bedf76ffd9b2d478ee11f3
    # This represents what comes from the API after get_piece() fetches the ruleSet
    stored_raga_data = {
        'name': 'Rageshree',
        'fundamental': 261.63,
        'ratios': [
            1.0,
            1.122462048309373,
            1.2599210498948732,
            1.3348398541700344,  # komal Ma
            1.681792830507429,   # D
            1.7817974362806785   # komal Ni
        ],
        'ruleSet': {
            'sa': True,
            're': {'lowered': False, 'raised': True},
            'ga': {'lowered': False, 'raised': True},
            'ma': {'lowered': True, 'raised': False},   # Only komal Ma
            'pa': False,                                 # No Pa
            'dha': {'lowered': False, 'raised': True},
            'ni': {'lowered': True, 'raised': False}     # Only komal Ni
        }
    }
    
    # Test loading with from_json (preserve_ratios=True)
    raga = Raga.from_json(stored_raga_data)
    
    # Ratios should be preserved exactly as stored
    assert raga.ratios == stored_raga_data['ratios'], "Ratios should be preserved exactly"
    assert len(raga.ratios) == 6, f"Expected 6 ratios, got {len(raga.ratios)}"
    
    # Verify specific problematic ratios are preserved
    assert abs(raga.ratios[3] - 1.3348398541700344) < 1e-10, "komal Ma ratio should be preserved"
    assert abs(raga.ratios[5] - 1.7817974362806785) < 1e-10, "komal Ni ratio should be preserved"
    
    # Verify no Pa (1.5) is added
    pa_ratio = 1.4983070768766815  # Approximate Pa ratio
    for ratio in raga.ratios:
        assert abs(ratio - pa_ratio) > 0.01, f"Pa (ratio ~{pa_ratio}) should not be present"
    
    # Get pitches and verify correct count in one octave
    pitches = raga.get_pitches(low=261.63, high=523.25)
    assert len(pitches) == 6, f"Expected 6 pitches in octave, got {len(pitches)}"
    
    # Get frequencies and verify correct count
    frequencies = raga.get_frequencies(low=261.63, high=523.25)
    assert len(frequencies) == 6, f"Expected 6 frequencies in octave, got {len(frequencies)}"
    
    # Verify pitch letters are correct (no Pa)
    pitch_letters = [p.sargam_letter for p in pitches]
    assert 'P' not in pitch_letters, "Pa should not be present in pitch letters"
    
    # Verify rule_set has correct number of pitches
    assert raga.rule_set_num_pitches == 6, f"Expected 6 pitches in rule_set, got {raga.rule_set_num_pitches}"


def test_from_json_frequencies_and_helper_mappings():
    r = Raga({'rule_set': additional_rule_set, 'fundamental': fundamental})
    json_obj = r.to_json()
    copy = Raga.from_json({**json_obj, 'rule_set': additional_rule_set})
    assert copy.to_json() == json_obj

    freqs = r.get_frequencies()
    expected_freqs = compute_freqs(r)
    assert len(freqs) == len(expected_freqs)
    for idx, f in enumerate(freqs):
        assert math.isclose(f, expected_freqs[idx], abs_tol=1e-6)

    chosen = freqs[4]
    p = r.pitch_from_log_freq(math.log2(chosen))
    assert isinstance(p, Pitch)
    assert math.isclose(p.frequency, chosen, abs_tol=1e-6)

    for idx, tup in enumerate(mapping_additional):
        assert r.ratio_idx_to_tuning_tuple(idx) == tup


def test_pitch_number_to_scale_number_edge_cases():
    r = Raga({'rule_set': additional_rule_set, 'fundamental': fundamental})
    allowed = r.get_pitch_numbers(0, 11)
    for idx, pn in enumerate(allowed):
        assert r.pitch_number_to_scale_number(pn) == idx
    disallowed = [2, 3, 8, 11]
    for pn in disallowed:
        with pytest.raises(ValueError):
            r.pitch_number_to_scale_number(pn)


def test_raga_conversion_helpers():
    r = Raga({'rule_set': additional_rule_set, 'fundamental': fundamental})
    pitch_numbers = [0, 1, 4, 5, 6, 7, 9, 10]
    letters = ['S', 'r', 'G', 'm', 'M', 'P', 'D', 'n']
    for idx, pn in enumerate(pitch_numbers):
        assert r.pitch_number_to_sargam_letter(pn) == letters[idx]
        assert r.pitch_number_to_scale_number(pn) == idx
        assert r.scale_number_to_pitch_number(idx) == pn
        assert r.scale_number_to_sargam_letter(idx) == letters[idx]
    length = len(pitch_numbers)
    assert r.scale_number_to_pitch_number(length) == pitch_numbers[0] + 12
    assert r.scale_number_to_pitch_number(length + 1) == pitch_numbers[1] + 12
    invalid_pns = [2, 3, 8, 11]
    for pn in invalid_pns:
        assert r.pitch_number_to_sargam_letter(pn) is None
        with pytest.raises(ValueError):
            r.pitch_number_to_scale_number(pn)

# Custom rule set utilities
custom_local_rule_set = {
    'sa': True,
    're': {'lowered': True, 'raised': False},
    'ga': {'lowered': False, 'raised': True},
    'ma': {'lowered': True, 'raised': True},
    'pa': True,
    'dha': {'lowered': True, 'raised': False},
    'ni': {'lowered': False, 'raised': True},
}

expected_ratios_local = [
    2 ** (0 / 12),
    2 ** (1 / 12),
    2 ** (4 / 12),
    2 ** (5 / 12),
    2 ** (6 / 12),
    2 ** (7 / 12),
    2 ** (8 / 12),
    2 ** (11 / 12),
]

ratio_mapping_local = [
    ('sa', True, expected_ratios_local[0]),
    ('re', False, expected_ratios_local[1]),
    ('ga', True, expected_ratios_local[2]),
    ('ma', False, expected_ratios_local[3]),
    ('ma', True, expected_ratios_local[4]),
    ('pa', True, expected_ratios_local[5]),
    ('dha', False, expected_ratios_local[6]),
    ('ni', True, expected_ratios_local[7]),
]

def compute_expected_pitches_local(r: Raga, low=100, high=800):
    pitches: list[Pitch] = []
    for swara, raised, ratio in ratio_mapping_local:
        freq = ratio * r.fundamental
        low_exp = math.ceil(math.log2(low / freq))
        high_exp = math.floor(math.log2(high / freq))
        for i in range(low_exp, high_exp + 1):
            pitches.append(Pitch({'swara': swara, 'oct': i, 'raised': raised,
                                  'fundamental': r.fundamental, 'ratios': r.stratified_ratios}))
    pitches.sort(key=lambda p: p.frequency)
    return [p for p in pitches if low <= p.frequency <= high]

def compute_expected_freqs_local(r: Raga, low=100, high=800):
    freqs: list[float] = []
    for ratio in expected_ratios_local:
        base = ratio * r.fundamental
        low_exp = math.ceil(math.log2(low / base))
        high_exp = math.floor(math.log2(high / base))
        for i in range(low_exp, high_exp + 1):
            freqs.append(base * 2 ** i)
    freqs.sort()
    return freqs

pitch_numbers_single_oct = [0, 1, 4, 5, 6, 7, 8, 11]

def test_custom_rule_set_basic_functions():
    r = Raga({'rule_set': custom_local_rule_set, 'fundamental': fundamental})
    assert r.rule_set_num_pitches == 8
    for idx, pn in enumerate(pitch_numbers_single_oct):
        assert r.get_pitch_numbers(0, 11)[idx] == pn
    for idx, ratio in enumerate(expected_ratios_local):
        assert math.isclose(r.ratios[idx], ratio, abs_tol=1e-6)
    from_set = r.set_ratios(custom_local_rule_set)
    for idx, ratio in enumerate(from_set):
        assert math.isclose(r.ratios[idx], ratio, abs_tol=1e-6)


def test_get_pitches_frequencies_and_flags():
    r = Raga({'rule_set': custom_local_rule_set, 'fundamental': fundamental})
    expected = compute_expected_pitches_local(r)
    result = r.get_pitches()
    assert len(result) == len(expected)
    for idx, p in enumerate(result):
        assert math.isclose(p.frequency, expected[idx].frequency, abs_tol=1e-6)
        assert p.swara == expected[idx].swara
        assert p.oct == expected[idx].oct
        assert p.raised == expected[idx].raised


def test_frequency_helpers_with_custom_rule_set():
    r = Raga({'rule_set': custom_local_rule_set, 'fundamental': fundamental})
    expected_freqs = compute_expected_freqs_local(r)
    freqs = r.get_frequencies()
    assert len(freqs) == len(expected_freqs)
    for idx, f in enumerate(freqs):
        assert math.isclose(f, expected_freqs[idx], abs_tol=1e-6)

    pick_freq = freqs[3]
    p = r.pitch_from_log_freq(math.log2(pick_freq))
    assert math.isclose(p.frequency, pick_freq, abs_tol=1e-6)

    mapping = [
        ('sa', None),
        ('re', 'lowered'),
        ('ga', 'raised'),
        ('ma', 'lowered'),
        ('ma', 'raised'),
        ('pa', None),
        ('dha', 'lowered'),
        ('ni', 'raised'),
    ]
    for idx, tup in enumerate(mapping):
        assert r.ratio_idx_to_tuning_tuple(idx) == tup


def test_model_raga_string_getters():
    r = Raga()
    pl = r.get_pitches(low=r.fundamental, high=r.fundamental * 1.999)
    solfege = [p.solfege_letter for p in pl]
    pcs = [str(p.chroma) for p in pl]
    western = [p.western_pitch for p in pl]
    assert r.solfege_strings == solfege
    assert r.pc_strings == pcs
    assert r.western_pitch_strings == western


# Parameter validation tests
def test_invalid_parameter_rules_instead_of_rule_set():
    """Test that 'rules' parameter raises helpful error message."""
    with pytest.raises(ValueError, match="Parameter 'rules' not supported. Did you mean 'rule_set'?"):
        Raga({'name': 'Bhimpalasi', 'fundamental': 157.6, 'rules': custom_rule_set})


def test_invalid_parameter_multiple_errors():
    """Test multiple invalid parameters with helpful suggestions."""
    with pytest.raises(ValueError) as exc_info:
        Raga({'name': 'Test', 'rules': custom_rule_set, 'fundamental_freq': 200, 'invalid_param': 123})
    
    error_msg = str(exc_info.value)
    assert "Parameter 'rules' not supported. Did you mean 'rule_set'?" in error_msg
    assert "Parameter 'fundamental_freq' not supported. Did you mean 'fundamental'?" in error_msg
    assert "Invalid parameter: 'invalid_param'" in error_msg
    assert "Allowed parameters:" in error_msg


def test_invalid_parameter_raga_name():
    """Test that 'raga_name' parameter raises helpful error message."""
    with pytest.raises(ValueError, match="Parameter 'raga_name' not supported. Did you mean 'name'?"):
        Raga({'raga_name': 'Test', 'fundamental': 200})


def test_invalid_parameter_base_freq():
    """Test that 'base_freq' parameter raises helpful error message."""
    with pytest.raises(ValueError, match="Parameter 'base_freq' not supported. Did you mean 'fundamental'?"):
        Raga({'name': 'Test', 'base_freq': 200})


def test_invalid_parameter_generic():
    """Test generic invalid parameter error."""
    with pytest.raises(ValueError) as exc_info:
        Raga({'name': 'Test', 'unknown_param': 123, 'another_invalid': 456})
    
    error_msg = str(exc_info.value)
    assert "Invalid parameter: 'unknown_param'" in error_msg
    assert "Invalid parameter: 'another_invalid'" in error_msg
    assert "Allowed parameters:" in error_msg


def test_valid_parameters_work():
    """Test that all valid parameters work correctly."""
    # This should not raise any errors
    raga_data = {
        'name': 'Test Raga',
        'fundamental': 200.0,
        'rule_set': custom_rule_set,
        'tuning': et_tuning,
        'ratios': [1.0, 1.1, 1.2]
    }
    r = Raga(raga_data)
    assert r.name == 'Test Raga'
    assert r.fundamental == 200.0


def test_empty_options_dict():
    """Test that empty options dict works (uses defaults)."""
    r = Raga({})
    assert r.name == 'Yaman'
    assert r.fundamental == 261.63


def test_none_options():
    """Test that None options work (uses defaults)."""
    r = Raga(None)
    assert r.name == 'Yaman'
    assert r.fundamental == 261.63


def test_camel_case_conversion():
    """Test that camelCase parameters are converted properly."""
    # This should work because humps.decamelize converts camelCase
    raga_data = {
        'name': 'Test',
        'fundamental': 200.0,
        'ruleSet': custom_rule_set  # camelCase version
    }
    r = Raga(raga_data)
    assert r.name == 'Test'
    assert r.fundamental == 200.0
    assert r.rule_set == custom_rule_set


def test_validation_issue_reproduction():
    """Reproduce the exact issue from the bug report."""
    # This was the problematic code that should now raise an error
    raga_data = {
        'name': 'Bhimpalasi',
        'fundamental': 157.6,
        'rules': {'sa': True, 'ga': {'lowered': True, 'raised': False}}  # Wrong parameter name
    }
    
    with pytest.raises(ValueError, match="Parameter 'rules' not supported. Did you mean 'rule_set'?"):
        Raga(raga_data)


def test_correct_usage_after_fix():
    """Test that the corrected usage works properly."""
    # This is the corrected version that should work (complete rule set)
    raga_data = {
        'name': 'Bhimpalasi',
        'fundamental': 157.6,
        'rule_set': {  # Correct parameter name with complete rule set
            'sa': True,
            're': {'lowered': False, 'raised': True},
            'ga': {'lowered': True, 'raised': False},  # komal ga (characteristic of Bhimpalasi)
            'ma': {'lowered': False, 'raised': True},
            'pa': True,
            'dha': {'lowered': True, 'raised': False},  # komal dha (characteristic of Bhimpalasi)
            'ni': {'lowered': True, 'raised': False}    # komal ni (characteristic of Bhimpalasi)
        }
    }
    
    raga = Raga(raga_data)
    assert raga.name == 'Bhimpalasi'
    assert raga.fundamental == 157.6
    # Test that the rule set was applied correctly for Bhimpalasi characteristics
    assert raga.rule_set['sa'] == True
    assert raga.rule_set['ga']['lowered'] == True   # komal ga
    assert raga.rule_set['ga']['raised'] == False
    assert raga.rule_set['dha']['lowered'] == True  # komal dha  
    assert raga.rule_set['ni']['lowered'] == True   # komal ni


# Comprehensive validation tests for type errors
def test_invalid_type_name():
    """Test that non-string name raises TypeError."""
    with pytest.raises(TypeError, match="Parameter 'name' must be a string"):
        Raga({'name': 123})


def test_invalid_type_fundamental():
    """Test that non-numeric fundamental raises TypeError."""
    with pytest.raises(TypeError, match="Parameter 'fundamental' must be a number"):
        Raga({'fundamental': 'not_a_number'})


def test_invalid_type_rule_set():
    """Test that non-dict rule_set raises TypeError."""
    with pytest.raises(TypeError, match="Parameter 'rule_set' must be a dict"):
        Raga({'rule_set': 'not_a_dict'})


def test_invalid_type_tuning():
    """Test that non-dict tuning raises TypeError."""
    with pytest.raises(TypeError, match="Parameter 'tuning' must be a dict"):
        Raga({'tuning': [1, 2, 3]})


def test_invalid_type_ratios():
    """Test that non-list ratios raises TypeError."""
    with pytest.raises(TypeError, match="Parameter 'ratios' must be a list"):
        Raga({'ratios': 'not_a_list'})


def test_invalid_ratios_non_numeric():
    """Test that non-numeric items in ratios raises TypeError."""
    with pytest.raises(TypeError, match="All items in 'ratios' must be numbers"):
        Raga({'ratios': [1.0, 2.0, 'not_a_number']})


# Value validation tests
def test_negative_fundamental():
    """Test that negative fundamental raises ValueError."""
    with pytest.raises(ValueError, match="Parameter 'fundamental' must be positive"):
        Raga({'fundamental': -100})


def test_zero_fundamental():
    """Test that zero fundamental raises ValueError."""
    with pytest.raises(ValueError, match="Parameter 'fundamental' must be positive"):
        Raga({'fundamental': 0})


def test_fundamental_out_of_range_warning():
    """Test that fundamental frequency outside typical range raises warning."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Raga({'fundamental': 50000})  # Very high frequency
        assert len(w) == 1
        assert "outside typical range" in str(w[0].message)


def test_negative_ratios():
    """Test that negative ratios raise ValueError."""
    with pytest.raises(ValueError, match="All ratios must be positive"):
        Raga({'ratios': [1.0, -2.0, 1.5]})


def test_too_many_ratios():
    """Test that too many ratios raise ValueError."""
    with pytest.raises(ValueError, match="Too many ratios"):
        Raga({'ratios': [1.0] * 13})  # 13 ratios, max is 12


# rule_set structure validation tests
def test_rule_set_missing_swaras():
    """Test that rule_set missing required swaras raises ValueError."""
    incomplete_rule_set = {'sa': True, 're': True}  # Missing other swaras
    with pytest.raises(ValueError, match="rule_set missing required swaras"):
        Raga({'rule_set': incomplete_rule_set})


def test_rule_set_invalid_swaras():
    """Test that rule_set with invalid swaras raises ValueError."""
    invalid_rule_set = {
        'sa': True, 're': True, 'ga': True, 'ma': True, 
        'pa': True, 'dha': True, 'ni': True,
        'invalid_swara': True  # Invalid swara
    }
    with pytest.raises(ValueError, match="rule_set contains invalid swaras"):
        Raga({'rule_set': invalid_rule_set})


def test_rule_set_sa_pa_non_boolean():
    """Test that sa/pa with non-boolean values raise TypeError."""
    invalid_rule_set = {
        'sa': 'not_boolean', 're': True, 'ga': True, 'ma': True,
        'pa': True, 'dha': True, 'ni': True
    }
    with pytest.raises(TypeError, match="rule_set\\['sa'\\] must be boolean"):
        Raga({'rule_set': invalid_rule_set})


def test_rule_set_variable_swara_invalid_type():
    """Test that variable swaras with invalid types raise TypeError."""
    invalid_rule_set = {
        'sa': True, 're': 'invalid_type', 'ga': True, 'ma': True,
        'pa': True, 'dha': True, 'ni': True
    }
    with pytest.raises(TypeError, match="rule_set\\['re'\\] must be boolean or dict"):
        Raga({'rule_set': invalid_rule_set})


def test_rule_set_variable_swara_missing_keys():
    """Test that variable swaras missing lowered/raised keys raise ValueError."""
    invalid_rule_set = {
        'sa': True, 're': {'only_lowered': True}, 'ga': True, 'ma': True,
        'pa': True, 'dha': True, 'ni': True
    }
    with pytest.raises(ValueError, match="rule_set\\['re'\\] missing required keys"):
        Raga({'rule_set': invalid_rule_set})


def test_rule_set_variable_swara_invalid_keys():
    """Test that variable swaras with invalid keys raise ValueError."""
    invalid_rule_set = {
        'sa': True, 
        're': {'lowered': True, 'raised': False, 'invalid_key': True}, 
        'ga': True, 'ma': True, 'pa': True, 'dha': True, 'ni': True
    }
    with pytest.raises(ValueError, match="rule_set\\['re'\\] contains invalid keys"):
        Raga({'rule_set': invalid_rule_set})


def test_rule_set_variable_swara_non_boolean_values():
    """Test that variable swaras with non-boolean values raise TypeError."""
    invalid_rule_set = {
        'sa': True, 
        're': {'lowered': 'not_boolean', 'raised': True}, 
        'ga': True, 'ma': True, 'pa': True, 'dha': True, 'ni': True
    }
    with pytest.raises(TypeError, match="All values in rule_set\\['re'\\] must be boolean"):
        Raga({'rule_set': invalid_rule_set})


# tuning structure validation tests
def test_tuning_missing_swaras():
    """Test that tuning missing required swaras raises ValueError."""
    incomplete_tuning = {'sa': 1.0, 're': 1.1}  # Missing other swaras
    with pytest.raises(ValueError, match="tuning missing required swaras"):
        Raga({'tuning': incomplete_tuning})


def test_tuning_invalid_swaras():
    """Test that tuning with invalid swaras raises ValueError."""
    invalid_tuning = {
        'sa': 1.0, 're': 1.1, 'ga': 1.2, 'ma': 1.3, 
        'pa': 1.5, 'dha': 1.6, 'ni': 1.8,
        'invalid_swara': 2.0  # Invalid swara
    }
    with pytest.raises(ValueError, match="tuning contains invalid swaras"):
        Raga({'tuning': invalid_tuning})


def test_tuning_sa_pa_non_numeric():
    """Test that sa/pa with non-numeric values raise TypeError."""
    invalid_tuning = {
        'sa': 'not_numeric', 're': 1.1, 'ga': 1.2, 'ma': 1.3,
        'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(TypeError, match="tuning\\['sa'\\] must be a number"):
        Raga({'tuning': invalid_tuning})


def test_tuning_sa_pa_negative():
    """Test that sa/pa with negative values raise ValueError."""
    invalid_tuning = {
        'sa': -1.0, 're': 1.1, 'ga': 1.2, 'ma': 1.3,
        'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(ValueError, match="tuning\\['sa'\\] must be positive"):
        Raga({'tuning': invalid_tuning})


def test_tuning_variable_swara_negative_numeric():
    """Test that variable swaras with negative numeric values raise ValueError."""
    invalid_tuning = {
        'sa': 1.0, 're': -1.1, 'ga': 1.2, 'ma': 1.3,
        'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(ValueError, match="tuning\\['re'\\] must be positive"):
        Raga({'tuning': invalid_tuning})


def test_tuning_variable_swara_invalid_type():
    """Test that variable swaras with invalid types raise TypeError."""
    invalid_tuning = {
        'sa': 1.0, 're': 'invalid_type', 'ga': 1.2, 'ma': 1.3,
        'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(TypeError, match="tuning\\['re'\\] must be number or dict"):
        Raga({'tuning': invalid_tuning})


def test_tuning_variable_swara_dict_missing_keys():
    """Test that variable swaras dicts missing keys raise ValueError."""
    invalid_tuning = {
        'sa': 1.0, 're': {'only_lowered': 1.05}, 'ga': 1.2, 'ma': 1.3,
        'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(ValueError, match="tuning\\['re'\\] missing required keys"):
        Raga({'tuning': invalid_tuning})


def test_tuning_variable_swara_dict_invalid_keys():
    """Test that variable swaras dicts with invalid keys raise ValueError."""
    invalid_tuning = {
        'sa': 1.0, 
        're': {'lowered': 1.05, 'raised': 1.1, 'invalid_key': 1.2}, 
        'ga': 1.2, 'ma': 1.3, 'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(ValueError, match="tuning\\['re'\\] contains invalid keys"):
        Raga({'tuning': invalid_tuning})


def test_tuning_variable_swara_dict_non_numeric_values():
    """Test that variable swaras dicts with non-numeric values raise TypeError."""
    invalid_tuning = {
        'sa': 1.0, 
        're': {'lowered': 'not_numeric', 'raised': 1.1}, 
        'ga': 1.2, 'ma': 1.3, 'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(TypeError, match="All values in tuning\\['re'\\] must be numbers"):
        Raga({'tuning': invalid_tuning})


def test_tuning_variable_swara_dict_negative_values():
    """Test that variable swaras dicts with negative values raise ValueError."""
    invalid_tuning = {
        'sa': 1.0, 
        're': {'lowered': -1.05, 'raised': 1.1}, 
        'ga': 1.2, 'ma': 1.3, 'pa': 1.5, 'dha': 1.6, 'ni': 1.8
    }
    with pytest.raises(ValueError, match="All values in tuning\\['re'\\] must be positive"):
        Raga({'tuning': invalid_tuning})


# Test comprehensive validation - everything should work with valid complete data
def test_complete_valid_data():
    """Test that comprehensive valid data works without errors."""
    valid_data = {
        'name': 'Test Raga',
        'fundamental': 220.0,
        'rule_set': {
            'sa': True,
            're': {'lowered': True, 'raised': False},
            'ga': {'lowered': False, 'raised': True},
            'ma': {'lowered': True, 'raised': True},
            'pa': True,
            'dha': {'lowered': False, 'raised': True},
            'ni': {'lowered': True, 'raised': False}
        },
        'tuning': {
            'sa': 1.0,
            're': {'lowered': 1.05, 'raised': 1.1},
            'ga': {'lowered': 1.15, 'raised': 1.2},
            'ma': {'lowered': 1.3, 'raised': 1.35},
            'pa': 1.5,
            'dha': {'lowered': 1.6, 'raised': 1.65},
            'ni': {'lowered': 1.75, 'raised': 1.8}
        },
        'ratios': [1.0, 1.05, 1.2, 1.3, 1.35, 1.5, 1.6, 1.75]
    }
    
    # This should work without any errors
    raga = Raga(valid_data)
    assert raga.name == 'Test Raga'
    assert raga.fundamental == 220.0


# Chikari pitch derivation tests

def test_chikari_pitches_no_pa():
    """Raga without Pa should have None for Pa chikari string."""
    rule_set = {
        'sa': True,
        're': {'lowered': False, 'raised': True},
        'ga': {'lowered': False, 'raised': True},
        'ma': {'lowered': True, 'raised': False},
        'pa': False,
        'dha': {'lowered': False, 'raised': True},
        'ni': {'lowered': True, 'raised': False},
    }
    r = Raga({'rule_set': rule_set})
    cp = r.chikari_pitches
    assert len(cp) == 4
    assert cp[0] is not None  # Sa always present
    assert cp[1] is not None  # Sa always present
    assert cp[2] is None  # No Pa in raga
    assert cp[3] is not None  # Only raised Ga → present
    assert cp[3].raised is True


def test_chikari_pitches_both_ga_variants():
    """Raga with both Ga variants should have None for Ga chikari (ambiguous)."""
    rule_set = {
        'sa': True,
        're': {'lowered': False, 'raised': True},
        'ga': {'lowered': True, 'raised': True},
        'ma': {'lowered': True, 'raised': False},
        'pa': True,
        'dha': {'lowered': False, 'raised': True},
        'ni': {'lowered': False, 'raised': True},
    }
    r = Raga({'rule_set': rule_set})
    cp = r.chikari_pitches
    assert cp[2] is not None  # Pa present
    assert cp[3] is None  # Both Ga variants → ambiguous → None


def test_chikari_pitches_no_ga():
    """Raga with neither Ga variant should have None for Ga chikari."""
    rule_set = {
        'sa': True,
        're': {'lowered': False, 'raised': True},
        'ga': {'lowered': False, 'raised': False},
        'ma': {'lowered': True, 'raised': False},
        'pa': True,
        'dha': {'lowered': False, 'raised': True},
        'ni': {'lowered': False, 'raised': True},
    }
    r = Raga({'rule_set': rule_set})
    cp = r.chikari_pitches
    assert cp[3] is None  # Neither Ga variant → None


def test_chikari_pitches_include_ratios():
    """Verify chikari pitches use stratified_ratios for correct tuning."""
    r = Raga({'fundamental': 200.0})
    cp = r.chikari_pitches
    # All non-None pitches should have proper frequencies
    for p in cp:
        if p is not None:
            assert p.frequency > 0


# ---------------------------------------------------------------------------
# stratified_ratios under rule-set/ratios count mismatch
# (port of the TS regression tests, src/ts/tests/raga.test.ts — idtap #24/#25/#26)
#
# Regression: transcription 63445d13 ("Mushtaq Ali Khan - Yaman", saved 2022)
# stored 7 ratios + a synced tuning block. The editor grafts the raga's
# CURRENT db rule set onto the saved raga, and Yaman's db rules later gained
# lowered ma (8 pitches). Positional mapping would misassign everything after
# ga and run off the end.
# ---------------------------------------------------------------------------

mismatch_fundamental = 248.8843372467667
mismatch_stored_ratios = [
    1, 1.122462048309373, 1.2591935011032134, 1.4142135623730951,
    1.4983070768766815, 1.6817928305074292, 1.8790454984280236,
]
mismatch_stored_tuning = {
    'sa': 1,
    're': {'lowered': 1.0594630943592953, 'raised': 1.122462048309373},
    'ga': {'lowered': 1.189207115002721, 'raised': 1.2591935011032134},
    'ma': {'lowered': 1.3348398541700344, 'raised': 1.4142135623730951},
    'pa': 1.4983070768766815,
    'dha': {'lowered': 1.5874010519681994, 'raised': 1.6817928305074292},
    'ni': {'lowered': 1.7817974362806785, 'raised': 1.8790454984280236},
}
mismatch_grafted_rule_set = {
    'sa': True,
    're': {'lowered': False, 'raised': True},
    'ga': {'lowered': False, 'raised': True},
    'ma': {'lowered': True, 'raised': True},  # gained lowered ma since save
    'pa': True,
    'dha': {'lowered': False, 'raised': True},
    'ni': {'lowered': False, 'raised': True},
}


def make_mismatch_raga() -> Raga:
    # from_json preserves transcription ratios verbatim (PROP-1) and warns
    # about the count mismatch.
    with pytest.warns(UserWarning):
        return Raga.from_json({
            'name': 'Yaman',
            'fundamental': mismatch_fundamental,
            'ratios': list(mismatch_stored_ratios),
            'tuning': mismatch_stored_tuning,
            'ruleSet': mismatch_grafted_rule_set,
        })


def make_aligned_raga() -> Raga:
    return Raga.from_json({
        'name': 'Yaman',
        'fundamental': mismatch_fundamental,
        'ratios': list(mismatch_stored_ratios),
        'tuning': mismatch_stored_tuning,
    })


def test_mismatch_falls_back_to_tuning_no_none_frequencies_preserved():
    raga = make_mismatch_raga()

    # PROP-1: stored ratios preserved verbatim
    assert raga.ratios == mismatch_stored_ratios

    flat = []
    for entry in raga.stratified_ratios:
        if isinstance(entry, list):
            flat.extend(entry)
        else:
            flat.append(entry)
    assert all(v is not None for v in flat)

    # existing variants keep their saved values (tuning was synced from the
    # same ratios at save time)
    sr = raga.stratified_ratios
    assert sr[0] == 1
    assert sr[1][1] == mismatch_stored_ratios[1]  # re raised
    assert sr[3][1] == mismatch_stored_ratios[3]  # ma raised (tivra)
    assert sr[4] == mismatch_stored_ratios[4]     # pa
    assert sr[6][1] == mismatch_stored_ratios[6]  # ni raised
    # the newly-enabled variant gets its tuning value
    assert sr[3][0] == mismatch_stored_tuning['ma']['lowered']

    # and Pitch construction no longer misbehaves
    p = Pitch({'swara': 'ma', 'raised': True,
               'fundamental': mismatch_fundamental, 'ratios': sr})
    assert math.isclose(
        p.frequency, mismatch_stored_ratios[3] * mismatch_fundamental,
        abs_tol=1e-6,
    )


def test_aligned_counts_keep_exact_positional_behavior():
    raga = make_aligned_raga()
    sr = raga.stratified_ratios
    assert sr[0] == mismatch_stored_ratios[0]
    assert sr[3][1] == mismatch_stored_ratios[3]
    flat = []
    for entry in sr:
        if isinstance(entry, list):
            flat.extend(entry)
        else:
            flat.append(entry)
    assert all(v is not None for v in flat)


def test_get_frequencies_pairs_1_to_1_with_get_pitches():
    # y-axis consumers zip frequencies with pitch-derived labels by index
    mismatched = make_mismatch_raga()
    aligned = make_aligned_raga()
    for raga in (mismatched, aligned):
        freqs = raga.get_frequencies(low=100, high=800)
        pitches = raga.get_pitches(low=100, high=800)
        assert len(freqs) == len(pitches)
        for f, p in zip(freqs, pitches):
            assert math.isclose(f, p.frequency, abs_tol=1e-10)
    # the mismatch raga's octave now genuinely has 8 degrees (komal ma added)
    in_octave = mismatched.get_frequencies(
        low=mismatch_fundamental, high=mismatch_fundamental * 1.999)
    assert len(in_octave) == 8


def test_pitch_from_log_freq_assigns_correct_swara_on_mismatch_ragas():
    raga = make_mismatch_raga()
    # used to index self.ratios (7) against the rule-set walk (8): drawing a
    # tivra ma came back labeled komal ma, pa came back as tivra ma, etc.
    cases = [
        (mismatch_stored_tuning['ma']['raised'], 3, True),   # tivra ma
        (mismatch_stored_tuning['ma']['lowered'], 3, False), # komal ma
        (mismatch_stored_tuning['pa'], 4, True),
        (mismatch_stored_tuning['ni']['raised'], 6, True),
        (1, 0, True),                                        # sa
    ]
    for ratio, swara, raised in cases:
        p = raga.pitch_from_log_freq(math.log2(ratio * mismatch_fundamental))
        assert (p.swara, p.raised) == (swara, raised)
        assert math.isclose(
            p.frequency, ratio * mismatch_fundamental, abs_tol=1e-6)
    # octaves + log_offset still behave
    up = raga.pitch_from_log_freq(
        math.log2(2 * mismatch_fundamental) + 0.01)
    assert up.swara == 0
    assert up.oct == 1
    assert math.isclose(up.log_offset, 0.01, abs_tol=1e-8)
