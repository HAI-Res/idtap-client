"""Vibrato v2 (idtap-contract PROP-6 / PROP-6b).

- the v2 curve matches the TypeScript reference golden (values at 21 and 201
  points, and the emitted wire vibObj) to 1e-9 relative;
- the v1 -> v2 heal is lossless: for phase in {0, pi} and integer P the v2
  curve equals the old v1 curve at 201 points to 1e-12 (the v1 id13 is kept
  here, and only here, as the reference);
- stored v1 fields may be strings / non-integer and are coerced, not truncated;
- validation: v2 keys strict, v1 keys accepted, rate > 0, extents >= 0;
- PROP-6b: vibObj is emitted only for id 13, accepted and ignored elsewhere;
- decompose_trajectory's cosine-chain view (``vibrato_as_cosines=True``)
  reproduces the v2 curve chunk-by-chunk; the default single ``vibrato``
  chunk is covered in simple_trajectory_test.py.
"""
import json
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath('.'))

import pytest

from idtap.classes.pitch import Pitch
from idtap.classes.piece import Piece
from idtap.classes.trajectory import Trajectory, default_vib_obj
from idtap.classes.simple_trajectory import decompose_trajectory
# rootdir-relative import, same convention as the other test modules (the
# tests directory is not a package)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simple_trajectory_test import assert_matches_compute, chunk_freq_at  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _v1_id13(traj: Trajectory, vib: dict, x: float) -> float:
    """The pre-PROP-6 (v1) id13 curve, verbatim from idtap <= 0.1.54, kept only
    as the reference for the lossless-heal proof."""
    periods = vib['periods']
    vert_offset = vib['vert_offset']
    init_up = vib['init_up']
    extent = vib['extent']
    if abs(vert_offset) > extent / 2:
        vert_offset = math.copysign(extent / 2, vert_offset)
    out = math.cos(x * 2 * math.pi * periods + int(init_up) * math.pi)
    if x < 1 / (2 * periods):
        start = traj.log_freqs[0]
        end = math.log2(_v1_id13(traj, vib, 1 / (2 * periods)))
        middle = (end + start) / 2
        ext = abs(end - start) / 2
        return 2 ** (out * ext + middle)
    elif x > 1 - 1 / (2 * periods):
        start = math.log2(_v1_id13(traj, vib, 1 - 1 / (2 * periods)))
        end = traj.log_freqs[0]
        middle = (end + start) / 2
        ext = abs(end - start) / 2
        return 2 ** (out * ext + middle)
    else:
        return 2 ** (out * extent / 2 + vert_offset + traj.log_freqs[0])


def _vib13(vib_obj, dur_tot=1.0, **extra):
    return Trajectory({'id': 13, 'pitches': [Pitch()], 'dur_tot': dur_tot,
                       'vib_obj': vib_obj, **extra})


# ----------------------------------------------------------------- golden

def _golden():
    with open(os.path.join(FIXTURES, 'vibrato-v2-golden.json')) as f:
        return json.load(f)


GOLDEN = _golden()


@pytest.mark.parametrize('case', GOLDEN['cases'], ids=lambda c: c['name'])
def test_matches_typescript_reference_golden(case):
    """Curve values (21 and 201 points) and emitted vibObj match the TS
    reference implementation to 1e-9 relative."""
    t = Trajectory.from_json(case['input'], ratios=GOLDEN['ratios'],
                             fundamental=GOLDEN['fundamental'])
    assert t.log_freqs[0] == pytest.approx(case['logFreqs'][0], rel=1e-12)
    for xs_key, vals_key in (('xs21', 'values21'), ('xs201', 'values201')):
        for x, expected in zip(GOLDEN[xs_key], case[vals_key]):
            assert t.compute(x) == pytest.approx(expected, rel=1e-9), (case['name'], x)
    got = t.to_json()['vibObj']
    assert set(got) == set(case['json']['vibObj'])
    for k, v in case['json']['vibObj'].items():
        assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-15), k


# ----------------------------------------------------------- lossless heal

@pytest.mark.parametrize('periods', [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize('init_up', [True, False])
@pytest.mark.parametrize('dur_tot', [0.37, 1.0, 2.5])
@pytest.mark.parametrize('vert_offset', [0.0, 0.01, -0.5])
def test_heal_is_lossless_against_v1_curve(periods, init_up, dur_tot, vert_offset):
    """For phase in {0, pi} and integer P, v2 == v1 at 201 points to 1e-12."""
    vib = {'periods': periods, 'vert_offset': vert_offset,
           'init_up': init_up, 'extent': 0.06}
    t = _vib13(vib, dur_tot, unique_id='heal')
    assert t.vib_obj == {
        'rate': periods / dur_tot, 'extent_start': 0.06, 'extent_end': 0.06,
        'vert_offset': vert_offset, 'phase': math.pi if init_up else 0.0,
    }
    for k in range(201):
        x = k / 200
        assert t.id13(x) == pytest.approx(_v1_id13(t, vib, x), rel=1e-12), x


def test_heal_coerces_string_fields_and_keeps_noninteger_periods():
    """Real stored data (Babul Mora) carries the v1 fields as strings and a
    non-integer periods; the heal uses float(), never int()."""
    t = _vib13({'periods': '3.5', 'vertOffset': '0.0209', 'initUp': 'true',
                'extent': '0.055'}, dur_tot=0.474)
    assert t.vib_obj['rate'] == pytest.approx(3.5 / 0.474)
    assert t.vib_obj['extent_start'] == 0.055
    assert t.vib_obj['extent_end'] == 0.055
    assert t.vib_obj['vert_offset'] == 0.0209
    assert t.vib_obj['phase'] == math.pi
    assert all(isinstance(v, float) for v in t.vib_obj.values())
    # bool-like init_up spellings
    for spelled, phase in (('false', 0.0), ('0', 0.0), (0, 0.0), (1, math.pi),
                           (False, 0.0), (True, math.pi)):
        assert _vib13({'periods': 2, 'init_up': spelled}).vib_obj['phase'] == phase


def test_heal_fills_missing_v1_keys_with_old_defaults():
    t = _vib13({'periods': 4}, dur_tot=2.0)
    assert t.vib_obj == {'rate': 2.0, 'extent_start': 0.05, 'extent_end': 0.05,
                         'vert_offset': 0.0, 'phase': math.pi}


def test_heal_changes_dur_tot_semantics():
    """After migration, changing dur_tot keeps the rate and changes the cycle
    count (v1 kept the count)."""
    t = _vib13({'periods': 4}, dur_tot=2.0)
    assert t.vib_obj['rate'] == 2.0
    t.dur_tot = 4.0
    assert t.vib_obj['rate'] == 2.0
    assert len(t.vib_breakpoints()) - 1 == 16  # 8 cycles -> 16 half periods


# --------------------------------------------------------------- validation

def test_default_vib_obj():
    assert Trajectory().vib_obj == default_vib_obj()
    assert default_vib_obj() == {'rate': 5.5, 'extent_start': 0.05,
                                 'extent_end': 0.05, 'vert_offset': 0.0,
                                 'phase': math.pi}


def test_v2_input_accepted_and_coerced():
    t = _vib13({'rate': '6', 'extentStart': 0, 'extentEnd': '0.08',
                'vertOffset': -0.01, 'phase': 1.3})
    assert t.vib_obj == {'rate': 6.0, 'extent_start': 0.0, 'extent_end': 0.08,
                         'vert_offset': -0.01, 'phase': 1.3}


def test_v2_missing_keys_take_defaults():
    t = _vib13({'rate': 3})
    assert t.vib_obj == {**default_vib_obj(), 'rate': 3.0}


def test_v2_unknown_keys_rejected():
    with pytest.raises(ValueError, match='invalid keys'):
        _vib13({'rate': 5, 'extent': 0.05})
    with pytest.raises(ValueError, match='invalid keys'):
        _vib13({'rate': 5, 'init_up': True})
    with pytest.raises(ValueError, match='invalid keys'):
        _vib13({'rate': 5, 'bogus': 1})


def test_v1_unknown_keys_rejected():
    with pytest.raises(ValueError, match='invalid keys'):
        _vib13({'periods': 5, 'rate': 5})
    with pytest.raises(ValueError, match='invalid keys'):
        _vib13({'periods': 5, 'bogus': 1})


@pytest.mark.parametrize('bad', [
    {'rate': 0}, {'rate': -1}, {'rate': '0'},
    {'extentStart': -0.01}, {'extentEnd': -1},
    {'periods': 0}, {'periods': '-2'}, {'periods': 2, 'extent': -0.1},
])
def test_out_of_range_values_rejected_on_id13(bad):
    with pytest.raises(ValueError):
        _vib13(bad)


@pytest.mark.parametrize('bad', [
    {'rate': 'fast'}, {'rate': None}, {'rate': True}, {'phase': [1]},
    {'periods': 'many'}, {'periods': 2, 'init_up': 'maybe'},
    {'periods': 2, 'init_up': 2},
])
def test_uncoercible_types_rejected_on_id13(bad):
    with pytest.raises(TypeError):
        _vib13(bad)


def test_zero_extent_allowed():
    t = _vib13({'rate': 5, 'extentStart': 0, 'extentEnd': 0})
    for x in (0, 0.3, 0.77, 1):
        assert t.compute(x) == pytest.approx(t.freqs[0], rel=1e-12)
    # v1 form too: a flat vibrato is not a crash
    assert _vib13({'periods': 3, 'extent': 0}).vib_obj['extent_end'] == 0.0


def test_vib_obj_must_be_dict():
    with pytest.raises(TypeError):
        Trajectory({'id': 13, 'vib_obj': [5.5]})


# ------------------------------------------------------------------ PROP-6b

def test_to_json_emits_vib_obj_only_for_id13():
    t13 = _vib13({'rate': 4, 'extentStart': 0.02, 'extentEnd': 0.04,
                  'vertOffset': 0.01, 'phase': 0.5})
    assert t13.to_json()['vibObj'] == {
        'rate': 4.0, 'extentStart': 0.02, 'extentEnd': 0.04,
        'vertOffset': 0.01, 'phase': 0.5,
    }
    for id_ in (0, 1, 6, 12):
        t = Trajectory({'id': id_, 'pitches': [Pitch()] * 4,
                        'dur_array': [0.25] * 4, 'fund_id12': 220})
        assert 'vibObj' not in t.to_json()


def test_vib_obj_on_other_ids_is_accepted_and_ignored():
    j = {'id': 0, 'pitches': [Pitch().to_json()], 'durTot': 1.0,
         'vibObj': {'periods': 8, 'vertOffset': 0, 'initUp': True, 'extent': 0.05}}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        t = Trajectory.from_json(j)
    assert 'vibObj' not in t.to_json()
    # a well-formed one is kept in memory, so retyping 0 -> 13 sees it
    assert t.vib_obj['rate'] == 8.0
    j['vibObj'] = {'rate': 3, 'extentStart': 0.1, 'extentEnd': 0.1,
                   'vertOffset': 0, 'phase': 0}
    assert 'vibObj' not in Trajectory.from_json(j).to_json()


def test_malformed_vib_obj_on_other_ids_warns_instead_of_raising():
    j = {'id': 0, 'pitches': [Pitch().to_json()], 'durTot': 1.0,
         'vibObj': {'periods': 'eight', 'extent': -1, 'weird': True}}
    with pytest.warns(UserWarning, match='ignoring malformed vibObj'):
        t = Trajectory.from_json(j)
    assert t.vib_obj == default_vib_obj()
    j['id'] = 13
    with pytest.raises(ValueError):
        Trajectory.from_json(j)


def test_round_trip_is_stable_and_v2():
    t = _vib13({'periods': '3.5', 'extent': '0.055', 'vertOffset': '0.02',
                'initUp': 'true'}, dur_tot=0.474)
    j1 = json.loads(json.dumps(t.to_json()))
    t2 = Trajectory.from_json(j1)
    j2 = json.loads(json.dumps(t2.to_json()))
    assert j1 == j2
    assert set(j1['vibObj']) == {'rate', 'extentStart', 'extentEnd', 'vertOffset', 'phase'}
    for x in (0, 0.1, 0.5, 0.9, 1):
        assert t2.compute(x) == pytest.approx(t.compute(x), rel=1e-12)


def _load_fixture(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return json.load(f)


def _trajs(d):
    return [t for row in d['phraseGrid'] for ph in row
            for r in ph['trajectoryGrid'] for t in r]


def test_legacy_v1_fixture_loads_identically_to_v2_fixture():
    """serialization_test_v1.json is the old fixture (a vibObj on all 34
    trajectories, v1 form); serialization_test.json is its PROP-6/6b
    canonical form (one healed v2 vibObj, on the id-13 trajectory only)."""
    v1_json = _load_fixture('serialization_test_v1.json')
    v2_json = _load_fixture('serialization_test.json')
    assert sum('vibObj' in t for t in _trajs(v1_json)) == 34
    assert [t['id'] for t in _trajs(v2_json) if 'vibObj' in t] == [13]
    assert set(_trajs(v2_json)[15]['vibObj']) == {
        'rate', 'extentStart', 'extentEnd', 'vertOffset', 'phase'}
    with warnings.catch_warnings():
        # neither fixture may trip the malformed-vibObj fallback
        warnings.filterwarnings('error', message='.*malformed vibObj.*')
        p1 = Piece.from_json(v1_json)
        p2 = Piece.from_json(v2_json)
    j1, j2 = p1.to_json(), p2.to_json()
    # trajectories without a stored uniqueId get a fresh uuid on every load
    for t in _trajs(j1) + _trajs(j2):
        t.pop('uniqueId', None)
    assert j1 == j2
    vib = [t for t in p1.all_trajectories() if t.id == 13]
    assert len(vib) == 1
    vib2 = [t for t in p2.all_trajectories() if t.id == 13][0]
    for k in range(201):
        assert vib[0].compute(k / 200) == pytest.approx(vib2.compute(k / 200), rel=1e-12)
    # canonical output: vibObj only on id 13, in v2 form
    out = _trajs(p1.to_json())
    assert [t['id'] for t in out if 'vibObj' in t] == [13]
    assert 'periods' not in next(t for t in out if t['id'] == 13)['vibObj']


# ------------------------------------------------------------ curve shape

def test_curve_attaches_at_ends_and_first_extreme_for_every_phase():
    for phase in [0, 0.3, 1.0, math.pi / 2, 2.0, math.pi, 4.5, 2 * math.pi - 0.1]:
        t = _vib13({'rate': 3.3, 'extentStart': 0.05, 'extentEnd': 0.05,
                    'vertOffset': 0.0, 'phase': phase}, dur_tot=1.7)
        lf0 = t.log_freqs[0]
        assert math.log2(t.id13(0)) == pytest.approx(lf0, abs=1e-12)
        assert math.log2(t.id13(1)) == pytest.approx(lf0, abs=1e-12)
        xs = t.vib_breakpoints()
        P = 3.3 * 1.7
        assert xs[1] >= 1 / (4 * P) - 1e-12
        assert xs[-2] <= 1 - 1 / (4 * P) + 1e-12
        # every interior breakpoint is an extreme: |y - centre| == A
        for x in xs[1:-1]:
            assert abs(math.log2(t.id13(x)) - lf0) == pytest.approx(0.025, abs=1e-12)
        # smooth attach: finite-difference slope ~ 0 at x1 from both sides
        x1 = xs[1]
        h = 1e-6
        left = (math.log2(t.id13(x1)) - math.log2(t.id13(x1 - h))) / h
        right = (math.log2(t.id13(x1 + h)) - math.log2(t.id13(x1))) / h
        assert abs(left) < 1e-3 and abs(right) < 1e-3


def test_sub_cycle_computes_with_p_equal_one():
    a = _vib13({'rate': 0.3, 'extentStart': 0.05, 'extentEnd': 0.05,
                'vertOffset': 0, 'phase': 0.7}, dur_tot=1.0)
    b = _vib13({'rate': 1.0, 'extentStart': 0.05, 'extentEnd': 0.05,
                'vertOffset': 0, 'phase': 0.7}, dur_tot=1.0)
    assert a.vib_obj['rate'] == 0.3  # not stored as 1
    for k in range(101):
        assert a.id13(k / 100) == pytest.approx(b.id13(k / 100), rel=1e-12)


def test_degenerate_short_trajectory_two_tapers_meet_at_one_extreme():
    # P = 1 with phase 0.2 pi: k1 == k2, so x2 == x1 and the middle is empty
    t = _vib13({'rate': 1, 'extentStart': 0.05, 'extentEnd': 0.05,
                'vertOffset': 0, 'phase': 0.2 * math.pi}, dur_tot=1.0)
    xs = t.vib_breakpoints()
    assert len(xs) == 3
    assert xs[1] == pytest.approx(0.4)
    assert abs(math.log2(t.id13(0.4)) - t.log_freqs[0]) == pytest.approx(0.025)
    assert len(decompose_trajectory(t, vibrato_as_cosines=True)) == 2
    assert_matches_compute(t, vibrato_as_cosines=True)
    assert_matches_compute(t)


def test_extent_ramp_and_offset_clamp():
    t = _vib13({'rate': 4, 'extentStart': 0.0, 'extentEnd': 0.08,
                'vertOffset': 0.01, 'phase': math.pi}, dur_tot=2.0)
    lf0 = t.log_freqs[0]
    xs = t.vib_breakpoints()
    # excursion grows along the ramp; offset is clamped to +-A(x) so the
    # centre fades in with the extent. With phase pi the first interior
    # breakpoint is a crest and crests/troughs alternate from there.
    crests = [math.log2(t.id13(x)) - lf0 for x in xs[1:-1:2]]
    troughs = [math.log2(t.id13(x)) - lf0 for x in xs[2:-1:2]]
    assert crests == sorted(crests)
    assert troughs == sorted(troughs, reverse=True)
    for x, y in zip(xs[1:-1:2], crests):
        A = 0.04 * x
        assert y == pytest.approx(min(0.01, A) + A, abs=1e-12)
    for x, y in zip(xs[2:-1:2], troughs):
        A = 0.04 * x
        assert y == pytest.approx(min(0.01, A) - A, abs=1e-12)


# --------------------------------------------------------------- decompose

@pytest.mark.parametrize('vib, dur_tot', [
    ({'rate': 5.5, 'extentStart': 0.05, 'extentEnd': 0.05, 'vertOffset': 0, 'phase': math.pi}, 2.0),
    ({'rate': 3.7, 'extentStart': 0.06, 'extentEnd': 0.06, 'vertOffset': -0.005, 'phase': 1.3}, 0.55),
    ({'rate': 1.0, 'extentStart': 0.05, 'extentEnd': 0.05, 'vertOffset': 0, 'phase': math.pi / 2}, 0.3),
    ({'rate': 2.5, 'extentStart': 0.04, 'extentEnd': 0.04, 'vertOffset': 0.9, 'phase': math.pi}, 2.0),
    ({'periods': '3.5', 'vertOffset': 0.0209, 'initUp': True, 'extent': '0.055'}, 0.474),
    ({'periods': 3, 'vertOffset': 0.01, 'initUp': False, 'extent': 0.08}, 2.0),
])
def test_decompose_reproduces_v2_curve_exactly(vib, dur_tot):
    t = _vib13(vib, dur_tot)
    chunks = decompose_trajectory(t, 1.5, vibrato_as_cosines=True)
    assert all(c.type == 'cosine' for c in chunks)
    assert [c.continuation for c in chunks] == [False] + [True] * (len(chunks) - 1)
    assert len(chunks) == len(t.vib_breakpoints()) - 1
    assert_matches_compute(t, 1.5, vibrato_as_cosines=True)
    # and the default view, one vibrato chunk carrying all five numbers
    # (vert_offset included), matches too
    default = decompose_trajectory(t, 1.5)
    assert [c.type for c in default] == ['vibrato']
    assert default[0].vert_offset == t.vib_obj['vert_offset']
    assert_matches_compute(t, 1.5)


def test_decompose_ramp_chunk_ends_sit_on_curve_and_interior_is_close():
    t = _vib13({'rate': 5.5, 'extentStart': 0.0, 'extentEnd': 0.08,
                'vertOffset': 0.01, 'phase': math.pi}, dur_tot=2.0)
    chunks = decompose_trajectory(t, vibrato_as_cosines=True)
    for c in chunks:
        assert chunk_freq_at(c, c.start.time) == pytest.approx(
            t.compute(c.start.time / 2.0), rel=1e-9)
        assert chunk_freq_at(c, c.end.time) == pytest.approx(
            t.compute(c.end.time / 2.0), rel=1e-9)
        # interior deviation bounded by the ramp increment over one half period
        # (A changes by 0.04 * (1 / 22) per half period here)
        for u in (0.25, 0.5, 0.75):
            tt = c.start.time + u * c.dur_tot
            assert abs(math.log2(chunk_freq_at(c, tt)) - math.log2(t.compute(tt / 2.0))) < 0.04 / 22


# ------------------------------------------------- durArray on a fresh id 13

def test_fresh_id13_gets_a_dur_array():
    """A vibrato is one segment, so it defaults to [1] like a `fixed`.

    Regression: the constructor defaulted `dur_array` for ids 0-11 and skipped
    13, so an id 13 built from scratch serialised with ``durArray: null``. The
    editor never hit it, because retyping an existing trajectory inherits its
    array, but `reconstruct`'s vibrato path builds one from nothing -- and the
    web app maps over `durArray` for every sounding trajectory and throws on
    null, so an uploaded machine transcription would not render at all.
    """
    t = _vib13({'rate': 5.0, 'extentStart': 0.05, 'extentEnd': 0.05,
                'vertOffset': 0.0, 'phase': 0.0}, dur_tot=2.0)
    assert t.dur_array == [1]
    assert t.to_json()['durArray'] == [1]


def test_id13_keeps_an_explicit_dur_array():
    t = _vib13({'rate': 5.0, 'extentStart': 0.05, 'extentEnd': 0.05,
                'vertOffset': 0.0, 'phase': 0.0}, dur_tot=2.0, dur_array=[1.0])
    assert t.dur_array == [1.0]


def test_reconstructed_vibrato_carries_a_dur_array():
    """The path that actually broke: a vibrato chunk through reconstruct_piece.

    Asserts it for every sounding trajectory, not just the vibrato, because the
    web app's failure mode is per-trajectory and this is the cheapest place to
    catch the next id that has no default.
    """
    from idtap.classes.simple_trajectory import simple_trajectories_from_dots
    from idtap.classes.reconstruct import reconstruct_piece
    from idtap.classes.raga import Raga
    from idtap.enums import Instrument

    chunks = simple_trajectories_from_dots(
        times=[0.0, 2.0], log_freqs=[0.0, 0.0], types=['vibrato'],
        vibratos=[{'rate': 5.0, 'extent_start': 0.05, 'extent_end': 0.05,
                   'phase': 0.0, 'vert_offset': 0.0}])
    piece = reconstruct_piece(chunks, Raga(), Instrument.Vocal_M,
                              synthetic=True)
    trajs = piece.all_trajectories()
    assert any(t.id == 13 for t in trajs)
    for t in trajs:
        if t.id == 12:                      # silence is skipped by the app
            continue
        assert t.dur_array, f"id {t.id} has no dur_array"
        assert t.to_json()['durArray'], f"id {t.id} serialises durArray as null"
