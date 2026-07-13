"""Tests for serialization sync: stripped fields, context threading, and backward compat.

These tests verify that:
1. to_json() strips redundant fields (ratios, fundamental, name, instrumentation, tags, raga)
2. from_json() with context params (ratios, fundamental) preserves frequencies
3. Legacy JSON with embedded fields still loads correctly (backward compat)
4. Multi-cycle round-trips stabilize and preserve data
"""

import json as json_mod
import pytest

from idtap.classes.pitch import Pitch
from idtap.classes.trajectory import Trajectory
from idtap.classes.phrase import Phrase
from idtap.classes.piece import Piece
from idtap.classes.raga import Raga


# ===================================================================
# A. Pitch Tests
# ===================================================================

class TestPitchSerializationSync:

    def test_to_json_strips_ratios_and_fundamental(self):
        """to_json() must NOT include ratios or fundamental."""
        raga = Raga({'name': 'Yaman'})
        p = Pitch({'swara': 're', 'raised': True, 'oct': 0,
                   'ratios': raga.stratified_ratios, 'fundamental': raga.fundamental})
        j = p.to_json()
        assert 'ratios' not in j
        assert 'fundamental' not in j
        assert j.keys() == {'swara', 'raised', 'oct', 'logOffset'}

    def test_round_trip_with_raga_context(self):
        """to_json() -> from_json(ratios, fundamental) preserves frequency."""
        raga = Raga({'name': 'Yaman'})
        p = Pitch({'swara': 'ga', 'raised': False, 'oct': 1,
                   'ratios': raga.stratified_ratios, 'fundamental': raga.fundamental,
                   'logOffset': 0.15})
        j = p.to_json()
        restored = Pitch.from_json(j, ratios=raga.stratified_ratios,
                                   fundamental=raga.fundamental)
        assert restored.frequency == pytest.approx(p.frequency, rel=1e-10)
        assert restored.to_json() == j

    def test_from_json_legacy_with_embedded_ratios(self):
        """Old JSON with ratios/fundamental embedded still works without context."""
        raga = Raga({'name': 'Yaman'})
        p = Pitch({'swara': 'ni', 'raised': True, 'oct': -1,
                   'ratios': raga.stratified_ratios, 'fundamental': raga.fundamental})
        # Simulate legacy JSON that includes ratios and fundamental
        legacy_json = {**p.to_json(),
                       'ratios': raga.stratified_ratios,
                       'fundamental': raga.fundamental}
        restored = Pitch.from_json(legacy_json)  # no context params
        assert restored.frequency == pytest.approx(p.frequency, rel=1e-10)

    def test_round_trip_all_swaras_multiple_octaves(self):
        """Exhaustive: every swara x octave round-trips with correct frequency."""
        raga = Raga({'name': 'Yaman', 'fundamental': 240})
        ratios = raga.stratified_ratios
        fund = raga.fundamental
        swaras = [
            ('sa', True), ('re', False), ('re', True),
            ('ga', False), ('ga', True), ('ma', False), ('ma', True),
            ('pa', True), ('dha', False), ('dha', True),
            ('ni', False), ('ni', True),
        ]
        for swara, raised in swaras:
            for oct in [-1, 0, 1]:
                p = Pitch({'swara': swara, 'raised': raised, 'oct': oct,
                           'ratios': ratios, 'fundamental': fund})
                j = p.to_json()
                restored = Pitch.from_json(j, ratios=ratios, fundamental=fund)
                assert restored.frequency == pytest.approx(p.frequency, rel=1e-10), \
                    f"Failed for {swara} raised={raised} oct={oct}"

    def test_context_params_override_embedded(self):
        """When both context params and embedded values exist, context wins."""
        raga1 = Raga({'name': 'Yaman', 'fundamental': 261.63})
        raga2 = Raga({'name': 'Yaman', 'fundamental': 240.0})
        p = Pitch({'swara': 'ga', 'raised': True, 'oct': 0,
                   'ratios': raga1.stratified_ratios, 'fundamental': raga1.fundamental})
        # Legacy JSON with raga1's values embedded
        legacy_json = {**p.to_json(),
                       'ratios': raga1.stratified_ratios,
                       'fundamental': raga1.fundamental}
        # But context says raga2 — context should win
        restored = Pitch.from_json(legacy_json,
                                   ratios=raga2.stratified_ratios,
                                   fundamental=raga2.fundamental)
        assert restored.fundamental == 240.0


# ===================================================================
# B. Trajectory Tests
# ===================================================================

class TestTrajectorySerializationSync:

    def test_to_json_strips_name_instrumentation_tags(self):
        """to_json() must NOT include name, instrumentation, or tags."""
        t = Trajectory({'id': 3, 'pitches': [Pitch()], 'durTot': 1})
        j = t.to_json()
        assert 'name' not in j
        assert 'instrumentation' not in j
        assert 'tags' not in j
        # Essential fields still present
        assert 'id' in j
        assert 'pitches' in j
        assert 'durTot' in j

    def test_round_trip_stripped_with_context(self):
        """Stripped format round-trips when raga context is provided."""
        raga = Raga({'name': 'Yaman'})
        ratios = raga.stratified_ratios
        fund = raga.fundamental
        p = Pitch({'swara': 'dha', 'raised': True, 'oct': 0,
                   'ratios': ratios, 'fundamental': fund})
        t = Trajectory({'id': 1, 'pitches': [p], 'durTot': 2.5})
        j = t.to_json()
        restored = Trajectory.from_json(j, ratios=ratios, fundamental=fund)
        assert restored.id == 1
        assert restored.dur_tot == 2.5
        assert restored.pitches[0].frequency == pytest.approx(p.frequency, rel=1e-10)
        # Derived fields restored from defaults
        assert restored.name is not None  # derived from id
        assert restored.tags == []

    def test_from_json_legacy_with_all_fields(self):
        """Legacy JSON with name/instrumentation/tags still loads correctly."""
        raga = Raga({'name': 'Yaman'})
        p = Pitch({'swara': 'sa', 'ratios': raga.stratified_ratios,
                   'fundamental': raga.fundamental})
        t = Trajectory({'id': 0, 'pitches': [p], 'durTot': 1})
        j = t.to_json()
        # Re-add stripped fields to simulate legacy
        j['name'] = 'Fixed'
        j['instrumentation'] = 'Sitar'
        j['tags'] = ['annotation-1', 'annotation-2']
        j['pitches'][0]['ratios'] = raga.stratified_ratios
        j['pitches'][0]['fundamental'] = raga.fundamental
        restored = Trajectory.from_json(j)  # no context params
        assert restored.id == 0
        assert restored.tags == ['annotation-1', 'annotation-2']
        assert restored.pitches[0].frequency == pytest.approx(p.frequency, rel=1e-10)

    def test_round_trip_all_ids(self):
        """Every trajectory type (id 0-13) round-trips with frequency preservation."""
        raga = Raga({'name': 'Yaman'})
        ratios = raga.stratified_ratios
        fund = raga.fundamental
        for tid in range(14):
            if tid == 11:
                continue  # same as id 7
            pitch_count = 2 if 4 <= tid <= 10 else 1
            pitches = [Pitch({'swara': i, 'ratios': ratios, 'fundamental': fund})
                       for i in range(pitch_count)]
            t = Trajectory({'id': tid, 'pitches': pitches, 'durTot': 1})
            j = t.to_json()
            restored = Trajectory.from_json(j, ratios=ratios, fundamental=fund)
            assert restored.id == tid
            assert len(restored.pitches) == pitch_count
            for i, pitch in enumerate(restored.pitches):
                assert pitch.frequency == pytest.approx(
                    t.pitches[i].frequency, rel=1e-10
                ), f"Failed for trajectory id={tid}, pitch {i}"

    def test_round_trip_preserves_articulations_and_automation(self):
        """Articulations and automation survive stripped round-trip."""
        raga = Raga({'name': 'Yaman'})
        ratios = raga.stratified_ratios
        fund = raga.fundamental
        p = Pitch({'swara': 'ga', 'ratios': ratios, 'fundamental': fund})
        t = Trajectory({
            'id': 0,
            'pitches': [p],
            'durTot': 1,
            'articulations': {'0.00': {'name': 'pluck', 'stroke': 'da'}},
            'automation': {'values': [
                {'normTime': 0.0, 'value': 0.5},
                {'normTime': 0.5, 'value': 0.8},
                {'normTime': 1.0, 'value': 1.0},
            ]},
        })
        j = t.to_json()
        restored = Trajectory.from_json(j, ratios=ratios, fundamental=fund)
        assert '0.00' in restored.articulations
        assert restored.articulations['0.00'].stroke == 'da'
        assert restored.automation is not None


# ===================================================================
# C. Phrase Tests
# ===================================================================

class TestPhraseSerializationSync:

    def test_to_json_strips_raga(self):
        """to_json() must NOT include raga."""
        raga = Raga({'name': 'Yaman'})
        phrase = Phrase({
            'trajectories': [Trajectory({'id': 0, 'pitches': [Pitch({
                'ratios': raga.stratified_ratios, 'fundamental': raga.fundamental
            })], 'durTot': 1})],
            'raga': raga,
        })
        j = phrase.to_json()
        assert 'raga' not in j

    def test_round_trip_with_raga_context(self):
        """Stripped phrase round-trips when raga context is threaded."""
        raga = Raga({'name': 'Yaman', 'fundamental': 240})
        ratios = raga.stratified_ratios
        fund = raga.fundamental
        p = Pitch({'swara': 'pa', 'ratios': ratios, 'fundamental': fund, 'logOffset': 0.1})
        t = Trajectory({'id': 0, 'pitches': [p], 'durTot': 1})
        phrase = Phrase({'trajectories': [t], 'raga': raga})
        j = phrase.to_json()
        restored = Phrase.from_json(j, ratios=ratios, fundamental=fund)
        assert restored.trajectories[0].pitches[0].frequency == pytest.approx(
            p.frequency, rel=1e-10)

    def test_from_json_legacy_with_embedded_raga(self):
        """Old JSON with raga embedded in phrase still works without context."""
        raga = Raga({'name': 'Yaman'})
        p = Pitch({'swara': 'ga', 'raised': True, 'ratios': raga.stratified_ratios,
                   'fundamental': raga.fundamental})
        t = Trajectory({'id': 0, 'pitches': [p], 'durTot': 1})
        phrase = Phrase({'trajectories': [t], 'raga': raga})
        # Build legacy JSON: raga present, ratios in pitches
        j = phrase.to_json()
        j['raga'] = raga.to_json()
        j['trajectoryGrid'][0][0]['pitches'][0]['ratios'] = raga.stratified_ratios
        j['trajectoryGrid'][0][0]['pitches'][0]['fundamental'] = raga.fundamental
        restored = Phrase.from_json(j)  # no context params
        assert restored.trajectories[0].pitches[0].frequency == pytest.approx(
            p.frequency, rel=1e-10)

    def test_context_overrides_embedded_raga(self):
        """When both context params and embedded raga exist, context wins."""
        raga1 = Raga({'name': 'Yaman', 'fundamental': 261.63})
        raga2 = Raga({'name': 'Yaman', 'fundamental': 240.0})
        p = Pitch({'swara': 'sa', 'ratios': raga1.stratified_ratios,
                   'fundamental': raga1.fundamental})
        t = Trajectory({'id': 0, 'pitches': [p], 'durTot': 1})
        phrase = Phrase({'trajectories': [t], 'raga': raga1})
        j = phrase.to_json()
        j['raga'] = raga1.to_json()  # legacy embedded raga
        # But context says raga2
        restored = Phrase.from_json(j, ratios=raga2.stratified_ratios,
                                    fundamental=raga2.fundamental)
        # Context should win
        assert restored.trajectories[0].pitches[0].fundamental == 240.0


# ===================================================================
# D. Piece Tests (Full Integration)
# ===================================================================

class TestPieceSerializationSync:

    def test_to_json_strips_nested_fields(self):
        """Verify stripped fields are absent at every nesting level."""
        raga = Raga({'name': 'Yaman', 'fundamental': 240})
        ratios = raga.stratified_ratios
        p1 = Pitch({'swara': 'ga', 'raised': False, 'ratios': ratios, 'fundamental': 240})
        p2 = Pitch({'swara': 'pa', 'ratios': ratios, 'fundamental': 240})
        t = Trajectory({'id': 1, 'pitches': [p1, p2], 'durTot': 1})
        phrase = Phrase({'trajectories': [t], 'raga': raga, 'isSectionStart': True})
        piece = Piece({'phrases': [phrase], 'raga': raga, 'instrumentation': ['Sitar']})
        j = piece.to_json()
        # Piece level
        assert 'durArray' not in j
        assert 'sectionCategorization' not in j
        assert 'sectionStarts' not in j
        assert 'sectionStartsGrid' not in j
        assert 'phrases' not in j
        # Phrase level
        phrase_json = j['phraseGrid'][0][0]
        assert 'raga' not in phrase_json
        # Trajectory level
        traj_json = phrase_json['trajectoryGrid'][0][0]
        assert 'name' not in traj_json
        assert 'instrumentation' not in traj_json
        assert 'tags' not in traj_json
        # Pitch level
        pitch_json = traj_json['pitches'][0]
        assert 'ratios' not in pitch_json
        assert 'fundamental' not in pitch_json

    def test_round_trip_stripped_preserves_frequencies(self):
        """Full round-trip through stripped format preserves all pitch frequencies."""
        raga = Raga({'name': 'Yaman', 'fundamental': 240})
        ratios = raga.stratified_ratios
        pitches = [
            Pitch({'swara': 'ga', 'raised': False, 'oct': 0, 'ratios': ratios,
                   'fundamental': 240, 'logOffset': 0.02}),
            Pitch({'swara': 'ni', 'raised': True, 'oct': 1, 'ratios': ratios,
                   'fundamental': 240}),
        ]
        t = Trajectory({'id': 5, 'pitches': pitches, 'durTot': 2})
        phrase = Phrase({'trajectories': [t], 'raga': raga})
        piece = Piece({'phrases': [phrase], 'raga': raga, 'instrumentation': ['Sitar']})
        orig_freqs = [p.frequency for p in piece.all_pitches()]
        j = piece.to_json()
        restored = Piece.from_json(j)
        restored_freqs = [p.frequency for p in restored.all_pitches()]
        assert len(restored_freqs) == len(orig_freqs)
        for orig, rest in zip(orig_freqs, restored_freqs):
            assert rest == pytest.approx(orig, rel=1e-10)

    def test_from_json_legacy_fixture(self):
        """The existing test fixture (old format) still loads correctly."""
        with open('idtap/tests/fixtures/serialization_test.json') as f:
            fixture = json_mod.load(f)
        piece = Piece.from_json(fixture)
        # Basic structure preserved
        assert len(piece.phrase_grid) > 0
        assert piece.raga is not None
        assert piece.raga.fundamental > 0
        # Re-serialize in new stripped format
        json2 = piece.to_json()
        assert 'durArray' not in json2
        # Load again from stripped format
        piece2 = Piece.from_json(json2)
        # Frequencies should match
        freqs1 = [p.frequency for p in piece.all_pitches()]
        freqs2 = [p.frequency for p in piece2.all_pitches()]
        assert len(freqs2) == len(freqs1)
        for f1, f2 in zip(freqs1, freqs2):
            assert f2 == pytest.approx(f1, rel=1e-10)

    def test_full_legacy_with_all_redundant_fields(self):
        """Piece with every legacy field re-added still loads and round-trips."""
        raga = Raga({'name': 'Yaman', 'fundamental': 240})
        ratios = raga.stratified_ratios
        p = Pitch({'swara': 'ga', 'raised': False, 'ratios': ratios, 'fundamental': 240})
        t = Trajectory({'id': 0, 'pitches': [p], 'durTot': 1})
        phrase = Phrase({'trajectories': [t], 'raga': raga, 'isSectionStart': True})
        piece = Piece({'phrases': [phrase], 'raga': raga, 'instrumentation': ['Sitar']})
        j = json_mod.loads(json_mod.dumps(piece.to_json()))
        # Re-add ALL legacy fields
        j['durArray'] = j['durArrayGrid'][0]
        j['sectionCategorization'] = j.get('sectionCatGrid', [[]])[0]
        j['sectionStarts'] = [0]
        j['sectionStartsGrid'] = [[0]]
        j['phrases'] = j['phraseGrid'][0]
        j['phraseGrid'][0][0]['raga'] = raga.to_json()
        traj_json = j['phraseGrid'][0][0]['trajectoryGrid'][0][0]
        traj_json['name'] = 'Fixed'
        traj_json['tags'] = ['legacy-tag']
        traj_json['instrumentation'] = 'Sitar'
        traj_json['pitches'][0]['ratios'] = ratios
        traj_json['pitches'][0]['fundamental'] = 240
        # Load legacy format
        restored = Piece.from_json(j)
        assert len(restored.phrases) == 1
        assert restored.phrases[0].trajectories[0].tags == ['legacy-tag']
        assert restored.all_pitches()[0].frequency == pytest.approx(p.frequency, rel=1e-10)

    def test_multi_cycle_round_trip(self):
        """3-cycle save/load preserves all data and stabilizes JSON size."""
        raga = Raga({'name': 'Yaman', 'fundamental': 220})
        ratios = raga.stratified_ratios
        pitches = [
            Pitch({'swara': 'pa', 'ratios': ratios, 'fundamental': 220}),
            Pitch({'swara': 'dha', 'raised': True, 'oct': 1, 'ratios': ratios,
                   'fundamental': 220, 'logOffset': 0.05}),
        ]
        t = Trajectory({'id': 1, 'pitches': pitches, 'durTot': 1})
        phrase = Phrase({'trajectories': [t], 'raga': raga, 'isSectionStart': True})
        piece = Piece({'phrases': [phrase], 'raga': raga, 'instrumentation': ['Sitar']})
        orig_freqs = [p.frequency for p in piece.all_pitches()]
        # Cycle 1
        j1 = json_mod.loads(json_mod.dumps(piece.to_json()))
        p2 = Piece.from_json(j1)
        # Cycle 2
        j2 = json_mod.loads(json_mod.dumps(p2.to_json()))
        p3 = Piece.from_json(j2)
        # Cycle 3
        j3 = json_mod.loads(json_mod.dumps(p3.to_json()))
        p4 = Piece.from_json(j3)
        # Frequencies survive all cycles
        final_freqs = [p.frequency for p in p4.all_pitches()]
        assert len(final_freqs) == len(orig_freqs)
        for orig, final in zip(orig_freqs, final_freqs):
            assert final == pytest.approx(orig, rel=1e-10)
        # JSON size stabilizes between cycle 2 and 3
        assert len(json_mod.dumps(j2)) == len(json_mod.dumps(j3))

    def test_round_trip_legacy_to_stripped_to_reload(self):
        """Load old format -> re-serialize (stripped) -> load stripped -> verify."""
        with open('idtap/tests/fixtures/serialization_test.json') as f:
            old_json = json_mod.load(f)
        # Load from old format
        piece1 = Piece.from_json(old_json)
        freqs1 = [p.frequency for p in piece1.all_pitches()]
        # Re-serialize in new stripped format
        stripped_json = json_mod.loads(json_mod.dumps(piece1.to_json()))
        # Load from stripped format
        piece2 = Piece.from_json(stripped_json)
        freqs2 = [p.frequency for p in piece2.all_pitches()]
        # All frequencies preserved
        assert len(freqs2) == len(freqs1)
        for f1, f2 in zip(freqs1, freqs2):
            assert f2 == pytest.approx(f1, rel=1e-10)
