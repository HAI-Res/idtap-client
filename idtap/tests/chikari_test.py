import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from idtap.classes.chikari import Chikari
from idtap.classes.pitch import Pitch

# Test mirrors src/js/tests/chikari.test.ts

def test_chikari_serialization():
    """New format: to_json() only includes fundamental and uniqueId, no pitches."""
    pitches = [Pitch({'swara': 's', 'oct': 1}), Pitch({'swara': 'p'})]
    c = Chikari({'pitches': pitches, 'fundamental': 440})
    assert isinstance(c.unique_id, str)
    json_obj = c.to_json()
    assert 'fundamental' in json_obj
    assert 'uniqueId' in json_obj
    assert 'pitches' not in json_obj
    # Round-trip preserves fundamental and uniqueId
    copy = Chikari.from_json(json_obj)
    assert copy.fundamental == c.fundamental
    assert copy.unique_id == c.unique_id
    assert copy.to_json() == json_obj


def test_chikari_from_json_old_format():
    """Backward compat: from_json() handles old format with pitches array."""
    old_json = {
        'fundamental': 261.63,
        'uniqueId': 'test-id-123',
        'pitches': [
            {'swara': 0, 'oct': 2, 'raised': True, 'logOffset': 0},
            {'swara': 0, 'oct': 1, 'raised': True, 'logOffset': 0},
        ]
    }
    c = Chikari.from_json(old_json)
    assert c.fundamental == 261.63
    assert c.unique_id == 'test-id-123'
    assert len(c.pitches) == 2
    assert c.pitches[0].swara == 0
    assert c.pitches[0].oct == 2


def test_chikari_from_json_new_format():
    """New format: from_json() without pitches leaves pitches EMPTY (PROP-3).
    Drone tuning is owned by the raga (Piece.chikari_freqs -> raga.chikari_pitches),
    so the canonical form no longer synthesizes 12-TET default pitches. Matches TS."""
    new_json = {
        'fundamental': 261.63,
        'uniqueId': 'test-id-456',
    }
    c = Chikari.from_json(new_json)
    assert c.fundamental == 261.63
    assert c.unique_id == 'test-id-456'
    assert len(c.pitches) == 0  # PROP-3: no 12-TET default footgun on canonical load
