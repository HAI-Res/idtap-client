"""Automation must serialise the keys the app reads."""
import json

from idtap.classes.automation import Automation


def test_to_json_camelises_norm_time():
    """The app reads normTime. Given norm_time it finds undefined at every
    point, so no point satisfies `normTime <= x`, and valueAtX throws on the
    first sample -- a transcription that loads and displays but will not play."""
    payload = Automation().to_json()
    assert [v['normTime'] for v in payload['values']] == [0.0, 1.0]
    assert all('norm_time' not in v for v in payload['values'])


def test_round_trips_through_json():
    original = Automation({'values': [{'normTime': 0.0, 'value': 0.2},
                                      {'normTime': 0.5, 'value': 0.9},
                                      {'normTime': 1.0, 'value': 0.4}]})
    back = Automation.from_json(json.loads(json.dumps(original.to_json())))
    assert [v['norm_time'] for v in back.values] == [0.0, 0.5, 1.0]
    assert [v['value'] for v in back.values] == [0.2, 0.9, 0.4]
    assert back.value_at_x(0.0) == 0.2
    assert back.value_at_x(1.0) == 0.4


def test_value_at_x_accepts_both_ends():
    a = Automation()
    assert a.value_at_x(0.0) == 1.0
    assert a.value_at_x(1.0) == 1.0
