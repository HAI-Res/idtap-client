import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import json
import pytest
import responses

from idtap.client import SwaraClient

BASE = 'https://swara.studio/'


@responses.activate
def test_start_transcription_sends_only_what_was_given():
    client = SwaraClient(auto_login=False)
    responses.post(BASE + 'api/transcribe', json={'jobId': 'j1', 'queuedAhead': 0})
    assert client.start_transcription('r1', model='student', tonic=220.0, voice='F', lam=0.05) == 'j1'
    assert json.loads(responses.calls[0].request.body) == {
        'recordingId': 'r1', 'model': 'student', 'voice': 'F', 'separate': True, 'tonic': 220.0, 'lam': 0.05}


def test_start_transcription_checks_arguments():
    client = SwaraClient(auto_login=False)
    with pytest.raises(ValueError):
        client.start_transcription('r1', model='nope')
    with pytest.raises(ValueError):
        client.start_transcription('r1', model='student')          # no tonic
    with pytest.raises(ValueError):
        client.start_transcription('r1', model='rg_world_tx', voice='X')
    with pytest.raises(ValueError):
        client.transcribe('r1', where='moon')
    with pytest.raises(ValueError):
        client.transcribe(where='server')


@responses.activate
def test_transcribe_on_the_server_polls_then_returns_the_piece():
    client = SwaraClient(auto_login=False)
    responses.post(BASE + 'api/transcribe', json={'jobId': 'j1'})
    job = BASE + 'api/transcribe/j1'
    responses.get(job, json={'status': 'queued', 'events': []})
    responses.get(job, json={'status': 'running', 'events': [{'event': 'phase', 'phase': 'separate'}]})
    responses.get(job, json={'status': 'done', 'pieceId': 'p1',
                             'events': [{'event': 'phase', 'phase': 'separate'}, {'event': 'done'}]})
    responses.get(BASE + 'api/user', json={'waiverAgreed': True})
    responses.get(BASE + 'api/transcription/p1', json={'_id': 'p1'})
    seen = []
    piece = client.transcribe('r1', model='rg_world_tx', on_event=seen.append, poll_s=0)
    assert piece == {'_id': 'p1'}
    assert [e['event'] for e in seen] == ['phase', 'done']        # each event once


@responses.activate
def test_transcribe_raises_on_failure():
    client = SwaraClient(auto_login=False)
    responses.post(BASE + 'api/transcribe', json={'jobId': 'j1'})
    responses.get(BASE + 'api/transcribe/j1', json={'status': 'failed', 'error': {'code': 'audio', 'message': 'no singing found'}})
    with pytest.raises(RuntimeError, match='no singing found'):
        client.transcribe('r1', model='rg_world_tx', poll_s=0)


def test_local_needs_the_engine_package():
    client = SwaraClient(auto_login=False)
    with pytest.raises(ValueError):
        client.transcribe(where='local')
    sys.modules.pop('idtap_transcribe', None)
    try:
        import idtap_transcribe  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match='idtap-transcribe'):
            client.transcribe(where='local', audio_path='x.wav', model='student', tonic=220.0)
