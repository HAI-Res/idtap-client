import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import responses
import pytest
import json

from idtap.client import SwaraClient

BASE = 'https://swara.studio/'

@responses.activate
def test_get_piece():
    client = SwaraClient(auto_login=False)
    
    # Mock the waiver check endpoint
    waiver_endpoint = BASE + 'api/user'
    responses.get(waiver_endpoint, json={'waiverAgreed': True}, status=200)
    
    # Mock the actual transcription endpoint
    endpoint = BASE + 'api/transcription/1'
    responses.get(endpoint, json={'_id': '1'}, status=200)
    
    result = client.get_piece('1')
    assert result == {'_id': '1'}


@responses.activate
def test_save_piece():
    client = SwaraClient(auto_login=False)
    endpoint = BASE + 'api/transcription'
    responses.post(endpoint, json={'ok': 1}, status=200)
    result = client.save_piece({'_id': '1'})
    assert result == {'ok': 1}


@responses.activate
def test_user_id_prefers_id(tmp_path):
    client = SwaraClient(auto_login=False)
    client.token = 'abc'
    client.user = {'_id': 'u1', 'sub': 's1'}
    assert client.user_id == 'u1'


@responses.activate
def test_user_id_fallback_sub(tmp_path):
    client = SwaraClient(auto_login=False)
    client.token = 'abc'
    client.user = {'sub': 's1'}
    assert client.user_id == 's1'



@responses.activate
def test_insert_new_transcription_uses_api_route():
    client = SwaraClient(auto_login=False)
    client.token = 'abc'
    client.user = {'_id': 'u1'}
    responses.post(BASE + 'api/transcription', json={'insertedId': 'n1'}, status=200)
    result = client.insert_new_transcription({'title': 't', 'userID': 'stale', '_id': None})
    assert result == {'insertedId': 'n1'}
    body = json.loads(responses.calls[0].request.body)
    # server derives owner and id; client must not send them
    assert '_id' not in body and 'userID' not in body


@responses.activate
def test_clone_transcription_uses_api_route():
    client = SwaraClient(auto_login=False)
    client.token = 'abc'
    client.user = {'_id': 'u1', 'name': 'N', 'family_name': 'F', 'given_name': 'G'}
    responses.post(BASE + 'api/transcription/p1/clone', json={'insertedId': 'c1'}, status=200)
    result = client.clone_transcription('p1', title='copy')
    assert result == {'insertedId': 'c1'}
    body = json.loads(responses.calls[0].request.body)
    assert body['title'] == 'copy'
    assert 'newOwner' not in body and 'id' not in body


@responses.activate
def test_delete_transcription_uses_api_route():
    client = SwaraClient(auto_login=False)
    client.token = 'abc'
    client.user = {'_id': 'u1'}
    responses.delete(BASE + 'api/transcription/p1', json={'deletedCount': 1}, status=200)
    result = client.delete_transcription('p1')
    assert result == {'deletedCount': 1}


@responses.activate
def test_get_audio_recording_uses_api_route():
    client = SwaraClient(auto_login=False)
    client.token = 'abc'
    client.user = {'_id': 'u1'}
    responses.get(BASE + 'api/audioRecording/a1', json={'_id': 'a1', 'duration': 5.0}, status=200)
    result = client.get_audio_recording('a1')
    assert result['duration'] == 5.0
