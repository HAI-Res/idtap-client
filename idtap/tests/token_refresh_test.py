"""Tests for automatic token refresh (issue #2).

All tests are offline: HTTP is mocked with `responses` and secure storage is
replaced with mocks so no real keyring/encrypted-file state is touched.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath('.'))

import responses

from idtap.auth import refresh_access_token
from idtap.client import SwaraClient

BASE = 'https://swara.studio/'
TOKEN_URL = BASE + 'oauth/token'


def make_storage(expired: bool = False) -> MagicMock:
    storage = MagicMock()
    storage.is_token_expired.return_value = expired
    storage.store_tokens.return_value = True
    return storage


def make_client(tokens, expired: bool = False) -> SwaraClient:
    """Build a client with mocked storage and stubbed initial token load."""
    with patch('idtap.client.SecureTokenStorage') as storage_cls, \
         patch('idtap.client.load_token', return_value=None):
        storage_cls.return_value = make_storage(expired)
        client = SwaraClient(auto_login=False)
    client._token_data = tokens
    client.token = tokens.get('id_token') if tokens else None
    client.user = tokens.get('profile') if tokens else None
    client.secure_storage.is_token_expired.return_value = expired
    return client


# ---- refresh_access_token (auth.py) ----

@responses.activate
def test_refresh_access_token_success():
    storage = make_storage()
    tokens = {'id_token': 'old', 'refresh_token': 'rt', 'profile': {'_id': 'u1'}}
    responses.post(TOKEN_URL, json={
        'id_token': 'new', 'access_token': 'at', 'refresh_token': None,
        'profile': {'_id': 'u1', 'name': 'Jon'},
    }, status=200)

    updated = refresh_access_token(BASE, storage=storage, tokens=tokens)

    assert updated['id_token'] == 'new'
    # Server omitted a new refresh token -> the old one is kept
    assert updated['refresh_token'] == 'rt'
    assert updated['profile'] == {'_id': 'u1', 'name': 'Jon'}
    storage.store_tokens.assert_called_once_with(updated)
    body = responses.calls[0].request.body
    assert b'"grant_type": "refresh_token"' in body
    assert b'"refresh_token": "rt"' in body


def test_refresh_access_token_no_refresh_token():
    storage = make_storage()
    assert refresh_access_token(BASE, storage=storage, tokens={'id_token': 'old'}) is None
    storage.store_tokens.assert_not_called()


@responses.activate
def test_refresh_access_token_server_rejects():
    # Revoked token (401) or a pre-refresh server (500/404) both return None.
    storage = make_storage()
    tokens = {'id_token': 'old', 'refresh_token': 'rt'}
    responses.post(TOKEN_URL, json={'error': 'invalid_grant'}, status=401)
    assert refresh_access_token(BASE, storage=storage, tokens=tokens) is None
    storage.store_tokens.assert_not_called()


@responses.activate
def test_refresh_access_token_missing_id_token():
    storage = make_storage()
    tokens = {'id_token': 'old', 'refresh_token': 'rt'}
    responses.post(TOKEN_URL, json={'access_token': 'at'}, status=200)
    assert refresh_access_token(BASE, storage=storage, tokens=tokens) is None


# ---- SwaraClient integration ----

def test_expired_tokens_refresh_on_load():
    stored = {'id_token': 'old', 'refresh_token': 'rt', 'profile': {'_id': 'u1'}}
    refreshed = {'id_token': 'new', 'refresh_token': 'rt', 'profile': {'_id': 'u1'}}
    with patch('idtap.client.SecureTokenStorage') as storage_cls, \
         patch('idtap.client.load_token', return_value=stored), \
         patch('idtap.client.refresh_access_token', return_value=refreshed) as mock_refresh:
        storage_cls.return_value = make_storage(expired=True)
        client = SwaraClient(auto_login=False)

    assert client.token == 'new'
    assert client._token_data == refreshed
    mock_refresh.assert_called_once()
    client.secure_storage.clear_tokens.assert_not_called()


def test_expired_tokens_cleared_when_refresh_fails():
    stored = {'id_token': 'old', 'refresh_token': 'rt'}
    with patch('idtap.client.SecureTokenStorage') as storage_cls, \
         patch('idtap.client.load_token', return_value=stored), \
         patch('idtap.client.refresh_access_token', return_value=None):
        storage_cls.return_value = make_storage(expired=True)
        client = SwaraClient(auto_login=False)

    assert client.token is None
    client.secure_storage.clear_tokens.assert_called_once()


@responses.activate
def test_proactive_refresh_before_request():
    tokens = {'id_token': 'old', 'refresh_token': 'rt'}
    refreshed = {'id_token': 'new', 'refresh_token': 'rt'}
    client = make_client(tokens, expired=True)
    responses.get(BASE + 'api/ragas', json=['Yaman'], status=200)

    with patch('idtap.client.refresh_access_token', return_value=refreshed) as mock_refresh:
        result = client.get_available_ragas()

    assert result == ['Yaman']
    mock_refresh.assert_called_once()
    assert responses.calls[0].request.headers['Authorization'] == 'Bearer new'


@responses.activate
def test_retry_once_on_401():
    tokens = {'id_token': 'old', 'refresh_token': 'rt'}
    refreshed = {'id_token': 'new', 'refresh_token': 'rt'}
    client = make_client(tokens, expired=False)
    responses.get(BASE + 'api/ragas', json={'error': 'expired'}, status=401)
    responses.get(BASE + 'api/ragas', json=['Yaman'], status=200)

    with patch('idtap.client.refresh_access_token', return_value=refreshed):
        result = client.get_available_ragas()

    assert result == ['Yaman']
    assert len(responses.calls) == 2
    assert responses.calls[0].request.headers['Authorization'] == 'Bearer old'
    assert responses.calls[1].request.headers['Authorization'] == 'Bearer new'


@responses.activate
def test_401_not_retried_when_refresh_fails():
    tokens = {'id_token': 'old', 'refresh_token': 'rt'}
    client = make_client(tokens, expired=False)
    responses.get(BASE + 'api/ragas', json={'error': 'expired'}, status=401)

    with patch('idtap.client.refresh_access_token', return_value=None):
        try:
            client.get_available_ragas()
            assert False, 'expected HTTPError'
        except Exception as e:
            assert '401' in str(e)

    assert len(responses.calls) == 1
