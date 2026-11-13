"""PROP-8 write guard, client side: every request sends
`X-IDTAP-Client: python/<contract version>` (the contract version, not the
package version), and a 409 write-guard refusal raises WriteGuardError with the
server's reasons and is never retried."""
import json

import pytest
import requests
import responses

from idtap import CONTRACT_VERSION, WriteGuardError
from idtap.client import SwaraClient
from idtap.classes.piece import Piece

BASE = "https://swara.studio/"
GUARD_BODY = {"error": "write-guard", "reasons": ["boundary", "tone"], "minContract": "0.3.0"}


def _client():
    c = SwaraClient(auto_login=False)
    c.token = "abc"
    c.user = {"_id": "u1"}
    return c


def test_header_value_is_the_contract_version():
    assert CONTRACT_VERSION == "0.3.1"
    assert _client()._auth_headers()["X-IDTAP-Client"] == "python/0.3.1"


@responses.activate
def test_header_on_bearer_save_route():
    responses.post(BASE + "api/transcription", json={"ok": 1}, status=200)
    _client().save_piece({"_id": "1"})
    assert responses.calls[0].request.headers["X-IDTAP-Client"] == "python/0.3.1"
    assert responses.calls[0].request.headers["Authorization"] == "Bearer abc"


@responses.activate
def test_header_on_insert_and_reads():
    responses.post(BASE + "api/transcription", json={"insertedId": "n1"}, status=200)
    responses.get(BASE + "api/transcriptions", json=[], status=200)
    c = _client()
    c.insert_new_transcription({"title": "t"})
    c._get("api/transcriptions")
    assert all(call.request.headers["X-IDTAP-Client"] == "python/0.3.1" for call in responses.calls)


@responses.activate
def test_header_without_a_token():
    responses.get(BASE + "api/transcriptions", json=[], status=200)
    SwaraClient(auto_login=False)._get("api/transcriptions")
    assert responses.calls[0].request.headers["X-IDTAP-Client"] == "python/0.3.1"


@responses.activate
def test_write_guard_refusal_raises_with_reasons_and_is_not_retried():
    responses.post(BASE + "api/transcription", json=GUARD_BODY, status=409)
    with pytest.raises(WriteGuardError) as ei:
        _client().save_piece({"_id": "1"})
    assert ei.value.reasons == ["boundary", "tone"]
    assert ei.value.min_contract == "0.3.0"
    assert "boundary" in str(ei.value)
    assert isinstance(ei.value, requests.HTTPError)  # existing handlers still catch it
    assert len(responses.calls) == 1


@responses.activate
def test_save_transcription_surfaces_the_refusal_and_keeps_the_piece():
    responses.post(BASE + "api/transcription", json=GUARD_BODY, status=409)
    piece = Piece({"title": "kept", "_id": "abc123"})
    with pytest.raises(WriteGuardError):
        _client().save_transcription(piece, fill_duration=False)
    assert piece.title == "kept"
    assert len(responses.calls) == 1


@responses.activate
def test_other_409_is_a_plain_http_error():
    responses.post(BASE + "api/transcription", json={"error": "conflict"}, status=409)
    with pytest.raises(requests.HTTPError) as ei:
        _client().save_piece({"_id": "1"})
    assert not isinstance(ei.value, WriteGuardError)
