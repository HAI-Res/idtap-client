"""A Piece must serialise whatever save_transcription puts on it."""
import json

from idtap.classes.piece import Piece
from idtap.enums import Instrument


def test_solo_instrument_enum_serialises():
    """save_transcription defaults solo_instrument to instrumentation[0], which
    is an Instrument. Left as an enum it reaches json.dumps and the upload
    fails at the POST, after every local check has passed."""
    piece = Piece({'instrumentation': [Instrument.Vocal_M]})
    piece.solo_instrument = piece.instrumentation[0]
    payload = piece.to_json()
    assert payload['soloInstrument'] == Instrument.Vocal_M.value
    json.dumps(payload)          # the step that actually failed


def test_solo_instrument_string_is_left_alone():
    piece = Piece({'instrumentation': [Instrument.Vocal_M],
                   'soloInstrument': 'Vocal (M)'})
    assert piece.to_json()['soloInstrument'] == 'Vocal (M)'


def test_whole_payload_is_json_serialisable():
    piece = Piece({'instrumentation': [Instrument.Vocal_M],
                   'title': 'serialisation check'})
    piece.solo_instrument = piece.instrumentation[0]
    json.dumps(piece.to_json(), allow_nan=False)
