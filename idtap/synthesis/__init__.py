"""Offline synthesis of IDTAP transcriptions.

Renders Piece transcriptions to audio using ports of the web app's synth
engines (Karplus-Strong sitar + chikari, filter-feedback sarangi, Klatt
voice), driven by trajectory f0 curves, automation envelopes, and
articulation events.

Usage:
    from idtap.synthesis import synthesize_piece
    audio = synthesize_piece(piece, out='render.wav')

or equivalently:
    piece.synthesize(out='render.wav')

Install numba for fast rendering: pip install idtap[synth]
"""
from .render import (synthesize_piece, render_track, render_sitar,
                     render_sarangi, render_vocal, write_wav)
from .control import extract_track_control, TrackControl

__all__ = [
    'synthesize_piece', 'render_track', 'render_sitar', 'render_sarangi',
    'render_vocal', 'write_wav', 'extract_track_control', 'TrackControl',
]
