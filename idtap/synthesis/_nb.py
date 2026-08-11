"""Numba shim: use @njit when numba is available, otherwise fall back to
pure Python (functional but slow — installing ``idtap[synth]`` is strongly
recommended for rendering anything longer than a few seconds)."""
from __future__ import annotations

import warnings

try:
    from numba import njit  # type: ignore
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrap(fn):
            return fn
        return wrap


def warn_if_slow() -> None:
    if not HAVE_NUMBA:
        warnings.warn(
            "numba is not installed; synthesis will run in pure Python and "
            "may be very slow. Install with: pip install idtap[synth]",
            RuntimeWarning,
            stacklevel=3,
        )
