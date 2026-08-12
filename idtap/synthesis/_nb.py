"""Numba shim.

numba is a required dependency of idtap; this shim exists only so the
package remains importable (with a loud warning and drastically slower
synthesis) in the unlikely event numba is unavailable — e.g. a brand-new
CPython version numba doesn't support yet.
"""
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
            "numba could not be imported; synthesis will run in pure Python "
            "and may be very slow. numba is a required dependency — check "
            "your installation (pip install numba).",
            RuntimeWarning,
            stacklevel=3,
        )
