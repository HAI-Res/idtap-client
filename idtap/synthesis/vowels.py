"""Vowel formant target table, copied from the web app (Synths.vue).

Each entry is a pair of targets [start, end]; each target is
[f1, f2, f3, b1, b2, b3]. The voice glides from the start target to the
end target over SHWAH_TIME seconds at each trajectory onset (a crude
diphthong/onglide). Indices 1 and 3 hold their start target (no glide).
"""
from __future__ import annotations

from typing import List, Optional

VOWEL_PARAMS = [
    [[310, 2020, 2960, 45, 200, 400], [290, 2070, 2960, 60, 200, 400]],
    [[400, 1800, 2570, 50, 100, 140], [470, 1600, 2600, 50, 100, 140]],
    [[480, 1720, 2520, 70, 100, 200], [330, 2020, 2600, 55, 100, 200]],
    [[530, 1680, 2500, 60, 90, 200], [620, 1530, 2530, 60, 90, 200]],
    [[620, 1660, 2430, 70, 150, 320], [650, 1490, 2470, 70, 100, 320]],
    [[700, 1220, 2600, 130, 70, 160], [700, 1220, 2600, 130, 70, 160]],
    [[600, 990, 2570, 90, 100, 80], [630, 1040, 2600, 90, 100, 80]],
    [[620, 1220, 2550, 80, 50, 140], [620, 1220, 2550, 80, 50, 140]],
    [[540, 1100, 2300, 80, 70, 70], [450, 900, 2300, 80, 70, 70]],
    [[450, 1100, 2350, 80, 100, 80], [500, 1180, 2390, 80, 100, 80]],
    [[350, 1250, 2200, 65, 110, 140], [320, 900, 2200, 65, 110, 140]],
    [[470, 1270, 1540, 100, 60, 110], [420, 1310, 1540, 100, 60, 110]],
    [[660, 1200, 2550, 100, 70, 200], [400, 1880, 2500, 70, 100, 200]],
    [[640, 1230, 2550, 80, 70, 140], [420, 940, 2350, 80, 70, 80]],
    [[550, 960, 2400, 80, 50, 130], [360, 1820, 2450, 60, 50, 160]],
]

VOWELS = ['a', 'ā', 'i', 'ī', 'u', 'ū', 'ē', 'ai', 'ō', 'au', '_']
VOWEL_PARAM_IDXS = [7, 6, 1, 0, 9, 10, 2, 3, 8, 5, 7]

# Klatt worklet defaults for the upper formants (never varied by the app)
DEFAULT_F = [520.0, 1006.0, 2831.0, 3168.0, 4135.0, 5020.0]
DEFAULT_B = [76.0, 102.0, 72.0, 102.0, 816.0, 596.0]

SHWAH_TIME = 0.3  # seconds; formant glide time at each trajectory onset

# targets are held (no glide) for these vowelParams indices — matches
# `idx === 1 || idx === 3` in Synths.vue playKlattTraj
HOLD_TARGET_IDXS = (1, 3)


def vowel_targets(vowel: Optional[str]) -> tuple[List[float], List[float]]:
    """Return ([f1,f2,f3,b1,b2,b3] start, same end) for a vowel string."""
    v_idx = VOWELS.index(vowel) if vowel in VOWELS else 0
    idx = VOWEL_PARAM_IDXS[v_idx]
    s0 = [float(x) for x in VOWEL_PARAMS[idx][0]]
    if idx in HOLD_TARGET_IDXS:
        s1 = list(s0)
    else:
        s1 = [float(x) for x in VOWEL_PARAMS[idx][1]]
    return s0, s1
