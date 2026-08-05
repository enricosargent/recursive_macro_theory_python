'Numerical check for Exercise 14.20.'

from __future__ import annotations

import math

import os

from pathlib import Path

import numpy as np

_cache = Path(__file__).resolve().parents[1] / '.mplcache'
_cache.mkdir(exist_ok=True)

os.environ.setdefault('MPLCONFIGDIR', str(_cache))

os.environ.setdefault('XDG_CACHE_HOME', str(_cache))

def exercise_1420() -> None:
    print('\nExercise 14.20, lambda=1')
    for t in [1, 5, 100, 1000, 10000]:
        p_le_1 = 0.5 * (1.0 + math.erf(math.sqrt(t) / (2.0 * math.sqrt(2.0))))
        median = math.exp(-0.5 * t) if t < 1500 else 0.0
        print(f't={t:5d}: P(xi_t <= 1)={p_le_1:.12f}, median={median:.12g}')

if __name__ == '__main__':
    exercise_1420()
