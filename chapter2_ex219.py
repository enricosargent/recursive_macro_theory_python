'Figure generation for Exercise 2.19.'

from __future__ import annotations

import os

from pathlib import Path

import numpy as np

_cache = Path(__file__).resolve().parents[1] / '.mplcache'
_cache.mkdir(exist_ok=True)

os.environ.setdefault('MPLCONFIGDIR', str(_cache))

os.environ.setdefault('XDG_CACHE_HOME', str(_cache))

np.set_printoptions(precision=10, suppress=False)

def make_ex219_figure() -> None:
    if os.environ.get('RMT_WRITE_FIGURES') != '1':
        print('skipping figure regeneration; set RMT_WRITE_FIGURES=1 to rebuild it.')
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'matplotlib unavailable; skipping Exercise 2.19 figure: {exc}')
        return
    rng = np.random.default_rng(20260506)
    t_grid = np.arange(51)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for _ in range(10):
        theta = rng.normal(100.0, 10.0)
        iq = np.empty(51)
        mean = 100.0
        variance = 100.0
        for t in range(51):
            iq[t] = mean
            y = theta + rng.normal(0.0, 10.0)
            gain = variance / (variance + 100.0)
            mean = mean + gain * (y - mean)
            variance = 100.0 * variance / (variance + 100.0)
        ax.plot(t_grid, iq, color='black', linewidth=0.55, alpha=0.75)
        ax.axhline(theta, color='black', linewidth=0.25, alpha=0.35)
    ax.set_xlabel('date')
    ax.set_ylabel('IQ')
    ax.set_title('Exercise 2.19: ten posterior-mean paths')
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / 'figures' / 'ch02_ex219_iq_paths.pdf'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')

if __name__ == '__main__':
    make_ex219_figure()
