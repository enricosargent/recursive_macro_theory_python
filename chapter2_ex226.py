'Figure generation for Exercise 2.26.'

from __future__ import annotations

import math

import os

from pathlib import Path

import numpy as np

_cache = Path(__file__).resolve().parents[1] / '.mplcache'
_cache.mkdir(exist_ok=True)

os.environ.setdefault('MPLCONFIGDIR', str(_cache))

os.environ.setdefault('XDG_CACHE_HOME', str(_cache))

np.set_printoptions(precision=10, suppress=False)

def make_ex226_figure() -> None:
    if os.environ.get('RMT_WRITE_FIGURES') != '1':
        print('skipping figure regeneration; set RMT_WRITE_FIGURES=1 to rebuild it.')
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'matplotlib unavailable; skipping Exercise 2.26 figure: {exc}')
        return
    rng = np.random.default_rng(20260506)
    t_grid = np.arange(51)
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
    g_even = np.array([[0.9, 0.1]])
    g_odd = np.array([[0.01, 0.99]])
    for _ in range(10):
        theta = rng.multivariate_normal(np.array([100.0, 100.0]), 100.0 * np.eye(2))
        mean = np.array([100.0, 100.0])
        covariance = 100.0 * np.eye(2)
        path = np.empty((51, 2))
        for t in range(51):
            path[t] = mean
            g = g_even if t % 2 == 0 else g_odd
            y = (g @ theta).item() + rng.normal(0.0, math.sqrt(50.0))
            innovation_var = (g @ covariance @ g.T).item() + 50.0
            gain = covariance @ g.T / innovation_var
            mean = mean + gain[:, 0] * (y - (g @ mean).item())
            covariance = covariance - gain @ g @ covariance
        axs[0].plot(t_grid, path[:, 0], color='black', linewidth=0.55, alpha=0.75)
        axs[0].axhline(theta[0], color='black', linewidth=0.25, alpha=0.35)
        axs[1].plot(t_grid, path[:, 1], color='black', linewidth=0.55, alpha=0.75)
        axs[1].axhline(theta[1], color='black', linewidth=0.25, alpha=0.35)
    axs[0].set_ylabel('math')
    axs[1].set_ylabel('verbal')
    axs[1].set_xlabel('date')
    axs[0].set_title('Exercise 2.26: ten posterior-mean paths')
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / 'figures' / 'ch02_ex226_iq_paths.pdf'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')

if __name__ == '__main__':
    make_ex226_figure()
