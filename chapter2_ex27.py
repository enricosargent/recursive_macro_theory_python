'Figure generation for Exercise 2.7.'

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

def make_arma_figure() -> None:
    if os.environ.get('RMT_WRITE_FIGURES') != '1':
        print('skipping figure regeneration; set RMT_WRITE_FIGURES=1 to rebuild it.')
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'matplotlib unavailable; skipping ARMA figure: {exc}')
        return
    cases = [('a', [], [1.0]), ('b', [], [1.0, 0.5]), ('c', [], [1.0, 0.5, 0.4]), ('d', [0.999], [1.0, -0.4]), ('e', [0.8], [1.0, 0.5, 0.4]), ('f', [-0.8], [1.0]), ('g', [], [1.0, -0.6])]
    rng = np.random.default_rng(20260506)
    sim_shocks = rng.normal(size=80)
    omega = np.linspace(0.0, math.pi, 500)
    fig, axs = plt.subplots(len(cases), 3, figsize=(7.0, 11.5))
    for row, (label, phi, theta) in enumerate(cases):
        sim = simulate_arma(phi, theta, sim_shocks)
        impulse_shocks = np.zeros(40)
        impulse_shocks[0] = 1.0
        impulse = simulate_arma(phi, theta, impulse_shocks)
        den = np.ones_like(omega, dtype=complex)
        for j, coef in enumerate(phi, start=1):
            den -= coef * np.exp(-1j * omega * j)
        num = np.zeros_like(omega, dtype=complex)
        for j, coef in enumerate(theta):
            num += coef * np.exp(-1j * omega * j)
        spectrum = np.abs(num / den) ** 2
        ax = axs[row, 0]
        ax.plot(np.arange(80), sim, color='black', linewidth=0.65)
        ax.set_ylabel(label, rotation=0, labelpad=10)
        ax.set_yticks([])
        if row == 0:
            ax.set_title('simulation')
        ax = axs[row, 1]
        ax.stem(np.arange(20), impulse[:20], linefmt='k-', markerfmt='ko', basefmt='k-')
        ax.set_yticks([])
        if row == 0:
            ax.set_title('impulse')
        ax = axs[row, 2]
        ax.plot(omega, spectrum / max(np.max(spectrum), 1e-12), color='black', linewidth=0.75)
        ax.set_xlim(0, math.pi)
        ax.set_yticks([])
        if row == 0:
            ax.set_title('spectrum')
    for ax in axs[-1, :]:
        ax.set_xlabel('period / frequency')
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / 'figures' / 'ch02_ex27_arma_panels.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')

def simulate_arma(phi: list[float], theta: list[float], shocks: np.ndarray) -> np.ndarray:
    y = np.zeros_like(shocks)
    for t in range(len(shocks)):
        ar_part = 0.0
        for j, coef in enumerate(phi, start=1):
            if t - j >= 0:
                ar_part += coef * y[t - j]
        ma_part = 0.0
        for j, coef in enumerate(theta):
            if t - j >= 0:
                ma_part += coef * shocks[t - j]
        y[t] = ar_part + ma_part
    return y

if __name__ == '__main__':
    make_arma_figure()
