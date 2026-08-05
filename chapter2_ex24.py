'Numerical check for Exercise 2.4.'

from __future__ import annotations

import os

from pathlib import Path

import numpy as np

_cache = Path(__file__).resolve().parents[1] / '.mplcache'
_cache.mkdir(exist_ok=True)

os.environ.setdefault('MPLCONFIGDIR', str(_cache))

os.environ.setdefault('XDG_CACHE_HOME', str(_cache))

np.set_printoptions(precision=10, suppress=False)

def solve_discrete_lyapunov_numpy(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    lhs = np.eye(n * n) - np.kron(a, a)
    return np.linalg.solve(lhs, q.reshape(-1, order='F')).reshape((n, n), order='F')

def exercise_24() -> None:
    beta = 0.95
    cases = [([1.2, -0.3, 0.0, 0.0], 10.0, 1.0), ([1.2, -0.3, 0.0, 0.0], 10.0, 2.0), ([0.9, 0.0, 0.0, 0.0], 5.0, 1.0), ([0.2, 0.0, 0.0, 0.5], 5.0, 1.0), ([0.8, 0.3, 0.0, 0.0], 5.0, 1.0)]
    print('\nExercise 2.4')
    for i, (rho_values, mu, c) in enumerate(cases, start=1):
        rho = np.array(rho_values, dtype=float)
        a0 = np.array([[rho[0], rho[1], rho[2], rho[3]], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        eigs = np.linalg.eigvals(a0)
        print(f'case {i}: eigenvalues {eigs}')
        if np.any(np.abs(eigs) >= 1.0):
            print('  not covariance stationary')
            continue
        cvec = np.array([[c], [0.0], [0.0], [0.0]])
        sigma = solve_discrete_lyapunov_numpy(a0, cvec @ cvec.T)
        aug = np.zeros((5, 5))
        aug[:4, :4] = a0
        aug[0, 4] = mu * (1.0 - rho.sum())
        aug[4, 4] = 1.0
        forecast = np.linalg.matrix_power(aug, 5)[0]
        discounted = (np.array([[1, 0, 0, 0, 0]]) @ np.linalg.inv(np.eye(5) - beta * aug))[0]
        autocovs = [(np.linalg.matrix_power(a0, k) @ sigma)[0, 0] for k in (1, 5, 10)]
        print(f'  mean {mu:.10g}; variance {sigma[0, 0]:.10g}')
        print(f'  y(t+5) intercept {forecast[4]:.10g}; h {forecast[:4]}')
        print(f'  discounted intercept {discounted[4]:.10g}; htilde {discounted[:4]}')
        print(f'  autocovariances k=1,5,10 {autocovs}')

if __name__ == '__main__':
    exercise_24()
