'Numerical check for Exercise 2.13.'

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

def solve_discounted_riccati(
    a: np.ndarray,
    b: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    beta: float,
    tol: float = 1e-13,
    max_iter: int = 200000,
) -> np.ndarray:
    p = np.zeros_like(r, dtype=float)
    for _ in range(max_iter):
        gain = np.linalg.solve(q + beta * b.T @ p @ b, beta * b.T @ p @ a)
        p_new = r + beta * a.T @ p @ a - beta * a.T @ p @ b @ gain
        p_new = 0.5 * (p_new + p_new.T)
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = p_new
    raise RuntimeError('Riccati iteration did not converge')

def exercise_213() -> None:
    beta = 0.95
    rho1, rho2 = (1.3, -0.4)
    a = np.array([[1.0, 0.0, 0.0, 0.0], [30.0 / beta, 1.0 / beta, -1.0 / beta, 0.0], [5.0 * (1.0 - rho1 - rho2), 0.0, rho1, rho2], [0.0, 0.0, 1.0, 0.0]])
    b = np.array([[0.0], [1.0 / beta], [0.0], [0.0]])
    c = np.array([[0.0], [0.0], [0.05], [0.0]])
    r = np.diag([0.0, 1e-06, 0.0, 0.0])
    q = np.array([[1.0]])
    p = solve_discounted_riccati(a, b, r, q, beta)
    f = beta * np.linalg.solve(q + beta * b.T @ p @ b, b.T @ p @ a)
    acl = a - b @ f
    roots = np.roots([0.4, -1.3, 1.0])
    print('\nExercise 2.13')
    print(f'A eigenvalues {np.linalg.eigvals(a)}')
    print(f'zeros of income lag polynomial {roots}')
    print(f'optimal F {f}')
    print(f'closed-loop eigenvalues {np.linalg.eigvals(acl)}')
    shock_response = []
    dx = np.zeros(4)
    for h in range(12):
        shock = 1.0 if h == 0 else 0.0
        dx = acl @ dx + c.flatten() * shock
        shock_response.append((h + 1, float(-(f @ dx)[0]), float(dx[1])))
    print('responses (h, consumption, debt)')
    for row in shock_response:
        print(row)

if __name__ == '__main__':
    exercise_213()
