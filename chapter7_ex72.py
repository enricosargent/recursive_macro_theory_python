'Numerical check for Exercise 7.2.'

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

np.set_printoptions(precision=10, suppress=False, linewidth=140)

def firm_rule(h0: float, h1: float, nfirms: int=1) -> QuadResult:
    a0, a1, beta, d = (100.0, 0.05, 0.95, 10.0)
    a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [h0, 0.0, h1]])
    b = np.array([[0.0], [1.0], [0.0]])
    r = np.zeros((3, 3))
    r[0, 1] = r[1, 0] = a0 / 2.0
    r[1, 2] = r[2, 1] = -a1 / 2.0
    r[1, 1] = -0.5 * d
    q = np.array([[-0.5 * d]])
    h = np.array([[0.0, d / 2.0, 0.0]])
    return solve_reward(a, b, r, q, h, beta)

@dataclass
class QuadResult:
    p: np.ndarray
    k: np.ndarray
    iterations: int
    diff: float

def solve_reward(a: np.ndarray, b: np.ndarray, r: np.ndarray, q: np.ndarray, h: np.ndarray, beta: float) -> QuadResult:
    """Maximize sum beta^t [x'Rx + u'Qu + 2u'Hx], x^+=Ax+Bu."""
    p = np.zeros_like(r, dtype=float)
    k = np.zeros((b.shape[1], a.shape[0]))
    for it in range(1, 200000):
        g = q + beta * b.T @ p @ b
        n = h + beta * b.T @ p @ a
        k = -np.linalg.solve(g, n)
        p_new = r + beta * a.T @ p @ a + (h.T + beta * a.T @ p @ b) @ k
        diff = float(np.max(np.abs(p_new - p)))
        p = 0.5 * (p_new + p_new.T)
        if diff < 1e-12:
            return QuadResult(p=p, k=k, iterations=it, diff=diff)
    return QuadResult(p=p, k=k, iterations=it, diff=diff)

import numpy as np

if __name__ == '__main__':
    print('Exercise 7.2')
    for pair in [(94.0888, 0.9211), (93.22, 0.9433), (95.08187459215024, 0.95245906270392)]:
        ans = firm_rule(*pair)
        implied = np.array([ans.k[0, 0], ans.k[0, 1] + ans.k[0, 2]])
        print(f'belief {pair}; firm rule {ans.k}; implied aggregate {implied}')
