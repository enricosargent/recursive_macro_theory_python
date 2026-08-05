'Numerical check for Exercise 5.12.'

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

np.set_printoptions(precision=10, suppress=False, linewidth=140)

def firm_adjustment(demand_shock: bool) -> tuple[LQResult, np.ndarray, float]:
    beta, d, a0, a1 = (0.95, 2.0, 100.0, 1.0)
    if not demand_shock:
        h0, h1 = (200.0, 0.8)
        a = np.array([[1.0, 0.0, 0.0], [h0, h1, 0.0], [0.0, 0.0, 0.0]])
        b = np.array([[0.0], [0.0], [1.0]])
        r = np.zeros((3, 3))
        r[0, 2] = r[2, 0] = a0 / 2.0
        r[1, 2] = r[2, 1] = -a1 / 2.0
        r[2, 2] += -0.5 * d
        q = np.array([[-0.5 * d]])
        h = np.array([[0.0, 0.0, d / 2.0]])
        ans = solve_discounted_reward(a, b, r, q, h, beta)
        return (ans, -ans.f, 0.0)
    h0, h1, h2, rho, sigma = (200.0, 0.8, 2.0, 0.9, 0.05)
    a = np.array([[1.0, 0.0, 0.0, 0.0], [h0, h1, 0.0, h2], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, rho]])
    b = np.array([[0.0], [0.0], [1.0], [0.0]])
    c = np.array([[0.0], [0.0], [0.0], [sigma]])
    r = np.zeros((4, 4))
    r[0, 2] = r[2, 0] = a0 / 2.0
    r[1, 2] = r[2, 1] = -a1 / 2.0
    r[3, 2] = r[2, 3] = 0.5
    r[2, 2] += -0.5 * d
    q = np.array([[-0.5 * d]])
    h = np.array([[0.0, 0.0, d / 2.0, 0.0]])
    ans = solve_discounted_reward(a, b, r, q, h, beta)
    return (ans, -ans.f, stochastic_constant(beta, ans.p, c))

def print_matrix(name: str, x: np.ndarray) -> None:
    print(f'{name} =')
    print(np.array2string(x, precision=10, suppress_small=False))

def solve_discounted_reward(a: np.ndarray, b: np.ndarray, r: np.ndarray, q: np.ndarray, h: np.ndarray, beta: float, *, tol: float=1e-12, max_iter: int=200000) -> LQResult:
    """Solve a discounted problem that maximizes x'Rx + u'Qu + 2u'Hx."""
    p = np.zeros_like(r, dtype=float)
    f = np.zeros((b.shape[1], a.shape[0]))
    diff = math.inf
    for it in range(1, max_iter + 1):
        g = q + beta * b.T @ p @ b
        n = h + beta * b.T @ p @ a
        f = np.linalg.solve(g, n)
        p_new = r + beta * a.T @ p @ a - (h.T + beta * a.T @ p @ b) @ f
        diff = float(np.max(np.abs(p_new - p)))
        p = 0.5 * (p_new + p_new.T)
        if diff < tol:
            return LQResult(p=p, f=f, iterations=it, diff=diff)
    return LQResult(p=p, f=f, iterations=max_iter, diff=diff)

@dataclass
class LQResult:
    p: np.ndarray
    f: np.ndarray
    iterations: int
    diff: float

def stochastic_constant(beta: float, p: np.ndarray, c: np.ndarray) -> float:
    return beta / (1.0 - beta) * float(np.trace(p @ c @ c.T))

if __name__ == '__main__':
    ans, rule, const = firm_adjustment(True)
    print('Exercise 5.12')
    print_matrix("Pi for V=x'Pi x+d, x=(1,Y,y,u)'", ans.p)
    print_matrix('decision y_{t+1}=K x_t', rule)
    print(f'additive constant d {const:.10g}')
