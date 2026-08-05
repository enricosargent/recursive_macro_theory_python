'Numerical check for Exercise 5.3.'

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

np.set_printoptions(precision=10, suppress=False, linewidth=140)

def exercise_53() -> None:
    beta = 0.95
    gross_r = beta ** (-1)
    bliss = 30.0
    gamma = 1.0
    rho1, rho2 = (1.2, -0.3)
    a = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, rho1, rho2], [0.0, 0.0, 1.0, 0.0]])
    b = np.array([[0.0], [1.0], [0.0], [0.0]])
    c = np.array([[-bliss, gross_r, 1.0, 0.0]])
    d = np.array([[-1.0]])
    r = c.T @ c
    q = d.T @ d + np.array([[gamma]])
    h = d.T @ c
    ans = solve_discounted_loss(a, b, r, q, h, beta)
    c_rule = np.array([[0.0, gross_r, 1.0, 0.0]]) + ans.f
    print('\nExercise 5.3')
    print_matrix("F_i for i_t=-F_i x_t, x=(1,a,y,y_-1)'", ans.f)
    print_matrix('consumption rule c_t=C_c x_t', c_rule)
    print(f'closed-loop eigenvalues {np.linalg.eigvals(a - b @ ans.f)}')

def solve_discounted_loss(a: np.ndarray, b: np.ndarray, r: np.ndarray, q: np.ndarray, h: np.ndarray, beta: float, *, tol: float=1e-12, max_iter: int=200000) -> LQResult:
    """Solve a discounted regulator that minimizes x'Rx + u'Qu + 2u'Hx.

    The optimal rule is u_t = -F x_t and the value is x'Px plus, in stochastic
    cases, the usual additive variance constant.
    """
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

def print_matrix(name: str, x: np.ndarray) -> None:
    print(f'{name} =')
    print(np.array2string(x, precision=10, suppress_small=False))

@dataclass
class LQResult:
    p: np.ndarray
    f: np.ndarray
    iterations: int
    diff: float

if __name__ == '__main__':
    exercise_53()
