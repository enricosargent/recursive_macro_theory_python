'Numerical check for Exercise 5.14.'

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

np.set_printoptions(precision=10, suppress=False, linewidth=140)

def permanent_income(order: int, eps_asset: float) -> tuple[LQResult, np.ndarray, float, np.ndarray]:
    beta = 0.95
    gross_r = beta ** (-1)
    bliss = 1000.0
    sigma_y = 0.05
    if order == 2:
        rho1, rho2 = (1.2, -0.4)
        n = 4
        a = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, gross_r, 1.0, 0.0], [1.0 - rho1 - rho2, 0.0, rho1, rho2], [0.0, 0.0, 1.0, 0.0]])
        b = np.array([[0.0], [-1.0], [0.0], [0.0]])
        c = np.array([[0.0], [0.0], [sigma_y], [0.0]])
    else:
        rho1, rho2 = (0.55, 0.3)
        n = 6
        a = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, gross_r, 1.0, 0.0, 0.0, 0.0], [1.0 - rho1 - rho2, 0.0, rho1, 0.0, 0.0, rho2], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        b = np.array([[0.0], [-1.0], [0.0], [0.0], [0.0], [0.0]])
        c = np.array([[0.0], [0.0], [sigma_y], [0.0], [0.0], [0.0]])
    r = np.zeros((n, n))
    r[0, 0] = 0.5 * bliss ** 2
    r[1, 1] = 0.5 * eps_asset
    q = np.array([[0.5]])
    h = np.zeros((1, n))
    h[0, 0] = -0.5 * bliss
    ans = solve_discounted_loss(a, b, r, q, h, beta, tol=5e-11)
    c_rule = -ans.f
    a_next_rule = np.zeros((1, n))
    a_next_rule[0, 1] = gross_r
    a_next_rule[0, 2] = 1.0
    a_next_rule -= c_rule
    d_const = stochastic_constant(beta, ans.p, c)
    eigs = np.linalg.eigvals(a - b @ ans.f)
    return (ans, np.vstack([c_rule, a_next_rule]), d_const, eigs)

def print_matrix(name: str, x: np.ndarray) -> None:
    print(f'{name} =')
    print(np.array2string(x, precision=10, suppress_small=False))

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

def stochastic_constant(beta: float, p: np.ndarray, c: np.ndarray) -> float:
    return beta / (1.0 - beta) * float(np.trace(p @ c @ c.T))

@dataclass
class LQResult:
    p: np.ndarray
    f: np.ndarray
    iterations: int
    diff: float

if __name__ == '__main__':
    print('Exercise 5.14')
    for eps_asset in (1e-06, 0.0):
        ans, rules, d_const, eigs = permanent_income(4, eps_asset)
        print(f"epsilon={eps_asset:g}; state x=(1,a,y,y_-1,y_-2,y_-3)'")
        print_matrix("P for loss value L=x'Px+d", ans.p)
        print_matrix('rows: c_t rule, a_{t+1} rule', rules)
        print(f'additive loss constant d {d_const:.10g}')
        print(f'closed-loop eigenvalues {eigs}')
