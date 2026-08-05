"""Numerical checks for Chapter 23 exercises.

The original Matlab files cited in the source are not present in this
workspace.  This script implements the Hopenhayn-Nicolini calibration and a
plain Python value-iteration approximation for Exercise 23.1.
"""

from __future__ import annotations

from math import exp, isfinite, log, sqrt


BETA = 0.999
SIGMA = 0.5
WAGE = 100.0


def u(c: float) -> float:
    return c ** (1.0 - SIGMA) / (1.0 - SIGMA)


def u_inv(x: float) -> float:
    return (x * (1.0 - SIGMA)) ** (1.0 / (1.0 - SIGMA))


def bisect(fn, lo: float, hi: float, tol: float = 1e-14) -> float:
    flo = fn(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = fn(mid)
        if abs(fmid) < tol or hi - lo < tol:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)


def golden_min(fn, lo: float, hi: float, tol: float = 1e-5) -> tuple[float, float]:
    gr = (sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = fn(c)
    fd = fn(d)
    for _ in range(180):
        if hi - lo < tol:
            break
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = fn(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = fn(d)
    x = 0.5 * (lo + hi)
    return x, fn(x)


def interpolate(grid: list[float], values: list[float], x: float) -> float:
    if x <= grid[0]:
        return values[0]
    if x >= grid[-1]:
        return values[-1]
    pos = (x - grid[0]) / (grid[-1] - grid[0]) * (len(grid) - 1)
    i = int(pos)
    frac = pos - i
    return (1.0 - frac) * values[i] + frac * values[i + 1]


def calibration() -> dict[str, float]:
    v_e = u(WAGE) / (1.0 - BETA)

    def equation_for_r(r: float) -> float:
        a_star = -log(0.9) / r
        v_aut = (-a_star + BETA * 0.1 * v_e) / (1.0 - BETA * 0.9)
        return BETA * r * 0.9 * (v_e - v_aut) - 1.0

    r = bisect(equation_for_r, 1e-6, 1e-2)
    a_star = -log(0.9) / r
    v_aut = (-a_star + BETA * 0.1 * v_e) / (1.0 - BETA * 0.9)
    v_bar = v_e - 1.0 / (BETA * r)
    return {"V_e": v_e, "r": r, "a_star": a_star, "V_aut": v_aut, "V_bar": v_bar}


def solve_contract(n: int = 900, max_iter: int = 260) -> dict[str, object]:
    cal = calibration()
    v_e = cal["V_e"]
    r = cal["r"]
    v_aut = cal["V_aut"]
    v_bar = cal["V_bar"]

    def p(a: float) -> float:
        return 1.0 - exp(-r * a)

    def effort(v_u: float) -> float:
        x = r * BETA * (v_e - v_u)
        if x <= 1.0:
            return 0.0
        return log(x) / r

    def consumption(v: float, v_u: float) -> tuple[float, float, float]:
        a = effort(v_u)
        pa = p(a)
        util = v + a - BETA * (pa * v_e + (1.0 - pa) * v_u)
        if util < 0.0 and util > -1e-7:
            util = 0.0
        if util < 0.0:
            return float("nan"), a, pa
        return u_inv(util), a, pa

    grid = [v_aut + (v_bar - v_aut) * i / (n - 1) for i in range(n)]
    costs = [u_inv(max(0.0, (1.0 - BETA) * (v - v_aut))) / (1.0 - BETA) for v in grid]
    costs[0] = 0.0
    policy_v = [v_aut for _ in grid]
    policy_c = [0.0 for _ in grid]
    policy_a = [cal["a_star"] for _ in grid]

    for _ in range(max_iter):
        new_costs = [0.0 for _ in grid]
        new_policy_v = [v_aut for _ in grid]
        new_policy_c = [0.0 for _ in grid]
        new_policy_a = [cal["a_star"] for _ in grid]
        for i, v in enumerate(grid):
            if i == 0:
                continue

            def objective(v_u: float) -> float:
                c, a, pa = consumption(v, v_u)
                if not isfinite(c):
                    return 1e100
                return c + BETA * (1.0 - pa) * interpolate(grid, costs, v_u)

            candidate_v, candidate_cost = golden_min(objective, v_aut, v_bar)
            candidates = [
                (candidate_v, candidate_cost),
                (v_aut, objective(v_aut)),
                (v_bar, objective(v_bar)),
            ]
            v_u, value = min(candidates, key=lambda item: item[1])
            c, a, _ = consumption(v, v_u)
            new_costs[i] = value
            new_policy_v[i] = v_u
            new_policy_c[i] = c
            new_policy_a[i] = a
        diff = max(abs(a - b) for a, b in zip(new_costs, costs))
        costs = new_costs
        policy_v = new_policy_v
        policy_c = new_policy_c
        policy_a = new_policy_a
        if diff < 1e-5:
            break

    def path(v0: float, horizon: int = 5) -> list[dict[str, float]]:
        out = []
        v = v0
        for t in range(horizon):
            v_u = interpolate(grid, policy_v, v)
            c = interpolate(grid, policy_c, v)
            a = interpolate(grid, policy_a, v)
            out.append(
                {
                    "t": float(t),
                    "V": v,
                    "replacement_ratio": c / WAGE,
                    "effort": a,
                    "hazard": p(a),
                    "V_next": v_u,
                }
            )
            v = v_u
        return out

    return {**cal, "paths": {v0: path(v0) for v0 in (16900.0, 16942.0, 16980.0)}}


def main() -> None:
    result = solve_contract()
    print("Exercise 23.1 calibration")
    for key in ("V_e", "r", "a_star", "V_aut", "V_bar"):
        print(f"{key} = {result[key]:.12f}")
    print()
    print("Exercise 23.1 Python value-iteration paths")
    for v0, rows in result["paths"].items():
        print(f"V0 = {v0:.0f}")
        for row in rows:
            print(
                "t={t:.0f}, V={V:.3f}, c/w={replacement_ratio:.6f}, "
                "a={effort:.3f}, p(a)={hazard:.6f}, V_next={V_next:.3f}".format(**row)
            )
        print()


if __name__ == "__main__":
    main()
