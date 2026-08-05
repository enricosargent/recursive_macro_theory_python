"""Numerical checks for Chapter 26 exercises."""

from __future__ import annotations

from math import sqrt


def u_l(t: float) -> float:
    return -0.5 * (t - 0.5) ** 2


def u_w(t: float) -> float:
    return -0.5 * t * t


def u_s(t: float) -> float:
    return u_w(t) - u_l(t)


T_N = 0.5
BETA_C = sqrt((u_l(T_N) - u_l(0.0)) / (u_s(0.0) - u_s(T_N)))


def cutoffs(beta: float) -> dict[str, float]:
    v_l_n = u_l(T_N) / (1.0 - beta)
    v_s_n = u_s(T_N) / (1.0 - beta)
    v_star = (v_l_n - u_l(0.0)) / beta
    v_starstar = (u_l(0.0) + beta * (u_s(0.0) - u_s(T_N))) / (1.0 - beta)
    v_max = beta * (-u_s(T_N)) / (1.0 - beta)
    return {"v_L_N": v_l_n, "v_S_N": v_s_n, "v_star": v_star, "v_starstar": v_starstar, "v_max": v_max}


def frontier(beta: float, v_l: float) -> tuple[str, float, float]:
    c = cutoffs(beta)
    if v_l <= c["v_starstar"]:
        return "I/III", 0.0, -v_l
    t = 0.5 - sqrt(max(0.0, -2.0 * (v_l - beta * 0.125 / (1.0 - beta))))
    p = u_s(t) + beta * c["v_S_N"]
    return "II", t, p


def region_iii_values(beta: float, v_l: float) -> dict[str, float]:
    c = cutoffs(beta)
    y = c["v_star"]
    e0 = v_l
    e1 = c["v_star"]
    return {"v_L": v_l, "e_S": e0, "e_S_prime": e1, "y": y}


def main() -> None:
    beta = 0.9
    print("Exercise 26.1")
    print(f"beta_c = {BETA_C:.12f}")
    c = cutoffs(beta)
    print(f"beta = {beta:.3f}")
    for key, value in c.items():
        print(f"{key} = {value:.12f}")
    print("frontier samples")
    for v_l in (0.0, 0.5 * c["v_star"], c["v_star"], c["v_starstar"], 0.5 * (c["v_starstar"] + c["v_max"]), c["v_max"]):
        region, tariff, value_s = frontier(beta, v_l)
        print(f"v_L={v_l:.12f}, region={region}, t_L={tariff:.12f}, P(v_L)={value_s:.12f}")
    vals = region_iii_values(beta, 0.5 * c["v_star"])
    print("region III example")
    for key, value in vals.items():
        print(f"{key} = {value:.12f}")


if __name__ == "__main__":
    main()
