"""Numerical checks for Chapter 24 exercises."""

from __future__ import annotations

from math import sqrt


U_STAR = 5.0
THETA = 1.0


def best_response(x: float, u_star: float = U_STAR, theta: float = THETA) -> float:
    y = theta * (u_star + theta * x) / (1.0 + theta * theta)
    return min(10.0, max(0.0, y))


def ce_return(y: float, u_star: float = U_STAR) -> float:
    return -0.5 * (u_star * u_star + y * y)


def temptation_return(y: float, u_star: float = U_STAR, theta: float = THETA) -> float:
    yy = best_response(y, u_star, theta)
    unemployment = u_star - theta * (yy - y)
    return -0.5 * (unemployment * unemployment + yy * yy)


def chapter_243_values(delta: float = 0.95) -> dict[str, float]:
    y_ramsey = 0.0
    y_nash = THETA * U_STAR
    y_stick = 10.0
    v_ramsey = ce_return(y_ramsey)
    v_nash = ce_return(y_nash)
    v_worst = temptation_return(y_stick)
    v_abreu = (1.0 - delta) * ce_return(y_stick) + delta * v_ramsey
    v1 = ((1.0 - delta) / delta) * (temptation_return(y_stick) - ce_return(y_stick)) + v_worst
    return {
        "y_ramsey": y_ramsey,
        "y_nash": y_nash,
        "y_stick": y_stick,
        "v_ramsey": v_ramsey,
        "v_nash": v_nash,
        "v_worst": v_worst,
        "v_abreu": v_abreu,
        "v1_worst": v1,
        "delta_nash_cutoff": 1.0 / 3.0,
        "delta_worst_cutoff": 1.0 / 8.0,
    }


def worst_sequence(delta: float = 0.95) -> list[float]:
    values = []
    v = temptation_return(10.0)
    r_stick = ce_return(10.0)
    v_bar = ce_return(0.0)
    while True:
        values.append(v)
        v_next = (v - (1.0 - delta) * r_stick) / delta
        if v_next > v_bar:
            break
        v = v_next
    return values


def final_intermediate_y(delta: float = 0.95) -> float:
    seq = worst_sequence(delta)
    v_last = seq[-1]
    v_bar = ce_return(0.0)
    target_return = (v_last - delta * v_bar) / (1.0 - delta)
    return sqrt(max(0.0, -2.0 * target_return - U_STAR * U_STAR))


def low_delta_values(delta: float = 0.08) -> dict[str, float]:
    # Nondegenerate solution of equations (24.13.5)--(24.13.8).
    y_under = 8.2
    v_under = temptation_return(y_under)
    v_bar = ((1.0 - delta) / delta) * (temptation_return(y_under) - ce_return(y_under)) + v_under
    y_bar = sqrt(max(0.0, -2.0 * v_bar - U_STAR * U_STAR))
    return {"delta": delta, "y_under": y_under, "v_under": v_under, "y_bar": y_bar, "v_bar": v_bar}


def main() -> None:
    vals = chapter_243_values()
    print("Exercise 24.3, delta=.95")
    for key in ("y_ramsey", "y_nash", "v_ramsey", "v_nash", "v_worst", "v_abreu", "v1_worst"):
        print(f"{key} = {vals[key]:.12f}")
    seq = worst_sequence()
    print(f"worst_sequence_length = {len(seq)}")
    print(f"first_values = {[round(v, 6) for v in seq[:5]]}")
    print(f"last_value_before_switch = {seq[-1]:.12f}")
    print(f"final_intermediate_y = {final_intermediate_y():.12f}")
    print(f"delta_nash_cutoff = {vals['delta_nash_cutoff']:.12f}")
    print(f"delta_worst_cutoff = {vals['delta_worst_cutoff']:.12f}")
    print()
    low = low_delta_values()
    print("Exercise 24.3, delta=.08")
    for key in ("y_under", "v_under", "y_bar", "v_bar"):
        print(f"{key} = {low[key]:.12f}")


if __name__ == "__main__":
    main()
