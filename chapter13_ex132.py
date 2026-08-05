import numpy as np


BETA = 0.75
DIVIDEND = np.array([0.0, 1.0])


def homogeneous_price(P):
    return BETA * np.linalg.solve(np.eye(2) - BETA * P, P @ DIVIDEND)


def hk_operator(p, Pa, Pb, optimistic=True):
    out = np.empty(2)
    for s in range(2):
        va = Pa[s, 0] * p[0] + Pa[s, 1] * (1.0 + p[1])
        vb = Pb[s, 0] * p[0] + Pb[s, 1] * (1.0 + p[1])
        out[s] = BETA * (max(va, vb) if optimistic else min(va, vb))
    return out


def fixed_point(Pa, Pb, optimistic=True, tol=1e-13):
    p = np.zeros(2)
    for _ in range(100000):
        new_p = hk_operator(p, Pa, Pb, optimistic)
        if np.max(np.abs(new_p - p)) < tol:
            return new_p
        p = new_p
    raise RuntimeError("fixed point did not converge")


def table(Pa, Pb):
    pa = homogeneous_price(Pa)
    pb = homogeneous_price(Pb)
    pbar = fixed_point(Pa, Pb, optimistic=True)
    pcheck = fixed_point(Pa, Pb, optimistic=False)
    hat_a = BETA * (Pa @ (DIVIDEND + pbar))
    hat_b = BETA * (Pb @ (DIVIDEND + pbar))
    return pa, pb, pbar, hat_a, hat_b, pcheck


def print_table(name, Pa, Pb):
    rows = table(Pa, Pb)
    labels = ["p_a", "p_b", "p_bar", "hat_p_a", "hat_p_b", "p_check"]
    print(name)
    for label, row in zip(labels, rows):
        print(f"{label:8s} {row[0]:.6f} {row[1]:.6f}")


if __name__ == "__main__":
    Pa_1 = np.array([[1 / 2, 1 / 2], [3 / 4, 1 / 4]], dtype=float)
    Pb_1 = np.array([[3 / 4, 1 / 4], [1 / 2, 1 / 2]], dtype=float)
    Pa_2 = np.array([[1 / 2, 1 / 2], [1 / 2, 1 / 2]], dtype=float)
    Pb_2 = np.array([[3 / 4, 1 / 4], [2 / 3, 1 / 3]], dtype=float)

    print_table("Exercise 13.2(a)", Pa_1, Pb_1)
    print()
    print_table("Exercise 13.2(b)", Pa_2, Pb_2)
