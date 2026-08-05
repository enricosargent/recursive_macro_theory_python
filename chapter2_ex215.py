"""Numerical check for Exercise 2.15."""

import math


if __name__ == "__main__":
    print("Exercise 2.15 peak locations")
    for m, n in [(10, 10), (10, 40), (40, 10), (120, 30)]:
        omega = math.acos((n - m) / (n + m))
        print(f"m={m}, n={n}: |omega|={omega:.10g}")
