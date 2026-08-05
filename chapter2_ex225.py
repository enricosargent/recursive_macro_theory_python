"""Numerical check for Exercise 2.25."""

import math


if __name__ == "__main__":
    print("Exercise 2.25 Kalman gains")
    beta = 0.95
    for sigma1, sigma2 in [(1.0, 1.0), (2.0, 1.0)]:
        q, r = sigma1**2, sigma2**2
        sigma = 0.5 * (q + math.sqrt(q * q + 4.0 * q * r))
        k = sigma / (sigma + r)
        print(
            f"sigma1={sigma1:g}, sigma2={sigma2:g}: K={k:.10g}, "
            f"hidden-consumption coefficient={1 - beta * (1 - k):.10g}"
        )
