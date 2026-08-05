"""Numerical checks for Chapter 16 exercises."""

from __future__ import annotations

import math


def exercise_16_5(beta: float = 0.95) -> tuple[float, float]:
    odd = (1.0 - math.sqrt(1.0 - 0.8 * beta / (1.0 + beta))) / 2.0
    even = (1.0 - math.sqrt(1.0 - 0.8 / (1.0 + beta))) / 2.0
    return odd, even


if __name__ == "__main__":
    odd, even = exercise_16_5()
    print(f"Exercise 16.5 odd-spending tau:  {odd:.8f}")
    print(f"Exercise 16.5 even-spending tau: {even:.8f}")
