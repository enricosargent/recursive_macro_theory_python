"""Numerical checks for Chapter 6 search exercises."""

from __future__ import annotations

import numpy as np


np.set_printoptions(precision=10, suppress=False, linewidth=120)


def markov_wages(beta: float) -> tuple[np.ndarray, np.ndarray]:
    c = 1.0
    w = np.array([1, 2, 3, 4, 5.0])
    p = np.array(
        [
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.18, 0.8, 0.02, 0.0, 0.0],
            [0.25, 0.25, 0.0, 0.25, 0.25],
            [0.0, 0.0, 0.02, 0.8, 0.18],
            [0.0, 0.0, 0.0, 0.2, 0.8],
        ]
    )
    v = w / (1 - beta)
    for _ in range(100_000):
        accept = w / (1 - beta)
        reject = c + beta * p @ v
        new_v = np.maximum(accept, reject)
        if np.max(np.abs(new_v - v)) < 1e-12:
            v = new_v
            break
        v = new_v
    policy = np.where(w / (1 - beta) >= c + beta * p @ v, "accept", "reject")
    return v, policy


def main() -> None:
    for beta in (0.95, 0.99):
        v, policy = markov_wages(beta)
        print(f"Exercise 6.25 beta={beta}")
        print("value", v)
        print("policy", policy)


if __name__ == "__main__":
    main()
