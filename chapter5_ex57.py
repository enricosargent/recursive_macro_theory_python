'Numerical check for Exercise 5.7.'

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

np.set_printoptions(precision=10, suppress=False, linewidth=140)

def exercise_57() -> None:

    def roots_for(g: float) -> tuple[float, float, float]:
        gamma1, gamma2 = (100.0, 50.0)
        a = (gamma1 + gamma2 - g) / gamma2
        b = gamma1 / gamma2
        roots = np.roots([1.0, -a, b])
        roots = np.sort(np.real_if_close(roots))
        pi_low, pi_high = (float(roots[0]), float(roots[1]))
        p0_min = 100.0 / (gamma1 - g - gamma2 * pi_low)
        return (pi_low, pi_high, p0_min)
    print('\nExercise 5.7')
    for g in (0.05, 0.075):
        pi_low, pi_high, p0_min = roots_for(g)
        print(f'g={g:g}: low inflation {pi_low:.12f}; high inflation {pi_high:.12f}; minimal p0 {p0_min:.12f}')

if __name__ == '__main__':
    exercise_57()
