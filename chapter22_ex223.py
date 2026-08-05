'Numerical check for Exercise 22.3.'

from __future__ import annotations

from pathlib import Path

import numpy as np


def exercise_223(beta: float=0.8) -> dict[str, float]:
    a = 2.5
    b = -8.0
    c = 6.0
    roots = np.roots([a, b, c])
    c_high = float(min((root.real for root in roots if root.real > 1.0 - 1e-12)))
    c_low = 2.0 - c_high
    aut_ex_ante = 0.5 * (u_223(2.0) + u_223(0.0)) / (1.0 - beta)
    value = 0.5 * (u_223(c_high) + u_223(c_low)) / (1.0 - beta)
    full_insurance_threshold = (u_223(2.0) - u_223(1.0)) / (u_223(2.0) - 0.5 * (u_223(2.0) + u_223(0.0)))
    return {'c_high': c_high, 'c_low': c_low, 'aut_ex_ante': aut_ex_ante, 'value': value, 'threshold': full_insurance_threshold}

def u_223(c: float) -> float:
    return 4.0 * c - 0.5 * c * c

if __name__ == '__main__':
    ex223 = exercise_223()
    print('Exercise 22.3')
    for key, value in ex223.items():
        print(f'{key} = {value:.12f}')
