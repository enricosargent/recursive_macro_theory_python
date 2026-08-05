'Numerical check for Exercise 5.5.'

from __future__ import annotations

import math

from dataclasses import dataclass

import numpy as np

np.set_printoptions(precision=10, suppress=False, linewidth=140)

def exercise_55() -> None:
    beta, rho1, rho2, gamma = (0.95, 1.2, -0.4, 0.5)
    a = np.array([[rho1, rho2, gamma], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    coeff = np.array([[1.0, 0.0, 0.0]]) @ np.linalg.inv(np.eye(3) - beta * a)
    print('\nExercise 5.5')
    print_matrix('discounted forecast coefficients on (y_t,y_{t-1},w_t)', coeff)

def print_matrix(name: str, x: np.ndarray) -> None:
    print(f'{name} =')
    print(np.array2string(x, precision=10, suppress_small=False))

if __name__ == '__main__':
    exercise_55()
