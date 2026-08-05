"""Numerical checks for Chapter 18 exercises."""

from __future__ import annotations

import numpy as np


def crra_utility(c: np.ndarray, sigma: float) -> np.ndarray:
    values = np.full_like(c, -1.0e18, dtype=float)
    mask = c > 1.0e-14
    values[mask] = c[mask] ** (1.0 - sigma) / (1.0 - sigma)
    return values


def exercise_18_2(step: float = 0.005) -> None:
    beta = 0.95
    prob_same = 0.8
    gross_return = 1.02
    wage_good = 1.4
    wage_bad = 1.0
    mobility_cost = 0.9
    sigma = 4.0

    grid = np.round(np.arange(0.0, 3.0 + 0.5 * step, step), 12)
    n = grid.size
    value_good = np.zeros(n)
    value_bad = np.zeros(n)
    policy_good = np.zeros(n, dtype=int)
    policy_bad = np.zeros(n, dtype=int)
    move_bad = np.zeros(n, dtype=bool)

    for iteration in range(5000):
        continuation_good = prob_same * value_good + (1.0 - prob_same) * value_bad
        continuation_bad = (1.0 - prob_same) * value_good + prob_same * value_bad

        new_good = np.empty(n)
        new_bad = np.empty(n)
        new_policy_good = np.empty(n, dtype=int)
        new_policy_bad = np.empty(n, dtype=int)
        new_move_bad = np.empty(n, dtype=bool)

        for i, assets in enumerate(grid):
            cash_good = gross_return * assets + wage_good
            values_good = (
                crra_utility(cash_good - grid, sigma)
                + beta * continuation_good
            )
            values_good[grid > cash_good + 1.0e-12] = -1.0e18
            best_good = int(np.argmax(values_good))
            new_good[i] = values_good[best_good]
            new_policy_good[i] = best_good

            cash_stay = gross_return * assets + wage_bad
            values_stay = (
                crra_utility(cash_stay - grid, sigma)
                + beta * continuation_bad
            )
            values_stay[grid > cash_stay + 1.0e-12] = -1.0e18
            best_stay = int(np.argmax(values_stay))

            cash_move = gross_return * assets + wage_good - mobility_cost
            values_move = (
                crra_utility(cash_move - grid, sigma)
                + beta * continuation_good
            )
            values_move[grid > cash_move + 1.0e-12] = -1.0e18
            best_move = int(np.argmax(values_move))

            if values_move[best_move] > values_stay[best_stay]:
                new_bad[i] = values_move[best_move]
                new_policy_bad[i] = best_move
                new_move_bad[i] = True
            else:
                new_bad[i] = values_stay[best_stay]
                new_policy_bad[i] = best_stay
                new_move_bad[i] = False

        diff = max(
            np.max(np.abs(new_good - value_good)),
            np.max(np.abs(new_bad - value_bad)),
        )
        value_good = new_good
        value_bad = new_bad
        policy_good = new_policy_good
        policy_bad = new_policy_bad
        move_bad = new_move_bad
        if diff < 1.0e-10:
            break

    first_move = grid[np.where(move_bad)[0][0]]
    print("Exercise 18.2")
    print(f"first moving asset level: {first_move:.3f}")
    for assets in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        i = int(round(assets / step))
        bad_cash = gross_return * assets + (
            wage_good - mobility_cost if move_bad[i] else wage_bad
        )
        print(
            f"A={assets:.1f}: "
            f"Ap_g={grid[policy_good[i]]:.3f}, "
            f"c_g={gross_return * assets + wage_good - grid[policy_good[i]]:.3f}, "
            f"Ap_b={grid[policy_bad[i]]:.3f}, "
            f"c_b={bad_cash - grid[policy_bad[i]]:.3f}, "
            f"move={move_bad[i]}"
        )

    transition = np.zeros((2 * n, 2 * n))
    for i in range(n):
        next_good = policy_good[i]
        transition[i, next_good] = prob_same
        transition[i, n + next_good] = 1.0 - prob_same

        next_bad = policy_bad[i]
        if move_bad[i]:
            transition[n + i, next_bad] = prob_same
            transition[n + i, n + next_bad] = 1.0 - prob_same
        else:
            transition[n + i, next_bad] = 1.0 - prob_same
            transition[n + i, n + next_bad] = prob_same

    invariant = np.ones(2 * n) / (2 * n)
    for _ in range(200000):
        updated = invariant @ transition
        if np.max(np.abs(updated - invariant)) < 1.0e-14:
            invariant = updated
            break
        invariant = updated

    assets_all = np.r_[grid, grid]
    order = np.argsort(assets_all)
    cdf = np.cumsum(invariant[order])
    quantiles = {
        q: assets_all[order][np.searchsorted(cdf, q)]
        for q in [0.25, 0.50, 0.75, 0.90]
    }

    print(f"mass good wage: {invariant[:n].sum():.4f}")
    print(f"mass bad wage:  {invariant[n:].sum():.4f}")
    print(f"mean assets:    {np.dot(invariant, assets_all):.4f}")
    for q, value in quantiles.items():
        print(f"asset quantile {q:.2f}: {value:.3f}")


if __name__ == "__main__":
    exercise_18_2()
