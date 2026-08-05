"""Numerical check for Exercise 27.1."""


def alpha(real_R, inflation_gross):
    # Text formula alpha=(R-1)/(R-p_{t-1}/p_t), where inflation_gross=p_t/p_{t-1}.
    return (real_R - 1.0) / (real_R - 1.0 / inflation_gross)


if __name__ == "__main__":
    print("Exercise 27.1")
    print(f"alpha, zero net inflation = {alpha(1.02, 1.0):.12f}")
    print(f"alpha, 100 percent net inflation = {alpha(1.02, 2.0):.12f}")
