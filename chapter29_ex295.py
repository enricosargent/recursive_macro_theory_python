'Illustrative numerical check for Exercise 29.5.'

from math import isfinite

R = 0.01
BETA = 1.0 / (1.0 + R)
S = 0.05
C = 0.5
ALPHA = 0.5
PHI = ALPHA
A = 1.0

def exercise_295():
    distributions = {'none': [1.0], 'mild': [0.8, 0.9, 1.0, 1.1, 1.2], 'high': [0.6, 0.8, 1.0, 1.2, 1.4]}
    benefits = [0.0, 0.2, 0.4]
    rows = []
    for benefit in benefits:
        row = [benefit]
        for values in distributions.values():
            theta = theta_match_value(values, benefit)
            ubar = benefit + PHI * theta * C / (1.0 - PHI)
            accept_prob = sum((1 for p in values if p >= ubar)) / len(values)
            row.append(unemployment(theta, accept_prob))
        rows.append(row)
    return rows

def theta_match_value(prod_values, benefit):
    D = 1.0 - BETA * (1.0 - S)

    def residual(theta):
        ubar = benefit + PHI * theta * C / (1.0 - PHI)
        expected_surplus = sum((max(p - ubar, 0.0) for p in prod_values)) / len(prod_values)
        expected_surplus /= D
        return BETA * q(theta) * (1.0 - PHI) * expected_surplus - C
    return bisect(residual)

def unemployment(theta, accept_prob=1.0):
    return S / (S + theta * q(theta) * accept_prob)

def bisect(func, lo=1e-10, hi=10000.0, tol=1e-13):
    flo = func(lo)
    fhi = func(hi)
    while flo * fhi > 0:
        hi *= 2.0
        fhi = func(hi)
        if hi > 1000000000000.0:
            raise RuntimeError('root is not bracketed')
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        fmid = func(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)

def q(theta):
    return A * theta ** (-ALPHA)

if __name__ == '__main__':
    print('Exercise 29.5 aggregate unemployment')
    print('benefit       no spread     mild spread   high spread')
    for row in exercise_295():
        print(f'{row[0]:7.3f} {row[1]:13.6f} {row[2]:13.6f} {row[3]:13.6f}')
