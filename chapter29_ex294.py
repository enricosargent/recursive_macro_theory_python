'Illustrative numerical check for Exercise 29.4.'

from math import isfinite

def exercise_294():
    skill_sets = {'none': [1.0, 1.0, 1.0, 1.0, 1.0], 'mild': [0.8, 0.9, 1.0, 1.1, 1.2], 'high': [0.6, 0.8, 1.0, 1.2, 1.4]}
    benefits = [0.0, 0.2, 0.4, 0.5]
    rows = []
    for benefit in benefits:
        row = [benefit]
        for skills in skill_sets.values():
            rates = []
            for h in skills:
                theta = theta_skill_market(h, benefit)
                rates.append(float('nan') if not isfinite(theta) else unemployment(theta))
            row.append(sum(rates) / len(rates))
        rows.append(row)
    return rows

def theta_skill_market(h, benefit):
    target = 1.0 - benefit / h
    if target <= 0:
        return float('nan')

    def residual(theta):
        rhs = C / (1.0 - ALPHA) * ((R + S) / A * theta ** ALPHA + ALPHA * theta)
        return rhs - target
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

C = 0.5

ALPHA = 0.5

A = 1.0

R = 0.01

S = 0.05

def q(theta):
    return A * theta ** (-ALPHA)

if __name__ == '__main__':
    print('Exercise 29.4 aggregate unemployment')
    print('benefit       no spread     mild spread   high spread')
    for row in exercise_294():
        print(f'{row[0]:7.3f} {row[1]:13.6f} {row[2]:13.6f} {row[3]:13.6f}')
