'Numerical check for Exercise 21.2.'

from __future__ import annotations

import os

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'figures'
TIKZ = ROOT / 'tikz'
FIGURES.mkdir(exist_ok=True)
TIKZ.mkdir(exist_ok=True)

_cache = Path(__file__).resolve().parents[1] / '.mplcache'
_cache.mkdir(exist_ok=True)

os.environ.setdefault('MPLCONFIGDIR', str(_cache))

os.environ.setdefault('XDG_CACHE_HOME', str(_cache))

def run_contract_calculations() -> None:
    data = contract()
    y = data['y']
    c_bar = data['c_bar']
    w_bar = data['w_bar']
    profits = data['profits']
    cum_pi = data['cum_pi']
    pi = data['pi']
    beta = data['beta']
    display = [0, 4, 9, 14, 19]
    print('Chapter 21 calculations')
    print(f'v_aut = {data['v_aut']:.15f}')
    print(f'v_pool = {data['v_pool']:.15f}')
    print(f'E y = {data['ey']:.15f}')
    for i in display:
        print(f's={i + 1:2d}: y={y[i]:.0f}, c_bar={c_bar[i]:.6f}, w_bar={w_bar[i]:.6f}, P(w_bar)={profits[i]:.6f}')
    print(f'P(v_aut) = {profits[0]:.15f}')
    print(f'P(w_S) = {profits[-1]:.15f}')
    zero = data['zero_profit']
    print('Zero-profit contract')
    print(f'k = {zero['k']}')
    print(f'tilde_c = {zero['tilde_c']:.15f}')
    print(f'v0 = {zero['v0']:.15f}')
    print('CDF values at c_1,c_5,c_10,c_15,c_20')
    for t in [0, 5, 10, 500]:
        vals = cdf_values(cum_pi, t, display)
        print(f't={t}: ' + ' '.join((f'{val:.12g}' for val in vals)))
    balances = bank_balances(pi, y, c_bar, beta, 100)
    print('Autarky-value bank balances')
    for t in [0, 1, 5, 10, 25, 50, 100]:
        print(f'B_{t} = {balances[t]:.15g}')
    k = int(zero['k']) - 1
    c_zero = c_bar.copy()
    c_zero[:k] = float(zero['tilde_c'])
    asymptote = (c_bar[-1] - float(data['ey'])) / (beta ** (-1) - 1.0)
    print('Zero-profit bank balances')
    print(f'zero-profit limiting balance = {asymptote:.15g}')
    expected_wage = np.array([densities(cum_pi, t) @ c_bar for t in range(101)])
    print('Expected company wage by tenure')
    for t in [0, 1, 5, 10, 20, 40, 100]:
        print(f'tenure {t}: {expected_wage[t]:.6f}')
    make_tikz_figures(data)

def contract(beta: float=0.5, s_count: int=20, gamma: float=2.0, lam: float=0.95):
    s = np.arange(1, s_count + 1, dtype=float)
    y = s + 5.0
    pi = (1.0 - lam) * lam ** (s - 1.0) / (1.0 - lam ** s_count)
    u_y = utility(y, gamma)
    v_aut = float(pi @ u_y / (1.0 - beta))
    ey = float(pi @ y)
    v_pool = float(utility(ey, gamma) / (1.0 - beta))
    c_bar = np.empty(s_count)
    w_bar = np.empty(s_count)
    cum_pi = np.cumsum(pi)
    for j in range(s_count):
        rhs = u_y[j] - beta * np.sum(pi[:j + 1] * (u_y[j] - u_y[:j + 1]))
        c_bar[j] = inverse_utility(rhs, gamma)
        w_bar[j] = (u_y[j] + beta * v_aut - utility(c_bar[j], gamma)) / beta
    profits = np.empty(s_count)
    profits[-1] = (ey - c_bar[-1]) / (1.0 - beta)
    for j in range(s_count - 2, -1, -1):
        low = np.sum(pi[:j + 1] * (y[:j + 1] - c_bar[j]))
        high = np.sum(pi[j + 1:] * (y[j + 1:] - c_bar[j + 1:]))
        continuation = beta * np.sum(pi[j + 1:] * profits[j + 1:])
        profits[j] = (low + high + continuation) / (1.0 - beta * cum_pi[j])
    crossing = int(np.argmax(profits <= 0.0))
    if crossing == 0 and profits[0] <= 0:
        raise RuntimeError('Unexpected crossing at the first grid point.')
    if profits[-1] > 0:
        zero_profit = {'k': None, 'tilde_c': ey, 'v0': v_pool, 'profits': 0.0}
    else:
        k = crossing
        low_prob = float(np.sum(pi[:k]))
        numerator = np.sum(pi[:k] * y[:k]) + np.sum(pi[k:] * (y[k:] - c_bar[k:] + beta * profits[k:]))
        tilde_c = float(numerator / low_prob)
        high_value = np.sum(pi[k:] * (utility(c_bar[k:], gamma) + beta * w_bar[k:]))
        v0 = float((low_prob * utility(tilde_c, gamma) + high_value) / (1.0 - beta * low_prob))
        zero_profit = {'k': k + 1, 'tilde_c': tilde_c, 'v0': v0, 'profits': 0.0}
    return {'beta': beta, 'gamma': gamma, 'lambda': lam, 'S': s_count, 's': s, 'y': y, 'pi': pi, 'cum_pi': cum_pi, 'c_bar': c_bar, 'w_bar': w_bar, 'profits': profits, 'v_aut': v_aut, 'v_pool': v_pool, 'ey': ey, 'zero_profit': zero_profit}

def bank_balances(pi: np.ndarray, y: np.ndarray, consumption: np.ndarray, beta: float, periods: int) -> np.ndarray:
    ey = float(pi @ y)
    balance = np.empty(periods + 1)
    previous = 0.0
    cum_pi = np.cumsum(pi)
    for t in range(periods + 1):
        density = densities(cum_pi, t)
        current = beta ** (-1) * previous + ey - float(density @ consumption)
        balance[t] = current
        previous = current
    return balance

def make_tikz_figures(data: dict[str, object]) -> None:
    c_bar = data['c_bar']
    cum_pi = data['cum_pi']
    beta = data['beta']
    pi = data['pi']
    y = data['y']
    x_min, x_max = (5.5, 16.2)
    body = '\\draw[gray!45] (0.55,2.80) -- (4.75,2.80);\n'
    body += '\\node[left] at (0.55,0.35) {0};\\node[left] at (0.55,2.80) {1};\n'
    body += '\\node[below] at (0.75,0.35) {6};\\node[below] at (4.60,0.35) {16};\n'
    styles = [(0, 'solid'), (5, 'dashed'), (10, 'densely dotted'), (500, 'dash dot')]
    labels = []
    for idx, (t, style) in enumerate(styles):
        x_step = np.repeat(c_bar, 2)[1:]
        y_vals = cum_pi ** (t + 1)
        y_step = np.repeat(y_vals, 2)[:-1]
        x_path = np.r_[c_bar[0], x_step]
        y_path = np.r_[0.0, y_step]
        body += f'\\draw[black,{style},line width=0.55pt] plot coordinates {{{_coord_pairs(x_path, y_path, x_min, x_max, 0.0, 1.0)}}};\n'
        labels.append((t, style, 2.62 - 0.18 * idx))
    for t, style, yloc in labels:
        body += f'\\draw[black,{style},line width=0.55pt] (3.75,{yloc:.2f}) -- (4.05,{yloc:.2f});\\node[right] at (4.07,{yloc:.2f}) {{$t={t}$}};\n'
    out = TIKZ / 'ch21_cdf_vaut.tex'
    _write_axes(out, body, 'consumption', 'CDF')
    print(f'wrote {out}')
    balances = bank_balances(pi, y, c_bar, beta, 100)
    tgrid = np.arange(101)
    log_bal = np.log10(balances)
    body = '\\node[left] at (0.55,0.35) {0};\\node[left] at (0.55,2.80) {30};\n'
    body += '\\node[below] at (0.55,0.35) {0};\\node[below] at (4.75,0.35) {100};\n'
    body += f'\\draw[black,line width=0.6pt] plot coordinates {{{_coord_pairs(tgrid, log_bal, 0.0, 100.0, 0.0, 31.0)}}};\n'
    out = TIKZ / 'ch21_bank_balance_vaut.tex'
    _write_axes(out, body, '$t$', '$\\log_{10} B_t$')
    print(f'wrote {out}')
    expected_wage = np.array([densities(cum_pi, t) @ c_bar for t in range(41)])
    body = '\\node[left] at (0.55,0.35) {11};\\node[left] at (0.55,2.80) {16};\n'
    body += '\\node[below] at (0.55,0.35) {0};\\node[below] at (4.75,0.35) {40};\n'
    body += f'\\draw[black,line width=0.6pt] plot coordinates {{{_coord_pairs(np.arange(41), expected_wage, 0.0, 40.0, 11.0, 16.0)}}};\n'
    out = TIKZ / 'ch21_wage_tenure.tex'
    _write_axes(out, body, 'tenure', 'expected wage')
    print(f'wrote {out}')

def cdf_values(cum_pi: np.ndarray, t: int, indices: list[int]) -> list[float]:
    return [float(cum_pi[i] ** (t + 1)) for i in indices]

def densities(cum_pi: np.ndarray, t: int) -> np.ndarray:
    cdf = cum_pi ** (t + 1)
    return np.diff(np.r_[0.0, cdf])

def utility(c: np.ndarray | float, gamma: float) -> np.ndarray | float:
    if abs(gamma - 1.0) < 1e-14:
        return np.log(c)
    return np.asarray(c) ** (1.0 - gamma) / (1.0 - gamma)

def inverse_utility(u: np.ndarray | float, gamma: float) -> np.ndarray | float:
    if abs(gamma - 1.0) < 1e-14:
        return np.exp(u)
    return ((1.0 - gamma) * np.asarray(u)) ** (1.0 / (1.0 - gamma))


def _write_axes(path: Path, body: str, xlabel: str, ylabel: str) -> None:
    path.write_text(f'\\begin{{tikzpicture}}[x=1in,y=1in]\n\\draw[->] (0.45,0.35) -- (4.95,0.35);\n\\draw[->] (0.55,0.25) -- (0.55,2.95);\n\\node[below] at (2.75,0.08) {{{xlabel}}};\n\\node[rotate=90,above] at (0.04,1.65) {{{ylabel}}};\n{body}\n\\end{{tikzpicture}}\n', encoding='utf-8')

def _coord_pairs(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    coords = []
    for xv, yv in zip(x, y):
        xs = 0.55 + 4.2 * (float(xv) - x_min) / (x_max - x_min)
        ys = 0.35 + 2.45 * (float(yv) - y_min) / (y_max - y_min)
        coords.append(f'({xs:.4f},{ys:.4f})')
    return ' '.join(coords)


if __name__ == '__main__':
    run_contract_calculations()
