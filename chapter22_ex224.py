'Numerical check for Exercise 22.4.'

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TIKZ = ROOT / 'tikz'
TIKZ.mkdir(exist_ok=True)

def exercise_224(beta: float=0.8, b: float=5.0, eps: float=0.5) -> dict[str, float]:
    y_high = 1.0 + eps
    y_low = 1.0 - eps
    denom = 1.0 - beta ** 2
    v_aut_h = (u_224(y_high, b) + beta * u_224(y_low, b)) / denom
    v_aut_l = (u_224(y_low, b) + beta * u_224(y_high, b)) / denom
    v_ce = u_224(1.0, b) / (1.0 - beta)

    def high_pc(c):
        return u_224(c, b) + beta * u_224(2.0 - c, b) - (u_224(y_high, b) + beta * u_224(y_low, b))
    c_check = bisect_root(high_pc, 1.0, y_high)
    v_h = (u_224(c_check, b) + beta * u_224(2.0 - c_check, b)) / denom
    v_l = (u_224(2.0 - c_check, b) + beta * u_224(c_check, b)) / denom
    q_enforcement = beta * max(up_224(2.0 - c_check, b) / up_224(c_check, b), up_224(c_check, b) / up_224(2.0 - c_check, b))
    r_enforcement = 1.0 / q_enforcement
    r_complete = 1.0 / beta
    beta_star = (u_224(y_high, b) - u_224(1.0, b)) / (u_224(1.0, b) - u_224(y_low, b))
    return {'v_aut_h': v_aut_h, 'v_aut_l': v_aut_l, 'v_ce': v_ce, 'c_check': c_check, 'c_other': 2.0 - c_check, 'v_h': v_h, 'v_l': v_l, 'R_complete': r_complete, 'R_enforcement': r_enforcement, 'q_enforcement': q_enforcement, 'beta_star': beta_star}

def make_figures(data: dict[str, float]) -> None:
    t = np.arange(8)
    aut1 = np.where(t % 2 == 0, 1.5, 0.5)
    aut2 = 2.0 - aut1
    cm1 = np.ones_like(t, dtype=float)
    cm2 = np.ones_like(t, dtype=float)
    se1 = np.where(t % 2 == 0, data['c_check'], data['c_other'])
    se2 = 2.0 - se1
    body = '\\node[left] at (0.55,0.35) {0.5};\\node[left] at (0.55,2.80) {1.5};\n'
    body += '\\node[below] at (0.55,0.35) {0};\\node[below] at (4.75,0.35) {7};\n'
    for series, style, label, yloc in [(aut1, 'dotted', 'autarky type 1', 2.65), (cm1, 'dashed', 'complete markets', 2.45), (se1, 'solid', 'enforcement', 2.25)]:
        body += f'\\draw[black,{style},line width=0.6pt] plot coordinates {{{coords(t, series, 0, 7, 0.5, 1.5)}}};\n'
        body += f'\\draw[black,{style},line width=0.6pt] (3.25,{yloc:.2f}) -- (3.55,{yloc:.2f});\\node[right] at (3.57,{yloc:.2f}) {{{label}}};\n'
    write_tikz_axes(TIKZ / 'ch22_ex224_consumption.tex', body, '$t$', 'type 1 consumption')
    aut_v = np.where(t % 2 == 0, data['v_aut_h'], data['v_aut_l'])
    cm_v = np.full_like(t, data['v_ce'], dtype=float)
    se_v = np.where(t % 2 == 0, data['v_h'], data['v_l'])
    body = '\\node[left] at (0.55,0.35) {-0.850};\\node[left] at (0.55,2.80) {-0.830};\n'
    body += '\\node[below] at (0.55,0.35) {0};\\node[below] at (4.75,0.35) {7};\n'
    for series, style, label, yloc in [(aut_v, 'dotted', 'autarky type 1', 2.65), (cm_v, 'dashed', 'complete markets', 2.45), (se_v, 'solid', 'enforcement', 2.25)]:
        body += f'\\draw[black,{style},line width=0.6pt] plot coordinates {{{coords(t, series, 0, 7, -0.85, -0.83)}}};\n'
        body += f'\\draw[black,{style},line width=0.6pt] (3.25,{yloc:.2f}) -- (3.55,{yloc:.2f});\\node[right] at (3.57,{yloc:.2f}) {{{label}}};\n'
    write_tikz_axes(TIKZ / 'ch22_ex224_values.tex', body, '$t$', 'type 1 continuation value')
    body = '\\node[left] at (0.55,0.35) {1.10};\\node[left] at (0.55,2.80) {1.26};\n'
    body += '\\node[below] at (0.55,0.35) {0};\\node[below] at (4.75,0.35) {7};\n'
    body += f'\\draw[black,dashed,line width=0.6pt] plot coordinates {{{coords(t, np.full_like(t, data['R_complete'], dtype=float), 0, 7, 1.1, 1.26)}}};\n'
    body += f'\\draw[black,solid,line width=0.6pt] plot coordinates {{{coords(t, np.full_like(t, data['R_enforcement'], dtype=float), 0, 7, 1.1, 1.26)}}};\n'
    body += '\\draw[black,dashed,line width=0.6pt] (3.25,2.60) -- (3.55,2.60);\\node[right] at (3.57,2.60) {complete markets};\n'
    body += '\\draw[black,solid,line width=0.6pt] (3.25,2.40) -- (3.55,2.40);\\node[right] at (3.57,2.40) {enforcement};\n'
    write_tikz_axes(TIKZ / 'ch22_ex224_interest.tex', body, '$t$', 'gross interest rate')


def bisect_root(fn, lo: float, hi: float, tol: float=1e-14) -> float:
    flo = fn(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = fn(mid)
        if abs(fmid) < tol or hi - lo < tol:
            return mid
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return 0.5 * (lo + hi)

def u_224(c: float, b: float=5.0) -> float:
    return -1.0 / (c + b)

def up_224(c: float, b: float=5.0) -> float:
    return 1.0 / (c + b) ** 2

def write_tikz_axes(path: Path, body: str, xlabel: str, ylabel: str) -> None:
    path.write_text(f'\\begin{{tikzpicture}}[x=1in,y=1in]\n\\draw[->] (0.45,0.35) -- (4.95,0.35);\n\\draw[->] (0.55,0.25) -- (0.55,2.95);\n\\node[below] at (2.75,0.08) {{{xlabel}}};\n\\node[rotate=90,above] at (0.04,1.65) {{{ylabel}}};\n{body}\n\\end{{tikzpicture}}\n', encoding='utf-8')

def coords(x, y, xmin, xmax, ymin, ymax) -> str:
    out = []
    for xv, yv in zip(x, y):
        xs = 0.55 + 4.2 * (float(xv) - xmin) / (xmax - xmin)
        ys = 0.35 + 2.45 * (float(yv) - ymin) / (ymax - ymin)
        out.append(f'({xs:.4f},{ys:.4f})')
    return ' '.join(out)

if __name__ == '__main__':
    ex224 = exercise_224()
    print('Exercise 22.4')
    for key, value in ex224.items():
        print(f'{key} = {value:.12f}')
    make_figures(ex224)
    print(f'wrote {TIKZ / 'ch22_ex224_consumption.tex'}')
    print(f'wrote {TIKZ / 'ch22_ex224_values.tex'}')
    print(f'wrote {TIKZ / 'ch22_ex224_interest.tex'}')
