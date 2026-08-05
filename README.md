# Numerical verification scripts

This directory contains numerical checks and figure-generation support for the
solutions manual. The scripts retain the notation and calculations used by the
manual.

## Requirements

- Python 3.12 or newer
- NumPy
- Matplotlib for the optional PDF figure-generation routines

## Project layout

The scripts are intended to remain in the `python/` directory of the project.
They resolve paths from their own file locations and may use the sibling
`figures/` and `tikz/` directories for generated assets. Matplotlib's local
cache is stored in `.mplcache/` at the project root; that directory is ignored
by Git.

Some scripts regenerate PDF figures only when `RMT_WRITE_FIGURES=1` is set. For
example, from the project root:

```sh
RMT_WRITE_FIGURES=1 python3.12 python/chapter2_ex219.py
```

Without that setting, the applicable scripts perform their numerical checks
without replacing the existing figures.
