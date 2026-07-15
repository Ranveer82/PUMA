"""
utils.py
========

Shared helpers: consistent, colour-blind-safe styling and goodness-of-fit
metrics used across the plotting routines.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Colour-blind-safe palette (Wong, 2011, "Points of view: Color blindness")
# Used consistently so prior/posterior read the same across every figure.
PRIOR_COLOR = "#7f7f7f"      # neutral grey
POST_COLOR = "#0072B2"       # blue
MEAS_COLOR = "#D55E00"       # vermillion
ACCENT = "#009E73"           # bluish green
WARN_COLOR = "#CC3311"       # red

# A qualitative cycle for per-group colouring.
GROUP_CYCLE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#999999",
    "#117733", "#882255", "#44AA99", "#332288",
]


def apply_style() -> None:
    """Apply a clean, publication-oriented matplotlib style."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def group_color(index: int) -> str:
    """Return a stable colour for the ``index``-th group."""
    return GROUP_CYCLE[index % len(GROUP_CYCLE)]


def grid_shape(n: int, ncols: int = 3) -> tuple[int, int]:
    """Rows/cols for ``n`` sub-plots given a preferred column count."""
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


# ----------------------------------------------------------------------
# Goodness-of-fit metrics
def gof_metrics(measured: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
    """Compute standard calibration metrics.

    Returns a dict with n, RMSE, MAE, ME (mean error / bias), NSE
    (Nash-Sutcliffe), R2 (Pearson r squared), PBIAS (%) and the
    RMSE-observations-standard-deviation ratio (RSR).
    """
    measured = np.asarray(measured, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    mask = np.isfinite(measured) & np.isfinite(simulated)
    measured, simulated = measured[mask], simulated[mask]
    n = measured.size
    if n == 0:
        return {k: np.nan for k in
                ["n", "rmse", "mae", "me", "nse", "r2", "pbias", "rsr"]}

    resid = simulated - measured
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    me = float(np.mean(resid))
    denom = np.sum((measured - measured.mean()) ** 2)
    nse = float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else np.nan
    if measured.std() > 0 and simulated.std() > 0:
        r = np.corrcoef(measured, simulated)[0, 1]
        r2 = float(r ** 2)
    else:
        r2 = np.nan
    sum_meas = np.sum(measured)
    pbias = float(100.0 * np.sum(resid) / sum_meas) if sum_meas != 0 else np.nan
    std_obs = np.std(measured)
    rsr = float(rmse / std_obs) if std_obs > 0 else np.nan
    return {
        "n": int(n), "rmse": rmse, "mae": mae, "me": me,
        "nse": nse, "r2": r2, "pbias": pbias, "rsr": rsr,
    }


def coverage_fraction(measured: np.ndarray,
                      lower: np.ndarray,
                      upper: np.ndarray) -> float:
    """Fraction of measured values contained within [lower, upper].

    A well-calibrated ensemble whose bounds are a p% credible interval
    should contain roughly p% of the measurements.
    """
    measured = np.asarray(measured, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(measured) & np.isfinite(lower) & np.isfinite(upper)
    if mask.sum() == 0:
        return np.nan
    inside = (measured[mask] >= lower[mask]) & (measured[mask] <= upper[mask])
    return float(inside.mean())


def fmt_metrics(m: Dict[str, float]) -> str:
    """One-line label string for annotating plots."""
    return (
        f"n={m['n']}  RMSE={m['rmse']:.3g}\n"
        f"NSE={m['nse']:.3f}  R$^2$={m['r2']:.3f}\n"
        f"bias={m['me']:.3g}  PBIAS={m['pbias']:.1f}%"
    )
