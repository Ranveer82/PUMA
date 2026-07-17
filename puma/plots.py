"""
plots.py
========

Plotting routines for post-processing a PEST++ IES run.  Every function
takes an :class:`~puma.results.IesResults` instance plus an output
directory and returns the path(s) of the figure(s) written.  All functions
degrade gracefully: if the data they need is not available they log a message
and return ``None`` instead of raising.

Figure catalogue
----------------
Convergence & objective function
    * ``plot_phi_convergence``      -- mean phi vs iteration with ensemble band
    * ``plot_phi_distribution``     -- per-realisation phi, prior vs posterior
    * ``plot_phi_by_group``         -- objective-function decomposition by group

Fit quality / calibration efficacy
    * ``plot_one_to_one``           -- measured vs simulated 1:1 with ensemble
                                       uncertainty range, per observation group
    * ``plot_residual_histograms``  -- residual distributions prior vs posterior
    * ``plot_residual_vs_simulated``-- bias / heteroscedasticity diagnostic

Parameter & predictive uncertainty
    * ``plot_parameter_distributions``       -- prior vs posterior by group
    * ``plot_parameter_uncertainty_reduction`` -- variance-reduction bars
    * ``plot_forecast_uncertainty``          -- prediction PDFs prior vs post
    * ``plot_ensemble_coverage``             -- reliability of the posterior CI

Data screening
    * ``plot_prior_data_conflict``  -- highlight obs in conflict with the prior
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless / batch use
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .results import IesResults
from . import utils
from .utils import (PRIOR_COLOR, POST_COLOR, MEAS_COLOR, ACCENT, WARN_COLOR,
                    gof_metrics, coverage_fraction, fmt_metrics)


# ----------------------------------------------------------------------
def _ensure_dir(output_dir: str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig, path: Path) -> Path:
    fig.savefig(path)
    plt.close(fig)
    print(f"[puma]   saved {path.name}")
    return path


def _obs_meta(res: IesResults) -> pd.DataFrame:
    """Observation-data frame restricted to non-zero-weight targets."""
    obs = res.nonzero_obs()
    # make numeric columns robust
    obs["obsval"] = pd.to_numeric(obs["obsval"], errors="coerce")
    obs["weight"] = pd.to_numeric(obs["weight"], errors="coerce")
    return obs


def _resolve_iters(res: IesResults, iteration, prior_iteration):
    """Resolve (posterior, prior) iteration numbers, defaulting to the case."""
    post_it = res.posterior_iter if iteration is None else int(iteration)
    prior_it = res.prior_iter if prior_iteration is None else int(prior_iteration)
    return post_it, prior_it


# river / flux observation types get volumetric units; everything else metres
_FLOW_HINTS = ("debit", "débit", "flow", "flux", "q_")


def _unit_for_type(otype: str) -> str:
    t = str(otype).lower()
    if any(h in t for h in _FLOW_HINTS):
        return "$m^3/s$"
    if "haut" in t or "stage" in t or "niveau" in t:
        return "m"
    return "m"


def _normalise_type_map(obs_type_map: dict) -> "list[tuple[str, str]]":
    """Return ``[(prefix_lower, type_label), ...]`` sorted longest-prefix first.

    Accepts either ``{type_label: [prefixes]}`` (e.g.
    ``{"Charge": ["c", "h", "w"]}``) or ``{prefix: type_label}`` (e.g.
    ``{"c": "Charge"}``).  Matching is by case-insensitive prefix; the longest
    matching prefix wins so more specific rules take precedence.
    """
    pairs: "list[tuple[str, str]]" = []
    for k, v in obs_type_map.items():
        if isinstance(v, (list, tuple, set)):
            for pref in v:                      # {type: [prefixes]}
                pairs.append((str(pref).lower(), str(k)))
        else:                                   # {prefix: type}
            pairs.append((str(k).lower(), str(v)))
    pairs.sort(key=lambda t: len(t[0]), reverse=True)
    return pairs


def _classify_by_prefix(groups: pd.Series, obs_type_map: dict) -> pd.Series:
    """Classify each group name by the first matching prefix rule."""
    rules = _normalise_type_map(obs_type_map)
    gl = groups.astype(str).str.strip().str.lower()

    def _lookup(g: str) -> Optional[str]:
        for pref, label in rules:
            if g.startswith(pref):
                return label
        return None

    # cache per unique group for speed on large obs sets
    uniq = {g: _lookup(g) for g in gl.unique()}
    return gl.map(uniq)


def _obs_panel_map(res: IesResults, obs: pd.DataFrame,
                   histo_file: Optional[str],
                   obs_type_map: Optional[dict] = None):
    """Map each non-zero observation to a panel key.

    Panel key precedence:

    1. ``obs_type_map`` - classify the observation group (obgnme) by a prefix
       dictionary, e.g. ``{"Charge": ["c", "h", "w"], "Debit": ["d", "q"]}``
       (group starting with c/h/w -> Charge, ...).  Useful when there is no
       ``.histo`` or its names do not match the pst groups.
    2. a Marthe ``.histo`` - panel key is the **observation type**
       (Charge, Debit_Rivi, Hauteu_Rivi, ...) read from the file.
    3. otherwise the panel key is the observation group (obgnme).

    Returns ``(panel_of, units, source)`` where ``panel_of`` is a Series
    indexed like ``obs`` giving the panel key, ``units`` maps panel key -> unit
    label, and ``source`` describes the grouping.
    """
    if obs_type_map:
        mapped = _classify_by_prefix(obs["obgnme"], obs_type_map)
        n_typed = int(mapped.notna().sum())
        if n_typed > 0:
            panel_of = mapped.where(mapped.notna(), "unclassified")
            n_types = int(pd.unique(panel_of).size)
            n_unc = int((mapped.isna()).sum())
            msg = (f"[puma] obs-type-map: classified {n_typed}/{len(obs)} obs "
                   f"into {n_types} type(s)")
            if n_unc:
                msg += f" ({n_unc} unclassified)"
            print(msg)
            units = {t: _unit_for_type(t) for t in pd.unique(panel_of)}
            return panel_of, units, "obs type (prefix map)"
        print("[puma] obs-type-map matched no groups; falling back")

    if histo_file and Path(histo_file).exists():
        try:
            from .spatial import parse_marthe_histo
            hd = parse_marthe_histo(histo_file)
        except Exception as exc:  # noqa: BLE001
            print(f"[puma] could not parse .histo ({exc}); grouping by group")
            hd = None
        if hd is not None and not hd.empty:
            name2type = dict(zip(
                hd["name"].astype(str).str.strip().str.lower(),
                hd["obs_type"].astype(str).str.strip()))
            groups = obs["obgnme"].astype(str)
            gl = groups.str.strip().str.lower()
            mapped = gl.map(name2type)

            # substring fallback for groups that don't match a histo name
            # exactly (e.g. a prefix/suffix differs between pst group and site)
            unmatched = mapped.isna()
            if unmatched.any():
                histo_names = list(name2type)
                fb = {}
                for g in gl[unmatched].unique():
                    hit = next((h for h in histo_names
                                if h and (h in g or g in h)), None)
                    if hit is not None:
                        fb[g] = name2type[hit]
                if fb:
                    mapped = mapped.where(~unmatched, gl.map(fb))

            n_typed = int(mapped.notna().sum())
            n_groups_typed = int(gl[mapped.notna()].nunique())
            n_groups = int(gl.nunique())
            if n_typed > 0:
                panel_of = mapped.where(mapped.notna(), groups)
                n_types = int(mapped.dropna().nunique())
                print(f"[puma] .histo: matched {n_groups_typed}/{n_groups} "
                      f"obs groups to {n_types} observation type(s)")
                units = {t: _unit_for_type(t) for t in pd.unique(panel_of)}
                return panel_of, units, "obs type (.histo)"
            print("[puma] .histo parsed but no group names matched its "
                  "site names; grouping by observation group instead")
    panel_of = obs["obgnme"].astype(str)
    return panel_of, {g: "" for g in pd.unique(panel_of)}, "obs group"


# ======================================================================
# 1. Phi convergence
# ======================================================================
def plot_phi_convergence(res: IesResults, output_dir: str,
                         log_scale: bool = True) -> Optional[Path]:
    """Mean phi vs iteration with the ensemble min/max and inter-quartile band.

    This is the headline "is it converging?" figure.  The measured-phi file
    is preferred (it includes measurement-noise realisations) but the code
    falls back to actual-phi.
    """
    pa = res.phi_actual()
    pm = res.phi_meas()
    if pa is None and pm is None:
        print("[puma] no phi.*.csv found; skipping phi convergence")
        return None

    utils.apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    df = pa if pa is not None else pm

    it = df["iteration"].values
    ax.fill_between(it, df["min"], df["max"], color=PRIOR_COLOR, alpha=0.15,
                    label="ensemble min-max")
    # inter-quartile band from per-realisation values if available
    meta = {"iteration", "total_runs", "mean", "standard_deviation",
            "min", "max"}
    real_cols = [c for c in df.columns if c not in meta]
    if real_cols:
        q25 = df[real_cols].quantile(0.25, axis=1)
        q75 = df[real_cols].quantile(0.75, axis=1)
        ax.fill_between(it, q25, q75, color=POST_COLOR, alpha=0.20,
                        label="inter-quartile range")
    ax.plot(it, df["mean"], color=POST_COLOR, lw=2.2, marker="o",
            ms=4, label="mean $\\Phi$")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("IES iteration")
    ax.set_ylabel("$\\Phi$ (measurement objective function)")
    phi0, phiN = df["mean"].iloc[0], df["mean"].iloc[-1]
    redux = 100.0 * (1 - phiN / phi0) if phi0 else np.nan
    ax.set_title(
        f"Objective-function convergence\n"
        f"$\\Phi$: {phi0:,.0f} $\\to$ {phiN:,.0f}  ({redux:.1f}% reduction)"
    )
    ax.set_xticks(it)
    ax.legend()
    out = _ensure_dir(output_dir) / f"{res.case}_01_phi_convergence.png"
    return _save(fig, out)


# ======================================================================
# 2. Phi distribution (prior vs posterior)
# ======================================================================
def plot_phi_distribution(res: IesResults, output_dir: str) -> Optional[Path]:
    """Violin + box of the per-realisation phi at prior vs posterior.

    Shows not just that the *mean* improved but how the whole ensemble of
    objective-function values collapsed - the hallmark of a good IES run.
    """
    prior = res.realized_phi(res.prior_iter)
    post = res.realized_phi(res.posterior_iter)
    if prior is None and post is None:
        print("[puma] no realised phi available; skipping phi distribution")
        return None

    utils.apply_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    data, labels, colors = [], [], []
    if prior is not None:
        data.append(prior.dropna().values)
        labels.append(f"Prior\n(iter {res.prior_iter})")
        colors.append(PRIOR_COLOR)
    if post is not None:
        data.append(post.dropna().values)
        labels.append(f"Posterior\n(iter {res.posterior_iter})")
        colors.append(POST_COLOR)

    parts = ax.violinplot(data, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.35)
    bp = ax.boxplot(data, widths=0.15, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    for med in bp["medians"]:
        med.set_color("black")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("$\\Phi$ per realisation")
    ax.set_yscale("log")
    ax.set_title("Ensemble objective-function collapse")
    out = _ensure_dir(output_dir) / f"{res.case}_02_phi_distribution.png"
    return _save(fig, out)


# ======================================================================
# 3. Phi by observation group
# ======================================================================
def plot_phi_by_group(res: IesResults, output_dir: str) -> Optional[Path]:
    """Decompose phi into per-group contributions, prior vs posterior.

    Reveals *which* observation groups the calibration was (or was not) able
    to fit, and where residual objective function concentrates after IES.
    """
    pg = res.phi_group()
    if pg is None:
        print("[puma] no phi.group.csv; skipping phi-by-group")
        return None

    # PEST++ writes phi.group.csv with one row per realisation per iteration
    # (columns: iteration,total_runs,obs_realization,par_realization,<groups>);
    # older/summary variants use mean/std/min/max instead - handle both.
    meta = {"iteration", "total_runs", "mean", "standard_deviation",
            "min", "max", "obs_realization", "par_realization", "real_name"}
    grp_cols = [c for c in pg.columns if c not in meta
                and pd.api.types.is_numeric_dtype(pg[c])]
    # drop pure-regularisation / all-zero groups so the plot stays readable
    grp_cols = [c for c in grp_cols if pg[c].abs().sum() > 0]
    if not grp_cols:
        print("[puma] phi.group.csv has no non-zero group columns; skipping")
        return None

    first = pg.loc[pg.iteration == pg.iteration.min(), grp_cols].mean()
    last = pg.loc[pg.iteration == pg.iteration.max(), grp_cols].mean()
    order = last.sort_values(ascending=False).index
    # with many groups show only the largest posterior contributors
    max_bars = 30
    total = len(order)
    truncated = total > max_bars
    if truncated:
        order = order[:max_bars]
    first, last = first[order], last[order]

    utils.apply_style()
    n = len(order)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n + 4), max(6, 0.3 * n)))
    y = np.arange(n)
    ax.barh(y - 0.2, first.values, height=0.4, color=PRIOR_COLOR,
            alpha=0.8, label="prior")
    ax.barh(y + 0.2, last.values, height=0.4, color=POST_COLOR,
            alpha=0.9, label="posterior")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("symlog")
    ax.set_xlabel("mean group $\\Phi$ contribution (symlog)")
    ttl = "Objective-function decomposition by observation group"
    if truncated:
        ttl += f"\n(top {max_bars} of {total} groups by posterior $\\Phi$)"
    ax.set_title(ttl)
    ax.legend()
    out = _ensure_dir(output_dir) / f"{res.case}_03_phi_by_group.png"
    return _save(fig, out)


# ======================================================================
# 4. One-to-one (measured vs simulated) with ensemble uncertainty range
#    *** the group-wise scatter with uncertainty band requested ***
# ======================================================================
def plot_one_to_one(res: IesResults, output_dir: str,
                    iteration: Optional[int] = None,
                    prior_iteration: Optional[int] = None,
                    ci: tuple[float, float] = (0.05, 0.95),
                    max_points: int = 10000,
                    max_group_panels: int = 12,
                    histo_file: Optional[str] = None,
                    obs_type_map: Optional[dict] = None) -> Optional[Path]:
    """Measured vs simulated 1:1 scatter with posterior uncertainty.

    Each marker is the *posterior ensemble mean* simulated value; the error
    bar spans the posterior ensemble credible interval (default 5-95%), so
    the plot communicates both fit and predictive uncertainty at every
    observation.  Prior ensemble means are shown faintly for comparison and
    a metrics box (RMSE / NSE / R2 / bias / PBIAS and CI coverage) quantifies
    the calibration efficacy.

    Panels are one per **observation type** when a Marthe ``histo_file`` is
    given (heads, river flow, river stage, ...), which keeps regional models
    with hundreds of per-site groups to a handful of physically meaningful
    panels with correct units.  Without a histo the panel is the observation
    group; and if that still yields more than ``max_group_panels`` panels the
    figure collapses to a single combined panel over all weighted obs.
    """
    phi_actual = res.phi_actual()
    
    real_mask = [v>0 for v in phi_actual.iloc[res.posterior_iter, 6:-1].values]
    ac_idx = phi_actual.columns[6:-1][real_mask]

    post_it, prior_it = _resolve_iters(res, iteration, prior_iteration)

    post = res.obs_ensemble(post_it)
    post = post.loc[post.index.isin(ac_idx)]
    
    if post is None:
        print("[puma] no posterior obs ensemble; skipping 1:1 plot")
        return None

    prior = res.obs_ensemble(prior_it)
    prior = prior.loc[prior.index.isin(ac_idx)]
    obs = _obs_meta(res)

    # Remove invalid measured
    obsval = pd.to_numeric(obs["obsval"], errors="coerce")
    valid_obs = obsval.notna() & obsval.ne(-9999.0)

    obs = obs.loc[valid_obs].copy()
    obs["obsval"] = obsval.loc[valid_obs]

    
    post_cols = set(post.columns)
    obs = obs.loc[obs.index.isin(post_cols)]

    if obs.empty:
        print("[puma] no valid observations found; skipping 1:1 plot")
        return None

    panel_of, units, source = _obs_panel_map(
        res, obs, histo_file, obs_type_map
    )
    panels = sorted(pd.unique(panel_of), key=str)

    # still too many panels and no type grouping -> single combined panel
    if len(panels) > max_group_panels:
        return _one_to_one_combined(res, output_dir, obs, post, prior,
                                    ci, max_points, len(panels), post_it)

    utils.apply_style()
    nrows, ncols = utils.grid_shape(len(panels), ncols=min(3, len(panels)))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.2 * ncols, 4.8 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    lo_q, hi_q = ci
    for i, g in enumerate(panels):
        ax = axes[i]
        names = panel_of.index[panel_of == g].tolist()
        meas = obs.loc[names, "obsval"].astype(float).values
        sub = post[names]

        sim_mean = sub.mean(axis=0).values
        sim_lo = sub.quantile(lo_q, axis=0).values
        sim_hi = sub.quantile(hi_q, axis=0).values

        # thin very large panels so the figure stays readable / cheap
        if len(names) > max_points:
            idx = np.random.default_rng(0).choice(
                len(names), max_points, replace=False)
        else:
            idx = np.arange(len(names))

        # a skewed ensemble mean can fall just outside [q_lo, q_hi]; clip the
        # error-bar half-widths at 0 so matplotlib does not reject them
        yerr = np.vstack([sim_mean[idx] - sim_lo[idx],
                          sim_hi[idx] - sim_mean[idx]])
        yerr = np.clip(yerr, 0, None)
        ax.errorbar(meas[idx], sim_mean[idx], yerr=yerr, fmt="none",
                    ecolor=POST_COLOR, elinewidth=0.6, alpha=0.3, zorder=1)
        ax.scatter(meas[idx], sim_mean[idx], s=12, color=POST_COLOR,
                   edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3,
                   label="posterior mean")

        if prior is not None:
            pnames = [n for n in names if n in prior.columns]
            if pnames:
                pmean = prior[pnames].mean(axis=0).values
                pmeas = obs.loc[pnames, "obsval"].astype(float).values
                ax.scatter(pmeas, pmean, s=8, color=PRIOR_COLOR,
                           alpha=0.3, zorder=2, label="prior mean")

        lims = _shared_lims(meas, sim_mean)
        ax.plot(lims, lims, "k--", lw=1, zorder=4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        m = gof_metrics(meas, sim_mean)
        cov = coverage_fraction(meas, sim_lo, sim_hi)
        txt = fmt_metrics(m) + f"\nCI cov={cov*100:.0f}%"
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=8, bbox=dict(boxstyle="round", fc="white",
                                      ec="0.8", alpha=0.85))
        unit = units.get(g, "")
        u = f"  ({unit})" if unit else ""
        ax.set_title(f"{g}  [n={len(names)}]", fontsize=11)
        ax.set_xlabel(f"measured{u}")
        ax.set_ylabel(f"simulated{u}")

    handles = [
        Line2D([0], [0], marker="o", ls="none", color=POST_COLOR,
               label="posterior mean"),
        Line2D([0], [0], marker="o", ls="none", color=PRIOR_COLOR,
               label="prior mean"),
        Line2D([0], [0], color=POST_COLOR, lw=4, alpha=0.4,
               label=f"posterior {round((hi_q-lo_q)*100)}% range"),
        Line2D([0], [0], color="k", ls="--", label="1:1 line"),
    ]
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Measured vs simulated with posterior uncertainty  "
        f"[{res.case}, iter {post_it}] - by {source}",
        fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    out = _ensure_dir(output_dir) / f"{res.case}_04_one_to_one_uncertainty.png"
    return _save(fig, out)


def _one_to_one_combined(res, output_dir, obs, post, prior, ci, max_points,
                         n_groups, post_it=None):
    """Single combined 1:1 panel over all weighted obs (many-group models)."""
    if post_it is None:
        post_it = res.posterior_iter
    utils.apply_style()
    lo_q, hi_q = ci
    names = [n for n in obs.index if n in post.columns]
    meas = obs.loc[names, "obsval"].astype(float).values
    sub = post[names]
    sim_mean = sub.mean(axis=0).values
    sim_lo = sub.quantile(lo_q, axis=0).values
    sim_hi = sub.quantile(hi_q, axis=0).values

    n = len(names)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
    else:
        idx = np.arange(n)

    fig, ax = plt.subplots(figsize=(8.5, 8))
    yerr = np.clip(np.vstack([sim_mean[idx] - sim_lo[idx],
                              sim_hi[idx] - sim_mean[idx]]), 0, None)
    ax.errorbar(meas[idx], sim_mean[idx], yerr=yerr, fmt="none",
                ecolor=POST_COLOR, elinewidth=0.5, alpha=0.25, zorder=1)
    ax.scatter(meas[idx], sim_mean[idx], s=8, color=POST_COLOR,
               edgecolor="none", alpha=0.6, zorder=3, label="posterior mean")
    if prior is not None:
        pnames = [n for n in names if n in prior.columns]
        if pnames:
            pmeas = obs.loc[pnames, "obsval"].astype(float).values
            pmean = prior[pnames].mean(axis=0).values
            pi = idx if len(pnames) == n else np.arange(len(pnames))
            if len(pnames) > max_points:
                pi = np.random.default_rng(1).choice(len(pnames), max_points,
                                                      replace=False)
            ax.scatter(pmeas[pi], pmean[pi], s=6, color=PRIOR_COLOR,
                       alpha=0.25, zorder=2, label="prior mean")
    lims = _shared_lims(meas, sim_mean)
    ax.plot(lims, lims, "k--", lw=1, zorder=4, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    m = gof_metrics(meas, sim_mean)
    cov = coverage_fraction(meas, sim_lo, sim_hi)
    txt = fmt_metrics(m) + f"\nCI cov={cov*100:.0f}%"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
    ax.set_xlabel("measured")
    ax.set_ylabel("simulated")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Measured vs simulated with posterior uncertainty\n"
                 f"[{res.case}, iter {post_it}] - all "
                 f"{n:,} weighted obs across {n_groups} groups")
    out = _ensure_dir(output_dir) / f"{res.case}_04_one_to_one_uncertainty.png"
    return _save(fig, out)


def _shared_lims(a, b, pad=0.05):
    vals = np.concatenate([np.asarray(a, float), np.asarray(b, float)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0, 1)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    if lo == hi:
        lo -= 1
        hi += 1
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


# ======================================================================
# 5. Residual histograms prior vs posterior
# ======================================================================
def plot_residual_histograms(res: IesResults, output_dir: str,
                             iteration: Optional[int] = None,
                             prior_iteration: Optional[int] = None,
                             clip_pct: tuple[float, float] = (2, 98),
                             max_group_panels: int = 12,
                             histo_file: Optional[str] = None,
                             obs_type_map: Optional[dict] = None) -> Optional[Path]:
    """Residual (simulated - measured) distributions, prior vs posterior.

    One panel per **observation type** when a Marthe ``histo_file`` is given,
    otherwise per observation group - falling back to a single combined
    histogram over all weighted observations above ``max_group_panels`` panels.
    """
    post_it, prior_it = _resolve_iters(res, iteration, prior_iteration)
    post = res.obs_ensemble(post_it)
    if post is None:
        print("[puma] no posterior obs ensemble; skipping residual hist")
        return None
    prior = res.obs_ensemble(prior_it)
    obs = _obs_meta(res)
    obs = obs.loc[obs.index.isin(set(post.columns))]
    panel_of, units, source = _obs_panel_map(res, obs, histo_file, obs_type_map)
    panels = sorted(pd.unique(panel_of), key=str)
    combined = len(panels) > max_group_panels
    if combined:
        panels = ["__all__"]

    utils.apply_style()
    nrows, ncols = utils.grid_shape(len(panels), ncols=min(3, len(panels)))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.2 * ncols, 4.0 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for i, g in enumerate(panels):
        ax = axes[i]
        if g == "__all__":
            names = list(obs.index)
        else:
            names = panel_of.index[panel_of == g].tolist()
        if not names:
            ax.axis("off")
            continue
        meas = obs.loc[names, "obsval"].astype(float)
        res_post = post[names].subtract(meas, axis=1).values.flatten()
        res_post = res_post[np.isfinite(res_post)]
        allvals = [res_post]
        res_prior = None
        if prior is not None:
            pn = [n for n in names if n in prior.columns]
            if pn:
                pmeas = obs.loc[pn, "obsval"].astype(float)
                res_prior = prior[pn].subtract(pmeas, axis=1).values.flatten()
                res_prior = res_prior[np.isfinite(res_prior)]
                allvals.append(res_prior)

        stacked = np.concatenate(allvals)
        rmin, rmax = np.percentile(stacked, clip_pct)
        if rmin == rmax:
            rmin, rmax = rmin - 1, rmax + 1
        if res_prior is not None:
            ax.hist(res_prior, bins=50, range=(rmin, rmax), density=True,
                    color=PRIOR_COLOR, alpha=0.45, label="prior")
        ax.hist(res_post, bins=50, range=(rmin, rmax), density=True,
                color=POST_COLOR, alpha=0.6, label="posterior")
        ax.axvline(0, color="k", lw=1)
        ax.axvline(np.mean(res_post), color=POST_COLOR, ls="--", lw=1.4)
        ax.text(0.03, 0.97,
                f"n={len(names)}\n$\\mu$={np.mean(res_post):.3g}\n"
                f"$\\sigma$={np.std(res_post):.3g}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.8))
        title = (f"all weighted obs ({len(pd.unique(panel_of))} {source}s)"
                 if g == "__all__" else f"{g}  [n={len(names)}]")
        ax.set_title(title, fontsize=11)
        unit = units.get(g, "")
        u = f" ({unit})" if unit else ""
        ax.set_xlabel(f"residual (sim - meas){u}")
        ax.set_ylabel("density")
        if i == 0:
            ax.legend()

    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Residual distributions  [{res.case}] - by {source}",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = _ensure_dir(output_dir) / f"{res.case}_05_residual_histograms.png"
    return _save(fig, out)


# ======================================================================
# 6. Residual vs simulated (bias / heteroscedasticity)
# ======================================================================
def plot_residual_vs_simulated(res: IesResults, output_dir: str,
                               iteration: Optional[int] = None,
                               histo_file: Optional[str] = None,
                               obs_type_map: Optional[dict] = None
                               ) -> Optional[Path]:
    """Posterior mean residual against simulated value, coloured by obs type.

    A structureless cloud centred on zero indicates an unbiased fit; trends
    or fanning reveal conditional bias or heteroscedastic error.  Points are
    coloured by **observation type** when a Marthe ``histo_file`` is given
    (else by group), collapsing to one colour above 12 panels.
    """
    post_it, _ = _resolve_iters(res, iteration, None)
    post = res.obs_ensemble(post_it)
    if post is None:
        print("[puma] no posterior obs ensemble; skipping residual-vs-sim")
        return None
    obs = _obs_meta(res)
    obs = obs.loc[obs.index.isin(set(post.columns))]
    panel_of, _units, source = _obs_panel_map(res, obs, histo_file, obs_type_map)
    panels = sorted(pd.unique(panel_of), key=str)

    utils.apply_style()
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(panels) <= 12:
        for i, g in enumerate(panels):
            names = panel_of.index[panel_of == g].tolist()
            if not names:
                continue
            meas = obs.loc[names, "obsval"].astype(float).values
            sim = post[names].mean(axis=0).values
            ax.scatter(sim, sim - meas, s=12, alpha=0.5,
                       color=utils.group_color(i), label=str(g))
        ax.legend(fontsize=8, ncol=2, title=source)
        ax.set_title("Residual vs simulated - bias / heteroscedasticity check")
    else:
        names = list(obs.index)
        meas = obs.loc[names, "obsval"].astype(float).values
        sim = post[names].mean(axis=0).values
        ax.scatter(sim, sim - meas, s=6, alpha=0.3, color=POST_COLOR,
                   edgecolor="none")
        ax.set_title(f"Residual vs simulated ({len(names):,} weighted obs, "
                     f"{len(panels)} {source}s)")
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xlabel("simulated (posterior mean)")
    ax.set_ylabel("residual (sim - meas)")
    out = _ensure_dir(output_dir) / f"{res.case}_06_residual_vs_simulated.png"
    return _save(fig, out)


# ======================================================================
# 7. Parameter distributions prior vs posterior
# ======================================================================
def plot_parameter_distributions(res: IesResults, output_dir: str,
                                 iteration: Optional[int] = None,
                                 prior_iteration: Optional[int] = None
                                 ) -> Optional[Path]:
    """Prior vs posterior parameter distributions, one panel per group.

    Whether a panel is drawn on a log10 axis is decided from the control
    file's ``partrans`` (a group whose parameters are log-transformed is
    shown in log space) - never guessed from the group name.
    """
    post_it, prior_it = _resolve_iters(res, iteration, prior_iteration)
    prior = res.par_ensemble(prior_it)
    post = res.par_ensemble(post_it)
    if prior is None or post is None:
        print("[puma] par ensembles unavailable; skipping par distributions")
        return None

    pdata = res.pst.parameter_data
    adj = pdata.loc[pdata.partrans != "fixed"]
    par_by_group = adj.groupby("pargp")["parnme"].apply(list).to_dict()
    # fraction of each group that is log-transformed, from the control file
    log_frac = (adj.assign(is_log=(adj.partrans == "log").astype(float))
                .groupby("pargp")["is_log"].mean().to_dict())
    groups = sorted(par_by_group.keys())

    utils.apply_style()
    nrows, ncols = utils.grid_shape(len(groups), ncols=4)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.3 * ncols, 3.8 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for i, g in enumerate(groups):
        ax = axes[i]
        names = [n for n in par_by_group[g]
                 if n in prior.columns and n in post.columns]
        if not names:
            ax.axis("off")
            continue
        pri = prior[names].values.flatten()
        pos = post[names].values.flatten()
        pri = pri[np.isfinite(pri)]
        pos = pos[np.isfinite(pos)]
        # log axis when the control file says this group is log-transformed
        use_log = (log_frac.get(g, 0.0) >= 0.5
                   and pri.size and pos.size
                   and np.all(pri > 0) and np.all(pos > 0))
        if use_log:
            pri, pos = np.log10(pri), np.log10(pos)
            xlab = f"log10({g})"
        else:
            xlab = g
        stacked = np.concatenate([pri, pos])
        rmin, rmax = np.percentile(stacked, [1, 99])
        if rmin == rmax:
            rmin, rmax = rmin - 1, rmax + 1
        ax.hist(pri, bins=40, range=(rmin, rmax), density=True,
                color=PRIOR_COLOR, alpha=0.45, label="prior")
        ax.hist(pos, bins=40, range=(rmin, rmax), density=True,
                color=POST_COLOR, alpha=0.6, label="posterior")
        # uncertainty reduction annotation
        red = 100.0 * (1 - np.std(pos) / np.std(pri)) if np.std(pri) > 0 else np.nan
        ax.text(0.03, 0.97, f"npar={len(names)}\nunc.$\\downarrow${red:.0f}%",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.8))
        ax.set_title(g, fontsize=10)
        ax.set_xlabel(xlab)
        ax.set_ylabel("density")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(groups), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Parameter prior vs posterior  [{res.case}]",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = _ensure_dir(output_dir) / f"{res.case}_07_parameter_distributions.png"
    return _save(fig, out)


# ======================================================================
# 8. Parameter uncertainty reduction
# ======================================================================
def plot_parameter_uncertainty_reduction(res: IesResults, output_dir: str,
                                         iteration: Optional[int] = None,
                                         prior_iteration: Optional[int] = None
                                         ) -> Optional[Path]:
    """Percent reduction in parameter standard deviation, prior -> posterior.

    Reduction = 1 - sigma_post / sigma_prior.  Large values mark parameters
    strongly informed by the data; near-zero values mark parameters the data
    could not constrain (candidate for prior sensitivity).  Aggregated by
    parameter group with whiskers spanning the per-parameter spread.
    """
    post_it, prior_it = _resolve_iters(res, iteration, prior_iteration)
    prior = res.par_ensemble(prior_it)
    post = res.par_ensemble(post_it)
    if prior is None or post is None:
        print("[puma] par ensembles unavailable; skipping unc-reduction")
        return None

    pdata = res.pst.parameter_data
    common = [p for p in pdata.loc[pdata.partrans != "fixed", "parnme"]
              if p in prior.columns and p in post.columns]
    if not common:
        return None
    sd_pri = prior[common].std(axis=0)
    sd_pos = post[common].std(axis=0)
    reduction = (1 - sd_pos / sd_pri.replace(0, np.nan)) * 100.0
    df = pd.DataFrame({"pargp": pdata.set_index("parnme").loc[common, "pargp"].values,
                       "reduction": reduction.values})
    grp = df.groupby("pargp")["reduction"]
    order = grp.median().sort_values().index

    utils.apply_style()
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(order) + 4), 6))
    y = np.arange(len(order))
    med = grp.median().loc[order]
    q1 = grp.quantile(0.25).loc[order]
    q3 = grp.quantile(0.75).loc[order]
    ax.barh(y, med.values, color=POST_COLOR, alpha=0.85)
    ax.errorbar(med.values, y,
                xerr=[med.values - q1.values, q3.values - med.values],
                fmt="none", ecolor="0.3", capsize=3, lw=1)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("uncertainty reduction  $1-\\sigma_{post}/\\sigma_{prior}$  (%)")
    ax.set_title("How much did the data inform each parameter group?")
    out = _ensure_dir(output_dir) / f"{res.case}_08_parameter_uncertainty_reduction.png"
    return _save(fig, out)


# ======================================================================
# 9. Forecast / prediction uncertainty
# ======================================================================
def plot_forecast_uncertainty(res: IesResults, output_dir: str,
                              iteration: Optional[int] = None,
                              prior_iteration: Optional[int] = None
                              ) -> Optional[Path]:
    """Prior vs posterior predictive distributions for declared forecasts.

    Uses the ``++forecasts`` / prediction observations recorded in the
    control file.  Shows the collapse (or not) of predictive uncertainty and
    where the posterior prediction sits relative to the prior.
    """
    forecasts = res.pst.forecast_names
    if forecasts is None or len(forecasts) == 0:
        print("[puma] no forecasts declared; skipping forecast plot")
        return None
    post_it, prior_it = _resolve_iters(res, iteration, prior_iteration)
    prior = res.obs_ensemble(prior_it)
    post = res.obs_ensemble(post_it)
    if prior is None or post is None:
        return None
    forecasts = [f for f in forecasts
                 if f in prior.columns and f in post.columns]
    if not forecasts:
        return None

    utils.apply_style()
    nrows, ncols = utils.grid_shape(len(forecasts), ncols=3)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 3.8 * nrows),
                             squeeze=False)
    axes = axes.flatten()
    obs = res.pst.observation_data

    for i, f in enumerate(forecasts):
        ax = axes[i]
        pri = prior[f].values
        pos = post[f].values
        stacked = np.concatenate([pri, pos])
        rmin, rmax = np.percentile(stacked, [1, 99])
        bins = np.linspace(rmin, rmax, 40)
        ax.hist(pri, bins=bins, density=True, color=PRIOR_COLOR,
                alpha=0.45, label="prior")
        ax.hist(pos, bins=bins, density=True, color=POST_COLOR,
                alpha=0.6, label="posterior")
        red = 100.0 * (1 - np.std(pos) / np.std(pri)) if np.std(pri) > 0 else np.nan
        ax.text(0.03, 0.97, f"unc.$\\downarrow${red:.0f}%",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.8))
        # overlay the measured value if this forecast also has a weight/obsval
        try:
            oval = float(obs.loc[f, "obsval"])
            if np.isfinite(oval) and rmin <= oval <= rmax:
                ax.axvline(oval, color=MEAS_COLOR, lw=1.6, label="measured")
        except (KeyError, ValueError, TypeError):
            pass
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("forecast value")
        ax.set_ylabel("density")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(forecasts), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Predictive uncertainty (prior vs posterior)  [{res.case}]",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = _ensure_dir(output_dir) / f"{res.case}_09_forecast_uncertainty.png"
    return _save(fig, out)


# ======================================================================
# 10. Ensemble coverage / reliability
# ======================================================================
def plot_ensemble_coverage(res: IesResults, output_dir: str,
                           iteration: Optional[int] = None
                           ) -> Optional[Path]:
    """Reliability diagram: nominal vs empirical coverage of the posterior.

    For a range of nominal credible intervals (e.g. 10-90%) it computes the
    fraction of measurements that actually fall inside the posterior ensemble
    interval.  A perfectly reliable ensemble sits on the 1:1 line; below the
    line means the posterior is over-confident (intervals too narrow), above
    means under-confident.
    """
    post_it, _ = _resolve_iters(res, iteration, None)
    post = res.obs_ensemble(post_it)
    if post is None:
        print("[puma] no posterior obs ensemble; skipping coverage plot")
        return None
    obs = _obs_meta(res)
    names = [n for n in obs.index if n in post.columns]
    if not names:
        return None
    meas = obs.loc[names, "obsval"].astype(float).values
    valid = (
            np.isfinite(meas)
            & (meas != -9999)
            )
    meas = meas[valid]
    
    nominal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    empirical = []
    for p in nominal:
        lo = post[names].quantile((1 - p) / 2, axis=0).values
        lo = lo[valid]
        hi = post[names].quantile(1 - (1 - p) / 2, axis=0).values
        hi = hi[valid]
        empirical.append(coverage_fraction(meas, lo, hi))
    empirical = np.array(empirical)

    utils.apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfect reliability")
    ax.plot(nominal, empirical, "o-", color=POST_COLOR, lw=2,
            label="posterior ensemble")
    ax.fill_between(nominal, nominal, empirical,
                    where=empirical >= nominal, color=ACCENT, alpha=0.15,
                    label="under-confident")
    ax.fill_between(nominal, nominal, empirical,
                    where=empirical < nominal, color=WARN_COLOR, alpha=0.15,
                    label="over-confident")
    ax.set_xlabel("nominal credible interval")
    ax.set_ylabel("empirical coverage of measurements")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Posterior reliability / coverage diagram")
    ax.legend(loc="upper left", fontsize=8)
    out = _ensure_dir(output_dir) / f"{res.case}_10_ensemble_coverage.png"
    return _save(fig, out)


# ======================================================================
# 11. Prior data conflict
# ======================================================================
def plot_prior_data_conflict(res: IesResults, output_dir: str,
                             prior_iteration: Optional[int] = None
                             ) -> Optional[Path]:
    """Screen observations for prior-data conflict.

    If PEST++ wrote a ``*.pdc.csv`` it is used directly.  Otherwise conflict
    is derived from the prior observation ensemble: an observation is *in
    conflict* when the measured value lies outside the prior ensemble range,
    meaning the prior cannot reproduce it and it may bias the calibration.
    """
    obs = _obs_meta(res)
    pdc = res.pdc()

    # Build a per-group summary (sum = #conflicted, count = #obs) - vectorised,
    # so it stays fast even for millions of observations.  The prior ensemble
    # is only loaded for the fallback path (no pdc.csv), never when PEST++'s
    # own *.pdc.csv is available.
    summ = None
    source = None
    if pdc is not None and "name" in {c.lower() for c in pdc.columns}:
        cols = {c.lower(): c for c in pdc.columns}
        name_col = cols["name"]
        flagged = pdc[name_col].astype(str).str.lower()
        if "distance" in cols:
            keep = pd.to_numeric(pdc[cols["distance"]], errors="coerce").notna()
            flagged = flagged[keep.values]
        flagged_set = set(flagged)
        obs_all = res.pst.observation_data
        grp = obs_all["obgnme"].astype(str)
        # index names lower-cased to match the pdc names
        is_flagged = pd.Series(obs_all.index.str.lower().isin(flagged_set),
                               index=obs_all.index)
        total = grp.value_counts()
        conf = grp[is_flagged.values].value_counts()
        summ = pd.DataFrame({"count": total})
        summ["sum"] = conf.reindex(summ.index).fillna(0).astype(int)
        summ["pct"] = 100.0 * summ["sum"] / summ["count"]
        source = "pdc.csv"
    else:
        _, prior_it = _resolve_iters(res, None, prior_iteration)
        prior = res.obs_ensemble(prior_it)
        if prior is None:
            print("[puma] no data for prior-data-conflict; skipping")
            return None
        names = [n for n in obs.index if n in prior.columns]
        if not names:
            print("[puma] cannot assess prior-data conflict; skipping")
            return None
        meas = obs.loc[names, "obsval"].astype(float)
        lo = prior[names].min(axis=0)
        hi = prior[names].max(axis=0)
        in_conflict = (meas < lo) | (meas > hi)
        summ = (pd.DataFrame({"obgnme": obs.loc[names, "obgnme"].values,
                              "conflict": in_conflict.values})
                .groupby("obgnme")["conflict"].agg(["sum", "count"]))
        summ["pct"] = 100.0 * summ["sum"] / summ["count"]
        source = "prior ensemble range"

    utils.apply_style()
    if summ is not None and not summ.empty:
        n_conf_groups = int((summ["sum"] > 0).sum())
        total_groups = len(summ)
        # with many groups keep the worst offenders so the chart stays readable
        max_bars = 30
        ranked = summ.sort_values("pct", ascending=False)
        if total_groups > max_bars:
            conflicted = ranked[ranked["sum"] > 0].head(max_bars)
            # if nothing is in conflict, still show the top groups (all 0%)
            ranked = conflicted if not conflicted.empty else ranked.head(max_bars)
        summ = ranked.sort_values("pct", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(summ) + 2)))
        y = np.arange(len(summ))
        colors = [WARN_COLOR if p > 0 else ACCENT for p in summ["pct"]]
        ax.barh(y, summ["pct"].values, color=colors, alpha=0.85)
        # fix the x-axis before annotating so labels never fall off-canvas
        pmax = float(summ["pct"].max()) if len(summ) else 0.0
        if not np.isfinite(pmax):
            pmax = 0.0
        xmax = max(pmax * 1.15, 1.0)
        ax.set_xlim(0, xmax)
        for yi, (s, c) in enumerate(zip(summ["sum"], summ["count"])):
            ax.text(summ["pct"].iloc[yi] + xmax * 0.01, yi,
                    f"{int(s)}/{int(c)}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(summ.index, fontsize=8)
        ax.set_xlabel("% of observations in prior-data conflict")
        ttl = f"Prior-data conflict by group  (source: {source})"
        if total_groups > max_bars:
            ttl += (f"\n{n_conf_groups} of {total_groups} groups in conflict; "
                    f"showing worst {len(summ)}")
        ax.set_title(ttl)
        out = _ensure_dir(output_dir) / f"{res.case}_11_prior_data_conflict.png"
        return _save(fig, out)
    print("[puma] pdc.csv present but unexpected format; skipping plot")
    return None
