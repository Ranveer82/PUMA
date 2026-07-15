"""
report.py
=========

High-level orchestration: run the full post-processing suite on a PEST++ IES
run with a single call.  Each plot is wrapped so that a failure in one figure
never aborts the others - the goal is a hands-off, fully autonomous report.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional

from .results import IesResults
from . import plots
from . import timeseries
from . import marthe
from . import spatial


# (function, kwargs, human label) - order controls figure numbering / flow.
_PIPELINE = [
    (plots.plot_phi_convergence, {}, "phi convergence"),
    (plots.plot_phi_distribution, {}, "phi distribution"),
    (plots.plot_phi_by_group, {}, "phi by group"),
    (plots.plot_one_to_one, {}, "1:1 with uncertainty"),
    (plots.plot_residual_histograms, {}, "residual histograms"),
    (plots.plot_residual_vs_simulated, {}, "residual vs simulated"),
    (plots.plot_parameter_distributions, {}, "parameter distributions"),
    (plots.plot_parameter_uncertainty_reduction, {}, "parameter unc. reduction"),
    (plots.plot_forecast_uncertainty, {}, "forecast uncertainty"),
    (plots.plot_ensemble_coverage, {}, "ensemble coverage"),
    (plots.plot_prior_data_conflict, {}, "prior-data conflict"),
]


def run_full_report(pst_file: str,
                    output_dir: str = "PLOTS_PESTPP_IES",
                    iteration: Optional[int] = None,
                    prior_iteration: Optional[int] = None,
                    obs_meta_csv: Optional[str] = None,
                    marthe_pastp: Optional[str] = None,
                    marthe_mart: Optional[str] = None,
                    histo_file: Optional[str] = None,
                    pp_file: Optional[str] = None,
                    shapefiles: Optional[List[str]] = None,
                    marthe_rma: Optional[str] = None,
                    marthe_config: Optional[str] = None,
                    field_prop: Optional[str] = None,
                    field_layer: int = 0,
                    field_max_reals: Optional[int] = None,
                    make_timeseries: bool = True,
                    verbose: bool = True) -> List[Path]:
    """Run every applicable plot for a PEST++ IES case.

    Parameters
    ----------
    pst_file:
        Path to the ``.pst`` control file (or its directory).
    output_dir:
        Where PNGs are written (created if needed).
    iteration:
        Which IES iteration to treat as the *posterior* (default: the last
        iteration found).  Every posterior/uncertainty figure uses it, so you
        can inspect the fit and spread at any iteration.
    prior_iteration:
        Which iteration to treat as the *prior* baseline (default: 0).
    obs_meta_csv:
        Optional CSV with ``obsnme,site,datetime`` for time-series plots.
    marthe_pastp:
        Optional path to the Marthe ``.pastp`` file.  When given, the transient
        observation datetimes are reconstructed from the model definition (see
        :mod:`pestpp_ies_post.marthe`) and used for the time-series figures -
        no external date lookup required.  Takes precedence over
        ``obs_meta_csv``.
    marthe_mart:
        Optional path to the Marthe ``.mart`` file (used with ``marthe_pastp``
        to honour the time-step budget / warm-up count).
    histo_file:
        Optional Marthe ``.histo`` file.  When given, spatial maps of posterior
        accuracy / uncertainty / reliability are drawn at the observation
        locations (:mod:`pestpp_ies_post.spatial`).
    pp_file:
        Optional pilot-point file (``name x y zone parval``).  When given,
        spatial maps of pilot-point posterior value / spread / uncertainty
        reduction are drawn.
    shapefiles:
        Optional list of boundary shapefiles overlaid on the spatial maps
        (needs ``geopandas``).
    marthe_rma:
        Optional Marthe ``.rma`` project file.  When given (with ``pymarthe``
        installed) the base property ``field_prop`` is rendered on the true
        model grid for ``field_layer`` with pilot-point uncertainty overlaid.
    marthe_config:
        Optional ``pymarthe`` parameterisation config file.  When given, the
        posterior parameter ensemble is replayed through the config to
        reconstruct a *per-cell* posterior field ensemble (mixed pilot-point +
        ZPC aware) and map its mean / std / CV on the true grid.
    field_prop, field_layer:
        Property name (``None`` = first grid property in the config) and layer
        for the ``pymarthe`` field figures.
    field_max_reals:
        Cap on the number of realisations reconstructed (speed control).
    make_timeseries:
        Whether to attempt per-site time-series figures.

    Returns
    -------
    list of Path
        Every figure successfully written.
    """
    res = IesResults(pst_file, verbose=verbose)
    # let the caller pin which iterations are the prior / posterior; every plot
    # reads res.posterior_iter / res.prior_iter, so this propagates everywhere
    if iteration is not None:
        if iteration not in res.available_iters:
            print(f"[ies-post] warning: iteration {iteration} not found "
                  f"(available {res.available_iters}); using "
                  f"{res.posterior_iter}")
        else:
            res.posterior_iter = iteration
    if prior_iteration is not None:
        res.prior_iter = prior_iteration
    print("\n" + "=" * 64)
    print(res.summary())
    print("=" * 64 + "\n")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # write the inventory next to the plots for provenance
    (out / f"{res.case}_00_summary.txt").write_text(res.summary())

    # plots that can group observations by type when a .histo is available
    _histo_aware = {
        plots.plot_one_to_one,
        plots.plot_residual_histograms,
        plots.plot_residual_vs_simulated,
    }

    written: List[Path] = []
    for func, kwargs, label in _PIPELINE:
        print(f"[ies-post] --> {label}")
        call_kwargs = dict(kwargs)
        if histo_file and func in _histo_aware:
            call_kwargs["histo_file"] = histo_file
        try:
            result = func(res, str(out), **call_kwargs)
            if result is not None:
                written.append(result)
        except Exception:  # noqa: BLE001 - keep the pipeline alive
            print(f"[ies-post] !! '{label}' failed:")
            traceback.print_exc()

    if make_timeseries:
        print("[ies-post] --> time series")
        try:
            ts_dir = out / "timeseries"
            obs_meta = None
            if marthe_pastp:
                # reconstruct datetimes straight from the Marthe model files
                obs_meta = marthe.build_obs_meta(
                    res, marthe_pastp, mart_file=marthe_mart)
            paths = timeseries.plot_obs_timeseries(
                res, str(ts_dir), obs_meta=obs_meta, obs_meta_csv=obs_meta_csv)
            written.extend(paths)
        except Exception:  # noqa: BLE001
            print("[ies-post] !! time series failed:")
            traceback.print_exc()

    # ---- spatial maps (only when coordinates / model files are supplied) ----
    if histo_file:
        print("[ies-post] --> spatial obs performance")
        try:
            written.extend(spatial.plot_spatial_obs_performance(
                res, str(out), histo_file=histo_file, shapefiles=shapefiles,
                model_rma=marthe_rma))
        except Exception:  # noqa: BLE001
            print("[ies-post] !! spatial obs performance failed:")
            traceback.print_exc()
    if pp_file:
        print("[ies-post] --> pilot-point uncertainty")
        try:
            written.extend(spatial.plot_pilotpoint_uncertainty(
                res, str(out), pp_file=pp_file, shapefiles=shapefiles))
        except Exception:  # noqa: BLE001
            print("[ies-post] !! pilot-point uncertainty failed:")
            traceback.print_exc()
    if marthe_rma:
        print("[ies-post] --> Marthe property field")
        try:
            fig = spatial.plot_marthe_property_field(
                res, str(out), model_rma=marthe_rma,
                prop=field_prop or "permh",
                layer=field_layer, pp_file=pp_file)
            if fig is not None:
                written.append(fig)
        except Exception:  # noqa: BLE001
            print("[ies-post] !! Marthe property field failed:")
            traceback.print_exc()
    if marthe_config:
        print("[ies-post] --> posterior field statistics (mixed pp + ZPC)")
        try:
            written.extend(spatial.plot_posterior_field_stats(
                res, str(out), configfile=marthe_config, prop=field_prop,
                layer=field_layer, max_reals=field_max_reals))
        except Exception:  # noqa: BLE001
            print("[ies-post] !! posterior field statistics failed:")
            traceback.print_exc()

    print(f"\n[ies-post] DONE - {len(written)} figure(s) written to {out}\n")
    return written
