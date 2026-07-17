"""
spatial.py
==========

Spatial post-processing of a PEST++ IES run for a **Marthe** model:
maps of calibration *accuracy* and posterior *uncertainty* in both

* **data space** - at the observation (HISTO) locations, and
* **parameter space** - at the pilot points / zones that were adjusted.

The module is layered so it is useful with or without ``pymarthe`` installed:

* Observation-location and pilot-point maps need only the IES ensembles plus
  ``x, y`` coordinates.  Those coordinates are parsed straight from the Marthe
  ``.histo`` file (:func:`parse_marthe_histo`) or a pilot-point file
  (:func:`parse_pilot_points`), so these figures work anywhere.
* :func:`plot_marthe_property_field` additionally uses ``pymarthe``'s
  :class:`MartheField` to draw the adjusted property on the true model grid and
  overlay the pilot-point posterior uncertainty.  It imports ``pymarthe`` lazily
  and explains what it needs if the package or the model files are absent.

Optional dependencies (``pymarthe``, ``geopandas``) are imported inside the
functions that use them so importing this module never fails.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .results import IesResults
from . import utils
from .utils import POST_COLOR, coverage_fraction


# ======================================================================
# Marthe .histo parsing (observation coordinates)
# ======================================================================
_HISTO_COLS = ["name", "obs_type", "x", "y", "layer",
               "col", "line", "plane", "affluent", "troncon"]


def parse_marthe_histo(histo_file: str) -> pd.DataFrame:
    """Parse a Marthe ``.histo`` file into an observation-metadata table.

    Handles the three location formats Marthe writes:

    * projected coordinates ``/XCOO:X= .. Y= .. L= .. ; Name= ..``,
    * grid-cell references  ``/MAIL:C= .. L= .. P= .. ; Name= ..`` (column,
      line, plane) - x/y are filled later from the model grid, and
    * river affluent/troncon ``/AFFL:Afflu= .. ,Tronc= .. ; Name= ..``.

    Always returns a DataFrame with the full column set (empty if nothing
    matched), so callers never hit a missing-column error.
    """
    xcoo = re.compile(
        r"/(?P<type>[^/]+?)\s*/HISTO/\s*=\s*/XCOO:X=\s*"
        r"(?P<x>[-\d\.]+)\s*Y=\s*(?P<y>[-\d\.]+)\s*L=\s*(?P<layer>\d+)\s*;"
        r"\s*Name=(?P<name>.+)")
    mail = re.compile(
        r"/(?P<type>[^/]+?)\s*/HISTO/\s*=\s*/MAIL:C=\s*"
        r"(?P<col>\d+)\s*L=\s*(?P<line>\d+)\s*P=\s*(?P<plane>\d+)\s*;"
        r"\s*Name=(?P<name>.+)")
    riv = re.compile(
        r"/(?P<type>[^/]+?)\s*/HISTO/\s*=\s*/AFFL:Afflu=\s*"
        r"(?P<affluent>\d+)\s*,\s*Tron[^=]*=\s*(?P<troncon>\d+)\s*;"
        r"\s*Name=(?P<name>.+)")
    rows: List[dict] = []
    with open(histo_file, "r", encoding="latin-1") as fh:
        for line in fh:
            line = line.strip().rstrip("\r")
            m = xcoo.search(line)
            if m:
                d = m.groupdict()
                rows.append({"name": d["name"].strip(),
                             "obs_type": d["type"].strip(),
                             "x": float(d["x"]), "y": float(d["y"]),
                             "layer": int(d["layer"])})
                continue
            m = mail.search(line)
            if m:
                d = m.groupdict()
                rows.append({"name": d["name"].strip(),
                             "obs_type": d["type"].strip(),
                             "col": int(d["col"]), "line": int(d["line"]),
                             "plane": int(d["plane"]),
                             "layer": int(d["plane"])})
                continue
            m = riv.search(line)
            if m:
                d = m.groupdict()
                rows.append({"name": d["name"].strip(),
                             "obs_type": d["type"].strip(),
                             "affluent": int(d["affluent"]),
                             "troncon": int(d["troncon"])})
    df = pd.DataFrame(rows)
    return df.reindex(columns=_HISTO_COLS) if not df.empty \
        else pd.DataFrame(columns=_HISTO_COLS)


def build_obs_coords(res: IesResults,
                     histo_file: Optional[str] = None,
                     coords: Optional[pd.DataFrame] = None,
                     model_rma: Optional[str] = None
                     ) -> Optional[pd.DataFrame]:
    """Map each non-zero observation group (a HISTO site) to ``x, y, layer``.

    ``coords`` (if given) must have columns ``site, x, y`` and optionally
    ``layer``/``obs_type``; otherwise coordinates come from ``histo_file``.
    The Marthe convention (used in the reference workflow) is that the PEST
    observation-group name equals the lower-cased HISTO ``Name``.

    For a ``.histo`` using grid-cell (``/MAIL``) references instead of
    projected coordinates, pass ``model_rma`` and the cell col/line/plane are
    converted to x,y via the model grid (needs ``pymarthe``).
    """
    if coords is not None:
        df = coords.copy()
        if "site" not in df.columns:
            df = df.rename(columns={df.columns[0]: "site"})
        df["site"] = df["site"].astype(str).str.lower()
        return df
    if histo_file is None or not Path(histo_file).exists():
        return None
    hd = parse_marthe_histo(histo_file)
    if hd.empty:
        return None
    hd["site"] = hd["name"].astype(str).str.strip().str.lower()

    have_xy = hd.dropna(subset=["x", "y"])
    if not have_xy.empty:
        return have_xy[["site", "x", "y", "layer", "obs_type"]]

    # only grid-cell references available -> convert via the model grid
    need = hd.dropna(subset=["col", "line"])
    if need.empty:
        print("[puma] spatial: .histo has no usable coordinates; skipping")
        return None
    if model_rma is None or not Path(model_rma).exists():
        print("[puma] spatial: .histo uses grid-cell (/MAIL) references; "
              "pass model_rma to convert them to x,y. Skipping.")
        return None
    xy = _cell_to_xy(model_rma, need)
    return xy


def _cell_to_xy(model_rma: str, need: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Convert Marthe grid-cell (col/line/plane) references to x,y.

    Marthe's ``/MAIL`` uses 1-based column (C=j), line (L=i) and plane
    (P=layer); the model grid recarray carries 0-based ``i, j, layer`` with
    ``x, y``.  Best-effort match on the main grid (``inest == 0``).
    """
    try:
        from pymarthe import MartheModel
        from pymarthe.mfield import MartheField
    except Exception as exc:  # noqa: BLE001
        print(f"[puma] spatial: pymarthe needed for cell->xy ({exc})")
        return None
    mm = MartheModel(_winpath(model_rma), spatial_index=False)
    prop = next(iter(getattr(mm, "prop", {})), None) or "permh"
    mm.load_prop(prop)
    rec = mm.prop[prop].get_data()
    grid = pd.DataFrame({k: rec[k] for k in ("layer", "inest", "i", "j",
                                             "x", "y")})
    main = grid.loc[grid["inest"] == 0]
    lut = main.set_index(["layer", "i", "j"])[["x", "y"]]
    out = []
    for _, r in need.iterrows():
        key = (int(r["plane"]) - 1, int(r["line"]) - 1, int(r["col"]) - 1)
        if key in lut.index:
            xy = lut.loc[key]
            out.append((r["site"], float(xy["x"]), float(xy["y"]),
                        r["layer"], r["obs_type"]))
    if not out:
        print("[puma] spatial: no grid cells matched the /MAIL references")
        return None
    return pd.DataFrame(out, columns=["site", "x", "y", "layer", "obs_type"])


# ======================================================================
# helper: per-site accuracy / uncertainty / coverage from the ensemble
# ======================================================================
def _site_performance(res: IesResults,
                      ci: tuple[float, float] = (0.05, 0.95),
                      iteration: Optional[int] = None) -> pd.DataFrame:
    """Aggregate posterior obs ensemble to one row per site (obgnme).

    Columns: site, obs_type-ish (from group), n_weighted, rmse (accuracy),
    mean_std (posterior spread), coverage (reliability).
    """
    it = res.posterior_iter if iteration is None else int(iteration)
    post = res.obs_ensemble(it)
    if post is None:
        return pd.DataFrame()
    obs = res.pst.observation_data.copy()
    obs["obsval"] = pd.to_numeric(obs["obsval"], errors="coerce")
    obs["weight"] = pd.to_numeric(obs["weight"], errors="coerce")
    lo_q, hi_q = ci

    records = []
    for site, sub in obs.groupby("obgnme"):
        names = [n for n in sub.index if n in post.columns]
        if not names:
            continue
        ens = post[names]
        sim_mean = ens.mean(axis=0)
        sim_std = ens.std(axis=0)
        sim_lo = ens.quantile(lo_q, axis=0)
        sim_hi = ens.quantile(hi_q, axis=0)
        meas = obs.loc[names, "obsval"]
        wt = obs.loc[names, "weight"]
        w = (wt > 0) & np.isfinite(meas) & (meas > -9000)
        n_w = int(w.sum())
        if n_w == 0:
            # unmeasured site: still report predictive spread
            records.append((site, 0, np.nan, float(sim_std.mean()), np.nan))
            continue
        resid = (sim_mean[w.values] - meas[w.values]).values
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        cov = coverage_fraction(meas[w.values].values,
                                sim_lo[w.values].values,
                                sim_hi[w.values].values)
        records.append((site, n_w, rmse, float(sim_std[w.values].mean()), cov))
    return pd.DataFrame(records,
                        columns=["site", "n_weighted", "rmse",
                                 "mean_std", "coverage"])


# ======================================================================
# 1. Spatial observation performance (data space)
# ======================================================================
def plot_spatial_obs_performance(res: IesResults, output_dir: str,
                                 histo_file: Optional[str] = None,
                                 coords: Optional[pd.DataFrame] = None,
                                 shapefiles: Optional[List[str]] = None,
                                 model_rma: Optional[str] = None,
                                 iteration: Optional[int] = None,
                                 layer: Optional[int] = None,
                                 ci: tuple[float, float] = (0.05, 0.95)
                                 ) -> List[Path]:
    """Maps of posterior accuracy, uncertainty and reliability at obs sites.

    Produces three spatial scatter maps (one panel each), coloured by:

    * **accuracy**    - posterior RMSE at each site (lower = better fit),
    * **uncertainty** - mean posterior ensemble standard deviation,
    * **reliability** - fraction of measurements inside the posterior CI.

    ``iteration`` selects which iteration's ensemble to map (default posterior);
    ``layer`` restricts the map to sites in that model layer (when the ``.histo``
    carries a layer).  Requires site coordinates from a Marthe ``.histo`` file
    or a ``coords`` table.  Optional ``shapefiles`` (needs ``geopandas``) draw
    the model outline.  Returns the figure paths written.
    """
    site_xy = build_obs_coords(res, histo_file=histo_file, coords=coords,
                               model_rma=model_rma)
    if site_xy is None:
        print("[puma] spatial: no obs coordinates (need .histo or coords); "
              "skipping")
        return []
    if layer is not None and "layer" in site_xy.columns:
        site_xy = site_xy.loc[
            pd.to_numeric(site_xy["layer"], errors="coerce") == layer]
        if site_xy.empty:
            print(f"[puma] spatial: no sites in layer {layer}; skipping")
            return []
    perf = _site_performance(res, ci=ci, iteration=iteration)
    if perf.empty:
        print("[puma] spatial: no posterior obs ensemble; skipping")
        return []

    df = perf.merge(site_xy, on="site", how="inner").dropna(subset=["x", "y"])
    if df.empty:
        print("[puma] spatial: no sites matched coordinates; skipping")
        return []
    lay_tag = "" if layer is None else f"_L{layer}"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    utils.apply_style()

    # optional model outline
    gdfs = _read_shapefiles(shapefiles)

    written: List[Path] = []
    panels = [
        ("rmse", "Posterior RMSE (accuracy)", "RMSE", "viridis_r", False),
        ("mean_std", "Posterior ensemble std (uncertainty)",
         "std", "plasma", False),
        ("coverage", "CI coverage of measurements (reliability)",
         "coverage", "RdYlGn", True),
    ]
    for col, title, clabel, cmap, is_frac in panels:
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8.5))
        for g in gdfs:
            g.plot(ax=ax, color="none", edgecolor="#2c3e50", lw=1.2, zorder=1)
        vmin, vmax = (0, 1) if is_frac else (None, None)
        sc = ax.scatter(sub.x, sub.y, c=sub[col], cmap=cmap, s=55,
                        vmin=vmin, vmax=vmax, edgecolor="k", linewidth=0.5,
                        alpha=0.9, zorder=3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label(clabel)
        it_used = res.posterior_iter if iteration is None else iteration
        lay_txt = "" if layer is None else f", layer {layer}"
        ax.set_title(f"{title}\n[{res.case}, iter {it_used}{lay_txt}]")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        fname = f"{res.case}_SPATIAL_{col}{lay_tag}.png"
        fig.savefig(out / fname)
        plt.close(fig)
        written.append(out / fname)
        print(f"[puma]   saved {fname}")
    return written


# ======================================================================
# Pilot-point parsing + parameter-space uncertainty maps
# ======================================================================
def parse_pilot_points(pp_file: str) -> pd.DataFrame:
    """Parse a standard (pyemu/PEST) pilot-point file.

    Expected whitespace-delimited columns ``name x y zone parval`` (extra
    columns are ignored).  Returns ``parnme, x, y, zone`` (parval dropped).
    """
    df = pd.read_csv(pp_file, sep=r"\s+", header=None, comment="#")
    ncol = df.shape[1]
    names = ["parnme", "x", "y", "zone", "parval"][:ncol]
    df = df.iloc[:, :len(names)]
    df.columns = names
    df["parnme"] = df["parnme"].astype(str)
    return df[["parnme", "x", "y"] + (["zone"] if "zone" in df.columns else [])]


def plot_pilotpoint_uncertainty(res: IesResults, output_dir: str,
                                pp_file: Optional[str] = None,
                                pp_coords: Optional[pd.DataFrame] = None,
                                shapefiles: Optional[List[str]] = None,
                                iteration: Optional[int] = None,
                                prior_iteration: Optional[int] = None
                                ) -> List[Path]:
    """Spatial maps of pilot-point posterior value, spread and unc. reduction.

    For every pilot-point parameter present in both the prior and posterior
    parameter ensembles, draws three maps coloured by:

    * **posterior mean** value (the calibrated field),
    * **posterior std** (remaining parameter uncertainty),
    * **uncertainty reduction** ``1 - sigma_post/sigma_prior`` (how strongly the
      data constrained each location).

    ``pp_coords`` (columns ``parnme, x, y``) or a ``pp_file`` supplies the
    coordinates.  Log-transformed parameters are shown in native units for the
    mean and in log-space for the spread/reduction.  Returns figure paths.
    """
    post_it = res.posterior_iter if iteration is None else int(iteration)
    prior_it = res.prior_iter if prior_iteration is None else int(prior_iteration)
    prior = res.par_ensemble(prior_it)
    post = res.par_ensemble(post_it)
    if prior is None or post is None:
        print("[puma] spatial: parameter ensembles unavailable; skipping pp")
        return []
    if pp_coords is None:
        if pp_file is None or not Path(pp_file).exists():
            print("[puma] spatial: no pilot-point coordinates; skipping pp")
            return []
        pp_coords = parse_pilot_points(pp_file)

    pp_coords = pp_coords.copy()
    pp_coords["parnme"] = pp_coords["parnme"].astype(str)
    common = [p for p in pp_coords["parnme"]
              if p in prior.columns and p in post.columns]
    if not common:
        print("[puma] spatial: pilot-point names do not match the "
              "ensemble columns; skipping pp")
        return []
    pp = pp_coords.set_index("parnme").loc[common]

    # log flag from the control file
    pdata = res.pst.parameter_data
    is_log = (pdata.set_index("parnme")["partrans"] == "log").to_dict() \
        if "partrans" in pdata.columns else {}

    stat = pd.DataFrame(index=common)
    stat["x"] = pp["x"].values
    stat["y"] = pp["y"].values
    stat["post_mean"] = post[common].mean(axis=0).values
    # spread/reduction computed in the space the parameter was estimated in
    def _est_space(df):
        out = df[common].copy()
        for p in common:
            if is_log.get(p, False):
                out[p] = np.log10(out[p].clip(lower=1e-30))
        return out
    pri_e, pos_e = _est_space(prior), _est_space(post)
    stat["post_std"] = pos_e.std(axis=0).values
    sd_pri = pri_e.std(axis=0).replace(0, np.nan)
    stat["unc_reduction"] = (1 - pos_e.std(axis=0) / sd_pri).values * 100.0

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    utils.apply_style()
    gdfs = _read_shapefiles(shapefiles)

    written: List[Path] = []
    panels = [
        ("post_mean", "Pilot-point posterior mean (calibrated field)",
         "value", "viridis", (None, None)),
        ("post_std", "Pilot-point posterior std (parameter uncertainty)",
         "std (est. space)", "plasma", (None, None)),
        ("unc_reduction", "Parameter uncertainty reduction",
         "1 - $\\sigma_{post}/\\sigma_{prior}$ (%)", "RdYlGn", (0, 100)),
    ]
    for col, title, clabel, cmap, (vmin, vmax) in panels:
        sub = stat.dropna(subset=[col])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 8.5))
        for g in gdfs:
            g.plot(ax=ax, color="none", edgecolor="#2c3e50", lw=1.2, zorder=1)
        sc = ax.scatter(sub.x, sub.y, c=sub[col], cmap=cmap, s=60,
                        vmin=vmin, vmax=vmax, edgecolor="k", linewidth=0.5,
                        alpha=0.9, zorder=3)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label(clabel)
        ax.set_title(f"{title}\n[{res.case}, iter {post_it}]")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        fname = f"{res.case}_SPATIAL_pp_{col}.png"
        fig.savefig(out / fname)
        plt.close(fig)
        written.append(out / fname)
        print(f"[puma]   saved {fname}")
    return written


# ======================================================================
# 3. pymarthe field overlay (optional)
# ======================================================================
def plot_marthe_property_field(res: IesResults, output_dir: str,
                               model_rma: str,
                               prop: str = "permh",
                               layer: int = 0,
                               pp_file: Optional[str] = None,
                               pp_coords: Optional[pd.DataFrame] = None,
                               iteration: Optional[int] = None,
                               prior_iteration: Optional[int] = None,
                               log: bool = True) -> Optional[Path]:
    """Draw a Marthe property field and overlay pilot-point uncertainty.

    Uses ``pymarthe`` to render the property (e.g. ``permh``) for a given
    ``layer`` on the true model grid via :meth:`MartheField.plot`, then overlays
    the pilot points sized by posterior std and coloured by uncertainty
    reduction - i.e. *where the calibrated field is well constrained and where
    it is not*.

    Requires ``pymarthe`` and access to the Marthe model files referenced by
    ``model_rma`` (the ``.rma`` project file).  Returns ``None`` (with an
    explanatory message) if either is unavailable.
    """
    try:
        from pymarthe import MartheModel
        from pymarthe.mfield import MartheField
    except Exception as exc:  # noqa: BLE001
        print(f"[puma] spatial: pymarthe not available ({exc}); "
              f"skipping property-field plot")
        return None
    if not Path(model_rma).exists():
        print(f"[puma] spatial: model file '{model_rma}' not found; "
              f"skipping property-field plot")
        return None

    # coordinates + posterior stats for the pilot points
    if pp_coords is None and pp_file is not None and Path(pp_file).exists():
        pp_coords = parse_pilot_points(pp_file)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    utils.apply_style()

    mm = MartheModel(_winpath(model_rma), spatial_index=False, modelgrid=True)
    mm.load_prop(prop)                # load base property field from its file
    mf = mm.prop[prop]

    fig, ax = plt.subplots(figsize=(12, 9))
    mf.plot(ax=ax, layer=layer, log=log)

    if pp_coords is not None:
        post_it = res.posterior_iter if iteration is None else int(iteration)
        prior_it = res.prior_iter if prior_iteration is None \
            else int(prior_iteration)
        post = res.par_ensemble(post_it)
        prior = res.par_ensemble(prior_it)
        pp_coords = pp_coords.copy()
        pp_coords["parnme"] = pp_coords["parnme"].astype(str)
        common = [p for p in pp_coords["parnme"]
                  if post is not None and p in post.columns]
        if common:
            pp = pp_coords.set_index("parnme").loc[common]
            pdata = res.pst.parameter_data
            is_log = (pdata.set_index("parnme")["partrans"] == "log").to_dict()

            def _sp(df):
                o = df[common].copy()
                for p in common:
                    if is_log.get(p, False):
                        o[p] = np.log10(o[p].clip(lower=1e-30))
                return o
            std_post = _sp(post).std(axis=0)
            if prior is not None:
                sd_pri = _sp(prior).std(axis=0).replace(0, np.nan)
                redux = (1 - std_post / sd_pri) * 100.0
            else:
                redux = std_post * 0
            sizes = 30 + 300 * (std_post / std_post.max()).fillna(0).values
            sc = ax.scatter(pp["x"].values, pp["y"].values, s=sizes,
                            c=redux.values, cmap="RdYlGn", vmin=0, vmax=100,
                            edgecolor="k", linewidth=0.6, zorder=4)
            cb = fig.colorbar(sc, ax=ax, shrink=0.6)
            cb.set_label("uncertainty reduction (%)")

    ax.set_title(f"{prop} (layer {layer}) with pilot-point uncertainty\n"
                 f"[{res.case}]")
    ax.set_aspect("equal")
    fname = f"{res.case}_SPATIAL_field_{prop}_L{layer}.png"
    fig.savefig(out / fname)
    plt.close(fig)
    print(f"[puma]   saved {fname}")
    return out / fname


# ======================================================================
# 4. Full posterior field reconstruction (mixed pilot points + ZPC)
# ======================================================================
def _winpath(p: str) -> str:
    """Normalise a (possibly Windows) path so it resolves on any OS."""
    return str(p).replace("\\", "/").strip()


def reconstruct_field_ensemble(res: IesResults,
                               configfile: str,
                               prop: Optional[str] = None,
                               iteration: Optional[int] = None,
                               max_reals: Optional[int] = None,
                               model_dir: Optional[str] = None):
    """Rebuild a per-cell posterior ensemble of a Marthe property field.

    Handles **mixed pilot-point + ZPC** parameterisations natively by replaying
    the exact reconstruction ``pymarthe`` uses in the forward run: for each
    realisation the parameter values are written into the model's template
    inputs (``pst.write_input_files``) and then, for the grid property, the
    pilot-point + ZPC parameter files are applied through
    :meth:`MartheField.set_data_from_parfile` with the property's ``izone`` -
    which kriges the pilot points (via their ``.fac`` factors) and fills the
    ZPC zones.  Per-cell values are stacked across realisations.

    The parameterisation is read from the ``pymarthe`` config file, so no
    assumptions about the pp/ZPC split are hard-coded.  This is version-tolerant
    (it uses only the config fields it needs) and normalises Windows paths, so a
    config written on Windows reconstructs on Linux/macOS too.

    Parameters
    ----------
    prop:
        Property to reconstruct (e.g. ``permh``).  If ``None`` the first grid
        property found in the config is used.
    model_dir:
        Directory the config's relative paths resolve against (defaults to the
        config file's own directory).

    Returns
    -------
    ``(mm, prop, layers, stack)`` or ``None``
        ``mm`` is the loaded ``MartheModel``, ``prop`` the property name,
        ``layers`` the per-cell layer index (for per-layer plotting) and
        ``stack`` a ``(n_real, n_cell)`` array of the property over active cells.
    """
    try:
        from pymarthe import MartheModel
        from pymarthe.mfield import MartheField
        from pymarthe.utils import pest_utils
    except Exception as exc:  # noqa: BLE001
        print(f"[puma] spatial: pymarthe unavailable ({exc}); "
              f"cannot reconstruct fields")
        return None
    if not Path(configfile).exists():
        print(f"[puma] spatial: config '{configfile}' not found")
        return None

    it = res.posterior_iter if iteration is None else iteration
    ens = res._load_ensemble(it, "par")  # native pst (estimation) space
    if ens is None:
        print("[puma] spatial: parameter ensemble unavailable")
        return None
    reals = list(ens.index)
    if max_reals:
        reals = reals[:max_reals]

    hdic, pdics, _ = pest_utils.read_config(configfile)
    cfg_dir = Path(configfile).resolve().parent if model_dir is None \
        else Path(model_dir)

    # pick the grid parameter block for the requested property
    grid_pdics = [p for p in pdics if p.get("type") == "grid"]
    if not grid_pdics:
        print("[puma] spatial: no 'grid' parameter block in config")
        return None
    pdic = next((p for p in grid_pdics if p["property name"] == prop),
                grid_pdics[0])
    prop = pdic["property name"]
    use_imask = pdic.get("use_imask", "True") == "True"
    btrans = pdic.get("btrans", "none")
    parfiles = [_winpath(pf) for pf in pdic["parfile"].split(",")]

    # build the model once (paths in the config are resolved against cfg_dir)
    import os as _os
    cwd = _os.getcwd()
    _os.chdir(cfg_dir)
    try:
        mm = MartheModel(_winpath(hdic["Model full path"]), spatial_index=False)
        mm.load_prop(prop, use_imask=use_imask)
        izone = MartheField(f"i{prop}", _winpath(pdic["izone"]), mm,
                            use_imask=use_imask)
        pst = res.pst
        # layer index per active cell (constant across realisations)
        layers = mm.prop[prop].get_data()["layer"]

        stack = []
        for r in reals:
            pst.parameter_data["parval1"] = ens.loc[r, pst.par_names].values
            pst.write_input_files()          # templates -> parfiles
            for pf in parfiles:
                mm.prop[prop].set_data_from_parfile(
                    parfile=pf, izone=izone, btrans=btrans)
            stack.append(mm.prop[prop].get_data()["value"].copy())
    finally:
        _os.chdir(cwd)

    import numpy as _np
    print(f"[puma] spatial: reconstructed {prop} for {len(reals)} "
          f"realisation(s) over {stack[0].size} cells")
    return mm, prop, _np.asarray(layers), _np.vstack(stack)


def plot_posterior_field_stats(res: IesResults, output_dir: str,
                               configfile: str,
                               prop: Optional[str] = None,
                               layer: int = 0,
                               iteration: Optional[int] = None,
                               max_reals: Optional[int] = None,
                               masked_values=(-9999.0, 0.0, 9999.0),
                               log: bool = True) -> List[Path]:
    """Cell-by-cell posterior **mean**, **std** and **CV** maps of a property.

    Reconstructs the field ensemble via :func:`reconstruct_field_ensemble`
    (mixed pilot-point + ZPC aware) and renders the three statistics on the true
    Marthe grid via :meth:`MartheField.plot` for the requested ``layer``.
    ``std`` and ``CV`` are the model-grounded *spatial uncertainty*; ``CV``
    (std/|mean|) is the dimensionless version.  Returns the figure paths.
    """
    result = reconstruct_field_ensemble(
        res, configfile, prop=prop, iteration=iteration, max_reals=max_reals)
    if result is None:
        return []
    import numpy as _np
    from pymarthe.mfield import MartheField

    mm, prop, _layers, stack = result
    masked = _np.asarray(masked_values, dtype=float)
    clean = _np.where(_np.isin(stack, masked), _np.nan, stack)
    mean = _np.nanmean(clean, axis=0)
    std = _np.nanstd(clean, axis=0)
    cv = _np.where(_np.abs(mean) > 0, std / _np.abs(mean), _np.nan)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    utils.apply_style()
    written: List[Path] = []

    it = iteration if iteration is not None else res.posterior_iter
    for stat_name, values, use_log in [
            ("mean", mean, log), ("std", std, False), ("cv", cv, False)]:
        # build a MartheField carrying this statistic for plotting
        fld = MartheField(prop, mm.prop[prop].data.copy(), mm,
                          use_imask=mm.prop[prop].use_imask)
        rec = fld.data
        rec["value"] = _np.where(_np.isfinite(values), values, -9999.0)
        fld.set_data(rec)
        fig, ax = plt.subplots(figsize=(12, 9))
        try:
            fld.plot(ax=ax, layer=layer, log=use_log)
        except Exception as exc:  # noqa: BLE001
            print(f"[puma] spatial: field plot failed for {stat_name} "
                  f"({exc})")
            plt.close(fig)
            continue
        ax.set_title(f"Posterior {prop} {stat_name} (layer {layer})\n"
                     f"[{res.case}, iter {it}]")
        ax.set_aspect("equal")
        fname = f"{res.case}_SPATIAL_fieldstat_{prop}_{stat_name}_L{layer}.png"
        fig.savefig(out / fname)
        plt.close(fig)
        written.append(out / fname)
        print(f"[puma]   saved {fname}")
    return written


# ======================================================================
def _read_shapefiles(shapefiles: Optional[List[str]]):
    """Load optional boundary shapefiles; empty list if geopandas missing."""
    if not shapefiles:
        return []
    try:
        import geopandas as gpd
    except Exception:  # noqa: BLE001
        print("[puma] spatial: geopandas not installed; "
              "skipping shapefile overlay")
        return []
    gdfs = []
    for shp in shapefiles:
        if Path(shp).exists():
            gdfs.append(gpd.read_file(shp))
    return gdfs
