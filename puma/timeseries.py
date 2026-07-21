"""
timeseries.py
=============

Optional time-series plotting of observation ensembles.

PEST++ IES does not itself store observation times, so this module needs a
small metadata table telling it which observation belongs to which site and
at what date.  Three sources are supported, tried in order:

1. an explicit ``obs_meta`` DataFrame passed by the caller with columns
   ``obsnme`` (index or column), ``site`` and ``datetime``;
2. a CSV given by ``obs_meta_csv`` with the same columns;
3. automatic parsing of ``time``/``date`` columns already present in the
   control file's ``* observation_data`` external block (some PEST++
   workflows carry these).

If none is available the module returns ``None`` and the rest of the toolbox
is unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .results import IesResults
from . import utils
from .utils import PRIOR_COLOR, POST_COLOR, MEAS_COLOR
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_obs_lookup(csv_path: str) -> dict[str, str]:
    df = pd.read_csv(csv_path)

    return {
        str(obgnme).strip().lower(): obs_name
        for obgnme, obs_name in zip(df["obgnme"], df["obs_name"])
    }


def _obs_group_name(obgnme: str, lookup_csv: str) -> str:
    lookup = _load_obs_lookup(lookup_csv)
    key = str(obgnme).strip().lower()

    return lookup.get(key, obgnme)


def _resolve_meta(res: IesResults,
                  obs_meta: Optional[pd.DataFrame],
                  obs_meta_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if obs_meta is not None:
        meta = obs_meta.copy()
    elif obs_meta_csv is not None and Path(obs_meta_csv).exists():
        meta = pd.read_csv(obs_meta_csv)
    else:
        # try to pull time/site straight from observation_data
        od = res.pst.observation_data
        lower = {c.lower(): c for c in od.columns}
        time_col = next((lower[c] for c in ("datetime", "date", "time")
                         if c in lower), None)
        site_col = next((lower[c] for c in ("site", "usecol", "obsnme_base",
                                            "obgnme")
                         if c in lower), None)
        if time_col is None:
            return None
        meta = pd.DataFrame({
            "obsnme": od.index,
            "datetime": od[time_col].values,
            "site": od[site_col].values if site_col else od["obgnme"].values,
        })
    # normalise
    if "obsnme" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "obsnme"})
    meta["obsnme"] = meta["obsnme"].astype(str)
    if "site" not in meta.columns:
        meta["site"] = meta["obsnme"]
    meta["datetime"] = pd.to_datetime(meta["datetime"], errors="coerce")
    meta = meta.dropna(subset=["datetime"])
    return meta if not meta.empty else None


def plot_obs_timeseries(res: IesResults, output_dir: str,
                        obs_meta: Optional[pd.DataFrame] = None,
                        obs_meta_csv: Optional[str] = None,
                        site_lookup: Optional[str] = None,
                        iteration: Optional[int] = None,
                        min_points: int = 5,
                        max_sites: int = 60,
                        ci: tuple[float, float] = (0.05, 0.95)) -> list[Path]:
    """One time-series figure per site: posterior ensemble band + measured.

    Returns the list of figures written (possibly empty).
    """
    meta = _resolve_meta(res, obs_meta, obs_meta_csv)
    if meta is None:
        print("[puma] no time metadata available; skipping time series")
        return []

    it = res.posterior_iter if iteration is None else iteration
    post = res.obs_ensemble(it)
    if post is None:
        return []
    obs = res.pst.observation_data
    obs_val = pd.to_numeric(obs["obsval"], errors="coerce")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.apply_style()

    written: list[Path] = []
    lo_q, hi_q = ci
    sites = meta["site"].unique().tolist()[:max_sites]
    for site in sites:
        if site_lookup is not None and Path(site_lookup).exists():
            site_name = _obs_group_name(site, site_lookup)
            
        sm = meta.loc[meta.site == site.lower()].sort_values("datetime")
        #sm = meta.loc[meta["site"].str.strip().str.casefold() == site.strip().casefold()]

        names = [n for n in sm["obsnme"] if n in post.columns]
        if len(names) < min_points:
            continue
        sm = sm.loc[sm["obsnme"].isin(names)]
        dates = sm["datetime"].values
        sub = post[sm["obsnme"].tolist()]
        med = sub.median(axis=0).values
        lo = sub.quantile(lo_q, axis=0).values
        hi = sub.quantile(hi_q, axis=0).values
        meas = obs_val.reindex(sm["obsnme"]).values
        wt = pd.to_numeric(obs["weight"], errors="coerce").reindex(
            sm["obsnme"]).values

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.fill_between(dates, lo, hi, color=POST_COLOR, alpha=0.2,
                        label=f"posterior {round((hi_q-lo_q)*100)}% range")
        ax.plot(dates, med, color=POST_COLOR, lw=1.4, label="posterior median")
        mmask = np.isfinite(meas) & (wt > 0)
        ax.plot(np.asarray(dates)[mmask], meas[mmask], "o", color=MEAS_COLOR,
                ms=3.5, label="measured")
        ax.set_title(f"Time-series fit: {site}--{site_name}", fontsize=12)
        ax.set_xlabel("date")
        ax.set_ylabel("value")
        ax.legend(loc="best", fontsize=8)
        clean = str(site).replace("/", "_").replace(" ", "_")
        path = out_dir / f"{res.case}_TS_{clean}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    print(f"[puma]   saved {len(written)} time-series figure(s)")
    return written
