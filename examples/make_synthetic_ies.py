#!/usr/bin/env python3
"""
make_synthetic_ies.py
=====================

Generate a small, self-contained *synthetic* PEST++ IES output set so the
toolbox can be exercised without a real model run.  It writes, into the target
directory, files that mimic the real PEST++ IES naming/format:

    synth_ies.pst
    synth_ies.0.par.csv / synth_ies.<N>.par.csv       (parameter ensembles)
    synth_ies.0.obs.csv / synth_ies.<N>.obs.csv       (observation ensembles)
    synth_ies.phi.actual.csv                          (phi bookkeeping)
    synth_ies.phi.group.csv

This is purely for testing/demonstration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyemu


def build(out_dir: str, noptmax: int = 4, nreal: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    case = "synth_ies"

    # ---- design parameters across a few groups --------------------
    par_groups = {
        "hk": 12,     # log-transformed hydraulic conductivity
        "sy": 6,      # specific yield (native)
        "rch": 8,     # recharge multiplier (log)
    }
    par_names, par_gp, par_trans = [], [], []
    for gp, n in par_groups.items():
        for i in range(n):
            par_names.append(f"{gp}_{i:03d}")
            par_gp.append(gp)
            par_trans.append("log" if gp in ("hk", "rch") else "none")

    # ---- observations across groups, some with a time axis --------
    obs_specs = {
        "head_shallow": 40,
        "head_deep": 30,
        "flux_river": 20,
    }
    obs_names, obs_gp, obs_val, obs_wt = [], [], [], []
    obs_time, obs_site = [], []
    for gp, n in obs_specs.items():
        base = 50.0 if gp.startswith("head") else 2.0
        nsite = 4
        for i in range(n):
            name = f"{gp}_{i:03d}"
            obs_names.append(name)
            obs_gp.append(gp)
            obs_val.append(base + rng.normal(0, 5 if gp.startswith("head") else 0.5))
            obs_wt.append(1.0)
            site = f"{gp}_site{i % nsite}"
            obs_site.append(site)
            obs_time.append(pd.Timestamp("2015-01-01") +
                            pd.Timedelta(days=30 * (i // nsite)))
    # one forecast (zero-weight prediction)
    obs_names.append("forecast_drawdown")
    obs_gp.append("forecast")
    obs_val.append(3.0)
    obs_wt.append(0.0)
    obs_site.append("forecast")
    obs_time.append(pd.Timestamp("2030-01-01"))

    # ---- assemble a minimal but valid Pst -------------------------
    pst = pyemu.Pst.from_par_obs_names(par_names=par_names, obs_names=obs_names)
    pdata = pst.parameter_data
    pdata.loc[par_names, "pargp"] = par_gp
    pdata.loc[par_names, "partrans"] = par_trans
    for name, tr in zip(par_names, par_trans):
        if tr == "log":
            pdata.loc[name, "parval1"] = 1.0
            pdata.loc[name, "parlbnd"] = 0.01
            pdata.loc[name, "parubnd"] = 100.0
        else:
            pdata.loc[name, "parval1"] = 0.15
            pdata.loc[name, "parlbnd"] = 0.02
            pdata.loc[name, "parubnd"] = 0.35

    odata = pst.observation_data
    odata.loc[obs_names, "obgnme"] = obs_gp
    odata.loc[obs_names, "obsval"] = obs_val
    odata.loc[obs_names, "weight"] = obs_wt
    # carry optional time/site so the timeseries module can auto-detect them
    odata.loc[obs_names, "datetime"] = [str(t.date()) for t in obs_time]
    odata.loc[obs_names, "site"] = obs_site

    pst.control_data.noptmax = noptmax
    pst.pestpp_options["forecasts"] = "forecast_drawdown"
    pst.write(str(out / f"{case}.pst"), version=2)

    # ---- fabricate prior & posterior parameter ensembles ----------
    # prior: broad; posterior: narrower and shifted toward a "truth"
    def par_ensemble(scale, shift):
        data = {}
        for name, gp, tr in zip(par_names, par_gp, par_trans):
            if tr == "log":
                mu = shift.get(gp, 0.0)
                vals = rng.normal(mu, scale, nreal)   # in log10 space
            else:
                mu = 0.15 + shift.get(gp, 0.0) * 0.05
                vals = np.clip(rng.normal(mu, 0.05 * scale, nreal), 0.02, 0.35)
            data[name] = vals
        return pd.DataFrame(data,
                            index=[f"real_{i}" for i in range(nreal)])

    prior_par = par_ensemble(scale=0.8, shift={})
    post_par = par_ensemble(scale=0.25, shift={"hk": 0.3, "rch": -0.2})

    # ---- observation ensembles: posterior tracks obsval closely ---
    def obs_ensemble(spread_frac, bias_frac):
        data = {}
        for name, val, wt in zip(obs_names, obs_val, obs_wt):
            sd = max(abs(val) * spread_frac, 0.5)
            center = val + bias_frac * sd
            data[name] = rng.normal(center, sd, nreal)
        return pd.DataFrame(data,
                            index=[f"real_{i}" for i in range(nreal)])

    prior_obs = obs_ensemble(spread_frac=0.4, bias_frac=0.8)
    post_obs = obs_ensemble(spread_frac=0.08, bias_frac=0.1)

    prior_par.to_csv(out / f"{case}.0.par.csv", index_label="real_name")
    post_par.to_csv(out / f"{case}.{noptmax}.par.csv", index_label="real_name")
    prior_obs.to_csv(out / f"{case}.0.obs.csv", index_label="real_name")
    post_obs.to_csv(out / f"{case}.{noptmax}.obs.csv", index_label="real_name")

    # ---- phi files across iterations ------------------------------
    nz = [n for n, w in zip(obs_names, obs_wt) if w > 0]
    real_cols = [f"real_{i}" for i in range(nreal)]

    def phi_of(ens):
        # simple sum-of-squared weighted residuals per realisation
        r = ens[nz].subtract(pd.Series(dict(zip(obs_names, obs_val)))[nz], axis=1)
        return (r ** 2).sum(axis=1)

    phi_prior = phi_of(prior_obs)
    phi_post = phi_of(post_obs)
    # phi.actual.csv: one row per iteration (summary + per-realisation columns)
    rows_actual = []
    # phi.group.csv: one row per realisation per iteration, matching the real
    # PEST++ layout (iteration,total_runs,obs_realization,par_realization,<grp>)
    rows_group = []
    grp_names = list(obs_specs.keys())
    truth = pd.Series(dict(zip(obs_names, obs_val)))
    for it in range(noptmax + 1):
        frac = it / noptmax
        phi = phi_prior * (1 - frac) + phi_post * frac
        rows_actual.append(
            [it, nreal, phi.mean(), phi.std(), phi.min(), phi.max()]
            + phi.loc[real_cols].tolist())
        ens = prior_obs * (1 - frac) + post_obs * frac
        for rc in real_cols:
            grow = [it, nreal, rc, rc]
            for gp in grp_names:
                gnames = [n for n, g in zip(obs_names, obs_gp)
                          if g == gp and n in nz]
                rr = ens.loc[rc, gnames] - truth[gnames]
                grow.append(float((rr ** 2).sum()))
            rows_group.append(grow)

    cols_meta = ["iteration", "total_runs", "mean", "standard_deviation",
                 "min", "max"]
    pd.DataFrame(rows_actual, columns=cols_meta + real_cols).to_csv(
        out / f"{case}.phi.actual.csv", index=False)
    pd.DataFrame(rows_group,
                 columns=["iteration", "total_runs", "obs_realization",
                          "par_realization"] + grp_names).to_csv(
        out / f"{case}.phi.group.csv", index=False)

    print(f"Synthetic PEST++ IES case written to: {out}/{case}.pst")
    return out / f"{case}.pst"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="synthetic_case")
    ap.add_argument("--noptmax", type=int, default=4)
    ap.add_argument("--nreal", type=int, default=60)
    build(**{"out_dir": ap.parse_args().out,
             "noptmax": ap.parse_args().noptmax,
             "nreal": ap.parse_args().nreal})
