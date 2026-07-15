# PUMA — **P**ost-processing for **U**ncertainty analysis and **M**ARTHE c**A**libration
## Autonomous post-processing for PEST++ IES

A self-contained Python toolbox for diagnosing the **efficacy of calibration**
and the **quality of uncertainty analysis** from a
[PEST++](https://github.com/usgs/pestpp) **IES** (iterative ensemble smoother)
run.

Point it at a `.pst` control file and it discovers the standard PEST++ IES
output files, loads the prior and posterior ensembles, and produces a complete
suite of publication-ready diagnostic figures — **no model-specific
configuration required**. It works for MODFLOW, Marthe, GSFLOW, or any model
driven through PEST++.

---

## Why this toolbox

- **Autonomous** — auto-detects the case name, `noptmax`, which iterations
  have ensembles, and whether files are text (`.csv`) or binary (`.jcb`).
- **Robust** — every figure degrades gracefully; a missing file or a failed
  panel never aborts the run.
- **Correct by construction** — log/native axes come from the control file's
  `partrans`, not from guessing at names; parameter ensembles are
  back-transformed to native units before plotting.
- **Colour-blind-safe, consistent styling** across every figure.

---

## Installation

Install the package (and its console script) directly from GitHub:

```bash
pip install git+https://github.com/Ranveer82/PUMA.git
```

or from a local clone:

```bash
git clone https://github.com/Ranveer82/PUMA.git
cd PUMA
pip install .            # or  pip install -e .   for a live/editable install
```

Core dependencies (`pyemu`, `pandas`, `numpy`, `matplotlib`, `scipy`) install
automatically. Optional extras:

```bash
pip install "pestpp_ies_post[spatial]"   # geopandas, for shapefile overlays
# pymarthe (Marthe field rendering) is installed separately from GitHub:
pip install git+https://github.com/pymartheproject/pymarthe.git
```

Installing adds a **`pestpp-ies-post`** console command; the toolbox also runs
as `python -m pestpp_ies_post`.

---

## Quick start

### Command line (fully autonomous)

```bash
# analyse a single case (console script installed with the package)
pestpp-ies-post path/to/case.pst

# equivalently, without installing:
python -m pestpp_ies_post path/to/case.pst
python run_ies_post.py path/to/case.pst          # legacy wrapper

# let it find the .pst inside a directory, custom output folder
pestpp-ies-post /runs/model_ies -o my_plots

# analyse an earlier iteration as the posterior; pick a prior baseline
pestpp-ies-post case.pst --iteration 3 --prior-iteration 0

# Marthe: obs datetimes from the model, group fit plots by obs type,
# and reconstruct per-cell posterior fields for layer 2
pestpp-ies-post case.pst --marthe-pastp mrn_v11.pastp --marthe-mart mrn_v11.mart \
    --histo mrn_v11.histo --marthe-config configuration.config --field-layer 2

# just print what was discovered and exit
pestpp-ies-post case.pst --summary-only
```

Run `pestpp-ies-post --help` for the full flag list (grouped into iteration
selection, time series, and spatial options).

### Python API

```python
from pestpp_ies_post import run_full_report

# choose which iteration is the posterior, and the layer for field maps
run_full_report("case.pst", output_dir="PLOTS",
                iteration=3, field_layer=2,
                histo_file="model.histo", marthe_config="configuration.config")
```

Or drive individual figures — every plot takes flexible inputs (iteration,
layer, credible interval, grouping):

```python
from pestpp_ies_post import IesResults, plots, spatial
res = IesResults("case.pst")
print(res.summary())                       # inventory of discovered files

# 1:1 fit at iteration 2, grouped by observation type, with a 10-90% band
plots.plot_one_to_one(res, "PLOTS", iteration=2, ci=(0.1, 0.9),
                      histo_file="model.histo")

# posterior K-field mean/std/CV on the grid for layer 2, iteration 4
spatial.plot_posterior_field_stats(res, "PLOTS", "configuration.config",
                                   iteration=4, layer=2, field_prop="permh")
```

Every plotting function accepts `iteration` (and most `prior_iteration`); the
spatial field/obs functions also accept `layer`. Omit them to use the defaults
(last iteration as posterior, iteration 0 as prior).

---

## Figure catalogue

| # | Figure | What it tells you |
|---|--------|-------------------|
| 01 | **Phi convergence** | Mean Φ vs iteration with ensemble min–max and inter-quartile band — did the smoother converge, and how far did Φ fall? |
| 02 | **Phi distribution** | Violin/box of per-realisation Φ, prior vs posterior — did the *whole ensemble* of objective values collapse? |
| 03 | **Phi by group** | Objective-function decomposition by observation group — *which* data types were (not) fit. |
| 04 | **1:1 with uncertainty** | **Measured vs simulated, per observation group**, markers = posterior ensemble mean, error bars = posterior credible interval, with RMSE / NSE / R² / bias / PBIAS and CI coverage per group. *The central calibration-efficacy figure.* |
| 05 | **Residual histograms** | Residual distributions prior vs posterior by group — bias and spread reduction. |
| 06 | **Residual vs simulated** | Conditional bias / heteroscedasticity check. |
| 07 | **Parameter distributions** | Prior vs posterior per parameter group (log/native axis from `partrans`), annotated with uncertainty reduction. |
| 08 | **Parameter uncertainty reduction** | `1 − σ_post/σ_prior` per group — how strongly the data informed each parameter. |
| 09 | **Forecast uncertainty** | Prior vs posterior predictive PDFs for declared `++forecasts`. |
| 10 | **Ensemble coverage** | Reliability diagram: nominal vs empirical coverage — is the posterior over- or under-confident? |
| 11 | **Prior-data conflict** | Fraction of observations each group cannot reproduce with the prior (from `*.pdc.csv` or derived from the prior ensemble). |
| TS | **Time-series fits** | Per-site posterior ensemble band + measured points on a real calendar axis (dates from an `obsnme,site,datetime` lookup, control-file time columns, or reconstructed from Marthe `.pastp`/`.mart` — see below). |

---

## Spatial accuracy & uncertainty (Marthe / pymarthe)

Beyond the aspatial diagnostics, the toolbox can map *where* the calibration is
good and *where* the posterior is still uncertain (module
`pestpp_ies_post.spatial`). Three tiers, by what you can supply:

**1. Observation-location maps** — need only site coordinates from a Marthe
`.histo` file (parsed natively — projected `/XCOO`, grid-cell `/MAIL`, and
river `/AFFL` formats; `/MAIL` cell references are converted to x,y via the
model grid when `--marthe-rma` is given). Three maps, coloured per site:
- posterior **RMSE** (accuracy),
- posterior ensemble **std** (predictive uncertainty),
- **CI coverage** of measurements (reliability).

```bash
python run_ies_post.py case.pst --histo mrn_v11.histo --shapefile domain.shp
```

**2. Pilot-point / parameter maps** — need a pilot-point file
(`name x y zone parval`). Three maps: posterior **mean** (the calibrated
field), posterior **std**, and **uncertainty reduction**
`1 − σ_post/σ_prior` — showing which locations the data actually constrained.

```bash
python run_ies_post.py case.pst --pp-file hkpp.dat
```

**3. Property field on the model grid (`pymarthe`)** — with `pymarthe`
installed and the model files present, the adjusted property (e.g. `permh`) is
rendered on the true Marthe grid via `MartheField.plot`, with pilot points
overlaid (size = posterior std, colour = uncertainty reduction):

```bash
python run_ies_post.py case.pst --marthe-rma mrn_v1.rma --field-prop permh \
    --field-layer 2 --pp-file hkpp.dat
```

**4. Cell-by-cell posterior field statistics (mixed pilot points + ZPC)** —
`spatial.plot_posterior_field_stats(res, configfile, prop="permh", layer=2)`
reconstructs a *per-cell* posterior ensemble of the property field and maps its
**mean**, **std** and **CV** on the true grid:

```bash
python run_ies_post.py case.pst --marthe-config configuration.config \
    --field-prop permh --field-layer 2 --field-max-reals 50
```

It handles mixed pilot-point + ZPC parameterisations natively: for each
realisation the parameter values are written through the templates and the
property is rebuilt with pymarthe's `MartheField.set_data_from_parfile` +
`izone` — kriging the pilot points (via their `.fac` factors) and filling the
ZPC zones, exactly as the forward run does — so nothing about the pp/ZPC split
is hard-coded. The config is read directly (version-tolerant) and Windows paths
are normalised, so a config written on Windows reconstructs on Linux/macOS.
This needs the pymarthe **config file** and the model setup it references
(`.rma` + grid geometry, `izone` fields, `.fac` kriging factors, base property
files and templates).

*Validated end-to-end on a real nested Marthe IES model with a mixed
pilot-point + ZPC `permh` parameterisation.*

`pymarthe` and `geopandas` are **optional** — imported only inside the
functions that use them, so the rest of the toolbox runs without either.
Install `pymarthe` from source:
`pip install git+https://github.com/pymartheproject/pymarthe.git`.

## Marthe models: reconstructing transient observation datetimes

PEST++ IES ensembles carry **no time stamp** for each observation, so plotting
a transient series by `obgnme` needs the dates supplied separately. For a
**Marthe** model these can be reconstructed directly from the model definition
(module `pestpp_ies_post.marthe`), following Marthe's own rules:

1. **`.mart`** — the transient time-step budget
   (*"Nombre maximal possible de pas de temps … Transitoire"*, `0` = all) and
   the warm-up count to ignore (*"Nombre de pas de temps de démarrage à
   ignorer …"*).
2. **`.pastp`** — the simulation start date, the model time-step dates
   (*"Le pas : N: se termine à la date …"*), and the hydrodynamic-control file
   named by `/CALCUL_HDYNAM/ACTION`.
3. **`/CALCUL_HDYNAM/ACTION` file** — a `flag date` table where `0` = skipped
   step and `1`/`2` = computed step. Only computed steps produce simulated
   output; the **first time step** (the start date) is always added by default.

The ordered list of computed datetimes is then assigned chronologically to each
transient group's observations. Use it via:

```python
from pestpp_ies_post import IesResults, marthe, timeseries
res  = IesResults("mrn_ies.pst")
meta = marthe.build_obs_meta(res, "mrn_v11.pastp", mart_file="mrn_v11.mart")
timeseries.plot_obs_timeseries(res, "PLOTS/ts", obs_meta=meta)
```

or simply pass `--marthe-pastp` / `--marthe-mart` to `run_ies_post.py`. Parsing
is done directly (no `pymarthe` dependency); if `pymarthe` is available it can
be substituted without changing the rest of the toolbox.

## Large models (many observation groups)

For models with many observation groups (hundreds of per-site groups is common
in regional Marthe/MODFLOW calibrations), the per-group figures adapt
automatically so nothing tries to draw hundreds of panels:

* **Group the fit plots by observation type.** Pass the Marthe `--histo` file
  and the **1:1**, **residual histogram** and **residual-vs-simulated** figures
  are panelled by *observation type* (Charge, Débit_Rivi, Hauteu_Rivi, …) read
  from the `.histo`, with correct per-type units (m vs m³/s) — a handful of
  physically meaningful panels instead of one per site. This is the
  recommended way to read a regional model.
* Without a `.histo` those figures use the observation groups, collapsing to a
  single combined panel above ~12 groups.
* **phi-by-group** and **prior-data-conflict** show the top ~30 groups by
  posterior objective-function contribution / conflict, with the total noted in
  the title.

The field-reconstruction tier runs `pymarthe` once per realisation, so cap it
with `--field-max-reals` on large ensembles. Nothing here re-runs Marthe: the
observation and parameter ensembles produced by PEST++ are used directly, and
the spatial field maps only apply the posterior parameters to the grid
(kriging + zone-fill), not a model run.

## Input files it looks for

Placed next to `<case>.pst` following standard PEST++ IES conventions:

- `<case>.phi.actual.csv`, `<case>.phi.meas.csv`, `<case>.phi.group.csv`
- `<case>.0.par.csv` / `.parjcb` … `<case>.<N>.par.csv` / `.parjcb`
- `<case>.0.obs.csv` / `.obsjcb` … `<case>.<N>.obs.csv` / `.obsjcb`
- `<case>.pdc.csv` *(optional)*
- `<case>.<N>.pcs.csv` *(optional)*

Iteration `0` is treated as the **prior**; the highest iteration found is the
**posterior**. Both `.csv` (text) and `.jcb`/`.parjcb`/`.obsjcb` (binary)
ensemble formats are handled automatically.

---

## Trying it without a real run

A synthetic PEST++ IES case generator is included for testing/demonstration:

```bash
python examples/make_synthetic_ies.py -o synthetic_case
python run_ies_post.py synthetic_case/synth_ies.pst -o synthetic_plots
```

This writes a valid `.pst`, prior/posterior parameter and observation
ensembles, and phi files, then produces the full figure suite.

---

## Package layout

```
pestpp_ies_post/
    __init__.py        package API + __version__
    results.py         IesResults — autonomous discovery & loading
    plots.py           all diagnostic figure functions
    timeseries.py      optional per-site time-series plots
    marthe.py          reconstruct transient obs datetimes from Marthe files
    spatial.py         spatial accuracy/uncertainty maps (+ optional pymarthe)
    utils.py           styling + goodness-of-fit metrics
    report.py          run_full_report — the orchestrator
    cli.py             command-line interface (pestpp-ies-post)
    __main__.py        enables `python -m pestpp_ies_post`
pyproject.toml         packaging / console-script entry point
run_ies_post.py        legacy CLI wrapper (delegates to cli.py)
examples/
    make_synthetic_ies.py   synthetic test-case generator
```

## Extending

Each plot is a standalone function `plot_xxx(res: IesResults, output_dir, ...)`
returning the saved path. Add a new one and register it in
`report._PIPELINE` to include it in the autonomous run.
