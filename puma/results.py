"""
results.py
==========

Autonomous discovery and loading of PEST++ IES (iterative ensemble smoother)
output files.

The :class:`IesResults` object is the single entry point for the whole
toolbox.  Point it at a ``.pst`` control file (or the directory that holds
one) and it will:

* locate the control file and read the case name / ``noptmax``;
* discover every iteration for which parameter and observation ensembles
  were written (both ``.csv`` and binary ``.jcb``/``.parjcb``/``.obsjcb``
  flavours are handled);
* load the *prior* (iteration 0) and *posterior* (last available iteration)
  ensembles for both parameters and observations;
* load the phi bookkeeping files (``*.phi.actual.csv``,
  ``*.phi.meas.csv``, ``*.phi.group.csv``);
* load optional extras when present: prior-data-conflict (``*.pdc.csv``) and
  the per-iteration parameter-change summaries (``*.pcs.csv``).

Nothing here is model-specific: it relies only on standard PEST++ IES output
conventions, so it works for MODFLOW, Marthe, GSFLOW or any other model that
is driven through PEST++.

The heavy lifting of reading ensembles is delegated to ``pyemu`` so that
parameter transforms declared in the control file are honoured.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pyemu
except ImportError as exc:  # pragma: no cover - pyemu is a hard dependency
    raise ImportError(
        "pyemu is required by puma. Install it with "
        "`pip install pyemu`."
    ) from exc


# Matches ``<case>.<iter>.<kind>`` where kind is one of the ensemble suffixes.
_ENS_RE = re.compile(
    r"^(?P<case>.+)\.(?P<iter>\d+)\.(?P<kind>par|obs)(?P<binary>jcb)?"
    r"(?:\.csv)?$"
)


@dataclass
class IesResults:
    """Container that discovers and lazily loads PEST++ IES results.

    Parameters
    ----------
    pst_file:
        Path to a ``.pst`` control file, or a directory containing exactly
        one ``.pst`` file.
    verbose:
        Print progress messages while discovering / loading.
    """

    pst_file: str
    verbose: bool = True

    # ---- populated during __post_init__ -------------------------------
    pst_path: Path = field(init=False)
    base_dir: Path = field(init=False)
    case: str = field(init=False)
    pst: "pyemu.Pst" = field(init=False)
    noptmax: int = field(init=False)
    available_iters: List[int] = field(init=False, default_factory=list)
    prior_iter: int = field(init=False, default=0)
    posterior_iter: int = field(init=False, default=0)

    # caches
    _cache: Dict[str, object] = field(init=False, default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        p = Path(self.pst_file)
        if p.is_dir():
            candidates = sorted(p.glob("*.pst"))
            if not candidates:
                raise FileNotFoundError(f"No .pst file found in {p}")
            if len(candidates) > 1:
                warnings.warn(
                    f"Multiple .pst files in {p}; using {candidates[0].name}"
                )
            p = candidates[0]
        if not p.exists():
            raise FileNotFoundError(f"Control file not found: {p}")

        self.pst_path = p.resolve()
        self.base_dir = self.pst_path.parent
        self.case = self.pst_path.stem

        self._log(f"Loading control file: {self.pst_path.name}")
        self.pst = pyemu.Pst(str(self.pst_path))
        self.noptmax = int(self.pst.control_data.noptmax)

        self._discover_iterations()
        self.prior_iter = 0
        self.posterior_iter = (
            max(self.available_iters) if self.available_iters else 0
        )
        self._log(
            f"Case '{self.case}': noptmax={self.noptmax}, "
            f"iterations found={self.available_iters}, "
            f"posterior iteration={self.posterior_iter}"
        )

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[puma] {msg}")

    # ------------------------------------------------------------------
    def _discover_iterations(self) -> None:
        """Scan the base directory for ensemble files and record iterations."""
        iters = set()
        for f in self.base_dir.iterdir():
            if not f.is_file():
                continue
            if not f.name.startswith(self.case + "."):
                continue
            m = _ENS_RE.match(f.name)
            if m and m.group("case") == self.case:
                iters.add(int(m.group("iter")))
        self.available_iters = sorted(iters)

    # ------------------------------------------------------------------
    def _find_ensemble_file(self, iteration: int, kind: str) -> Optional[Path]:
        """Return the ensemble path for ``kind`` in {'par','obs'}.

        Prefers binary (``.jcb``) files, falling back to ``.csv``.
        """
        stems = [
            f"{self.case}.{iteration}.{kind}jcb",   # binary
            f"{self.case}.{iteration}.{kind}.jcb",  # occasional variant
            f"{self.case}.{iteration}.{kind}.csv",  # text
        ]
        for s in stems:
            path = self.base_dir / s
            if path.exists():
                return path
        return None

    # ------------------------------------------------------------------
    def _load_ensemble(self, iteration: int, kind: str) -> Optional[pd.DataFrame]:
        """Load a parameter or observation ensemble as a plain DataFrame."""
        cache_key = f"{kind}_{iteration}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        path = self._find_ensemble_file(iteration, kind)
        if path is None:
            self._log(f"  (no {kind} ensemble for iteration {iteration})")
            self._cache[cache_key] = None
            return None

        self._log(f"  loading {kind} ensemble: {path.name}")
        ens_cls = (
            pyemu.ParameterEnsemble
            if kind == "par"
            else pyemu.ObservationEnsemble
        )
        try:
            if path.suffix == ".csv":
                ens = ens_cls.from_csv(pst=self.pst, filename=str(path))
            else:
                ens = ens_cls.from_binary(pst=self.pst, filename=str(path))
            df = ens._df.copy() if hasattr(ens, "_df") else ens.loc[:, :].copy()
        except Exception as exc:  # noqa: BLE001
            self._log(f"  ! failed to load {path.name}: {exc}")
            self._cache[cache_key] = None
            return None

        self._cache[cache_key] = df
        return df

    # ------------------- public ensemble accessors --------------------
    def par_ensemble(self, iteration: int) -> Optional[pd.DataFrame]:
        """Parameter ensemble (realisations x parameters) for ``iteration``.

        Log-transformed parameters are returned in their *native* (back
        transformed) space so plots are physically meaningful.
        """
        df = self._load_ensemble(iteration, "par")
        if df is None:
            return None
        return self._backtransform_pars(df)

    def obs_ensemble(self, iteration: int) -> Optional[pd.DataFrame]:
        """Observation ensemble (realisations x observations)."""
        return self._load_ensemble(iteration, "obs")

    @property
    def prior_par(self) -> Optional[pd.DataFrame]:
        return self.par_ensemble(self.prior_iter)

    @property
    def posterior_par(self) -> Optional[pd.DataFrame]:
        return self.par_ensemble(self.posterior_iter)

    @property
    def prior_obs(self) -> Optional[pd.DataFrame]:
        return self.obs_ensemble(self.prior_iter)

    @property
    def posterior_obs(self) -> Optional[pd.DataFrame]:
        return self.obs_ensemble(self.posterior_iter)

    # ------------------------------------------------------------------
    def _backtransform_pars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Undo log10 transforms declared in the control file."""
        pdata = self.pst.parameter_data
        if "partrans" not in pdata.columns:
            return df
        log_pars = pdata.loc[
            pdata.partrans == "log", "parnme"
        ].tolist()
        log_pars = [p for p in log_pars if p in df.columns]
        if not log_pars:
            return df
        out = df.copy()
        out[log_pars] = 10.0 ** out[log_pars]
        return out

    # ------------------------- phi files ------------------------------
    def phi_actual(self) -> Optional[pd.DataFrame]:
        return self._read_csv_cached(self.pst_path.with_suffix(".phi.actual.csv"))

    def phi_meas(self) -> Optional[pd.DataFrame]:
        return self._read_csv_cached(self.pst_path.with_suffix(".phi.meas.csv"))

    def phi_group(self) -> Optional[pd.DataFrame]:
        return self._read_csv_cached(self.pst_path.with_suffix(".phi.group.csv"))

    def pcs(self, iteration: int) -> Optional[pd.DataFrame]:
        path = self.base_dir / f"{self.case}.{iteration}.pcs.csv"
        return self._read_csv_cached(path)

    def pdc(self) -> Optional[pd.DataFrame]:
        """Prior-data-conflict summary if PEST++ wrote one."""
        for name in (
            f"{self.case}.pdc.csv",
            f"{self.case}.0.pdc.csv",
        ):
            path = self.base_dir / name
            if path.exists():
                return self._read_csv_cached(path)
        return None

    def _read_csv_cached(self, path: Path) -> Optional[pd.DataFrame]:
        key = f"csv::{path}"
        if key in self._cache:
            return self._cache[key]  # type: ignore[return-value]
        if not Path(path).exists():
            self._cache[key] = None
            return None
        df = pd.read_csv(path)
        self._cache[key] = df
        return df

    # --------------------- observation helpers ------------------------
    def nonzero_obs(self) -> pd.DataFrame:
        """Observation-data rows with non-zero weight (the calibration targets)."""
        obs = self.pst.observation_data
        return obs.loc[obs.weight.astype(float) > 0].copy()

    def obs_groups(self, nonzero_only: bool = True) -> List[str]:
        obs = self.nonzero_obs() if nonzero_only else self.pst.observation_data
        return sorted(obs.obgnme.unique().tolist())

    def par_groups(self) -> List[str]:
        pdata = self.pst.parameter_data
        adj = pdata.loc[pdata.partrans != "fixed"]
        return sorted(adj.pargp.unique().tolist())

    def realized_phi(self, iteration: int) -> Optional[pd.Series]:
        """Per-realisation total phi at ``iteration`` from the actual-phi file.

        Returns a Series indexed by realisation name.
        """
        pa = self.phi_actual()
        if pa is None:
            return None
        row = pa.loc[pa.iteration == iteration]
        if row.empty:
            return None
        # columns after the 6 bookkeeping columns are the realisations
        meta = {"iteration", "total_runs", "mean", "standard_deviation",
                "min", "max"}
        real_cols = [c for c in pa.columns if c not in meta]
        return row[real_cols].iloc[0].astype(float)

    # ------------------------------------------------------------------
    def discover_marthe_files(self) -> Dict[str, Optional[str]]:
        """Locate the Marthe model input files sitting next to the ``.pst``.

        A PEST++/Marthe run keeps the model files in the same directory as the
        control file, all sharing the Marthe *model* stem (e.g. ``mrn_v11``),
        which usually differs from the PEST case stem (``mrn_v11_v5``).  The
        model stem is taken from the ``.rma`` project file; the ``.histo``,
        ``.pastp`` and ``.mart`` companions are then just extension swaps.  The
        ``pymarthe`` config is found by its ``.config`` extension.

        Returns a dict with keys ``rma, histo, pastp, mart, config`` mapping to
        absolute paths (as ``str``) or ``None`` when a file is absent.  Values
        are only filled when the file actually exists, so callers can use them
        as defaults directly.
        """
        found: Dict[str, Optional[str]] = {
            "rma": None, "histo": None, "pastp": None,
            "mart": None, "config": None,
        }
        d = self.base_dir
        # model stem from the .rma project file (prefer a unique one)
        rmas = sorted(d.glob("*.rma"))
        model_stem = None
        if len(rmas) == 1:
            found["rma"] = str(rmas[0])
            model_stem = rmas[0].stem
        elif len(rmas) > 1:
            # multiple: prefer one whose stem is a prefix of the case name
            pick = next((r for r in rmas if self.case.startswith(r.stem)),
                        rmas[0])
            found["rma"] = str(pick)
            model_stem = pick.stem

        if model_stem is not None:
            for key, ext in (("histo", ".histo"), ("pastp", ".pastp"),
                             ("mart", ".mart")):
                cand = d / f"{model_stem}{ext}"
                if cand.exists():
                    found[key] = str(cand)
        else:
            # no .rma: still try to find the companions by extension
            for key, ext in (("histo", ".histo"), ("pastp", ".pastp"),
                             ("mart", ".mart")):
                hits = sorted(d.glob(f"*{ext}"))
                if len(hits) == 1:
                    found[key] = str(hits[0])

        configs = sorted(d.glob("*.config"))
        if configs:
            # prefer 'configuration.config' if present, else the first
            pick = next((c for c in configs if "config" in c.stem.lower()),
                        configs[0])
            found["config"] = str(pick)
        return found

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Human-readable inventory of what was discovered."""
        lines = [
            f"Case            : {self.case}",
            f"Control file    : {self.pst_path}",
            f"noptmax         : {self.noptmax}",
            f"Adj. parameters : {self.pst.npar_adj} / {self.pst.npar}",
            f"Non-zero obs    : {self.pst.nnz_obs} / {self.pst.nobs}",
            f"Obs groups (nz) : {len(self.obs_groups())}",
            f"Par groups      : {len(self.par_groups())}",
            f"Iterations      : {self.available_iters}",
            f"Prior iter      : {self.prior_iter}",
            f"Posterior iter  : {self.posterior_iter}",
            f"phi.actual.csv  : {'yes' if self.phi_actual() is not None else 'no'}",
            f"phi.group.csv   : {'yes' if self.phi_group() is not None else 'no'}",
            f"pdc.csv         : {'yes' if self.pdc() is not None else 'no'}",
        ]
        forecasts = list(self.pst.forecast_names) if self.pst.forecast_names is not None else []
        lines.append(f"Forecasts       : {len(forecasts)}")
        return "\n".join(lines)
