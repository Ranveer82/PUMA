"""
marthe.py
=========

Marthe-model helpers for reconstructing the **datetime index** of transient
observations, which PEST++ IES output files do not carry.

Why this is needed
------------------
The observation ensembles written by PEST++ (`*.obs.csv` / `*.obsjcb`) store
one column per observation but no time stamp.  For a Marthe model the simulated
history of a given site (a HISTO location, i.e. one ``obgnme``) is written only
at the model time steps where the hydrodynamic solver actually runs.  To place
those simulated values on a real calendar axis we reconstruct the sequence of
"computed" dates directly from the model definition files, following exactly
the rules Marthe uses:

1. **``.mart`` file** - read the transient time-step budget
   (*"Nombre maximal possible de 'pas de temps de modele' en regime
   Transitoire"*, ``0`` => all) and the warm-up count to ignore
   (*"Nombre de pas de temps de 'demarrage' a ignorer ..."*).
2. **``.pastp`` file** - read the simulation start date, every model time-step
   date (*"Le pas : N: se termine a la date : DD/MM/YYYY"*), and the
   hydrodynamic-control file referenced by ``/CALCUL_HDYNAM/ACTION``.
3. **``/CALCUL_HDYNAM/ACTION`` file** - a two-column ``flag  date`` table
   where ``0`` marks a skipped time step and ``1``/``2`` a computed one.  Only
   computed steps produce simulated output.  The **first time step**
   (the simulation start date, ``I=2`` in the ``.pastp``) is always added by
   default even though it is absent from this table.

The reconstructed ordered list of computed datetimes is then assigned, in
chronological order, to the observations of each transient group so the
generic time-series plotter can draw them on a date axis.

This module parses the files directly (latin-1, as Marthe writes them) so it
carries no dependency on ``pymarthe``; the same metadata ``pymarthe`` exposes is
reproduced here.  If ``pymarthe`` is installed it can be substituted for the
parsing without changing the rest of the toolbox.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .results import IesResults


_ENC = "latin-1"

# --- regexes (kept ASCII-only so accented bytes never break matching) ------
_RE_START = re.compile(r"date\s*:\s*(\d{2}/\d{2}/\d{4})")
_RE_PAS = re.compile(r"Le pas\s*:\s*(\d+)\s*:.*?(\d{2}/\d{2}/\d{4})")
_RE_ACTION = re.compile(
    r"CALCUL_HDYNAM/ACTION\s+I=\s*(\d+)\s*;\s*File\s*=\s*(\S+)")
_RE_STARTUP = re.compile(
    r"Nombre de pas de temps.*?d.marrage.*?ignorer", re.IGNORECASE)
_RE_MAXTS = re.compile(
    r"Nombre maximal possible de.*?pas de temps.*?Transitoire", re.IGNORECASE)


# ======================================================================
def read_mart_options(mart_file: str) -> Dict[str, int]:
    """Return ``{'n_max_timesteps': int, 'n_startup_ignore': int}``.

    ``n_max_timesteps == 0`` means all transient time steps are used.
    """
    opts = {"n_max_timesteps": 0, "n_startup_ignore": 0}
    path = Path(mart_file)
    if not path.exists():
        return opts
    with open(path, "r", encoding=_ENC) as fh:
        for line in fh:
            # value sits before the first '=' on each Marthe option line
            val_str = line.split("=", 1)[0].strip()
            try:
                val = int(float(val_str))
            except (ValueError, TypeError):
                continue
            if _RE_MAXTS.search(line):
                opts["n_max_timesteps"] = val
            elif _RE_STARTUP.search(line):
                opts["n_startup_ignore"] = val
    return opts


# ======================================================================
def parse_pastp(pastp_file: str,
                collect_all_steps: bool = False
                ) -> Dict[str, object]:
    """Parse a ``.pastp`` file.

    Returns a dict with:
        ``start_date``  : Timestamp of the simulation start (first time step),
        ``action_file`` : filename referenced by /CALCUL_HDYNAM/ACTION (or None),
        ``action_mode`` : the ``I=`` integer on that line (or None),
        ``step_dates``  : list[Timestamp] of every model time step
                          (only populated when ``collect_all_steps`` is True).
    """
    path = Path(pastp_file)
    start_date: Optional[pd.Timestamp] = None
    action_file: Optional[str] = None
    action_mode: Optional[int] = None
    step_dates: List[pd.Timestamp] = []

    with open(path, "r", encoding=_ENC) as fh:
        for line in fh:
            if start_date is None and "simulation" in line.lower():
                m = _RE_START.search(line)
                if m:
                    start_date = pd.to_datetime(m.group(1), dayfirst=True)
                continue
            if action_file is None and "CALCUL_HDYNAM/ACTION" in line:
                m = _RE_ACTION.search(line)
                if m:
                    action_mode = int(m.group(1))
                    action_file = m.group(2).strip()
                continue
            if collect_all_steps and "Le pas" in line:
                m = _RE_PAS.search(line)
                if m:
                    step_dates.append(
                        pd.to_datetime(m.group(2), dayfirst=True))
    return {
        "start_date": start_date,
        "action_file": action_file,
        "action_mode": action_mode,
        "step_dates": step_dates,
    }


# ======================================================================
def parse_calcul_hydro(txt_file: str) -> pd.DataFrame:
    """Parse a ``/CALCUL_HDYNAM/ACTION`` control file.

    Returns a DataFrame with integer ``flag`` and datetime ``date`` columns.
    Comment lines (starting with ``!``) and blanks are ignored.
    """
    rows = []
    with open(txt_file, "r", encoding=_ENC) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                flag = int(float(parts[0]))
            except ValueError:
                continue
            date = pd.to_datetime(parts[1], dayfirst=True, errors="coerce")
            if pd.isna(date):
                continue
            rows.append((flag, date))
    return pd.DataFrame(rows, columns=["flag", "date"])


# ======================================================================
def build_computed_datetimes(pastp_file: str,
                             mart_file: Optional[str] = None,
                             computed_flags=(1, 2)) -> pd.DatetimeIndex:
    """Reconstruct the ordered datetimes at which simulated output exists.

    Implements the three checks: the ``.pastp`` start date is always the first
    computed step; the ``/CALCUL_HDYNAM/ACTION`` file selects the remaining
    computed steps (flag in ``computed_flags``); if no action file is present
    every model time step is computed.  The ``.mart`` budget then truncates the
    tail (``n_max_timesteps``) and drops the warm-up head (``n_startup_ignore``).
    """
    pastp_path = Path(pastp_file)
    info = parse_pastp(str(pastp_path), collect_all_steps=False)
    start_date = info["start_date"]
    action_file = info["action_file"]

    dates: List[pd.Timestamp] = []
    if start_date is not None:
        dates.append(start_date)  # first time step always included

    if action_file:
        action_path = pastp_path.parent / action_file
        if not action_path.exists():
            # allow the caller to have placed it beside the pastp under any case
            hits = list(pastp_path.parent.glob(Path(action_file).name))
            action_path = hits[0] if hits else action_path
        if action_path.exists():
            df = parse_calcul_hydro(str(action_path))
            computed = df.loc[df["flag"].isin(computed_flags), "date"]
            dates.extend(computed.tolist())
        else:
            print(f"[ies-post] Marthe: action file '{action_file}' not found "
                  f"next to the .pastp; falling back to all model steps")
            info = parse_pastp(str(pastp_path), collect_all_steps=True)
            dates = ([start_date] if start_date is not None else []) \
                + info["step_dates"]
    else:
        info = parse_pastp(str(pastp_path), collect_all_steps=True)
        dates = ([start_date] if start_date is not None else []) \
            + info["step_dates"]

    idx = pd.DatetimeIndex(dates).sort_values()

    # apply .mart budget
    if mart_file:
        opts = read_mart_options(mart_file)
        k = opts.get("n_startup_ignore", 0)
        if k > 0:
            idx = idx[k:]
        nmax = opts.get("n_max_timesteps", 0)
        if nmax and nmax > 0:
            idx = idx[:nmax]
    return idx


# ======================================================================
def build_obs_meta(res: IesResults,
                   pastp_file: str,
                   mart_file: Optional[str] = None,
                   nonzero_only: bool = False) -> Optional[pd.DataFrame]:
    """Build an ``obsnme, site, datetime`` table from the Marthe model files.

    The reconstructed computed datetimes are assigned, in chronological order,
    to the observations of each transient group (one Marthe HISTO site =
    one ``obgnme``).  Groups whose observation count matches the datetime
    sequence are aligned directly; a shorter group is aligned to the **tail**
    of the sequence (interpreting the missing steps as dropped warm-up), and a
    longer group is skipped with a warning.

    Returns ``None`` if the datetimes cannot be reconstructed.
    """
    dt = build_computed_datetimes(pastp_file, mart_file=mart_file)
    if len(dt) == 0:
        print("[ies-post] Marthe: no computed datetimes reconstructed")
        return None
    print(f"[ies-post] Marthe: reconstructed {len(dt)} computed datetimes "
          f"({dt[0].date()} -> {dt[-1].date()})")

    obs = res.pst.observation_data
    if nonzero_only:
        obs = obs.loc[pd.to_numeric(obs.weight, errors="coerce") > 0]

    records = []
    n_ok, n_skip = 0, 0
    for grp, sub in obs.groupby("obgnme"):
        names = list(sub.index)  # preserves control-file order
        n = len(names)
        if n < 2:
            continue
        if n == len(dt):
            grp_dt = dt
        elif n < len(dt):
            grp_dt = dt[len(dt) - n:]        # align to the tail (warm-up dropped)
        else:
            n_skip += 1
            print(f"[ies-post] Marthe: group '{grp}' has {n} obs > "
                  f"{len(dt)} datetimes; skipping")
            continue
        for name, d in zip(names, grp_dt):
            records.append((name, grp, d))
        n_ok += 1

    if not records:
        return None
    print(f"[ies-post] Marthe: dated {n_ok} group(s), skipped {n_skip}")
    return pd.DataFrame(records, columns=["obsnme", "site", "datetime"])
