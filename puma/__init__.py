"""
puma
===============

An autonomous post-processing toolbox for PEST++ IES (iterative ensemble
smoother) runs.  It discovers the standard output files, loads the prior and
posterior ensembles, and produces a full suite of diagnostic figures for
judging calibration efficacy and uncertainty analysis - with no model-specific
configuration required.

Quick start
-----------
>>> from puma import run_full_report
>>> run_full_report("path/to/case.pst", output_dir="PLOTS")

Or drive individual pieces:
>>> from puma import IesResults, plots
>>> res = IesResults("case.pst")
>>> print(res.summary())
>>> plots.plot_one_to_one(res, "PLOTS")
"""

__version__ = "0.1.1"

from .results import IesResults
from .report import run_full_report
from . import plots
from . import timeseries
from . import utils
from . import marthe
from . import spatial

__all__ = [
    "IesResults",
    "run_full_report",
    "plots",
    "timeseries",
    "utils",
    "marthe",
    "spatial",
    "__version__",
]
