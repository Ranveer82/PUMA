"""
cli.py
======

Command-line interface for the ``puma`` toolbox.  Installed as the
``pestpp-ies-post`` console script (see ``pyproject.toml``); also runnable as
``python -m puma``.

Point it at a ``.pst`` file (or a directory containing one) and it produces the
full diagnostic figure suite.  Flexible inputs let you pick the iteration to
analyse, the layer for spatial field maps, the property to reconstruct, and so
on.

Examples
--------
    # analyse a single case (all defaults)
    pestpp-ies-post case.pst

    # inspect an earlier iteration as the "posterior"
    pestpp-ies-post case.pst --iteration 3

    # group obs fit plots by observation type, spatial maps + field stats
    pestpp-ies-post case.pst --histo model.histo --marthe-rma model.rma \\
        --marthe-config configuration.config --field-prop permh --field-layer 2

    # just print what was discovered and exit
    pestpp-ies-post case.pst --summary-only
"""

from __future__ import annotations

import argparse
import sys

from .results import IesResults
from .report import run_full_report
from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pestpp-ies-post",
        description="Autonomous post-processing of a PEST++ IES run.")
    p.add_argument("pst", help="path to .pst control file or its directory")
    p.add_argument("-o", "--output", default="PLOTS_PESTPP_IES",
                   help="output directory for figures "
                        "(default: PLOTS_PESTPP_IES)")
    p.add_argument("--version", action="version",
                   version=f"puma {__version__}")

    g_it = p.add_argument_group("iteration selection")
    g_it.add_argument("--iteration", type=int, default=None,
                      help="IES iteration to treat as the posterior "
                           "(default: last available)")
    g_it.add_argument("--prior-iteration", type=int, default=None,
                      help="iteration to treat as the prior baseline "
                           "(default: 0)")

    g_ts = p.add_argument_group("time series")
    g_ts.add_argument("--obs-meta", default=None,
                      help="CSV with columns obsnme,site,datetime for "
                           "per-site time-series plots")
    g_ts.add_argument("--marthe-pastp", default=None,
                      help="Marthe .pastp file; reconstructs transient obs "
                           "datetimes from the model definition")
    g_ts.add_argument("--marthe-mart", default=None,
                      help="matching Marthe .mart file (time-step budget)")
    g_ts.add_argument("--no-timeseries", action="store_true",
                      help="skip per-site time-series figures")

    g_sp = p.add_argument_group("spatial (Marthe / pymarthe)")
    g_sp.add_argument("--histo", default=None,
                      help="Marthe .histo file; groups obs fit plots by "
                           "observation type and draws spatial obs maps")
    g_sp.add_argument("--pp-file", default=None,
                      help="pilot-point file (name x y zone parval) for "
                           "pilot-point uncertainty maps")
    g_sp.add_argument("--shapefile", action="append", default=None,
                      dest="shapefiles",
                      help="boundary shapefile to overlay (repeatable; "
                           "needs geopandas)")
    g_sp.add_argument("--marthe-rma", default=None,
                      help="Marthe .rma project file; renders the base "
                           "property field on the grid (needs pymarthe)")
    g_sp.add_argument("--marthe-config", default=None,
                      help="pymarthe config; reconstructs a per-cell posterior "
                           "field ensemble (mixed pilot-point + ZPC) and maps "
                           "its mean/std/CV on the grid")
    g_sp.add_argument("--field-prop", default=None,
                      help="property for the field figures "
                           "(default: first grid property in the config)")
    g_sp.add_argument("--field-layer", type=int, default=0,
                      help="layer for the field figures (default 0)")
    g_sp.add_argument("--field-max-reals", type=int, default=None,
                      help="cap on realisations reconstructed for field stats")

    p.add_argument("--summary-only", action="store_true",
                   help="print the discovered inventory and exit")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="reduce logging")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.summary_only:
        res = IesResults(args.pst, verbose=not args.quiet)
        print(res.summary())
        return 0

    written = run_full_report(
        pst_file=args.pst,
        output_dir=args.output,
        iteration=args.iteration,
        prior_iteration=args.prior_iteration,
        obs_meta_csv=args.obs_meta,
        marthe_pastp=args.marthe_pastp,
        marthe_mart=args.marthe_mart,
        histo_file=args.histo,
        pp_file=args.pp_file,
        shapefiles=args.shapefiles,
        marthe_rma=args.marthe_rma,
        marthe_config=args.marthe_config,
        field_prop=args.field_prop,
        field_layer=args.field_layer,
        field_max_reals=args.field_max_reals,
        make_timeseries=not args.no_timeseries,
        verbose=not args.quiet,
    )
    if not written:
        print("No figures were produced - check that the ensemble/phi files "
              "exist next to the .pst.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
