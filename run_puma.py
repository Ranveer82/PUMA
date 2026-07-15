#!/usr/bin/env python3
"""
run_puma.py
===============

Thin wrapper kept for backwards compatibility.  The command-line interface now
lives in :mod:`pestpp_ies_post.cli`; once the package is installed you can use
the ``pestpp-ies-post`` console script or ``python -m pestpp_ies_post`` instead.

    python run_ies_post.py case.pst --iteration 3 --histo model.histo
"""

from puma.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
