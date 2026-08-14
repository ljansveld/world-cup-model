"""
Dixon-Coles international football model with a cross-confederation scale correction.

Typical use:

    from model.matchup import fit_model, matchup

    dc = fit_model()                      # fit + de-bias, once
    r = matchup(dc, "Spain", "France")    # xG, W/D/L, scoreline grid, totals
"""

from model.paths import DATA_DIR, REPO_ROOT, data_path

__all__ = ["DATA_DIR", "REPO_ROOT", "data_path"]
