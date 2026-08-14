"""
validation/debias_diagnostic.py — isolate what squad_debias does to Egypt's defence
and how it moves Argentina's xG.

Run from the repo root:
    python validation/debias_diagnostic.py
    python validation/debias_diagnostic.py "Argentina" "Egypt"     # or any pair
"""

# make the repo root importable when this file is run directly
# (python validation/foo.py) as well as from the root (python -m validation.foo)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import numpy as np
import pandas as pd

from model.dixon_coles import DixonColesModel, load_results
from model.squad_ratings import SquadRatings
from model.squad_debias import debias_coefficients, _fit_line
from model.squad_strength import compute_squad_strength

# --- mirror model/matchup.py's fit settings exactly --------------------------------
XI, LOOKBACK_YEARS, MIN_MATCHES = 0.0019, 4, 5
SPLIT_HOME = True
WEIGHT = 0.7

TEAM_A = sys.argv[1] if len(sys.argv) > 2 else "Argentina"
TEAM_B = sys.argv[2] if len(sys.argv) > 2 else "Egypt"


def fit_raw() -> DixonColesModel:
    """fit_model() from model/matchup.py, minus the debias step."""
    df = load_results()
    ref = df.date.max()
    cutoff = ref - pd.DateOffset(years=LOOKBACK_YEARS)
    sub = df[(df.date >= cutoff) & (df.date <= ref)].copy()
    counts = pd.concat([sub.home_team, sub.away_team]).value_counts()
    valid = set(counts[counts >= MIN_MATCHES].index)
    sub = sub[sub.home_team.isin(valid) & sub.away_team.isin(valid)].reset_index(drop=True)
    dc = DixonColesModel(xi=XI)
    dc.fit(sub, ref_date=ref, split_home=SPLIT_HOME)
    return dc


def neutral_xg(dc, a, b):
    """xG at a neutral venue: lambda = exp(att + opp_def)."""
    return (float(np.exp(dc.attack[a] + dc.defence[b])),
            float(np.exp(dc.attack[b] + dc.defence[a])))


def world_rank(coefs: dict, team: str, better_is_lower: bool = True):
    """1 = best. Defence: more negative = better (concedes less)."""
    vals = sorted(coefs.items(), key=lambda kv: kv[1], reverse=not better_is_lower)
    return [t for t, _ in vals].index(team) + 1, len(vals)


def main():
    dc = fit_raw()
    for t in (TEAM_A, TEAM_B):
        if t not in dc.attack:
            sys.exit(f"'{t}' not in the fitted model — check name normalization.")

    # ---- BEFORE ----
    att_raw = {t: v for t, v in dc.attack.items()}
    def_raw = {t: v for t, v in dc.defence.items()}
    xg_a0, xg_b0 = neutral_xg(dc, TEAM_A, TEAM_B)

    # squad-implied line for TEAM_B's defence (recomputed the same way the module does)
    squad = SquadRatings()
    S = compute_squad_strength(pd.read_csv(squad.source, low_memory=False))
    att_sq = S["squad_attack"].dropna().to_dict()
    def_sq = S["squad_defense"].dropna().to_dict()
    teams = [t for t in dc.attack if t in att_sq and t in def_sq]
    d1, d0, dr2 = _fit_line([def_sq[t] for t in teams], [def_raw[t] for t in teams])

    print("=" * 72)
    print(f"BEFORE de-bias   ({TEAM_A} vs {TEAM_B}, neutral venue)")
    print("=" * 72)
    for t in (TEAM_A, TEAM_B):
        dr, dn = world_rank(def_raw, t)
        ar, an = world_rank(att_raw, t, better_is_lower=False)
        line = ""
        if t in def_sq:
            hat = d1 * def_sq[t] + d0
            line = f"   squad-implied def={hat:+.4f}  resid={def_raw[t]-hat:+.4f}"
        print(f"  {t:<12} att={att_raw[t]:+.4f} (rank {ar}/{an})   "
              f"def={def_raw[t]:+.4f} (rank {dr}/{dn}){line}")
    print(f"  xG: {TEAM_A} {xg_a0:.3f} — {TEAM_B} {xg_b0:.3f}")
    print(f"  (def~squad line R^2 = {dr2:.2f}; more negative def = better defence)")

    # ---- APPLY (verbose shows the top-6 corrections line you normally see) ----
    print()
    diag = debias_coefficients(dc, squad.source, weight=WEIGHT, verbose=True)
    print()

    # ---- AFTER ----
    xg_a1, xg_b1 = neutral_xg(dc, TEAM_A, TEAM_B)
    print("=" * 72)
    print(f"AFTER de-bias (w={WEIGHT})")
    print("=" * 72)
    for t in (TEAM_A, TEAM_B):
        dr, dn = world_rank(dc.defence, t)
        ar, an = world_rank(dc.attack, t, better_is_lower=False)
        print(f"  {t:<12} att={dc.attack[t]:+.4f} (rank {ar}/{an})  Δatt={dc.attack[t]-att_raw[t]:+.4f}   "
              f"def={dc.defence[t]:+.4f} (rank {dr}/{dn})  Δdef={dc.defence[t]-def_raw[t]:+.4f}")
    print(f"  xG: {TEAM_A} {xg_a1:.3f} — {TEAM_B} {xg_b1:.3f}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  {TEAM_A} xG moved {xg_a0:.3f} -> {xg_a1:.3f}  ({(xg_a1/xg_a0-1)*+100:+.1f}%)")
    print(f"  {TEAM_B} xG moved {xg_b0:.3f} -> {xg_b1:.3f}  ({(xg_b1/xg_b0-1)*+100:+.1f}%)")
    dd = dc.defence[TEAM_B] - def_raw[TEAM_B]
    print(f"  {TEAM_B} defence coefficient shifted {dd:+.4f} "
          f"(multiplies {TEAM_A}'s lambda by e^{dd:+.4f} = x{np.exp(dd):.3f})")
    if dd > 0.02:
        print(f"  -> de-bias DOWNGRADED {TEAM_B}'s defence; this is (at least part of) "
              f"why {TEAM_A}'s model xG sits above the market.")
    elif dd < -0.02:
        print(f"  -> de-bias UPGRADED {TEAM_B}'s defence; the gap to market is coming "
              f"from somewhere else (attack side, or the fit itself).")
    else:
        print(f"  -> de-bias barely touched {TEAM_B}'s defence; look elsewhere "
              f"(e.g. {TEAM_A}'s attack Δ above, or bunker/regime effects).")


if __name__ == "__main__":
    main()
