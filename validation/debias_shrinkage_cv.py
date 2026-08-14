"""
validation/debias_shrinkage_cv.py — is residual-shrinkage a viable upgrade to the de-bias?

Two tests, run in order:

  1. HISTORICAL CV (the decision criterion)
     Refit the DC model as of the eve of WC 2018 and WC 2022 (same settings as
     matchup.fit_model: xi, 4y lookback, min 5 matches, split-home), apply each
     de-bias variant to a fresh copy of the fit, and score every completed WC
     game with matchup.matchup() — i.e. the exact rho-corrected grid production
     uses. Metrics match the module docstring: H/D/A multiclass log loss,
     team-to-score binary log loss, BTTS, Over 2.5. Lower is better.

     PASS = a shrink variant holds the uniform w=0.7 H/D/A gain (within noise)
     without losing the team-market gains. The w=0.7 uniform row should roughly
     reproduce the docstring numbers (H/D/A ~0.969 etc.) — if it doesn't, the
     harness and the original CV disagree somewhere and nothing else here
     should be trusted until that's resolved.

  2. CURRENT-FIT SANITY PANEL (the motivating case)
     Refit on all current data (exactly fit_model minus the baked-in debias),
     apply each variant, and print def_EGY, def_ARG and the Argentina–Egypt xG
     against the market-implied references measured 2026-07-07:
         def_EGY(market) ~ -0.481,  lambda_ARG(market) ~ 2.11
     PASS = the winning CV variant lands def_EGY meaningfully closer to -0.48
     than uniform w=0.7 does (-0.314).

Usage:
    python validation/debias_shrinkage_cv.py            # full grid
    python validation/debias_shrinkage_cv.py --quick    # skip 2018 (faster sanity run)

NOTE: uses the CURRENT ratings CSV to de-bias the 2018/2022 fits — anachronistic,
but identical to how the original w=0.7 CV was run, so the comparison is fair.
"""
from __future__ import annotations

# make the repo root importable when this file is run directly
# (python validation/foo.py) as well as from the root (python -m validation.foo)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import copy
import sys

import numpy as np
import pandas as pd

from model.dixon_coles import DixonColesModel, load_results
from model.squad_ratings import SquadRatings
from model.squad_debias import debias_coefficients

# import matchup for its exact scoreline machinery; module-level fit only runs
# under __main__ so this is side-effect free (MATCHUP_SCALE stays 1.0).
from model import matchup as MU

XI, LOOKBACK_YEARS, MIN_MATCHES = 0.0019, 4, 5
SPLIT_HOME = True

# (label, weight, shrink)
VARIANTS = [
    ("pure DC (w=0)",            0.0, None),
    ("uniform w=0.5",            0.5, None),
    ("uniform w=0.7 (prod)",     0.7, None),
    ("w=0.7 shrink 2.5sd",       0.7, 2.5),
    ("w=0.7 shrink 2.0sd",       0.7, 2.0),
    ("w=0.7 shrink 1.5sd",       0.7, 1.5),
    ("w=0.7 shrink 1.0sd",       0.7, 1.0),
]

WORLD_CUPS = [
    ("WC 2018", pd.Timestamp("2018-06-14"), pd.Timestamp("2018-07-15")),
    ("WC 2022", pd.Timestamp("2022-11-20"), pd.Timestamp("2022-12-18")),
]

# Market-implied references, Argentina vs Egypt, measured 2026-07-07
MKT_DEF_EGY = -0.481
MKT_LAM_ARG = 2.11

EPS = 1e-12


def fit_asof(df: pd.DataFrame, ref: pd.Timestamp) -> DixonColesModel:
    """matchup.fit_model()'s fit block, parameterised by ref date, no debias."""
    cutoff = ref - pd.DateOffset(years=LOOKBACK_YEARS)
    sub = df[(df.date >= cutoff) & (df.date <= ref)].copy()
    counts = pd.concat([sub.home_team, sub.away_team]).value_counts()
    valid = set(counts[counts >= MIN_MATCHES].index)
    sub = sub[sub.home_team.isin(valid) & sub.away_team.isin(valid)].reset_index(drop=True)
    dc = DixonColesModel(xi=XI)
    dc.fit(sub, ref_date=ref, split_home=SPLIT_HOME)
    return dc


def apply_variant(dc_base: DixonColesModel, ratings_csv: str,
                  weight: float, shrink: float | None) -> DixonColesModel:
    dc = copy.deepcopy(dc_base)
    if weight > 0:
        debias_coefficients(dc, ratings_csv, weight=weight, shrink=shrink,
                            verbose=False)
    return dc


def score_games(dc: DixonColesModel, games: pd.DataFrame) -> dict:
    """Log losses over a set of completed games, via matchup()'s exact grid."""
    ll_hda, ll_ts, ll_btts, ll_o25 = [], [], [], []
    skipped = 0
    for g in games.itertuples():
        if g.home_team not in dc.attack or g.away_team not in dc.attack:
            skipped += 1
            continue
        r = MU.matchup(dc, g.home_team, g.away_team,
                       neutral=bool(g.neutral), scale=1.0)

        # H/D/A
        if g.home_score > g.away_score:
            p = r["p_home"]
        elif g.home_score < g.away_score:
            p = r["p_away"]
        else:
            p = r["p_draw"]
        ll_hda.append(-np.log(max(p, EPS)))

        # team to score (both teams pooled)
        p_h = r["ou"]["home"][0.5][0]   # P(home over 0.5)
        p_a = r["ou"]["away"][0.5][0]
        y_h = 1.0 if g.home_score >= 1 else 0.0
        y_a = 1.0 if g.away_score >= 1 else 0.0
        for p_, y_ in ((p_h, y_h), (p_a, y_a)):
            ll_ts.append(-np.log(max(p_ if y_ else 1 - p_, EPS)))

        # BTTS
        p_b = r["btts"]
        y_b = 1.0 if (g.home_score >= 1 and g.away_score >= 1) else 0.0
        ll_btts.append(-np.log(max(p_b if y_b else 1 - p_b, EPS)))

        # total over 2.5
        p_o = r["ou"]["total"][2.5][0]
        y_o = 1.0 if (g.home_score + g.away_score) > 2.5 else 0.0
        ll_o25.append(-np.log(max(p_o if y_o else 1 - p_o, EPS)))

    return {"n": len(ll_hda), "skipped": skipped,
            "hda": float(np.mean(ll_hda)) if ll_hda else np.nan,
            "to_score": float(np.mean(ll_ts)) if ll_ts else np.nan,
            "btts": float(np.mean(ll_btts)) if ll_btts else np.nan,
            "over25": float(np.mean(ll_o25)) if ll_o25 else np.nan}


def main():
    quick = "--quick" in sys.argv
    cups = WORLD_CUPS[1:] if quick else WORLD_CUPS

    df = load_results()
    ratings_csv = SquadRatings().source

    # ---------------- 1. HISTORICAL CV ----------------
    print("=" * 78)
    print("1. HISTORICAL CV  —  " + " + ".join(name for name, *_ in cups))
    print("=" * 78)

    pools = []   # (cup_name, dc_base, games)
    for name, start, end in cups:
        ref = start - pd.Timedelta(days=1)
        games = df[(df.tournament == "FIFA World Cup")
                   & (df.date >= start) & (df.date <= end)
                   & df.home_score.notna() & df.away_score.notna()].copy()
        print(f"[fit] {name}: fitting as of {ref.date()}  "
              f"({len(games)} tournament games to score) ...")
        pools.append((name, fit_asof(df, ref), games))

    rows = []
    for label, w, k in VARIANTS:
        per_cup = []
        for name, dc_base, games in pools:
            dc = apply_variant(dc_base, ratings_csv, w, k)
            per_cup.append(score_games(dc, games))
        n = sum(s["n"] for s in per_cup)
        agg = {m: float(np.sum([s[m] * s["n"] for s in per_cup]) / n)
               for m in ("hda", "to_score", "btts", "over25")}
        agg.update({"label": label, "n": n,
                    "skipped": sum(s["skipped"] for s in per_cup)})
        rows.append(agg)

    base = next(r for r in rows if r["label"].startswith("pure DC"))
    print(f"\n{'variant':<24}{'H/D/A':>9}{'d':>8}{'to-score':>10}{'BTTS':>8}"
          f"{'O2.5':>8}   (n={rows[0]['n']}, log loss, lower=better, "
          f"d = H/D/A gain vs pure DC)")
    print("-" * 78)
    for r in rows:
        d = base["hda"] - r["hda"]
        print(f"{r['label']:<24}{r['hda']:>9.4f}{d:>+8.4f}{r['to_score']:>10.4f}"
              f"{r['btts']:>8.4f}{r['over25']:>8.4f}")
    if rows[0]["skipped"]:
        print(f"(skipped {rows[0]['skipped']} games with teams missing from the fit)")

    # ---------------- 2. CURRENT-FIT SANITY PANEL ----------------
    print()
    print("=" * 78)
    print("2. CURRENT FIT  —  Argentina vs Egypt vs the market")
    print("=" * 78)
    ref_now = df.date.max()
    print(f"[fit] fitting as of {ref_now.date()} ...")
    dc_now = fit_asof(df, ref_now)

    print(f"\nmarket reference (implied team totals, 2026-07-07): "
          f"def_EGY ~ {MKT_DEF_EGY:+.3f},  lambda_ARG ~ {MKT_LAM_ARG:.2f}\n")
    print(f"{'variant':<24}{'def_EGY':>9}{'|mkt gap|':>10}{'def_ARG':>9}"
          f"{'xG ARG':>8}{'xG EGY':>8}")
    print("-" * 78)
    for label, w, k in VARIANTS:
        dc = apply_variant(dc_now, ratings_csv, w, k)
        if "Egypt" not in dc.attack or "Argentina" not in dc.attack:
            print(f"{label:<24}  (Egypt/Argentina missing from fit)")
            continue
        r = MU.matchup(dc, "Argentina", "Egypt", neutral=True, scale=1.0)
        de = dc.defence["Egypt"]
        print(f"{label:<24}{de:>+9.4f}{abs(de - MKT_DEF_EGY):>10.4f}"
              f"{dc.defence['Argentina']:>+9.4f}"
              f"{r['xg_home']:>8.3f}{r['xg_away']:>8.3f}")

    print("\nDecision rule:")
    print("  * Section 1: pick the shrink k whose H/D/A delta is within ~0.005 of")
    print("    uniform w=0.7 AND whose team markets (to-score/BTTS/O2.5) are no")
    print("    worse. If uniform w=0.7 doesn't roughly reproduce the docstring")
    print("    numbers (H/D/A ~0.969), debug the harness before concluding anything.")
    print("  * Section 2: that k should also cut the |mkt gap| on def_EGY vs the")
    print("    uniform row. If CV says shrink is free and the panel says it closes")
    print("    half the Egypt gap, ship it: pass shrink=k in fit_model()'s")
    print("    debias_coefficients() call in model/matchup.py.")


if __name__ == "__main__":
    main()
