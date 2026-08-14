"""
validation/squad_blend_cv.py — does folding squad strength into the matchup model improve
out-of-sample accuracy? Tested on the 2018 (FIFA19) and 2022 (FIFA21) World
Cups, the two folds we have squad data for, using the same train-before /
test-on protocol as model/dixon_coles.py.

Method: keep the Dixon-Coles TOTAL (DC is well-calibrated on totals), correct
the SUPREMACY (lam_home - lam_away) by blending toward a reference, rebuild the
rho-corrected scoreline matrix, score H/D/A log loss. References compared:
  - pure DC (status quo, w=0)
  - DC + Elo supremacy
  - DC + squad-strength supremacy        <- the proposed fix
  - DC + (Elo & squad averaged) supremacy
"""
from __future__ import annotations

# make the repo root importable when this file is run directly
# (python validation/foo.py) as well as from the root (python -m validation.foo)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import poisson
from model.dixon_coles import load_results, DixonColesModel, EloModel
from model.squad_strength import compute_squad_strength, load_fifa

# (World Cup year, FIFA edition to draw squad ratings from). The edition CSVs are
# downloaded into data/ on first use by load_fifa -- they are not committed.
FOLDS = [(2018, 2019), (2022, 2021)]
XI, LOOKBACK, MINM, MAXG = 0.0019, 4, 5, 10
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 2)


def squad_rating(fifa_year):
    s = compute_squad_strength(load_fifa(fifa_year))
    return s["squad_top11_mean"].dropna()


def fit_dc(train, ref):
    cutoff = ref - pd.DateOffset(years=LOOKBACK)
    sub = train[(train.date >= cutoff) & (train.date <= ref)].copy()
    c = pd.concat([sub.home_team, sub.away_team]).value_counts()
    valid = set(c[c >= MINM].index)
    sub = sub[sub.home_team.isin(valid) & sub.away_team.isin(valid)].reset_index(drop=True)
    dc = DixonColesModel(xi=XI); dc.fit(sub, ref_date=ref)
    return dc


def slope(train, rating, key):
    """LS slope mapping a strength differential to goal supremacy (origin)."""
    x, y = [], []
    for r in train.itertuples():
        if key == "squad":
            if r.home_team not in rating or r.away_team not in rating:
                continue
            d = rating[r.home_team] - rating[r.away_team]
        else:
            d = r.elodiff
        x.append(d); y.append(r.home_score - r.away_score)
    x, y = np.array(x, float), np.array(y, float)
    return float((x @ y) / (x @ x))


def matrix(lh, la, rho):
    i = np.arange(MAXG + 1)
    M = np.outer(poisson.pmf(i, lh), poisson.pmf(i, la))
    M[0, 0] *= 1 - lh * la * rho; M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho; M[1, 1] *= 1 - rho
    M = np.clip(M, 0, None); return M / M.sum()


def hda(M):
    return np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()


def run():
    df = load_results()
    pool = {name: {w: [] for w in WEIGHTS}
            for name in ["DC+Elo", "DC+Squad", "DC+Both"]}
    dc_pool = []
    print("Out-of-sample H/D/A log loss (lower is better)\n")
    for year, fifa_year in FOLDS:
        rating = squad_rating(fifa_year)
        test = df[(df.tournament == "FIFA World Cup") & (df.date.dt.year == year)]
        ref = test.date.min()
        train = df[df.date < ref].copy().reset_index(drop=True)
        dc = fit_dc(train, ref)
        elo = EloModel(); elo.fit(train)
        # pre-match elo diffs on training set for the elo slope
        e2 = EloModel(); diffs = []
        for r in train.itertuples():
            rh = e2.rating(r.home_team) + (0 if r.neutral else e2.home_advantage)
            ra = e2.rating(r.away_team)
            diffs.append(rh - ra); e2.update(r)
        train = train.assign(elodiff=diffs)
        b_sq = slope(train, rating, "squad")
        b_el = slope(train, rating, "elo")

        dc_ll = []
        rows = {name: {w: [] for w in WEIGHTS} for name in pool}
        for r in test.itertuples():
            if r.home_team not in dc.attack or r.away_team not in dc.attack:
                continue
            res = "H" if r.home_score > r.away_score else ("A" if r.home_score < r.away_score else "D")
            ht = 0.0 if r.neutral else dc.home_adv
            lh = float(np.exp(dc.attack[r.home_team] + dc.defence[r.away_team] + ht))
            la = float(np.exp(dc.attack[r.away_team] + dc.defence[r.home_team]))
            tot, s_dc = lh + la, lh - la
            # references in goal-supremacy units
            rh = elo.rating(r.home_team) + (0 if r.neutral else elo.home_advantage)
            s_elo = b_el * (rh - elo.rating(r.away_team))
            if r.home_team in rating and r.away_team in rating:
                s_sq = b_sq * (rating[r.home_team] - rating[r.away_team])
            else:
                s_sq = s_dc
            s_both = 0.5 * (s_elo + s_sq)
            # baseline DC
            Md = matrix(lh, la, dc.rho); h, d, a = hda(Md)
            dc_ll.append(-np.log(max({"H": h, "D": d, "A": a}[res], 1e-12)))
            for name, s_ref in [("DC+Elo", s_elo), ("DC+Squad", s_sq), ("DC+Both", s_both)]:
                for w in WEIGHTS:
                    s = (1 - w) * s_dc + w * s_ref
                    s = np.clip(s, -tot + .05, tot - .05)
                    M = matrix((tot + s) / 2, (tot - s) / 2, dc.rho)
                    hh, dd, aa = hda(M)
                    rows[name][w].append(-np.log(max({"H": hh, "D": dd, "A": aa}[res], 1e-12)))
        print(f"  {year} WC  (n={len(dc_ll)})  squad slope={b_sq:.4f} goals/pt, "
              f"elo slope={b_el:.5f}")
        print(f"    pure DC: {np.mean(dc_ll):.4f}")
        for name in pool:
            best_w = min(WEIGHTS, key=lambda w: np.mean(rows[name][w]))
            print(f"    {name:<9} best w={best_w:.1f} -> {np.mean(rows[name][best_w]):.4f}")
            for w in WEIGHTS:
                pool[name][w].extend(rows[name][w])
        dc_pool.extend(dc_ll)
        print()

    print("=" * 60)
    print("POOLED across 2018+2022 (the real test):")
    print(f"  pure DC: {np.mean(dc_pool):.4f}")
    for name in ["DC+Elo", "DC+Squad", "DC+Both"]:
        means = {w: float(np.mean(pool[name][w])) for w in WEIGHTS}
        best = min(means, key=means.get)
        print(f"  {name:<9} per-w: " + " ".join(f"{means[w]:.3f}" for w in WEIGHTS))
        print(f"  {'':<9} best w={best:.1f} -> {means[best]:.4f}  "
              f"({np.mean(dc_pool)-means[best]:+.4f} vs DC)")


if __name__ == "__main__":
    run()
