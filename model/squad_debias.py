"""
model/squad_debias.py — remove cross-confederation scale bias from Dixon-Coles
attack/defence coefficients, using globally-scaled squad ratings as the anchor.

ROOT CAUSE (France/Paraguay post-mortem)
----------------------------------------
DC's per-team attack and defence coefficients are NOT on a shared cross-
confederation scale: there are too few inter-confederation matches in the fit
window to tie the clusters together. Low-scoring confederations (CONMEBOL, AFC,
CONCACAF) get their DEFENCES systematically over-credited against the UEFA
attack scale they are crossed with. Measured mean defence residual (DC minus the
global squad-implied line): UEFA +0.01, CAF -0.15, CONMEBOL -0.25, AFC -0.38,
CONCACAF -0.41. Paraguay's DC defence sits at -1.022 (world rank 13) when the
squad line puts it near -0.57 — a -0.455 outlier. That single over-credited
coefficient is exactly what suppressed France's xG from ~2.5 to 1.5 and
manufactured the phantom "unders / NO-favourite" edges (up to ~30%).

WHY COEFFICIENT SPACE (not a lambda/supremacy blend)
----------------------------------------------------
France xG = exp(att_France + def_Paraguay). The broken term is def_Paraguay.
Blends that act on the lambda PRODUCT (supremacy-only, per-lambda) move both
teams together — they cannot raise France without disturbing Paraguay, because
Paraguay's own xG = exp(att_Paraguay + def_France) shares the total. Correcting
in COEFFICIENT space fixes def_Paraguay directly: France's xG rises, Paraguay's
is untouched (its terms — att_Paraguay, def_France — are already unbiased).

THE CORRECTION
--------------
Regress each DC coefficient on the squad rating (ONE global line => confederation
-neutral) and pull every team's coefficient a fraction `weight` toward that line.
Properties:
  * surgical   — only coefficients OFF the line (the biased ones) move; well-
                 connected UEFA teams sit on the line and barely budge.
  * asymmetric — attack and defence are corrected SEPARATELY.
  * cheap      — applied once, in place, after the fit; every matchup() inherits
                 corrected coefficients at zero per-call cost.

CV (2018+2022, 128 matches) vs pure DC, best weight ~0.7-0.8:
    H/D/A     1.0309 -> 0.9687  (+0.062)
    to-score  0.6055 -> 0.5897  (+0.016)
    BTTS      0.7498 -> 0.7320  (+0.018)
    Over 2.5  0.7379 -> 0.7101  (+0.028)
beating the supremacy blend (+0.042 H/D/A, team markets underwater) AND the
per-lambda blend (+0.055) on every market at once.

RESIDUAL SHRINKAGE (the tail fix — Argentina/Egypt post-mortem)
---------------------------------------------------------------
A single global weight cannot serve both kinds of off-line teams:
  * Antigua (+1.01 defence residual): tiny federation, near-zero connectivity —
    the residual is almost pure scale bias. Full correction is right.
  * Egypt (-0.62): 4x the CAF mean residual, on a line with def R^2 ~ 0.50 —
    a large chunk of that residual is plausibly REAL defensive signal the squad
    ratings can't see (organised unit, domestic-league defenders with cheap EA
    cards). Full 0.7 pull over-corrects: for ARG-EGY it pushed def_EGY from
    -0.749 to -0.314 (x1.55 on Argentina's lambda) when market-implied team totals
    implied -0.481 — an effective market weight of ~0.43, not 0.70.
`shrink=k` caps the removable residual at k standard deviations of the line's
own scatter: the effective per-team weight becomes
    w_t = weight * min(1, k*sigma / |resid|)
so teams within k*sigma get the full pull (the CV-validated average behaviour
is preserved) and tail teams — where bias and genuine signal are confounded —
get a proportionally gentler one. shrink=None reproduces the old behaviour
EXACTLY. Attack and defence are shrunk against their own sigmas, separately.

A residual UNIFORM total under-bias (~-0.5 goals on these WCs) is NOT this
module's job — it is left to the --scale calibration. De-bias fixes the SHAPE
(which team gets the goals); --scale fixes the LEVEL. They are orthogonal.

British spelling (defence) kept to match the codebase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from model.squad_strength import compute_squad_strength


def _fit_line(xs, ys):
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    b1, b0 = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 2 else float("nan")
    return b1, b0, r2


def debias_coefficients(dc, ratings_csv: str, weight: float = 0.7,
                        min_teams: int = 30, verbose: bool = True,
                        shrink: float | None = None) -> dict:
    """Pull dc.attack / dc.defence toward the global squad-implied line, IN PLACE.

    Parameters
    ----------
    dc          : a fitted DixonColesModel (its .attack/.defence dicts are mutated).
    ratings_csv : EA-FC / FIFA player CSV (Nationality, Overall, Position).
    weight      : fraction of the DC->squad residual to remove.
                  0 = off, 1 = snap fully onto the squad line. CV-best ~0.7-0.8.
    min_teams   : abort (leave DC untouched) if fewer squad-covered teams than this.
    shrink      : None = uniform weight for every team (legacy behaviour, exact).
                  k (e.g. 2.0) = cap the removable residual at k*sigma of the
                  regression's residual scatter; effective per-team weight is
                  weight * min(1, k*sigma/|resid|). Attack and defence use their
                  own sigmas.

    Returns a diagnostics dict. Teams lacking squad coverage keep their DC
    coefficients (they are still re-centred with everyone else).
    """
    if weight <= 0:
        return {"weight": 0.0, "applied": False}

    S = compute_squad_strength(pd.read_csv(ratings_csv, low_memory=False))
    att_sq = S["squad_attack"].dropna().to_dict()
    def_sq = S["squad_defense"].dropna().to_dict()
    teams = [t for t in dc.attack if t in att_sq and t in def_sq]
    if len(teams) < min_teams:
        if verbose:
            print(f"[debias] only {len(teams)} squad-covered teams (<{min_teams}); skipped")
        return {"weight": weight, "applied": False, "n_teams": len(teams)}

    a1, a0, ar2 = _fit_line([att_sq[t] for t in teams], [dc.attack[t] for t in teams])
    d1, d0, dr2 = _fit_line([def_sq[t] for t in teams], [dc.defence[t] for t in teams])

    # residuals for every team FIRST (shrinkage needs the scatter before applying)
    a_res = {t: dc.attack[t] - (a1 * att_sq[t] + a0) for t in teams}
    d_res = {t: dc.defence[t] - (d1 * def_sq[t] + d0) for t in teams}
    s_att = float(np.std(list(a_res.values())))
    s_def = float(np.std(list(d_res.values())))

    removed = []   # (|defence residual|, team, signed residual, effective def weight)
    n_shrunk = 0
    for t in teams:
        w_att = w_def = weight
        if shrink is not None:
            w_att = weight * min(1.0, shrink * s_att / max(abs(a_res[t]), 1e-12))
            w_def = weight * min(1.0, shrink * s_def / max(abs(d_res[t]), 1e-12))
            if w_att < weight - 1e-12 or w_def < weight - 1e-12:
                n_shrunk += 1
        dc.attack[t] = dc.attack[t] - w_att * a_res[t]
        dc.defence[t] = dc.defence[t] - w_def * d_res[t]
        removed.append((abs(d_res[t]), t, d_res[t], w_def))

    # re-centre attack to zero mean (DC identifiability convention); this is a
    # global constant shift, so it does not change any single game's supremacy,
    # only the shared level (which --scale owns).
    m = float(np.mean(list(dc.attack.values())))
    for t in dc.attack:
        dc.attack[t] -= m

    removed.sort(reverse=True)
    if verbose:
        tag = f"shrink={shrink} (att sd={s_att:.2f}, def sd={s_def:.2f}, " \
              f"{n_shrunk} teams shrunk)" if shrink is not None else "uniform"
        print(f"[debias] w={weight} {tag}  att~squad R^2={ar2:.2f}  "
              f"def~squad R^2={dr2:.2f}  ({len(teams)} teams)")
        tops = ", ".join(f"{t} {r:+.2f} (w_eff {we:.2f})" for _, t, r, we in removed[:6])
        print(f"[debias] largest defence over-credits corrected: {tops}")
    return {"weight": weight, "applied": True, "att_r2": ar2, "def_r2": dr2,
            "n_teams": len(teams), "shrink": shrink, "n_shrunk": n_shrunk,
            "att_sigma": s_att, "def_sigma": s_def,
            "att_line": (a1, a0), "def_line": (d1, d0)}


def estimate_total_calibration(dc, df, ref_date, xi: float = 0.0019,
                               lookback_years: int = 4, verbose: bool = True,
                               wc_window_days: int = 100, wc_min_games: int = 30) -> float:
    """Multiplicative goal calibration (the LEVEL knob), MEASURED not guessed.

    Priority 1 -- the live tournament. Once enough games of the current World Cup
    have been played, calibrate directly on them: actual/predicted total over the
    completed WC games (each game predicted with its own venue via home_adv /
    home_def). This is the honest regime factor for the exact games being bet, so
    there is no hand-tuned "WC runs ~15% hot" prior -- it is read off the data and
    self-updates every refit as the tournament progresses.

    Priority 2 -- pre-tournament fallback. Before a live WC sample exists, use the
    closest leakage-free analog: recent NEUTRAL, COMPETITIVE games (major-
    tournament / Nations-League, not friendlies), recency-weighted with the fit's
    xi. After the home-split fix this fallback is typically ~1.00 (the model self-
    calibrates on neutral games); the WC still tends to run hotter, which is
    exactly what Priority 1 captures once games exist.

    Returns sum(actual)/sum(predicted) over the chosen pool; 1.0 if none.
    """
    hd = getattr(dc, "home_def", 0.0)

    def _ratio(rows, decay):
        num = den = 0.0
        n = 0
        for r in rows.itertuples():
            if r.home_team not in dc.attack or r.away_team not in dc.attack:
                continue
            nn = 0.0 if r.neutral else 1.0
            lh = np.exp(dc.attack[r.home_team] + dc.defence[r.away_team] + dc.home_adv * nn)
            la = np.exp(dc.attack[r.away_team] + dc.defence[r.home_team] - hd * nn)
            w = np.exp(-xi * (ref_date - r.date).days) if decay else 1.0
            num += w * (r.home_score + r.away_score)
            den += w * (lh + la)
            n += 1
        return (float(num / den) if den > 0 else None), n

    # Priority 1: the live World Cup
    wc = df[(df.tournament == "FIFA World Cup")
            & (df.date >= ref_date - pd.DateOffset(days=wc_window_days))
            & (df.date <= ref_date)
            & df.home_score.notna() & df.away_score.notna()]
    r_wc, n_wc = _ratio(wc, decay=False)
    if r_wc is not None and n_wc >= wc_min_games:
        if verbose:
            print(f"[scale] {r_wc:.3f} -- measured from {n_wc} completed games of the "
                  f"current World Cup (live tournament regime)")
        return r_wc

    # Priority 2: pre-tournament neutral-competitive proxy
    cutoff = ref_date - pd.DateOffset(years=lookback_years)
    sub = df[(df.date >= cutoff) & (df.date <= ref_date)
             & (df.neutral == True) & (df.tournament != "Friendly")
             & df.home_score.notna() & df.away_score.notna()]
    r_pre, n_pre = _ratio(sub, decay=True)
    if r_pre is None:
        return 1.0
    if verbose:
        print(f"[scale] {r_pre:.3f} -- pre-tournament, from {n_pre} neutral competitive "
              f"games (no live WC sample yet; the WC usually runs hotter)")
    return r_pre
