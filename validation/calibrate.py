#!/usr/bin/env python3
"""
validation/calibrate.py  -  Are the model's goal totals calibrated, or is it biased low?

This answers ONE question with evidence instead of vibes: when the matchup model
says P(under X.5) = p, do totals actually land under that line ~p of the time? A
model can rank teams correctly and still get the LEVEL of scoring wrong, which
silently distorts every total, BTTS and correct-score number it produces. If the
totals are off, this script also fits the correction (a single scale factor c).

Run it from the repo root (it imports the model to rebuild predictions in Mode B).
Realized scores come from the SAME martj42 results.csv your model trains on, so there
is no separate results file to maintain.

------------------------------------------------------------------------------------
WHAT IT PRODUCES
------------------------------------------------------------------------------------
  1. Total-goals bias test ....... mean predicted total vs mean realized total (+CI).
                                   The headline number. Low here => model biased low.
  2. Per-line hit-rate table ..... for lines 0.5..6.5: model mean P(under) vs the
                                   empirical fraction that went under, with a z-score.
  3. Reliability curve ........... bins P(under 2.5); plots predicted vs empirical.
  4. Calibration slope/intercept . one logistic fit. intercept!=0 => directional bias;
                                   slope<1 => overconfident.
  5. Brier / log-loss ............ scalar accuracy scores for under 2.5 and 3.5.
  6. Recalibration fit ........... the lambda scale factor c that makes the model match
                                   reality (MLE), plus an optional joint (c, rho) refit
                                   and a "rho off" sensitivity pass.
  7. Boundary audit .............. prints the discrete CDF around a line, confirming the
                                   half-goal line is read off the right cells (under 2.5
                                   must equal P(total in {0,1,2}), nothing else).

------------------------------------------------------------------------------------
INPUTS  (two ways to supply model predictions)
------------------------------------------------------------------------------------
MODE A  (preferred, zero leakage):  --pred predictions.csv
    A log of what your model said, with columns (header required):
        date,home_team,away_team,lambda_home,lambda_away,rho
    These are the model's expected goals for each side and the DC rho for that game.
    If you logged these at prediction time, calibration is on exactly what you claimed.
    (rho may be a constant column if your fit uses a single global rho.)

MODE B  (reconstruct from the model):  --from-model
    Imports your matchup module and calls matchup() for every completed WC fixture to
    rebuild lambda_home / lambda_away / rho, then calibrates those. Convenient but note
    the LEAKAGE CAVEAT below. Use --as-of-cutoff if your fit accepts a date cutoff.

REALIZED RESULTS (always):  --results results.csv  (path or the raw martj42 URL)
    Uses tournament == "FIFA World Cup" and date >= --since (default 2026-06-01).

------------------------------------------------------------------------------------
LEAKAGE CAVEAT (Mode B)
------------------------------------------------------------------------------------
model/matchup.py fits once on all data with no date cutoff. If you reconstruct a match's
prediction by re-running the model NOW, that match's own result is already in the
training set, so the model looks slightly better-calibrated than it really was. With
4 years of data the per-match leakage is small, but it biases toward "looks fine."
For a clean read either (a) log predictions live and use MODE A, or (b) pass
--as-of-cutoff and make your fit_model accept a date cutoff so each match is predicted
out-of-sample.

------------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------------
    python validation/calibrate.py --selftest                 # validate the math (no model needed)
    python validation/calibrate.py --pred predictions.csv --results results.csv
    python validation/calibrate.py --from-model --results results.csv
"""

# make the repo root importable when this file is run directly
# (python validation/foo.py) as well as from the root (python -m validation.foo)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import csv
import math
import sys
import unicodedata
from dataclasses import dataclass, field

import numpy as np

MAX_GOALS = 15
LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
MARTJ42_URL = ("https://raw.githubusercontent.com/martj42/"
               "international_results/master/results.csv")


# --------------------------------------------------------------------------------------
# Dixon-Coles scoreline matrix (matches model/matchup.py's rho-corrected grid)
# --------------------------------------------------------------------------------------
def _poisson_pmf(lam, kmax=MAX_GOALS):
    k = np.arange(kmax + 1)
    # exp(-lam) * lam^k / k!  computed in log space for stability
    logp = -lam + k * math.log(max(lam, 1e-12)) - np.array(
        [math.lgamma(i + 1) for i in k])
    return np.exp(logp)


def dc_matrix(lh, la, rho, kmax=MAX_GOALS):
    """P(home=i, away=j) with the Dixon-Coles low-score tau correction, renormalised."""
    h = _poisson_pmf(lh, kmax)
    a = _poisson_pmf(la, kmax)
    M = np.outer(h, a)
    M[0, 0] *= 1.0 - lh * la * rho
    M[0, 1] *= 1.0 + lh * rho
    M[1, 0] *= 1.0 + la * rho
    M[1, 1] *= 1.0 - rho
    M = np.clip(M, 0.0, None)
    s = M.sum()
    return M / s if s > 0 else M


def total_dist(M):
    """Distribution of total goals (home+away) from a scoreline matrix."""
    n = M.shape[0] + M.shape[1] - 1
    t = np.zeros(n)
    for i in range(M.shape[0]):
        row = M[i]
        t[i:i + len(row)] += row
    return t


def p_under(M, line):
    """P(total < line). For an X.5 line this is P(total <= floor(line))."""
    t = total_dist(M)
    cap = int(math.floor(line))
    return float(t[:cap + 1].sum())


def expected_total(M):
    t = total_dist(M)
    return float((np.arange(len(t)) * t).sum())


def dc_cell_prob(hg, ag, lh, la, rho):
    """P(exact score hg-ag) under the DC model; used by the recalibration MLE."""
    M = dc_matrix(lh, la, rho)
    hg = min(hg, M.shape[0] - 1)
    ag = min(ag, M.shape[1] - 1)
    return max(M[hg, ag], 1e-12)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass
class Match:
    date: str
    home: str
    away: str
    hg: int            # realized home goals
    ag: int            # realized away goals
    lh: float = None   # model expected home goals
    la: float = None   # model expected away goals
    rho: float = None
    stage: str = "all"
    matrix: object = None  # the model's own rho-corrected score matrix, if available

    @property
    def total(self):
        return self.hg + self.ag


def mat(m):
    """The scoreline matrix for a match: use the model's own matrix if it gave us one
    (exact, rho already baked in); otherwise rebuild it from lambdas + rho."""
    if m.matrix is not None:
        M = np.asarray(m.matrix, dtype=float)
        s = M.sum()
        return M / s if s > 0 else M
    return dc_matrix(m.lh, m.la, m.rho)


def _norm(s):
    """Accent-insensitive, case-insensitive team-name key (matches model/matchup.py logic)."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip().lower()


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------
def load_results(path, since="2026-06-01", tournament="FIFA World Cup"):
    """Realized WC scores from the martj42 results.csv (local path or raw URL)."""
    rows = _read_csv_any(path)
    out = {}
    for r in rows:
        if r.get("tournament", "").strip() != tournament:
            continue
        d = r.get("date", "").strip()
        if d < since:
            continue
        hs, as_ = r.get("home_score", ""), r.get("away_score", "")
        try:
            hg, ag = int(float(hs)), int(float(as_))
        except (TypeError, ValueError):
            continue  # not yet played (martj42 uses 'NA' for unplayed fixtures)
        key = (d, _norm(r["home_team"]), _norm(r["away_team"]))
        neutral = str(r.get("neutral", "")).strip().lower() in ("true", "1", "yes")
        out[key] = (hg, ag, r["home_team"].strip(), r["away_team"].strip(), neutral)
    return out


def load_predictions(path):
    """Model predictions log: date,home_team,away_team,lambda_home,lambda_away,rho."""
    rows = _read_csv_any(path)
    preds = {}
    for r in rows:
        key = (r["date"].strip(), _norm(r["home_team"]), _norm(r["away_team"]))
        preds[key] = (float(r["lambda_home"]), float(r["lambda_away"]),
                      float(r["rho"]), None)
    return preds


def _read_csv_any(path):
    if str(path).startswith("http"):
        import urllib.request
        with urllib.request.urlopen(path) as resp:        # nosec - user-supplied
            text = resp.read().decode("utf-8", "replace")
        return list(csv.DictReader(text.splitlines()))
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def join_matches(results, preds):
    """Inner-join realized results with model predictions on (date, home, away)."""
    matches = []
    missing = 0
    for key, (hg, ag, home, away, _neutral) in results.items():
        if key not in preds:
            missing += 1
            continue
        lh, la, rho, M = preds[key]
        matches.append(Match(date=key[0], home=home, away=away, hg=hg, ag=ag,
                             lh=lh, la=la, rho=rho, matrix=M))
    return matches, missing


# --------------------------------------------------------------------------------------
# Mode B: reconstruct predictions by importing the user's model
# --------------------------------------------------------------------------------------
def predictions_from_model(results, as_of_cutoff=False, default_rho=-0.05):
    """
    Build a prediction for each completed fixture by calling the user's matchup().
    Wired to model/matchup.py's actual interface:

        from model import matchup
        dc = matchup.fit_model()                       # returns a DixonColesModel
        m  = matchup.matchup(dc, home, away, neutral)  # rho-corrected score matrix

    matchup() returns the score matrix, which is used directly (exact, rho baked in).
    The real `neutral` flag per fixture is read from results.csv and passed through.
    """
    try:
        from model import matchup
    except Exception as e:                                    # pragma: no cover
        print(f"[from-model] could not import model.matchup: {e}", file=sys.stderr)
        print("            run this from the repo root, or as python validation/calibrate.py",
              file=sys.stderr)
        sys.exit(2)

    try:
        import inspect
        print(f"[from-model] matchup() signature: "
              f"{inspect.signature(matchup.matchup)}", file=sys.stderr)
    except (TypeError, ValueError):
        pass

    # --- fit the DixonColesModel ONCE (the first positional arg matchup() needs) --------
    dc = _fit_dc(matchup)
    if dc is None:
        print("[from-model] could not build a DixonColesModel via fit_model(); "
              "see the error above.", file=sys.stderr)
        sys.exit(2)

    preds = {}
    first = True
    for (date, hkey, akey), (_, _, home, away, neutral) in results.items():
        if as_of_cutoff:
            dc_match = _fit_dc(matchup, cutoff=date) or dc   # out-of-sample if supported
        else:
            dc_match = dc
        try:
            m = matchup.matchup(dc_match, home, away, neutral)   # <-- real signature
        except Exception as e:
            print(f"[from-model] matchup({home} v {away}) failed: {e}", file=sys.stderr)
            continue
        if first:
            _describe_return(m)                                  # one-time diagnostic
            first = False
        lh, la, rho, M = _extract_lambdas(m, default_rho=default_rho)
        if lh is None:
            print(f"[from-model] couldn't read a prediction for {home} v {away}; "
                  f"matchup() returned {type(m).__name__}. Paste the line above and "
                  f"I'll adjust _extract_lambdas().", file=sys.stderr)
            continue
        preds[(date, hkey, akey)] = (lh, la, rho, M)
    print(f"[from-model] built {len(preds)} predictions.", file=sys.stderr)
    return preds


def _fit_dc(matchup, cutoff=None):
    """Get a fitted DixonColesModel. Tries fit_model() with/without a cutoff, then a few
    common fallbacks (fit_model(load_results()) etc.)."""
    fm = getattr(matchup, "fit_model", None)
    if fm is None:
        print("[from-model] model/matchup.py has no fit_model(); set dc manually.", file=sys.stderr)
        return None
    attempts = []
    if cutoff is not None:
        attempts += [lambda: fm(cutoff=cutoff)]
    attempts += [lambda: fm()]
    if hasattr(matchup, "load_results"):
        attempts += [lambda: fm(matchup.load_results())]
    for call in attempts:
        try:
            dc = call()
            if dc is not None:
                return dc
        except TypeError:
            continue
        except Exception as e:
            print(f"[from-model] fit_model() raised: {e}", file=sys.stderr)
            return None
    return None


def _describe_return(m):
    """Print what matchup() returns so extraction can be fixed in one shot if needed."""
    try:
        import numpy as _np
        arr = _np.asarray(m, dtype=float)
        shape = getattr(arr, "shape", None)
    except Exception:
        shape = None
    attrs = [a for a in dir(m) if not a.startswith("_")][:20]
    print(f"[from-model] matchup() returns {type(m).__name__}; array-shape={shape}; "
          f"attrs={attrs}", file=sys.stderr)


def _extract_lambdas(m, default_rho=-0.05):
    """Return (lambda_home, lambda_away, rho, matrix) from whatever matchup() returns.

    If matchup() returns its rho-corrected score matrix (the common case), that matrix is
    carried through and used directly for all probability calculations -- exact, with rho
    already baked in. Lambdas are still derived from its marginals so the recalibration
    scale-factor in section [6] has something to scale; rho there falls back to
    `default_rho` (your global rho) since it can't be read back out of a matrix.
    """
    # dict-like  (your matchup() returns: xg_home, xg_away, ..., matrix)
    if isinstance(m, dict):
        lh = m.get("xg_home", m.get("lambda_home", m.get("lh", m.get("home_xg"))))
        la = m.get("xg_away", m.get("lambda_away", m.get("la", m.get("away_xg"))))
        rho = m.get("rho", default_rho)
        M = m.get("matrix", m.get("score_matrix"))
        if lh is not None:
            return float(lh), float(la), float(rho), (np.asarray(M, float)
                                                       if M is not None else None)
    # attribute-style object (e.g. a result object exposing .lambda_home / .matrix)
    for ln in ("lambda_home", "lh", "home_xg", "mu_home"):
        if hasattr(m, ln):
            la_attr = next((x for x in ("lambda_away", "la", "away_xg", "mu_away")
                            if hasattr(m, x)), None)
            rho = getattr(m, "rho", default_rho)
            M = None
            for mn in ("matrix", "score_matrix", "grid", "M"):
                if hasattr(m, mn):
                    M = np.asarray(getattr(m, mn), float)
                    break
            return (float(getattr(m, ln)), float(getattr(m, la_attr)),
                    float(rho), M)
    # matchup() returned the matrix itself -> use it directly, derive lambdas from marginals
    arr = np.asarray(m, dtype=float) if not isinstance(m, dict) else None
    if arr is not None and arr.ndim == 2:
        gh = np.arange(arr.shape[0])
        ga = np.arange(arr.shape[1])
        lh = float((gh * arr.sum(axis=1)).sum())
        la = float((ga * arr.sum(axis=0)).sum())
        return lh, la, default_rho, arr
    return None, None, None, None


# --------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------
def _bootstrap_ci(diffs, n=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def bias_test(matches):
    pred = np.array([expected_total(mat(m)) for m in matches])
    real = np.array([m.total for m in matches], dtype=float)
    diff = pred - real                       # >0 model high, <0 model LOW
    lo, hi = _bootstrap_ci(diff)
    return {
        "n": len(matches),
        "mean_pred": float(pred.mean()),
        "mean_real": float(real.mean()),
        "mean_diff": float(diff.mean()),
        "ci": (lo, hi),
        "biased_low": hi < 0,                # whole CI below 0 => significantly low
        "biased_high": lo > 0,
    }


def per_line_table(matches):
    rows = []
    for L in LINES:
        ps = np.array([p_under(mat(m), L) for m in matches])
        emp = np.array([1.0 if m.total < L else 0.0 for m in matches])
        n = len(matches)
        mp, me = ps.mean(), emp.mean()
        se = math.sqrt(max(mp * (1 - mp), 1e-9) / n)
        z = (me - mp) / se if se > 0 else 0.0   # +z => unders landed MORE than modelled
        rows.append({"line": L, "model_p_under": mp, "emp_under": me,
                     "n_under": int(emp.sum()), "n": n, "z": z})
    return rows


def reliability(matches, line=2.5, nbins=5):
    ps = np.array([p_under(mat(m), line) for m in matches])
    y = np.array([1.0 if m.total < line else 0.0 for m in matches])
    order = np.argsort(ps)
    ps, y = ps[order], y[order]
    bins = np.array_split(np.arange(len(ps)), min(nbins, max(1, len(ps))))
    out = []
    for b in bins:
        if len(b) == 0:
            continue
        out.append({"pred": float(ps[b].mean()), "emp": float(y[b].mean()),
                    "n": int(len(b))})
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def calibration_slope_intercept(matches, line=2.5, ridge=1e-6, iters=50):
    """Logistic fit  outcome ~ b0 + b1*logit(p).  Perfect => b0=0, b1=1."""
    p = np.array([p_under(mat(m), line) for m in matches])
    p = np.clip(p, 1e-4, 1 - 1e-4)
    x = np.log(p / (1 - p))
    y = np.array([1.0 if m.total < line else 0.0 for m in matches])
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iters):
        eta = X @ b
        mu = _sigmoid(eta)
        W = np.clip(mu * (1 - mu), 1e-6, None)
        z = eta + (y - mu) / W
        XtW = X.T * W
        H = XtW @ X + ridge * np.eye(2)
        b_new = np.linalg.solve(H, XtW @ z)
        if np.max(np.abs(b_new - b)) < 1e-8:
            b = b_new
            break
        b = b_new
    return {"intercept": float(b[0]), "slope": float(b[1])}


def brier_logloss(matches, line=2.5):
    p = np.clip(np.array([p_under(mat(m), line)
                          for m in matches]), 1e-9, 1 - 1e-9)
    y = np.array([1.0 if m.total < line else 0.0 for m in matches])
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return {"line": line, "brier": brier, "logloss": ll}


# --------------------------------------------------------------------------------------
# Recalibration
# --------------------------------------------------------------------------------------
def _neg_loglik_scale(c, matches):
    if c <= 0:
        return 1e18
    s = 0.0
    for m in matches:
        s += math.log(dc_cell_prob(m.hg, m.ag, c * m.lh, c * m.la, m.rho))
    return -s


def _golden_min(f, lo, hi, tol=1e-4):
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1, f2 = f(c1), f(c2)
    while b - a > tol:
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = f(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = f(c2)
    return (a + b) / 2


def fit_scale(matches, lo=0.5, hi=2.0):
    """MLE multiplicative factor c on both lambdas. c>1 => your model was too LOW."""
    c = _golden_min(lambda x: _neg_loglik_scale(x, matches), lo, hi)
    # implied shift in mean predicted total
    before = np.mean([expected_total(mat(m)) for m in matches])
    after = np.mean([expected_total(dc_matrix(c * m.lh, c * m.la, m.rho))
                     for m in matches])
    return {"c": float(c), "mean_total_before": float(before),
            "mean_total_after": float(after)}


def fit_scale_and_rho(matches):
    """Joint (c, rho) refit. Coarse grid on rho, golden-section on c per rho."""
    best = None
    for rho in np.linspace(-0.20, 0.20, 21):
        ms = [Match(m.date, m.home, m.away, m.hg, m.ag, m.lh, m.la, rho)
              for m in matches]
        c = _golden_min(lambda x: _neg_loglik_scale(x, ms), 0.5, 2.0)
        nll = _neg_loglik_scale(c, ms)
        if best is None or nll < best[2]:
            best = (c, rho, nll)
    return {"c": float(best[0]), "rho": float(best[1]), "neg_loglik": float(best[2])}


def rho_sensitivity(matches):
    """How much of the unders lean is attributable to rho? Compares P(under 2.5) under
    the model's rho vs rho=0, reconstructing both from lambdas so it's apples-to-apples
    (independent of whether matchup() handed us a prebuilt matrix)."""
    on = np.mean([p_under(dc_matrix(m.lh, m.la, m.rho), 2.5) for m in matches])
    off = np.mean([p_under(dc_matrix(m.lh, m.la, 0.0), 2.5) for m in matches])
    return {"under25_model_with_rho": float(on),
            "under25_model_rho_off": float(off),
            "shift": float(on - off)}


def boundary_audit(matches, line=2.5):
    """Print the discrete CDF around `line` for the first match to catch off-by-ones."""
    if not matches:
        return None
    m = matches[0]
    t = total_dist(mat(m))
    cap = int(math.floor(line))
    return {
        "example": f"{m.home} v {m.away}",
        "cells": {k: float(t[k]) for k in range(0, min(6, len(t)))},
        f"P(total<{line}) [=under, total in 0..{cap}]": float(t[:cap + 1].sum()),
        "WRONG P(total<=line as 3)": float(t[:cap + 2].sum()),
        "WRONG P(total<line-1)": float(t[:cap].sum()),
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _verdict(bias, slope):
    if bias["biased_low"]:
        return ("MODEL BIASED LOW. Predicted totals sit below realized totals with the "
                "whole CI under zero -> the model systematically under-predicts goals and "
                "over-states P(under). Apply the scale factor c below and re-check.")
    if bias["biased_high"]:
        return ("Model biased HIGH on totals -> the model expects more goals than reality "
                "delivered; apply the scale factor c below to bring the level down.")
    return ("Totals are unbiased (CI straddles zero): the model's goal level is sound, so "
            "P(under X.5) can be read at face value.")


def render_report(res, plot_path=None):
    L = []
    P = L.append
    P("=" * 78)
    P("DIXON-COLES TOTALS CALIBRATION REPORT")
    P("=" * 78)
    b = res["bias"]
    P(f"\nMatches calibrated: {b['n']}")
    P("\n[1] TOTAL-GOALS BIAS  (the headline)")
    P(f"    mean predicted total : {b['mean_pred']:.3f}")
    P(f"    mean realized total  : {b['mean_real']:.3f}")
    P(f"    mean diff (pred-real): {b['mean_diff']:+.3f}  "
      f"95% CI [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]")
    P(f"    --> {_verdict(b, res['calib']['slope'])}")

    P("\n[2] PER-LINE UNDER HIT RATES   (+z => unders landed MORE than the model said)")
    P(f"    {'line':>5} {'model P(under)':>15} {'empirical':>11} {'under/N':>9} {'z':>7}")
    for r in res["lines"]:
        P(f"    {r['line']:>5} {r['model_p_under']:>15.3f} {r['emp_under']:>11.3f} "
          f"{str(r['n_under'])+'/'+str(r['n']):>9} {r['z']:>+7.2f}")

    P("\n[3] RELIABILITY (under 2.5)   predicted vs empirical per bin")
    for r in res["reliab"]:
        flag = "  <-- under-predicted" if r["emp"] - r["pred"] > 0.08 else ""
        P(f"    pred {r['pred']:.3f}  emp {r['emp']:.3f}  n={r['n']}{flag}")

    c = res["calib"]
    P("\n[4] CALIBRATION FIT (under 2.5)")
    P(f"    intercept {c['intercept']:+.3f}  (0 = unbiased; >0 = unders under-predicted)")
    P(f"    slope     {c['slope']:.3f}  (1 = perfectly sharp; <1 = overconfident)")

    P("\n[5] ACCURACY")
    for s in res["scores"]:
        P(f"    line {s['line']}: Brier {s['brier']:.4f}  logloss {s['logloss']:.4f}")

    s = res["scale"]
    P("\n[6] RECALIBRATION")
    P(f"    MLE lambda scale c = {s['c']:.3f}   "
      f"({'model too LOW, scale UP' if s['c'] > 1.02 else 'model too HIGH, scale DOWN' if s['c'] < 0.98 else 'already ~unbiased'})")
    P(f"    mean total {s['mean_total_before']:.3f} -> {s['mean_total_after']:.3f} after scaling")
    if "scale_rho" in res:
        sr = res["scale_rho"]
        P(f"    joint refit: c = {sr['c']:.3f}, rho = {sr['rho']:+.3f}")
    if "rho_sens" in res:
        rs = res["rho_sens"]
        P(f"    rho sensitivity @2.5: P(under) {rs['under25_model_with_rho']:.3f} "
          f"-> {rs['under25_model_rho_off']:.3f} with rho off "
          f"(shift {rs['shift']:+.3f})")

    if res.get("boundary"):
        ba = res["boundary"]
        P("\n[7] BOUNDARY AUDIT  (under 2.5 must equal P(total in 0..2))")
        P(f"    example: {ba['example']}")
        P(f"    cells P(total=k): " +
          ", ".join(f"{k}:{v:.3f}" for k, v in ba["cells"].items()))
        for k, v in ba.items():
            if k.startswith("P(") or k.startswith("WRONG"):
                P(f"    {k} = {v:.3f}")

    if plot_path:
        P(f"\nReliability plot written to: {plot_path}")
    P("\n" + "=" * 78)
    return "\n".join(L)


def make_plot(reliab, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    pred = [r["pred"] for r in reliab]
    emp = [r["emp"] for r in reliab]
    ns = [r["n"] for r in reliab]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect")
    ax.scatter(pred, emp, s=[20 + 6 * n for n in ns], color="#c0392b", zorder=3,
               label="under 2.5")
    ax.set_xlabel("model P(under 2.5)")
    ax.set_ylabel("empirical fraction under")
    ax.set_title("Reliability: points above the line = unders under-predicted")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run(matches, plot_path="reliability.png", full=True):
    res = {
        "bias": bias_test(matches),
        "lines": per_line_table(matches),
        "reliab": reliability(matches, 2.5, nbins=5),
        "calib": calibration_slope_intercept(matches, 2.5),
        "scores": [brier_logloss(matches, 2.5), brier_logloss(matches, 3.5)],
        "scale": fit_scale(matches),
        "boundary": boundary_audit(matches, 2.5),
    }
    if full:
        res["scale_rho"] = fit_scale_and_rho(matches)
        res["rho_sens"] = rho_sensitivity(matches)
    if plot_path:
        make_plot(res["reliab"], plot_path)
    return res


# --------------------------------------------------------------------------------------
# Self-test: inject a KNOWN low bias and confirm the toolkit detects + fixes it
# --------------------------------------------------------------------------------------
def selftest(n=400, true_scale=1.0, model_scale=0.85, seed=7):
    """
    Truth-generating lambdas are drawn realistically; the MODEL is fed lambdas that are
    `model_scale` too low (0.85 => 15% low). Realized scores come from TRUTH. A correct
    toolkit must: (a) flag model biased low, (b) show empirical unders < model's claimed
    P(under) ... wait, biased-low model OVER-predicts unders -> empirical unders LOWER
    than model says -> +emp deficit; and (c) recover c ~= true/model = 1/0.85 ~= 1.176.
    """
    rng = np.random.default_rng(seed)
    matches = []
    for i in range(n):
        base = rng.uniform(0.7, 2.2)
        supremacy = rng.normal(0, 0.6)
        lh_true = max(0.15, base + supremacy / 2)
        la_true = max(0.15, base - supremacy / 2)
        rho = -0.05
        hg = rng.poisson(lh_true)
        ag = rng.poisson(la_true)
        matches.append(Match(date=f"2026-06-{(i % 27)+1:02d}", home=f"H{i}", away=f"A{i}",
                             hg=int(hg), ag=int(ag),
                             lh=lh_true * model_scale, la=la_true * model_scale, rho=rho))
    res = run(matches, plot_path="reliability_selftest.png", full=True)
    print(render_report(res, "reliability_selftest.png"))
    print("\nSELF-TEST EXPECTATIONS")
    print(f"  injected model_scale = {model_scale}  =>  expected recovered c ~ "
          f"{1/model_scale:.3f}")
    print(f"  recovered c          = {res['scale']['c']:.3f}")
    ok = abs(res['scale']['c'] - 1 / model_scale) < 0.06 and res['bias']['biased_low']
    print(f"  bias flagged low     = {res['bias']['biased_low']}")
    print(f"  RESULT: {'PASS' if ok else 'CHECK'}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--results", help="martj42 results.csv path or raw URL")
    ap.add_argument("--pred", help="predictions log CSV (Mode A)")
    ap.add_argument("--from-model", action="store_true", help="reconstruct via model/matchup.py (Mode B)")
    ap.add_argument("--as-of-cutoff", action="store_true",
                    help="refit per match with a date cutoff (needs fit_model(cutoff=...))")
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--results-url", action="store_true",
                    help="fetch results.csv from the martj42 raw URL")
    ap.add_argument("--rho", type=float, default=-0.05,
                    help="your model's global rho; used by the recalibration step in "
                         "--from-model when matchup() returns a matrix (default -0.05)")
    ap.add_argument("--plot", default="reliability.png")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    results_path = MARTJ42_URL if args.results_url else args.results
    if not results_path:
        ap.error("need --results PATH (or --results-url)")
    results = load_results(results_path, since=args.since)
    if not results:
        print("No completed FIFA World Cup matches found after "
              f"{args.since}. Nothing to calibrate yet.", file=sys.stderr)
        sys.exit(1)

    if args.from_model:
        preds = predictions_from_model(results, as_of_cutoff=args.as_of_cutoff,
                                       default_rho=args.rho)
    elif args.pred:
        preds = load_predictions(args.pred)
    else:
        ap.error("supply --pred predictions.csv  OR  --from-model")

    matches, missing = join_matches(results, preds)
    if missing:
        print(f"[info] {missing} completed matches had no matching prediction "
              f"(name/date mismatch?) and were skipped.", file=sys.stderr)
    if not matches:
        print("\n[stop] 0 matches joined -- nothing to calibrate. Most likely the "
              "matchup() call shifted args (check the signature echoed above) so every "
              "prediction failed, OR team-name/date keys don't line up between the model "
              "and martj42. Fix the matchup() call in predictions_from_model() and rerun.",
              file=sys.stderr)
        sys.exit(1)
    if len(matches) < 8:
        print(f"[warn] only {len(matches)} matches joined -- calibration will be very "
              f"noisy. Treat as directional, not conclusive.", file=sys.stderr)

    res = run(matches, plot_path=args.plot)
    print(render_report(res, args.plot))


if __name__ == "__main__":
    main()
