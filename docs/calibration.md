# Totals calibration harness

Supplementary doc for `validation/calibrate.py`. See [README.md](README.md) for the project itself.

Answers one question with evidence: **when the model says P(under X.5) = p, do totals
actually land under that line ~p of the time?**

This is a distinct failure mode from getting matchups wrong. A model can rank teams
correctly and still get the *level* of scoring wrong, which quietly distorts every total,
BTTS and correct-score number it produces without ever showing up in W/D/L accuracy.
If the level is off, this script also fits the correction: a single scale factor `c` on
both lambdas.

Runs locally next to `model/matchup.py` / `model/dixon_coles.py`. Realized scores come from the same
martj42 `results.csv` the model trains on, so there is no separate results file.

## Quick start

```bash
# 0) sanity-check the math on synthetic data with a KNOWN injected bias (no model needed)
python validation/calibrate.py --selftest

# 1a) PREFERRED: calibrate the predictions you actually logged (zero leakage)
python validation/calibrate.py --pred predictions.csv --results results.csv

# 1b) OR reconstruct predictions by re-running the model (read the leakage caveat)
python validation/calibrate.py --from-model --results results.csv
```

`--results` takes a local path or the raw martj42 URL (or pass `--results-url`).

## Input format

**predictions.csv** (Mode A — what the model said, one row per game):

```
date,home_team,away_team,lambda_home,lambda_away,rho
2026-06-24,Switzerland,Canada,1.32,1.04,-0.06
```

`lambda_*` are expected goals per side; `rho` is the DC low-score correction (a constant
column is fine if the fit uses one global rho). To dump these from the model in one pass:

```python
import matchup, csv

dc = matchup.fit_model()
fixtures = [("Switzerland", "Canada", "2026-06-24"), ...]   # completed games

with open("predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["date", "home_team", "away_team", "lambda_home", "lambda_away", "rho"])
    for h, a, d in fixtures:
        m = matchup.matchup(dc, h, a)
        w.writerow([d, h, a, m["xg_home"], m["xg_away"], dc.rho])
```

## Reading the output

| Section | How to read it |
|---|---|
| **[1] Total-goals bias** | The headline. `mean diff (pred − real)` with the whole 95% CI **below zero** ⇒ the model under-predicts goals and over-states P(under). CI straddling zero ⇒ totals are sound. |
| **[2] Per-line table** | `z` is how far empirical unders sit from the model's P(under). All-negative z is the fingerprint of a model running low. |
| **[3] Reliability + plot** | Points above the diagonal = unders under-predicted. Writes `reliability.png`. |
| **[4] Calibration fit** | Intercept ≠ 0 = directional bias; slope < 1 = overconfident. |
| **[6] Recalibration** | `c` is the MLE scale on both lambdas. `c > 1` ⇒ the model was low by that factor. This is what `model/matchup.py --scale` applies. A joint `(c, rho)` fit and a "rho off" pass are also reported, so you can see whether a too-negative rho is contributing. |
| **[7] Boundary audit** | Confirms `under 2.5 == P(total ∈ {0,1,2})` — catches off-by-one errors in how a half-goal line is read off the discrete grid. |

## Two caveats that matter

1. **Leakage (Mode B only).** `model/matchup.py` fits once on all data with no cutoff, so
   reconstructing a match's prediction *now* includes that match's own result in training
   and makes the model look better-calibrated than it was. Prefer Mode A (log predictions
   as you make them), or add a `cutoff=` argument to `fit_model` and run `--as-of-cutoff`
   for a true out-of-sample read.
2. **Sample size.** Early in a tournament the CIs are wide (the self-test shows `c` going
   noisy around n ≈ 60). Treat results as directional until most of the group stage is in;
   the *sign* of the bias stabilizes well before its magnitude does.

## Where Elo fits, and where it doesn't

Decompose a prediction into **supremacy** (expected goal *difference*) and **expectancy**
(expected goal *total*). A totals bias is an expectancy problem — fix it with the scale
factor `c`, not with Elo, which informs supremacy and is near-silent on the total.

Elo is still worth considering for a separate reason: international Dixon-Coles is poorly
identified across confederations that rarely meet, and Elo propagates strength across those
sparse links. That is the same structural problem the squad de-bias attacks, and on the
2018/2022 folds the de-bias won (see `validation/squad_blend_cv.py` and the README). Keep the two jobs
separate: `c` fixes the level, the de-bias fixes the shape.
