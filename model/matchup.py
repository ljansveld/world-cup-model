"""
model/matchup.py

Expected goals (and full scoreline / W-D-L) for any World Cup matchup, straight
from the offence/defence coefficients of your Dixon-Coles fit.

Core formula (exactly what DixonColesModel.predict_proba uses internally):

    xG(i vs j) = exp( att_i + def_j + home_term )

      att_i      = team i's attack coefficient   (offence)
      def_j      = team j's defence coefficient   (defence)
      home_term  = 0 at a neutral venue; = home_adv if i is the home side

This refits the model once (xi=0.0019, no shrinkage -- the CV-best config) so it
has att, def, home_adv and rho all consistent, then exposes matchup().

matchup() returns the full rho-corrected scoreline grid, so every derived market
-- W/D/L, exact score, team and match totals, winning margin, BTTS -- is read off
one internally consistent distribution.

This is the LIBRARY, not the command line. simulate.py and the validation scripts
import it; predict.py is the CLI wrapper around main() below. Use:

    from model.matchup import fit_model, matchup

    dc = fit_model()                      # fit + de-bias, once (slow)
    r = matchup(dc, "Spain", "France")    # cheap per fixture thereafter

Run it as `python predict.py "Spain" "France"` (or `python -m model.matchup ...`).
Running `python model/matchup.py` directly will NOT work -- as a module inside the
package its own imports need the repo root on sys.path.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy.stats import poisson

from model.dixon_coles import DixonColesModel, load_results
from model.squad_ratings import SquadRatings
from model.squad_debias import debias_coefficients, estimate_total_calibration

# --- cross-confederation scale de-bias (see model/squad_debias.py) ----------------
# The DC attack/defence coefficients are not on a shared cross-confederation
# scale (too few inter-confederation games to tie the clusters together), so
# non-UEFA defences get systematically over-credited against the UEFA attack
# scale they are crossed with. That single-coefficient bias -- e.g. Paraguay's
# defence rated world-13 when the global squad line says ~world-55 -- is what
# collapsed France's xG from ~2.5 to 1.5 and dragged its match totals down with it.
# We correct it ONCE, in coefficient space, right after the DC fit: pull each
# team's att/def a fraction SQUAD_DEBIAS_WEIGHT toward the global squad-implied
# line. Surgical (only off-line/biased teams move), asymmetric (att & def fixed
# separately, so an over-credited opponent defence is corrected without touching
# the underdog's own scoring), and free per matchup (coefficients are already
# fixed). CV (2018+2022) best ~0.7-0.8; kept conservative at 0.7 given n=128.
SQUAD = None                 # SquadRatings, for the ratings CSV path / coverage
SQUAD_DEBIAS_WEIGHT = 0.7
SQUAD_COV_FULL = 12          # (kept for back-compat with any callers)

# Uniform total calibration -- the LEVEL knob, independent of the de-bias above
# (which acts on SHAPE: which side gets the goals). OFF by default; the raw fit
# is what you get, and nothing downstream assumes a standing correction.
# It exists because the goal environment of a given tournament can run above or
# below the four-year training window, and that is a level effect no rating
# correction addresses. Pass --scale to measure the factor from the completed
# games of the live tournament (see estimate_total_calibration), or --scale=1.12
# to set one by hand.
# Internally: 1.0 = off, None = "measure it at fit time", a number = that factor.
MATCHUP_SCALE = 1.0

# Split home advantage into home_adv (home attack boost) + home_def (away
# suppression). ON by default: live host-game betting showed the single-param
# model manufacturing fake OVERS on host opponents, because it applied zero
# suppression to the away side. The data backs this -- the effect is ~80% away-
# suppression, and stronger still in competitive games (home_def~0.31, away xG
# x0.74) like WC host matches. Crucially this only touches NON-NEUTRAL games, so
# at the WC it changes host games ONLY and leaves every neutral game identical.
# (For friendly-heavy non-WC use it can mildly over-suppress; disable with
# --no-split-home there.)
SPLIT_HOME = True

XI, LOOKBACK_YEARS, MIN_MATCHES = 0.0019, 4, 5
MAX_GOALS = 10


def fit_model() -> DixonColesModel:
    df = load_results()
    ref = df.date.max()
    cutoff = ref - pd.DateOffset(years=LOOKBACK_YEARS)
    sub = df[(df.date >= cutoff) & (df.date <= ref)].copy()
    counts = pd.concat([sub.home_team, sub.away_team]).value_counts()
    valid = set(counts[counts >= MIN_MATCHES].index)
    sub = sub[sub.home_team.isin(valid) & sub.away_team.isin(valid)].reset_index(drop=True)
    dc = DixonColesModel(xi=XI)
    dc.fit(sub, ref_date=ref, split_home=globals().get("SPLIT_HOME", False))

    # remove cross-confederation scale bias from the coefficients, in place
    global SQUAD
    if SQUAD_DEBIAS_WEIGHT > 0:
        try:
            SQUAD = SquadRatings()
            debias_coefficients(dc, SQUAD.source, weight=SQUAD_DEBIAS_WEIGHT)
        except Exception as e:
            print(f"[debias] disabled ({e})")
            SQUAD = None

    # calibrate the uniform total scale only if requested (MATCHUP_SCALE is None,
    # set by --scale). Default is 1.0 (off). Done AFTER de-bias so it measures the
    # corrected coefficients against the live tournament.
    global MATCHUP_SCALE
    if MATCHUP_SCALE is None:
        try:
            MATCHUP_SCALE = estimate_total_calibration(dc, df, ref, xi=XI,
                                                       lookback_years=LOOKBACK_YEARS)
        except Exception as e:
            print(f"[scale] calibration failed ({e}); using 1.0")
            MATCHUP_SCALE = 1.0
    return dc


def matchup(dc: DixonColesModel, home: str, away: str, neutral: bool = True,
            scale: float = None):
    """Return xG for both sides plus the Dixon-Coles scoreline distribution.

    `scale` multiplies both xG values -- an optional, uniform adjustment to the goal
    LEVEL, off by default (1.0). Use it when a tournament's scoring environment runs
    above or below the training window. When None, it reads the module-level
    MATCHUP_SCALE (set by the --scale CLI flag), so it flows through report() and the
    knockout simulator automatically. Every downstream market -- matrix, totals,
    spread, BTTS -- is built from these xG values, so all inherit it.
    """
    for t in (home, away):
        if t not in dc.attack:
            raise KeyError(f"'{t}' not in the fitted ratings. Check spelling "
                           f"(uses martj42 names, e.g. 'United States', 'South Korea').")

    home_term = 0.0 if neutral else dc.home_adv
    away_suppress = 0.0 if neutral else dc.home_def   # 0 unless --split-home fit
    xg_home = float(np.exp(dc.attack[home] + dc.defence[away] + home_term))
    xg_away = float(np.exp(dc.attack[away] + dc.defence[home] - away_suppress))

    # dc.attack / dc.defence are already de-biased in fit_model() (squad_debias),
    # so the confederation-scale correction is baked into the coefficients and
    # every downstream market inherits it. --scale is the separate, optional
    # LEVEL knob and is 1.0 (no-op) unless explicitly requested.
    if scale is None:
        scale = globals().get("MATCHUP_SCALE") or 1.0
    xg_home *= scale
    xg_away *= scale

    # full scoreline grid with the Dixon-Coles low-score (rho) correction
    i = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(i, xg_home)
    pa = poisson.pmf(i, xg_away)
    mat = np.outer(ph, pa)
    rho = dc.rho
    mat[0, 0] *= 1 - xg_home * xg_away * rho
    mat[0, 1] *= 1 + xg_home * rho
    mat[1, 0] *= 1 + xg_away * rho
    mat[1, 1] *= 1 - rho
    mat = np.clip(mat, 0, None)
    mat /= mat.sum()

    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())

    # most likely scorelines
    flat = [((h, a), mat[h, a]) for h in range(MAX_GOALS + 1) for a in range(MAX_GOALS + 1)]
    flat.sort(key=lambda x: -x[1])
    top_scores = flat[:5]

    # goal-line over/under, from the marginals of the (rho-corrected) grid
    home_goals = mat.sum(axis=1)          # P(home team scores k)
    away_goals = mat.sum(axis=0)          # P(away team scores k)
    total_goals = np.zeros(2 * MAX_GOALS + 1)
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            total_goals[h + a] += mat[h, a]

    def over_under(marg, lines=(0.5, 1.5, 2.5)):
        cdf = np.cumsum(marg)
        return {L: (float(1 - cdf[int(np.floor(L))]), float(cdf[int(np.floor(L))]))
                for L in lines}   # (over, under)

    ou = {
        "home": over_under(home_goals),
        "away": over_under(away_goals),
        "total": over_under(total_goals),
    }

    # winning margin (spread / handicap): distribution of home_goals - away_goals
    margin = np.zeros(2 * MAX_GOALS + 1)
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            margin[h - a + MAX_GOALS] += mat[h, a]

    def win_by(side, ks=(1, 2, 3)):
        out = {}
        for k in ks:
            if side == "home":
                out[k] = float(margin[k + MAX_GOALS:].sum())       # h - a >= k
            else:
                out[k] = float(margin[: MAX_GOALS - k + 1].sum())  # a - h >= k
        return out

    spread = {"home": win_by("home"), "away": win_by("away")}

    # both teams to score: 1 - P(home=0) - P(away=0) + P(0-0)
    btts_yes = float(1 - home_goals[0] - away_goals[0] + mat[0, 0])

    return {
        "xg_home": xg_home, "xg_away": xg_away,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "top_scores": top_scores, "ou": ou, "spread": spread,
        "btts": btts_yes, "matrix": mat,
    }


def report(dc, home, away, neutral=True):
    r = matchup(dc, home, away, neutral)
    venue = "neutral" if neutral else f"{home} at home"
    print(f"\n{home}  vs  {away}   ({venue})")
    print("-" * 46)
    print(f"  expected goals : {home} {r['xg_home']:.2f}  -  {r['xg_away']:.2f} {away}")
    _sq = globals().get("SQUAD")
    _w = globals().get("SQUAD_DEBIAS_WEIGHT", 0)
    if _sq is not None and _w > 0 and home in _sq.rating and away in _sq.rating:
        _cov = min(_sq.coverage(home), _sq.coverage(away))
        _thin = "  [thin squad data]" if _cov < globals().get("SQUAD_COV_FULL", 12) else ""
        print(f"  squad strength : {home} {_sq.rating[home]:.1f}  -  "
              f"{_sq.rating[away]:.1f} {away}   (coefficients de-biased toward "
              f"the global squad line, w={_w}){_thin}")
    print(f"  win / draw / win: {r['p_home']*100:4.1f}%  /  {r['p_draw']*100:4.1f}%  /  {r['p_away']*100:4.1f}%")
    print(f"  most likely scores:")
    for (h, a), p in r["top_scores"]:
        print(f"      {h}-{a}   {p*100:4.1f}%")

    ou = r["ou"]
    h_lab = home[:12]
    a_lab = away[:12]
    print(f"  over / under goal lines:")
    print(f"      {'line':<6}{h_lab:>14}{a_lab:>14}{'match total':>16}")
    for L in (0.5, 1.5, 2.5):
        ho, hu = ou["home"][L]
        ao, au = ou["away"][L]
        to, tu = ou["total"][L]
        print(f"      {('O/U '+str(L)):<6}"
              f"{f'{ho*100:4.0f}/{hu*100:<4.0f}':>14}"
              f"{f'{ao*100:4.0f}/{au*100:<4.0f}':>14}"
              f"{f'{to*100:4.0f}/{tu*100:<4.0f}':>16}")

    sp = r["spread"]
    print(f"  winning margin (team wins by N+ goals):")
    print(f"      {'margin':<10}{h_lab:>12}{a_lab:>14}")
    labels = {1: "1+ (win)", 2: "2+ goals", 3: "3+ goals"}
    for k in (1, 2, 3):
        hv = f"{sp['home'][k]*100:4.1f}%"
        av = f"{sp['away'][k]*100:4.1f}%"
        print(f"      {labels[k]:<10}{hv:>12}{av:>14}")

    btts = r["btts"]
    print(f"  both teams to score: yes {btts*100:4.1f}%  /  no {(1-btts)*100:4.1f}%")



def apply_cli_flags(flags) -> None:
    """Apply the shared model flags to this module's globals.

    Shared with simulate.py so both entry points expose the same knobs with the
    same names and defaults.
    """
    global MATCHUP_SCALE, SQUAD_DEBIAS_WEIGHT, SPLIT_HOME

    # Total calibration is OFF by default (raw model, scale=1.0).
    #   --scale        -> measure it from data at fit time (live-WC regime).
    #   --scale=1.12   -> manual factor.
    for f in flags:
        if f == "--scale":
            MATCHUP_SCALE = None          # None => fit_model measures it
        elif f.startswith("--scale="):
            MATCHUP_SCALE = float(f.split("=", 1)[1])
    if MATCHUP_SCALE is not None and MATCHUP_SCALE != 1.0:
        print(f"[scale] manual override: xG multiplied by {MATCHUP_SCALE}")

    for f in flags:
        if f == "--no-debias":
            SQUAD_DEBIAS_WEIGHT = 0.0
        elif f.startswith("--debias-weight="):
            SQUAD_DEBIAS_WEIGHT = float(f.split("=", 1)[1])
    if "--split-home" in flags:
        SPLIT_HOME = True
    if "--no-split-home" in flags:
        SPLIT_HOME = False


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    neutral = "--home" not in flags

    apply_cli_flags(flags)

    dc = fit_model()
    if len(args) >= 2:
        report(dc, args[0], args[1], neutral=neutral)
    else:
        # demo a few matchups
        print(f"(fitted home_adv={dc.home_adv:.3f}, rho={dc.rho:.3f})")
        report(dc, "Spain", "France")
        report(dc, "Argentina", "Brazil")
        report(dc, "Germany", "Japan")
        report(dc, "United States", "Norway", neutral=False)


if __name__ == "__main__":
    main()
