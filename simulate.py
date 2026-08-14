"""
simulate.py
===========

Round-of-32-onward Monte Carlo for the 2026 World Cup, driven by the SAME
prediction engine as model/matchup.py -- so every game inherits the EA FC 26
squad-talent correction with zero duplicated model code.

Design
------
1. Single source of truth: every tie is scored by matchup.matchup(), not a
   re-implemented rating blend. Fix the model once, it fixes everywhere.
2. Squad talent: inherited from model/matchup.py as a COEFFICIENT DE-BIAS. The DC
   attack/defence coefficients are pulled toward the global (confederation-
   neutral) squad line at fit time, removing the cross-confederation scale bias
   that was over-crediting non-UEFA defences. This is baked into dc.attack /
   dc.defence, so it flows through here automatically -- nothing to configure.
3. No rotation penalty: a clinched-team haircut only makes sense for dead-rubber
   group finales. There are no dead rubbers in single-elimination.
4. Real bracket: we seed the actual R32 field and bracket slots, and auto-lock
   any games already played (fetch_completed_results reads martj42 live before
   each run, so finished ties are fixed, not simulated -- no hand-editing).
   Since the 2026 tournament is complete, a default run locks all 32 games and
   replays the real bracket; use --no-fetch for the pre-tournament forecast.

Goal-scale
----------
model/matchup.py can measure a uniform total scale from the live tournament, but it is
OFF by default (raw model). For who-advances it mostly acts through the draw rate
(higher totals -> fewer draws -> fewer shootouts), a small effect. Enable it with
--scale (or --scale=1.12 for a manual factor); ablate the de-bias with --no-debias
(alias: --no-squad), and the host away-suppression with --no-split-home.

Bracket order
-------------
R32_BRACKET is 16 consecutive pairs = the 16 R32 ties, laid out so the binary
tree folds cleanly: pairs (0,1) feed an R16 game, (0..3) feed a QF, (0..7) feed
an SF, top half = slots 0-15, bottom half = 16-31. Verified against the official
schedule (R16 pairings, July 4-7).

Usage
-----
    python simulate.py                 # auto-lock results, 50k sims
    python simulate.py --no-fetch      # skip live check (manual LOCKED only)
    python simulate.py --lock="Morocco>Canada"   # force a just-finished result
    python simulate.py --no-debias      # ablate squad de-bias (sanity check)
    python simulate.py --scale          # apply measured total calibration
    python simulate.py --sims=200000   # tighter Monte Carlo error

Live results lag: martj42 is community-updated and trails live games by hours,
so a game you just watched may still be blank in the feed. --lock lets you fix a
result immediately without waiting or editing source; it is overridden by
nothing and wins over the fetched feed.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
from scipy.stats import poisson

from model import matchup as mu   # reuse the production predictor (squad lives here)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SIMS = 50_000
SEED = 42

# Host nations get the fitted home_adv when they meet a non-host (huge home crowds
# across all NA venues). With matchup.SPLIT_HOME on (default), this also suppresses
# the opponent's xG -- fixing the fake OVERS the single-param model produced on host
# opponents. Set to an empty set to treat every knockout game as neutral.
HOSTS = {"United States", "Mexico", "Canada"}

# Penalty shootouts (reached only if extra time is also level): a coin flip.
# Empirically shootouts are ~50/50; nudge this with a shootout model if you like.
SHOOTOUT_P = 0.50


# ---------------------------------------------------------------------------
# The actual Round-of-32 bracket (martj42 names, to match the DC fit)
# ---------------------------------------------------------------------------
# 16 ties, in bracket order. Consecutive pairs fold up the tree.

R32_BRACKET = [
    # ----- TOP HALF (slots 0-15): SF1 = QF1 winner vs QF2 winner -----
    # QF1 branch (France's quarter)
    ("South Africa", "Canada"),                    # R16-A: Canada vs Morocco
    ("Netherlands", "Morocco"),                    # R16-A
    ("Germany", "Paraguay"),                       # R16-B: Paraguay vs France
    ("France", "Sweden"),                          # R16-B
    # QF2 branch (Spain's quarter) -- SAME half as France: they meet in the SF,
    # not the final (verified vs official bracket: SF1 = France/Morocco winner
    # vs Spain/Belgium winner, July 14 Dallas)
    ("Spain", "Austria"),                          # R16-C: Spain vs Portugal
    ("Croatia", "Portugal"),                       # R16-C
    ("Belgium", "Senegal"),                        # R16-D: Belgium vs USA
    ("United States", "Bosnia and Herzegovina"),   # R16-D
    # ----- BOTTOM HALF (slots 16-31): SF2 = QF3 winner vs QF4 winner -----
    # QF3 branch (Norway/England quarter)
    ("Brazil", "Japan"),                           # R16-E: Brazil vs Norway
    ("Ivory Coast", "Norway"),                     # R16-E
    ("Mexico", "Ecuador"),                         # R16-F: Mexico vs England
    ("England", "DR Congo"),                       # R16-F
    # QF4 branch (Argentina/Switzerland quarter)
    ("Australia", "Egypt"),                        # R16-G: Egypt vs Argentina
    ("Argentina", "Cape Verde"),                   # R16-G
    ("Switzerland", "Algeria"),                    # R16-H: Switzerland vs Colombia
    ("Colombia", "Ghana"),                         # R16-H
]

# Manual result overrides -- winners fixed by hand, keyed by the (order-independent)
# pair of teams. You normally don't need to touch this: fetch_completed_results()
# below auto-locks finished games from martj42 before each run. Use this only to
# force a result the dataset hasn't picked up yet (manual entries win over fetched).
# Example:  frozenset({"Germany", "Paraguay"}): "Paraguay",
LOCKED: dict[frozenset, str] = {}

# Auto-lock source: the same martj42 repo the model fits on. results.csv carries
# scores (after extra time); shootouts.csv carries the penalty winner when a
# knockout game is level. KO_START excludes the group stage.
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/shootouts.csv"
)
KO_START = "2026-06-28"   # first Round-of-32 match


def fetch_completed_results(start=KO_START, verbose=True):
    """Return {frozenset({A, B}): winner} for every World Cup knockout game that
    has already finished, read live from martj42.

    A decisive scoreline gives the winner directly; a level game is resolved via
    the penalty winner in shootouts.csv. Games still unplayed (NaN score) or level
    with no shootout recorded yet are skipped, not guessed. Network failure returns
    an empty dict so the sim still runs on whatever is in LOCKED.
    """
    import pandas as pd  # local import: only needed when fetching

    try:
        res = pd.read_csv(RESULTS_URL, parse_dates=["date"])
        sho = pd.read_csv(SHOOTOUTS_URL, parse_dates=["date"])
    except Exception as e:                       # offline / URL moved / parse error
        if verbose:
            print(f"  [results] fetch failed ({e}); using manual LOCKED only")
        return {}

    cutoff = pd.Timestamp(start)
    res = res[(res.date >= cutoff) & (res.tournament == "FIFA World Cup")]
    pens = {
        frozenset({r.home_team, r.away_team}): r.winner
        for r in sho[sho.date >= cutoff].itertuples()
    }

    found: dict[frozenset, str] = {}
    for row in res.itertuples():
        if pd.isna(row.home_score) or pd.isna(row.away_score):
            continue                              # not played yet
        pair = frozenset({row.home_team, row.away_team})
        if row.home_score > row.away_score:
            found[pair] = row.home_team
        elif row.away_score > row.home_score:
            found[pair] = row.away_team
        else:                                     # level after ET -> shootout
            w = pens.get(pair)
            if w is not None:
                found[pair] = w
    if verbose:
        if found:
            for pair, w in found.items():
                a, b = sorted(pair)
                print(f"  [results] {a} vs {b}  ->  {w}")
        else:
            print("  [results] no completed knockout games found")
    return found


# ---------------------------------------------------------------------------
# Prediction: P(t1 advances past t2) for a single knockout game
# ---------------------------------------------------------------------------

_ADV_CACHE: dict[tuple, float] = {}


def _pair_probs(dc, t1, t2):
    """(P(t1 win), P(draw), P(t2 win), xg_t1, xg_t2) at the correct venue.

    Host advantage: if exactly one side is a host, it plays as the home team
    (neutral=False so matchup() adds home_adv); otherwise neutral. The squad
    de-bias is baked into dc.attack/dc.defence by fit_model, and matchup()
    applies the auto-calibrated scale, so both flow through here automatically.
    xG are returned oriented to (t1, t2) so the caller can model extra time.
    """
    if t1 in HOSTS and t2 not in HOSTS:
        r = mu.matchup(dc, t1, t2, neutral=False)
        return r["p_home"], r["p_draw"], r["p_away"], r["xg_home"], r["xg_away"]
    if t2 in HOSTS and t1 not in HOSTS:
        r = mu.matchup(dc, t2, t1, neutral=False)
        return r["p_away"], r["p_draw"], r["p_home"], r["xg_away"], r["xg_home"]
    r = mu.matchup(dc, t1, t2, neutral=True)
    return r["p_home"], r["p_draw"], r["p_away"], r["xg_home"], r["xg_away"]


def _extra_time_adv(xg1, xg2, frac=30.0 / 90.0, max_goals=8):
    """P(t1 advances | level after 90 min).

    Extra time is a 30-minute Poisson mini-match at `frac` of the regulation
    scoring rates; if it's still level, a 50/50 shootout decides it. This gives
    the stronger side its extra-time edge instead of an even coin flip.
    """
    i = np.arange(max_goals + 1)
    pa = poisson.pmf(i, xg1 * frac)
    pb = poisson.pmf(i, xg2 * frac)
    mat = np.outer(pa, pb)
    win1 = float(np.tril(mat, -1).sum())
    draw = float(np.trace(mat))
    return win1 + SHOOTOUT_P * draw


def advance_prob(dc, t1, t2):
    """P(t1 wins the tie): win in 90, else win extra time, else a 50/50 shootout.
    Cached (and the mirror image stored as 1 - p)."""
    locked = LOCKED.get(frozenset({t1, t2}))
    if locked is not None:
        return 1.0 if locked == t1 else 0.0
    key = (t1, t2)
    cached = _ADV_CACHE.get(key)
    if cached is not None:
        return cached
    p1, pd, p2, xg1, xg2 = _pair_probs(dc, t1, t2)
    s = p1 + pd + p2
    p1, pd = p1 / s, pd / s
    out = p1 + pd * _extra_time_adv(xg1, xg2)
    _ADV_CACHE[key] = out
    _ADV_CACHE[(t2, t1)] = 1.0 - out
    return out


# ---------------------------------------------------------------------------
# One simulated tournament
# ---------------------------------------------------------------------------


def play(dc, t1, t2, rng):
    return t1 if rng.random() < advance_prob(dc, t1, t2) else t2


def simulate_once(dc, rng, reached):
    """Play R32 -> Final once; tally how far each team got into `reached`."""
    # Round of 32: 16 ties -> 16 winners
    winners = [play(dc, a, b, rng) for a, b in R32_BRACKET]
    for t in winners:
        reached["r16"][t] += 1

    # Fold the binary tree: R16 -> QF -> SF -> Final -> Champion
    for round_name in ("qf", "sf", "final", "champion"):
        nxt = [play(dc, winners[i], winners[i + 1], rng)
               for i in range(0, len(winners), 2)]
        for t in nxt:
            reached[round_name][t] += 1
        winners = nxt
    return winners[0]  # champion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_locks(flags) -> dict[frozenset, str]:
    """Parse --lock="Winner>Loser" / --lock=Winner:Loser (comma-separated) into
    {frozenset({winner, loser}): winner}. Accepts '>' or ':' as the separator so
    the ':' form survives an unquoted shell."""
    out: dict[frozenset, str] = {}
    for f in flags:
        if not f.startswith("--lock="):
            continue
        for token in f.split("=", 1)[1].split(","):
            token = token.strip()
            sep = ">" if ">" in token else (":" if ":" in token else None)
            if not sep:
                continue
            w, l = (x.strip() for x in token.split(sep, 1))
            if w and l:
                out[frozenset({w, l})] = w
    return out


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    n_sims = N_SIMS
    for f in flags:
        if f.startswith("--sims="):
            n_sims = int(f.split("=", 1)[1])
    # Model knobs (--scale, --no-debias, --debias-weight=, --split-home,
    # --no-split-home) are applied by matchup.apply_cli_flags, so predict.py and
    # simulate.py expose exactly the same options and defaults.
    # --no-squad is kept as an alias for --no-debias (backward compat): the squad
    # talent now enters as a coefficient de-bias baked in by fit_model, not a
    # per-matchup blend, so turning it off means turning off the de-bias.
    model_flags = set(flags)
    if "--no-squad" in model_flags:
        model_flags.add("--no-debias")
    mu.apply_cli_flags(model_flags)

    # Manual CLI locks, for live results martj42 has not committed yet (it lags
    # live games by hours). Syntax: --lock="Winner>Loser" or --lock=Winner:Loser,
    # comma-separated for several. The ':' form needs no shell quoting; the '>'
    # form MUST be quoted or the shell reads it as a redirect. Manual locks take
    # precedence over fetched results (applied before the fetch, which only
    # setdefault()s), so this also lets you override a bad/again-updated feed.
    cli_locks = _parse_locks(flags)
    if any(f.startswith("--lock=") for f in flags) and not cli_locks:
        print("  [lock] WARNING: --lock given but nothing parsed -- quote it, e.g. "
              "--lock=\"Morocco>Canada\"  (or use --lock=Morocco:Canada)")
    bracket_teams = {t for pair in R32_BRACKET for t in pair}
    for pair, w in cli_locks.items():
        bad = [t for t in pair if t not in bracket_teams]
        if bad:
            print(f"  [lock] WARNING: {bad} not in the R32 field -- check spelling "
                  f"(uses martj42 names, e.g. 'United States', 'DR Congo')")
        LOCKED[pair] = w
        a, b = sorted(pair)
        print(f"  [lock] manual: {a} vs {b} -> {w}")

    # Auto-lock any knockout games that have already finished (manual LOCKED wins).
    if "--no-fetch" not in flags:
        print("Checking martj42 for completed knockout results...")
        for pair, winner in fetch_completed_results().items():
            LOCKED.setdefault(pair, winner)

    print("Fitting Dixon-Coles + de-biased squad engine (via model/matchup.py)...")
    dc = mu.fit_model()          # de-biases coefficients; scale off unless --scale
    db = "off" if mu.SQUAD_DEBIAS_WEIGHT == 0 else f"on (w={mu.SQUAD_DEBIAS_WEIGHT})"
    scl = mu.MATCHUP_SCALE or 1.0
    hsplit = (f"  home_def={dc.home_def:.3f} (away-suppression on)"
              if getattr(dc, "home_def", 0.0) else "")
    print(f"  home_adv={dc.home_adv:.3f}  rho={dc.rho:.3f}  "
          f"squad de-bias={db}  scale={scl:.3f}{hsplit}")
    print(f"  locked games: {len(LOCKED)}  |  hosts get home edge: {sorted(HOSTS)}")

    reached = {r: defaultdict(int) for r in ("r16", "qf", "sf", "final", "champion")}
    rng = np.random.default_rng(SEED)

    print(f"\n--- Simulating {n_sims:,} knockout runs from the R32 bracket ---")
    champ = defaultdict(int)
    for _ in range(n_sims):
        champ[simulate_once(dc, rng, reached)] += 1

    # ---- report ----
    all_teams = set()
    for r in reached.values():
        all_teams |= set(r)
    teams = sorted(all_teams, key=lambda t: (-reached["champion"][t],
                                             -reached["final"][t],
                                             -reached["sf"][t]))

    print("\n" + "=" * 86)
    print("2026 WORLD CUP -- KNOCKOUT PROBABILITIES (from current R32 bracket)")
    print("=" * 86)
    print(f"{'Team':<26}{'Champ':>9}{'Final':>9}{'SF':>9}{'QF':>9}{'R16':>9}")
    print("-" * 86)
    for t in teams:
        row = [reached[k][t] / n_sims for k in ("champion", "final", "sf", "qf", "r16")]
        print(f"{t:<26}" + "".join(f"{x:>8.1%} " for x in row))

    p_top = max(champ.values()) / n_sims
    se = (p_top * (1 - p_top) / n_sims) ** 0.5
    print(f"\n(Monte Carlo std error on the top champ prob ~ {se:.3%}. "
          f"Use --sims=200000 to tighten.)")


if __name__ == "__main__":
    main()
