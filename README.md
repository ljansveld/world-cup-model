# 2026 World Cup Match Prediction & Knockout Simulator

A Dixon-Coles goals model for international football, corrected for a structural bias
in how it rates teams across confederations, plus a Monte Carlo simulator that plays
out the 2026 World Cup knockout bracket.

Two things it does:

1. **Predict any single match** — expected goals, W/D/L, the full scoreline
   distribution, over/under lines, winning margin, and both-teams-to-score.
2. **Simulate the tournament** — run the bracket 50,000 times to get each team's
   probability of reaching every round.

```bash
python predict.py "Spain" "France"
python simulate.py --sims=50000
```

---

## The model

### Base layer: Dixon-Coles

Each team gets a fitted **attack** and **defence** coefficient, estimated by maximum
likelihood on every international result in a rolling 4-year window. Expected goals
for a fixture are:

```
xG(i vs j) = exp( att_i + def_j + home_term )
```

Matches are exponentially down-weighted by age (`xi = 0.0019`, chosen by
cross-validation in `model/dixon_coles.py`), and the low-score cells of the scoreline grid get
the Dixon-Coles `rho` correction, which fixes the well-known tendency of a naive
independent-Poisson model to misprice 0-0, 1-0, 0-1 and 1-1.

Every derived market — exact score, totals, margin, BTTS — is read off that one
rho-corrected grid, so the numbers are mutually consistent by construction.

### The problem: coefficients aren't on a shared scale

Dixon-Coles fits all teams jointly, which implicitly assumes the rating scale is tied
together across the whole population. In international football it isn't. Confederations
mostly play *within* themselves, and there are too few inter-confederation matches in a
4-year window to anchor the clusters to each other.

The result is a systematic bias. Measured mean defence residual against a globally-scaled
talent line:

| Confederation | Mean defence residual |
|---|---|
| UEFA | +0.01 |
| CAF | −0.15 |
| CONMEBOL | −0.25 |
| AFC | −0.38 |
| CONCACAF | −0.41 |

Non-UEFA defences are over-credited against the UEFA attack scale they get crossed with.
Concretely: Paraguay's fitted defence ranked 13th in the world, where a global talent
line puts it around 55th. Because `xG(France) = exp(att_France + def_Paraguay)`, that one
inflated coefficient collapsed France's expected goals from ~2.5 to ~1.5.

### The fix: de-bias in coefficient space

`model/squad_debias.py` regresses each fitted coefficient on an independent, globally-scaled
talent measure — squad ratings aggregated from EA FC 26 player data (`data/fc26.csv`) — and
pulls every team's coefficient a fraction `w` toward that single line.

Correcting in **coefficient space** rather than blending final lambdas is the key choice.
A supremacy or lambda blend moves both teams together and can't raise France without
disturbing Paraguay, since they share the match total. Fixing `def_Paraguay` directly
raises France's xG and leaves Paraguay's own attack untouched.

The correction is:
- **surgical** — only coefficients off the line move; well-connected UEFA teams barely budge
- **asymmetric** — attack and defence are corrected separately
- **free** — applied once after the fit, so every prediction inherits it at zero per-call cost

**Out-of-sample results** (2018 + 2022 World Cups, 128 matches, `w = 0.7`), log loss,
lower is better:

| Market | Pure DC | De-biased | Gain |
|---|---|---|---|
| H/D/A | 1.0309 | 0.9687 | +0.062 |
| Team to score | 0.6055 | 0.5897 | +0.016 |
| BTTS | 0.7498 | 0.7320 | +0.018 |
| Over 2.5 | 0.7379 | 0.7101 | +0.028 |

This beat the two alternatives tested — a supremacy blend (+0.042 H/D/A, and *worse* on
team markets) and a per-lambda blend (+0.055) — while being the only one to improve
every market simultaneously.

### Home advantage, split

Host nations get the fitted home advantage, but split into two terms: a boost to the
host's attack (`home_adv`) and a suppression of the opponent's (`home_def`). The data puts
roughly 80% of the effect in away-suppression. A single-parameter home term applies no
suppression at all to the away side, which manufactures spurious high totals in host
matches. This only affects non-neutral games, so at a World Cup it changes host fixtures
and leaves every neutral fixture identical.

---

## Layout

```
predict.py              CLI: one fixture
simulate.py             CLI: Monte Carlo over the knockout bracket
model/                  the library
├── matchup.py          fits the model, applies the de-bias, exposes matchup()
├── dixon_coles.py      DixonColesModel + EloModel, and the CV harness that picked xi
├── squad_debias.py     the cross-confederation scale correction
├── squad_strength.py   aggregates player data to per-nation talent ratings
├── squad_ratings.py    loads the squad CSV from data/
└── paths.py            repo-relative paths, so cwd never matters
data/
└── fc26.csv            EA FC 26 player data — the talent anchor
tools/
├── scrape_sofifa.py    builds data/fc26.csv from SoFIFA
└── ingest_squad_csv.py adapts any Kaggle / EA FC player CSV to the expected schema
validation/             the evidence behind the modelling choices
├── squad_blend_cv.py       does squad strength help out-of-sample? (WC 2018 + 2022)
├── debias_shrinkage_cv.py  is residual shrinkage better than a flat de-bias weight?
├── debias_diagnostic.py    what does the de-bias do to one team's coefficient?
└── calibrate.py            are the goal totals calibrated? fits the fix if not
docs/
└── calibration.md      how to read the calibration report
```

Everything resolves paths through `model/paths.py`, so any script runs correctly from
any directory. The scripts in `tools/` and `validation/` work both as
`python validation/calibrate.py` and `python -m validation.calibrate`.

Match results are pulled live from the
[martj42 international results](https://github.com/martj42/international_results)
dataset, so there is no results file to maintain. The historical FIFA editions the
validation folds need (~10MB each) download into `data/` on first use rather than being
committed.

---

## Usage

Python 3.10+.

```bash
pip install -r requirements.txt
```

The core model needs only `numpy`, `pandas`, `scipy` and `requests`. `matplotlib` is
optional (the calibration report skips its plot without it) and `beautifulsoup4` is only
needed to re-run the scraper.

### Single match

```bash
python predict.py "Spain" "France"
```

```
Spain  vs  France   (neutral)
----------------------------------------------
  expected goals : Spain 1.23  -  1.38 France
  squad strength : Spain 86.5  -  86.7 France   (coefficients de-biased, w=0.7)
  win / draw / win: 32.1%  /  28.7%  /  39.2%
  most likely scores:
      1-1   13.7%
      0-1    8.8%
      1-2    8.6%
  over / under goal lines:
      line           Spain        France     match total
      O/U 2.5       13/87         16/84           49/51
  both teams to score: yes 54.4%  /  no 45.6%
```

Team names follow the martj42 dataset (`"United States"`, `"South Korea"`).

Flags:

| Flag | Effect |
|---|---|
| `--home` | Give team 1 home advantage instead of a neutral venue. |
| `--no-debias` | Ablate the confederation correction (sanity check). |
| `--debias-weight=0.5` | Change the pull strength (default 0.7). |
| `--scale` | Optional: measure a uniform goal-level factor from completed games and apply it. Off by default. |
| `--scale=1.12` | Optional: set that goal-level factor by hand. |
| `--no-split-home` | Revert to a single-parameter home advantage. |

### Knockout simulation

```bash
python simulate.py --no-fetch --sims=50000
```

```
Team                          Champ    Final       SF       QF      R16
Argentina                    18.4%    33.1%    57.1%    80.3%    91.8%
France                       15.0%    24.1%    37.1%    54.3%    79.3%
Spain                        10.5%    18.5%    31.9%    46.6%    75.5%
Brazil                       10.5%    18.9%    31.8%    52.8%    73.1%
England                       9.4%    17.7%    30.6%    51.7%    77.1%
```

Each tie is drawn from the same rho-corrected scoreline grid the single-match predictor
uses; level ties go to extra time (a scaled continuation of the same rates) and then to a
coin-flip shootout.

By default the simulator fetches completed results and **locks** them rather than
simulating them, so probabilities stay honest mid-tournament. Since the 2026 tournament is
now complete, the default run locks all 32 knockout games and returns a finished bracket —
use `--no-fetch` to see the model's actual pre-tournament forecast from the R32 field.

| Flag | Effect |
|---|---|
| `--no-fetch` | Don't lock live results; simulate the whole bracket. |
| `--sims=200000` | More runs, tighter Monte Carlo error. |
| `--lock="Morocco>Canada"` | Force a just-finished result the feed hasn't picked up yet. |
| `--no-debias`, `--debias-weight=`, `--scale`, `--no-split-home` | Passed through to the model. |

### Reproducing the validation

```bash
python validation/squad_blend_cv.py      # the out-of-sample squad-signal test
python validation/debias_diagnostic.py   # what the de-bias does to one fixture
python validation/calibrate.py --selftest # verify the calibration math on synthetic data
```

`squad_blend_cv.py` downloads the FIFA 2019 and 2021 player datasets into `data/` on first
run (~10MB each) and refits the model as of the eve of each tournament, so it takes a few
minutes. It prints the pooled result the README table is drawn from. See
[docs/calibration.md](docs/calibration.md) for the calibration harness.

---

## Notes and limitations

- The de-bias weight `w = 0.7` was validated on 128 matches (WC 2018 + 2022). That's the
  full set of folds where both squad data and a World Cup exist, but it is a small sample,
  and `w` is deliberately held at the conservative end of the 0.7–0.8 CV-optimal range.
- The talent anchor is video-game ratings. They are a genuinely independent, globally-scaled
  signal, which is exactly what the correction needs — but they carry their own biases
  (players in less-watched domestic leagues get cheaper cards), which is why the correction
  is a partial pull toward the line rather than a replacement of the fitted coefficient.
- Goal *level* (how many goals a fixture produces) and goal *shape* (which side gets them)
  are separate axes. The de-bias acts on shape, and measurably improved the totals markets
  too — Over 2.5 log loss fell from 0.7379 to 0.7101 on the validation folds. `--scale` is
  the independent lever on level; it is off by default and the model is not fit assuming
  any standing correction.
- Shootouts are modelled as coin flips. Attempts to predict them from team strength are
  not well supported by the historical record.
