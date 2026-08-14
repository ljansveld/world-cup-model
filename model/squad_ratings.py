"""
model/squad_ratings.py — squad-strength talent ratings for the matchup model.

Loads a FIFA/EA-FC player CSV from data/ (default fc26.csv, falling back to the
newest edition present), aggregates it to a per-nation top-11 overall rating via
squad_strength.compute_squad_strength, and estimates the empirical "goals per
rating point" slope from recent results.

This is the orthogonal talent signal that pure Dixon-Coles and Elo both lack, and
the globally-scaled anchor the confederation de-bias regresses against — see
model/squad_debias.py. Cross-validation on the 2018 and 2022 World Cups
(validation/squad_blend_cv.py) is what established the signal is real.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from model.squad_strength import compute_squad_strength
from model.paths import data_path

# Searched in order inside data/; first non-empty file wins. fc26.csv = current
# tournament. The fifa20xx names match what squad_strength.load_fifa() writes.
SQUAD_CSV_CANDIDATES = ["fc26.csv", "fc26_clean.csv", "fifa2021.csv", "fifa2019.csv"]


class SquadRatings:
    def __init__(self, csv: str | None = None):
        path = csv or self._find()
        if path is None:
            raise FileNotFoundError(
                "no squad CSV found in data/ (looked for "
                + ", ".join(SQUAD_CSV_CANDIDATES) + "). "
                "Run tools/scrape_sofifa.py to build fc26.csv, or drop a "
                "Kaggle EA-FC CSV through tools/ingest_squad_csv.py.")
        raw = pd.read_csv(path, low_memory=False)
        s = compute_squad_strength(raw)
        self.rating = s["squad_top11_mean"].dropna().to_dict()
        self.counts = s["squad_n_players"].to_dict()
        self.source = str(path)

    @staticmethod
    def _find():
        """First candidate present in data/. Resolved relative to the repo root,
        so it does not matter which directory you run from."""
        for c in SQUAD_CSV_CANDIDATES:
            p = data_path(c)
            if p.exists() and p.stat().st_size > 100:
                return p
        return None

    def get(self, team):
        return self.rating.get(team)

    def coverage(self, team):
        return int(self.counts.get(team, 0))


def estimate_squad_slope(df: pd.DataFrame, rating: dict,
                         xi: float, lookback_years: int,
                         ref_date) -> float:
    """Least-squares slope (through origin) mapping a squad top-11 rating
    differential to realised goal supremacy, over the same recent window the
    Dixon-Coles fit uses. Typically ~0.10-0.11 goals per rating point."""
    cutoff = ref_date - pd.DateOffset(years=lookback_years)
    sub = df[(df.date >= cutoff) & (df.date <= ref_date)]
    x, y = [], []
    for r in sub.itertuples():
        if r.home_team in rating and r.away_team in rating:
            x.append(rating[r.home_team] - rating[r.away_team])
            y.append(r.home_score - r.away_score)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 50 or (x @ x) == 0:
        return 0.0
    return float((x @ y) / (x @ x))
