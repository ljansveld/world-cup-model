"""
Stage 3: Cross-validation across 4 World Cups (2010, 2014, 2018, 2022).

For each tournament:
  - Train on everything before that tournament started
  - Evaluate on that tournament

This gives us 256 test matches instead of 64, which makes the noise more
manageable. We report mean log_loss across tournaments and per-tournament
breakdowns so we can see if any model is consistently good vs lucky.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

DATA_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_results() -> pd.DataFrame:
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)

    def label(row):
        if row.home_score > row.away_score:
            return "H"
        if row.home_score < row.away_score:
            return "A"
        return "D"

    df["result"] = df.apply(label, axis=1)
    return df


# ---------------------------------------------------------------------------
# Models (same as before, slightly cleaned up)
# ---------------------------------------------------------------------------


@dataclass
class EloModel:
    base_rating: float = 1500.0
    home_advantage: float = 65.0
    k_friendly: float = 20.0
    k_competitive: float = 40.0
    k_world_cup: float = 60.0
    draw_sigma: float = 200.0
    draw_peak: float = 0.30
    ratings: Dict[str, float] = field(default_factory=dict)

    def rating(self, team):
        return self.ratings.get(team, self.base_rating)

    def _k(self, tournament):
        if tournament == "Friendly":
            return self.k_friendly
        if "World Cup" in tournament:
            return self.k_world_cup
        return self.k_competitive

    @staticmethod
    def _expected(r_a, r_b):
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    @staticmethod
    def _gd_multiplier(gd):
        gd = abs(gd)
        if gd <= 1:
            return 1.0
        if gd == 2:
            return 1.5
        return (11 + gd) / 8.0

    def predict_proba(self, home, away, neutral):
        r_h = self.rating(home) + (0 if neutral else self.home_advantage)
        r_a = self.rating(away)
        diff = r_h - r_a
        p_home_or_draw = self._expected(r_h, r_a)
        p_away_or_draw = 1.0 - p_home_or_draw
        p_draw = self.draw_peak * np.exp(-(diff ** 2) / (2 * self.draw_sigma ** 2))
        p_home = max(p_home_or_draw - p_draw / 2, 1e-6)
        p_away = max(p_away_or_draw - p_draw / 2, 1e-6)
        s = p_home + p_draw + p_away
        return p_home / s, p_draw / s, p_away / s

    def update(self, row):
        home, away = row.home_team, row.away_team
        r_h = self.rating(home) + (0 if row.neutral else self.home_advantage)
        r_a = self.rating(away)
        exp_h = self._expected(r_h, r_a)
        score_h = 1.0 if row.result == "H" else 0.5 if row.result == "D" else 0.0
        k = self._k(row.tournament)
        mult = self._gd_multiplier(row.home_score - row.away_score)
        delta = k * mult * (score_h - exp_h)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    def fit(self, df):
        for row in df.itertuples():
            self.update(row)


@dataclass
class DixonColesModel:
    xi: float = 0.0019
    attack: Dict[str, float] = field(default_factory=dict)
    defence: Dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.25
    home_def: float = 0.0        # away-suppression home term; 0 = single-param (default)
    rho: float = -0.10

    def fit(self, df, ref_date, split_home: bool = False):
        """Fit the model. If split_home, home advantage is split into two terms:
        home_adv (boosts the home team's attack) and home_def (suppresses the
        away team's attack). In-sample the split loads ~80% onto home_def, but
        out-of-sample it is a wash on H/D/A and slightly worse on total goals,
        so it is OFF by default. Left in for host-game shape experiments."""
        teams = sorted(set(df.home_team) | set(df.away_team))
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}
        days = (ref_date - df["date"]).dt.days.values
        weights = np.exp(-self.xi * days)
        home_idx = df["home_team"].map(idx).values
        away_idx = df["away_team"].map(idx).values
        hs = df["home_score"].values.astype(float)
        as_ = df["away_score"].values.astype(float)
        neutral = df["neutral"].values.astype(float)
        not_neutral = 1.0 - neutral
        log_fact_hs = gammaln(hs + 1)
        log_fact_as = gammaln(as_ + 1)

        def neg_log_lik(params):
            atk = params[:n]
            dfn = params[n : 2 * n]
            if split_home:
                home_adv, home_def, rho = params[-3], params[-2], params[-1]
            else:
                home_adv, home_def, rho = params[-2], 0.0, params[-1]
            atk = atk - atk.mean()
            lam = np.exp(atk[home_idx] + dfn[away_idx] + home_adv * not_neutral)
            mu = np.exp(atk[away_idx] + dfn[home_idx] - home_def * not_neutral)
            log_p = (
                hs * np.log(lam) - lam - log_fact_hs
                + as_ * np.log(mu) - mu - log_fact_as
            )
            tau = np.ones_like(lam)
            mask_00 = (hs == 0) & (as_ == 0)
            mask_01 = (hs == 0) & (as_ == 1)
            mask_10 = (hs == 1) & (as_ == 0)
            mask_11 = (hs == 1) & (as_ == 1)
            tau[mask_00] = 1 - lam[mask_00] * mu[mask_00] * rho
            tau[mask_01] = 1 + lam[mask_01] * rho
            tau[mask_10] = 1 + mu[mask_10] * rho
            tau[mask_11] = 1 - rho
            tau = np.clip(tau, 1e-10, None)
            log_p = log_p + np.log(tau)
            return -np.sum(weights * log_p)

        tail = [0.20, 0.05, -0.1] if split_home else [0.25, -0.1]
        x0 = np.concatenate([np.zeros(n), np.zeros(n), tail])
        res = minimize(neg_log_lik, x0, method="L-BFGS-B", options={"maxiter": 200})
        atk = res.x[:n] - res.x[:n].mean()
        dfn = res.x[n : 2 * n]
        self.attack = dict(zip(teams, atk))
        self.defence = dict(zip(teams, dfn))
        if split_home:
            self.home_adv = float(res.x[-3])
            self.home_def = float(res.x[-2])
        else:
            self.home_adv = float(res.x[-2])
            self.home_def = 0.0
        self.rho = float(res.x[-1])

    def predict_proba(self, home, away, neutral, max_goals=10):
        atk_h = self.attack.get(home, 0.0)
        atk_a = self.attack.get(away, 0.0)
        dfn_h = self.defence.get(home, 0.0)
        dfn_a = self.defence.get(away, 0.0)
        lam = float(np.exp(atk_h + dfn_a + (0 if neutral else self.home_adv)))
        mu = float(np.exp(atk_a + dfn_h - (0 if neutral else self.home_def)))
        i = np.arange(max_goals + 1)
        ph = poisson.pmf(i, lam)
        pa = poisson.pmf(i, mu)
        mat = np.outer(ph, pa)
        mat[0, 0] *= 1 - lam * mu * self.rho
        mat[0, 1] *= 1 + lam * self.rho
        mat[1, 0] *= 1 + mu * self.rho
        mat[1, 1] *= 1 - self.rho
        mat = np.clip(mat, 0, None)
        mat /= mat.sum()
        p_home = np.tril(mat, -1).sum()
        p_draw = np.trace(mat)
        p_away = np.triu(mat, 1).sum()
        return float(p_home), float(p_draw), float(p_away)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def log_loss(preds):
    eps = 1e-12
    y_prob = np.where(
        preds.result == "H",
        preds.p_H,
        np.where(preds.result == "D", preds.p_D, preds.p_A),
    )
    return -np.log(np.clip(y_prob, eps, 1)).mean()


def predict_set(model, matches):
    rows = []
    for row in matches.itertuples():
        p_h, p_d, p_a = model.predict_proba(row.home_team, row.away_team, row.neutral)
        rows.append({"result": row.result, "p_H": p_h, "p_D": p_d, "p_A": p_a})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CV harness
# ---------------------------------------------------------------------------


WORLD_CUP_YEARS = [2010, 2014, 2018, 2022]


def cv_evaluate(df: pd.DataFrame, build_model_fn, label: str) -> dict:
    """
    For each World Cup year, train on everything before and predict on it.
    build_model_fn(train_df, ref_date) returns a fitted model with predict_proba.
    Returns per-tournament log_loss and the overall (pooled) log_loss.
    """
    per_tourney = {}
    all_preds = []
    for year in WORLD_CUP_YEARS:
        mask = (df.tournament == "FIFA World Cup") & (df.date.dt.year == year)
        test = df[mask].copy().reset_index(drop=True)
        if len(test) == 0:
            continue
        train = df[df.date < test.date.min()].copy().reset_index(drop=True)
        ref_date = test.date.min()
        model = build_model_fn(train, ref_date)
        preds = predict_set(model, test)
        per_tourney[year] = log_loss(preds)
        all_preds.append(preds)
    pooled = pd.concat(all_preds, ignore_index=True)
    return {
        "model": label,
        "mean_log_loss": float(np.mean(list(per_tourney.values()))),
        "pooled_log_loss": float(log_loss(pooled)),
        **{f"ll_{y}": v for y, v in per_tourney.items()},
    }


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_elo_default(train, ref_date):
    elo = EloModel()
    elo.fit(train)
    return elo


def make_dc_builder(xi: float, lookback_years: int = 4):
    def build(train, ref_date):
        cutoff = ref_date - pd.DateOffset(years=lookback_years)
        sub = train[train.date >= cutoff].copy()
        counts = pd.concat([sub.home_team, sub.away_team]).value_counts()
        valid = set(counts[counts >= 5].index)
        sub = sub[sub.home_team.isin(valid) & sub.away_team.isin(valid)].reset_index(
            drop=True
        )
        dc = DixonColesModel(xi=xi)
        dc.fit(sub, ref_date=ref_date)
        return dc

    return build


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Loading data...")
    df = load_results()
    print(f"  {len(df):,} matches")

    # First: how many matches do we have in each WC year?
    print("\nMatches per World Cup year:")
    for y in WORLD_CUP_YEARS:
        n = ((df.tournament == "FIFA World Cup") & (df.date.dt.year == y)).sum()
        print(f"  {y}: {n}")

    # === Default Elo across CV ===
    print("\n--- Elo (default) across CV ---")
    elo_result = cv_evaluate(df, build_elo_default, "Elo (default)")
    print(elo_result)

    # === Dixon-Coles xi sweep with CV ===
    print("\n--- Dixon-Coles xi sweep across CV ---")
    dc_results = []
    for xi in [0.0005, 0.001, 0.0019, 0.003, 0.005]:
        r = cv_evaluate(df, make_dc_builder(xi), f"DC xi={xi}")
        dc_results.append(r)
        print(
            f"  xi={xi:.4f}  pooled={r['pooled_log_loss']:.4f}  "
            + "  ".join(f"{y}={r[f'll_{y}']:.3f}" for y in WORLD_CUP_YEARS)
        )

    # === Summary table ===
    print("\n" + "=" * 75)
    print("Cross-validated log loss (lower is better)")
    print("=" * 75)
    all_results = [elo_result] + dc_results
    summary = pd.DataFrame(all_results)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nNotes:")
    print("  - mean_log_loss = simple mean across the 4 tournaments")
    print("  - pooled_log_loss = log loss computed on all 256 matches together")
    print("  - These differ when tournaments have very different difficulties")


if __name__ == "__main__":
    main()
