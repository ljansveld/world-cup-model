"""
Stage 12: squad strength features from FIFA video game player data.

For each (country, FIFA edition year), aggregate the player ratings to produce
team-strength signals. Standard approaches in the literature:

  squad_top11_mean  : mean overall of top 11 players (starting XI proxy)
  squad_top23_mean  : mean overall of top 23 (full match-day squad)
  squad_top11_max   : best player's rating (star quality)
  squad_attack      : mean overall of top 4 forward-position players
  squad_defense     : mean overall of top 4 defender-position players
  squad_depth       : std of top 23 (high = top-heavy, low = balanced)

These give us multiple views of squad quality that aren't perfectly correlated.
The hypothesis is that these features add orthogonal information to Elo
because Elo only sees match results, not the underlying player quality.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import requests

from model.paths import data_path, ensure_data_dir


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

FIFA_DATASETS = {
    # historical editions, used by the cross-validation folds (WC 2018 / 2022)
    2019: "https://raw.githubusercontent.com/4m4n5/fifa18-all-player-statistics/master/2019/data.csv",
    2021: "https://raw.githubusercontent.com/4m4n5/fifa18-all-player-statistics/master/2021/data.csv",
}


def load_fifa(year: int) -> pd.DataFrame:
    """Load a historical FIFA edition, downloading it into data/ on first use.

    These are only needed by the validation folds -- the live model reads
    data/fc26.csv instead -- so they are fetched on demand rather than committed
    (they are ~10MB each). Cached after the first call.
    """
    if year not in FIFA_DATASETS:
        raise KeyError(f"no dataset URL for FIFA {year}; known: {sorted(FIFA_DATASETS)}")
    ensure_data_dir()
    local_path = data_path(f"fifa{year}.csv")
    if not local_path.exists():
        print(f"[squad_strength] downloading FIFA {year} player data -> {local_path} ...")
        resp = requests.get(FIFA_DATASETS[year], timeout=120)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
    return pd.read_csv(local_path, low_memory=False)


# ---------------------------------------------------------------------------
# Nationality normalization to match our match-results dataset
# ---------------------------------------------------------------------------

NATION_MAP = {
    "Korea Republic": "South Korea",
    "Korea, Republic of": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "Republic of Ireland": "Republic of Ireland",
    "Ireland": "Republic of Ireland",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Cote d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Iran": "Iran",
    "FYR Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    "Turkey": "Turkey",
}


def normalize_nation(n: str) -> str:
    return NATION_MAP.get(n, n)


# ---------------------------------------------------------------------------
# Position classification (the FIFA datasets use sometimes-cryptic codes)
# ---------------------------------------------------------------------------


def classify_position(pos):
    """Return 'GK', 'DEF', 'MID', or 'FWD'."""
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return "MID"  # default
    p = str(pos).upper().strip()
    if "GK" in p:
        return "GK"
    if any(t in p for t in ["CB", "RB", "LB", "RWB", "LWB", "SW", "DEF"]):
        return "DEF"
    if any(t in p for t in ["ST", "CF", "LW", "RW", "LF", "RF", "FWD"]):
        return "FWD"
    return "MID"


# ---------------------------------------------------------------------------
# Compute per-country squad strength from a FIFA dataset
# ---------------------------------------------------------------------------


def compute_squad_strength(fifa_df: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame indexed by country with squad strength features."""
    df = fifa_df.copy()
    # The 4m4n5 datasets use these column names
    df = df.rename(columns={
        "Nationality": "nationality",
        "Overall": "overall",
        "Position": "position",
    })
    df["nationality"] = df["nationality"].astype(str).map(normalize_nation)
    df["pos_class"] = df["position"].apply(classify_position)

    out_rows = []
    for nation, sub in df.groupby("nationality"):
        sub = sub.sort_values("overall", ascending=False)
        top23 = sub.head(23)
        top11 = sub.head(11)

        forwards = sub[sub.pos_class == "FWD"].head(4)
        defenders = sub[sub.pos_class == "DEF"].head(4)
        midfielders = sub[sub.pos_class == "MID"].head(4)

        out_rows.append({
            "nationality": nation,
            "squad_top11_mean": top11["overall"].mean() if len(top11) > 0 else np.nan,
            "squad_top23_mean": top23["overall"].mean() if len(top23) > 0 else np.nan,
            "squad_top11_max": top11["overall"].max() if len(top11) > 0 else np.nan,
            "squad_attack": forwards["overall"].mean() if len(forwards) > 0 else np.nan,
            "squad_defense": defenders["overall"].mean() if len(defenders) > 0 else np.nan,
            "squad_midfield": midfielders["overall"].mean() if len(midfielders) > 0 else np.nan,
            "squad_depth_std": top23["overall"].std() if len(top23) >= 2 else np.nan,
            "squad_n_players": len(sub),
        })
    return pd.DataFrame(out_rows).set_index("nationality")


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------


def main():
    print("Loading FIFA player data...")
    for year in [2019, 2021]:
        print(f"\n=== FIFA {year} ===")
        fifa = load_fifa(year)
        print(f"  loaded {len(fifa)} players")
        strength = compute_squad_strength(fifa)
        print(f"  computed squad strength for {len(strength)} nations")

        # Show top 15 by top11 mean
        print(f"\n  Top 15 nations by top-11 mean rating (FIFA {year}):")
        top = strength.sort_values("squad_top11_mean", ascending=False).head(15)
        for nation, row in top.iterrows():
            print(
                f"    {nation:<22} top11={row.squad_top11_mean:.1f} "
                f"top23={row.squad_top23_mean:.1f} "
                f"best={row.squad_top11_max:.0f} "
                f"atk={row.squad_attack:.1f} "
                f"def={row.squad_defense:.1f} "
                f"(n={int(row.squad_n_players)})"
            )

        # Save for reuse
        out = data_path(f"squad_strength_{year}.csv")
        strength.to_csv(out)
        print(f"  saved to {out}")


if __name__ == "__main__":
    main()
