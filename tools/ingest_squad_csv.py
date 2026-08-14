"""
tools/ingest_squad_csv.py - turnkey adapter for any FIFA / EA FC player CSV.

Kaggle EA FC datasets, SoFIFA scrapes, and the older 4m4n5 datasets all use
DIFFERENT column names for the same things. This script auto-detects the
relevant columns (name, nationality, overall, position, club), normalizes
them to the schema our pipeline expects (Name / Nationality / Overall /
Position / Club), reconciles nationality names against the 48 World Cup
qualifiers, and writes a clean CSV ready to drop into model/squad_strength.py.

USAGE:
    python tools/ingest_squad_csv.py --in players_25.csv --out data/fc25_clean.csv

    # If auto-detection picks the wrong column, override explicitly:
    python tools/ingest_squad_csv.py --in players_25.csv --out data/fc25_clean.csv \
        --col-overall overall_rating --col-nationality nation

After it runs, it prints which of the 48 WC teams matched and which didn't,
so you can add any stragglers to NATION_ALIASES below and re-run.

Plug the output in:
    Write it into data/ under a name model/squad_ratings.py already looks for --
    SQUAD_CSV_CANDIDATES is ["fc26.csv", "fc26_clean.csv", "fifa2021.csv", "fifa2019.csv"],
    searched in order, first one present wins. So:

        python tools/ingest_squad_csv.py --in players_26.csv --out data/fc26_clean.csv

    No other file needs editing.
"""

from __future__ import annotations

# make the repo root importable when this file is run directly
# (python validation/foo.py) as well as from the root (python -m validation.foo)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import sys
from typing import Optional, List

import pandas as pd


# ---------------------------------------------------------------------------
# The 48 teams in the 2026 draw (normalized to our match-data names).
# Used only for the reconciliation report.
# ---------------------------------------------------------------------------

WC2026_TEAMS = [
    "Mexico", "South Korea", "South Africa", "Czech Republic",
    "Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina",
    "Brazil", "Morocco", "Scotland", "Haiti",
    "United States", "Australia", "Paraguay", "Turkey",
    "Germany", "Ecuador", "Ivory Coast", "Curaçao",
    "Netherlands", "Japan", "Tunisia", "Sweden",
    "Belgium", "Iran", "Egypt", "New Zealand",
    "Spain", "Uruguay", "Saudi Arabia", "Cape Verde",
    "France", "Senegal", "Norway", "Iraq",
    "Argentina", "Austria", "Algeria", "Jordan",
    "Portugal", "Colombia", "Uzbekistan", "DR Congo",
    "England", "Croatia", "Panama", "Ghana",
]


# ---------------------------------------------------------------------------
# Column auto-detection. We list known aliases for each target field, checked
# case-insensitively. First match wins; --col-* flags override.
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    "Name": ["name", "short_name", "long_name", "player_name", "full_name",
             "player"],
    "Nationality": ["nationality", "nationality_name", "nation", "country",
                    "nation_name", "country_name"],
    "Overall": ["overall", "overall_rating", "ovr", "oa", "rating",
                "current_rating"],
    "Position": ["position", "player_positions", "positions", "best_position",
                 "club_position", "pos"],
    "Club": ["club", "club_name", "team", "team_name", "club_team"],
}


def detect_column(df: pd.DataFrame, target: str,
                  override: Optional[str]) -> Optional[str]:
    if override:
        if override not in df.columns:
            sys.exit(f"ERROR: --col-{target.lower()} '{override}' not found. "
                     f"Available columns: {list(df.columns)}")
        return override
    lower_map = {c.lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES[target]:
        if alias in lower_map:
            return lower_map[alias]
    return None


# ---------------------------------------------------------------------------
# Nationality normalization: map source-dataset names to our match-data names.
# Add stragglers here after reading the reconciliation report.
# ---------------------------------------------------------------------------

NATION_ALIASES = {
    "Korea Republic": "South Korea",
    "Korea, Republic of": "South Korea",
    "Republic of Korea": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
    "United States of America": "United States",
    "Czechia": "Czech Republic",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "China PR": "China",
    "China": "China",
    "Chinese Taipei": "Taiwan",
    "IR Iran": "Iran",
    "Iran": "Iran",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "FYR Macedonia": "North Macedonia",
    "North Macedonia": "North Macedonia",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Curacao": "Curaçao",
    "Republic of Ireland": "Republic of Ireland",
    "Ireland": "Republic of Ireland",
}


def normalize_nation(n) -> str:
    if not isinstance(n, str):
        return ""
    n = n.strip()
    return NATION_ALIASES.get(n, n)


def simplify_position(pos) -> str:
    """Take the first listed position. EA CSVs often store 'ST, LW' or 'CB|RB'."""
    if not isinstance(pos, str):
        return ""
    for sep in [",", "|", "/"]:
        if sep in pos:
            return pos.split(sep)[0].strip()
    return pos.strip()


def main():
    ap = argparse.ArgumentParser(description="Adapt any FIFA/EA FC CSV to our schema.")
    ap.add_argument("--in", dest="infile", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="outfile", required=True, help="Output CSV path")
    ap.add_argument("--col-name", default=None)
    ap.add_argument("--col-nationality", default=None)
    ap.add_argument("--col-overall", default=None)
    ap.add_argument("--col-position", default=None)
    ap.add_argument("--col-club", default=None)
    args = ap.parse_args()

    print(f"Reading {args.infile} ...")
    df = pd.read_csv(args.infile, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Detect columns
    detected = {}
    overrides = {
        "Name": args.col_name, "Nationality": args.col_nationality,
        "Overall": args.col_overall, "Position": args.col_position,
        "Club": args.col_club,
    }
    for target in ["Name", "Nationality", "Overall", "Position", "Club"]:
        col = detect_column(df, target, overrides[target])
        detected[target] = col
        status = col if col else "NOT FOUND"
        print(f"  {target:<12} <- {status}")

    # Overall and Nationality are mandatory
    for required in ["Overall", "Nationality"]:
        if detected[required] is None:
            sys.exit(f"\nERROR: couldn't find a '{required}' column. "
                     f"Use --col-{required.lower()} to specify it. "
                     f"Available: {list(df.columns)}")

    # Build output frame
    out = pd.DataFrame()
    out["Name"] = df[detected["Name"]] if detected["Name"] else ""
    out["Nationality"] = df[detected["Nationality"]].map(normalize_nation)
    out["Overall"] = pd.to_numeric(df[detected["Overall"]], errors="coerce")
    out["Position"] = (df[detected["Position"]].map(simplify_position)
                       if detected["Position"] else "")
    out["Club"] = df[detected["Club"]] if detected["Club"] else ""

    # Drop rows with no usable overall
    before = len(out)
    out = out.dropna(subset=["Overall"]).copy()
    out["Overall"] = out["Overall"].astype(int)
    print(f"\n  dropped {before - len(out)} rows with no numeric Overall")
    print(f"  final: {len(out):,} players")

    out.to_csv(args.outfile, index=False)
    print(f"  wrote {args.outfile}")

    # ---- Reconciliation report ----
    print("\n" + "=" * 60)
    print("NATIONALITY RECONCILIATION vs 48 World Cup teams")
    print("=" * 60)
    available = set(out["Nationality"].unique())
    matched, missing = [], []
    for team in WC2026_TEAMS:
        if team in available:
            n = (out["Nationality"] == team).sum()
            matched.append((team, n))
        else:
            missing.append(team)

    print(f"\nMatched {len(matched)}/48 teams:")
    for team, n in sorted(matched, key=lambda x: -x[1]):
        print(f"  {team:<25} {n:>4} players")

    if missing:
        print(f"\n*** {len(missing)} teams NOT matched (need aliases): ***")
        for team in missing:
            # Try to suggest a close name from the data
            suggestions = [a for a in available
                           if team.split()[0].lower() in a.lower()
                           or a.split()[0].lower() in team.lower()]
            sug = f"  (similar in data: {suggestions[:3]})" if suggestions else ""
            print(f"  {team}{sug}")
        print("\n  -> Add the correct source-name -> our-name mappings to")
        print("     NATION_ALIASES at the top of this file, then re-run.")
    else:
        print("\n  All 48 teams matched. You're ready to plug this in.")


if __name__ == "__main__":
    main()
