#!/usr/bin/env python3
"""
predict.py -- expected goals and full market probabilities for a single fixture.

    python predict.py "Spain" "France"
    python predict.py "United States" "Wales" --home
    python predict.py "Brazil" "Morocco" --scale

Team names follow the martj42 results dataset ("United States", "South Korea").
Run with no teams for a short demo of several fixtures.

Flags:
    --home                give team 1 home advantage (default: neutral venue)
    --no-debias           ablate the cross-confederation correction
    --debias-weight=0.5   change the pull strength (default 0.7)
    --scale               measure the uniform total calibration and apply it
    --scale=1.12          apply a manual goal-level factor
    --no-split-home       revert to a single-parameter home advantage

The model itself lives in model/matchup.py; this is just the CLI.
"""

from model.matchup import main

if __name__ == "__main__":
    main()
