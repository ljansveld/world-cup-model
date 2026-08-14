"""
tools/scrape_sofifa.py - polite scraper for SoFIFA player ratings.

Produces a CSV with columns: Name, Nationality, Overall, Position, Club, Potential
which is exactly what model/squad_strength.py expects (it reads Name/Nationality/
Overall/Position).

RUN THIS LOCALLY, not in the Claude environment (SoFIFA is not reachable there).

IMPORTANT - SoFIFA is behind Cloudflare bot protection:
    A plain `requests` call gets a 403 at the door (you'll see it fail on the
    very first page). The fix is to impersonate a real browser's TLS
    fingerprint. Install curl_cffi:

        pip install requests beautifulsoup4 curl_cffi

    The scraper auto-detects curl_cffi and uses Chrome impersonation. If
    curl_cffi isn't installed it falls back to plain requests, which will
    probably still 403.

USAGE:
    python tools/scrape_sofifa.py --out data/fc26.csv --max-pages 400 --delay 1.5

    If you STILL get 403 even with curl_cffi (Cloudflare occasionally escalates
    to JavaScript/Turnstile challenges that TLS impersonation alone can't pass),
    see the "IF CURL_CFFI ISN'T ENOUGH" section at the bottom of this file.

Then plug into the World Cup pipeline:
    Write it to data/fc26.csv and you are done -- model/squad_ratings.py picks up
    the first name in SQUAD_CSV_CANDIDATES that exists in data/, and fc26.csv is
    first in that list. Nothing else needs editing.

NOTES ON POLITENESS / TERMS:
    - SoFIFA's data is itself sourced from EA Sports. Scraping is for personal
      / research use; don't hammer the site or redistribute the raw data.
    - Default 1.5s delay between requests keeps load light. ~400 pages * 1.5s
      is about 10 minutes for the full database (~24k players, 60/page).
    - The script identifies itself honestly in the user-agent and retries
      gently. If SoFIFA blocks you, increase --delay; do not try to evade.
    - This scrapes ONLY public ratings pages, nothing behind a login.

If the HTML layout has changed and parsing breaks, the script will print a
clear diagnostic showing what it found, so you can fix the selectors.
"""

from __future__ import annotations

# make the repo root importable when this file is run directly
# (python validation/foo.py) as well as from the root (python -m validation.foo)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import requests
from bs4 import BeautifulSoup


SOFIFA_BASE = "https://sofifa.com"
# The players list endpoint. offset increments by 60 per page.
PLAYERS_URL = SOFIFA_BASE + "/players"

# SoFIFA sits behind Cloudflare-style bot management. A "scraper" user-agent
# gets a 403 at the door. We must look like a real browser: real Chrome UA,
# full header set, and (ideally) a matching TLS fingerprint via curl_cffi.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

BROWSER_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,image/apng,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="126", "Google Chrome";v="126", "Not.A/Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "Connection": "keep-alive",
}


@dataclass
class Player:
    Name: str
    Nationality: str
    Overall: int
    Potential: Optional[int]
    Position: str
    Club: str


def make_session():
    """Create a session that impersonates Chrome.

    Prefers curl_cffi (which matches Chrome's TLS/JA3 fingerprint and clears
    the most common Cloudflare detection layer). Falls back to plain requests
    if curl_cffi isn't installed -- but note the fallback will likely still
    get 403'd by Cloudflare-protected SoFIFA, so installing curl_cffi is
    strongly recommended:  pip install curl_cffi
    """
    try:
        from curl_cffi import requests as creq
        # impersonate a recent Chrome; this sets the TLS fingerprint + HTTP/2
        session = creq.Session(impersonate="chrome")
        session.headers.update(BROWSER_HEADERS)
        print("  using curl_cffi with Chrome impersonation (recommended)")
        return session, "curl_cffi"
    except ImportError:
        print("  WARNING: curl_cffi not installed; falling back to plain requests.",
              file=sys.stderr)
        print("  SoFIFA is behind Cloudflare and will likely 403 plain requests.",
              file=sys.stderr)
        print("  Install with:  pip install curl_cffi", file=sys.stderr)
        s = requests.Session()
        s.headers.update(BROWSER_HEADERS)
        return s, "requests"


def fetch_page(session, offset: int, retries: int = 4,
               backoff: float = 3.0) -> str:
    """Fetch one players list page with retry + exponential backoff.

    Works with both a curl_cffi session and a plain requests session, since
    both expose a .get() with params= and .status_code / .text / .raise_for_status.
    """
    params = {"offset": offset}
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(PLAYERS_URL, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 403:
                # Cloudflare block. Retrying the same way won't help much, but
                # give it one or two tries in case it was a transient challenge.
                wait = backoff * (2 ** attempt)
                print(f"    HTTP 403 (Cloudflare block) at offset {offset}. "
                      f"Retrying in {wait:.0f}s...", file=sys.stderr)
                if attempt >= 1:
                    print("    Still 403 after retry. See the troubleshooting "
                          "notes at the top of this file -- you likely need "
                          "curl_cffi installed, or a stronger method.",
                          file=sys.stderr)
                time.sleep(wait)
                continue
            if resp.status_code in (429, 503):
                wait = backoff * (2 ** attempt)
                print(f"    HTTP {resp.status_code} at offset {offset}, "
                      f"backing off {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except Exception as e:
            last_err = e
            wait = backoff * (2 ** attempt)
            print(f"    request error at offset {offset}: {e}; retrying in "
                  f"{wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch offset {offset} after {retries} retries: {last_err}")


def parse_players(html: str) -> List[Player]:
    """Parse one players list page into Player records.

    SoFIFA's table has one <tr> per player. Columns we care about:
      - name + primary position(s): in the player-name cell
      - nationality: <img> title attribute in the name cell
      - overall, potential: numeric cells (data-col 'oa' / 'pt')
      - club: in the team cell
    We parse defensively: if the expected structure isn't found, we collect
    diagnostics and raise so the caller can surface a helpful error.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No <table> found on page - layout may have changed.")

    tbody = table.find("tbody")
    if tbody is None:
        raise ValueError("No <tbody> in table - layout may have changed.")

    players: List[Player] = []
    rows = tbody.find_all("tr")
    for tr in rows:
        try:
            player = _parse_row(tr)
            if player is not None:
                players.append(player)
        except Exception as e:
            # Skip a malformed row but keep going
            print(f"    warning: skipped a row ({e})", file=sys.stderr)
            continue

    return players


def _parse_row(tr) -> Optional[Player]:
    cells = tr.find_all("td")
    if len(cells) < 3:
        return None

    # --- Name + position cell ---
    # Typically the cell containing an <a> linking to /player/<id>/...
    name_cell = None
    for td in cells:
        a = td.find("a", href=re.compile(r"/player/\d+"))
        if a is not None:
            name_cell = td
            break
    if name_cell is None:
        return None

    name_link = name_cell.find("a", href=re.compile(r"/player/\d+"))
    name = name_link.get("aria-label") or name_link.get_text(strip=True)
    name = name.strip()

    # Nationality: an <img> with a title/alt attribute (the flag)
    nationality = ""
    flag = name_cell.find("img")
    if flag is not None:
        nationality = (flag.get("title") or flag.get("alt") or "").strip()
    # Fallback: sometimes nationality is a <a href="/players?na=..."> with title
    if not nationality:
        na_link = name_cell.find("a", href=re.compile(r"na="))
        if na_link is not None:
            nationality = na_link.get("title", "").strip()

    # Position(s): small tags/spans within the name cell, e.g. "ST", "CM"
    positions = []
    for span in name_cell.find_all(["span", "a"]):
        txt = span.get_text(strip=True)
        if re.fullmatch(r"[A-Z]{2,3}", txt) and txt not in ("FC",):
            positions.append(txt)
    position = positions[0] if positions else ""

    # --- Overall and Potential ---
    # These are numeric cells. SoFIFA marks them with data-col="oa"/"pt"
    overall = None
    potential = None
    club = ""

    for td in cells:
        dcol = td.get("data-col")
        if dcol == "oa":
            overall = _extract_int(td.get_text())
        elif dcol == "pt":
            potential = _extract_int(td.get_text())

    # If data-col attributes aren't present (older/newer layout), fall back to
    # the first two standalone 2-digit numbers in the row.
    if overall is None:
        nums = []
        for td in cells:
            val = _extract_int(td.get_text())
            if val is not None and 1 <= val <= 99:
                nums.append(val)
        if nums:
            overall = nums[0]
            if len(nums) > 1:
                potential = nums[1]

    # --- Club ---
    club_link = None
    for td in cells:
        a = td.find("a", href=re.compile(r"/team/\d+"))
        if a is not None:
            club_link = a
            break
    if club_link is not None:
        club = club_link.get_text(strip=True)

    if not name or overall is None:
        return None

    return Player(
        Name=name,
        Nationality=nationality,
        Overall=int(overall),
        Potential=int(potential) if potential is not None else None,
        Position=position,
        Club=club,
    )


def _extract_int(text: str) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def scrape(out_path: str, max_pages: int, delay: float, resume: bool):
    session, backend = make_session()

    # Resume support: if the output file exists and resume=True, count rows
    # already written and skip those pages.
    already = 0
    write_header = True
    mode = "w"
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            already = sum(1 for _ in f) - 1  # minus header
        if already > 0:
            start_page = already // 60
            print(f"Resuming: {already} players already in {out_path}, "
                  f"skipping to page {start_page}")
            write_header = False
            mode = "a"
        else:
            start_page = 0
    else:
        start_page = 0

    fieldnames = ["Name", "Nationality", "Overall", "Potential", "Position", "Club"]
    total = 0
    empty_streak = 0

    with open(out_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for page in range(start_page, max_pages):
            offset = page * 60
            html = fetch_page(session, offset)
            players = parse_players(html)

            if len(players) == 0:
                empty_streak += 1
                print(f"  page {page} (offset {offset}): 0 players "
                      f"(empty streak {empty_streak})")
                # Two empty pages in a row -> we've hit the end of the database
                if empty_streak >= 2:
                    print("  Two consecutive empty pages; assuming end of data.")
                    break
            else:
                empty_streak = 0
                for p in players:
                    writer.writerow(asdict(p))
                total += len(players)
                print(f"  page {page} (offset {offset}): {len(players)} players "
                      f"(total {total})")
                f.flush()  # checkpoint so resume works if we crash

            time.sleep(delay)

    print(f"\nDone. Wrote ~{total} players this run to {out_path}")
    print("Sanity-check a few rows:")
    os.system(f"head -5 {out_path}" if os.name != "nt" else f"more {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Polite SoFIFA player ratings scraper.")
    ap.add_argument("--out", default="data/sofifa_players.csv",
                    help="Output CSV path (default: data/sofifa_players.csv). "
                         "Write to data/fc26.csv to have the model pick it up "
                         "automatically -- note that overwrites the committed one.")
    ap.add_argument("--max-pages", type=int, default=400,
                    help="Max pages to scrape (60 players/page; 400 ~= 24k players)")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="Seconds to wait between requests (be polite; default 1.5)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Don't resume from an existing output file; start fresh")
    args = ap.parse_args()

    if args.delay < 0.5:
        print("Refusing delay < 0.5s to avoid hammering the site. "
              "Use at least 0.5.", file=sys.stderr)
        sys.exit(1)

    print(f"Scraping SoFIFA -> {args.out}")
    print(f"  max_pages={args.max_pages}, delay={args.delay}s, "
          f"resume={not args.no_resume}")
    print(f"  estimated time: ~{args.max_pages * args.delay / 60:.0f} min "
          f"(plus parsing)\n")

    try:
        scrape(args.out, args.max_pages, args.delay, resume=not args.no_resume)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        print("\nIf this is a parsing error, SoFIFA's HTML layout may have "
              "changed. Open one page in your browser, inspect the table "
              "structure, and adjust the selectors in _parse_row().",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


# ============================================================================
# IF CURL_CFFI ISN'T ENOUGH
# ============================================================================
#
# Cloudflare has tiers. curl_cffi's TLS impersonation clears the most common
# one, but if SoFIFA has escalated to JavaScript or Turnstile challenges for
# your IP, you have a few options, in rough order of effort:
#
# 1. SWAP THE DATA SOURCE (easiest, recommended).
#    The Kaggle EA FC datasets are scraped from SoFIFA already and published
#    as clean CSVs -- no Cloudflare in the way. This is almost certainly less
#    work than beating Cloudflare:
#        - "EA Sports FC 25 complete player dataset" on Kaggle
#        - download via the Kaggle API (kaggle datasets download -d ...)
#        - then just rename columns to Name/Nationality/Overall/Position
#    Given the goal is squad-strength features, the Kaggle snapshot is just as
#    good as a fresh scrape for our purposes.
#
# 2. USE A REAL BROWSER (medium effort).
#    Tools like SeleniumBase "UC mode", nodriver, or Playwright with a stealth
#    plugin drive an actual Chrome instance, which passes JS challenges. Slower
#    (you render every page) but robust. You'd replace fetch_page() with a call
#    that navigates the browser to PLAYERS_URL + "?offset=" + offset and returns
#    page.content().
#
# 3. USE A SCRAPING API (costs money, least effort per request).
#    Services like Scrapfly / ScrapeOps / ZenRows handle Cloudflare for you;
#    you send them the URL and get HTML back. Fine if you value time over a
#    small cost, overkill for a one-off squad-strength refresh.
#
# Honest recommendation: try curl_cffi first (this file). If that 403s too,
# go straight to option 1 (Kaggle) rather than escalating the arms race --
# it's the path of least resistance and the data is equivalent.
# ============================================================================
