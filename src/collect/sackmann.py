"""
Download historical ATP match data from Jeff Sackmann's tennis_atp GitHub repo.

Each year CSV includes match results, serve statistics, player rankings, and
court surface. We collect from 2000 onward to allow Elo ratings to stabilise
before the 2005 training window starts.

Source: https://github.com/JeffSackmann/tennis_atp
"""

import io
import time
import logging
from datetime import datetime
from pathlib import Path

from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "sackmann"
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
BASE_URL_MAIN = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/main"

START_YEAR = 2000
_GITHUB_API_URL = "https://api.github.com/repos/JeffSackmann/tennis_atp/contents/"


def _latest_available_year() -> int:
    """Query the GitHub API once to find the most recent atp_matches_YYYY.csv."""
    try:
        r = requests.get(_GITHUB_API_URL, timeout=15)
        r.raise_for_status()
        names = [item["name"] for item in r.json() if isinstance(item, dict)]
        years = []
        for name in names:
            if name.startswith("atp_matches_") and name.endswith(".csv"):
                try:
                    years.append(int(name[len("atp_matches_"):-len(".csv")]))
                except ValueError:
                    pass
        return max(years) if years else datetime.now().year
    except Exception:
        return datetime.now().year


CURRENT_YEAR = _latest_available_year()

# Approximate day offset from tournament start date per round.
# Used to construct a match date (Sackmann gives only tournament start date).
ROUND_OFFSETS: dict[str, int] = {
    "RR": 0, "R128": 0, "R64": 1, "R32": 2, "R16": 3,
    "QF": 4, "SF": 5, "F": 6, "BR": 5,
}

COLS_KEEP = [
    "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date",
    "match_num", "winner_id", "winner_name", "winner_hand", "winner_ioc",
    "winner_age", "loser_id", "loser_name", "loser_hand", "loser_ioc",
    "loser_age", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
]

NUMERIC_COLS = [
    "winner_id", "loser_id", "winner_rank", "loser_rank",
    "winner_rank_points", "loser_rank_points", "minutes", "best_of",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]


def _year_url(year: int, branch: str = "master") -> str:
    base = BASE_URL if branch == "master" else BASE_URL_MAIN
    return f"{base}/atp_matches_{year}.csv"


def _download_year(year: int, retries: int = 3) -> pd.DataFrame:
    # Determine which branch has this year's file
    url = None
    for branch in ("master", "main"):
        candidate = _year_url(year, branch)
        try:
            r = requests.get(candidate, timeout=30)
            r.raise_for_status()
            url = candidate
            break
        except Exception:
            continue
    if url is None:
        raise RuntimeError(f"atp_matches_{year}.csv not found on master or main branch")

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            cols = [c for c in COLS_KEEP if c in df.columns]
            df = df[cols].copy()

            # tourney_date is an integer like 20240101 — parse to datetime
            df["tourney_date"] = pd.to_datetime(
                df["tourney_date"].astype(str).str[:8],
                format="%Y%m%d",
                errors="coerce",
            )
            # Approximate match date = tournament start + round offset
            df["round"] = df["round"].fillna("RR").astype(str)
            df["_round_offset"] = df["round"].map(ROUND_OFFSETS).fillna(0).astype(int)
            df["date"] = df["tourney_date"] + pd.to_timedelta(df["_round_offset"], unit="D")
            df = df.drop(columns=["_round_offset"])

            df = df.dropna(subset=["date", "winner_id", "loser_id"])
            df["year"] = year

            for col in NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop walkovers and retirements with no score
            df = df[df["score"].notna()]
            df = df[~df["score"].astype(str).str.contains("W/O", na=False)]

            logger.info(f"  Downloaded {year}: {len(df)} matches")
            return df
        except Exception as exc:
            logger.warning(f"  Attempt {attempt + 1} failed for {year}: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download year {year} after {retries} retries")


def download_all(
    start_year: int = START_YEAR,
    end_year: Optional[int] = None,
    force_current: bool = True,
    supplement: bool = True,
) -> pd.DataFrame:
    """
    Download all years from start_year to end_year inclusive from Jeff Sackmann's
    tennis_atp repo, then automatically supplement with tennis-data.co.uk for any
    subsequent years (updated ~daily).

    Args:
        supplement: If True (default), append tennis-data.co.uk rows for years
                    beyond Sackmann's latest available year.  Set False to use
                    only Sackmann data (e.g. for historical-only backtests).
    """
    if end_year is None:
        end_year = CURRENT_YEAR

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for year in range(start_year, end_year + 1):
        cache_path = RAW_DIR / f"atp_matches_{year}.csv"
        is_current = year == CURRENT_YEAR
        should_download = not cache_path.exists() or (force_current and is_current)

        if not should_download:
            logger.info(f"  Using cached {cache_path.name}")
            df = pd.read_csv(cache_path, parse_dates=["date", "tourney_date"], low_memory=False)
        else:
            try:
                logger.info(f"  Downloading {year}...")
                df = _download_year(year)
                df.to_csv(cache_path, index=False)
            except Exception as exc:
                logger.warning(f"  Skipping {year}: {exc}")
                if cache_path.exists():
                    logger.warning(f"  Falling back to cached {year}")
                    df = pd.read_csv(cache_path, parse_dates=["date", "tourney_date"], low_memory=False)
                else:
                    continue

        frames.append(df)

    if not frames:
        raise RuntimeError("No ATP match data could be loaded.")

    sackmann_df = pd.concat(frames, ignore_index=True)
    sackmann_df = sackmann_df.sort_values("date").reset_index(drop=True)
    logger.info(f"Sackmann ATP matches loaded: {len(sackmann_df)} (up to {CURRENT_YEAR})")

    # Supplement with tennis-data.co.uk for seasons after Sackmann's last year
    if supplement:
        supplement_from = CURRENT_YEAR + 1
        current_cal_year = datetime.now().year
        if supplement_from <= current_cal_year:
            logger.info(
                f"Supplementing with tennis-data.co.uk for {supplement_from}–{current_cal_year}..."
            )
            from src.collect.tennis_data_co_uk import (
                build_name_lookup,
                download_supplement,
            )
            name_to_id = build_name_lookup(sackmann_df)
            td_df = download_supplement(
                from_year=supplement_from,
                name_to_id=name_to_id,
                force_current=force_current,
            )
            if not td_df.empty:
                logger.info(f"tennis-data.co.uk added {len(td_df)} matches")
                sackmann_df = pd.concat([sackmann_df, td_df], ignore_index=True)
                sackmann_df = sackmann_df.sort_values("date").reset_index(drop=True)

    logger.info(f"Total ATP matches loaded: {len(sackmann_df)}")
    return sackmann_df


def load_cached() -> pd.DataFrame:
    """Load all cached year CSVs without re-downloading."""
    frames = []
    for path in sorted(RAW_DIR.glob("atp_matches_*.csv")):
        frames.append(pd.read_csv(path, parse_dates=["date", "tourney_date"], low_memory=False))
    if not frames:
        raise FileNotFoundError(
            f"No cached data found in {RAW_DIR}. Run download_all() first."
        )
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = download_all()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Surfaces:\n{df['surface'].value_counts()}")
    print(f"Tournament levels:\n{df['tourney_level'].value_counts()}")
