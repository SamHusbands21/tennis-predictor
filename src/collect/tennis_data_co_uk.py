"""
Download ATP match results from tennis-data.co.uk.

Used as a supplement to Jeff Sackmann's data for years that Sackmann has not
yet published.  Updated ~daily by the site maintainer.

Match results are downloaded without odds (Betfair Exchange odds are not
available from this source).  A separate download_historical_odds() function
extracts Pinnacle Sports closing odds (PSW/PSL) for use in backtesting.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parents[2] / "data" / "raw" / "tennis_data_co_uk"
BASE_URL = "http://www.tennis-data.co.uk"

# tennis-data Series label → Sackmann tourney_level code
SERIES_LEVEL_MAP: dict[str, str] = {
    "grand slam": "G",
    "masters": "M",
    "masters 1000": "M",
    "masters cup": "M",
    "atp world tour finals": "M",
    "nitto atp finals": "M",
    "international gold": "A",
    "international": "A",
    "atp500": "A",
    "atp250": "A",
    "500": "A",
    "250": "A",
}

# Sackmann player IDs are ~100 000–210 000.
# Pseudo-IDs for players first seen in tennis-data start at 500 000.
_PSEUDO_ID_BASE = 500_000

# Sackmann round abbreviations used throughout the pipeline
ROUND_MAP: dict[str, str] = {
    "1st round": "R64",
    "2nd round": "R32",
    "3rd round": "R16",
    "4th round": "QF",
    "quarterfinals": "QF",
    "semifinals": "SF",
    "the final": "F",
    "final": "F",
    "round robin": "RR",
    "r128": "R128",
    "r64": "R64",
    "r32": "R32",
    "r16": "R16",
    "qf": "QF",
    "sf": "SF",
    "f": "F",
    "rr": "RR",
    "br": "BR",
}

# Output columns — must match Sackmann schema exactly so engineer.py is unchanged
_SACKMANN_COLS = [
    "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date",
    "match_num", "winner_id", "winner_name", "winner_hand", "winner_ioc",
    "winner_age", "loser_id", "loser_name", "loser_hand", "loser_ioc",
    "loser_age", "score", "best_of", "round", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points", "loser_rank", "loser_rank_points",
    "date", "year",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pseudo_id(name: str) -> int:
    """Stable numeric ID derived from a player name string."""
    h = int(hashlib.md5(name.lower().strip().encode()).hexdigest(), 16)
    return _PSEUDO_ID_BASE + (h % 400_000)


def _normalize(name: str) -> str:
    return name.lower().strip() if isinstance(name, str) else ""


def build_name_lookup(sackmann_df: pd.DataFrame) -> dict[str, int]:
    """
    Build a dict of normalised_name → player_id from Sackmann historical data.

    Registers multiple name variants per player so tennis-data's abbreviated
    format (e.g. "Djokovic N.") can be matched:
      - "novak djokovic"        (full name, first last)
      - "djokovic novak"        (reversed)
      - "djokovic n"            (last + first-initial, no dot)
    """
    lookup: dict[str, int] = {}

    for _, row in sackmann_df[["winner_id", "winner_name"]].drop_duplicates().iterrows():
        pid = int(row["winner_id"])
        name: str = str(row["winner_name"]).strip()
        _register(lookup, name, pid)

    for _, row in sackmann_df[["loser_id", "loser_name"]].drop_duplicates().iterrows():
        pid = int(row["loser_id"])
        name = str(row["loser_name"]).strip()
        _register(lookup, name, pid)

    return lookup


def _register(lookup: dict[str, int], name: str, pid: int) -> None:
    """Add several normalised variants of *name* pointing to *pid*."""
    parts = name.split()
    if not parts:
        return
    norm = _normalize(name)
    lookup.setdefault(norm, pid)

    # reversed: "Djokovic Novak"
    if len(parts) >= 2:
        rev = _normalize(" ".join(reversed(parts)))
        lookup.setdefault(rev, pid)

        # "djokovic n" — last name + first initial
        first_initial = parts[0][0].lower()   # Sackmann: "Firstname Lastname"
        last = parts[-1].lower()
        lookup.setdefault(f"{last} {first_initial}", pid)
        # also without the initial: just last name (weaker, only used as fallback)
        lookup.setdefault(last, pid)


def _resolve_id(name: str, lookup: dict[str, int]) -> int:
    """Map a tennis-data player name to a Sackmann player ID, or a pseudo-ID."""
    norm = _normalize(name)
    if norm in lookup:
        return lookup[norm]

    # tennis-data format: "Djokovic N." — try "lastname firstinitial"
    parts = norm.replace(".", "").split()
    if len(parts) >= 2:
        # last initial  e.g. ["djokovic", "n"]
        candidate = f"{parts[0]} {parts[-1]}"
        if candidate in lookup:
            return lookup[candidate]
        # first initial last  e.g. ["n", "djokovic"]
        candidate2 = f"{parts[-1]} {parts[0]}"
        if candidate2 in lookup:
            return lookup[candidate2]
        # bare last name
        if parts[0] in lookup:
            return lookup[parts[0]]

    return _pseudo_id(name)


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _year_url(year: int) -> str:
    ext = "xlsx" if year >= 2013 else "xls"
    return f"{BASE_URL}/{year}/{year}.{ext}"


def _download_year(
    year: int,
    name_to_id: dict[str, int],
    retries: int = 3,
) -> pd.DataFrame:
    url = _year_url(year)
    raw_bytes: bytes | None = None

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw_bytes = resp.content
            break
        except Exception as exc:
            logger.warning(f"  Attempt {attempt + 1} failed for tennis-data {year}: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    if raw_bytes is None:
        raise RuntimeError(f"Failed to download tennis-data year {year}")

    engine = "openpyxl" if year >= 2013 else "xlrd"
    df = pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
    df.columns = [str(c).strip() for c in df.columns]
    logger.info(f"  Downloaded tennis-data {year}: {len(df)} rows raw")

    # Drop walkovers
    if "Comment" in df.columns:
        df = df[df["Comment"].astype(str).str.lower() != "walkover"]

    df["date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df["tourney_date"] = df["date"]
    df = df.dropna(subset=["date", "Winner", "Loser"])

    # Surface
    surface_map = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Hard"}
    df["surface"] = (
        df["Surface"].astype(str).str.lower().map(surface_map).fillna("Hard")
        if "Surface" in df.columns else "Hard"
    )

    # Tournament level
    series_col = next((c for c in ("Series", "Tier") if c in df.columns), None)
    if series_col:
        df["tourney_level"] = (
            df[series_col].astype(str).str.lower()
            .map(SERIES_LEVEL_MAP).fillna("A")
        )
    else:
        df["tourney_level"] = "A"

    # Round — normalise to Sackmann abbreviations
    if "Round" in df.columns:
        df["round"] = (
            df["Round"].astype(str).str.lower()
            .map(ROUND_MAP).fillna(df["Round"].astype(str))
        )
    else:
        df["round"] = "F"

    # Best of
    df["best_of"] = (
        pd.to_numeric(df["Best of"], errors="coerce").fillna(3).astype(int)
        if "Best of" in df.columns else 3
    )

    # Tournament name / id
    df["tourney_name"] = df["Tournament"].astype(str) if "Tournament" in df.columns else "Unknown"
    df["tourney_id"] = (
        df["date"].dt.year.astype(str) + "_td_" + df["tourney_name"].astype(str)
    )

    # Rankings / points
    df["winner_rank"] = pd.to_numeric(df.get("WRank"), errors="coerce")
    df["loser_rank"] = pd.to_numeric(df.get("LRank"), errors="coerce")
    df["winner_rank_points"] = pd.to_numeric(df.get("WPts"), errors="coerce")
    df["loser_rank_points"] = pd.to_numeric(df.get("LPts"), errors="coerce")

    # Player names + ID resolution
    df["winner_name"] = df["Winner"].astype(str).str.strip()
    df["loser_name"] = df["Loser"].astype(str).str.strip()
    df["winner_id"] = df["winner_name"].apply(lambda n: _resolve_id(n, name_to_id))
    df["loser_id"] = df["loser_name"].apply(lambda n: _resolve_id(n, name_to_id))

    # Columns tennis-data doesn't have — NaN so engineer rolling windows still work
    for col in [
        "winner_hand", "loser_hand", "winner_ioc", "loser_ioc",
        "winner_age", "loser_age", "score", "minutes",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
    ]:
        df[col] = np.nan

    df["year"] = year
    df["match_num"] = range(len(df))

    logger.info(f"  tennis-data {year}: {len(df)} matches after cleaning")
    return df[_SACKMANN_COLS].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_supplement(
    from_year: int,
    name_to_id: dict[str, int],
    force_current: bool = True,
) -> pd.DataFrame:
    """
    Download tennis-data.co.uk results from *from_year* to the current year.

    Args:
        from_year:    First year to fetch (typically Sackmann's last year + 1).
        name_to_id:   Mapping from normalised player name → Sackmann player ID,
                      built via build_name_lookup().
        force_current: Always re-fetch the current calendar year (live season).

    Returns:
        DataFrame in Sackmann schema, sorted by date.
    """
    current_year = datetime.now().year
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for year in range(from_year, current_year + 1):
        cache_path = RAW_DIR / f"td_{year}.parquet"
        is_current = year == current_year
        should_download = not cache_path.exists() or (force_current and is_current)

        if not should_download:
            logger.info(f"  Using cached tennis-data {year}")
            frames.append(pd.read_parquet(cache_path))
            continue

        try:
            logger.info(f"  Downloading tennis-data {year}...")
            df = _download_year(year, name_to_id)
            df.to_parquet(cache_path, index=False)
            frames.append(df)
        except Exception as exc:
            logger.warning(f"  Skipping tennis-data {year}: {exc}")
            if cache_path.exists():
                logger.info(f"  Falling back to cache for tennis-data {year}")
                frames.append(pd.read_parquet(cache_path))

    if not frames:
        return pd.DataFrame(columns=_SACKMANN_COLS)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Historical odds (Pinnacle Sports) — for evaluation / backtesting
# ---------------------------------------------------------------------------

def _download_year_odds(year: int, name_to_id: dict[str, int], retries: int = 3) -> pd.DataFrame:
    """
    Download a tennis-data.co.uk year file and return Pinnacle Sports closing
    odds (PSW = winner odds, PSL = loser odds) alongside resolved player IDs.

    Returns DataFrame with columns:
        date, winner_id, loser_id, odds_winner, odds_loser
    Rows missing Pinnacle odds are dropped.
    """
    url = _year_url(year)
    raw_bytes: bytes | None = None

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw_bytes = resp.content
            break
        except Exception as exc:
            logger.warning(f"  Attempt {attempt + 1} for odds {year}: {exc}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    if raw_bytes is None:
        raise RuntimeError(f"Failed to download tennis-data {year} for Pinnacle odds")

    engine = "openpyxl" if year >= 2013 else "xlrd"
    df = pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
    df.columns = [str(c).strip() for c in df.columns]

    if "Comment" in df.columns:
        df = df[df["Comment"].astype(str).str.lower() != "walkover"]

    df["date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df = df.dropna(subset=["date", "Winner", "Loser"])

    df["winner_id"] = df["Winner"].astype(str).str.strip().apply(
        lambda n: _resolve_id(n, name_to_id)
    )
    df["loser_id"] = df["Loser"].astype(str).str.strip().apply(
        lambda n: _resolve_id(n, name_to_id)
    )

    df["odds_winner"] = pd.to_numeric(df.get("PSW"), errors="coerce")
    df["odds_loser"]  = pd.to_numeric(df.get("PSL"), errors="coerce")

    out = df[["date", "winner_id", "loser_id", "odds_winner", "odds_loser"]].copy()
    out = out.dropna(subset=["odds_winner", "odds_loser"])
    logger.info(f"  Pinnacle odds {year}: {len(out)} matches")
    return out


def download_historical_odds(
    from_year: int,
    to_year: int,
    name_to_id: dict[str, int],
    force_current: bool = True,
) -> pd.DataFrame:
    """
    Download Pinnacle Sports closing odds (PSW/PSL) from tennis-data.co.uk
    for the given year range.  Used by engineer.py to populate odds_p1/odds_p2
    on the historical feature matrix so that evaluate.py can run profitability
    analysis.

    Note: Betfair Exchange odds are not available from tennis-data.co.uk.
    Pinnacle Sports is used as a proxy — it carries the lowest bookmaker
    margin (~2%) and is the industry-standard reference odds source.

    Caches each year's odds to:
        data/raw/tennis_data_co_uk/odds_{year}.parquet

    Returns DataFrame with columns:
        date, winner_id, loser_id, odds_winner, odds_loser
    """
    current_year = datetime.now().year
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for year in range(from_year, to_year + 1):
        cache_path = RAW_DIR / f"odds_{year}.parquet"
        is_current = year == current_year
        should_download = not cache_path.exists() or (force_current and is_current)

        if not should_download:
            logger.info(f"  Using cached Pinnacle odds {year}")
            frames.append(pd.read_parquet(cache_path))
            continue

        try:
            logger.info(f"  Downloading Pinnacle odds for {year}...")
            df = _download_year_odds(year, name_to_id)
            if not df.empty:
                df.to_parquet(cache_path, index=False)
            frames.append(df)
        except Exception as exc:
            logger.warning(f"  Skipping Pinnacle odds for {year}: {exc}")
            if cache_path.exists():
                logger.info(f"  Falling back to cached Pinnacle odds {year}")
                frames.append(pd.read_parquet(cache_path))

    if not frames:
        return pd.DataFrame(
            columns=["date", "winner_id", "loser_id", "odds_winner", "odds_loser"]
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)
