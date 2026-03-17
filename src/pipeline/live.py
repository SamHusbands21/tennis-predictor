"""
Daily live pipeline entrypoint for the tennis betting model.

Steps:
  1. Download / refresh ATP match CSVs from Jeff Sackmann's GitHub
  2. Rebuild Elo and surface-Elo ratings to today
  3. Load trained models
  4. Fetch upcoming ATP matches + odds from Betfair Exchange
  5. Build features for each match
  6. Generate predictions and identify value bets
  7. Write output/recommendations.json
  8. Copy to sam_website/data/tennis_recommendations.json

This script is run daily by run_daily.ps1 on a local Windows machine (required
because the Betfair API demands a UK IP address).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.collect.betfair import get_upcoming_atp_matches
from src.collect.sackmann import download_all as download_atp
from src.features.elo import EloSystem
from src.features.surface_elo import SurfaceEloSystem, normalise_surface
from src.features.engineer import (
    MODEL_FEATURES,
    TOURNEY_LEVEL_MAP,
    MAX_DAYS_REST,
    FORM_WINDOW_SHORT,
    FORM_WINDOW_LONG,
    SURFACE_FORM_WINDOW,
    SERVE_WINDOW,
    _safe_div,
    _serve_stats_for_player,
    _build_long_form,
    _rolling_player_stats,
)
from src.models.train import load_models, ensemble_proba

logger = logging.getLogger(__name__)

OUTPUT_DIR   = ROOT / "output"
EV_THRESHOLD = 1.20
MAX_KELLY    = 0.25


# ── Rating computation ────────────────────────────────────────────────────────

def _build_current_ratings(hist_df: pd.DataFrame) -> tuple[
    dict[int, float],
    dict[str, dict[int, float]],
]:
    """
    Fit Elo and surface-Elo on all historical data.
    Returns (overall_ratings, surface_ratings_dict).
    """
    elo = EloSystem(k=32)
    elo.fit_transform(hist_df)

    s_elo = SurfaceEloSystem(k=32)
    s_elo.fit_transform(hist_df)

    return elo.get_current_ratings(), s_elo.get_current_ratings()


def _build_player_lookup(
    hist_df: pd.DataFrame,
) -> tuple[dict[int, dict], dict[str, int]]:
    """
    Build two lookups from the historical data:

    1. player_stats_lookup: player_id -> most recent pre-match rolling stats dict
    2. name_to_id: normalised_name -> player_id (for Betfair name matching)
    """
    long = _build_long_form(hist_df)
    all_stats = _rolling_player_stats(long)

    player_stats_lookup: dict[int, dict] = {}
    for player_id, grp in all_stats.groupby("player_id"):
        latest = grp.sort_values("date").iloc[-1]
        player_stats_lookup[int(player_id)] = latest.to_dict()

    # Name → id from most recent appearance as winner or loser
    name_to_id: dict[str, int] = {}
    for _, row in hist_df.sort_values("date").iterrows():
        name_to_id[str(row["winner_name"]).strip()] = int(row["winner_id"])
        name_to_id[str(row["loser_name"]).strip()]  = int(row["loser_id"])

    # Current rank info (use most recent match per player)
    player_rank: dict[int, tuple[float, float]] = {}  # id -> (rank, rank_pts)
    for _, row in hist_df.sort_values("date").iterrows():
        player_rank[int(row["winner_id"])] = (
            float(row.get("winner_rank") or np.nan),
            float(row.get("winner_rank_points") or 0),
        )
        player_rank[int(row["loser_id"])] = (
            float(row.get("loser_rank") or np.nan),
            float(row.get("loser_rank_points") or 0),
        )

    return player_stats_lookup, name_to_id, player_rank


def _match_betfair_name(
    betfair_name: str,
    name_to_id: dict[str, int],
) -> int | None:
    """
    Resolve a Betfair runner name to a Sackmann player_id.

    Attempts (in order):
      1. Exact match
      2. "F. LastName" → match by first initial + last name
      3. Last name only (fallback; risky for common surnames)
    """
    name = betfair_name.strip()

    # 1. Exact
    if name in name_to_id:
        return name_to_id[name]

    # 2. Abbreviated: "N. Djokovic"
    parts = name.split(". ", 1)
    if len(parts) == 2:
        first_initial = parts[0].strip().upper()
        last_name     = parts[1].strip().lower()
        candidates = [
            pid for n, pid in name_to_id.items()
            if len(n.split()) >= 2
            and n.split()[-1].lower() == last_name
            and n.split()[0][0].upper() == first_initial
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            # Multiple candidates: prefer the one seen most recently
            return candidates[0]

    # 3. Last name only
    last_name = name.split()[-1].lower()
    candidates = [
        pid for n, pid in name_to_id.items()
        if n.split()[-1].lower() == last_name
    ]
    if len(candidates) == 1:
        return candidates[0]

    return None


# ── Feature assembly for a single live match ──────────────────────────────────

def _player_rolling_stats(
    player_id: int | None,
    surface: str,
    fixture_date: datetime,
    player_stats_lookup: dict[int, dict],
) -> dict[str, float]:
    """
    Return pre-match rolling stat defaults for a player, looked up from the
    most recent entry in their rolling stats history.
    """
    defaults: dict[str, float] = {
        "win_rate_5":               0.5,
        "win_rate_10":              0.5,
        "surface_win_rate_10":      0.5,
        "first_serve_pct":          0.60,  # approximate ATP average
        "first_serve_won_pct":      0.73,  # approximate ATP average
        "bp_save_rate":             0.65,  # approximate ATP average
        "prev_match_minutes":       0.0,
        "days_rest":                MAX_DAYS_REST,
    }

    if player_id is None or player_id not in player_stats_lookup:
        return defaults

    stats = player_stats_lookup[player_id]
    surf = normalise_surface(surface)
    surf_col = f"surface_win_rate_10_{surf.lower()}"

    last_date = stats.get("last_date")
    if last_date is not None and pd.notna(last_date):
        try:
            last_dt = pd.to_datetime(last_date)
            fix_dt  = pd.to_datetime(fixture_date).tz_localize(None)
            if hasattr(last_dt, "tz_localize"):
                last_dt = last_dt.tz_localize(None)
            days = min(int((fix_dt - last_dt).days), MAX_DAYS_REST)
        except Exception:
            days = MAX_DAYS_REST
    else:
        days = MAX_DAYS_REST

    return {
        "win_rate_5":           float(stats.get("win_rate_5")           or 0.5),
        "win_rate_10":          float(stats.get("win_rate_10")          or 0.5),
        "surface_win_rate_10":  float(stats.get(surf_col)               or 0.5),
        "first_serve_pct":      float(stats.get("first_serve_pct")      or 0.60),
        "first_serve_won_pct":  float(stats.get("first_serve_won_pct")  or 0.73),
        "bp_save_rate":         float(stats.get("bp_save_rate")         or 0.65),
        "prev_match_minutes":   float(stats.get("prev_minutes")         or 0.0),
        "days_rest":            float(days),
    }


def _h2h_rate_live(
    p1_id: int | None,
    p2_id: int | None,
    hist_df: pd.DataFrame,
    n: int = 10,
) -> float:
    """Compute P1's H2H win rate in the last n meetings from historical data."""
    if p1_id is None or p2_id is None:
        return 0.5

    mask = (
        ((hist_df["winner_id"] == p1_id) & (hist_df["loser_id"] == p2_id))
        | ((hist_df["winner_id"] == p2_id) & (hist_df["loser_id"] == p1_id))
    )
    past = hist_df[mask].sort_values("date").tail(n)
    if len(past) == 0:
        return 0.5
    p1_wins = (past["winner_id"] == p1_id).sum()
    return float(p1_wins / len(past))


def _build_match_features(
    match: dict,
    p1_id: int | None,
    p2_id: int | None,
    elo_ratings: dict[int, float],
    surface_ratings: dict[str, dict[int, float]],
    player_stats_lookup: dict[int, dict],
    player_rank: dict[int, tuple],
    hist_df: pd.DataFrame,
) -> dict[str, float]:
    """Assemble the full feature vector for a single upcoming match."""
    surface = normalise_surface(str(match.get("surface", "Hard")))
    default_elo = 1500.0

    elo_p1 = float(elo_ratings.get(p1_id, default_elo)) if p1_id else default_elo
    elo_p2 = float(elo_ratings.get(p2_id, default_elo)) if p2_id else default_elo

    surf_pool = surface_ratings.get(surface, {})
    surface_elo_p1 = float(surf_pool.get(p1_id, default_elo)) if p1_id else default_elo
    surface_elo_p2 = float(surf_pool.get(p2_id, default_elo)) if p2_id else default_elo

    fix_date = match["date"]
    p1_stats = _player_rolling_stats(p1_id, surface, fix_date, player_stats_lookup)
    p2_stats = _player_rolling_stats(p2_id, surface, fix_date, player_stats_lookup)

    p1_rank,    p1_rank_pts    = player_rank.get(p1_id, (np.nan, 0.0)) if p1_id else (np.nan, 0.0)
    p2_rank,    p2_rank_pts    = player_rank.get(p2_id, (np.nan, 0.0)) if p2_id else (np.nan, 0.0)

    rank_diff = (float(p2_rank) - float(p1_rank)) if (
        pd.notna(p1_rank) and pd.notna(p2_rank)
    ) else 0.0

    h2h = _h2h_rate_live(p1_id, p2_id, hist_df)

    tourney_level     = str(match.get("tourney_level", "A"))
    tourney_level_enc = TOURNEY_LEVEL_MAP.get(tourney_level, 2)
    best_of           = int(match.get("best_of", 3))

    return {
        "elo_p1":                   elo_p1,
        "elo_p2":                   elo_p2,
        "elo_diff":                 elo_p1 - elo_p2,
        "surface_elo_p1":           surface_elo_p1,
        "surface_elo_p2":           surface_elo_p2,
        "surface_elo_diff":         surface_elo_p1 - surface_elo_p2,
        "p1_win_rate_5":            p1_stats["win_rate_5"],
        "p2_win_rate_5":            p2_stats["win_rate_5"],
        "p1_win_rate_10":           p1_stats["win_rate_10"],
        "p2_win_rate_10":           p2_stats["win_rate_10"],
        "p1_surface_win_rate_10":   p1_stats["surface_win_rate_10"],
        "p2_surface_win_rate_10":   p2_stats["surface_win_rate_10"],
        "p1_first_serve_pct":       p1_stats["first_serve_pct"],
        "p2_first_serve_pct":       p2_stats["first_serve_pct"],
        "p1_first_serve_won_pct":   p1_stats["first_serve_won_pct"],
        "p2_first_serve_won_pct":   p2_stats["first_serve_won_pct"],
        "p1_bp_save_rate":          p1_stats["bp_save_rate"],
        "p2_bp_save_rate":          p2_stats["bp_save_rate"],
        "p1_days_rest":             p1_stats["days_rest"],
        "p2_days_rest":             p2_stats["days_rest"],
        "p1_prev_match_minutes":    p1_stats["prev_match_minutes"],
        "p2_prev_match_minutes":    p2_stats["prev_match_minutes"],
        "rank_diff":                rank_diff,
        "p1_rank_pts":              float(p1_rank_pts),
        "p2_rank_pts":              float(p2_rank_pts),
        "h2h_p1_win_rate":          h2h,
        "tourney_level_encoded":    tourney_level_enc,
        "best_of":                  best_of,
        "surface_hard":             int(surface == "Hard"),
        "surface_clay":             int(surface == "Clay"),
        "surface_grass":            int(surface == "Grass"),
    }


def _value_bets_for_match(
    p_p1: float,
    betfair_odds: dict,
) -> list[dict]:
    """Return value bets for a match, filtered by EV threshold."""
    bets = []
    for outcome, p, o in [
        ("p1", p_p1,       betfair_odds.get("p1")),
        ("p2", 1 - p_p1,   betfair_odds.get("p2")),
    ]:
        if o is None or o <= 1.0 or p <= 0:
            continue
        ev = p * o
        if ev < EV_THRESHOLD:
            continue
        kelly = min(MAX_KELLY, (p * o - 1) / (o - 1))
        bets.append({
            "outcome":        outcome,
            "ev":             round(ev, 4),
            "kelly_fraction": round(kelly, 4),
        })
    return sorted(bets, key=lambda x: x["ev"], reverse=True)


# ── Pipeline entrypoint ───────────────────────────────────────────────────────

def run_pipeline(days_ahead: int = 7) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading / refreshing ATP match data...")
    hist_df = download_atp(force_current=True)

    logger.info("Building current Elo and surface-Elo ratings...")
    elo_ratings, surface_ratings = _build_current_ratings(hist_df)

    logger.info("Building player stats lookup...")
    player_stats_lookup, name_to_id, player_rank = _build_player_lookup(hist_df)

    logger.info("Loading trained models...")
    xgb_model, rf_model = load_models()

    logger.info("Fetching upcoming ATP matches from Betfair Exchange...")
    try:
        matches = get_upcoming_atp_matches(days_ahead=days_ahead)
    except Exception as exc:
        logger.warning(f"Betfair unavailable ({exc}). Writing empty recommendations.")
        matches = []

    if not matches:
        output = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "matches": [],
        }
        with open(OUTPUT_DIR / "recommendations.json", "w") as f:
            json.dump(output, f, indent=2)
        return

    logger.info(f"Generating predictions for {len(matches)} matches...")
    output_matches = []

    for match in matches:
        try:
            player1 = match["player1"]
            player2 = match["player2"]

            p1_id = _match_betfair_name(player1, name_to_id)
            p2_id = _match_betfair_name(player2, name_to_id)

            if p1_id is None:
                logger.debug(f"  Unknown player: {player1}")
            if p2_id is None:
                logger.debug(f"  Unknown player: {player2}")

            features = _build_match_features(
                match, p1_id, p2_id,
                elo_ratings, surface_ratings,
                player_stats_lookup, player_rank,
                hist_df,
            )

            X = np.array([[features[col] for col in MODEL_FEATURES]])
            p_p1 = float(ensemble_proba(xgb_model, rf_model, X)[0])

            model_probs = {
                "p1": round(p_p1, 4),
                "p2": round(1 - p_p1, 4),
            }

            betfair_odds = match["betfair_odds"]
            value_bets   = _value_bets_for_match(p_p1, betfair_odds)

            date_val = match["date"]
            date_str = date_val.isoformat() if hasattr(date_val, "isoformat") else str(date_val)

            surface    = normalise_surface(str(match.get("surface", "Hard")))
            tournament = match.get("tournament", "Unknown")
            round_name = match.get("round", "")

            output_matches.append({
                "player1":     player1,
                "player2":     player2,
                "tournament":  tournament,
                "surface":     surface,
                "round":       round_name,
                "date":        date_str,
                "model_probs": model_probs,
                "betfair_odds": {
                    k: round(v, 2) if v is not None else None
                    for k, v in betfair_odds.items()
                },
                "value_bets": value_bets,
            })

            logger.info(
                f"  {player1} vs {player2} [{surface}] "
                f"P1={model_probs['p1']} P2={model_probs['p2']} "
                f"| {len(value_bets)} value bet(s)"
            )

        except Exception as exc:
            logger.warning(f"  Skipping {match.get('player1')} vs {match.get('player2')}: {exc}")
            continue

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "matches":    sorted(output_matches, key=lambda x: x["date"]),
    }

    out_path = OUTPUT_DIR / "recommendations.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nRecommendations written to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline()
