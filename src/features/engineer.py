"""
Full feature engineering pipeline for the tennis ATP betting model.

Takes the raw ATP match DataFrame (from sackmann.py), computes Elo ratings,
surface-specific Elo, rolling per-player statistics, and contextual features.
All features are strictly pre-match — no information from the current match
or any later matches is ever used (shift-then-roll pattern throughout).

Binary classification normalisation
------------------------------------
Raw Sackmann data is always winner/loser. To prevent the model from learning
"the player in column 1 always wins", each match is randomly assigned to
player1/player2 with the winner assigned as player1 approximately 50% of the
time. The random seed is fixed for reproducibility.

Output parquet columns
-----------------------
  Identifiers:  date, player1_id, player2_id, player1_name, player2_name,
                surface, tournament, tourney_level, round, year
  Elo:          elo_p1, elo_p2, elo_diff,
                surface_elo_p1, surface_elo_p2, surface_elo_diff
  Form:         p1_win_rate_5, p2_win_rate_5,
                p1_win_rate_10, p2_win_rate_10,
                p1_surface_win_rate_10, p2_surface_win_rate_10
  Serve:        p1_first_serve_pct, p2_first_serve_pct,
                p1_first_serve_won_pct, p2_first_serve_won_pct,
                p1_bp_save_rate, p2_bp_save_rate
  Fatigue:      p1_days_rest, p2_days_rest,
                p1_prev_match_minutes, p2_prev_match_minutes
  Context:      rank_diff, p1_rank_pts, p2_rank_pts,
                h2h_p1_win_rate, tourney_level_encoded, best_of,
                surface_hard, surface_clay, surface_grass
  Target:       target  (1 = player1 wins, 0 = player2 wins)
  Odds:         odds_p1, odds_p2  (kept for evaluation; NOT fed to model)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from typing import Optional

import numpy as np
import pandas as pd

from src.features.elo import EloSystem
from src.features.surface_elo import SurfaceEloSystem, normalise_surface

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"
RANDOM_SEED = 42

# Rolling window sizes
FORM_WINDOW_SHORT = 5
FORM_WINDOW_LONG = 10
SURFACE_FORM_WINDOW = 10
SERVE_WINDOW = 10

MAX_DAYS_REST = 21         # cap days of rest — anything beyond 3 weeks is irrelevant
PREV_MINUTES_DEFAULT = 90  # assumed neutral match length for first known match

TOURNEY_LEVEL_MAP: dict[str, int] = {
    "G": 4,  # Grand Slam
    "M": 3,  # Masters (ATP 1000)
    "F": 3,  # ATP Finals
    "A": 2,  # ATP 500 / 250
    "D": 1,  # Davis Cup / team events
    "C": 1,  # Challenger
}


# ── Model feature list ────────────────────────────────────────────────────────

MODEL_FEATURES = [
    # Elo
    "elo_p1", "elo_p2", "elo_diff",
    "surface_elo_p1", "surface_elo_p2", "surface_elo_diff",
    # Form
    "p1_win_rate_5", "p2_win_rate_5",
    "p1_win_rate_10", "p2_win_rate_10",
    "p1_surface_win_rate_10", "p2_surface_win_rate_10",
    # Serve quality
    "p1_first_serve_pct", "p2_first_serve_pct",
    "p1_first_serve_won_pct", "p2_first_serve_won_pct",
    "p1_bp_save_rate", "p2_bp_save_rate",
    # Fatigue / recovery
    "p1_days_rest", "p2_days_rest",
    "p1_prev_match_minutes", "p2_prev_match_minutes",
    # Context
    "rank_diff",
    "p1_rank_pts", "p2_rank_pts",
    "h2h_p1_win_rate",
    "tourney_level_encoded",
    "best_of",
    # Surface one-hot
    "surface_hard", "surface_clay", "surface_grass",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _safe_div(num: Optional[float], denom: Optional[float], default: float = np.nan) -> float:
    if num is None or denom is None:
        return default
    try:
        n, d = float(num), float(denom)
    except (TypeError, ValueError):
        return default
    if np.isnan(n) or np.isnan(d) or d == 0:
        return default
    return n / d


def _serve_stats_for_player(row: pd.Series, won: bool) -> dict[str, float]:
    """Extract per-match serve stat ratios from a raw match row."""
    prefix = "w_" if won else "l_"
    svpt     = row.get(f"{prefix}svpt")
    first_in = row.get(f"{prefix}1stIn")
    first_won = row.get(f"{prefix}1stWon")
    bp_saved  = row.get(f"{prefix}bpSaved")
    bp_faced  = row.get(f"{prefix}bpFaced")

    return {
        "first_serve_pct":     _safe_div(first_in,  svpt),
        "first_serve_won_pct": _safe_div(first_won, first_in),
        "bp_save_rate":        _safe_div(bp_saved,  bp_faced),
    }


def _build_long_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long-form table: one row per (player, match).

    Each match contributes two rows — one for the winner and one for the loser.
    The 'won' column is 1 for the winner row and 0 for the loser row.
    """
    df = df.sort_values("date").reset_index(drop=True)
    winner_rows, loser_rows = [], []

    for _, row in df.iterrows():
        surface = normalise_surface(str(row.get("surface", "Hard")))
        minutes = row.get("minutes")
        if pd.isna(minutes):
            minutes = np.nan

        w_stats = _serve_stats_for_player(row, won=True)
        l_stats = _serve_stats_for_player(row, won=False)

        winner_rows.append({
            "date":                row["date"],
            "player_id":           int(row["winner_id"]),
            "surface":             surface,
            "won":                 1.0,
            "minutes":             float(minutes) if pd.notna(minutes) else np.nan,
            "first_serve_pct":     w_stats["first_serve_pct"],
            "first_serve_won_pct": w_stats["first_serve_won_pct"],
            "bp_save_rate":        w_stats["bp_save_rate"],
        })
        loser_rows.append({
            "date":                row["date"],
            "player_id":           int(row["loser_id"]),
            "surface":             surface,
            "won":                 0.0,
            "minutes":             float(minutes) if pd.notna(minutes) else np.nan,
            "first_serve_pct":     l_stats["first_serve_pct"],
            "first_serve_won_pct": l_stats["first_serve_won_pct"],
            "bp_save_rate":        l_stats["bp_save_rate"],
        })

    long = pd.DataFrame(winner_rows + loser_rows)
    long = long.sort_values(["player_id", "date"]).reset_index(drop=True)
    return long


def _rolling_player_stats(long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute shifted rolling statistics per player.

    The .shift(1) before rolling ensures the current match is never included
    in its own rolling window — preventing data leakage.

    Returns a DataFrame with columns:
        player_id, date,
        win_rate_5, win_rate_10,
        surface_win_rate_10_hard, surface_win_rate_10_clay, surface_win_rate_10_grass,
        first_serve_pct, first_serve_won_pct, bp_save_rate,
        prev_minutes, last_date
    """
    def rolling_shifted(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=1).mean()

    all_stats: list[pd.DataFrame] = []

    for player_id, grp in long.groupby("player_id"):
        grp = grp.sort_values("date").reset_index(drop=True)

        # Surface-specific win rate: shift(1) then rolling over surface-filtered values
        def surface_win_rate(surf: str) -> np.ndarray:
            # Set non-matching surface rows to NaN, then forward-fill over the window
            won_filtered = grp["won"].where(grp["surface"] == surf, other=np.nan)
            # shift + rolling while skipping NaN (min_periods=1 means we use whatever is available)
            return (
                won_filtered.shift(1)
                .rolling(SURFACE_FORM_WINDOW, min_periods=1)
                .mean()
                .values
            )

        player_df = pd.DataFrame({
            "player_id":                player_id,
            "date":                     grp["date"].values,
            "win_rate_5":               rolling_shifted(grp["won"], FORM_WINDOW_SHORT).values,
            "win_rate_10":              rolling_shifted(grp["won"], FORM_WINDOW_LONG).values,
            "surface_win_rate_10_hard": surface_win_rate("Hard"),
            "surface_win_rate_10_clay": surface_win_rate("Clay"),
            "surface_win_rate_10_grass":surface_win_rate("Grass"),
            "first_serve_pct":          rolling_shifted(grp["first_serve_pct"], SERVE_WINDOW).values,
            "first_serve_won_pct":      rolling_shifted(grp["first_serve_won_pct"], SERVE_WINDOW).values,
            "bp_save_rate":             rolling_shifted(grp["bp_save_rate"], SERVE_WINDOW).values,
            "prev_minutes":             grp["minutes"].shift(1).values,
            "last_date":                grp["date"].shift(1).values,
        })
        all_stats.append(player_df)

    return pd.concat(all_stats, ignore_index=True)


def _attach_player_stats(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    player_id_col: str,
    prefix: str,
    current_surface_col: str,
) -> pd.DataFrame:
    """
    Merge pre-match rolling stats onto the match DataFrame for one player role.

    For the surface-specific win rate, we pick the column that matches the
    current match's surface rather than keeping all three surface columns.
    """
    # Rename stats columns with the given prefix
    rename_map = {
        "win_rate_5":               f"{prefix}_win_rate_5",
        "win_rate_10":              f"{prefix}_win_rate_10",
        "first_serve_pct":          f"{prefix}_first_serve_pct",
        "first_serve_won_pct":      f"{prefix}_first_serve_won_pct",
        "bp_save_rate":             f"{prefix}_bp_save_rate",
        "prev_minutes":             f"{prefix}_prev_match_minutes",
        "last_date":                f"_{prefix}_last_date",
        # Surface columns kept separately then merged
        "surface_win_rate_10_hard":  f"_{prefix}_swr_hard",
        "surface_win_rate_10_clay":  f"_{prefix}_swr_clay",
        "surface_win_rate_10_grass": f"_{prefix}_swr_grass",
    }
    stats_renamed = stats.rename(columns=rename_map)
    stats_renamed = stats_renamed.rename(columns={"player_id": player_id_col})

    df = df.merge(
        stats_renamed,
        on=[player_id_col, "date"],
        how="left",
    )

    # Pick the surface-specific win rate matching the current match surface
    surface_map = {
        "Hard":  f"_{prefix}_swr_hard",
        "Clay":  f"_{prefix}_swr_clay",
        "Grass": f"_{prefix}_swr_grass",
    }
    surf_col = f"{prefix}_surface_win_rate_10"
    df[surf_col] = df.apply(
        lambda r: r.get(surface_map.get(normalise_surface(str(r[current_surface_col])),
                                        f"_{prefix}_swr_hard")),
        axis=1,
    )
    # Drop temporary columns
    df = df.drop(columns=[c for c in [f"_{prefix}_swr_hard", f"_{prefix}_swr_clay",
                                       f"_{prefix}_swr_grass"] if c in df.columns])

    # Compute days rest from last_date
    last_date_col = f"_{prefix}_last_date"
    if last_date_col in df.columns:
        df[f"{prefix}_days_rest"] = (
            df["date"] - pd.to_datetime(df[last_date_col])
        ).dt.days.clip(upper=MAX_DAYS_REST).fillna(MAX_DAYS_REST)
        df = df.drop(columns=[last_date_col])

    return df


def _h2h_win_rate(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """
    For each match row compute player1's win rate in the last n H2H meetings,
    strictly before the current match date.

    Processes rows in chronological order, building the H2H history incrementally
    to avoid any leakage. Default 0.5 when no prior meetings exist.
    """
    df = df.sort_values("date").reset_index(drop=True)
    rates = []
    # canonical_pair -> list of (date, winner_player_id)
    h2h: defaultdict[tuple, list] = defaultdict(list)

    for _, row in df.iterrows():
        p1 = int(row["player1_id"])
        p2 = int(row["player2_id"])
        key = (min(p1, p2), max(p1, p2))

        past = h2h[key]
        if not past:
            rates.append(0.5)
        else:
            last_n = past[-n:]
            p1_wins = sum(1 for _, winner in last_n if winner == p1)
            rates.append(p1_wins / len(last_n))

        # Record actual outcome (who won)
        winner_id = p1 if row["target"] == 1 else p2
        h2h[key].append((row["date"], winner_id))

    return pd.Series(rates, index=df.index)


def _random_flip(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Randomly assign each match's winner/loser to player1/player2 with ~50%
    probability. When flipped, player1 is the loser and target = 0; when not
    flipped, player1 is the winner and target = 1.

    This prevents the model from learning a positional bias.
    """
    rng = np.random.default_rng(seed)
    flipped = rng.integers(0, 2, size=len(df)).astype(bool)

    out = df.copy()

    # Swap identifiers and stats for flipped rows
    flip_pairs = [
        ("winner_id",        "loser_id"),
        ("winner_name",      "loser_name"),
        ("elo_winner",       "elo_loser"),
        ("surface_elo_winner", "surface_elo_loser"),
        ("winner_rank",      "loser_rank"),
        ("winner_rank_points", "loser_rank_points"),
    ]
    for col_a, col_b in flip_pairs:
        if col_a in out.columns and col_b in out.columns:
            tmp = out.loc[flipped, col_a].copy()
            out.loc[flipped, col_a] = out.loc[flipped, col_b]
            out.loc[flipped, col_b] = tmp

    out = out.rename(columns={
        "winner_id":            "player1_id",
        "loser_id":             "player2_id",
        "winner_name":          "player1_name",
        "loser_name":           "player2_name",
        "elo_winner":           "elo_p1",
        "elo_loser":            "elo_p2",
        "surface_elo_winner":   "surface_elo_p1",
        "surface_elo_loser":    "surface_elo_p2",
        "winner_rank":          "p1_rank",
        "loser_rank":           "p2_rank",
        "winner_rank_points":   "p1_rank_pts_raw",
        "loser_rank_points":    "p2_rank_pts_raw",
    })

    # Target: 1 = player1 wins (winner was assigned as p1 = not flipped)
    out["target"] = (~flipped).astype(int)
    return out


def build_features(
    raw_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Main feature engineering entrypoint.

    Takes the raw ATP match DataFrame (from sackmann.download_all()),
    computes all features, and returns a clean feature DataFrame.
    """
    logger.info(f"Starting feature engineering on {len(raw_df)} matches...")

    df = raw_df.sort_values("date").reset_index(drop=True)

    # ── Elo ratings ──────────────────────────────────────────────────────────
    logger.info("Computing overall Elo ratings...")
    elo = EloSystem(k=32)
    df = elo.fit_transform(df)

    logger.info("Computing surface Elo ratings...")
    s_elo = SurfaceEloSystem(k=32)
    df = s_elo.fit_transform(df)

    # ── Rolling player statistics ─────────────────────────────────────────────
    logger.info("Building player history for rolling stats...")
    long = _build_long_form(df)

    logger.info("Computing rolling per-player stats (shift-then-roll)...")
    player_stats = _rolling_player_stats(long)

    # ── Random flip: assign winner/loser → player1/player2 ───────────────────
    logger.info("Randomising player1/player2 assignment...")
    df = _random_flip(df)

    # ── Attach rolling stats for player1 ─────────────────────────────────────
    logger.info("Attaching rolling stats for player1...")
    df = _attach_player_stats(df, player_stats, "player1_id", "p1", "surface")

    # ── Attach rolling stats for player2 ─────────────────────────────────────
    logger.info("Attaching rolling stats for player2...")
    df = _attach_player_stats(df, player_stats, "player2_id", "p2", "surface")

    # ── Rank and context features ─────────────────────────────────────────────
    df["p1_rank_pts"] = pd.to_numeric(df["p1_rank_pts_raw"], errors="coerce").fillna(0)
    df["p2_rank_pts"] = pd.to_numeric(df["p2_rank_pts_raw"], errors="coerce").fillna(0)

    p1_rank = pd.to_numeric(df["p1_rank"], errors="coerce")
    p2_rank = pd.to_numeric(df["p2_rank"], errors="coerce")
    # rank_diff: positive means p1 is better ranked (lower rank number)
    df["rank_diff"] = p2_rank - p1_rank

    # Elo diffs
    df["elo_diff"] = df["elo_p1"] - df["elo_p2"]
    df["surface_elo_diff"] = df["surface_elo_p1"] - df["surface_elo_p2"]

    # Tournament level encoding
    df["tourney_level_encoded"] = (
        df["tourney_level"].map(TOURNEY_LEVEL_MAP).fillna(1).astype(int)
    )

    # best_of: 3 or 5 (Grand Slams are best of 5)
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)

    # Surface one-hot
    df["_surf_norm"] = df["surface"].apply(normalise_surface)
    df["surface_hard"]  = (df["_surf_norm"] == "Hard").astype(int)
    df["surface_clay"]  = (df["_surf_norm"] == "Clay").astype(int)
    df["surface_grass"] = (df["_surf_norm"] == "Grass").astype(int)
    df = df.drop(columns=["_surf_norm"])

    # ── H2H win rate ──────────────────────────────────────────────────────────
    logger.info("Computing head-to-head win rates...")
    df["h2h_p1_win_rate"] = _h2h_win_rate(df)

    # ── Odds placeholder (filled by caller or left NaN) ───────────────────────
    if "odds_p1" not in df.columns:
        df["odds_p1"] = np.nan
    if "odds_p2" not in df.columns:
        df["odds_p2"] = np.nan

    # ── Select output columns ─────────────────────────────────────────────────
    id_cols = [
        "date", "player1_id", "player2_id", "player1_name", "player2_name",
        "surface", "tourney_name", "tourney_level", "round", "year",
    ]
    out_cols = id_cols + MODEL_FEATURES + ["target", "odds_p1", "odds_p2"]

    missing = [c for c in out_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns (filled with NaN): {missing}")
        for c in missing:
            df[c] = np.nan

    out = df[[c for c in out_cols if c in df.columns]].copy()
    out = out.dropna(subset=["target", "player1_id", "player2_id"])

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        path = PROCESSED_DIR / "features.parquet"
        out.to_parquet(path, index=False)
        logger.info(f"Saved features to {path} ({len(out)} rows)")

    logger.info(
        f"Feature engineering complete. "
        f"Rows: {len(out)} | Target balance: "
        f"{out['target'].mean():.3f} (should be ~0.50)"
    )
    return out


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from src.collect.sackmann import download_all

    try:
        raw = download_all()
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    features = build_features(raw)
    print(features[MODEL_FEATURES + ["target"]].head())
    print(f"\nShape: {features.shape}")
    print(f"\nTarget distribution:\n{features['target'].value_counts()}")
    print(f"\nFeature null counts:\n{features[MODEL_FEATURES].isnull().sum().to_string()}")
