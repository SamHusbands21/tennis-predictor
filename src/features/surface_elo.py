"""
Surface-specific Elo rating system for tennis.

Maintains separate Elo rating pools for Hard, Clay, and Grass courts.
Only the rating for the surface on which a match was played is updated;
the other surface ratings are unchanged.

This captures surface specialisation — a dominant clay-court player like Nadal
carries a very different Clay Elo vs Grass Elo, which a single overall Elo
cannot represent.

Usage:
    from src.features.surface_elo import SurfaceEloSystem
    s_elo = SurfaceEloSystem(k=32)
    ratings_df = s_elo.fit_transform(matches_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

SURFACES = ("Hard", "Clay", "Grass")


def normalise_surface(surface: str) -> str:
    """Map raw surface strings (e.g. 'Hard', 'Clay', 'Grass', 'Carpet') to canonical keys."""
    s = str(surface).strip()
    if s.lower().startswith("hard"):
        return "Hard"
    if s.lower().startswith("clay"):
        return "Clay"
    if s.lower().startswith("grass"):
        return "Grass"
    # Carpet (very rare in modern ATP) treated as Hard for model purposes
    return "Hard"


@dataclass
class SurfaceEloSystem:
    k: float = 32.0
    initial_rating: float = 1500.0
    # {surface_name -> {player_id -> rating}}
    ratings: dict[str, dict[int, float]] = field(
        default_factory=lambda: {s: {} for s in SURFACES}
    )

    def _get_rating(self, surface: str, player_id: int) -> float:
        surf = normalise_surface(surface)
        return self.ratings[surf].get(int(player_id), self.initial_rating)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def process_match(
        self,
        winner_id: int,
        loser_id: int,
        surface: str,
    ) -> tuple[float, float]:
        """
        Update surface-specific ratings for one match.
        Returns (pre_match_winner_surface_elo, pre_match_loser_surface_elo).
        """
        surf = normalise_surface(surface)
        pre_winner = self._get_rating(surf, winner_id)
        pre_loser = self._get_rating(surf, loser_id)

        e_winner = self._expected(pre_winner, pre_loser)
        e_loser = 1.0 - e_winner

        self.ratings[surf][int(winner_id)] = pre_winner + self.k * (1.0 - e_winner)
        self.ratings[surf][int(loser_id)] = pre_loser + self.k * (0.0 - e_loser)

        return pre_winner, pre_loser

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches chronologically. Adds surface_elo_winner and
        surface_elo_loser columns with pre-match surface-specific ratings.

        Expects columns: date, winner_id, loser_id, surface
        """
        matches = matches.sort_values("date").reset_index(drop=True)
        self.ratings = {s: {} for s in SURFACES}

        pre_winner_elos, pre_loser_elos = [], []

        for _, row in matches.iterrows():
            w_elo, l_elo = self.process_match(
                int(row["winner_id"]), int(row["loser_id"]),
                str(row.get("surface", "Hard")),
            )
            pre_winner_elos.append(w_elo)
            pre_loser_elos.append(l_elo)

        result = matches.copy()
        result["surface_elo_winner"] = pre_winner_elos
        result["surface_elo_loser"] = pre_loser_elos
        return result

    def get_current_ratings(self) -> dict[str, dict[int, float]]:
        """Return all current surface-specific ratings per player."""
        return {s: dict(r) for s, r in self.ratings.items()}
