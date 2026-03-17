"""
Standard Elo rating system for tennis match outcomes.

No home advantage is applied — tennis matches are played at neutral tournament
venues. Each match has a clear winner and loser (no draws).

Ratings are computed in chronological order; the value stored for each match is
the rating BEFORE that match was played. This prevents any data leakage.

Usage:
    from src.features.elo import EloSystem
    elo = EloSystem(k=32, initial_rating=1500)
    ratings_df = elo.fit_transform(matches_df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EloSystem:
    k: float = 32.0
    initial_rating: float = 1500.0
    ratings: dict[int, float] = field(default_factory=dict)

    def _get_rating(self, player_id: int) -> float:
        return self.ratings.get(int(player_id), self.initial_rating)

    def _expected(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def process_match(
        self,
        winner_id: int,
        loser_id: int,
    ) -> tuple[float, float]:
        """
        Update ratings from one match (winner beat loser).
        Returns (pre_match_winner_elo, pre_match_loser_elo).
        """
        pre_winner = self._get_rating(winner_id)
        pre_loser = self._get_rating(loser_id)

        e_winner = self._expected(pre_winner, pre_loser)
        e_loser = 1.0 - e_winner

        self.ratings[int(winner_id)] = pre_winner + self.k * (1.0 - e_winner)
        self.ratings[int(loser_id)] = pre_loser + self.k * (0.0 - e_loser)

        return pre_winner, pre_loser

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches chronologically. Adds elo_winner and elo_loser
        columns to the DataFrame containing pre-match ratings.

        Expects columns: date, winner_id, loser_id
        """
        matches = matches.sort_values("date").reset_index(drop=True)
        self.ratings = {}

        pre_winner_elos, pre_loser_elos = [], []

        for _, row in matches.iterrows():
            w_elo, l_elo = self.process_match(
                int(row["winner_id"]), int(row["loser_id"])
            )
            pre_winner_elos.append(w_elo)
            pre_loser_elos.append(l_elo)

        result = matches.copy()
        result["elo_winner"] = pre_winner_elos
        result["elo_loser"] = pre_loser_elos
        return result

    def get_current_ratings(self) -> dict[int, float]:
        """Return post-all-matches Elo ratings for every player seen."""
        return dict(self.ratings)
