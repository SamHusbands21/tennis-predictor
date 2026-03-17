"""
Betfair Exchange API client for tennis.

Fetches upcoming ATP tennis match odds using betfairlightweight.
Requires credentials in .env:
  BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY
  BETFAIR_CERT_PATH, BETFAIR_KEY_PATH

Tennis MATCH_ODDS markets have exactly two runners (the two players — no draw),
so the parsing is simpler than football.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Betfair event type IDs
TENNIS_EVENT_TYPE_ID = "2"
MATCH_ODDS_MARKET = "MATCH_ODDS"


def _get_client():
    """Authenticate and return a betfairlightweight trading client."""
    import betfairlightweight as bfl

    username = os.environ["BETFAIR_USERNAME"]
    password = os.environ["BETFAIR_PASSWORD"]
    app_key = os.environ["BETFAIR_APP_KEY"]
    certs_dir = str(Path(__file__).parents[2] / "certs")

    client = bfl.APIClient(
        username=username,
        password=password,
        app_key=app_key,
        certs=certs_dir,
    )
    client.login()
    logger.info("Betfair login successful")
    return client


def get_upcoming_atp_matches(days_ahead: int = 7) -> list[dict]:
    """
    Return upcoming ATP tennis matches with best back odds from the Exchange.

    Returns a list of dicts:
      {
        "player1": str,
        "player2": str,
        "tournament": str,
        "date": datetime (UTC),
        "betfair_odds": {"p1": float, "p2": float},
        "market_id": str,
      }
    """
    import betfairlightweight.filters as bfl_filters

    trading = _get_client()

    now = datetime.now(timezone.utc)
    from_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    to_time = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%SZ")

    market_filter = bfl_filters.market_filter(
        event_type_ids=[TENNIS_EVENT_TYPE_ID],
        market_type_codes=[MATCH_ODDS_MARKET],
        market_start_time=bfl_filters.time_range(from_=from_time, to=to_time),
    )

    catalogues = trading.betting.list_market_catalogue(
        filter=market_filter,
        market_projection=["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
        max_results=200,
    )

    if not catalogues:
        logger.info("No upcoming tennis markets found.")
        trading.logout()
        return []

    # Filter to men's singles markets: event names typically contain "ATP" or "Men's Singles"
    atp_cats = []
    for c in catalogues:
        event_name = (c.event.name or "") if c.event else ""
        # Keep ATP events; skip WTA, ITF, Davis Cup
        name_lower = event_name.lower()
        if any(skip in name_lower for skip in ["wta", "women", "itf", "davis", "fed cup", "billie"]):
            continue
        if len(c.runners or []) == 2:  # tennis singles: exactly 2 runners
            atp_cats.append(c)

    if not atp_cats:
        logger.info("No ATP singles markets found.")
        trading.logout()
        return []

    market_ids = [c.market_id for c in atp_cats]

    # Betfair rejects requests with too many market IDs — batch in chunks of 40
    price_proj = bfl_filters.price_projection(price_data=["EX_BEST_OFFERS"])
    market_books = []
    batch_size = 40
    for i in range(0, len(market_ids), batch_size):
        batch = market_ids[i : i + batch_size]
        market_books.extend(
            trading.betting.list_market_book(
                market_ids=batch,
                price_projection=price_proj,
            )
        )

    id_to_book = {b.market_id: b for b in market_books}
    matches = []

    for cat in atp_cats:
        book = id_to_book.get(cat.market_id)
        if book is None:
            continue

        runners = cat.runners or []
        if len(runners) != 2:
            continue

        runner_map = {r.selection_id: r.runner_name for r in runners}
        odds = {}
        for runner in book.runners:
            name = runner_map.get(runner.selection_id)
            if not name:
                continue
            best_back = (
                runner.ex.available_to_back[0].price
                if runner.ex and runner.ex.available_to_back
                else None
            )
            if best_back:
                odds[name] = best_back

        runner_names = [r.runner_name for r in runners]
        if len(runner_names) < 2:
            continue

        player1, player2 = runner_names[0], runner_names[1]

        event_name = cat.event.name if cat.event else "Unknown Tournament"
        # Strip year from tournament name for cleaner display
        tournament = event_name.split(" - ")[0].strip() if " - " in event_name else event_name

        try:
            match = {
                "player1": player1,
                "player2": player2,
                "tournament": tournament,
                "date": cat.market_start_time,
                "betfair_odds": {
                    "p1": odds.get(player1),
                    "p2": odds.get(player2),
                },
                "market_id": cat.market_id,
            }
            matches.append(match)
            logger.info(
                f"  {player1} vs {player2} [{tournament}] | "
                f"P1:{odds.get(player1)} P2:{odds.get(player2)}"
            )
        except Exception as exc:
            logger.warning(f"  Skipping market {cat.market_id}: {exc}")
            continue

    trading.logout()
    logger.info(f"Found {len(matches)} upcoming ATP singles matches.")
    return matches


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    matches = get_upcoming_atp_matches(days_ahead=7)
    for m in matches:
        print(m)
