"""Market-implied probability loaders and de-vigging helpers."""

from __future__ import annotations  # consistent type-hint behaviour

import json  # parse cached live market API responses
import re  # extract teams and odds from market strings
from datetime import datetime, timezone  # snapshot timestamps
from pathlib import Path  # optional raw file paths

import httpx  # live public API/page pulls for the market snapshot
import numpy as np  # probability vectors
import pandas as pd  # file IO and table outputs

from gyan.config import (  # market configuration and live raw source paths
    BOOKMAKER_WORLD_CUP_WINNER_FILE,
    DEVIG_METHOD,
    KALSHI_WORLD_CUP_WINNER_FILE,
    MARKET_SOURCE_WEIGHTS_LIVE,
    POLYMARKET_WORLD_CUP_WINNER_FILE,
)
from gyan.features.socioeconomic import canonical_football_team  # shared team aliases
from gyan.ensemble.pooling import linear_opinion_pool  # within-market source pooling


MARKET_TEAM_ALIASES: dict[str, str] = {  # market labels -> project labels
    "USA": "United States",
    "USMNT": "United States",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
    "DR Congo": "Democratic Republic of the Congo",
    "Congo DR": "Democratic Republic of the Congo",
    "Curaçao": "Curacao",
    "Curacao": "Curacao",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
}

DEFAULT_MARKET_SOURCE_PATHS: dict[str, Path] = {  # live/cached raw source locations
    "polymarket": POLYMARKET_WORLD_CUP_WINNER_FILE,
    "kalshi": KALSHI_WORLD_CUP_WINNER_FILE,
    "bookmaker": BOOKMAKER_WORLD_CUP_WINNER_FILE,
}

LIVE_MARKET_URLS: dict[str, str] = {  # public endpoints/pages, no auth required
    "polymarket": "https://gamma-api.polymarket.com/events/slug/world-cup-winner",
    "kalshi": "https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker=KXMENWORLDCUP-26&limit=100",
    "bookmaker": "https://www.bookmakersreview.com/fifa-world-cup/",
}


def canonical_market_team(team: object) -> str:
    """Return the project team name for a market-source team label."""
    label = str(team).strip()  # normalise text input
    return canonical_football_team(MARKET_TEAM_ALIASES.get(label, label))  # shared aliases


def devig(vector: pd.Series, method: str = DEVIG_METHOD) -> pd.Series:
    """Return a de-vigged probability vector.

    Parameters
    ----------
    vector : pandas.Series
        Raw implied probabilities indexed by team.
    method : str
        De-vig method; only `"proportional"` is currently implemented.

    Returns
    -------
    pandas.Series
        Probability vector summing to one.
    """
    if method != "proportional":  # only method specified by config/PRD
        raise ValueError(f"Unsupported devig method: {method}")  # fail loudly
    clean = vector.astype(float).clip(lower=0.0)  # remove impossible negative values
    total = float(clean.sum())  # raw implied mass
    if total <= 0.0:  # all-zero source cannot be de-vigged
        raise ValueError("market vector has no positive probability mass")  # fail loudly
    return clean / total  # proportional overround removal


def _read_vector_file(path: Path | str, team_index: list[str], value_col: str = "raw_probability") -> pd.Series:
    """Read a raw market vector from CSV or parquet.

    Parameters
    ----------
    path : Path | str
        File containing `team` and probability columns.
    team_index : list[str]
        Shared team ordering.
    value_col : str
        Probability column name.

    Returns
    -------
    pandas.Series
        Raw implied probabilities aligned to `team_index`.
    """
    file_path = Path(path)  # normalise path
    if file_path.suffix == ".parquet":  # parquet source
        frame = pd.read_parquet(file_path)  # read parquet
    else:  # CSV or text source
        frame = pd.read_csv(file_path)  # read CSV
    lookup = dict(zip(frame["team"], frame[value_col], strict=False))  # team -> raw probability
    return pd.Series({team: float(lookup.get(team, 0.0)) for team in team_index}, dtype=float)  # aligned vector


def _json_list(value: object) -> list[object]:
    """Parse a Gamma JSON-encoded array field."""
    if isinstance(value, list):  # already parsed
        return value
    if value is None:  # absent optional field
        return []
    return json.loads(str(value))  # Gamma stores arrays as JSON strings


def _midpoint_probability(bid: object, ask: object, fallback: object) -> float:
    """Return a source price using bid/ask midpoint with fallback."""
    bid_value = pd.to_numeric(bid, errors="coerce")  # best bid
    ask_value = pd.to_numeric(ask, errors="coerce")  # best ask
    if not pd.isna(bid_value) and not pd.isna(ask_value) and ask_value >= bid_value:  # valid spread
        return float((bid_value + ask_value) / 2.0)  # midpoint
    fallback_value = pd.to_numeric(fallback, errors="coerce")  # last/outcome price
    if pd.isna(fallback_value):  # no usable price
        return 0.0  # absent market price
    return float(fallback_value)  # fallback price


def parse_polymarket_event(path: Path | str) -> pd.DataFrame:
    """Parse cached Polymarket Gamma event JSON into raw YES probabilities."""
    event = json.loads(Path(path).read_text(encoding="utf-8"))  # load raw event
    rows: list[dict[str, object]] = []  # collect markets
    for market in event.get("markets", []):  # one binary country market
        if not market.get("active") or market.get("closed"):  # open live markets only
            continue  # skip inactive/resolved
        question = str(market.get("question", ""))  # e.g. Will Spain win...
        match = re.search(r"Will (?:the )?(.+?) win the 2026 FIFA World Cup", question)  # team
        if not match:  # skip non-team rows/placeholders
            continue  # not a winner country market
        prices = _json_list(market.get("outcomePrices"))  # [yes, no]
        yes_fallback = prices[0] if prices else market.get("lastTradePrice")  # fallback price
        rows.append(  # source row
            {
                "team": canonical_market_team(match.group(1)),
                "raw_probability": _midpoint_probability(market.get("bestBid"), market.get("bestAsk"), yes_fallback),
                "source_market_id": market.get("id"),
                "source_question": question,
            }
        )
    frame = pd.DataFrame(rows)  # source table
    if frame.empty:  # failed parse/source unavailable
        raise ValueError(f"No active Polymarket World Cup winner prices parsed from {path}")  # fail
    return frame.groupby("team", as_index=False)["raw_probability"].max()  # one row per team


def parse_kalshi_markets(path: Path | str) -> pd.DataFrame:
    """Parse cached Kalshi market JSON into raw YES probabilities."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))  # load raw API response
    rows: list[dict[str, object]] = []  # collect market rows
    for market in payload.get("markets", []):  # one country market
        if market.get("status") != "active":  # active live markets only
            continue  # skip inactive
        team = market.get("yes_sub_title")  # clean country label in Kalshi API
        if not team:  # fallback parse title
            match = re.search(r"Will (?:the )?(.+?) win the 2026 Men's World Cup", str(market.get("title", "")))
            team = match.group(1) if match else None  # parsed team
        if not team:  # not a country market
            continue  # skip
        rows.append(  # source row
            {
                "team": canonical_market_team(team),
                "raw_probability": _midpoint_probability(
                    market.get("yes_bid_dollars"),
                    market.get("yes_ask_dollars"),
                    market.get("last_price_dollars"),
                ),
                "source_market_id": market.get("ticker"),
                "source_question": market.get("title"),
            }
        )
    frame = pd.DataFrame(rows)  # source table
    if frame.empty:  # failed parse/source unavailable
        raise ValueError(f"No active Kalshi World Cup winner prices parsed from {path}")  # fail
    return frame.groupby("team", as_index=False)["raw_probability"].max()  # one row per team


def _american_to_probability(value: object) -> float:
    """Convert American odds to raw implied probability."""
    text = str(value).strip().replace("+", "")  # normalise +450
    if text in {"", "—", "-", "nan"}:  # missing odds cell
        return float("nan")  # no price
    odds = float(text)  # American odds
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)  # implied


def parse_bookmaker_html(path: Path | str) -> pd.DataFrame:
    """Parse cached bookmaker odds-comparison HTML into median raw probabilities."""
    tables = pd.read_html(path)  # bookmaker table is an HTML table
    odds_tables = [table for table in tables if "Best Odds" in table.columns]  # bookmaker odds grid
    if not odds_tables:  # failed parse/source unavailable
        raise ValueError(f"No bookmaker odds table parsed from {path}")  # fail
    table = odds_tables[0].copy()  # first odds grid
    team_col = table.columns[0]  # unnamed country column
    bookmaker_cols = [col for col in table.columns if col not in {team_col, "Best Odds"}]  # sportsbook odds columns
    rows: list[dict[str, object]] = []  # collect one row per team
    for _, row in table.iterrows():  # one team row
        team = canonical_market_team(row[team_col])  # first column is team name
        probabilities = []  # sportsbook implied probabilities
        for value in row[bookmaker_cols]:  # each bookmaker quote
            probability = _american_to_probability(value)  # raw implied probability
            if not np.isnan(probability):  # usable odds
                probabilities.append(probability)  # collect
        if probabilities:  # only keep quoted teams
            rows.append({"team": team, "raw_probability": float(np.median(probabilities))})  # median book
    frame = pd.DataFrame(rows)  # source table
    if frame.empty:  # failed parse/source unavailable
        raise ValueError(f"No bookmaker prices parsed from {path}")  # fail
    return frame.groupby("team", as_index=False)["raw_probability"].max()  # one row per team


def _align_source_frame(frame: pd.DataFrame, team_index: list[str]) -> pd.Series:
    """Align a parsed source frame to the tournament team index."""
    lookup = dict(zip(frame["team"], frame["raw_probability"], strict=False))  # team -> probability
    vector = pd.Series({team: float(lookup.get(team, 0.0)) for team in team_index}, dtype=float)  # aligned
    if vector.sum() <= 0.0:  # no usable prices for this source/team field
        raise ValueError("market source has no positive probability mass for the 2026 field")  # fail
    return vector  # raw implied vector


def load_polymarket_vector(team_index: list[str], raw_path: Path | str | None = None, anchor: pd.Series | None = None) -> pd.Series:
    """Load a Polymarket winner vector from cached Gamma JSON.

    Parameters
    ----------
    team_index : list[str]
        Shared team ordering.
    raw_path : Path | str | None
        Optional cached raw source file.
    anchor : pandas.Series | None
        Deprecated; ignored. Kept for API compatibility.

    Returns
    -------
    pandas.Series
        Raw implied probabilities.
    """
    if raw_path is None:  # live source cache is required
        raise ValueError("Polymarket raw_path is required; benchmark proxy fallback has been removed")  # fail
    if Path(raw_path).suffix in {".csv", ".parquet"}:  # prepared vector for tests/manual cache
        return _read_vector_file(raw_path, team_index)  # parse generic vector
    return _align_source_frame(parse_polymarket_event(raw_path), team_index)  # parse and align


def load_kalshi_vector(team_index: list[str], raw_path: Path | str | None = None, anchor: pd.Series | None = None) -> pd.Series:
    """Load a Kalshi winner vector from cached API JSON."""
    if raw_path is None:  # live source cache is required
        raise ValueError("Kalshi raw_path is required; benchmark proxy fallback has been removed")  # fail
    if Path(raw_path).suffix in {".csv", ".parquet"}:  # prepared vector for tests/manual cache
        return _read_vector_file(raw_path, team_index)  # parse generic vector
    return _align_source_frame(parse_kalshi_markets(raw_path), team_index)  # parse and align


def load_bookmaker_vector(team_index: list[str], raw_path: Path | str | None = None, anchor: pd.Series | None = None) -> pd.Series:
    """Load a bookmaker-consensus winner vector from cached odds HTML/CSV."""
    if raw_path is None:  # live source cache is required
        raise ValueError("Bookmaker raw_path is required; benchmark proxy fallback has been removed")  # fail
    file_path = Path(raw_path)  # normalise
    if file_path.suffix in {".csv", ".parquet"}:  # allow test/local prepared vectors
        return _read_vector_file(file_path, team_index)  # parse generic vector
    return _align_source_frame(parse_bookmaker_html(file_path), team_index)  # parse and align


def refresh_live_market_source_cache(
    source_paths: dict[str, Path | str] | None = None,
    force: bool = True,
) -> dict[str, Path]:
    """Pull current public market source files into the raw cache."""
    paths = {key: Path(value) for key, value in (source_paths or DEFAULT_MARKET_SOURCE_PATHS).items()}  # paths
    headers = {"User-Agent": "Mozilla/5.0 GYAN-WorldCupModel/0.1", "Accept": "*/*"}  # public data UA
    with httpx.Client(headers=headers, timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        for source, url in LIVE_MARKET_URLS.items():  # one source at a time
            path = paths[source]  # destination
            if path.exists() and not force:  # cached source allowed
                continue  # keep existing file
            try:
                response = client.get(url)  # live pull
                response.raise_for_status()  # fail on blocked/bad response
            except (httpx.HTTPStatusError, httpx.RequestError):
                if path.exists():  # preserve last raw evidence when the public endpoint throttles/errors
                    continue  # source parsers will validate the existing raw file
                raise  # no raw evidence exists for this required source
            path.parent.mkdir(parents=True, exist_ok=True)  # ensure raw dir
            path.write_bytes(response.content)  # cache raw response
    return paths  # raw source paths


def build_live_market_snapshot(
    team_index: list[str],
    output_path: Path | str,
    anchor: pd.Series | None = None,
    source_paths: dict[str, Path | str] | None = None,
    snapshot_time_utc: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build and persist the live market-implied champion vector.

    Parameters
    ----------
    team_index : list[str]
        Shared team ordering.
    output_path : Path | str
        Processed parquet path to write.
    anchor : pandas.Series | None
        Optional long-tail allocation anchor.
    source_paths : dict[str, Path | str] | None
        Optional cached source files keyed by source name.
    snapshot_time_utc : str | None
        Snapshot timestamp override; defaults to current UTC.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, object]]
        Persisted market table and metadata.
    """
    paths = source_paths or DEFAULT_MARKET_SOURCE_PATHS  # live raw source file map
    missing_sources = [source for source in MARKET_SOURCE_WEIGHTS_LIVE if source not in paths]  # required sources
    if missing_sources:  # all configured live market sources must be explicit
        raise ValueError(f"Missing live market source paths: {missing_sources}")  # fail
    timestamp = snapshot_time_utc or datetime.now(timezone.utc).isoformat()  # market snapshot time
    raw_vectors = {  # raw implied probabilities per source
        "polymarket": load_polymarket_vector(team_index, paths.get("polymarket"), anchor),
        "kalshi": load_kalshi_vector(team_index, paths.get("kalshi"), anchor),
        "bookmaker": load_bookmaker_vector(team_index, paths.get("bookmaker"), anchor),
    }
    devigged = {source: devig(vector) for source, vector in raw_vectors.items()}  # de-vig each source
    source_order = list(MARKET_SOURCE_WEIGHTS_LIVE.keys())  # configured source order
    source_weights = np.asarray([MARKET_SOURCE_WEIGHTS_LIVE[source] for source in source_order], dtype=float)  # weights
    source_matrix = np.vstack([devigged[source].reindex(team_index).to_numpy(dtype=float) for source in source_order])  # matrix
    pooled = linear_opinion_pool(source_matrix, source_weights)  # within-market linear pool
    frame = pd.DataFrame(  # output table
        {
            "team": team_index,
            "p_champion": pooled,
            "snapshot_time_utc": timestamp,
            "market_source_note": "live_market_pull_cached_raw",
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # ensure processed directory
    frame.to_parquet(output_path, index=False)  # persist processed market vector
    metadata = {  # run-record metadata
        "snapshot_time_utc": timestamp,
        "source_weights": MARKET_SOURCE_WEIGHTS_LIVE,
        "source_note": frame["market_source_note"].iloc[0],
        "devig_method": DEVIG_METHOD,
        "source_paths": {key: str(value) for key, value in paths.items()},
    }
    return frame, metadata  # market table and metadata
