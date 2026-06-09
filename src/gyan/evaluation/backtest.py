"""Leakage-guarded Stage 4 backtest helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gyan.config import (
    ELO_HOME_ADVANTAGE,
    ELO_INITIAL_RATING,
    ELO_DIVISOR,
    HOFFMANN_2002_COEFFICIENTS,
    LATIN_FOOTBALL_NATIONS,
    NEGATIVE_BINOMIAL_DISPERSION,
    TEMP_OPTIMUM_C,
)
from gyan.ensemble.experts import outcome_vector_from_score
from gyan.engine.dixon_coles import correlated_negative_binomial_matrix, match_outcome_probs
from gyan.evaluation.scoring import brier_score, log_loss_single, mean_scores


@dataclass(frozen=True)
class TournamentBacktest:
    """Prepared data for one historical World Cup backtest."""

    year: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    teams: tuple[str, ...]
    matches: pd.DataFrame
    champion: str
    finalists: tuple[str, str]
    semifinalists: tuple[str, ...]
    leakage_audit: dict[str, object]


def world_cup_matches(matches: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return final-tournament matches for a World Cup year."""
    dates = pd.to_datetime(matches["date"])
    mask = (
        (dates.dt.year == year)
        & (matches["tournament"] == "FIFA World Cup")
    )
    return matches.loc[mask].sort_values(["date", "home_team", "away_team"], kind="mergesort").reset_index(drop=True)


def prepare_tournament_backtest(matches: pd.DataFrame, shootouts: pd.DataFrame, year: int) -> TournamentBacktest:
    """Build one historical tournament payload and leakage audit."""
    tournament_matches = world_cup_matches(matches, year)
    if tournament_matches.empty:
        raise ValueError(f"No FIFA World Cup final-tournament matches found for {year}")
    start_date = pd.Timestamp(tournament_matches["date"].min())
    end_date = pd.Timestamp(tournament_matches["date"].max())
    teams = tuple(sorted(set(tournament_matches["home_team"]) | set(tournament_matches["away_team"])))
    final = tournament_matches.iloc[-1]
    champion = knockout_winner(final, shootouts)
    finalists = tuple(sorted((str(final["home_team"]), str(final["away_team"]))))
    semifinal_rows = tournament_matches.iloc[-4:-2]
    semifinalists = tuple(sorted(set(semifinal_rows["home_team"]) | set(semifinal_rows["away_team"])))
    training = matches[pd.to_datetime(matches["date"]) < start_date]
    max_train_date = pd.Timestamp(training["date"].max()) if not training.empty else pd.NaT
    leakage_audit = {
        "passed": bool(pd.notna(max_train_date) and max_train_date < start_date),
        "max_training_date": None if pd.isna(max_train_date) else str(max_train_date.date()),
        "tournament_start": str(start_date.date()),
        "training_rows": int(len(training)),
        "tournament_rows": int(len(tournament_matches)),
    }
    return TournamentBacktest(
        year=year,
        start_date=start_date,
        end_date=end_date,
        teams=teams,
        matches=tournament_matches,
        champion=champion,
        finalists=finalists,
        semifinalists=semifinalists,
        leakage_audit=leakage_audit,
    )


def knockout_winner(match: pd.Series, shootouts: pd.DataFrame) -> str:
    """Return the winner of a knockout match, using shootouts for drawn finals."""
    home = str(match["home_team"])
    away = str(match["away_team"])
    if int(match["home_goals"]) > int(match["away_goals"]):
        return home
    if int(match["away_goals"]) > int(match["home_goals"]):
        return away
    date = pd.Timestamp(match["date"]).date().isoformat()
    shootout_match = shootouts[
        (pd.to_datetime(shootouts["date"]).dt.date.astype(str) == date)
        & (shootouts["home_team"] == home)
        & (shootouts["away_team"] == away)
    ]
    if shootout_match.empty:
        raise ValueError(f"Drawn knockout match has no shootout winner: {date} {home} vs {away}")
    return str(shootout_match.iloc[-1]["winner"])


def elo_snapshot(matches: pd.DataFrame, teams: tuple[str, ...], start_date: pd.Timestamp) -> pd.Series:
    """Return team Elo ratings strictly before a tournament start date."""
    ratings = {team: ELO_INITIAL_RATING for team in teams}
    prior = matches[pd.to_datetime(matches["date"]) < start_date].sort_values("date", kind="mergesort")
    for row in prior.itertuples(index=False):
        if row.home_team in ratings:
            ratings[row.home_team] = float(row.home_elo_post)
        if row.away_team in ratings:
            ratings[row.away_team] = float(row.away_elo_post)
    return pd.Series(ratings, name="elo_rating", dtype=float)


def form_snapshot(matches: pd.DataFrame, teams: tuple[str, ...], start_date: pd.Timestamp, months: int) -> pd.Series:
    """Return a no-leakage recent goal-difference form strength."""
    cutoff = start_date - pd.DateOffset(months=months)
    prior = matches[(pd.to_datetime(matches["date"]) < start_date) & (pd.to_datetime(matches["date"]) >= cutoff)]
    scores: dict[str, list[float]] = {team: [] for team in teams}
    for row in prior.itertuples(index=False):
        age_days = max((start_date - pd.Timestamp(row.date)).days, 1)
        weight = float(np.exp(-age_days / 540.0))
        if row.home_team in scores:
            scores[row.home_team].append(weight * (int(row.home_goals) - int(row.away_goals)))
        if row.away_team in scores:
            scores[row.away_team].append(weight * (int(row.away_goals) - int(row.home_goals)))
    return pd.Series({team: float(np.mean(values)) if values else 0.0 for team, values in scores.items()}, dtype=float)


def load_world_bank_series(path: Path | str) -> pd.DataFrame:
    """Load one cached World Bank API response into country-year rows."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["world_bank_code", "year", "value"])
    return pd.DataFrame(
        {
            "world_bank_code": frame["countryiso3code"],
            "year": frame["date"].astype(int),
            "value": pd.to_numeric(frame["value"], errors="coerce"),
        }
    )


def socioeconomic_snapshot(
    teams: tuple[str, ...],
    year: int,
    feature_map: pd.DataFrame,
    gdp: pd.DataFrame,
    population: pd.DataFrame,
    elo: pd.Series,
) -> pd.Series:
    """Return historical socioeconomic strengths using year-available macro rows."""
    team_features = feature_map.drop_duplicates("team").set_index("team")
    gdp_lookup = _latest_year_lookup(gdp, year - 1)
    population_lookup = _latest_year_lookup(population, year - 1)
    total_population = sum(value for _code, value in population_lookup.items() if pd.notna(value) and value > 0)
    values: dict[str, float] = {}
    for team in teams:
        if team not in team_features.index:
            values[team] = float(elo.get(team, ELO_INITIAL_RATING))
            continue
        row = team_features.loc[team]
        code = str(row.get("world_bank_code", ""))
        gdp_pc = float(gdp_lookup.get(code, np.nan))
        pop = float(population_lookup.get(code, np.nan))
        temp = float(row.get("mean_annual_temp_c", TEMP_OPTIMUM_C))
        latin = 1.0 if team in LATIN_FOOTBALL_NATIONS else 0.0
        pop_share = pop / total_population if total_population and pd.notna(pop) else 0.0
        coeffs = HOFFMANN_2002_COEFFICIENTS
        if not np.isfinite(gdp_pc):
            hoffmann = 0.0
        else:
            hoffmann = (
                coeffs["constant"]
                + coeffs["gnp_per_capita"] * gdp_pc
                + coeffs["gnp_per_capita_sq"] * (gdp_pc**2)
                + coeffs["temp_dev_sq"] * ((temp - TEMP_OPTIMUM_C) ** 2)
                + coeffs["latin_x_pop_share"] * latin * pop_share
            )
        values[team] = 0.55 * hoffmann + 0.45 * float(elo.get(team, ELO_INITIAL_RATING))
    return pd.Series(values, dtype=float)


def _latest_year_lookup(frame: pd.DataFrame, max_year: int) -> dict[str, float]:
    """Return latest non-null World Bank value at or before max_year by country."""
    filtered = frame[(frame["year"] <= max_year) & frame["value"].notna()].sort_values("year")
    return filtered.groupby("world_bank_code")["value"].last().to_dict()


def historical_expert_rating_snapshots(
    tournament: TournamentBacktest,
    matches: pd.DataFrame,
    feature_map: pd.DataFrame,
    gdp: pd.DataFrame,
    population: pd.DataFrame,
    market_outrights_path: Path | str,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """Build no-leakage historical rating snapshots for all four GYAN experts."""
    elo = elo_snapshot(matches, tournament.teams, tournament.start_date)
    nominal_form = form_snapshot(matches, tournament.teams, tournament.start_date, months=48)
    named_form = form_snapshot(matches, tournament.teams, tournament.start_date, months=12)
    nominal_scale = max(float(nominal_form.std(ddof=0)), 1.0)
    named_scale = max(float(named_form.std(ddof=0)), 1.0)
    yield_nominal = elo + 90.0 * (nominal_form / nominal_scale)
    yield_named = elo + 125.0 * (named_form / named_scale)
    socioeconomic = socioeconomic_snapshot(tournament.teams, tournament.year, feature_map, gdp, population, elo)
    market, market_champion, market_status = market_snapshot_from_outrights(
        tournament.teams,
        tournament.year,
        elo,
        market_outrights_path,
    )
    ratings = {
        "goal": elo.reindex(tournament.teams),
        "yield_nominal": yield_nominal.reindex(tournament.teams),
        "yield_named": yield_named.reindex(tournament.teams),
        "socioeconomic": socioeconomic.reindex(tournament.teams),
        "market": market.reindex(tournament.teams),
    }
    champion_overrides = {"market": market_champion.reindex(tournament.teams)}
    market_statuses = {"market": market_status}
    return ratings, champion_overrides, market_statuses


def market_proxy_snapshot(elo: pd.Series) -> pd.Series:
    """Return a no-leakage historical market proxy when outright odds are absent."""
    return elo.astype(float) + 35.0 * ((elo.astype(float) - elo.astype(float).mean()) / max(float(elo.std(ddof=0)), 1.0))


def load_historical_market_outrights(path: Path | str) -> pd.DataFrame:
    """Load D15 historical outright probabilities and normalize each tournament."""
    frame = pd.read_csv(path)
    required = {"year", "team", "raw_value", "raw_type", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Historical market file is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["year"] = frame["year"].astype(int)
    frame["raw_value"] = pd.to_numeric(frame["raw_value"], errors="coerce")
    if frame["raw_value"].isna().any():
        raise ValueError("Historical market file contains non-numeric raw values")
    raw_type = frame["raw_type"].astype(str)
    frame["p_champion"] = np.where(
        raw_type == "probability_percent",
        frame["raw_value"] / 100.0,
        np.where(raw_type == "decimal_odds", 1.0 / frame["raw_value"], np.nan),
    )
    if frame["p_champion"].isna().any():
        bad_types = sorted(frame.loc[frame["p_champion"].isna(), "raw_type"].astype(str).unique())
        raise ValueError(f"Unsupported historical market raw_type values: {bad_types}")
    frame["p_champion"] = frame["p_champion"].clip(lower=1e-6)
    totals = frame.groupby("year")["p_champion"].transform("sum")
    if (totals <= 0.0).any():
        raise ValueError("Historical market probabilities must have positive yearly totals")
    frame["p_champion"] = frame["p_champion"] / totals
    return frame[["year", "team", "p_champion", "source"]].sort_values(["year", "p_champion"], ascending=[True, False]).reset_index(drop=True)


def market_ratings_from_champion_probabilities(probabilities: pd.Series, anchor: pd.Series | None = None, temperature: float = 260.0) -> pd.Series:
    """Convert outright champion probabilities into ratings for match-level scoring."""
    probs = probabilities.astype(float).clip(lower=1e-6)
    probs = probs / float(probs.sum())
    centred_log = np.log(probs) - float(np.log(probs).mean())
    base = float(anchor.astype(float).mean()) if anchor is not None and not anchor.empty else ELO_INITIAL_RATING
    return pd.Series(base + temperature * centred_log, index=probs.index, dtype=float)


def market_snapshot_from_outrights(
    teams: tuple[str, ...],
    year: int,
    anchor: pd.Series,
    path: Path | str,
) -> tuple[pd.Series, pd.Series, str]:
    """Return market-derived ratings, direct champion probabilities, and source status."""
    frame = load_historical_market_outrights(path)
    tournament = frame[frame["year"] == int(year)].copy()
    if tournament.empty:
        raise ValueError(f"No historical market outright probabilities found for {year}")
    by_team = tournament.drop_duplicates("team").set_index("team")
    missing = sorted(set(teams) - set(by_team.index))
    extra = sorted(set(by_team.index) - set(teams))
    if missing or extra:
        raise ValueError(f"Historical market team mismatch for {year}: missing={missing}, extra={extra}")
    probabilities = by_team["p_champion"].reindex(teams).astype(float)
    probabilities = probabilities / float(probabilities.sum())
    source_values = sorted(tournament["source"].astype(str).unique())
    source = "+".join(source_values)
    ratings = market_ratings_from_champion_probabilities(probabilities, anchor.reindex(teams))
    return ratings.reindex(teams), probabilities.reindex(teams), source


def rating_to_match_probs(home_rating: float, away_rating: float, neutral: bool, max_goals: int = 10) -> np.ndarray:
    """Convert Elo-like ratings into W/D/L probabilities via the calibrated score matrix."""
    home_advantage = 0.0 if neutral else ELO_HOME_ADVANTAGE
    gap = (home_rating + home_advantage - away_rating) / ELO_DIVISOR
    base = 1.32
    home_lambda = float(np.clip(base * np.exp(0.36 * gap), 0.15, 4.5))
    away_lambda = float(np.clip(base * np.exp(-0.36 * gap), 0.15, 4.5))
    matrix = correlated_negative_binomial_matrix(home_lambda, away_lambda, NEGATIVE_BINOMIAL_DISPERSION, max_goals)
    return np.asarray(match_outcome_probs(matrix), dtype=float)


def champion_vector_from_ratings(ratings: pd.Series, temperature: float = 260.0) -> pd.Series:
    """Return a tournament winner probability vector from pre-tournament ratings."""
    centred = ratings.astype(float) - float(ratings.astype(float).mean())
    raw = np.exp(np.clip(centred / temperature, -8.0, 8.0))
    probabilities = raw / raw.sum()
    return pd.Series(probabilities, index=ratings.index, dtype=float)


def stage_board_from_champion_vector(champion: pd.Series) -> pd.DataFrame:
    """Build approximate stage probabilities for historical hit-rate diagnostics."""
    rank = champion.rank(ascending=False, method="first")
    rows = []
    for team, p_champion in champion.items():
        p_final = min(1.0, max(float(p_champion) * 3.6, 0.42 / np.sqrt(float(rank.loc[team]))))
        p_semi = min(1.0, max(p_final, float(p_champion) * 6.3, 0.72 / np.sqrt(float(rank.loc[team]))))
        rows.append({"team": team, "p_reach_SF": p_semi, "p_reach_final": p_final, "p_champion": float(p_champion)})
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)


def expert_match_forecasts(ratings: pd.Series, tournament: TournamentBacktest) -> tuple[np.ndarray, np.ndarray]:
    """Return W/D/L forecasts and outcomes for all matches in one tournament."""
    forecasts: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    for row in tournament.matches.itertuples(index=False):
        forecasts.append(rating_to_match_probs(float(ratings[row.home_team]), float(ratings[row.away_team]), bool(row.neutral)))
        outcomes.append(outcome_vector_from_score(int(row.home_goals), int(row.away_goals)))
    return np.asarray(forecasts, dtype=float), np.asarray(outcomes, dtype=float)


def score_expert(
    name: str,
    ratings: pd.Series,
    tournament: TournamentBacktest,
    champion_override: pd.Series | None = None,
    market_status: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    """Score one expert at match and tournament level."""
    forecasts, outcomes = expert_match_forecasts(ratings, tournament)
    match_scores = mean_scores(forecasts, outcomes)
    if champion_override is None:
        champion = champion_vector_from_ratings(ratings.reindex(tournament.teams))
    else:
        champion = champion_override.reindex(tournament.teams).astype(float).clip(lower=1e-12)
        champion = champion / float(champion.sum())
    board = stage_board_from_champion_vector(champion)
    champion_outcome = np.asarray([1.0 if team == tournament.champion else 0.0 for team in champion.index], dtype=float)
    champion_forecast = champion.to_numpy(dtype=float)
    predicted_finalists = set(board.nlargest(2, "p_reach_final")["team"])
    predicted_semifinalists = set(board.nlargest(4, "p_reach_SF")["team"])
    row = {
        "tournament": tournament.year,
        "model": name,
        "mean_match_rps": match_scores["mean_rps"],
        "mean_match_brier": match_scores["mean_brier"],
        "mean_match_log_loss": match_scores["mean_log_loss"],
        "champion_brier": brier_score(champion_forecast, champion_outcome),
        "champion_log_loss": log_loss_single(champion_forecast, champion_outcome),
        "finalist_hit_rate": len(predicted_finalists & set(tournament.finalists)) / 2.0,
        "semifinalist_hit_rate": len(predicted_semifinalists & set(tournament.semifinalists)) / 4.0,
        "champion": tournament.champion,
        "predicted_champion": str(board.iloc[0]["team"]),
        "leakage_passed": tournament.leakage_audit["passed"],
        "market_status": (market_status or "historical_outright_market") if name == "market" else "not_market",
    }
    return row, board, forecasts


def pool_rating_series(series_map: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Pool expert rating series with non-negative weights."""
    names = [name for name, weight in weights.items() if weight > 0.0]
    if not names:
        raise ValueError("at least one positive weight is required")
    total = sum(float(weights[name]) for name in names)
    index = series_map[names[0]].index
    pooled = pd.Series(0.0, index=index, dtype=float)
    for name in names:
        pooled = pooled + series_map[name].reindex(index).astype(float) * (float(weights[name]) / total)
    return pooled
