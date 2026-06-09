"""Expert interfaces for Stage 3 ensemble forecasts."""

from __future__ import annotations  # modern type hints

from dataclasses import dataclass  # lightweight expert containers

import numpy as np  # vectorised probability and strength maths
import pandas as pd  # expert input tables

from gyan.config import (  # shared modelling constants
    DEFAULT_SCORE_DISTRIBUTION,
    DIXON_COLES_MAX_GOALS,
    NEGATIVE_BINOMIAL_DISPERSION,
    STRENGTH_MODEL_DEFENSE_SHARE,
    STRENGTH_RATING_DIVISOR,
)
from gyan.engine.dixon_coles import DixonColesModel, match_outcome_probs  # goal probabilities


STAGE_PROB_COLUMNS: tuple[str, ...] = (  # common tournament probability columns
    "p_reach_R32",
    "p_reach_R16",
    "p_reach_QF",
    "p_reach_SF",
    "p_reach_final",
    "p_champion",
)


def _normalise_probabilities(values: np.ndarray) -> np.ndarray:
    """Return non-negative probabilities normalised to one.

    Parameters
    ----------
    values : numpy.ndarray
        Raw probability values.

    Returns
    -------
    numpy.ndarray
        Probability vector summing to one.
    """
    vector = np.asarray(values, dtype=float).clip(min=0.0)  # numeric non-negative vector
    total = float(vector.sum())  # total probability mass
    if total <= 0.0:  # invalid all-zero vector
        raise ValueError("probability vector must have positive mass")  # fail loudly
    return vector / total  # normalised vector


def outcome_vector_from_score(home_goals: int, away_goals: int) -> np.ndarray:
    """Return a one-hot W/D/L vector from a realised scoreline.

    Parameters
    ----------
    home_goals, away_goals : int
        Realised full-time goals.

    Returns
    -------
    numpy.ndarray
        `[home_win, draw, away_win]` one-hot vector.
    """
    if home_goals > away_goals:  # home win
        return np.asarray([1.0, 0.0, 0.0])  # home outcome
    if home_goals == away_goals:  # draw
        return np.asarray([0.0, 1.0, 0.0])  # draw outcome
    return np.asarray([0.0, 0.0, 1.0])  # away outcome


def ratings_from_feature(feature: pd.Series, teams: list[str], rating_spread: float = 220.0) -> dict[str, float]:
    """Convert a feature series into Elo-like ratings.

    Parameters
    ----------
    feature : pandas.Series
        Team-indexed numeric feature where higher is stronger.
    teams : list[str]
        Shared team ordering.
    rating_spread : float
        Standard-deviation multiplier in Elo-like rating points.

    Returns
    -------
    dict[str, float]
        Team to Elo-like rating.
    """
    aligned = feature.reindex(teams).astype(float)  # align to tournament teams
    missing = aligned[aligned.isna()].index.tolist()  # feature gaps after alignment
    if missing:  # every expert feature must cover the tournament field
        raise ValueError(f"Missing expert feature values for tournament teams: {missing}")  # fail loudly
    std = float(aligned.std(ddof=0))  # population standard deviation
    z_score = (aligned - float(aligned.mean())) / (std if std > 0.0 else 1.0)  # standardised strength
    return {team: float(1500.0 + rating_spread * z_score.loc[team]) for team in teams}  # Elo-like ratings


def strength_model_from_ratings(
    ratings: dict[str, float],
    home_field: float = 0.0,
    selected_engine: str = "plain_poisson",
) -> DixonColesModel:
    """Build a Poisson strength model from Elo-like ratings.

    Parameters
    ----------
    ratings : dict[str, float]
        Team to Elo-like strength rating.
    home_field : float
        Non-neutral home-field log-goal effect.
    selected_engine : str
        Engine label, normally `"plain_poisson"` for feature experts.

    Returns
    -------
    DixonColesModel
        Stage 2-compatible goal model.
    """
    teams = sorted(ratings)  # stable team order
    mean_rating = float(np.mean([ratings[team] for team in teams]))  # centre ratings
    ability = {team: (ratings[team] - mean_rating) / STRENGTH_RATING_DIVISOR for team in teams}  # log-goal scale
    attack = {team: float(value) for team, value in ability.items()}  # stronger teams attack better
    defense = {team: float(-STRENGTH_MODEL_DEFENSE_SHARE * value) for team, value in ability.items()}  # stronger concede less
    return DixonColesModel(  # Stage 2-compatible model object
        teams=teams,
        attack=attack,
        defense=defense,
        home_field=home_field,
        rho=0.0,
        xi=0.0,
        max_goals=DIXON_COLES_MAX_GOALS,
        selected_engine=selected_engine,
        score_distribution=DEFAULT_SCORE_DISTRIBUTION,
        score_dispersion=NEGATIVE_BINOMIAL_DISPERSION,
    )


def board_from_champion_vector(engine_board: pd.DataFrame, champion_probabilities: pd.Series) -> pd.DataFrame:
    """Construct a full stage board from a champion vector and engine path shape.

    Parameters
    ----------
    engine_board : pandas.DataFrame
        Stage 2 engine-only board with all stage probability columns.
    champion_probabilities : pandas.Series
        Team-indexed champion probabilities summing to one.

    Returns
    -------
    pandas.DataFrame
        Full tournament board aligned to the engine board's teams.
    """
    teams = engine_board["team"].tolist()  # preserve shared team order
    champion = pd.Series(_normalise_probabilities(champion_probabilities.reindex(teams).fillna(0.0).to_numpy()), index=teams)  # aligned
    rows: list[dict[str, object]] = []  # collect full stage rows
    for row in engine_board.set_index("team").loc[teams].itertuples():  # engine path shape per team
        p_champion_engine = max(float(row.p_champion), 1e-9)  # avoid divide by zero
        scale = float(champion.loc[row.Index]) / p_champion_engine  # champion probability replacement scale
        stage_values = {column: min(1.0, max(float(getattr(row, column)) * scale, float(champion.loc[row.Index]))) for column in STAGE_PROB_COLUMNS[:-1]}  # scaled stages
        previous = 1.0  # enforce monotone decreasing from R32 to champion
        clean_values: dict[str, float] = {}  # monotone stage values
        for column in STAGE_PROB_COLUMNS[:-1]:  # each pre-champion stage
            value = min(previous, max(stage_values[column], float(champion.loc[row.Index])))  # monotone clamp
            clean_values[column] = value  # store cleaned value
            previous = value  # next stage cannot exceed this one
        clean_values["p_champion"] = float(champion.loc[row.Index])  # exact champion vector
        rows.append({"team": row.Index, **clean_values})  # output row
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)  # ranked board


@dataclass
class BaseExpert:
    """Shared interface for Stage 3 experts."""

    name: str  # short expert name
    tournament_board: pd.DataFrame  # full stage-probability board

    def predict_match(self, home: str, away: str, neutral: bool) -> np.ndarray:
        """Return `[P_home, P_draw, P_away]` for one match."""
        raise NotImplementedError  # subclasses implement match forecasts

    def predict_tournament(self) -> pd.DataFrame:
        """Return the expert's full tournament probability board."""
        return self.tournament_board.copy()  # defensive copy for callers


@dataclass
class GoalExpert(BaseExpert):
    """Goal expert wrapping the fitted Stage 1 Dixon-Coles/plain-Poisson model."""

    model: DixonColesModel  # fitted goal model

    def predict_match(self, home: str, away: str, neutral: bool) -> np.ndarray:
        """Return W/D/L probabilities from the fitted goal model."""
        matrix = self.model.predict_fixture(home, away, neutral)  # scoreline distribution
        return np.asarray(match_outcome_probs(matrix), dtype=float)  # W/D/L probabilities


@dataclass
class StrengthExpert(BaseExpert):
    """Feature-strength expert converted into a Poisson match model."""

    ratings: dict[str, float]  # team -> Elo-like strength rating
    model: DixonColesModel  # feature-derived Poisson model

    def predict_match(self, home: str, away: str, neutral: bool) -> np.ndarray:
        """Return W/D/L probabilities from the feature-derived strength model."""
        matrix = self.model.predict_fixture(home, away, neutral)  # configured score matrix
        return np.asarray(match_outcome_probs(matrix), dtype=float)  # W/D/L probabilities


def build_yield_feature_series(squad_features: pd.DataFrame, named: bool) -> pd.Series:
    """Return the Yield expert feature series.

    Parameters
    ----------
    squad_features : pandas.DataFrame
        Stage 1.6 squad feature table.
    named : bool
        Whether to use named-squad, age, scorer, and injury adjustments.

    Returns
    -------
    pandas.Series
        Team-indexed Yield feature where higher is stronger.
    """
    frame = squad_features.set_index("team")  # team-indexed features
    if not named:  # nominal team-level value variant
        return np.log1p(frame["raw_squad_value_eur"].astype(float))  # raw squad value signal
    age_ratio = (frame["age_weighted_value"] / frame["selected_squad_value_eur"].replace(0.0, np.nan)).fillna(1.0)  # age effect
    named_value = frame["uefa_adjusted_value"].astype(float) * age_ratio - frame["injury_adjustment_eur"].astype(float)  # named value
    return np.log1p(named_value.clip(lower=1.0))  # log value signal


def build_socioeconomic_feature_series(socioeconomic_features: pd.DataFrame, teams: list[str]) -> pd.Series:
    """Return the socioeconomic expert feature series.

    Parameters
    ----------
    socioeconomic_features : pandas.DataFrame
        Stage 1.5 feature table.
    teams : list[str]
        2026 field teams.

    Returns
    -------
    pandas.Series
        Team-indexed socioeconomic feature where higher is stronger.
    """
    frame = socioeconomic_features.set_index("team")  # team-indexed features
    subset = frame.reindex(teams)  # keep only 2026 teams
    hoffmann_z = (subset["hoffmann_prior_score"] - subset["hoffmann_prior_score"].mean()) / subset["hoffmann_prior_score"].std(ddof=0)  # prior z
    fifa_z = (subset["fifa_points"] - subset["fifa_points"].mean()) / subset["fifa_points"].std(ddof=0)  # FIFA z
    return (0.55 * hoffmann_z.fillna(0.0) + 0.45 * fifa_z.fillna(0.0)).rename("socioeconomic_strength")  # blended signal


def validate_expert_board(board: pd.DataFrame) -> dict[str, object]:
    """Validate one tournament board.

    Parameters
    ----------
    board : pandas.DataFrame
        Expert stage probability board.

    Returns
    -------
    dict[str, object]
        Board validity metrics.
    """
    stage_frame = board[list(STAGE_PROB_COLUMNS)]  # stage probability columns
    return {  # validation metrics
        "rows": int(len(board)),
        "champion_sum": float(board["p_champion"].sum()),
        "in_range": bool(((stage_frame >= 0.0) & (stage_frame <= 1.0)).all().all()),
        "monotone": bool((stage_frame.diff(axis=1).iloc[:, 1:] <= 1e-12).all().all()),
    }
