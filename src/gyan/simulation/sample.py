"""Scoreline sampling helpers for the Monte-Carlo simulator."""

from __future__ import annotations  # modern type hints

import numpy as np  # vectorised random sampling and probability helpers

from gyan.config import (  # simulation constants
    EXTRA_TIME_SCALE,
    NEGATIVE_BINOMIAL_DISPERSION,
    PENALTY_SKILL_ELO_DIVISOR,
    SCORE_DISTRIBUTION_CORRELATED_NB,
    SCORE_DISTRIBUTION_POISSON,
)
from gyan.engine.dixon_coles import correlated_negative_binomial_matrix, scoreline_matrix  # score matrices


def sample_scoreline(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Draw one `(home_goals, away_goals)` pair from a scoreline matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Scoreline probabilities with home goals on rows and away goals on columns.
    rng : numpy.random.Generator
        Random generator used for the draw.

    Returns
    -------
    tuple[int, int]
        Sampled home and away goals.
    """
    flat = matrix.reshape(-1)  # flatten row-major, preserving home/away grid mapping
    chosen = int(rng.choice(len(flat), p=flat / flat.sum()))  # draw one cell index
    return divmod(chosen, matrix.shape[1])  # convert flat index to row/column goals


def sample_scorelines(matrices: list[np.ndarray], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Draw one scoreline for each matrix in a fixture batch.

    Parameters
    ----------
    matrices : list[numpy.ndarray]
        Scoreline matrices, one per fixture.
    rng : numpy.random.Generator
        Random generator used for all draws.

    Returns
    -------
    list[tuple[int, int]]
        Sampled scorelines in the same order as the input matrices.
    """
    if not matrices:  # no fixtures in the batch
        return []  # return an empty result
    stacked = np.stack([matrix / matrix.sum() for matrix in matrices], axis=0)  # normalised batch
    flat = stacked.reshape(stacked.shape[0], -1)  # one probability row per fixture
    cumulative = np.cumsum(flat, axis=1)  # CDF per fixture
    draws = rng.random(stacked.shape[0])[:, None]  # one uniform draw per fixture
    chosen = (cumulative < draws).sum(axis=1)  # sampled flat cell index per fixture
    home_goals = chosen // stacked.shape[2]  # row index
    away_goals = chosen % stacked.shape[2]  # column index
    return list(zip(home_goals.astype(int).tolist(), away_goals.astype(int).tolist(), strict=True))  # pairs


def penalty_home_win_probability(home_elo: float, away_elo: float) -> float:
    """Return a near-50/50 penalty shootout probability tilted by Elo skill.

    Parameters
    ----------
    home_elo, away_elo : float
        Current team Elo ratings.

    Returns
    -------
    float
        Probability the listed home-side team wins the shootout.
    """
    elo_gap = home_elo - away_elo  # positive means home/listed team is stronger
    probability = 1.0 / (1.0 + 10.0 ** (-elo_gap / PENALTY_SKILL_ELO_DIVISOR))  # mild logistic
    return float(np.clip(probability, 0.40, 0.60))  # keep shootouts close to coin flips


def sample_knockout_scoreline(
    lam: float,
    mu: float,
    rho: float,
    home_elo: float,
    away_elo: float,
    rng: np.random.Generator,
    max_goals: int,
    score_distribution: str = SCORE_DISTRIBUTION_POISSON,
    score_dispersion: float | None = None,
) -> dict[str, object]:
    """Sample a knockout match through extra time and penalties if needed.

    Parameters
    ----------
    lam, mu : float
        Regulation-time home and away Poisson means.
    rho : float
        Dixon-Coles dependence parameter.
    home_elo, away_elo : float
        Elo ratings used for the small shootout tilt.
    rng : numpy.random.Generator
        Random generator.
    max_goals : int
        Scoreline matrix truncation.

    Returns
    -------
    dict[str, object]
        Goals and winner side; winner is `"home"` or `"away"`.
    """
    regulation = _knockout_matrix(lam, mu, rho, max_goals, score_distribution, score_dispersion)  # regulation
    home_goals, away_goals = sample_scoreline(regulation, rng)  # regulation draw
    regulation_draw = home_goals == away_goals  # whether the match needed extra time
    decided_by = "regulation"  # default decision route
    if regulation_draw:  # knockout draw after 90 minutes
        extra = _knockout_matrix(  # extra-time matrix with scaled means
            lam * EXTRA_TIME_SCALE,
            mu * EXTRA_TIME_SCALE,
            rho,
            max_goals,
            score_distribution,
            score_dispersion,
        )
        extra_home, extra_away = sample_scoreline(extra, rng)  # extra-time draw
        home_goals += extra_home  # add extra-time home goals
        away_goals += extra_away  # add extra-time away goals
        decided_by = "extra_time"  # at least extra time was required
    if home_goals == away_goals:  # still level after extra time
        p_home_penalty = penalty_home_win_probability(home_elo, away_elo)  # shootout probability
        winner = "home" if rng.random() < p_home_penalty else "away"  # draw shootout result
        decided_by = "penalties"  # final decision route
    else:  # scoreline decided match
        winner = "home" if home_goals > away_goals else "away"  # winner by goals
    return {  # return complete knockout draw result
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "winner": winner,
        "decided_by": decided_by,
        "regulation_draw": bool(regulation_draw),
    }


def _knockout_matrix(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int,
    score_distribution: str,
    score_dispersion: float | None,
) -> np.ndarray:
    """Return the configured knockout score matrix for regulation or extra time."""
    if score_distribution == SCORE_DISTRIBUTION_CORRELATED_NB:  # calibrated shared-frailty matrix
        return correlated_negative_binomial_matrix(lam, mu, score_dispersion or NEGATIVE_BINOMIAL_DISPERSION, max_goals)  # NB
    if score_distribution != SCORE_DISTRIBUTION_POISSON:  # persisted config must be known
        raise ValueError(f"Unknown score_distribution: {score_distribution}")  # fail loudly
    return scoreline_matrix(lam, mu, rho, max_goals=max_goals)  # Poisson/DC matrix
