"""Opinion pools and RPS weight optimisation for expert forecasts."""

from __future__ import annotations  # keep type hints modern

import numpy as np  # vectorised pooling maths
from scipy.optimize import minimize  # constrained simplex optimisation

from gyan.evaluation.scoring import ranked_probability_score  # objective scoring rule


def _normalise_weights(weights: np.ndarray) -> np.ndarray:
    """Return non-negative weights normalised to the simplex.

    Parameters
    ----------
    weights : numpy.ndarray
        Raw weight vector.

    Returns
    -------
    numpy.ndarray
        Non-negative vector summing to one.
    """
    vector = np.asarray(weights, dtype=float).reshape(-1)  # flatten weights
    if np.any(vector < 0.0) or not np.all(np.isfinite(vector)):  # valid finite non-negative weights
        raise ValueError("weights must be finite and non-negative")  # fail loudly
    total = float(vector.sum())  # weight mass
    if total <= 0.0:  # all-zero weights are invalid
        raise ValueError("weights must sum to positive mass")  # fail loudly
    return vector / total  # simplex normalisation


def _normalise_distribution(probabilities: np.ndarray) -> np.ndarray:
    """Return a probability vector normalised to sum to one.

    Parameters
    ----------
    probabilities : numpy.ndarray
        Raw category probabilities.

    Returns
    -------
    numpy.ndarray
        Probability vector summing to one.
    """
    vector = np.asarray(probabilities, dtype=float)  # numeric probability vector
    if np.any(vector < 0.0) or not np.all(np.isfinite(vector)):  # valid finite probabilities
        raise ValueError("probabilities must be finite and non-negative")  # fail loudly
    total = float(vector.sum())  # probability mass
    if total <= 0.0:  # all-zero vector invalid
        raise ValueError("probabilities must sum to positive mass")  # fail loudly
    return vector / total  # normalised distribution


def linear_opinion_pool(expert_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a weighted arithmetic mean of expert probability vectors.

    Parameters
    ----------
    expert_probs : numpy.ndarray
        Array with shape `(n_experts, n_categories)`.
    weights : numpy.ndarray
        Non-negative expert weights summing to one.

    Returns
    -------
    numpy.ndarray
        Pooled probability vector.
    """
    expert_array = np.asarray(expert_probs, dtype=float)  # numeric expert matrix
    weight_vector = _normalise_weights(weights)  # enforce simplex weights
    pooled = np.tensordot(weight_vector, expert_array, axes=([0], [0]))  # weighted arithmetic mean
    return _normalise_distribution(pooled)  # renormalise for numerical safety


def log_opinion_pool(expert_probs: np.ndarray, weights: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Return a renormalised weighted geometric mean of expert probabilities.

    Parameters
    ----------
    expert_probs : numpy.ndarray
        Array with shape `(n_experts, n_categories)`.
    weights : numpy.ndarray
        Non-negative expert weights summing to one.
    eps : float
        Clipping floor before logs.

    Returns
    -------
    numpy.ndarray
        Pooled probability vector.
    """
    expert_array = np.asarray(expert_probs, dtype=float)  # numeric expert matrix
    weight_vector = _normalise_weights(weights)  # enforce simplex weights
    clipped = np.clip(expert_array, eps, 1.0)  # avoid log zero
    log_pooled = np.tensordot(weight_vector, np.log(clipped), axes=([0], [0]))  # weighted log probs
    pooled = np.exp(log_pooled)  # back to probability scale
    return _normalise_distribution(pooled)  # renormalise to one


def pool_many(expert_forecasts: np.ndarray, weights: np.ndarray, pool_fn=linear_opinion_pool) -> np.ndarray:
    """Pool many observations of expert forecasts.

    Parameters
    ----------
    expert_forecasts : numpy.ndarray
        Array with shape `(n_observations, n_experts, n_categories)`.
    weights : numpy.ndarray
        Non-negative expert weights.
    pool_fn : callable
        Opinion-pool function taking `(expert_probs, weights)`.

    Returns
    -------
    numpy.ndarray
        Array with shape `(n_observations, n_categories)`.
    """
    forecast_array = np.asarray(expert_forecasts, dtype=float)  # numeric forecast tensor
    return np.asarray([pool_fn(observation, weights) for observation in forecast_array], dtype=float)  # pooled rows


def mean_rps_for_weights(
    weights: np.ndarray,
    expert_forecasts: np.ndarray,
    outcomes: np.ndarray,
    pool_fn=linear_opinion_pool,
) -> float:
    """Return mean RPS for a candidate expert-weight vector.

    Parameters
    ----------
    weights : numpy.ndarray
        Candidate expert weights.
    expert_forecasts : numpy.ndarray
        Forecast tensor `(n_observations, n_experts, n_categories)`.
    outcomes : numpy.ndarray
        One-hot outcome matrix `(n_observations, n_categories)`.
    pool_fn : callable
        Opinion-pool function.

    Returns
    -------
    float
        Mean Ranked Probability Score.
    """
    pooled = pool_many(expert_forecasts, weights, pool_fn=pool_fn)  # pooled forecast per observation
    scores = [ranked_probability_score(forecast, outcome) for forecast, outcome in zip(pooled, outcomes)]  # RPS
    return float(np.mean(scores))  # mean score


def optimise_weights(
    expert_forecasts: np.ndarray,
    outcomes: np.ndarray,
    pool_fn=linear_opinion_pool,
    min_weight: float = 0.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Optimise simplex expert weights to minimise mean RPS.

    Parameters
    ----------
    expert_forecasts : numpy.ndarray
        Forecast tensor `(n_observations, n_experts, n_categories)`.
    outcomes : numpy.ndarray
        One-hot outcome matrix `(n_observations, n_categories)`.
    pool_fn : callable
        Opinion-pool function.
    min_weight : float
        Lower bound for every expert weight; use 0.0 for the unconstrained PRD default.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, object]]
        Optimised weights and optimisation diagnostics.
    """
    forecast_array = np.asarray(expert_forecasts, dtype=float)  # numeric forecast tensor
    outcome_array = np.asarray(outcomes, dtype=float)  # numeric outcome matrix
    n_experts = forecast_array.shape[1]  # expert dimension
    floor = float(min_weight)  # numeric lower bound for every expert
    if floor < 0.0 or floor * n_experts >= 1.0:  # simplex feasibility guard
        raise ValueError("min_weight must be non-negative and leave positive free weight mass")  # fail loudly
    initial = np.full(n_experts, 1.0 / n_experts, dtype=float)  # equal-weight starting point
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]  # simplex sum constraint
    bounds = [(floor, 1.0)] * n_experts  # non-negative or floor-constrained weights
    result = minimize(  # solve constrained SLSQP problem
        lambda weights: mean_rps_for_weights(weights, forecast_array, outcome_array, pool_fn=pool_fn),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    weights = _normalise_weights(result.x if result.success else initial)  # use fitted weights if successful
    diagnostics = {  # serialisable optimisation diagnostics
        "success": bool(result.success),
        "message": str(result.message),
        "nit": int(result.nit),
        "fun": float(mean_rps_for_weights(weights, forecast_array, outcome_array, pool_fn=pool_fn)),
        "min_weight": floor,
    }
    return weights, diagnostics  # fitted weights and diagnostics
