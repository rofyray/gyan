"""Forecast scoring rules used by Stage 3 and Stage 4."""

from __future__ import annotations  # keep annotations modern and consistent

import numpy as np  # vectorised probability maths


def _as_probability_vector(forecast: np.ndarray) -> np.ndarray:
    """Return a finite probability vector normalised to sum to one.

    Parameters
    ----------
    forecast : numpy.ndarray
        Raw or already-normalised probability vector.

    Returns
    -------
    numpy.ndarray
        One-dimensional finite probability vector summing to one.
    """
    vector = np.asarray(forecast, dtype=float).reshape(-1)  # flatten to one category axis
    if np.any(vector < 0.0) or not np.all(np.isfinite(vector)):  # probabilities must be valid
        raise ValueError("forecast must contain finite non-negative probabilities")  # fail loudly
    total = float(vector.sum())  # probability mass
    if total <= 0.0:  # all-zero forecasts are invalid
        raise ValueError("forecast probability mass must be positive")  # fail loudly
    return vector / total  # normalise for numerical safety


def _as_outcome_vector(outcome: np.ndarray, expected_size: int) -> np.ndarray:
    """Return a validated one-hot outcome vector.

    Parameters
    ----------
    outcome : numpy.ndarray
        Realised outcome indicator.
    expected_size : int
        Number of forecast categories.

    Returns
    -------
    numpy.ndarray
        One-dimensional outcome vector with the expected size.
    """
    vector = np.asarray(outcome, dtype=float).reshape(-1)  # flatten outcome axis
    if len(vector) != expected_size:  # category count must match
        raise ValueError("forecast and outcome must have the same length")  # fail loudly
    if np.any(vector < 0.0) or not np.all(np.isfinite(vector)):  # outcome must be valid
        raise ValueError("outcome must contain finite non-negative entries")  # fail loudly
    if not np.isclose(vector.sum(), 1.0):  # one-hot or fractional-realisation mass
        raise ValueError("outcome must sum to one")  # fail loudly
    return vector  # validated outcome vector


def ranked_probability_score(forecast: np.ndarray, outcome: np.ndarray) -> float:
    """Return Ranked Probability Score for one ordered-categorical forecast.

    Parameters
    ----------
    forecast : numpy.ndarray
        Probabilities over ordered categories, such as `[P_home, P_draw, P_away]`.
    outcome : numpy.ndarray
        One-hot vector for the realised ordered category.

    Returns
    -------
    float
        Ranked Probability Score; lower is better.

    Notes
    -----
    Formula follows CONVENTIONS Section 10:
    `1/(r-1) * sum_{i=1}^{r-1}(CDF_forecast_i - CDF_outcome_i)^2`.
    """
    probabilities = _as_probability_vector(forecast)  # normalised forecast probabilities
    realised = _as_outcome_vector(outcome, len(probabilities))  # validated outcome vector
    cumulative_forecast = np.cumsum(probabilities)  # forecast cumulative distribution
    cumulative_outcome = np.cumsum(realised)  # realised cumulative distribution
    gaps = (cumulative_forecast - cumulative_outcome)[:-1]  # final CDF gap is always zero
    return float(np.sum(gaps ** 2) / (len(probabilities) - 1))  # normalised score


def brier_score(forecast: np.ndarray, outcome: np.ndarray) -> float:
    """Return multiclass Brier score for one forecast.

    Parameters
    ----------
    forecast : numpy.ndarray
        Category probabilities.
    outcome : numpy.ndarray
        One-hot realised outcome.

    Returns
    -------
    float
        Sum of squared probability errors; lower is better.
    """
    probabilities = _as_probability_vector(forecast)  # normalised forecast probabilities
    realised = _as_outcome_vector(outcome, len(probabilities))  # validated outcome vector
    return float(np.sum((probabilities - realised) ** 2))  # multiclass Brier score


def log_loss_single(forecast: np.ndarray, outcome: np.ndarray, eps: float = 1e-15) -> float:
    """Return clipped log loss for one forecast.

    Parameters
    ----------
    forecast : numpy.ndarray
        Category probabilities.
    outcome : numpy.ndarray
        One-hot realised outcome.
    eps : float
        Lower clipping bound to avoid `log(0)`.

    Returns
    -------
    float
        Negative log probability assigned to the realised category.
    """
    probabilities = _as_probability_vector(forecast)  # normalised forecast probabilities
    realised = _as_outcome_vector(outcome, len(probabilities))  # validated outcome vector
    clipped = np.clip(probabilities, eps, 1.0)  # numerical guard for zero probabilities
    return float(-np.sum(realised * np.log(clipped)))  # one-observation log loss


def mean_scores(forecasts: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """Return mean RPS, Brier, and log loss over aligned forecasts.

    Parameters
    ----------
    forecasts : numpy.ndarray
        Two-dimensional array of forecast probability vectors.
    outcomes : numpy.ndarray
        Two-dimensional array of one-hot outcome vectors.

    Returns
    -------
    dict[str, float]
        Mean scoring-rule values.
    """
    forecast_array = np.asarray(forecasts, dtype=float)  # ensure numeric forecasts
    outcome_array = np.asarray(outcomes, dtype=float)  # ensure numeric outcomes
    if forecast_array.shape != outcome_array.shape:  # aligned observations/categories required
        raise ValueError("forecasts and outcomes must have the same shape")  # fail loudly
    rps_values = [ranked_probability_score(forecast, outcome) for forecast, outcome in zip(forecast_array, outcome_array)]  # RPS
    brier_values = [brier_score(forecast, outcome) for forecast, outcome in zip(forecast_array, outcome_array)]  # Brier
    log_loss_values = [log_loss_single(forecast, outcome) for forecast, outcome in zip(forecast_array, outcome_array)]  # log loss
    return {  # mean metrics
        "mean_rps": float(np.mean(rps_values)),
        "mean_brier": float(np.mean(brier_values)),
        "mean_log_loss": float(np.mean(log_loss_values)),
    }
