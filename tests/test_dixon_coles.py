"""Unit tests for Dixon-Coles helper formulas."""

import numpy as np  # build independent-Poisson reference matrices
import pytest  # approximate floating-point assertions
from scipy.stats import poisson  # hand reference for rho=0 matrix

from gyan.engine.dixon_coles import (  # public helpers under test
    correlated_negative_binomial_matrix,
    dixon_coles_tau,
    match_outcome_probs,
    scoreline_matrix,
    time_decay_weight,
)


def test_dixon_coles_tau_hand_values() -> None:
    """Check the four low-score tau branches against hand calculations."""
    lam = 1.4  # simple home mean
    mu = 0.9  # simple away mean
    rho = -0.1  # dependence parameter
    assert dixon_coles_tau(0, 0, lam, mu, rho) == pytest.approx(1.126)  # 1 - lam*mu*rho
    assert dixon_coles_tau(0, 1, lam, mu, rho) == pytest.approx(0.86)  # 1 + lam*rho
    assert dixon_coles_tau(1, 0, lam, mu, rho) == pytest.approx(0.91)  # 1 + mu*rho
    assert dixon_coles_tau(1, 1, lam, mu, rho) == pytest.approx(1.1)  # 1 - rho
    assert dixon_coles_tau(2, 1, lam, mu, rho) == pytest.approx(1.0)  # unchanged cell


def test_scoreline_matrix_and_outcomes_sum_to_one() -> None:
    """Check scoreline and W/D/L probabilities are normalised."""
    matrix = scoreline_matrix(1.5, 1.1, -0.05, max_goals=10)  # build corrected matrix
    assert matrix.sum() == pytest.approx(1.0, abs=1e-9)  # full matrix normalisation
    assert sum(match_outcome_probs(matrix)) == pytest.approx(1.0, abs=1e-9)  # W/D/L


def test_scoreline_matrix_rho_zero_matches_independent_poisson() -> None:
    """Check rho=0 reduces to the independent-Poisson outer product after truncation."""
    lam = 1.2  # home mean
    mu = 0.8  # away mean
    max_goals = 8  # small test support
    matrix = scoreline_matrix(lam, mu, 0.0, max_goals=max_goals)  # model matrix
    goal_range = np.arange(max_goals + 1)  # same support
    expected = np.outer(poisson.pmf(goal_range, lam), poisson.pmf(goal_range, mu))  # base
    expected = expected / expected.sum()  # same truncation normalisation
    assert np.allclose(matrix, expected)  # elementwise equality


def test_correlated_negative_binomial_matrix_preserves_means_and_adds_draw_mass() -> None:
    """Check the calibrated NB matrix keeps means and raises draw mass."""
    lam = 1.35
    mu = 1.05
    max_goals = 14
    poisson_matrix = scoreline_matrix(lam, mu, 0.0, max_goals=max_goals)
    nb_matrix = correlated_negative_binomial_matrix(lam, mu, dispersion=8.0, max_goals=max_goals)
    goals = np.arange(max_goals + 1)
    assert nb_matrix.sum() == pytest.approx(1.0, abs=1e-9)
    assert float((nb_matrix * goals[:, None]).sum()) == pytest.approx(lam, abs=0.01)
    assert float((nb_matrix * goals[None, :]).sum()) == pytest.approx(mu, abs=0.01)
    assert match_outcome_probs(nb_matrix)[1] > match_outcome_probs(poisson_matrix)[1]


def test_time_decay_weight_hand_values() -> None:
    """Check the exponential time-decay formula."""
    weights = time_decay_weight(np.asarray([0.0, 10.0]), xi=0.01)  # two simple ages
    assert weights[0] == pytest.approx(1.0)  # current match weight
    assert weights[1] == pytest.approx(np.exp(-0.1))  # exp(-xi*t)
