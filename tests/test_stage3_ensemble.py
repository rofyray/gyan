"""Stage 3 scoring, pooling, market, and expert tests."""

from __future__ import annotations  # modern type hints

import numpy as np  # arrays for hand-computed checks
import pandas as pd  # small expert/market tables

from gyan.ensemble.experts import board_from_champion_vector, ratings_from_feature, strength_model_from_ratings
from gyan.ensemble.market import build_live_market_snapshot, devig
from gyan.ensemble.pooling import linear_opinion_pool, log_opinion_pool, mean_rps_for_weights, optimise_weights
from gyan.evaluation.scoring import brier_score, log_loss_single, ranked_probability_score


def test_scoring_rules_match_hand_computed_values() -> None:
    """RPS, Brier, and log loss match simple hand calculations."""
    forecast = np.asarray([0.2, 0.3, 0.5])  # forecast probabilities
    outcome = np.asarray([0.0, 1.0, 0.0])  # draw realised
    assert np.isclose(ranked_probability_score(forecast, outcome), 0.145)  # ((0.2)^2 + (-0.5)^2)/2
    assert np.isclose(brier_score(forecast, outcome), 0.78)  # 0.04 + 0.49 + 0.25
    assert np.isclose(log_loss_single(forecast, outcome), -np.log(0.3))  # realised draw probability
    assert ranked_probability_score(np.asarray([0.0, 1.0, 0.0]), outcome) == 0.0  # perfect forecast


def test_opinion_pools_return_valid_distributions() -> None:
    """Linear and log pools both return normalised non-negative distributions."""
    experts = np.asarray([[0.7, 0.2, 0.1], [0.4, 0.3, 0.3]])  # two expert forecasts
    weights = np.asarray([0.75, 0.25])  # expert weights
    linear = linear_opinion_pool(experts, weights)  # arithmetic pool
    logged = log_opinion_pool(experts, weights)  # geometric pool
    assert np.isclose(linear.sum(), 1.0)  # linear normalises
    assert np.isclose(logged.sum(), 1.0)  # log normalises
    assert np.all(linear >= 0.0)  # no negative probabilities
    assert np.all(logged >= 0.0)  # no negative probabilities


def test_optimise_weights_improves_training_rps() -> None:
    """SLSQP optimisation beats or matches equal weights on its training objective."""
    expert_forecasts = np.asarray(
        [
            [[0.8, 0.1, 0.1], [0.2, 0.4, 0.4]],
            [[0.7, 0.2, 0.1], [0.2, 0.5, 0.3]],
            [[0.1, 0.2, 0.7], [0.6, 0.2, 0.2]],
        ]
    )
    outcomes = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # outcomes
    equal = np.asarray([0.5, 0.5])  # equal weights
    weights, diagnostics = optimise_weights(expert_forecasts, outcomes, pool_fn=linear_opinion_pool)  # optimise
    assert diagnostics["success"]  # optimiser converged
    assert np.isclose(weights.sum(), 1.0)  # simplex weights
    assert mean_rps_for_weights(weights, expert_forecasts, outcomes) <= mean_rps_for_weights(equal, expert_forecasts, outcomes)  # fit improves


def test_optimise_weights_respects_minimum_expert_floor() -> None:
    """Constrained optimisation keeps every shipped expert active."""
    expert_forecasts = np.asarray(  # three observations, three experts, W/D/L
        [
            [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.4, 0.4]],
            [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
            [[0.1, 0.2, 0.7], [0.2, 0.2, 0.6], [0.6, 0.2, 0.2]],
        ]
    )
    outcomes = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # realised outcomes
    weights, diagnostics = optimise_weights(expert_forecasts, outcomes, pool_fn=linear_opinion_pool, min_weight=0.05)  # fit floor
    assert diagnostics["success"]  # optimiser converged
    assert np.isclose(weights.sum(), 1.0)  # simplex weights
    assert np.all(weights >= 0.05 - 1e-12)  # every expert remains active


def test_devig_and_market_snapshot_sum_to_one(tmp_path) -> None:
    """Market de-vigging and live snapshot persistence produce champion mass one."""
    vector = pd.Series({"A": 0.6, "B": 0.3, "C": 0.3})  # overround vector
    assert np.isclose(devig(vector).sum(), 1.0)  # proportional de-vigging
    output = tmp_path / "market.parquet"  # temporary output path
    source_paths = {}  # prepared raw vectors stand in for cached source pulls
    for source in ("polymarket", "kalshi", "bookmaker"):
        path = tmp_path / f"{source}.csv"  # source-specific vector
        pd.DataFrame(
            {
                "team": ["Spain", "France", "England", "A"],
                "raw_probability": [0.2, 0.15, 0.1, 0.05],
            }
        ).to_csv(path, index=False)
        source_paths[source] = path
    market, metadata = build_live_market_snapshot(["Spain", "France", "England", "A"], output, source_paths=source_paths)  # cached snapshot
    assert output.exists()  # snapshot persisted
    assert np.isclose(market["p_champion"].sum(), 1.0)  # champion mass
    assert metadata["source_note"] == "live_market_pull_cached_raw"  # live pull with raw evidence cache


def test_strength_model_and_champion_board_are_valid() -> None:
    """Feature ratings create a usable model and market board stage probabilities."""
    teams = ["A", "B", "C"]  # toy teams
    ratings = ratings_from_feature(pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}), teams)  # feature ratings
    model = strength_model_from_ratings(ratings)  # feature-derived model
    matrix = model.predict_fixture("A", "C", neutral=True)  # toy scoreline matrix
    assert np.isclose(matrix.sum(), 1.0)  # model produces probability matrix
    engine_board = pd.DataFrame(  # toy engine board
        {
            "team": teams,
            "p_reach_R32": [1.0, 0.8, 0.7],
            "p_reach_R16": [0.7, 0.5, 0.4],
            "p_reach_QF": [0.5, 0.3, 0.2],
            "p_reach_SF": [0.3, 0.2, 0.1],
            "p_reach_final": [0.2, 0.1, 0.05],
            "p_champion": [0.1, 0.05, 0.02],
        }
    )
    board = board_from_champion_vector(engine_board, pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}))  # full board
    assert np.isclose(board["p_champion"].sum(), 1.0)  # champion vector sums to one
    assert ((board[["p_reach_R32", "p_reach_R16", "p_reach_QF", "p_reach_SF", "p_reach_final", "p_champion"]] >= 0.0).all().all())  # bounds
