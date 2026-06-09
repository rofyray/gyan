"""Unit tests for World Football Elo helper formulas."""

import pandas as pd  # small synthetic tables for sequential Elo feature tests
import pytest  # approximate floating-point assertions for hand-computed values

from gyan.features.elo import (  # import the public Elo formula helpers under test
    EloTracker,
    compute_elo_features,
    expected_score,
    goal_difference_index,
)


def test_goal_difference_index_hand_values() -> None:
    """Check the eloratings.net G multiplier at each branch boundary."""
    assert goal_difference_index(0) == 1.0  # draws use G = 1
    assert goal_difference_index(1) == 1.0  # one-goal wins use G = 1
    assert goal_difference_index(2) == 1.5  # two-goal wins use G = 1.5
    assert goal_difference_index(3) == 1.75  # three-goal wins use G = 1.75
    assert goal_difference_index(4) == 1.875  # four-goal wins use 1.75 + (4 - 3) / 8


def test_expected_score_with_home_advantage() -> None:
    """Check a hand-computed 1500-vs-1500 expectancy with 100 Elo home advantage."""
    expectancy = expected_score(1500.0, 1500.0, 100.0)  # 1 / (10 ** (-100 / 400) + 1)
    assert expectancy == pytest.approx(0.6400649998028851)  # hand-computed D2 logistic


def test_tracker_applies_rounded_zero_sum_update() -> None:
    """Check a one-match rounded Elo update remains zero-sum."""
    tracker = EloTracker(round_changes=True)  # official eloratings rounds each match delta
    result = tracker.update_match("Ghana", "Brazil", 1, 0, 20.0, neutral=False)  # friendly win
    assert result["home_elo_pre"] == 1500.0  # unseen teams start at 1500
    assert result["away_elo_pre"] == 1500.0  # unseen teams start at 1500
    assert result["home_win_expectancy"] == pytest.approx(0.6400649998028851)  # with home edge
    assert result["home_elo_post"] == 1507.0  # round(20 * 1 * (1 - 0.6400649998)) = 7
    assert result["away_elo_post"] == 1493.0  # away loses the exact same rounded points


def test_compute_elo_features_prefers_cleaned_importance_weight() -> None:
    """Check cleaned importance weights are used before tournament-name derivation."""
    matches = pd.DataFrame(  # one synthetic match with an intentionally unknown tournament
        {
            "date": [pd.Timestamp("2026-01-01")],  # deterministic chronological key
            "home_team": ["Alpha"],                # home team name
            "away_team": ["Beta"],                 # away team name
            "home_goals": [1],                     # final home score
            "away_goals": [0],                     # final away score
            "neutral": [False],                    # genuine home venue
            "tournament": ["Unknown Invitational"],  # would derive K=30 if used
            "importance_weight": [20.0],           # cleaned upstream K should win
        }
    )
    enriched = compute_elo_features(matches, round_changes=True)  # build Elo columns
    assert int(enriched.loc[0, "k_factor"]) == 20  # preserve the cleaned K-factor
    assert enriched.loc[0, "home_elo_post"] == 1507.0  # confirms K=20, not K=30
