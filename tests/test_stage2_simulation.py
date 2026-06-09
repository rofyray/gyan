"""Stage 2 tournament simulation tests."""

from __future__ import annotations  # modern type hints

import numpy as np  # random generators and empirical probabilities
import pandas as pd  # fixture and standings tables

from gyan.config import (  # project artifact paths and seed
    BRACKET_PAIRINGS_2026_FILE,
    DIXON_COLES_PARAMS_LATEST_FILE,
    FIFA_OFFICIAL_2026_MATCH_SCHEDULE_PDF,
    GLOBAL_SEED,
    GROUPS_2026_FILE,
    SCHEDULE_2026_FILE,
)
from gyan.engine.dixon_coles import DixonColesModel  # load fitted model for smoke simulation
from gyan.simulation.sample import sample_knockout_scoreline, sample_scorelines  # score samplers
from gyan.simulation.tiebreakers import group_standings, rank_best_thirds  # ranking logic
from gyan.simulation.tournament import (  # tournament simulation helpers
    aggregate_chunks,
    load_structure,
    prepare_simulation_inputs,
    run_tournaments_for_indices,
    validate_probabilities,
)
from gyan.simulation.structure import validate_against_official_fifa_schedule  # official D7 guard


def test_group_standings_resolves_head_to_head_after_base_tie() -> None:
    """A hand-built group uses head-to-head once points/GD/GF are tied."""
    matches = pd.DataFrame(
        [
            {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0},
            {"home": "A", "away": "C", "home_goals": 0, "away_goals": 1},
            {"home": "A", "away": "D", "home_goals": 0, "away_goals": 0},
            {"home": "B", "away": "C", "home_goals": 1, "away_goals": 0},
            {"home": "B", "away": "D", "home_goals": 1, "away_goals": 1},
            {"home": "C", "away": "D", "home_goals": 0, "away_goals": 0},
        ]
    )
    standings = group_standings(matches, ["A", "B", "C", "D"], np.random.default_rng(7))
    assert standings["team"].tolist() == ["B", "C", "A", "D"]


def test_best_thirds_rank_points_goal_diff_goals_then_lots() -> None:
    """Best-third ranking follows points, GD, goals-for, fair play, then lots."""
    thirds = pd.DataFrame(
        [
            {"team": "T1", "group": "A", "points": 6, "goal_diff": 1, "goals_for": 2, "fair_play_points": 0},
            {"team": "T2", "group": "B", "points": 4, "goal_diff": 3, "goals_for": 5, "fair_play_points": 0},
            {"team": "T3", "group": "C", "points": 4, "goal_diff": 3, "goals_for": 4, "fair_play_points": 0},
            {"team": "T4", "group": "D", "points": 4, "goal_diff": 2, "goals_for": 6, "fair_play_points": 0},
            {"team": "T5", "group": "E", "points": 3, "goal_diff": 4, "goals_for": 6, "fair_play_points": 0},
            {"team": "T6", "group": "F", "points": 3, "goal_diff": 1, "goals_for": 3, "fair_play_points": 0},
            {"team": "T7", "group": "G", "points": 3, "goal_diff": 0, "goals_for": 2, "fair_play_points": 0},
            {"team": "T8", "group": "H", "points": 2, "goal_diff": 0, "goals_for": 2, "fair_play_points": 0},
            {"team": "T9", "group": "I", "points": 1, "goal_diff": 4, "goals_for": 6, "fair_play_points": 0},
            {"team": "T10", "group": "J", "points": 1, "goal_diff": 3, "goals_for": 6, "fair_play_points": 0},
            {"team": "T11", "group": "K", "points": 1, "goal_diff": 2, "goals_for": 6, "fair_play_points": 0},
            {"team": "T12", "group": "L", "points": 0, "goal_diff": 9, "goals_for": 9, "fair_play_points": 0},
        ]
    )
    ranked = rank_best_thirds(thirds, np.random.default_rng(7))
    assert ranked.head(4)["team"].tolist() == ["T1", "T2", "T3", "T4"]
    assert ranked["qualified"].sum() == 8
    assert not bool(ranked.loc[ranked["team"] == "T9", "qualified"].iloc[0])


def test_sample_scorelines_converges_to_matrix_probabilities() -> None:
    """A long batched sample approximates the source scoreline probabilities."""
    rng = np.random.default_rng(11)
    matrix = np.asarray([[0.10, 0.20], [0.30, 0.40]], dtype=float)
    draws = sample_scorelines([matrix] * 30_000, rng)
    empirical = np.zeros_like(matrix)
    for home_goals, away_goals in draws:
        empirical[home_goals, away_goals] += 1
    empirical = empirical / empirical.sum()
    assert np.abs(empirical - matrix).sum() < 0.025


def test_knockout_sampler_always_returns_winner_for_drawn_match() -> None:
    """Penalty fallback guarantees a knockout winner even with zero goal means."""
    rng = np.random.default_rng(13)
    result = sample_knockout_scoreline(0.0, 0.0, 0.0, 1500.0, 1500.0, rng, max_goals=2)
    assert result["home_goals"] == result["away_goals"] == 0
    assert result["winner"] in {"home", "away"}
    assert result["decided_by"] == "penalties"


def test_processed_stage2_structure_artifacts_are_complete() -> None:
    """Processed Stage 2 structure artifacts cover the 2026 format."""
    structure = load_structure(GROUPS_2026_FILE, SCHEDULE_2026_FILE, BRACKET_PAIRINGS_2026_FILE)
    groups = structure["groups"]
    schedule = structure["schedule"]
    bracket = structure["bracket"]
    assert len(groups) == 48
    assert groups["group"].nunique() == 12
    assert len(schedule) == 104
    assert (schedule["stage"] == "group").sum() == 72
    assert len(bracket["third_place_combinations"]) == 495
    assert {row["match_id"] for row in bracket["round_of_32"]} == set(range(73, 89))


def test_processed_stage2_structure_matches_official_fifa_pdf() -> None:
    """Official FIFA schedule PDF validates parsed groups and fixture anchors."""
    structure = load_structure(GROUPS_2026_FILE, SCHEDULE_2026_FILE, BRACKET_PAIRINGS_2026_FILE)
    metrics = validate_against_official_fifa_schedule(
        structure["groups"],
        structure["schedule"],
        FIFA_OFFICIAL_2026_MATCH_SCHEDULE_PDF,
    )
    assert metrics["official_fifa_pdf_group_teams"] == 48
    assert metrics["official_fifa_pdf_anchor_fixtures_checked"] == 8


def test_tournament_simulation_is_deterministic_by_global_indices() -> None:
    """Per-index seeding gives exact aggregation stability across chunking."""
    model = DixonColesModel.load(DIXON_COLES_PARAMS_LATEST_FILE)
    structure = load_structure(GROUPS_2026_FILE, SCHEDULE_2026_FILE, BRACKET_PAIRINGS_2026_FILE)
    prepared = prepare_simulation_inputs(model, structure)
    one_chunk = run_tournaments_for_indices(list(range(12)), GLOBAL_SEED, prepared)
    split_chunks = [
        run_tournaments_for_indices(list(range(0, 5)), GLOBAL_SEED, prepared),
        run_tournaments_for_indices(list(range(5, 12)), GLOBAL_SEED, prepared),
    ]
    teams = prepared["groups"]["team"].tolist()
    single_probs, single_metrics = aggregate_chunks([one_chunk], teams, 12)
    split_probs, split_metrics = aggregate_chunks(split_chunks, teams, 12)
    assert single_probs.equals(split_probs)
    assert single_metrics["champion_trace"] == split_metrics["champion_trace"]
    validation = validate_probabilities(single_probs)
    assert validation["probabilities_in_range"]
    assert validation["probabilities_monotone"]
    assert validation["champion_probability_sum"] == 1.0
