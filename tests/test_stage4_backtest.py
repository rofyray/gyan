"""Stage 4 backtest helper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gyan.evaluation.backtest import (
    champion_vector_from_ratings,
    elo_snapshot,
    load_historical_market_outrights,
    market_ratings_from_champion_probabilities,
    prepare_tournament_backtest,
    rating_to_match_probs,
    score_expert,
)
from scripts.s4_backtest import EXPERT_ORDER, _benchmark_market_lookup, _build_modal_bracket, _run_backtests, _top_team_divergence_check


def test_prepare_tournament_backtest_uses_pre_start_training_only() -> None:
    """Tournament preparation records a strict pre-start leakage audit."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-06-01", "2014-06-12", "2014-06-13", "2014-07-08", "2014-07-09", "2014-07-12", "2014-07-13"]),
            "home_team": ["A", "A", "C", "A", "C", "B", "A"],
            "away_team": ["B", "B", "D", "B", "D", "C", "C"],
            "home_goals": [1, 2, 1, 1, 2, 0, 1],
            "away_goals": [0, 0, 0, 0, 0, 1, 0],
            "tournament": ["Friendly", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup"],
            "neutral": [True] * 7,
            "home_elo_post": [1510.0, 1520.0, 1515.0, 1530.0, 1525.0, 1540.0, 1550.0],
            "away_elo_post": [1490.0, 1480.0, 1485.0, 1470.0, 1475.0, 1460.0, 1450.0],
        }
    )
    shootouts = pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    tournament = prepare_tournament_backtest(matches, shootouts, 2014)
    assert tournament.leakage_audit["passed"]
    assert tournament.leakage_audit["max_training_date"] == "2014-06-01"
    assert tournament.champion == "A"


def test_elo_snapshot_ignores_tournament_matches() -> None:
    """Elo snapshots use the last post rating before tournament start."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2014-06-01", "2014-06-12"]),
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
            "home_elo_post": [1510.0, 1600.0],
            "away_elo_post": [1490.0, 1400.0],
        }
    )
    ratings = elo_snapshot(matches, ("A", "B"), pd.Timestamp("2014-06-12"))
    assert ratings["A"] == 1510.0
    assert ratings["B"] == 1490.0


def test_rating_probabilities_and_champion_vector_are_valid() -> None:
    """Backtest probability helpers return normalised distributions."""
    probs = rating_to_match_probs(1600.0, 1500.0, neutral=True)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)
    champion = champion_vector_from_ratings(pd.Series({"A": 1600.0, "B": 1500.0, "C": 1400.0}))
    assert np.isclose(champion.sum(), 1.0)
    assert champion.idxmax() == "A"


def test_historical_market_outrights_are_normalized(tmp_path) -> None:
    """Historical D15 market inputs support consensus percentages and raw odds."""
    path = tmp_path / "markets.csv"
    path.write_text(
        "\n".join(
            [
                "year,team,raw_value,raw_type,source",
                "2022,A,2.0,decimal_odds,test_book",
                "2022,B,4.0,decimal_odds,test_book",
                "2022,C,4.0,decimal_odds,test_book",
                "2018,A,60.0,probability_percent,test_consensus",
                "2018,B,40.0,probability_percent,test_consensus",
            ]
        ),
        encoding="utf-8",
    )
    frame = load_historical_market_outrights(path)
    assert np.isclose(frame[frame["year"] == 2022]["p_champion"].sum(), 1.0)
    assert np.isclose(frame[(frame["year"] == 2022) & (frame["team"] == "A")]["p_champion"].iloc[0], 0.5)


def test_score_expert_can_use_direct_market_champion_vector() -> None:
    """Market champion scoring can use outright probabilities directly."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-11-20"]),
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [0],
            "away_goals": [1],
            "tournament": ["FIFA World Cup"],
            "neutral": [True],
            "home_elo_post": [1500.0],
            "away_elo_post": [1510.0],
        }
    )
    tournament = prepare_tournament_backtest(matches, pd.DataFrame(columns=["date", "home_team", "away_team", "winner"]), 2022)
    champion = pd.Series({"A": 0.2, "B": 0.8})
    ratings = market_ratings_from_champion_probabilities(champion)
    row, _board, _forecasts = score_expert(
        "market",
        ratings,
        tournament,
        champion_override=champion,
        market_status="test_market_source",
    )
    assert row["market_status"] == "test_market_source"
    assert row["predicted_champion"] == "B"


def test_ablation_includes_shipped_gyan_and_compares_against_static_configs(monkeypatch, tmp_path) -> None:
    """Stage 4 ablation reports shipped GYAN beside static ablation configs."""
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2014-06-01", "2014-06-12", "2014-06-13", "2014-07-08", "2014-07-09", "2014-07-12", "2014-07-13",
                    "2018-06-01", "2018-06-14", "2018-06-15", "2018-07-10", "2018-07-11", "2018-07-14", "2018-07-15",
                    "2022-11-01", "2022-11-20", "2022-11-21", "2022-12-13", "2022-12-14", "2022-12-17", "2022-12-18",
                ]
            ),
            "home_team": ["A", "A", "C", "A", "C", "B", "A"] * 3,
            "away_team": ["B", "B", "D", "B", "D", "C", "C"] * 3,
            "home_goals": [1, 2, 1, 1, 2, 0, 1] * 3,
            "away_goals": [0, 0, 0, 0, 0, 1, 0] * 3,
            "tournament": ["Friendly", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "FIFA World Cup"] * 3,
            "neutral": [True] * 21,
            "home_elo_post": [1510.0, 1520.0, 1515.0, 1530.0, 1525.0, 1540.0, 1550.0] * 3,
            "away_elo_post": [1490.0, 1480.0, 1485.0, 1470.0, 1475.0, 1460.0, 1450.0] * 3,
        }
    )
    shootouts = pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    feature_map = pd.DataFrame({"team": ["A", "B", "C", "D"], "world_bank_code": ["AAA", "BBB", "CCC", "DDD"], "mean_annual_temp_c": [14.0] * 4})
    gdp = pd.DataFrame({"world_bank_code": ["AAA", "BBB", "CCC", "DDD"], "year": [2013, 2013, 2013, 2013], "value": [1.0, 1.0, 1.0, 1.0]})
    population = pd.DataFrame({"world_bank_code": ["AAA", "BBB", "CCC", "DDD"], "year": [2013, 2013, 2013, 2013], "value": [1.0, 1.0, 1.0, 1.0]})
    market_path = tmp_path / "markets.csv"
    market_rows = ["year,team,raw_value,raw_type,source"]
    for year in (2014, 2018, 2022):
        for team, raw in {"A": 2.0, "B": 4.0, "C": 5.0, "D": 6.0}.items():
            market_rows.append(f"{year},{team},{raw},decimal_odds,test")
    market_path.write_text("\n".join(market_rows), encoding="utf-8")

    import scripts.s4_backtest as stage4

    monkeypatch.setattr(stage4, "BACKTEST_TOURNAMENTS", (2014, 2018, 2022))
    monkeypatch.setattr(stage4, "BACKTEST_MARKET_OUTRIGHTS_FILE", market_path)
    monkeypatch.setattr(stage4, "prepare_tournament_backtest", lambda _matches, _shootouts, year: prepare_tournament_backtest(_matches, shootouts, year))
    shipped_weights = {expert: 1.0 / len(EXPERT_ORDER) for expert in EXPERT_ORDER}
    _backtest, ablation, _calibration, _audits = _run_backtests(matches, shootouts, feature_map, gdp, population, shipped_weights)
    assert "shipped_gyan" in set(ablation["model"])
    assert set(["goal_yield_market", "full_gyan_equal"]).issubset(set(ablation["model"]))


def test_top_team_divergence_check_resolves_france_when_not_far_from_both() -> None:
    """T-G4-style benchmark check flags only teams far from both Goldman and market."""
    benchmark = pd.DataFrame(
        {
            "team": ["France", "Spain"],
            "gyan_p_champion": [0.11, 0.04],
            "goldman_p_champion": [0.189, 0.257],
            "market_p_champion": [0.16, 0.16],
        }
    )
    check = _top_team_divergence_check(benchmark)
    assert check["france_blocker_resolved"]
    assert "Spain" in check["tripped_teams"]


def test_stage4_benchmark_requires_refreshed_stage3_market(monkeypatch, tmp_path) -> None:
    """Stage 4 should not silently benchmark against static market fallback constants."""
    import scripts.s4_backtest as stage4

    missing_market = tmp_path / "missing_market.parquet"
    monkeypatch.setattr(stage4, "MARKET_IMPLIED_LIVE_FILE", missing_market)
    with pytest.raises(FileNotFoundError, match="rerun scripts/s3_build_ensemble.py"):
        _benchmark_market_lookup()


def test_modal_bracket_is_slot_resolved_and_coherent() -> None:
    """Modal bracket figure source is a coherent match tree, not independent stage lists."""
    board = pd.read_csv("outputs/tables/gyan_forecast_2026_latest.csv")
    modal, metadata = _build_modal_bracket(board)
    final = modal[modal["match_id"] == 104].iloc[0]
    semifinal_winners = set(modal[modal["match_id"].isin([101, 102])]["winner"])
    finalists = {final["home_team"], final["away_team"]}
    assert metadata["validation"]["coherent"]
    assert final["winner"] in finalists
    assert semifinal_winners == finalists
    assert modal[["home_team", "away_team", "winner"]].notna().all().all()
    assert ((modal["winner"] == modal["home_team"]) | (modal["winner"] == modal["away_team"])).all()
