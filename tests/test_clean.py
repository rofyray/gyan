"""Tests for Stage 1 canonical match cleaning."""

import pandas as pd  # DataFrame construction for focused unit tests

from gyan.data.clean import build_team_name_map, clean_match_table, validate_clean_matches  # cleaners


def test_build_team_name_map_applies_former_name() -> None:
    """Check former names resolve to their current canonical name."""
    raw_matches = pd.DataFrame(  # minimal match table with one former team name
        {
            "home_team": ["Dahomey"],
            "away_team": ["Ghana"],
        }
    )
    former_names = pd.DataFrame(  # martj42-style former-name row
        {
            "current": ["Benin"],
            "former": ["Dahomey"],
            "start_date": ["1959-11-08"],
            "end_date": ["1975-11-30"],
        }
    )
    name_map = build_team_name_map(raw_matches, former_names)  # build source-to-canonical map
    lookup = dict(zip(name_map["source_name"], name_map["canonical_name"]))  # map for assertion
    assert lookup["Dahomey"] == "Benin"  # former names should canonicalise to current names


def test_clean_match_table_drops_unplayed_rows_and_validates_schema() -> None:
    """Check unplayed rows are dropped and completed rows become canonical."""
    raw_matches = pd.DataFrame(  # one completed row and one scheduled future row
        {
            "date": ["2026-06-06", "2026-06-11"],
            "home_team": ["Ghana", "Mexico"],
            "away_team": ["Benin", "South Africa"],
            "home_score": [2.0, None],
            "away_score": [1.0, None],
            "tournament": ["Friendly", "FIFA World Cup"],
            "city": ["Accra", "Mexico City"],
            "country": ["Ghana", "Mexico"],
            "neutral": [False, False],
        }
    )
    former_names = pd.DataFrame(  # empty but schema-compatible former-name table
        columns=["current", "former", "start_date", "end_date"]
    )
    matches, name_map = clean_match_table(raw_matches, former_names)  # clean the mini table
    metrics = validate_clean_matches(matches, name_map)  # validate and collect metrics
    assert metrics["rows"] == 1  # scheduled row with null score should be excluded
    assert matches.loc[0, "home_goals"] == 2  # completed score should become integer goals
    assert matches.loc[0, "importance_weight"] == 20.0  # friendlies use K=20
