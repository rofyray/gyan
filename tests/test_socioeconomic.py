"""Unit tests for socioeconomic feature helpers."""

import json  # write compact parser fixtures

import pandas as pd  # build synthetic feature rows
import pytest  # approximate assertions for formula output

from gyan.config import HOFFMANN_2002_COEFFICIENTS  # published coefficient constants
from gyan.features.socioeconomic import (  # public helpers under test
    hoffmann_prior_score,
    parse_fifa_rankings,
)


def test_hoffmann_prior_score_hand_value() -> None:
    """Check the published Hoffmann linear predictor against a hand calculation."""
    features = pd.DataFrame(  # one synthetic team row with simple numeric terms
        {
            "gdp_per_capita_ppp": [10_000.0],  # GNP/capita proxy
            "gdp_per_capita_ppp_sq": [100_000_000.0],  # squared GDP term
            "temp_dev_sq": [16.0],  # (18 - 14)^2 temperature term
            "host": [1],  # host dummy
            "latin_x_population_share": [0.01],  # interaction term
        }
    )
    coefficients = HOFFMANN_2002_COEFFICIENTS  # shorthand for hand formula
    expected = (  # direct hand calculation from the PRD formula
        coefficients["constant"]
        + coefficients["gnp_per_capita"] * 10_000.0
        + coefficients["gnp_per_capita_sq"] * 100_000_000.0
        + coefficients["temp_dev_sq"] * 16.0
        + coefficients["host_dummy"] * 1
        + coefficients["latin_x_pop_share"] * 0.01
    )
    assert hoffmann_prior_score(features).iloc[0] == pytest.approx(expected)  # formula check


def test_parse_fifa_rankings_applies_team_alias(tmp_path) -> None:
    """Check FIFA ranking parser canonicalises known team aliases."""
    ranking_path = tmp_path / "ranking.json"  # temporary cached FIFA-like artifact
    ranking_path.write_text(  # minimal official-shape JSON fixture
        json.dumps(
            {
                "Results": [
                    {
                        "TeamName": [{"Locale": "en-GB", "Description": "USA"}],
                        "IdCountry": "USA",
                        "ConfederationName": "CONCACAF",
                        "Rank": 16,
                        "DecimalTotalPoints": 1673.13,
                        "PubDate": "2026-04-01T13:00:00+00:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    parsed = parse_fifa_rankings(ranking_path)  # parse the fixture
    assert parsed.loc[0, "team"] == "United States"  # project canonical name
    assert parsed.loc[0, "fifa_team"] == "USA"  # original FIFA label preserved
    assert parsed.loc[0, "fifa_points"] == pytest.approx(1673.13)  # points parsed
