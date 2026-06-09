"""Unit tests for squad-value helper formulas."""

import pytest  # approximate assertions

from gyan.features.squad_value import (  # public helpers under test
    age_value_multiplier,
    parse_market_value_eur,
)


def test_parse_market_value_eur_units() -> None:
    """Check Transfermarkt display values parse to raw euros."""
    assert parse_market_value_eur("€1.37bn") == pytest.approx(1_370_000_000.0)  # bn
    assert parse_market_value_eur("€957.65m") == pytest.approx(957_650_000.0)  # m
    assert parse_market_value_eur("€250k") == pytest.approx(250_000.0)  # k


def test_age_value_multiplier_branches() -> None:
    """Check the PELE-style age weighting branches."""
    assert age_value_multiplier(22.0) == pytest.approx(1.12)  # young premium
    assert age_value_multiplier(26.0) == pytest.approx(1.00)  # prime neutral
    assert age_value_multiplier(31.0) == pytest.approx(0.88)  # early decline
    assert age_value_multiplier(35.0) == pytest.approx(0.72)  # late-career discount
