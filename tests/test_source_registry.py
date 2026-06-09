"""Tests for the Stage 1 raw-source registry."""

from gyan.data.source_registry import stage1_source_files  # Stage 1 source definitions


def test_stage1_source_registry_has_unique_paths() -> None:
    """Ensure every configured source writes to a unique raw artifact path."""
    source_files = stage1_source_files()  # collect all configured source artifacts
    raw_paths = [source_file.raw_path for source_file in source_files]  # extract target paths
    assert len(raw_paths) == len(set(raw_paths))  # duplicate targets would overwrite raw files


def test_stage1_source_registry_covers_required_source_ids() -> None:
    """Ensure the Stage 1 registry covers every required source family."""
    source_ids = {source_file.source_id for source_file in stage1_source_files()}  # ID coverage
    expected_ids = {"D1", "D2", "D2b", "D3", "D3b", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12"}  # noqa: E501
    assert expected_ids.issubset(source_ids)  # all Stage 1 source families must be present


def test_stage1_registry_uses_international_spi_and_official_fifa_schedule() -> None:
    """Ensure D3/D7 registry entries cover the validation and schedule fixes."""
    source_files = stage1_source_files()  # collect configured artifacts
    labels = {source_file.label: source_file for source_file in source_files}  # label lookup
    assert labels["spi_matches_intl_csv"].raw_path.name == "d3_spi_matches_intl.csv"
    assert "spi_matches_intl.csv" in labels["spi_matches_intl_csv"].url
    assert labels["fifa_official_2026_match_schedule_pdf"].required
    assert labels["fifa_official_2026_match_schedule_pdf"].raw_path.name == "d7_fifa_official_2026_match_schedule.pdf"
