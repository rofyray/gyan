"""Source registry for Stage 1 raw-data downloads."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.data import (  # import each source module named by CONVENTIONS Section 7
    d1_match_results,
    d2_elo_reference,
    d3_spi,
    d4_transfermarkt,
    d5_fifa_ranking,
    d6_squads,
    d7_schedule,
    d8_hoffmann,
    d9_d10_benchmarks,
    d11_world_bank,
    d12_climate,
)
from gyan.data.download_common import SourceFile  # source metadata record


def stage1_source_files() -> list[SourceFile]:
    """Return every raw artifact required or useful for Stage 1 ingestion."""
    files: list[SourceFile] = []  # accumulate source files in PRD order
    files.extend(d1_match_results.source_files())  # D1 match results and helpers
    files.extend(d2_elo_reference.source_files())  # D2/D2b Elo reference material
    files.extend(d3_spi.source_files())  # D3/D3b SPI and WC prediction snapshots
    files.extend(d4_transfermarkt.source_files())  # D4 Transfermarkt values page
    files.extend(d5_fifa_ranking.source_files())  # D5 FIFA ranking page
    files.extend(d6_squads.source_files())  # D6 named 2026 squads
    files.extend(d7_schedule.source_files())  # D7 2026 draw and schedule
    files.extend(d8_hoffmann.source_files())  # D8 socioeconomic paper
    files.extend(d9_d10_benchmarks.source_files())  # D9/D10 benchmark reports
    files.extend(d11_world_bank.source_files())  # D11 macroeconomic indicators
    files.extend(d12_climate.source_files())  # D12 climate source page
    return files  # hand the ordered source list to the script
