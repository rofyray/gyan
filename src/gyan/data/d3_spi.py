"""D3/D3b downloader definitions: FiveThirtyEight SPI and World Cup forecasts."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for FiveThirtyEight SPI validation sources."""
    return [  # capture both match forecasts and ranking snapshots for validation
        SourceFile(  # international SPI match archive from FiveThirtyEight via Internet Archive
            source_id="D3",
            label="spi_matches_intl_csv",
            url=(
                "https://web.archive.org/web/20250306125415id_/"
                "https://projects.fivethirtyeight.com/soccer-api/international/spi_matches_intl.csv"
            ),
            raw_path=DATA_RAW / "d3_spi_matches_intl.csv",
            min_bytes=100_000,
        ),
        SourceFile(  # global rankings archive, useful for SPI context
            source_id="D3",
            label="spi_global_rankings_csv",
            url="https://datahub.io/fivethirtyeight/soccer-spi/_r/-/data/spi_global_rankings.csv",
            raw_path=DATA_RAW / "d3_spi_global_rankings.csv",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # stable mirror metadata for archived World Cup prediction snapshots
            source_id="D3b",
            label="world_cup_predictions_datapackage_json",
            url="https://datahub.io/fivethirtyeight/world-cup-predictions/datapackage.json",
            raw_path=DATA_RAW / "d3b_world_cup_predictions_datapackage.json",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # earliest 2014 snapshot, enough to verify DataHub access
            source_id="D3b",
            label="world_cup_predictions_20140609_csv",
            url="https://datahub.io/fivethirtyeight/world-cup-predictions/_r/-/data/wc-20140609-140000.csv",
            raw_path=DATA_RAW / "d3b_wc_20140609_140000.csv",
            required=False,
            min_bytes=1_000,
        ),
    ]
