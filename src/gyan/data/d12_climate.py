"""D12 downloader definitions: climate temperature source pages."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for capital-city/climate temperature sources."""
    return [  # cache the Climate Knowledge Portal entry point for later source audit
        SourceFile(  # World Bank CCKP open-data collection documentation
            source_id="D12",
            label="world_bank_climate_data_collections_readme_html",
            url="https://worldbank.github.io/climateknowledgeportal/README.html",
            raw_path=DATA_RAW / "d12_world_bank_climate_data_collections_readme.html",
            required=True,
            min_bytes=1_000,
        ),
        SourceFile(  # World-Bank-sourced fallback country mean annual temperatures
            source_id="D12",
            label="country_average_yearly_temperature_wikipedia_html",
            url="https://en.wikipedia.org/wiki/List_of_countries_by_average_yearly_temperature",
            raw_path=DATA_RAW / "d12_country_average_yearly_temperature_wikipedia.html",
            required=False,
            min_bytes=10_000,
        ),
    ]
