"""D11 downloader definitions: World Bank macroeconomic data."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for World Bank GDP/capita PPP and population."""
    return [  # use the World Bank API to cache full country-year JSON responses
        SourceFile(  # GDP per capita, PPP current international dollars
            source_id="D11",
            label="world_bank_gdp_per_capita_ppp_json",
            url="https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.PP.CD?format=json&per_page=20000",
            raw_path=DATA_RAW / "d11_world_bank_gdp_per_capita_ppp.json",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # total population, needed for socioeconomic features
            source_id="D11",
            label="world_bank_population_json",
            url="https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000",
            raw_path=DATA_RAW / "d11_world_bank_population.json",
            required=False,
            min_bytes=1_000,
        ),
    ]
