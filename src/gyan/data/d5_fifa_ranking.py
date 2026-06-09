"""D5 downloader definitions: FIFA/Coca-Cola men's ranking pages."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for FIFA ranking source pages."""
    return [  # capture the live ranking page; parsing happens in a later task
        SourceFile(  # current men's ranking page from the FIFA source registry
            source_id="D5",
            label="fifa_mens_ranking_html",
            url="https://inside.fifa.com/fifa-world-ranking/men",
            raw_path=DATA_RAW / "d5_fifa_mens_ranking.html",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # official rankings JSON used for current FIFA points
            source_id="D5",
            label="fifa_mens_ranking_api_json",
            url="https://api.fifa.com/api/v3/rankings?gender=1&count=250&language=en",
            raw_path=DATA_RAW / "d5_fifa_rankings_api.json",
            required=False,
            min_bytes=10_000,
        ),
    ]
