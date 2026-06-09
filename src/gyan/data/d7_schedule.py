"""D7 downloader definitions: 2026 group draw and match schedule."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for 2026 World Cup schedule sources."""
    return [  # cache both the encyclopedic and official tournament pages
        SourceFile(  # Wikipedia tournament page includes groups and schedule tables
            source_id="D7",
            label="wikipedia_2026_world_cup_html",
            url="https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
            raw_path=DATA_RAW / "d7_wikipedia_2026_world_cup.html",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # FIFA official tournament page listed in the registry
            source_id="D7",
            label="fifa_official_2026_world_cup_html",
            url="https://inside.fifa.com/tournaments/mens/worldcup/canadamexicousa2026",
            raw_path=DATA_RAW / "d7_fifa_official_2026_world_cup.html",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # official FIFA wall-chart schedule, current final post-playoff version
            source_id="D7",
            label="fifa_official_2026_match_schedule_pdf",
            url="https://digitalhub.fifa.com/m/58eca676379df6de/original/FWC26-Match-Schedule_French.pdf",
            raw_path=DATA_RAW / "d7_fifa_official_2026_match_schedule.pdf",
            min_bytes=100_000,
        ),
        SourceFile(  # Wikipedia knockout-stage page carries Annex C in parseable table form
            source_id="D7",
            label="wikipedia_2026_world_cup_knockout_stage_html",
            url="https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage",
            raw_path=DATA_RAW / "d7_wikipedia_2026_world_cup_knockout_stage.html",
            required=False,
            min_bytes=1_000,
        ),
    ]
