"""D6 downloader definitions: 2026 FIFA World Cup named squads."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for named 2026 squad source pages."""
    return [  # use multiple pages because late squad news is fragile and time-sensitive
        SourceFile(  # Wikipedia squad page listed in CONVENTIONS Section 7
            source_id="D6",
            label="wikipedia_2026_world_cup_squads_html",
            url="https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads",
            raw_path=DATA_RAW / "d6_wikipedia_2026_world_cup_squads.html",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # ESPN all-48 squad list from the source registry
            source_id="D6",
            label="espn_2026_world_cup_squad_lists_html",
            url="https://www.espn.com/soccer/story/_/id/48757621/2026-world-cup-squad-lists-players-announced-all-48-teams",
            raw_path=DATA_RAW / "d6_espn_2026_world_cup_squad_lists.html",
            required=False,
            min_bytes=1_000,
        ),
    ]
