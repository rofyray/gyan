"""D2/D2b downloader definitions: World Football Elo reference material."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for Elo formula/reference sources."""
    return [  # prefer stable pages and mirror landing pages over repeated live scraping
        SourceFile(  # formula page with K-factor and update-method documentation
            source_id="D2",
            label="eloratings_about_html",
            url="https://www.eloratings.net/about",
            raw_path=DATA_RAW / "d2_eloratings_about.html",
            min_bytes=1_000,
        ),
        SourceFile(  # current eloratings JavaScript data used for spot checks
            source_id="D2",
            label="eloratings_current_ratings_js",
            url="https://www.eloratings.net/scripts/ratings.js",
            raw_path=DATA_RAW / "d2_eloratings_current_ratings.js",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # current world rating table loaded by the eloratings app
            source_id="D2",
            label="eloratings_world_tsv",
            url="https://www.eloratings.net/World.tsv",
            raw_path=DATA_RAW / "d2_eloratings_world.tsv",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # eloratings team-code metadata used to decode World.tsv
            source_id="D2",
            label="eloratings_teams_tsv",
            url="https://www.eloratings.net/teams.tsv",
            raw_path=DATA_RAW / "d2_eloratings_teams.tsv",
            required=False,
            min_bytes=100,
        ),
        SourceFile(  # English team labels used to decode eloratings team codes
            source_id="D2",
            label="eloratings_en_teams_tsv",
            url="https://www.eloratings.net/en.teams.tsv",
            raw_path=DATA_RAW / "d2_eloratings_en_teams.tsv",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # GitHub mirror page, cached to avoid hammering eloratings.net
            source_id="D2b",
            label="demetriodor_elo_mirror_html",
            url="https://github.com/demetriodor/Footbal-Elo-Ratings",
            raw_path=DATA_RAW / "d2b_demetriodor_football_elo_ratings.html",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # second mirror page named in the PRD source registry
            source_id="D2b",
            label="jgravier_soccer_elo_html",
            url="https://github.com/JGravier/soccer-elo",
            raw_path=DATA_RAW / "d2b_jgravier_soccer_elo.html",
            required=False,
            min_bytes=1_000,
        ),
    ]
