"""D1 downloader definitions: martj42 international match results."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for the martj42 international results source."""
    return [  # include related files needed for later cleaning and shootout handling
        SourceFile(  # main match-result table required by Stage 1.2
            source_id="D1",
            label="martj42_results_csv",
            url="https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
            raw_path=DATA_RAW / "d1_martj42_results.csv",
            min_bytes=1_000_000,
        ),
        SourceFile(  # shootout outcomes help treat penalty-decided draws correctly
            source_id="D1",
            label="martj42_shootouts_csv",
            url="https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
            raw_path=DATA_RAW / "d1_martj42_shootouts.csv",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # former names are useful for canonical team-name mapping
            source_id="D1",
            label="martj42_former_names_csv",
            url="https://raw.githubusercontent.com/martj42/international_results/master/former_names.csv",
            raw_path=DATA_RAW / "d1_martj42_former_names.csv",
            required=False,
            min_bytes=100,
        ),
    ]
