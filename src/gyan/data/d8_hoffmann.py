"""D8 downloader definitions: Hoffmann, Ging & Ramasamy socioeconomic paper."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for the socioeconomic specification source."""
    return [  # cache the open PDF copy for coefficient auditability
        SourceFile(  # Redalyc-hosted paper copy listed in CONVENTIONS Section 7
            source_id="D8",
            label="hoffmann_ging_ramasamy_2002_pdf",
            url="https://www.redalyc.org/pdf/103/10305205.pdf",
            raw_path=DATA_RAW / "d8_hoffmann_ging_ramasamy_2002.pdf",
            required=False,
            min_bytes=10_000,
        ),
    ]
