"""D9/D10 downloader definitions: 2026 benchmark forecast reports."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for Goldman and Klement benchmark reports."""
    return [  # benchmark PDFs are raw inputs for later comparison tables
        SourceFile(  # Goldman Sachs 2026 World Cup report from the source registry
            source_id="D9",
            label="goldman_2026_world_cup_report_pdf",
            url="https://static.poder360.com.br/2026/05/The-World-Cup-and-Economics_-World-Cup-2026_-Predictions-Probabilities-and-Paths-to-Victory.pdf",
            raw_path=DATA_RAW / "d9_goldman_2026_world_cup_report.pdf",
            required=False,
            min_bytes=10_000,
        ),
        SourceFile(  # Panmure Liberum/Klement 2026 note from the source registry
            source_id="D10",
            label="klement_panmure_liberum_2026_pdf",
            url="https://panmureliberum.com/media/3179/strs_1031724.pdf",
            raw_path=DATA_RAW / "d10_klement_panmure_liberum_2026.pdf",
            required=False,
            min_bytes=10_000,
        ),
    ]
