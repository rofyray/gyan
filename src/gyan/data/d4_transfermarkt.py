"""D4 downloader definitions: Transfermarkt squad and market-value pages."""

from __future__ import annotations  # keep annotations consistent across modules

from gyan.config import DATA_RAW  # raw-data destination root
from gyan.data.download_common import SourceFile  # source metadata record


def source_files() -> list[SourceFile]:
    """Return raw artifacts for Transfermarkt market-value source pages."""
    return [  # cache Transfermarkt-derived player and audit artifacts
        SourceFile(  # weekly Transfermarkt player-profile dataset with individual values
            source_id="D4",
            label="transfermarkt_players_csv_gz",
            url="https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz",
            raw_path=DATA_RAW / "d4_transfermarkt_players.csv.gz",
            required=False,
            min_bytes=1_000_000,
        ),
        SourceFile(  # national-team ids and URLs for current squad pages
            source_id="D4",
            label="transfermarkt_national_teams_csv_gz",
            url="https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/national_teams.csv.gz",
            raw_path=DATA_RAW / "d4_transfermarkt_national_teams.csv.gz",
            required=False,
            min_bytes=1_000,
        ),
        SourceFile(  # Transfermarkt source listed in CONVENTIONS Section 7
            source_id="D4",
            label="transfermarkt_national_team_values_html",
            url="https://www.transfermarkt.com/marktwerte/wertvollstemannschaften/marktwertetop/plus/0/galerie/0?land_id=0&kontinent_id=0&ausrichtung=&spielerposition_id=&altersklasse=&jahrgang=0&spieler_id=&wettbewerb_id=&pos=&detailpos=",
            raw_path=DATA_RAW / "d4_transfermarkt_national_team_values.html",
            required=False,
            min_bytes=1_000,
        ),
    ]
