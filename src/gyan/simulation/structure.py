"""Tournament-structure parsing for the 2026 World Cup simulator."""

from __future__ import annotations  # use modern type hints consistently

import json  # persist bracket pairings as JSON
import re  # extract match numbers and clean source labels
import shutil  # locate pdftotext for official FIFA PDF validation
import subprocess  # extract text from the official FIFA PDF schedule
from pathlib import Path  # typed filesystem paths

import pandas as pd  # parse and write schedule/group tables
from bs4 import BeautifulSoup  # parse footballbox schedule markup

from gyan.config import (  # project constants and paths
    GROUP_LABELS,
    HOST_NATIONS_2026,
    N_BEST_THIRDS,
    N_GROUPS,
    N_MATCHES_TOTAL,
    N_TEAMS,
)
from gyan.features.socioeconomic import canonical_football_team  # reuse country aliases


D7_TEAM_ALIASES: dict[str, str] = {  # D7 labels that need canonical project names
    "Czechia": "Czech Republic",  # source current name -> D1/project label
    "Türkiye": "Turkey",  # accented official name -> project label
    "DR Congo": "Democratic Republic of the Congo",  # source short label -> canonical label
    "Congo DR": "Democratic Republic of the Congo",  # FIFA variant -> canonical label
    "Curaçao": "Curacao",  # ASCII project label
    "United States": "United States",  # explicit host identity
}

HOST_VENUE_MARKERS: dict[str, tuple[str, ...]] = {  # venue text markers by host country
    "Canada": ("Toronto", "Vancouver"),  # Canadian host cities
    "Mexico": ("Mexico City", "Guadalajara", "Monterrey", "Guadalupe"),  # Mexican venues
    "United States": (  # United States host cities/metro labels
        "Atlanta", "Arlington", "Dallas", "East Rutherford", "Foxborough", "Houston",
        "Inglewood", "Kansas City", "Miami Gardens", "Philadelphia", "Santa Clara", "Seattle",
    ),
}

R32_THIRD_MATCH_BY_WINNER_GROUP: dict[str, int] = {  # Annex C destination columns -> match IDs
    "A": 79,  # 1A vs assigned third-place team
    "B": 85,  # 1B vs assigned third-place team
    "D": 81,  # 1D vs assigned third-place team
    "E": 74,  # 1E vs assigned third-place team
    "G": 82,  # 1G vs assigned third-place team
    "I": 77,  # 1I vs assigned third-place team
    "K": 87,  # 1K vs assigned third-place team
    "L": 80,  # 1L vs assigned third-place team
}

TEAM_TO_FIFA_CODE_2026: dict[str, str] = {  # official FIFA schedule PDF team abbreviations
    "Mexico": "MEX",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Czech Republic": "CZE",
    "Canada": "CAN",
    "Bosnia and Herzegovina": "BIH",
    "Qatar": "QAT",
    "Switzerland": "SUI",
    "Brazil": "BRA",
    "Morocco": "MAR",
    "Haiti": "HAI",
    "Scotland": "SCO",
    "United States": "USA",
    "Paraguay": "PAR",
    "Australia": "AUS",
    "Turkey": "TUR",
    "Germany": "GER",
    "Curacao": "CUW",
    "Ivory Coast": "CIV",
    "Ecuador": "ECU",
    "Netherlands": "NED",
    "Japan": "JPN",
    "Sweden": "SWE",
    "Tunisia": "TUN",
    "Belgium": "BEL",
    "Egypt": "EGY",
    "Iran": "IRN",
    "New Zealand": "NZL",
    "Spain": "ESP",
    "Cape Verde": "CPV",
    "Saudi Arabia": "KSA",
    "Uruguay": "URU",
    "France": "FRA",
    "Senegal": "SEN",
    "Iraq": "IRQ",
    "Norway": "NOR",
    "Argentina": "ARG",
    "Algeria": "ALG",
    "Austria": "AUT",
    "Jordan": "JOR",
    "Portugal": "POR",
    "Democratic Republic of the Congo": "COD",
    "Uzbekistan": "UZB",
    "Colombia": "COL",
    "England": "ENG",
    "Croatia": "CRO",
    "Ghana": "GHA",
    "Panama": "PAN",
}

OFFICIAL_ANCHOR_FIXTURES: dict[int, tuple[str, str]] = {  # anchors called out in FIFA release/PDF
    1: ("Mexico", "South Africa"),
    3: ("Canada", "Bosnia and Herzegovina"),
    4: ("United States", "Paraguay"),
    7: ("Brazil", "Morocco"),
    10: ("Germany", "Curacao"),
    22: ("England", "Croatia"),
    36: ("Tunisia", "Japan"),
    104: ("Winner Match 101", "Winner Match 102"),
}


def canonical_team_name(team_name: object) -> str:
    """Return the project canonical team name for a D7 label.

    Parameters
    ----------
    team_name : object
        Team label scraped from the D7 source.

    Returns
    -------
    str
        Canonical football team label.
    """
    clean = re.sub(r"\[[^\]]+\]", "", str(team_name))  # remove footnote markers
    clean = re.sub(r"\s*\(H\)", "", clean).strip()  # remove host marker from group tables
    alias = D7_TEAM_ALIASES.get(clean, clean)  # apply D7-specific aliases
    return canonical_football_team(alias)  # apply shared FIFA/source aliases


def _stage_for_match(match_id: int) -> str:
    """Return the tournament stage label for a match number."""
    if match_id <= 72:  # first 72 matches are the group stage
        return "group"  # group stage
    if match_id <= 88:  # 16 Round-of-32 matches
        return "R32"  # Round of 32
    if match_id <= 96:  # 8 Round-of-16 matches
        return "R16"  # Round of 16
    if match_id <= 100:  # 4 quarterfinals
        return "QF"  # quarterfinal
    if match_id <= 102:  # 2 semifinals
        return "SF"  # semifinal
    if match_id == 103:  # third-place match
        return "third_place"  # third-place playoff
    return "final"  # match 104 final


def venue_is_host_home(team: str, venue_text: str) -> bool:
    """Return True when a host team is listed at a venue in its host country."""
    if team not in HOST_NATIONS_2026:  # only host nations receive home-field
        return False  # non-hosts are neutral even in North America
    return any(marker in venue_text for marker in HOST_VENUE_MARKERS[team])  # city marker check


def parse_groups(world_cup_html: Path | str) -> pd.DataFrame:
    """Parse 12 groups of four teams from the cached D7 World Cup page.

    Parameters
    ----------
    world_cup_html : Path | str
        Cached `2026_FIFA_World_Cup` HTML page.

    Returns
    -------
    pandas.DataFrame
        Columns: team, group, group_position, fifa_pot.
    """
    tables = pd.read_html(world_cup_html)  # read all source tables
    group_rows: list[dict[str, object]] = []  # collect group/team records
    for table in tables:  # scan every table for standings-shaped group tables
        if "Teamvte" not in table.columns or len(table) != 4:  # group standings tables have 4 teams
            continue  # skip non-group tables
        teams = [canonical_team_name(team) for team in table["Teamvte"].tolist()]  # canonical labels
        if any(team.startswith("Third place") for team in teams):  # skip third-place ranking table
            continue  # not an actual group
        group_label = GROUP_LABELS[len(group_rows) // 4]  # group A-L in page order
        for position, team in enumerate(teams, start=1):  # group draw order
            group_rows.append(  # append one team record
                {
                    "team": team,
                    "group": group_label,
                    "group_position": position,
                    "fifa_pot": position,
                }
            )
        if len(group_rows) == N_TEAMS:  # all 12 groups parsed
            break  # stop scanning
    groups = pd.DataFrame(group_rows)  # convert records to table
    validate_groups(groups)  # fail early if source parse drifted
    return groups  # parsed group table


def parse_schedule(world_cup_html: Path | str, groups: pd.DataFrame) -> pd.DataFrame:
    """Parse group and knockout match placeholders from cached D7 footballboxes.

    Parameters
    ----------
    world_cup_html : Path | str
        Cached main World Cup HTML page.
    groups : pandas.DataFrame
        Output of parse_groups.

    Returns
    -------
    pandas.DataFrame
        Match schedule with match_id, stage, teams/placeholders, date, venue, neutral.
    """
    group_lookup = dict(zip(groups["team"], groups["group"], strict=True))  # team -> group
    soup = BeautifulSoup(Path(world_cup_html).read_text(encoding="utf-8"), "lxml")  # parse HTML
    rows: list[dict[str, object]] = []  # collect one row per footballbox
    for box in soup.select("div.footballbox"):  # every fixture box on the page
        score_text = box.select_one(".fscore").get_text(" ", strip=True)  # e.g. Match 1
        match = re.search(r"Match\s+([0-9]+)", score_text)  # extract match number
        if match is None:  # defensive guard
            continue  # skip malformed fixtures
        match_id = int(match.group(1))  # numeric match id
        home = canonical_team_name(box.select_one(".fhome").get_text(" ", strip=True))  # home/slot
        away = canonical_team_name(box.select_one(".faway").get_text(" ", strip=True))  # away/slot
        date_node = box.select_one(".bday")  # hidden ISO date span
        date = date_node.get_text(strip=True) if date_node else None  # YYYY-MM-DD or None
        time_node = box.select_one(".ftime")  # visible local time
        time_local = time_node.get_text(" ", strip=True) if time_node else ""  # local time string
        venue_node = box.select_one(".fright")  # venue/city div
        venue = venue_node.get_text(" ", strip=True) if venue_node else ""  # venue text
        stage = _stage_for_match(match_id)  # stage from match number
        group = group_lookup.get(home) if stage == "group" else None  # group-stage group
        neutral = not (venue_is_host_home(home, venue))  # home field only for hosts at home venues
        rows.append(  # append fixture record
            {
                "match_id": match_id,
                "stage": stage,
                "group": group,
                "home": home,
                "away": away,
                "date": date,
                "time_local": time_local,
                "venue": venue,
                "neutral": neutral,
            }
        )
    schedule = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)  # stable order
    validate_schedule(schedule, groups)  # run structural checks
    return schedule  # parsed fixture table


def parse_annex_c(knockout_html: Path | str) -> dict[str, object]:
    """Parse Annex C third-place assignment and fixed knockout pairings.

    Parameters
    ----------
    knockout_html : Path | str
        Cached 2026 knockout-stage HTML page.

    Returns
    -------
    dict[str, object]
        JSON-serialisable bracket pairing structure.
    """
    tables = pd.read_html(knockout_html)  # read all knockout tables
    annex = tables[0].copy()  # first table is the 495-row Annex C assignment table
    destination_columns = list(annex.columns[14:22])  # 1A/1B/... destination columns
    combinations: dict[str, dict[str, str]] = {}  # sorted groups key -> winner group -> third group
    for row in annex.itertuples(index=False):  # parse each Annex C row
        values = list(row)  # tuple -> list for positional indexing
        advancing_groups = tuple(str(value) for value in values[1:13] if pd.notna(value))  # group letters
        key = "".join(sorted(advancing_groups))  # canonical combination key
        assignments: dict[str, str] = {}  # destination winner group -> third-place group
        for column, value in zip(destination_columns, values[14:22], strict=True):  # eight assignments
            winner_group = str(column).replace("1", "").replace(" vs", "").strip()  # e.g. A
            assigned_group = str(value).replace("3", "").strip()  # e.g. E
            assignments[winner_group] = assigned_group  # store assignment
        combinations[key] = assignments  # store row by qualifying-group set
    r32_pairings = _fixed_pairings_from_tables(tables[3:19])  # match 73-88 source pairings
    knockout_pairings = _fixed_pairings_from_tables(tables[19:35])  # match 89-104 source pairings
    bracket = {  # final JSON payload
        "third_place_combinations": combinations,
        "third_place_destination_matches": R32_THIRD_MATCH_BY_WINNER_GROUP,
        "round_of_32": r32_pairings,
        "knockout_tree": knockout_pairings,
        "third_place_slots": N_BEST_THIRDS,
        "source_note": "Annex C parsed from cached Wikipedia knockout-stage page, which mirrors FIFA regulations.",
    }
    validate_bracket(bracket)  # structural checks for the bracket payload
    return bracket  # JSON-serialisable bracket structure


def _fixed_pairings_from_tables(tables: list[pd.DataFrame]) -> list[dict[str, object]]:
    """Convert one-row footballbox tables into match pairing records."""
    pairings: list[dict[str, object]] = []  # collect pairings
    for table in tables:  # one table per match
        columns = list(table.columns)  # home, match, away
        match_id = int(re.search(r"Match\s+([0-9]+)", str(columns[1])).group(1))  # match id
        pairings.append(  # append fixed pairing record
            {
                "match_id": match_id,
                "stage": _stage_for_match(match_id),
                "home_slot": str(columns[0]),
                "away_slot": str(columns[2]),
            }
        )
    return sorted(pairings, key=lambda row: row["match_id"])  # stable match order


def validate_groups(groups: pd.DataFrame) -> None:
    """Validate parsed group structure against Stage 2 checks."""
    assert len(groups) == N_TEAMS  # exactly 48 teams
    assert groups["team"].is_unique  # no duplicate team entries
    assert groups["group"].nunique() == N_GROUPS  # exactly 12 groups
    assert (groups.groupby("group")["team"].count() == 4).all()  # four teams per group


def validate_schedule(schedule: pd.DataFrame, groups: pd.DataFrame) -> None:
    """Validate parsed schedule against Stage 2 checks."""
    assert len(schedule) == N_MATCHES_TOTAL  # 104 matches total
    assert schedule["match_id"].is_unique  # every match number exactly once
    assert set(schedule["match_id"]) == set(range(1, N_MATCHES_TOTAL + 1))  # no gaps
    assert len(schedule[schedule["stage"] == "group"]) == 72  # 72 group-stage matches
    group_schedule = schedule[schedule["stage"] == "group"]  # only group fixtures
    assert group_schedule["group"].notna().all()  # every group-stage match has group label
    assert set(group_schedule["home"]).union(set(group_schedule["away"])) == set(groups["team"])  # all teams


def validate_bracket(bracket: dict[str, object]) -> None:
    """Validate the parsed bracket pairing payload."""
    combinations = bracket["third_place_combinations"]  # 495-row Annex C mapping
    assert len(combinations) == 495  # all possible 8-of-12 combinations
    assert len(bracket["round_of_32"]) == 16  # 16 R32 matches
    assert len(bracket["knockout_tree"]) == 16  # R16 through final/third-place matches
    r32_ids = {row["match_id"] for row in bracket["round_of_32"]}  # R32 match ids
    assert r32_ids == set(range(73, 89))  # exact R32 match coverage
    assert {row["match_id"] for row in bracket["knockout_tree"]} == set(range(89, 105))  # exact tree


def _official_pdf_text(pdf_path: Path | str) -> str:
    """Extract text from the official FIFA schedule PDF."""
    pdftotext = shutil.which("pdftotext")  # Poppler CLI is available in the project environment
    if pdftotext is None:  # fail loudly instead of silently skipping the official guard
        raise RuntimeError("pdftotext is required to validate the official FIFA schedule PDF")  # clear error
    result = subprocess.run(  # extract layout text without writing sidecar files
        [pdftotext, "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout  # extracted PDF text


def _parse_official_group_codes(pdf_text: str) -> dict[str, list[str]]:
    """Return group -> ordered FIFA team codes from the official schedule PDF text."""
    group_block = pdf_text[pdf_text.find("GROUPE A"):]  # French PDF uses GROUPE labels
    code_rows = [re.findall(r"\(([A-Z]{3})\)", line) for line in group_block.splitlines()]  # team codes
    code_rows = [row for row in code_rows if len(row) == 6]  # four team rows per six-column band
    if len(code_rows) < 8:  # two bands, four rows each
        raise ValueError("Could not parse the 12 group code rows from official FIFA PDF")  # source drift
    groups: dict[str, list[str]] = {}  # group -> codes
    for band_index, labels in enumerate((GROUP_LABELS[:6], GROUP_LABELS[6:])):  # A-F then G-L
        for column, group_label in enumerate(labels):  # one group per PDF column
            row_offset = band_index * 4  # first four rows or second four rows
            groups[group_label] = [code_rows[row_offset + pos][column] for pos in range(4)]  # ordered codes
    return groups  # official group team codes


def validate_against_official_fifa_schedule(
    groups: pd.DataFrame,
    schedule: pd.DataFrame,
    official_pdf_path: Path | str,
) -> dict[str, int]:
    """Validate parsed Stage 2 structure against FIFA's official schedule PDF.

    The FIFA article HTML is still a JavaScript shell, but DigitalHub exposes
    the official wall-chart PDF. The PDF is not a clean schedule table, so this
    guard checks the reliable invariants it exposes: published date, grouped
    team codes, expected match count, and a small set of anchor fixtures.
    """
    text = _official_pdf_text(official_pdf_path)  # extract official PDF text
    if "10 April 2026" not in text:  # current post-playoff official version marker
        raise ValueError("Official FIFA schedule PDF is not the expected 10 April 2026 version")  # stale source
    official_groups = _parse_official_group_codes(text)  # group -> official FIFA codes
    parsed_groups = {  # group -> parsed team codes
        group: [TEAM_TO_FIFA_CODE_2026[team] for team in table["team"].tolist()]
        for group, table in groups.sort_values(["group", "group_position"]).groupby("group", sort=True)
    }
    if parsed_groups != official_groups:  # fail if Wikipedia-derived groups drift from FIFA PDF
        raise ValueError(f"Parsed groups do not match official FIFA PDF groups: {parsed_groups} != {official_groups}")
    if set(schedule["match_id"]) != set(range(1, N_MATCHES_TOTAL + 1)):  # official schedule is 104 matches
        raise ValueError("Parsed schedule does not cover the official 104-match programme")  # source drift
    by_match = schedule.set_index("match_id")  # lookup parsed fixtures
    for match_id, (home, away) in OFFICIAL_ANCHOR_FIXTURES.items():  # check high-signal anchor fixtures
        parsed_home = str(by_match.loc[match_id, "home"])  # parsed home/slot
        parsed_away = str(by_match.loc[match_id, "away"])  # parsed away/slot
        if (parsed_home, parsed_away) != (home, away):  # anchor mismatch
            raise ValueError(
                f"Match {match_id} does not match official FIFA schedule: "
                f"{parsed_home} v {parsed_away} != {home} v {away}"
            )
    return {  # metrics for run record
        "official_fifa_pdf_group_teams": sum(len(codes) for codes in official_groups.values()),
        "official_fifa_pdf_anchor_fixtures_checked": len(OFFICIAL_ANCHOR_FIXTURES),
    }


def write_structure_artifacts(
    groups: pd.DataFrame,
    schedule: pd.DataFrame,
    bracket: dict[str, object],
    groups_path: Path | str,
    schedule_path: Path | str,
    bracket_path: Path | str,
) -> None:
    """Write Stage 2 structure artifacts to processed data paths."""
    Path(groups_path).parent.mkdir(parents=True, exist_ok=True)  # ensure output directory
    groups.to_parquet(groups_path, index=False)  # canonical group parquet
    schedule.to_parquet(schedule_path, index=False)  # canonical schedule parquet
    Path(bracket_path).write_text(json.dumps(bracket, indent=2), encoding="utf-8")  # bracket JSON
