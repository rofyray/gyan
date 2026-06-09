"""Squad-value features for the 2026 Yield expert input table."""

from __future__ import annotations  # allow modern type hints consistently

import re  # parse ages and market-value strings
import time  # polite retry backoff for Transfermarkt page pulls
import unicodedata  # strip accents for stable player and club joins
from difflib import SequenceMatcher  # conservative same-team fuzzy player matching
from pathlib import Path  # typed filesystem paths

import httpx  # pull current Transfermarkt national-team pages when refreshing D4
import numpy as np  # numeric clipping and median helpers
import pandas as pd  # tabular parsing and feature engineering
from bs4 import BeautifulSoup  # parse the D6 squad headings reliably

from gyan.config import (  # centralised constants used by the feature builder
    SQUAD_STATUS_WEIGHT,
    UEFA_VALUE_DISCOUNT,
    repo_path_str,
)
from gyan.features.socioeconomic import canonical_football_team  # reuse project team aliases


TEAM_ALIASES: dict[str, str] = {  # D6 labels that differ from project/D1 labels
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",  # preserve full country label
    "Côte d'Ivoire": "Ivory Coast",  # common project football name
    "Ivory Coast": "Ivory Coast",  # explicit identity for readability
    "Congo DR": "Democratic Republic of the Congo",  # FIFA ordering -> project label
    "DR Congo": "Democratic Republic of the Congo",  # Wikipedia heading variant
    "Türkiye": "Turkey",  # D1 and most local tables use Turkey
    "Czechia": "Czech Republic",  # D1 historical/common label
    "Curaçao": "Curacao",  # project labels use ASCII spelling
}

D4_COUNTRY_ALIASES: dict[str, str] = {  # Transfermarkt country labels -> project labels
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",  # TM country spelling
    "Cote d'Ivoire": "Ivory Coast",  # TM ASCII spelling
    "Korea, South": "South Korea",  # TM country spelling
    "Korea, North": "North Korea",  # TM country spelling
    "Türkiye": "Turkey",  # TM spelling
    "Curacao": "Curacao",  # ASCII spelling used by TM
}

D4_NATIONAL_TEAM_URL_FALLBACKS: dict[str, tuple[int, str, str]] = {  # teams absent from CSV
    "Cape Verde": (4311, "Cape Verde", "https://www.transfermarkt.com/cape-verde/startseite/verein/4311"),
    "Curacao": (32364, "Curaçao", "https://www.transfermarkt.com/cura-ao/startseite/verein/32364"),
    "Democratic Republic of the Congo": (3854, "DR Congo", "https://www.transfermarkt.com/dr-congo/startseite/verein/3854"),
    "Haiti": (14161, "Haiti", "https://www.transfermarkt.com/haiti/startseite/verein/14161"),
    "Ivory Coast": (3591, "Ivory Coast", "https://www.transfermarkt.com/ivory-coast/startseite/verein/3591"),
}

PLAYER_NAME_ALIASES: dict[str, tuple[str, ...]] = {  # squad-list name -> TM display-name variants
    "Sphephelo Sithole": ("Yaya Sithole",),
    "Gabriel Magalhães": ("Gabriel",),
    "Munir Mohamedi": ("Munir El Kajoui",),
    "Gatito Fernández": ("Roberto Fernández",),
    "Maximilian Arfsten": ("Max Arfsten",),
    "Hannibal Mejbri": ("Hannibal",),
    "Hadj Mahmoud": ("Mohamed Belhadj Mahmoud",),
    "Nabil Emad": ("Nabil Dunga",),
    "Diney": ("Diney Borges",),
    "Maximiliano Araújo": ("Maxi Araújo",),
    "Prince Kwabena Adu": ("Prince Adu",),
    "Nour Bani Attiah": ("Noor Bane Ataya",),
    "Homam Ahmed": ("Homam Al-Amin",),
    "Kevin Pina": ("Kevin Lenini",),
    "Mohammad Abu Zrayq": ("Shararh",),
}

NON_UEFA_CLUB_MARKERS: tuple[str, ...] = (  # substrings for clearly non-European clubs
    "america", "atlanta united", "al qadsiah", "al-hilal", "al hilal", "al-nassr",
    "al nassr", "cruz azul", "chivas", "tijuana", "santos laguna", "toluca",
    "inter miami", "los angeles", "columbus", "vancouver", "toronto", "monterrey",
    "pachuca", "river plate", "boca juniors", "flamengo", "palmeiras", "sao paulo",
    "santos", "corinthians", "cruzeiro", "ulsan", "yokohama", "kawasaki", "kashima",
    "vissel", "al sadd", "al duhail", "al ahly", "zamalek", "mamelodi", "orlando pirates",
)

UEFA_CLUB_MARKERS: tuple[str, ...] = (  # substrings for common European clubs/leagues
    "real madrid", "barcelona", "atletico", "manchester", "liverpool", "arsenal",
    "chelsea", "tottenham", "newcastle", "brighton", "crystal palace", "fulham",
    "west ham", "bayern", "borussia", "leipzig", "leverkusen", "hoffenheim",
    "psv", "ajax", "feyenoord", "inter", "milan", "juventus", "roma", "napoli",
    "atalanta", "lazio", "paris saint-germain", "psg", "marseille", "monaco",
    "lyon", "porto", "benfica", "sporting", "betis", "sevilla", "valencia",
    "real sociedad", "girona", "slavia prague", "celtic", "rangers", "anderlecht",
    "fenerbahce", "fenerbahçe", "galatasaray", "besiktas", "beşiktaş", "dinamo",
)


def normalise_text(value: object) -> str:
    """Return a lowercase ASCII key for player, club, and team joins.

    Parameters
    ----------
    value : object
        Text-like value to normalise.

    Returns
    -------
    str
        Alphanumeric lowercase key with accents and punctuation removed.
    """
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()  # strip accents
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())  # keep stable join characters only


def clean_player_name(value: object) -> str:
    """Remove squad-list annotations such as captain markers from a player name."""
    return re.sub(r"\s*\([^)]*\)", "", str(value)).strip()  # strip parenthetical notes


def player_join_keys(value: object) -> set[str]:
    """Return robust name keys for Transfermarkt/Wikipedia player joins."""
    cleaned = clean_player_name(value)  # remove captain and similar annotations
    keys = {normalise_text(cleaned)}  # direct display-name key
    parts = cleaned.split()  # preserve hyphenated given names as one part
    if len(parts) >= 2:  # handle East Asian display order and TM first/last ordering
        keys.add(normalise_text(" ".join(parts[::-1])))  # full reversed order
        keys.add(normalise_text(" ".join([parts[-1], *parts[:-1]])))  # last-name-first
    for alias in PLAYER_NAME_ALIASES.get(cleaned, ()):  # known TM display-name variant
        keys.update(player_join_keys(alias) if alias != cleaned else {normalise_text(alias)})  # alias keys
    return {key for key in keys if key}  # drop any empty key


def canonical_squad_team(team_name: str) -> str:
    """Return the project team name for a D6 squad heading.

    Parameters
    ----------
    team_name : str
        Team heading from the 2026 squad page.

    Returns
    -------
    str
        Canonical team label used by downstream joins.
    """
    alias_or_original = TEAM_ALIASES.get(team_name, team_name)  # apply D6-specific aliases
    return canonical_football_team(alias_or_original)  # reuse FIFA/D1 alias rules where possible


def canonical_transfermarkt_country(country_name: object) -> str:
    """Return the project team name for a Transfermarkt country/national-team label."""
    label = D4_COUNTRY_ALIASES.get(str(country_name), str(country_name))  # TM-specific alias
    return canonical_squad_team(label)  # reuse D6/FIFA canonicalisation


def parse_market_value_eur(value: object) -> float:
    """Parse a Transfermarkt value string such as EUR1.37bn or EUR957.65m.

    Parameters
    ----------
    value : object
        Display string from the cached D4 Transfermarkt table.

    Returns
    -------
    float
        Euro market value.
    """
    text = str(value).replace("€", "").replace(",", "").strip().lower()  # normalise currency text
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)(bn|m|k)?", text)  # capture number and unit
    if match is None:  # missing or unexpected Transfermarkt display value
        return float("nan")  # callers can drop or impute
    amount = float(match.group(1))  # numeric part of the value
    unit = match.group(2) or ""  # optional unit suffix
    if unit == "bn":  # billions of euros
        return amount * 1_000_000_000.0  # scale to euros
    if unit == "m":  # millions of euros
        return amount * 1_000_000.0  # scale to euros
    if unit == "k":  # thousands of euros
        return amount * 1_000.0  # scale to euros
    return amount  # already a raw euro amount


def parse_transfermarkt_national_teams(path: Path | str) -> pd.DataFrame:
    """Parse Transfermarkt national-team metadata and page URLs."""
    teams = pd.read_csv(path)  # small gzipped CSV from the D4 dataset
    teams["team"] = teams["country_name"].map(canonical_transfermarkt_country)  # project label
    teams["team"] = np.where(  # prefer TM's national-team name when country is not enough
        teams["team"].astype(str).eq("nan"),
        teams["name"].map(canonical_transfermarkt_country),
        teams["team"],
    )
    output = teams[["team", "national_team_id", "name", "country_name", "total_market_value", "url"]].copy()  # cols
    fallback_rows = []  # source rows for teams missing from the published CSV
    for team, (team_id, source_name, url) in D4_NATIONAL_TEAM_URL_FALLBACKS.items():
        if team not in set(output["team"]):  # only add when absent
            fallback_rows.append(  # compatible metadata row
                {
                    "team": team,
                    "national_team_id": team_id,
                    "name": source_name,
                    "country_name": source_name,
                    "total_market_value": np.nan,
                    "url": url,
                }
            )
    if fallback_rows:  # append explicit URL fallbacks
        output = pd.concat([output, pd.DataFrame(fallback_rows)], ignore_index=True)  # combined
    return output  # normalized metadata


def refresh_transfermarkt_team_pages(
    national_teams_path: Path | str,
    output_dir: Path | str,
    teams: list[str],
    force: bool = False,
) -> list[Path]:
    """Download current Transfermarkt national-team squad pages for the supplied teams."""
    national_teams = parse_transfermarkt_national_teams(national_teams_path)  # team -> URL
    output_root = Path(output_dir)  # normalise once
    output_root.mkdir(parents=True, exist_ok=True)  # ensure cache directory exists
    wanted = set(teams)  # only fetch teams in the 2026 field
    rows = national_teams[national_teams["team"].isin(wanted)].drop_duplicates("team")  # 2026 teams
    paths: list[Path] = []  # downloaded/cache paths
    headers = {"User-Agent": "Mozilla/5.0 GYAN-WorldCupModel/0.1"}  # Transfermarkt blocks default UA
    with httpx.Client(headers=headers, timeout=httpx.Timeout(60.0), follow_redirects=True) as client:
        for row in rows.itertuples(index=False):  # one team page at a time
            page_path = output_root / f"{int(row.national_team_id)}.html"  # stable by TM id
            paths.append(page_path)  # return path whether cached or refreshed
            if page_path.exists() and not force:  # keep raw cache stable by default
                continue  # do not redownload existing page
            last_error: Exception | None = None  # track transient failures
            for attempt in range(3):  # retry occasional 5xx responses
                try:
                    url = str(row.url).replace("www.transfermarkt.co.uk", "www.transfermarkt.com")  # stable host
                    response = client.get(url)  # fetch TM national-team page
                    response.raise_for_status()  # fail loudly on blocked/invalid pages
                    page_path.write_bytes(response.content)  # cache raw page
                    last_error = None  # success
                    break  # done with this page
                except Exception as exc:  # transient network/server issue
                    last_error = exc  # remember for final failure
                    time.sleep(1.0 + attempt)  # small polite backoff
            if last_error is not None:  # all attempts failed
                raise last_error  # fail with the source error
    missing = wanted.difference(set(rows["team"]))  # teams with no D4 national-team URL
    if missing:  # missing URLs mean D4 cannot cover those squads
        raise ValueError(f"Missing Transfermarkt national-team URLs for: {sorted(missing)}")  # fail
    return paths  # cached/downloaded page paths


def _parse_transfermarkt_squad_table(path: Path) -> pd.DataFrame:
    """Parse one Transfermarkt national-team page into player market values."""
    tables = pd.read_html(path)  # the squad table is the wide table with Market value
    squad_tables = [table for table in tables if "Market value" in table.columns and "Player" in table.columns]
    if not squad_tables:  # blocked page or unexpected markup
        raise ValueError(f"No Transfermarkt squad table found in {path}")  # fail loudly
    table = squad_tables[0].copy()  # first matching table is the current squad
    rows: list[dict[str, object]] = []  # collect number rows plus following name/position rows
    for index, row in table.iterrows():  # TM repeats each player across detail rows
        if pd.isna(row.get("#")) or pd.isna(row.get("Market value")):  # only numbered rows carry values
            continue  # skip detail rows
        name = table.iloc[index + 1]["Player"] if index + 1 < len(table) else row["Player"]  # detail name row
        position = table.iloc[index + 2]["Player"] if index + 2 < len(table) else ""  # detail position row
        rows.append(  # compact player-value record
            {
                "player": clean_player_name(name),
                "player_key": normalise_text(clean_player_name(name)),
                "tm_position": str(position),
                "market_value_eur": parse_market_value_eur(row["Market value"]),
                "transfermarkt_page": repo_path_str(path),
            }
        )
    return pd.DataFrame(rows)  # page-level player values


def parse_transfermarkt_team_page_values(
    national_teams_path: Path | str,
    team_pages_dir: Path | str,
    teams: list[str],
) -> pd.DataFrame:
    """Parse cached Transfermarkt national-team pages into individual player values."""
    national_teams = parse_transfermarkt_national_teams(national_teams_path)  # metadata
    rows: list[pd.DataFrame] = []  # collect all parsed team pages
    for team_row in national_teams[national_teams["team"].isin(teams)].drop_duplicates("team").itertuples(index=False):
        page_path = Path(team_pages_dir) / f"{int(team_row.national_team_id)}.html"  # page cache
        if not page_path.exists():  # caller should refresh pages before parsing
            raise FileNotFoundError(f"Missing Transfermarkt team page: {page_path}")  # fail
        values = _parse_transfermarkt_squad_table(page_path)  # parse one page
        values["team"] = team_row.team  # project team label
        values["transfermarkt_team"] = team_row.name  # source team label
        rows.append(values)  # collect
    if not rows:  # no D4 values at all
        raise ValueError("No Transfermarkt national-team page values parsed")  # fail
    return pd.concat(rows, ignore_index=True)  # combined player value table


def parse_transfermarkt_player_profile_values(path: Path | str) -> pd.DataFrame:
    """Parse Transfermarkt player-profile CSV values as a secondary individual-value source."""
    players = pd.read_csv(  # read only the columns needed for joins/audit
        path,
        usecols=[
            "name",
            "country_of_citizenship",
            "current_club_name",
            "market_value_in_eur",
            "international_caps",
            "international_goals",
            "position",
            "sub_position",
        ],
    )
    players = players[players["market_value_in_eur"].notna()].copy()  # individual values only
    players["team"] = players["country_of_citizenship"].map(canonical_transfermarkt_country)  # team label
    players["player_key"] = players["name"].map(lambda value: normalise_text(clean_player_name(value)))  # key
    return players.rename(  # source-specific schema
        columns={
            "name": "transfermarkt_player",
            "current_club_name": "transfermarkt_current_club",
            "market_value_in_eur": "profile_market_value_eur",
        }
    )


def _expand_player_keys(frame: pd.DataFrame, name_col: str, index_col: str) -> pd.DataFrame:
    """Return one row per robust player join key for a player/value table."""
    rows: list[dict[str, object]] = []  # collect expanded keys
    for row in frame.itertuples(index=False):  # source row
        source_index = getattr(row, index_col)  # stable source index
        for key in player_join_keys(getattr(row, name_col)):  # robust direct/reversed keys
            rows.append({index_col: source_index, "player_join_key": key})  # expanded key
    return pd.DataFrame(rows)  # key expansion


def _name_similarity(left: object, right: object) -> float:
    """Return a conservative normalised-name similarity score."""
    return SequenceMatcher(None, normalise_text(clean_player_name(left)), normalise_text(clean_player_name(right))).ratio()


def _team_headings_from_wikipedia(path: Path | str) -> list[str]:
    """Return team headings from the D6 Wikipedia squads page.

    Parameters
    ----------
    path : Path | str
        Cached 2026 World Cup squads HTML.

    Returns
    -------
    list[str]
        Ordered team names matching the first squad tables on the page.
    """
    html = Path(path).read_text(encoding="utf-8")  # read cached squad HTML
    soup = BeautifulSoup(html, "lxml")  # parse robustly with lxml
    headings: list[str] = []  # ordered team names from h3 headings
    for heading in soup.find_all("h3"):  # squad pages use h3 for team sections
        label = heading.get_text(" ", strip=True)  # visible heading text
        if label in {"Statistics", "Age", "Notes", "References", "External links"}:  # non-team sections
            continue  # skip non-team headings
        if label.startswith("Player representation"):  # statistics section heading
            continue  # skip non-team headings
        if label.startswith("Average age") or label.startswith("Coach representation"):  # stats heading
            continue  # skip non-team headings
        headings.append(label)  # keep candidate team heading
    return headings[:48]  # the first 48 h3 headings are the 2026 teams


def parse_wikipedia_squads(path: Path | str) -> pd.DataFrame:
    """Parse the named 2026 World Cup squad tables from Wikipedia D6.

    Parameters
    ----------
    path : Path | str
        Cached 2026 World Cup squads HTML.

    Returns
    -------
    pandas.DataFrame
        One row per named player with team, position, age, caps, goals, and club.
    """
    team_headings = _team_headings_from_wikipedia(path)  # ordered team labels from headings
    squad_tables = pd.read_html(path)[: len(team_headings)]  # first 48 tables are squad lists
    rows: list[pd.DataFrame] = []  # collect normalised player tables
    for team_name, squad_table in zip(team_headings, squad_tables, strict=False):  # align heading/table order
        table = squad_table.copy()  # avoid mutating pandas parser output
        table["team"] = canonical_squad_team(team_name)  # canonical project team name
        table["source_team"] = team_name  # preserve exact source heading
        rows.append(table)  # collect this team's player rows
    players = pd.concat(rows, ignore_index=True)  # one table for all named players
    players = players.rename(  # stable column names for downstream code
        columns={
            "No.": "squad_number",
            "Pos.": "position",
            "Player": "player",
            "Date of birth (age)": "date_of_birth_age",
            "Caps": "caps",
            "Goals": "goals",
            "Club": "club",
        }
    )
    players["age"] = players["date_of_birth_age"].astype(str).str.extract(r"aged\s+([0-9]+)")[0].astype(float)  # age
    players["caps"] = pd.to_numeric(players["caps"], errors="coerce").fillna(0.0)  # numeric caps
    players["goals"] = pd.to_numeric(players["goals"], errors="coerce").fillna(0.0)  # numeric goals
    players["position"] = players["position"].astype(str).replace({"DF": "DF", "MF": "MF", "FW": "FW"})  # pos
    players["player_clean"] = players["player"].map(clean_player_name)  # strip annotations
    players["player_key"] = players["player_clean"].map(normalise_text)  # player join key
    players["club_key"] = players["club"].map(normalise_text)  # club join key
    return players.reset_index(drop=True)  # stable row index


def is_uefa_club(club: object, club_key: str, known_uefa_keys: set[str]) -> bool:
    """Infer whether a player's listed club is in UEFA.

    Parameters
    ----------
    club : object
        Club display name from the squad table.
    club_key : str
        Normalised club key.
    known_uefa_keys : set[str]
        Club keys from the cached Transfermarkt top-club table.

    Returns
    -------
    bool
        True when the club is likely a UEFA club.
    """
    club_text = str(club).lower()  # lowercase display text for substring checks
    if club_key in known_uefa_keys:  # exact top-club match from D4
        return True  # cached D4 top clubs are European
    if any(marker in club_text for marker in NON_UEFA_CLUB_MARKERS):  # known non-UEFA club
        return False  # do not discount non-European club players
    if any(marker in club_text for marker in UEFA_CLUB_MARKERS):  # known UEFA club marker
        return True  # likely European club
    return False  # conservative fallback when the club is unknown


def age_value_multiplier(age: float) -> float:
    """Return the PELE-style age multiplier for market value.

    Parameters
    ----------
    age : float
        Player age in years.

    Returns
    -------
    float
        Multiplier that favours younger-at-equal-value players.
    """
    if np.isnan(age):  # missing ages should not dominate the feature
        return 1.0  # neutral multiplier
    if age <= 23.0:  # young players retain upside
        return 1.12  # modest premium
    if age <= 28.0:  # prime age band
        return 1.00  # no adjustment
    if age <= 32.0:  # early decline band
        return 0.88  # moderate discount
    return 0.72  # late-career discount


def enrich_players_with_individual_values(
    players: pd.DataFrame,
    team_page_values: pd.DataFrame,
    profile_values: pd.DataFrame,
) -> pd.DataFrame:
    """Attach true Transfermarkt individual player values to named squad players."""
    squad = players.copy().reset_index(drop=True)  # preserve one row per named player
    squad["squad_index"] = squad.index  # stable key for expanded-name joins
    page_values = team_page_values.copy().reset_index(drop=True)  # page-value source rows
    page_values["page_index"] = page_values.index  # stable source row id
    profile = profile_values.copy().reset_index(drop=True)  # player-profile source rows
    profile["profile_index"] = profile.index  # stable source row id

    squad_keys = _expand_player_keys(squad.rename(columns={"player_clean": "join_name"}), "join_name", "squad_index")
    squad_keys = squad_keys.merge(squad[["squad_index", "team"]], on="squad_index", how="left")  # add team

    page_keys = _expand_player_keys(page_values.rename(columns={"player": "join_name"}), "join_name", "page_index")
    page_keys = page_keys.merge(page_values[["page_index", "team"]], on="page_index", how="left")  # add team
    page_matches = squad_keys.merge(page_keys, on=["team", "player_join_key"], how="left")  # page join
    page_matches = page_matches.dropna(subset=["page_index"]).drop_duplicates("squad_index")  # first page match

    profile_keys = _expand_player_keys(profile.rename(columns={"transfermarkt_player": "join_name"}), "join_name", "profile_index")
    profile_keys = profile_keys.merge(profile[["profile_index", "team"]], on="profile_index", how="left")  # add team
    profile_matches = squad_keys.merge(profile_keys, on=["team", "player_join_key"], how="left")  # profile join
    profile_matches = profile_matches.dropna(subset=["profile_index"]).drop_duplicates("squad_index")  # first profile match

    enriched = squad.merge(  # attach primary national-team page values
        page_matches[["squad_index", "page_index"]],
        on="squad_index",
        how="left",
    ).merge(
        page_values[["page_index", "player", "tm_position", "market_value_eur", "transfermarkt_page", "transfermarkt_team"]],
        on="page_index",
        how="left",
        suffixes=("", "_transfermarkt_page"),
    )
    enriched = enriched.rename(columns={"player_transfermarkt_page": "transfermarkt_page_player"})  # readable audit
    enriched = enriched.merge(  # attach secondary player-profile values
        profile_matches[["squad_index", "profile_index"]],
        on="squad_index",
        how="left",
    ).merge(
        profile[
            [
                "profile_index",
                "transfermarkt_player",
                "profile_market_value_eur",
                "transfermarkt_current_club",
                "country_of_citizenship",
            ]
        ],
        on="profile_index",
        how="left",
    )
    enriched["market_value_eur"] = enriched["market_value_eur"].combine_first(enriched["profile_market_value_eur"])  # value
    enriched["value_source"] = np.select(  # source audit
        [
            enriched["page_index"].notna(),
            enriched["profile_index"].notna(),
        ],
        [
            "transfermarkt_national_team_player_value",
            "transfermarkt_player_profile_value",
        ],
        default="missing_transfermarkt_player_value",
    )
    missing_mask = enriched["value_source"].eq("missing_transfermarkt_player_value")  # residual misses
    if missing_mask.any():  # conservative same-team fuzzy matching for source-name variants
        source_values = pd.concat(  # combined true individual-value candidates
            [
                page_values.rename(columns={"player": "source_player"})[
                    ["team", "source_player", "market_value_eur", "transfermarkt_page", "transfermarkt_team"]
                ].assign(fuzzy_source="transfermarkt_national_team_player_value_fuzzy"),
                profile.rename(columns={"transfermarkt_player": "source_player", "profile_market_value_eur": "market_value_eur"})[
                    ["team", "source_player", "market_value_eur"]
                ].assign(
                    transfermarkt_page=np.nan,
                    transfermarkt_team=np.nan,
                    fuzzy_source="transfermarkt_player_profile_value_fuzzy",
                ),
            ],
            ignore_index=True,
        )
        for index, player_row in enriched[missing_mask].iterrows():  # one unresolved player at a time
            candidates = source_values[source_values["team"] == player_row["team"]].copy()  # same-team only
            if candidates.empty:  # no D4 candidates for this team
                continue  # leave missing
            candidates["similarity"] = candidates["source_player"].map(lambda value: _name_similarity(player_row["player_clean"], value))
            best = candidates.sort_values(["similarity", "market_value_eur"], ascending=False).iloc[0]  # best source name
            if float(best["similarity"]) >= 0.75:  # avoid loose cross-player matches
                enriched.at[index, "market_value_eur"] = float(best["market_value_eur"])  # true D4 value
                enriched.at[index, "value_source"] = str(best["fuzzy_source"])  # audit fuzzy source
                enriched.at[index, "transfermarkt_page"] = best.get("transfermarkt_page", np.nan)  # source page
                enriched.at[index, "transfermarkt_team"] = best.get("transfermarkt_team", np.nan)  # source team
                enriched.at[index, "transfermarkt_page_player"] = best["source_player"]  # matched source name
    enriched["market_value_eur"] = enriched["market_value_eur"].fillna(0.0)  # unknown values contribute zero, not imputed
    enriched["is_uefa_club_player"] = [  # infer UEFA club status from the named squad club
        is_uefa_club(club, club_key, set())
        for club, club_key in zip(enriched["club"], enriched["club_key"], strict=False)
    ]
    enriched["injury_status"] = "available"  # default status before injury overrides
    enriched["injury_multiplier"] = 1.0  # full value before injury overrides
    enriched["injury_note"] = ""  # blank audit note by default
    enriched["effective_market_value_eur"] = enriched["market_value_eur"]  # injury-adjusted value
    enriched["uefa_adjusted_player_value_eur"] = np.where(  # apply PELE-style UEFA discount
        enriched["is_uefa_club_player"],
        enriched["effective_market_value_eur"] * (1.0 - UEFA_VALUE_DISCOUNT),
        enriched["effective_market_value_eur"],
    )
    enriched["age_weighted_player_value_eur"] = [  # age-weighted contribution
        value * age_value_multiplier(age)
        for value, age in zip(enriched["effective_market_value_eur"], enriched["age"], strict=False)
    ]
    return enriched.drop(columns=["squad_index", "page_index", "profile_index"], errors="ignore")  # audit table


def apply_injury_adjustments(players: pd.DataFrame, injuries_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply late-injury and absence adjustments from the editable injury CSV.

    Parameters
    ----------
    players : pandas.DataFrame
        Enriched named-player table.
    injuries_path : Path | str
        Editable `data/raw/injuries_2026.csv` snapshot.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Adjusted player table and one injury audit row per injury snapshot entry.
    """
    adjusted = players.copy()  # do not mutate caller data
    injuries = pd.read_csv(injuries_path)  # load editable injury tracker
    injuries["team"] = injuries["team"].map(canonical_squad_team)  # canonicalise team labels
    injuries["player_key"] = injuries["player"].map(normalise_text)  # player join key
    audit_rows: list[dict[str, object]] = []  # collect adjustment audit rows
    player_index = {(row.team, row.player_key): index for index, row in adjusted.iterrows()}  # lookup
    for injury in injuries.itertuples(index=False):  # process each injury row deterministically
        key = (injury.team, injury.player_key)  # team/player lookup key
        status = str(injury.status).strip().lower()  # normalised status
        in_final_squad = str(injury.in_final_squad).strip().lower()  # yes/no text
        configured_multiplier = SQUAD_STATUS_WEIGHT.get(status, 1.0)  # config default by status
        file_multiplier = float(injury.weight_multiplier) if not pd.isna(injury.weight_multiplier) else configured_multiplier  # CSV override
        if key not in player_index:  # injury row refers to a player not in the named squad
            audit_rows.append(  # record the unresolved or pre-lock absence
                {
                    "team": injury.team,
                    "player": injury.player,
                    "status": status,
                    "in_final_squad": in_final_squad,
                    "matched_named_squad": False,
                    "value_before_eur": 0.0,
                    "value_after_eur": 0.0,
                    "value_delta_eur": 0.0,
                    "adjustment_type": "pre_lock_what_if" if status == "out" and in_final_squad == "no" else "unmatched_injury_row",
                    "note": f"{injury.note}; no named-squad Transfermarkt value applied",
                }
            )
            continue  # no named-squad row to change
        index = player_index[key]  # row index of the named player
        before_value = float(adjusted.at[index, "market_value_eur"])  # pre-adjustment value
        if status == "out" and in_final_squad == "yes":  # named then ruled out
            multiplier = 0.0  # remove from squad-value contribution
            adjustment_type = "removed_named_player"  # audit label
        elif status == "doubtful":  # player remains named but uncertain
            multiplier = file_multiplier  # apply editable CSV/config multiplier
            adjustment_type = "doubtful_multiplier"  # audit label
        else:  # available or pre-lock rows that still matched
            multiplier = 1.0 if status == "available" else file_multiplier  # default full value
            adjustment_type = "available_or_no_change"  # audit label
        after_value = before_value * multiplier  # injury-adjusted player contribution
        adjusted.at[index, "injury_status"] = status  # expose status on player audit table
        adjusted.at[index, "injury_multiplier"] = multiplier  # expose applied multiplier
        adjusted.at[index, "injury_note"] = str(injury.note)  # preserve source note
        adjusted.at[index, "effective_market_value_eur"] = after_value  # update effective value
        audit_rows.append(  # record the adjustment calculation
            {
                "team": injury.team,
                "player": injury.player,
                "status": status,
                "in_final_squad": in_final_squad,
                "matched_named_squad": True,
                "value_before_eur": before_value,
                "value_after_eur": after_value,
                "value_delta_eur": before_value - after_value,
                "adjustment_type": adjustment_type,
                "note": injury.note,
            }
        )
    adjusted["uefa_adjusted_player_value_eur"] = np.where(  # recompute UEFA-adjusted value after injuries
        adjusted["is_uefa_club_player"],
        adjusted["effective_market_value_eur"] * (1.0 - UEFA_VALUE_DISCOUNT),
        adjusted["effective_market_value_eur"],
    )
    adjusted["age_weighted_player_value_eur"] = [  # recompute age-weighted value after injuries
        value * age_value_multiplier(age)
        for value, age in zip(adjusted["effective_market_value_eur"], adjusted["age"], strict=False)
    ]
    return adjusted, pd.DataFrame(audit_rows)  # adjusted players and injury audit


def aggregate_team_features(players: pd.DataFrame, injury_audit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate named-player rows into one squad-value feature row per team.

    Parameters
    ----------
    players : pandas.DataFrame
        Player table after value and injury enrichment.
    injury_audit : pandas.DataFrame
        Injury adjustment audit rows.

    Returns
    -------
    pandas.DataFrame
        One row per 2026 team with Yield-expert squad features.
    """
    grouped = players.groupby("team", sort=True)  # one group per national team
    features = grouped.agg(  # compute team-level squad features
        n_named_players=("player", "count"),
        selected_squad_value_eur=("effective_market_value_eur", "sum"),
        raw_squad_value_eur=("market_value_eur", "sum"),
        uefa_adjusted_value=("uefa_adjusted_player_value_eur", "sum"),
        age_weighted_value=("age_weighted_player_value_eur", "sum"),
        n_uefa_club_players=("is_uefa_club_player", "sum"),
        median_age=("age", "median"),
        mean_age=("age", "mean"),
        n_missing_player_values=("value_source", lambda values: int((values == "missing_transfermarkt_player_value").sum())),
    ).reset_index()
    features["n_champions_league_minutes"] = np.nan  # unavailable in cached D4/D6, explicit missing
    if not injury_audit.empty:  # only compute injury deltas when audit rows exist
        named_delta = injury_audit[injury_audit["matched_named_squad"]].groupby("team")["value_delta_eur"].sum()  # deltas
        what_if = injury_audit[injury_audit["adjustment_type"] == "pre_lock_what_if"].groupby("team")["value_delta_eur"].sum()  # pre-lock
        features["injury_adjustment_eur"] = features["team"].map(named_delta).fillna(0.0)  # named-squad removal/discount
        features["pre_lock_lost_value_what_if_eur"] = features["team"].map(what_if).fillna(0.0)  # case-study value
    else:  # no injury rows supplied
        features["injury_adjustment_eur"] = 0.0  # no named-squad adjustment
        features["pre_lock_lost_value_what_if_eur"] = 0.0  # no pre-lock what-if
    features["uefa_discount_eur"] = features["selected_squad_value_eur"] - features["uefa_adjusted_value"]  # discount
    features["value_data_note"] = "player values use cached Transfermarkt individual national-team/profile market values"  # note
    return features.sort_values("selected_squad_value_eur", ascending=False).reset_index(drop=True)  # value rank


def build_squad_value_features(
    wikipedia_squads_path: Path | str,
    transfermarkt_players_path: Path | str,
    transfermarkt_national_teams_path: Path | str,
    transfermarkt_team_pages_dir: Path | str,
    injuries_path: Path | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build Stage 1.6 squad-value features and audit tables.

    Parameters
    ----------
    wikipedia_squads_path : Path | str
        Cached D6 Wikipedia squads page.
    transfermarkt_players_path : Path | str
        Cached D4 Transfermarkt player-profile CSV with individual market values.
    transfermarkt_national_teams_path : Path | str
        Cached D4 Transfermarkt national-team metadata CSV.
    transfermarkt_team_pages_dir : Path | str
        Directory of cached D4 Transfermarkt national-team squad pages.
    injuries_path : Path | str
        Editable injury/absence CSV.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, dict[str, object]]
        Team features, player audit table, injury audit table, and metrics.
    """
    squads = parse_wikipedia_squads(wikipedia_squads_path)  # parse named squad rows
    team_page_values = parse_transfermarkt_team_page_values(  # true D4 national-team player values
        transfermarkt_national_teams_path,
        transfermarkt_team_pages_dir,
        sorted(squads["team"].unique()),
    )
    profile_values = parse_transfermarkt_player_profile_values(transfermarkt_players_path)  # secondary D4 values
    valued_players = enrich_players_with_individual_values(squads, team_page_values, profile_values)  # attach values
    adjusted_players, injury_audit = apply_injury_adjustments(valued_players, injuries_path)  # injuries
    team_features = aggregate_team_features(adjusted_players, injury_audit)  # aggregate to teams
    direct_value_rate = float((adjusted_players["value_source"] != "missing_transfermarkt_player_value").mean())  # coverage
    metrics = {  # validation and data-quality metrics for run records
        "teams": int(team_features["team"].nunique()),
        "named_players": int(len(adjusted_players)),
        "teams_with_26_players": int((team_features["n_named_players"] == 26).sum()),
        "min_named_players": int(team_features["n_named_players"].min()),
        "max_named_players": int(team_features["n_named_players"].max()),
        "direct_player_market_values_available": True,
        "direct_player_market_value_rate": direct_value_rate,
        "national_team_page_value_rate": float((adjusted_players["value_source"] == "transfermarkt_national_team_player_value").mean()),
        "player_profile_value_rate": float((adjusted_players["value_source"] == "transfermarkt_player_profile_value").mean()),
        "missing_player_value_rate": float((adjusted_players["value_source"] == "missing_transfermarkt_player_value").mean()),
        "most_valuable_team": str(team_features.iloc[0]["team"]),
        "most_valuable_squad_value_eur": float(team_features.iloc[0]["selected_squad_value_eur"]),
        "total_injury_adjustment_eur": float(team_features["injury_adjustment_eur"].sum()),
        "total_pre_lock_lost_value_what_if_eur": float(team_features["pre_lock_lost_value_what_if_eur"].sum()),
    }
    return team_features, adjusted_players, injury_audit, metrics  # feature table plus audits
