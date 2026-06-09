"""Clean raw international match results into the canonical match table."""

from __future__ import annotations  # use modern type hints consistently

from pathlib import Path  # typed filesystem paths for inputs and outputs

import pandas as pd  # canonical table cleaning uses pandas DataFrames

from gyan.config import DATA_INTERIM, DATA_RAW, MATCHES_CLEAN_FILE, TEAM_NAME_MAP_FILE  # shared paths
from gyan.features.elo import assign_k_factors  # shared eloratings K-factor mapping


RAW_RESULTS_FILE: Path = DATA_RAW / "d1_martj42_results.csv"  # D1 raw results artifact
RAW_FORMER_NAMES_FILE: Path = DATA_RAW / "d1_martj42_former_names.csv"  # D1 name map helper
def load_raw_results(path: Path = RAW_RESULTS_FILE) -> pd.DataFrame:
    """Load the martj42 raw results CSV.

    Parameters
    ----------
    path : pathlib.Path
        Raw D1 results CSV path.

    Returns
    -------
    pandas.DataFrame
        Raw match-result rows.
    """
    return pd.read_csv(path)  # read the raw CSV exactly once at the boundary


def load_former_names(path: Path = RAW_FORMER_NAMES_FILE) -> pd.DataFrame:
    """Load the martj42 former-name mapping CSV.

    Parameters
    ----------
    path : pathlib.Path
        Raw D1 former-name CSV path.

    Returns
    -------
    pandas.DataFrame
        Former-name rows with current and former names.
    """
    return pd.read_csv(path)  # read former-name rows for canonical mapping


def build_team_name_map(raw_matches: pd.DataFrame, former_names: pd.DataFrame) -> pd.DataFrame:
    """Build a canonical team-name map covering every D1 team string.

    Parameters
    ----------
    raw_matches : pandas.DataFrame
        Raw match table containing home_team and away_team.
    former_names : pandas.DataFrame
        Former-name table containing current and former columns.

    Returns
    -------
    pandas.DataFrame
        Name map with source_name and canonical_name columns.
    """
    match_team_names = pd.concat(  # combine home and away names into one Series
        [raw_matches["home_team"], raw_matches["away_team"]],
        ignore_index=True,
    ).dropna()  # ignore any malformed null team names before mapping
    identity_rows = pd.DataFrame(  # every observed current source name maps to itself
        {
            "source_name": sorted(match_team_names.unique()),
            "canonical_name": sorted(match_team_names.unique()),
            "priority": 0,
        }
    )
    former_rows = former_names.rename(  # translate martj42 column names to map columns
        columns={"former": "source_name", "current": "canonical_name"}
    )[["source_name", "canonical_name"]].dropna()  # keep only complete mapping rows
    former_rows["priority"] = 1  # explicit former-name mappings override identity rows
    combined = pd.concat([identity_rows, former_rows], ignore_index=True)  # merge map rows
    combined = combined.sort_values(["source_name", "priority"]).drop_duplicates(  # stable map
        subset=["source_name"],
        keep="last",
    )
    return combined.drop(columns=["priority"]).reset_index(drop=True)  # return a stable map


def apply_team_name_map(matches: pd.DataFrame, team_name_map: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical team names to home and away columns.

    Parameters
    ----------
    matches : pandas.DataFrame
        Match table with home_team and away_team columns.
    team_name_map : pandas.DataFrame
        Name map with source_name and canonical_name columns.

    Returns
    -------
    pandas.DataFrame
        Copy of matches with canonical home_team and away_team values.

    Raises
    ------
    ValueError
        If any team name is not covered by the map.
    """
    name_lookup = dict(zip(team_name_map["source_name"], team_name_map["canonical_name"]))  # map dict
    cleaned = matches.copy()  # avoid mutating the caller's DataFrame
    cleaned["home_team"] = cleaned["home_team"].map(name_lookup)  # canonicalise home teams
    cleaned["away_team"] = cleaned["away_team"].map(name_lookup)  # canonicalise away teams
    unmapped_rows = cleaned[["home_team", "away_team"]].isna().any(axis=1)  # rows with failures
    if bool(unmapped_rows.any()):  # fail loudly if mapping coverage is incomplete
        raise ValueError(f"Unmapped team rows: {int(unmapped_rows.sum())}")  # clear failure count
    return cleaned  # hand back canonicalised rows


def clean_match_table(raw_matches: pd.DataFrame, former_names: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the canonical completed-match table and team-name map.

    Parameters
    ----------
    raw_matches : pandas.DataFrame
        Raw D1 match table.
    former_names : pandas.DataFrame
        Raw D1 former-name table.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Canonical match table and team-name map.

    Notes
    -----
    Rows with null scores are scheduled/unplayed fixtures in the martj42 file,
    not completed matches. They are excluded from the historical training table
    because the Stage 1 checks require non-null scores and the model must not use
    future outcomes.
    """
    team_name_map = build_team_name_map(raw_matches, former_names)  # build full coverage map
    completed = raw_matches.dropna(subset=["home_score", "away_score"]).copy()  # drop unplayed rows
    completed = apply_team_name_map(completed, team_name_map)  # canonicalise team names
    canonical = pd.DataFrame(  # assemble only the required canonical columns
        {
            "date": pd.to_datetime(completed["date"], errors="raise"),  # typed match date
            "home_team": completed["home_team"].astype(str),  # canonical home team
            "away_team": completed["away_team"].astype(str),  # canonical away team
            "home_goals": completed["home_score"].astype("int16"),  # home goals as compact int
            "away_goals": completed["away_score"].astype("int16"),  # away goals as compact int
            "tournament": completed["tournament"].astype(str),  # competition name
            "neutral": completed["neutral"].astype(bool),  # neutral-venue flag
            "venue_country": completed["country"].astype(str),  # host country for the fixture
        }
    )
    canonical["importance_weight"] = assign_k_factors(canonical).astype("float64")  # K weight
    canonical = canonical.sort_values("date", kind="mergesort").reset_index(drop=True)  # stable order
    validate_clean_matches(canonical, team_name_map)  # fail before writing invalid outputs
    return canonical, team_name_map  # return both output tables


def validate_clean_matches(matches: pd.DataFrame, team_name_map: pd.DataFrame) -> dict[str, object]:
    """Validate the canonical match table and return check metrics.

    Parameters
    ----------
    matches : pandas.DataFrame
        Canonical match table.
    team_name_map : pandas.DataFrame
        Name map used to canonicalise source teams.

    Returns
    -------
    dict[str, object]
        Row counts, date span, and validation counters.

    Raises
    ------
    AssertionError
        If any required Stage 1.3 invariant fails.
    """
    required_columns = [  # exact canonical columns required by Task 1.3
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "tournament",
        "neutral",
        "venue_country",
        "importance_weight",
    ]
    assert list(matches.columns) == required_columns  # enforce the canonical schema
    assert int(matches[["home_team", "away_team"]].isna().sum().sum()) == 0  # no null teams
    assert int(matches[["home_goals", "away_goals"]].isna().sum().sum()) == 0  # no null scores
    assert int((matches[["home_goals", "away_goals"]] < 0).sum().sum()) == 0  # no negative goals
    assert team_name_map["source_name"].is_unique  # every source name maps once
    return {  # expose validation facts to logs and run records
        "rows": int(len(matches)),
        "date_min": matches["date"].min().date().isoformat(),
        "date_max": matches["date"].max().date().isoformat(),
        "team_name_map_rows": int(len(team_name_map)),
        "unique_teams": int(pd.concat([matches["home_team"], matches["away_team"]]).nunique()),
        "null_team_rows": int(matches[["home_team", "away_team"]].isna().any(axis=1).sum()),
        "null_score_rows": int(matches[["home_goals", "away_goals"]].isna().any(axis=1).sum()),
        "negative_goal_cells": int((matches[["home_goals", "away_goals"]] < 0).sum().sum()),
    }


def write_clean_outputs(matches: pd.DataFrame, team_name_map: pd.DataFrame) -> tuple[Path, Path]:
    """Write canonical matches and team-name map to data/interim.

    Parameters
    ----------
    matches : pandas.DataFrame
        Canonical match table.
    team_name_map : pandas.DataFrame
        Source-to-canonical team-name map.

    Returns
    -------
    (pathlib.Path, pathlib.Path)
        Written parquet path and CSV map path.
    """
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    matches.to_parquet(MATCHES_CLEAN_FILE, index=False)  # parquet is canonical typed storage
    team_name_map.to_csv(TEAM_NAME_MAP_FILE, index=False)  # CSV is human-auditable
    return MATCHES_CLEAN_FILE, TEAM_NAME_MAP_FILE  # return written paths
