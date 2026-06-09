"""Build World Football Elo features from the canonical match table."""

from __future__ import annotations  # use modern type hints consistently

from gyan.config import (  # import all paths and constants from the single source of truth
    ELO_REFERENCE_TEAM_LABELS_FILE,
    ELO_REFERENCE_WORLD_FILE,
    ELO_CURRENT_RATINGS_FILE,
    ELO_SPOTCHECK_FILE,
    GLOBAL_SEED,
    MATCHES_CLEAN_FILE,
    MATCHES_WITH_ELO_FILE,
    create_directories,
)
from gyan.features.elo import (  # Elo feature and spot-check helpers
    build_elo_spotcheck,
    compute_elo_features,
    final_ratings_from_elo_features,
    parse_eloratings_reference,
)
from gyan.utils.logging import RunRecord, get_run_logger  # required logging/run record helpers

import pandas as pd  # read and write tabular match/rating data


SELECTED_SPOTCHECK_TEAMS: tuple[str, ...] = (  # top teams with clean name alignment
    "Spain",
    "Argentina",
    "France",
    "England",
    "Portugal",
    "Germany",
    "Netherlands",
    "Croatia",
)
SPOTCHECK_TOLERANCE: float = 75.0  # PRD Task 1.4 tolerance against eloratings.net
CURRENT_RATING_ALIASES: dict[str, str] = {  # align D1 Elo labels to FIFA/2026 field labels
    "Curaçao": "Curacao",
    "DR Congo": "Democratic Republic of the Congo",
}


def _add_current_rating_aliases(final_ratings: pd.DataFrame) -> pd.DataFrame:
    """Add duplicate current-rating rows for known cross-source team aliases."""
    existing = set(final_ratings["team"].astype(str))  # current label set
    alias_rows: list[pd.Series] = []  # collect real-rating alias rows
    for source_label, target_label in CURRENT_RATING_ALIASES.items():  # one alias at a time
        if source_label not in existing or target_label in existing:  # only add needed aliases
            continue  # no action needed
        row = final_ratings.loc[final_ratings["team"] == source_label].iloc[0].copy()  # source rating row
        row["team"] = target_label  # downstream tournament label
        alias_rows.append(row)  # collect alias row
    if not alias_rows:  # no aliases needed
        return final_ratings  # unchanged
    return (  # include aliases and keep ranking order
        pd.concat([final_ratings, pd.DataFrame(alias_rows)], ignore_index=True)
        .sort_values("elo_rating", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    """Run the Stage 1.4 Elo feature generation step."""
    create_directories()  # ensure processed/output/log directories exist
    logger, log_path = get_run_logger(  # create the human-readable stage log
        "s1_build_elo",
        stage="stage1",
        step="build_elo",
    )
    with RunRecord(  # create the machine-readable audit record
        stage="stage1",
        step="build_elo",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_input(MATCHES_CLEAN_FILE)  # hash the canonical match input
        record.add_input(ELO_REFERENCE_WORLD_FILE)  # hash the D2 reference rating table
        record.add_input(ELO_REFERENCE_TEAM_LABELS_FILE)  # hash the D2 code labels
        record.add_output(log_path)  # include the human-readable log path
        matches = pd.read_parquet(MATCHES_CLEAN_FILE)  # load canonical completed matches
        matches_with_elo = compute_elo_features(matches, round_changes=True)  # official rounded Elo
        MATCHES_WITH_ELO_FILE.parent.mkdir(parents=True, exist_ok=True)  # ensure processed dir exists
        matches_with_elo.to_parquet(MATCHES_WITH_ELO_FILE, index=False)  # write feature table
        final_ratings_raw = final_ratings_from_elo_features(matches_with_elo)  # latest team ratings
        final_ratings = _add_current_rating_aliases(final_ratings_raw)  # add cross-source aliases
        final_ratings.to_csv(ELO_CURRENT_RATINGS_FILE, index=False)  # write latest ratings CSV
        reference_ratings = parse_eloratings_reference(  # parse eloratings World.tsv
            ELO_REFERENCE_WORLD_FILE,
            ELO_REFERENCE_TEAM_LABELS_FILE,
        )
        spotcheck = build_elo_spotcheck(  # compare against selected reference teams
            final_ratings,
            reference_ratings,
            selected_teams=SELECTED_SPOTCHECK_TEAMS,
            tolerance=SPOTCHECK_TOLERANCE,
        )
        spotcheck.to_csv(ELO_SPOTCHECK_FILE, index=False)  # write all merged reference rows
        selected = spotcheck[spotcheck["selected_for_check"]]  # formal threshold subset
        max_selected_delta = float(selected["abs_rating_delta"].max())  # largest selected delta
        n_selected_failures = int((~selected["within_tolerance"]).sum())  # selected failures
        record.add_param("round_changes", True)  # record official rounded update mode
        record.add_param("spotcheck_tolerance", SPOTCHECK_TOLERANCE)  # record threshold
        record.add_param("selected_spotcheck_teams", SELECTED_SPOTCHECK_TEAMS)  # record teams
        record.add_metric("rows", int(len(matches_with_elo)))  # record output row count
        record.add_metric("unique_teams", int(final_ratings["team"].nunique()))  # team count
        record.add_metric("current_rating_alias_rows_added", int(len(final_ratings) - len(final_ratings_raw)))  # aliases
        record.add_metric("max_selected_abs_delta", max_selected_delta)  # tolerance metric
        record.add_metric("n_selected_failures", n_selected_failures)  # selected failure count
        record.add_output_artifact(MATCHES_WITH_ELO_FILE)  # record hashed parquet output
        record.add_output_artifact(ELO_CURRENT_RATINGS_FILE)  # record hashed ratings output
        record.add_output_artifact(ELO_SPOTCHECK_FILE)  # record hashed spot-check output
        logger.info("Wrote Elo feature table to %s", MATCHES_WITH_ELO_FILE)  # log feature path
        logger.info("Wrote current ratings to %s", ELO_CURRENT_RATINGS_FILE)  # log rating path
        logger.info("Wrote spot-check table to %s", ELO_SPOTCHECK_FILE)  # log check path
        logger.info("Selected max abs delta %.1f with %s failures", max_selected_delta, n_selected_failures)
        if n_selected_failures > 0:  # selected teams must stay within PRD tolerance
            raise RuntimeError("Selected Elo spot-check teams exceeded tolerance")  # fail loudly


if __name__ == "__main__":  # allow direct execution as a stage script
    main()  # run the entry point
