"""Clean D1 match results into the canonical Stage 1 match table."""

from __future__ import annotations  # use modern type hints consistently

from gyan.config import GLOBAL_SEED, create_directories  # shared project constants
from gyan.data.clean import (  # import the cleaning pipeline functions
    RAW_FORMER_NAMES_FILE,
    RAW_RESULTS_FILE,
    clean_match_table,
    load_former_names,
    load_raw_results,
    validate_clean_matches,
    write_clean_outputs,
)
from gyan.utils.logging import RunRecord, get_run_logger  # required logging/run record helpers


def main() -> None:
    """Run the Stage 1.3 canonical match-table cleaning step."""
    create_directories()  # ensure output folders exist before writing
    logger, log_path = get_run_logger(  # create the human-readable stage log
        "s1_clean_matches",
        stage="stage1",
        step="clean_matches",
    )
    with RunRecord(  # create the machine-readable audit record
        stage="stage1",
        step="clean_matches",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_input(RAW_RESULTS_FILE)  # hash the raw match results input
        record.add_input(RAW_FORMER_NAMES_FILE)  # hash the raw former-name input
        record.add_output(log_path)  # include the human-readable log path
        raw_matches = load_raw_results()  # load D1 raw matches
        former_names = load_former_names()  # load D1 former-name helper table
        matches, team_name_map = clean_match_table(raw_matches, former_names)  # clean tables
        matches_path, name_map_path = write_clean_outputs(matches, team_name_map)  # persist outputs
        metrics = validate_clean_matches(matches, team_name_map)  # collect validation metrics
        record.add_metrics(metrics)  # record every check metric
        record.add_output(matches_path)  # record canonical parquet output
        record.add_output(name_map_path)  # record canonical name-map output
        logger.info("Wrote canonical matches to %s", matches_path)  # log match output path
        logger.info("Wrote team-name map to %s", name_map_path)  # log map output path
        logger.info("Clean-match metrics: %s", metrics)  # log check metrics


if __name__ == "__main__":  # allow direct execution as a stage script
    main()  # run the entry point
