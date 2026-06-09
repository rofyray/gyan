"""Build 2026 tournament structure artifacts for Stage 2.1."""

from __future__ import annotations  # keep type hints consistent

from gyan.config import (  # central paths and constants
    BRACKET_PAIRINGS_2026_FILE,
    FIFA_OFFICIAL_2026_WORLD_CUP_FILE,
    FIFA_OFFICIAL_2026_MATCH_SCHEDULE_PDF,
    GLOBAL_SEED,
    GROUPS_2026_FILE,
    SCHEDULE_2026_FILE,
    WIKIPEDIA_2026_KNOCKOUT_FILE,
    WIKIPEDIA_2026_WORLD_CUP_FILE,
    create_directories,
)
from gyan.simulation.structure import (  # parsing and writing helpers
    parse_annex_c,
    parse_groups,
    parse_schedule,
    validate_against_official_fifa_schedule,
    write_structure_artifacts,
)
from gyan.utils.logging import RunRecord, get_run_logger  # required run records/logging


def main() -> None:
    """Run the Stage 2.1 structure-building step."""
    create_directories()  # ensure output directories exist
    logger, log_path = get_run_logger("s2_build_structure", stage="stage2", step="build_structure")  # log
    with RunRecord(  # machine-readable audit record
        stage="stage2",
        step="build_structure",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_input(WIKIPEDIA_2026_WORLD_CUP_FILE)  # hash cached main D7 page
        record.add_input(WIKIPEDIA_2026_KNOCKOUT_FILE)  # hash cached knockout/Annex C page
        record.add_input(FIFA_OFFICIAL_2026_WORLD_CUP_FILE)  # hash official FIFA shell artifact
        record.add_input(FIFA_OFFICIAL_2026_MATCH_SCHEDULE_PDF)  # hash official FIFA schedule PDF
        record.add_output(log_path)  # record human-readable log path
        groups = parse_groups(WIKIPEDIA_2026_WORLD_CUP_FILE)  # parse 12 groups
        schedule = parse_schedule(WIKIPEDIA_2026_WORLD_CUP_FILE, groups)  # parse 104 fixtures
        official_checks = validate_against_official_fifa_schedule(  # official FIFA source guard
            groups,
            schedule,
            FIFA_OFFICIAL_2026_MATCH_SCHEDULE_PDF,
        )
        bracket = parse_annex_c(WIKIPEDIA_2026_KNOCKOUT_FILE)  # parse Annex C and tree
        write_structure_artifacts(  # write all Stage 2.1 processed artifacts
            groups,
            schedule,
            bracket,
            GROUPS_2026_FILE,
            SCHEDULE_2026_FILE,
            BRACKET_PAIRINGS_2026_FILE,
        )
        metrics = {  # structural checks for the run record
            "groups": int(groups["group"].nunique()),
            "teams": int(groups["team"].nunique()),
            "matches": int(len(schedule)),
            "group_matches": int((schedule["stage"] == "group").sum()),
            "r32_matches": int((schedule["stage"] == "R32").sum()),
            "annex_c_combinations": int(len(bracket["third_place_combinations"])),
            "non_neutral_matches": int((~schedule["neutral"]).sum()),
            **official_checks,
        }
        record.add_params(  # record source note and known limitation
            {
                "structure_source": "official FIFA match-schedule PDF validated against parsed D7 tables; Wikipedia supplies parseable dates/venues and Annex C table",
                "official_fifa_source_note": "FIFA HTML remains a JavaScript shell, but the official DigitalHub schedule PDF is now required and checked",
            }
        )
        record.add_metrics(metrics)  # record structural checks
        record.add_output_artifact(GROUPS_2026_FILE)  # hashed group output
        record.add_output_artifact(SCHEDULE_2026_FILE)  # hashed schedule output
        record.add_output_artifact(BRACKET_PAIRINGS_2026_FILE)  # hashed bracket output
        logger.info("Wrote groups to %s", GROUPS_2026_FILE)  # log group path
        logger.info("Wrote schedule to %s", SCHEDULE_2026_FILE)  # log schedule path
        logger.info("Wrote bracket pairings to %s", BRACKET_PAIRINGS_2026_FILE)  # log bracket path
        logger.info("Structure metrics: %s", metrics)  # log checks


if __name__ == "__main__":  # allow direct execution
    main()  # run script
