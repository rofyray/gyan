"""Build 2026 named-squad value features for Stage 1.6."""

from __future__ import annotations  # keep type-hint behaviour consistent

import argparse  # CLI switch for forced Transfermarkt page refreshes

from gyan.config import (  # all paths and constants come from config.py
    ESPN_2026_SQUADS_FILE,
    GLOBAL_SEED,
    INJURIES_FILE,
    SQUAD_FEATURES_2026_CSV,
    SQUAD_FEATURES_2026_FILE,
    SQUAD_PLAYERS_2026_FILE,
    TRANSFERMARKT_NATIONAL_TEAMS_FILE,
    TRANSFERMARKT_PLAYERS_FILE,
    TRANSFERMARKT_TEAM_PAGES_DIR,
    UEFA_VALUE_DISCOUNT,
    WIKIPEDIA_2026_SQUADS_FILE,
    create_directories,
)
from gyan.features.squad_value import (  # Stage 1.6 builder and D4 page cache refresher
    parse_wikipedia_squads,
    refresh_transfermarkt_team_pages,
    build_squad_value_features,
)
from gyan.utils.logging import RunRecord, get_run_logger  # required logging/run records


def parse_args() -> argparse.Namespace:
    """Parse command-line options for squad-value feature generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-team-pages",
        action="store_true",
        help="Overwrite existing Transfermarkt national-team page cache.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Stage 1.6 squad-value feature-generation step."""
    args = parse_args()
    create_directories()  # ensure output directories exist
    logger, log_path = get_run_logger(  # create human-readable stage log
        "s1_build_squad_value",
        stage="stage1",
        step="build_squad_value",
    )
    with RunRecord(  # create machine-readable audit record
        stage="stage1",
        step="build_squad_value",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_input(WIKIPEDIA_2026_SQUADS_FILE)  # hash primary D6 squad page
        record.add_input(ESPN_2026_SQUADS_FILE)  # hash D6 ESPN cross-check artifact
        record.add_input(TRANSFERMARKT_PLAYERS_FILE)  # hash D4 individual player values
        record.add_input(TRANSFERMARKT_NATIONAL_TEAMS_FILE)  # hash D4 national-team metadata
        record.add_input(INJURIES_FILE)  # hash editable injury snapshot
        record.add_output(log_path)  # record the human-readable log path
        record.add_param("force_transfermarkt_team_pages", args.force_team_pages)  # raw D4 page refresh mode
        squads = parse_wikipedia_squads(WIKIPEDIA_2026_SQUADS_FILE)  # get 2026 field/team labels
        page_paths = refresh_transfermarkt_team_pages(  # refresh/cache true player-value pages
            TRANSFERMARKT_NATIONAL_TEAMS_FILE,
            TRANSFERMARKT_TEAM_PAGES_DIR,
            sorted(squads["team"].unique()),
            force=args.force_team_pages,
        )
        for page_path in page_paths:  # hash every cached team page as an input
            record.add_input(page_path)  # D4 raw national-team page
        features, players, injury_audit, metrics = build_squad_value_features(  # build features
            wikipedia_squads_path=WIKIPEDIA_2026_SQUADS_FILE,
            transfermarkt_players_path=TRANSFERMARKT_PLAYERS_FILE,
            transfermarkt_national_teams_path=TRANSFERMARKT_NATIONAL_TEAMS_FILE,
            transfermarkt_team_pages_dir=TRANSFERMARKT_TEAM_PAGES_DIR,
            injuries_path=INJURIES_FILE,
        )
        SQUAD_FEATURES_2026_FILE.parent.mkdir(parents=True, exist_ok=True)  # ensure data dir
        SQUAD_FEATURES_2026_CSV.parent.mkdir(parents=True, exist_ok=True)  # ensure table dir
        features.to_parquet(SQUAD_FEATURES_2026_FILE, index=False)  # canonical typed output
        features.to_csv(SQUAD_FEATURES_2026_CSV, index=False)  # human-readable team audit
        players.to_csv(SQUAD_PLAYERS_2026_FILE, index=False)  # human-readable player audit
        injury_audit_path = SQUAD_PLAYERS_2026_FILE.with_name("squad_injury_adjustments_2026.csv")  # audit path
        injury_audit.to_csv(injury_audit_path, index=False)  # write injury adjustment audit
        record.add_param("uefa_value_discount", UEFA_VALUE_DISCOUNT)  # record PELE-style discount
        record.add_param("player_value_method", "D4 individual Transfermarkt player market values from national-team pages/profile CSV")  # note
        record.add_metrics(metrics)  # record feature/data-quality metrics
        record.add_output_artifact(SQUAD_FEATURES_2026_FILE)  # hashed parquet output
        record.add_output_artifact(SQUAD_FEATURES_2026_CSV)  # hashed team CSV output
        record.add_output_artifact(SQUAD_PLAYERS_2026_FILE)  # hashed player CSV output
        record.add_output_artifact(injury_audit_path)  # hashed injury audit output
        logger.info("Wrote squad feature table to %s", SQUAD_FEATURES_2026_FILE)  # log path
        logger.info("Wrote squad player audit to %s", SQUAD_PLAYERS_2026_FILE)  # log path
        logger.info("Squad value metrics: %s", metrics)  # log metrics for paper audit
        if metrics["teams"] != 48:  # Stage 1.6 expects every 2026 team
            raise RuntimeError(f"Expected 48 squad teams, found {metrics['teams']}")  # fail loudly
        if metrics["most_valuable_squad_value_eur"] < 900_000_000.0:  # PRD ballpark sanity
            raise RuntimeError("Most valuable squad is below expected one-billion-euro order")  # fail


if __name__ == "__main__":  # allow direct script execution
    main()  # run the entry point
