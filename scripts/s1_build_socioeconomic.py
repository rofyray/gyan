"""Build socioeconomic features and OLS diagnostics for Stage 1.5."""

from __future__ import annotations  # use modern type hints consistently

import pandas as pd  # load Task 1.4 Elo ratings for the OLS target

from gyan.config import (  # all paths and constants come from the single source of truth
    COUNTRY_TEMPERATURE_FILE,
    ELO_CURRENT_RATINGS_FILE,
    FIFA_RANKINGS_API_FILE,
    GLOBAL_SEED,
    KLEMENT_TARGET_R_SQUARED,
    SOCIOECONOMIC_FEATURES_CSV,
    SOCIOECONOMIC_FEATURES_FILE,
    WORLD_BANK_GDP_PPP_FILE,
    WORLD_BANK_POPULATION_FILE,
    HOFFMANN_2002_R_SQUARED,
    create_directories,
)
from gyan.features.socioeconomic import (  # Task 1.5 feature assembly helpers
    build_feature_table,
    fit_socioeconomic_models,
    parse_country_temperatures,
    parse_fifa_rankings,
    parse_world_bank_latest,
    validate_feature_table,
)
from gyan.utils.logging import RunRecord, get_run_logger  # required logs/run records


def main() -> None:
    """Run the Stage 1.5 socioeconomic feature-generation step."""
    create_directories()  # ensure output directories exist
    logger, log_path = get_run_logger(  # create the human-readable run log
        "s1_build_socioeconomic",
        stage="stage1",
        step="build_socioeconomic",
    )
    with RunRecord(  # create the machine-readable audit record
        stage="stage1",
        step="build_socioeconomic",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_input(FIFA_RANKINGS_API_FILE)  # hash the official FIFA ranking JSON
        record.add_input(WORLD_BANK_GDP_PPP_FILE)  # hash World Bank GDP/capita PPP input
        record.add_input(WORLD_BANK_POPULATION_FILE)  # hash World Bank population input
        record.add_input(COUNTRY_TEMPERATURE_FILE)  # hash the D12 temperature fallback table
        record.add_input(ELO_CURRENT_RATINGS_FILE)  # hash the Task 1.4 Elo ratings target
        record.add_output(log_path)  # include the human-readable log path
        fifa_rankings = parse_fifa_rankings(FIFA_RANKINGS_API_FILE)  # parse D5 current points
        gdp_latest = parse_world_bank_latest(WORLD_BANK_GDP_PPP_FILE, "gdp_per_capita_ppp")  # D11 GDP
        population_latest = parse_world_bank_latest(WORLD_BANK_POPULATION_FILE, "population")  # D11 pop
        temperatures = parse_country_temperatures(COUNTRY_TEMPERATURE_FILE)  # D12 country mean annual temp
        elo_ratings = pd.read_csv(ELO_CURRENT_RATINGS_FILE)  # current final Elo ratings
        features = build_feature_table(  # assemble one row per FIFA-ranked team
            fifa_rankings=fifa_rankings,
            gdp_latest=gdp_latest,
            population_latest=population_latest,
            temperatures=temperatures,
            elo_ratings=elo_ratings,
        )
        metrics = validate_feature_table(features)  # run coverage checks before writing
        model_summaries = fit_socioeconomic_models(features)  # fit pure and augmented OLS
        SOCIOECONOMIC_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)  # ensure data dir
        SOCIOECONOMIC_FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)  # ensure table dir
        features.to_parquet(SOCIOECONOMIC_FEATURES_FILE, index=False)  # canonical typed output
        features.to_csv(SOCIOECONOMIC_FEATURES_CSV, index=False)  # human-readable audit output
        pure_r_squared = model_summaries["pure_hoffmann"]["r_squared"]  # pure R^2 diagnostic
        augmented_r_squared = model_summaries["augmented_fifa_points"]["r_squared"]  # aug R^2
        r_squared_diagnosis = (  # explain threshold interpretation for the run record
            "OLS target is current final Elo because FIFA points are an augmented predictor; "
            "the PRD reference R^2 values are therefore diagnostic, not directly comparable. "
            "Country mean annual temperature is used directly as the documented D12 climate feature."
        )
        record.add_params(  # record source/threshold interpretation
            {
                "ols_target": "elo_rating",
                "temperature_source": "country_mean_annual_temperature",
                "pure_hoffmann_reference_r_squared": HOFFMANN_2002_R_SQUARED,
                "augmented_reference_r_squared": KLEMENT_TARGET_R_SQUARED,
            }
        )
        record.add_metrics(metrics)  # record feature coverage metrics
        record.add_metric("pure_hoffmann_r_squared", pure_r_squared)  # record pure R^2
        record.add_metric("augmented_fifa_points_r_squared", augmented_r_squared)  # augmented R^2
        record.add_metric("r_squared_diagnosis", r_squared_diagnosis)  # record threshold diagnosis
        record.record["model_summaries"] = model_summaries  # store coefficients and R^2 values
        record.add_output_artifact(SOCIOECONOMIC_FEATURES_FILE)  # hashed parquet output
        record.add_output_artifact(SOCIOECONOMIC_FEATURES_CSV)  # hashed CSV output
        logger.info("Wrote socioeconomic features to %s", SOCIOECONOMIC_FEATURES_FILE)  # log path
        logger.info("Wrote socioeconomic audit CSV to %s", SOCIOECONOMIC_FEATURES_CSV)  # log path
        logger.info("Pure Hoffmann R^2 %.3f", pure_r_squared)  # log pure model diagnostic
        logger.info("Augmented FIFA-points R^2 %.3f", augmented_r_squared)  # log augmented diagnostic
        logger.info("Feature coverage metrics: %s", metrics)  # log feature coverage metrics


if __name__ == "__main__":  # allow direct execution as a stage script
    main()  # run the entry point
