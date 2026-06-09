"""Validate the Stage 1 goal engine against a heldout match set."""

from __future__ import annotations  # consistent type-hint behaviour

import json  # update latest parameter metadata after validation
import shutil  # promote the selected full-fit candidate into the latest path
from dataclasses import replace  # clone fitted model with calibrated score matrix
from datetime import datetime, timezone  # timestamp validation outputs
from pathlib import Path  # filesystem paths

import pandas as pd  # validation tables and SPI archive inspection

from gyan.config import (  # central paths and model constants
    DATA_PROCESSED,
    DEFAULT_SCORE_DISTRIBUTION,
    DIXON_COLES_PARAMS_DC_FILE,
    DIXON_COLES_MAX_TRAIN_MATCHES,
    DIXON_COLES_PARAMS_PLAIN_FILE,
    DIXON_COLES_RECENT_MIN_DATE,
    DIXON_COLES_RIDGE,
    DIXON_COLES_VALIDATION_CUTOFF,
    DIXON_COLES_XI,
    DRAW_RATE_TOLERANCE,
    ENGINE_DRAW_CALIBRATION_FILE,
    ENGINE_VALIDATION_FILE,
    ENGINE_VALIDATION_SPI_FILE,
    GLOBAL_SEED,
    MATCHES_WITH_ELO_FILE,
    NEGATIVE_BINOMIAL_DISPERSION,
    OUTPUTS_TABLES,
    SCORE_DISTRIBUTION_POISSON,
    SPI_MATCHES_FILE,
    create_directories,
)
from gyan.engine.dixon_coles import (  # fit/evaluate helpers
    evaluate_model_rps,
    fit_dixon_coles,
    observed_outcome_index,
    ranked_probability_score,
)
from gyan.features.socioeconomic import canonical_football_team  # align SPI team labels to D1 labels
from gyan.utils.logging import RunRecord, get_run_logger  # required audit logging


VALIDATION_MAXITER: int = 90  # shorter fit budget for leakage-free validation refits
SPI_BENCHMARK_CUTOFF: str = "2022-01-01"  # D3 scored benchmark window after this date


def _timestamp() -> str:
    """Return a compact UTC timestamp for output filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # sortable timestamp tag


def _modern_train_test_split(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return leakage-free modern-era train and heldout validation tables."""
    dated = matches.copy()  # do not mutate caller input
    dated["date"] = pd.to_datetime(dated["date"])  # ensure datetime comparisons are valid
    modern = dated[dated["date"] >= pd.Timestamp(DIXON_COLES_RECENT_MIN_DATE)].copy()  # modern era
    train = modern[modern["date"] < pd.Timestamp(DIXON_COLES_VALIDATION_CUTOFF)].copy()  # pre-cutoff
    test = modern[modern["date"] >= pd.Timestamp(DIXON_COLES_VALIDATION_CUTOFF)].copy()  # heldout
    if len(train) > DIXON_COLES_MAX_TRAIN_MATCHES:  # cap optimisation rows if needed
        train = train.tail(DIXON_COLES_MAX_TRAIN_MATCHES).copy()  # most recent training rows
    known_teams = set(train["home_team"]).union(set(train["away_team"]))  # teams fitted in train
    test = test[test["home_team"].isin(known_teams) & test["away_team"].isin(known_teams)].copy()  # overlap
    test["row_id"] = range(len(test))  # stable validation row key for per-match merges
    return train.reset_index(drop=True), test.reset_index(drop=True)  # stable indices


def _spi_benchmark_rows(path: Path, train: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Return scored D3 international SPI rows for a leakage-free benchmark."""
    spi = pd.read_csv(path)  # load archived D3 international SPI match forecasts
    required_columns = {  # columns needed to score SPI against observed outcomes
        "date", "league", "team1", "team2", "prob1", "prob2", "probtie", "score1", "score2"
    }
    missing_columns = required_columns.difference(spi.columns)  # source schema drift guard
    if missing_columns:  # fail loudly if D3 is no longer the international forecast file
        raise ValueError(f"D3 SPI archive is missing required columns: {sorted(missing_columns)}")
    spi = spi.dropna(subset=["score1", "score2", "prob1", "prob2", "probtie"]).copy()  # scored forecasts only
    spi["date"] = pd.to_datetime(spi["date"])  # enable benchmark cutoff
    spi = spi[spi["date"] >= pd.Timestamp(SPI_BENCHMARK_CUTOFF)].copy()  # scored benchmark window
    spi["home_team"] = spi["team1"].map(canonical_football_team)  # align names
    spi["away_team"] = spi["team2"].map(canonical_football_team)  # align names
    spi["home_goals"] = spi["score1"].astype(int)  # model/evaluator column name
    spi["away_goals"] = spi["score2"].astype(int)  # model/evaluator column name
    known_teams = set(train["home_team"]).union(set(train["away_team"]))  # fitted teams only
    spi = spi[spi["home_team"].isin(known_teams) & spi["away_team"].isin(known_teams)].copy()  # known teams
    d1_neutral = matches.copy()  # use D1 only to fill neutral-site flags when exact rows match
    d1_neutral["date"] = pd.to_datetime(d1_neutral["date"])  # normalise date type
    d1_neutral = d1_neutral[["date", "home_team", "away_team", "home_goals", "away_goals", "neutral"]]  # needed cols
    spi = spi.merge(  # enrich with D1 neutral flags where possible
        d1_neutral,
        on=["date", "home_team", "away_team", "home_goals", "away_goals"],
        how="left",
        suffixes=("", "_d1"),
    )
    spi["neutral"] = spi["neutral"].fillna(False).astype(bool)  # SPI team1 is a home side when D1 cannot confirm
    spi["row_id"] = range(len(spi))  # stable benchmark row key
    return spi.reset_index(drop=True)  # benchmark rows


def _score_spi_benchmark(spi_rows: pd.DataFrame) -> pd.DataFrame:
    """Score FiveThirtyEight SPI W/D/L probabilities with ranked probability score."""
    rows: list[dict[str, object]] = []  # collect per-row SPI scores
    for row in spi_rows.itertuples(index=False):  # row-wise scoring keeps intent clear
        outcome = observed_outcome_index(int(row.home_goals), int(row.away_goals))  # observed W/D/L
        rows.append(  # match evaluate_model_rps shape where useful
            {
                "row_id": int(row.row_id),
                "league": row.league,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_goals": int(row.home_goals),
                "away_goals": int(row.away_goals),
                "spi_p_home": float(row.prob1),
                "spi_p_draw": float(row.probtie),
                "spi_p_away": float(row.prob2),
                "rps_spi": ranked_probability_score((float(row.prob1), float(row.probtie), float(row.prob2)), outcome),
            }
        )
    return pd.DataFrame(rows)  # per-match SPI benchmark score


def _update_latest_engine_selection(selected_engine: str, metrics: dict[str, object]) -> None:
    """Promote the selected full-fit engine candidate into the stable latest parameter file."""
    latest_path = DATA_PROCESSED / "dixon_coles_params_latest.json"  # stable parameter alias
    candidate_path = DIXON_COLES_PARAMS_DC_FILE if selected_engine == "dixon_coles" else DIXON_COLES_PARAMS_PLAIN_FILE
    if not candidate_path.exists():  # fitting may not have produced candidate params yet
        return  # validation can still write its own outputs
    shutil.copyfile(candidate_path, latest_path)  # latest must contain the selected candidate parameters
    payload = json.loads(latest_path.read_text(encoding="utf-8"))  # read latest params
    payload["selected_engine"] = selected_engine  # record DC or T-G3 plain fallback
    if selected_engine == "plain_poisson":  # keep plain means but use calibrated score matrix
        payload["score_distribution"] = DEFAULT_SCORE_DISTRIBUTION  # draw-calibrated matrix family
        payload["score_dispersion"] = NEGATIVE_BINOMIAL_DISPERSION  # shared-frailty NB shape
    else:  # Dixon-Coles already carries low-score dependence in rho
        payload["score_distribution"] = SCORE_DISTRIBUTION_POISSON  # DC/Poisson matrix path
        payload["score_dispersion"] = NEGATIVE_BINOMIAL_DISPERSION  # harmless persisted default
    payload["validation_metrics"] = metrics  # attach validation summary
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")  # persist update


def _draw_calibration_row(name: str, frame: pd.DataFrame, p_draw_column: str) -> dict[str, object]:
    """Return draw-rate calibration metrics for one validation forecast table."""
    observed_draw_rate = float((frame["home_goals"] == frame["away_goals"]).mean())  # actual heldout draws
    predicted_draw_rate = float(frame[p_draw_column].mean())  # average forecast draw probability
    delta = predicted_draw_rate - observed_draw_rate  # signed probability-point gap
    return {  # compact audit row
        "model": name,
        "rows": int(len(frame)),
        "observed_draw_rate": observed_draw_rate,
        "predicted_draw_rate": predicted_draw_rate,
        "delta_pred_minus_observed": delta,
        "abs_delta": abs(delta),
        "tolerance": DRAW_RATE_TOLERANCE,
        "passed": abs(delta) <= DRAW_RATE_TOLERANCE,
    }


def main() -> None:
    """Run the Stage 1.8 engine validation step."""
    create_directories()  # ensure output directories exist
    tag = _timestamp()  # shared output timestamp
    validation_path = OUTPUTS_TABLES / f"engine_validation_{tag}.csv"  # timestamped validation table
    logger, log_path = get_run_logger("s1_validate_engine", stage="stage1", step="validate_engine")  # log
    with RunRecord(  # create machine-readable audit record
        stage="stage1",
        step="validate_engine",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_input(MATCHES_WITH_ELO_FILE)  # hash Stage 1.4 match features
        record.add_input(SPI_MATCHES_FILE)  # hash cached SPI archive used for source audit
        record.add_input(DIXON_COLES_PARAMS_DC_FILE)  # hash full-fit DC candidate
        record.add_input(DIXON_COLES_PARAMS_PLAIN_FILE)  # hash full-fit no-rho candidate
        record.add_output(log_path)  # record human-readable log path
        matches = pd.read_parquet(MATCHES_WITH_ELO_FILE)  # load match feature table
        train, test = _modern_train_test_split(matches)  # leakage-free train/test split
        spi_train, _ = _modern_train_test_split(  # separate no-leakage train set for D3 benchmark window
            matches[pd.to_datetime(matches["date"]) < pd.Timestamp(SPI_BENCHMARK_CUTOFF)].copy()
        )
        spi_rows = _spi_benchmark_rows(SPI_MATCHES_FILE, spi_train, matches)  # real D3 international benchmark
        if len(spi_rows) == 0:  # D3 benchmark should never silently disappear
            raise RuntimeError("No scored D3 international SPI benchmark rows after filtering")  # fail source drift
        dc_model, dc_diagnostics = fit_dixon_coles(  # fit DC on train only
            train,
            xi=DIXON_COLES_XI,
            fit_rho=True,
            maxiter=VALIDATION_MAXITER,
            ridge=DIXON_COLES_RIDGE,
        )
        baseline_model, baseline_diagnostics = fit_dixon_coles(  # fit matching plain Poisson candidate
            train,
            xi=DIXON_COLES_XI,
            fit_rho=False,
            maxiter=VALIDATION_MAXITER,
            ridge=DIXON_COLES_RIDGE,
        )
        dc_eval = evaluate_model_rps(dc_model, test, force_rho_zero=False).rename(columns={"rps": "rps_dixon_coles"})  # DC
        baseline_eval = evaluate_model_rps(baseline_model, test, force_rho_zero=False).rename(columns={"rps": "rps_plain_poisson"})  # no-rho
        calibrated_model = replace(  # keep baseline means; swap only the scoreline matrix family
            baseline_model,
            score_distribution=DEFAULT_SCORE_DISTRIBUTION,
            score_dispersion=NEGATIVE_BINOMIAL_DISPERSION,
        )
        calibrated_eval = evaluate_model_rps(calibrated_model, test, force_rho_zero=False).rename(  # NB score matrix
            columns={
                "p_home": "p_home_calibrated_score",
                "p_draw": "p_draw_calibrated_score",
                "p_away": "p_away_calibrated_score",
                "rps": "rps_calibrated_score",
            }
        )
        validation = dc_eval.merge(  # align model and baseline per-match RPS
            baseline_eval[["row_id", "p_home", "p_draw", "p_away", "rps_plain_poisson"]].rename(
                columns={
                    "p_home": "p_home_plain_poisson",
                    "p_draw": "p_draw_plain_poisson",
                    "p_away": "p_away_plain_poisson",
                }
            ),
            on="row_id",
            how="inner",
        )
        validation = validation.merge(  # align calibrated draw/RPS diagnostics
            calibrated_eval[
                [
                    "row_id",
                    "p_home_calibrated_score",
                    "p_draw_calibrated_score",
                    "p_away_calibrated_score",
                    "rps_calibrated_score",
                ]
            ],
            on="row_id",
            how="inner",
        )
        validation["rps_delta_dc_minus_baseline"] = validation["rps_dixon_coles"] - validation["rps_plain_poisson"]  # delta
        mean_dc_rps = float(validation["rps_dixon_coles"].mean())  # DC mean RPS
        mean_baseline_rps = float(validation["rps_plain_poisson"].mean())  # baseline mean RPS
        mean_calibrated_rps = float(validation["rps_calibrated_score"].mean())  # calibrated matrix RPS
        selected_engine = "dixon_coles" if mean_dc_rps <= mean_baseline_rps else "plain_poisson"  # T1/T-G3
        t_g3_actioned = selected_engine == "plain_poisson"  # fallback flag
        spi_model, spi_model_diagnostics = fit_dixon_coles(  # fit no-leakage model for D3 benchmark
            spi_train,
            xi=DIXON_COLES_XI,
            fit_rho=selected_engine == "dixon_coles",
            maxiter=VALIDATION_MAXITER,
            ridge=DIXON_COLES_RIDGE,
        )
        model_spi_eval = evaluate_model_rps(spi_model, spi_rows, force_rho_zero=False).rename(columns={"rps": "rps_model"})
        spi_eval = _score_spi_benchmark(spi_rows)  # SPI probability RPS
        spi_validation = model_spi_eval.merge(  # align model and SPI probabilities per row
            spi_eval[["row_id", "league", "spi_p_home", "spi_p_draw", "spi_p_away", "rps_spi"]],
            on="row_id",
            how="inner",
        )
        spi_validation["rps_delta_model_minus_spi"] = spi_validation["rps_model"] - spi_validation["rps_spi"]  # benchmark delta
        draw_calibration = pd.DataFrame(  # leakage-free draw frequency audit
            [
                _draw_calibration_row("dixon_coles", validation, "p_draw"),
                _draw_calibration_row("plain_poisson", validation, "p_draw_plain_poisson"),
                _draw_calibration_row("calibrated_score_matrix", validation, "p_draw_calibrated_score"),
            ]
        )
        validation.to_csv(validation_path, index=False)  # timestamped validation artifact
        validation.to_csv(ENGINE_VALIDATION_FILE, index=False)  # stable latest validation artifact
        spi_validation.to_csv(ENGINE_VALIDATION_SPI_FILE, index=False)  # stable D3 SPI benchmark artifact
        draw_calibration.to_csv(ENGINE_DRAW_CALIBRATION_FILE, index=False)  # stable draw audit artifact
        metrics = {  # concise validation metrics for summary and params
            "train_rows": int(len(train)),
            "heldout_rows": int(len(test)),
            "mean_rps_dixon_coles": mean_dc_rps,
            "mean_rps_plain_poisson": mean_baseline_rps,
            "mean_rps_calibrated_score_matrix": mean_calibrated_rps,
            "selected_engine": selected_engine,
            "t_g3_actioned": t_g3_actioned,
            "score_distribution": DEFAULT_SCORE_DISTRIBUTION if selected_engine == "plain_poisson" else SCORE_DISTRIBUTION_POISSON,
            "score_dispersion": NEGATIVE_BINOMIAL_DISPERSION,
            "observed_draw_rate": float(draw_calibration.loc[draw_calibration["model"] == "plain_poisson", "observed_draw_rate"].iloc[0]),
            "plain_poisson_predicted_draw_rate": float(draw_calibration.loc[draw_calibration["model"] == "plain_poisson", "predicted_draw_rate"].iloc[0]),
            "plain_poisson_draw_delta": float(draw_calibration.loc[draw_calibration["model"] == "plain_poisson", "delta_pred_minus_observed"].iloc[0]),
            "plain_poisson_draw_frequency_passed": bool(draw_calibration.loc[draw_calibration["model"] == "plain_poisson", "passed"].iloc[0]),
            "calibrated_predicted_draw_rate": float(draw_calibration.loc[draw_calibration["model"] == "calibrated_score_matrix", "predicted_draw_rate"].iloc[0]),
            "calibrated_draw_delta": float(draw_calibration.loc[draw_calibration["model"] == "calibrated_score_matrix", "delta_pred_minus_observed"].iloc[0]),
            "calibrated_draw_frequency_passed": bool(draw_calibration.loc[draw_calibration["model"] == "calibrated_score_matrix", "passed"].iloc[0]),
            "spi_international_rows": int(len(spi_rows)),
            "spi_benchmark_cutoff": SPI_BENCHMARK_CUTOFF,
            "mean_rps_model_on_spi_benchmark": float(spi_validation["rps_model"].mean()),
            "mean_rps_spi_benchmark": float(spi_validation["rps_spi"].mean()),
            "mean_rps_delta_model_minus_spi": float(spi_validation["rps_delta_model_minus_spi"].mean()),
        }
        _update_latest_engine_selection(selected_engine, metrics)  # record selected engine in latest params
        record.add_params(  # record validation split and caveat
            {
                "validation_cutoff": DIXON_COLES_VALIDATION_CUTOFF,
                "recent_min_date": DIXON_COLES_RECENT_MIN_DATE,
                "xi_dixon_coles": DIXON_COLES_XI,
                "xi_plain_poisson": DIXON_COLES_XI,
                "validation_maxiter": VALIDATION_MAXITER,
                "validation_source": "D1 heldout for engine selection plus archived D3 international SPI scored benchmark",
                "spi_benchmark_cutoff": SPI_BENCHMARK_CUTOFF,
            }
        )
        record.record["dixon_coles_fit_diagnostics"] = dc_diagnostics  # full DC fit diagnostics
        record.record["plain_poisson_fit_diagnostics"] = baseline_diagnostics  # full baseline diagnostics
        record.record["spi_benchmark_fit_diagnostics"] = spi_model_diagnostics  # D3 benchmark fit diagnostics
        record.add_metrics(metrics)  # record headline validation metrics
        record.add_output_artifact(validation_path)  # hashed timestamped validation output
        record.add_output_artifact(ENGINE_VALIDATION_FILE)  # hashed latest validation output
        record.add_output_artifact(ENGINE_VALIDATION_SPI_FILE)  # hashed D3 SPI benchmark output
        record.add_output_artifact(ENGINE_DRAW_CALIBRATION_FILE)  # hashed draw audit output
        logger.info("Wrote engine validation table to %s", validation_path)  # log path
        logger.info("Validation metrics: %s", metrics)  # log metrics
        if int(len(test)) == 0:  # validation requires heldout rows
            raise RuntimeError("No heldout validation matches after team-overlap filtering")  # fail


if __name__ == "__main__":  # allow direct script execution
    main()  # run validation
