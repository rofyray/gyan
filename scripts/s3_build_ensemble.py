"""Build the Stage 3 GYAN ensemble forecast."""

from __future__ import annotations  # consistent type hints

import argparse  # CLI flags for simulation draw count and workers
import os  # thread caps and plotting backend
import time  # runtime metrics
from datetime import datetime, timezone  # timestamped outputs
from pathlib import Path  # typed output paths

os.environ.setdefault("OMP_NUM_THREADS", "1")  # cap BLAS/OpenMP threads per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # cap OpenBLAS threads
os.environ.setdefault("MKL_NUM_THREADS", "1")  # cap MKL threads
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # cap macOS Accelerate threads
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")  # cap numexpr threads
os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting backend
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-gyan")  # writable matplotlib cache
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")  # writable font cache

import matplotlib.pyplot as plt  # noqa: E402  # plotting after environment setup
import numpy as np  # noqa: E402  # arrays and numeric metrics
import pandas as pd  # noqa: E402  # table IO
from joblib import Parallel, delayed  # noqa: E402  # process-parallel simulations

from gyan.config import (  # noqa: E402  # central paths and constants
    BRACKET_PAIRINGS_2026_FILE,
    BACKTEST_MARKET_OUTRIGHTS_FILE,
    BACKTEST_TOURNAMENTS,
    DIXON_COLES_PARAMS_LATEST_FILE,
    ELO_CURRENT_RATINGS_FILE,
    ENSEMBLE_WEIGHTS_FILE,
    EXPERT_CORRELATION_DIAGNOSTICS_FILE,
    EXPERT_BOARDS_2026_FILE,
    FINAL_FREEZE_TIMESTAMP_UTC,
    FINAL_RELEASE_TAG,
    GLOBAL_SEED,
    GROUPS_2026_FILE,
    GYAN_FORECAST_LATEST,
    GOLDMAN_2026_TOP_PROBS,
    MARKET_IMPLIED_LIVE_FILE,
    MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE,
    MARKET_SOURCE_DIVERGENCE_FILE,
    MATCHES_WITH_ELO_FILE,
    N_SIMULATIONS_HIGH,
    OUTPUTS_FIGURES,
    OUTPUTS_TABLES,
    RELIABILITY_DIAGRAM_PDF,
    RELIABILITY_DIAGRAM_PNG,
    SCHEDULE_2026_FILE,
    SHOOTOUTS_RAW_FILE,
    SOCIOECONOMIC_FEATURES_FILE,
    SQUAD_FEATURES_2026_FILE,
    STAGE3_DEFAULT_N_SIMULATIONS,
    STAGE3_DEFAULT_N_WORKERS,
    STAGE3_MIN_SHIPPED_EXPERT_WEIGHT,
    STAGE3_SUMMARY_FILE,
    STAGE3_VALIDATION_CUTOFF,
    TEAM_ADVANCEMENT_ENGINEONLY_LATEST,
    T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD,
    WORLD_BANK_GDP_PPP_FILE,
    WORLD_BANK_POPULATION_FILE,
    YIELD_DELTA_TABLE_FILE,
    create_directories,
    repo_path_str,
)
from gyan.engine.dixon_coles import DixonColesModel  # noqa: E402  # Stage 1 goal model
from gyan.ensemble.experts import (  # noqa: E402  # expert interface and helpers
    GoalExpert,
    STAGE_PROB_COLUMNS,
    StrengthExpert,
    board_from_champion_vector,
    build_socioeconomic_feature_series,
    build_yield_feature_series,
    outcome_vector_from_score,
    ratings_from_feature,
    strength_model_from_ratings,
    validate_expert_board,
)
from gyan.ensemble.market import (  # noqa: E402  # market vector builder/live source pull
    build_live_market_snapshot,
    refresh_live_market_source_cache,
)
from gyan.ensemble.pooling import (  # noqa: E402  # opinion pools and optimisation
    linear_opinion_pool,
    log_opinion_pool,
    mean_rps_for_weights,
    optimise_weights,
    pool_many,
)
from gyan.evaluation.scoring import mean_scores  # noqa: E402  # RPS/Brier/log-loss
from gyan.evaluation.backtest import (  # noqa: E402  # no-leakage historical weight calibration
    expert_match_forecasts,
    historical_expert_rating_snapshots,
    load_world_bank_series,
    prepare_tournament_backtest,
)
from gyan.simulation.tournament import (  # noqa: E402  # Stage 2 simulator
    aggregate_chunks,
    load_structure,
    prepare_simulation_inputs,
    run_tournaments_for_indices,
)
from gyan.utils.logging import RunRecord, get_run_logger  # noqa: E402  # run records/logging


EXPERT_ORDER: tuple[str, ...] = ("goal", "yield_named", "socioeconomic", "market")  # shipped experts
MATCH_VALIDATION_EXPERT_ORDER: tuple[str, ...] = ("goal", "yield_named", "socioeconomic")  # real W/D/L experts
POOL_RPS_MATERIALITY: float = 0.001  # require >=0.1pp RPS gain before sacrificing calibration


def _timestamp() -> str:
    """Return a compact UTC timestamp for output filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # sortable timestamp


def _split_indices(n_sims: int, n_workers: int) -> list[list[int]]:
    """Split simulation indices into deterministic chunks."""
    chunks = np.array_split(np.arange(n_sims, dtype=int), n_workers)  # contiguous chunks
    return [chunk.astype(int).tolist() for chunk in chunks if len(chunk)]  # non-empty chunks only


def _run_strength_board(
    model: DixonColesModel,
    structure: dict[str, object],
    elo_ratings: pd.DataFrame,
    n_sims: int,
    n_workers: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate a tournament board for one feature-derived strength model."""
    prepared = prepare_simulation_inputs(model, structure, elo_ratings=elo_ratings)  # precompute fixtures
    teams = prepared["groups"]["team"].tolist()  # stable team order
    sim_chunks = _split_indices(n_sims, n_workers)  # deterministic chunks
    if len(sim_chunks) == 1:  # direct single-worker path
        chunks = [run_tournaments_for_indices(sim_chunks[0], seed, prepared)]  # run directly
    else:  # joblib process-parallel path
        chunks = Parallel(n_jobs=len(sim_chunks), backend="loky")(  # run each chunk
            delayed(run_tournaments_for_indices)(chunk, seed, prepared) for chunk in sim_chunks
        )
    probabilities, _metrics = aggregate_chunks(chunks, teams, n_sims)  # aggregate stage probabilities
    return probabilities  # expert board


def _write_expert_board(board_map: dict[str, pd.DataFrame], tag: str) -> tuple[Path, Path]:
    """Write all expert boards as a long table."""
    rows: list[pd.DataFrame] = []  # collect expert boards
    for expert_name, board in board_map.items():  # one expert board at a time
        expert_frame = board.copy()  # defensive copy
        expert_frame.insert(0, "expert", expert_name)  # add expert label
        rows.append(expert_frame)  # collect long frame
    long_frame = pd.concat(rows, ignore_index=True)  # combined expert table
    parquet_path = OUTPUTS_TABLES / f"expert_boards_2026_{tag}.parquet"  # timestamped parquet
    csv_path = OUTPUTS_TABLES / f"expert_boards_2026_{tag}.csv"  # timestamped csv
    long_frame.to_parquet(parquet_path, index=False)  # canonical typed output
    long_frame.to_csv(csv_path, index=False)  # human-readable output
    long_frame.to_csv(EXPERT_BOARDS_2026_FILE, index=False)  # latest CSV alias
    return parquet_path, csv_path  # timestamped paths


def _build_validation_frame(matches: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Return held-out validation matches with both teams in the 2026 field."""
    heldout = matches[pd.to_datetime(matches["date"]) >= pd.Timestamp(STAGE3_VALIDATION_CUTOFF)].copy()  # heldout date split
    filtered = heldout[heldout["home_team"].isin(teams) & heldout["away_team"].isin(teams)].copy()  # feature coverage
    return filtered.sort_values("date", kind="mergesort").reset_index(drop=True)  # stable order


def _expert_forecast_tensor(experts: list[object], validation_matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return expert W/D/L tensor and one-hot outcomes for validation matches."""
    forecasts: list[np.ndarray] = []  # observation forecast tensor rows
    outcomes: list[np.ndarray] = []  # one-hot outcome rows
    for row in validation_matches.itertuples(index=False):  # one match at a time
        expert_probs = [expert.predict_match(row.home_team, row.away_team, bool(row.neutral)) for expert in experts]  # expert WDL
        forecasts.append(np.vstack(expert_probs))  # one observation, all experts
        outcomes.append(outcome_vector_from_score(int(row.home_goals), int(row.away_goals)))  # realised WDL
    return np.asarray(forecasts, dtype=float), np.asarray(outcomes, dtype=float)  # tensor and outcomes


def _single_expert_scores(expert: object, validation_matches: pd.DataFrame, outcomes: np.ndarray) -> dict[str, float]:
    """Return validation scores for one expert."""
    forecasts = np.asarray(  # one W/D/L row per validation match
        [expert.predict_match(row.home_team, row.away_team, bool(row.neutral)) for row in validation_matches.itertuples(index=False)],
        dtype=float,
    )
    return mean_scores(forecasts, outcomes)  # scoring-rule means


def _split_train_validation(
    expert_forecasts: np.ndarray,
    outcomes: np.ndarray,
    validation_matches: pd.DataFrame,
) -> dict[str, object]:
    """Split held-out matches into weight-training and validation slices."""
    split_index = max(1, int(len(validation_matches) * 0.6))  # chronological 60/40 split
    return {  # split payload
        "train_forecasts": expert_forecasts[:split_index],
        "train_outcomes": outcomes[:split_index],
        "validation_forecasts": expert_forecasts[split_index:],
        "validation_outcomes": outcomes[split_index:],
        "train_rows": split_index,
        "validation_rows": len(validation_matches) - split_index,
    }


def _evaluate_pool_options(split: dict[str, object]) -> dict[str, object]:
    """Optimise and compare linear/log/equal-weight pools."""
    train_forecasts = split["train_forecasts"]  # training forecast tensor
    train_outcomes = split["train_outcomes"]  # training outcomes
    validation_forecasts = split["validation_forecasts"]  # validation forecast tensor
    validation_outcomes = split["validation_outcomes"]  # validation outcomes
    n_experts = train_forecasts.shape[1]  # expert count
    equal_weights = np.full(n_experts, 1.0 / n_experts, dtype=float)  # equal-weight baseline
    options: dict[str, dict[str, object]] = {}  # pool option diagnostics
    for pool_name, pool_fn in {"linear": linear_opinion_pool, "log": log_opinion_pool}.items():  # candidate pools
        optimised_weights, diagnostics = optimise_weights(train_forecasts, train_outcomes, pool_fn=pool_fn)  # fit weights
        equal_train_rps = mean_rps_for_weights(equal_weights, train_forecasts, train_outcomes, pool_fn=pool_fn)  # train equal
        opt_train_rps = mean_rps_for_weights(optimised_weights, train_forecasts, train_outcomes, pool_fn=pool_fn)  # train opt
        equal_validation_rps = mean_rps_for_weights(equal_weights, validation_forecasts, validation_outcomes, pool_fn=pool_fn)  # val equal
        opt_validation_rps = mean_rps_for_weights(optimised_weights, validation_forecasts, validation_outcomes, pool_fn=pool_fn)  # val opt
        ship_weights = optimised_weights if opt_validation_rps <= equal_validation_rps + 1e-6 else equal_weights  # avoid overfit
        ship_reason = "optimised" if opt_validation_rps <= equal_validation_rps + 1e-6 else "equal_weights_validation_guard"  # reason
        ship_validation_rps = min(opt_validation_rps, equal_validation_rps)  # selected validation RPS
        options[pool_name] = {  # diagnostics for this pool
            "pool_fn": pool_fn,
            "optimised_weights": optimised_weights,
            "equal_weights": equal_weights,
            "ship_weights": ship_weights,
            "ship_reason": ship_reason,
            "optimiser": diagnostics,
            "train_rps_equal": float(equal_train_rps),
            "train_rps_optimised": float(opt_train_rps),
            "validation_rps_equal": float(equal_validation_rps),
            "validation_rps_optimised": float(opt_validation_rps),
            "validation_rps_selected": float(ship_validation_rps),
        }
    chosen_name = min(options, key=lambda name: options[name]["validation_rps_selected"])  # best selected validation RPS
    return {"options": options, "chosen_pool": chosen_name}  # option payload


def _fit_historical_four_expert_pool(
    matches: pd.DataFrame,
    feature_map: pd.DataFrame,
    expert_order: tuple[str, ...],
) -> dict[str, object]:
    """Fit the shipped four-expert pool on historical no-leakage World Cup backtests."""
    shootouts = pd.read_csv(SHOOTOUTS_RAW_FILE)  # D1 shootout winners for historical knockout champions
    gdp = load_world_bank_series(WORLD_BANK_GDP_PPP_FILE)  # D11 GDP/capita PPP history
    population = load_world_bank_series(WORLD_BANK_POPULATION_FILE)  # D11 population history
    forecast_blocks: list[np.ndarray] = []  # per-tournament expert forecast tensors
    outcome_blocks: list[np.ndarray] = []  # per-tournament realised W/D/L outcomes
    forecasts_by_year: dict[int, np.ndarray] = {}  # year -> expert forecast tensor
    outcomes_by_year: dict[int, np.ndarray] = {}  # year -> realised outcomes
    match_counts_by_year: dict[int, int] = {}  # year -> matches in the weight-fit objective
    leakage_audits: dict[int, dict[str, object]] = {}  # no-leakage audit by tournament
    market_sources: dict[int, str] = {}  # historical D15 market source by tournament
    for year in BACKTEST_TOURNAMENTS:  # fit from 2014/2018/2022 held-out tournaments
        tournament = prepare_tournament_backtest(matches, shootouts, year)  # tournament payload
        leakage_audits[year] = tournament.leakage_audit  # retain leakage proof
        ratings, _champion_overrides, market_statuses = historical_expert_rating_snapshots(  # expert snapshots
            tournament,
            matches,
            feature_map,
            gdp,
            population,
            BACKTEST_MARKET_OUTRIGHTS_FILE,
        )
        market_sources[year] = market_statuses["market"]  # D15 source label
        expert_forecasts: list[np.ndarray] = []  # one tournament, all experts
        tournament_outcomes: np.ndarray | None = None  # shared realised outcomes
        for expert in expert_order:  # preserve the shipped expert order
            forecasts, outcomes = expert_match_forecasts(ratings[expert], tournament)  # W/D/L probabilities
            expert_forecasts.append(forecasts)  # collect this expert
            tournament_outcomes = outcomes if tournament_outcomes is None else tournament_outcomes  # first outcomes are shared
        year_forecasts = np.stack(expert_forecasts, axis=1)  # shape: matches x experts x classes
        year_outcomes = np.asarray(tournament_outcomes, dtype=float)  # shape: matches x classes
        forecasts_by_year[year] = year_forecasts  # retain fold block
        outcomes_by_year[year] = year_outcomes  # retain fold outcomes
        match_counts_by_year[year] = int(year_forecasts.shape[0])  # record objective composition
        forecast_blocks.append(year_forecasts)  # shape: matches x experts x classes
        outcome_blocks.append(year_outcomes)  # shape: matches x classes
    forecast_tensor = np.concatenate(forecast_blocks, axis=0)  # all historical matches
    outcome_matrix = np.concatenate(outcome_blocks, axis=0)  # all historical outcomes
    match_objective_profile = _historical_match_objective_profile(match_counts_by_year)  # group/knockout fit profile
    options: dict[str, dict[str, object]] = {}  # pool diagnostics
    for pool_name, pool_fn in {"linear": linear_opinion_pool, "log": log_opinion_pool}.items():  # candidate pools
        equal_weights = np.full(len(expert_order), 1.0 / len(expert_order), dtype=float)  # equal baseline
        unconstrained_weights, unconstrained_diagnostics = optimise_weights(  # unconstrained reference fit
            forecast_tensor,
            outcome_matrix,
            pool_fn=pool_fn,
            min_weight=0.0,
        )
        constrained_weights, diagnostics = optimise_weights(  # RPS-optimal weights with all experts active
            forecast_tensor,
            outcome_matrix,
            pool_fn=pool_fn,
            min_weight=STAGE3_MIN_SHIPPED_EXPERT_WEIGHT,
        )
        loto = _leave_one_tournament_out_pool_validation(
            forecasts_by_year,
            outcomes_by_year,
            expert_order,
            pool_fn=pool_fn,
            min_weight=STAGE3_MIN_SHIPPED_EXPERT_WEIGHT,
        )
        ship_weights = constrained_weights if loto["optimised_beats_equal_oos"] else equal_weights  # guard against overfit
        ship_reason = "optimised_weights_loto_beats_equal" if loto["optimised_beats_equal_oos"] else "equal_weights_loto_guard"
        pooled = pool_many(forecast_tensor, constrained_weights, pool_fn=pool_fn)  # fitted historical forecasts
        calibration = _pool_calibration_metrics(pooled, outcome_matrix)  # calibration/sharpness diagnostics
        options[pool_name] = {  # record this pool's historical fit
            "pool_fn": pool_fn,
            "equal_weights": equal_weights,
            "optimised_weights": constrained_weights,
            "ship_weights": ship_weights,
            "ship_reason": ship_reason,
            "unconstrained_weights": unconstrained_weights,
            "optimiser": diagnostics,
            "unconstrained_optimiser": unconstrained_diagnostics,
            "leave_one_tournament_out": loto,
            "historical_rps_equal": float(mean_rps_for_weights(equal_weights, forecast_tensor, outcome_matrix, pool_fn=pool_fn)),
            "historical_rps_unconstrained": float(mean_rps_for_weights(unconstrained_weights, forecast_tensor, outcome_matrix, pool_fn=pool_fn)),
            "historical_rps_optimised": float(mean_rps_for_weights(constrained_weights, forecast_tensor, outcome_matrix, pool_fn=pool_fn)),
            "historical_calibration": calibration,
        }
    chosen_pool, selection_reason = _choose_historical_pool(options)  # calibration-aware selection
    chosen = options[chosen_pool]  # selected historical fit
    return {  # serialisable historical pool payload
        "options": options,
        "chosen_pool": chosen_pool,
        "selection_reason": selection_reason,
        "pool_fn": chosen["pool_fn"],
        "weights": chosen["ship_weights"],
        "ship_reason": chosen["ship_reason"],
        "leave_one_tournament_out": chosen["leave_one_tournament_out"],
        "floor_policy": "implementation_addition_min_weight_0.05_to_keep_all_four_experts_active",
        "floor_binding_experts": [
            expert
            for expert, constrained_weight, unconstrained_weight in zip(expert_order, chosen["optimised_weights"], chosen["unconstrained_weights"], strict=True)
            if np.isclose(float(constrained_weight), STAGE3_MIN_SHIPPED_EXPERT_WEIGHT) and float(unconstrained_weight) < STAGE3_MIN_SHIPPED_EXPERT_WEIGHT
        ],
        "floor_rps_cost_vs_unconstrained": float(chosen["historical_rps_optimised"] - chosen["historical_rps_unconstrained"]),
        "selected_scores": {
            "in_sample_mean_rps": chosen["historical_rps_optimised"],
            "leave_one_tournament_out_mean_rps": chosen["leave_one_tournament_out"]["selected_oos_rps"],
        },
        "equal_scores": {
            "in_sample_mean_rps": chosen["historical_rps_equal"],
            "leave_one_tournament_out_mean_rps": chosen["leave_one_tournament_out"]["equal_oos_rps"],
        },
        "n_matches": int(forecast_tensor.shape[0]),
        "min_weight": STAGE3_MIN_SHIPPED_EXPERT_WEIGHT,
        "match_objective_profile": match_objective_profile,
        "market_expert_construction": {
            "historical_raw_input": "D15 bookmaker champion outrights only; no historical match-level market W/D/L prices are used.",
            "historical_match_level_transform": "Normalised champion probabilities are converted to log-probability ratings with market_ratings_from_champion_probabilities, anchored to the pre-tournament Elo mean, then converted to W/D/L through rating_to_match_probs using the correlated negative-binomial score matrix.",
            "historical_champion_scoring": "Historical champion probabilities are retained directly as champion_override when scoring tournament-level market performance.",
            "live_2026_raw_input": "Live Polymarket/Kalshi/bookmaker champion probabilities are blended and de-vigged directly into the 2026 Market p_champion vector.",
            "live_2026_stage_transform": "Live non-champion Market stages are reconstructed by board_from_champion_vector, scaling the Stage 2 engine path shape to the exact live champion vector.",
            "transfer_assessment": "The transfer is consistent at the champion-probability primitive, but historical match-level RPS weights a market-strength hybrid and live 2026 non-champion stages are engine-shaped; describe the Market weight as champion-market-informed with structural transforms rather than raw match-price weight.",
            "limitation_note_required": True,
        },
        "leakage_audits": leakage_audits,
        "market_sources": market_sources,
    }


def _historical_match_objective_profile(match_counts_by_year: dict[int, int]) -> dict[str, object]:
    """Return the historical match-level objective's group/knockout composition."""
    expected_matches_per_32_team_world_cup = 64  # 48 group + 16 knockout in 2014/2018/2022
    group_matches_per_tournament = 48
    knockout_matches_per_tournament = 16
    unexpected_counts = {
        year: n_matches
        for year, n_matches in match_counts_by_year.items()
        if n_matches != expected_matches_per_32_team_world_cup
    }
    if unexpected_counts:
        raise ValueError(f"Historical objective profile expected 64 matches per 32-team tournament, got {unexpected_counts}")
    group_matches_by_year = {year: group_matches_per_tournament for year in sorted(match_counts_by_year)}
    knockout_matches_by_year = {year: knockout_matches_per_tournament for year in sorted(match_counts_by_year)}
    total_matches = int(sum(match_counts_by_year.values()))
    group_matches = int(sum(group_matches_by_year.values()))
    knockout_matches = int(sum(knockout_matches_by_year.values()))
    return {
        "objective": "mean_match_rps",
        "historical_tournaments": sorted(match_counts_by_year),
        "matches_by_year": {year: int(match_counts_by_year[year]) for year in sorted(match_counts_by_year)},
        "matches_total": total_matches,
        "group_stage_matches_by_year": group_matches_by_year,
        "knockout_matches_by_year": knockout_matches_by_year,
        "group_stage_matches": group_matches,
        "knockout_matches": knockout_matches,
        "group_stage_share": float(group_matches / total_matches),
        "knockout_share": float(knockout_matches / total_matches),
        "format_assumption": "2014/2018/2022 use the 32-team World Cup format with 48 group-stage and 16 knockout matches per tournament.",
        "deployment_note": (
            "Weights are calibrated to group-stage-heavy match-level RPS and then applied to a "
            "2026 tournament board whose champion probabilities are sensitive to knockout dynamics, "
            "bracket paths, and variance; this can partially favor market/strength signals in the fitted weights."
        ),
    }


def _leave_one_tournament_out_pool_validation(
    forecasts_by_year: dict[int, np.ndarray],
    outcomes_by_year: dict[int, np.ndarray],
    expert_order: tuple[str, ...],
    pool_fn,
    min_weight: float,
) -> dict[str, object]:
    """Fit on two historical tournaments and evaluate on the held-out third."""
    years = sorted(forecasts_by_year)  # deterministic fold order
    equal_weights = np.full(len(expert_order), 1.0 / len(expert_order), dtype=float)  # equal baseline
    folds: list[dict[str, object]] = []  # per-holdout diagnostics
    weighted_opt_total = 0.0  # row-weighted RPS numerator
    weighted_equal_total = 0.0  # row-weighted RPS numerator
    n_total = 0  # held-out rows across folds
    for holdout_year in years:
        train_years = [year for year in years if year != holdout_year]  # fit on other tournaments
        train_forecasts = np.concatenate([forecasts_by_year[year] for year in train_years], axis=0)
        train_outcomes = np.concatenate([outcomes_by_year[year] for year in train_years], axis=0)
        holdout_forecasts = forecasts_by_year[holdout_year]
        holdout_outcomes = outcomes_by_year[holdout_year]
        fitted_weights, diagnostics = optimise_weights(
            train_forecasts,
            train_outcomes,
            pool_fn=pool_fn,
            min_weight=min_weight,
        )
        opt_rps = float(mean_rps_for_weights(fitted_weights, holdout_forecasts, holdout_outcomes, pool_fn=pool_fn))
        equal_rps = float(mean_rps_for_weights(equal_weights, holdout_forecasts, holdout_outcomes, pool_fn=pool_fn))
        n_matches = int(holdout_forecasts.shape[0])
        weighted_opt_total += opt_rps * n_matches
        weighted_equal_total += equal_rps * n_matches
        n_total += n_matches
        folds.append(
            {
                "holdout_year": int(holdout_year),
                "train_years": [int(year) for year in train_years],
                "n_holdout_matches": n_matches,
                "optimised_holdout_rps": opt_rps,
                "equal_holdout_rps": equal_rps,
                "delta_optimised_minus_equal": opt_rps - equal_rps,
                "train_optimised_weights": {expert: float(weight) for expert, weight in zip(expert_order, fitted_weights, strict=True)},
                "optimizer": diagnostics,
            }
        )
    opt_oos = weighted_opt_total / n_total
    equal_oos = weighted_equal_total / n_total
    optimised_beats_equal = bool(opt_oos <= equal_oos + 1e-9)
    return {
        "folds": folds,
        "optimised_oos_rps": float(opt_oos),
        "equal_oos_rps": float(equal_oos),
        "delta_optimised_minus_equal": float(opt_oos - equal_oos),
        "optimised_beats_equal_oos": optimised_beats_equal,
        "selected_oos_rps": float(opt_oos if optimised_beats_equal else equal_oos),
        "selected_weight_set": "optimised" if optimised_beats_equal else "equal",
    }


def _pool_calibration_metrics(forecasts: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """Return calibration and sharpness diagnostics for pooled historical forecasts."""
    scores = mean_scores(forecasts, outcomes)  # RPS/Brier/log-loss
    predicted = forecasts.reshape(-1)  # all class probabilities
    observed = outcomes.reshape(-1)  # all realised class indicators
    ece = 0.0  # weighted expected calibration error over coarse bins
    bins = np.linspace(0.0, 1.0, 7)  # match Stage 4 backtest-calibration bins
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (predicted >= lower) & (predicted < upper if upper < 1.0 else predicted <= upper)
        if mask.any():
            ece += float(mask.sum() / len(predicted) * abs(float(predicted[mask].mean()) - float(observed[mask].mean())))
    entropy = float(np.mean(-(forecasts * np.log(np.clip(forecasts, 1e-15, 1.0))).sum(axis=1)))  # lower is sharper
    max_probability = float(np.mean(forecasts.max(axis=1)))  # higher is sharper
    return {
        **scores,
        "ece_6_bins": ece,
        "mean_entropy": entropy,
        "mean_max_probability": max_probability,
    }


def _choose_historical_pool(options: dict[str, dict[str, object]]) -> tuple[str, str]:
    """Choose the shipped pool using RPS only when the gain is material and calibrated."""
    linear = options["linear"]["historical_calibration"]
    logged = options["log"]["historical_calibration"]
    rps_gain_log = float(linear["mean_rps"] - logged["mean_rps"])  # positive means log improves RPS
    log_calibration_guard = bool(
        logged["mean_log_loss"] <= linear["mean_log_loss"]
        and logged["ece_6_bins"] <= linear["ece_6_bins"]
    )
    if rps_gain_log > POOL_RPS_MATERIALITY and log_calibration_guard:
        return "log", f"log_selected_material_rps_gain_{rps_gain_log:.6f}_and_calibration_guard_passed"
    if rps_gain_log > 0.0:
        return "linear", f"linear_selected_log_rps_gain_{rps_gain_log:.6f}_below_materiality_or_calibration_guard_failed"
    return "linear", "linear_selected_best_historical_rps"


def _apply_thresholds(
    pool_result: dict[str, object],
    split: dict[str, object],
    expert_order: tuple[str, ...],
    market_board: pd.DataFrame,
) -> dict[str, object]:
    """Apply T-G1 and T-G2 decision gates to selected pool weights."""
    options = pool_result["options"]  # pool diagnostics
    chosen_name = pool_result["chosen_pool"]  # selected pool name
    chosen = options[chosen_name]  # selected pool diagnostics
    pool_fn = chosen["pool_fn"]  # selected pool function
    validation_forecasts = split["validation_forecasts"]  # validation forecasts
    validation_outcomes = split["validation_outcomes"]  # validation outcomes
    weights = np.asarray(chosen["ship_weights"], dtype=float).copy()  # selected weights
    actions: dict[str, object] = {"T-G1": "market_outrights_used_for_champion_benchmark_only", "T-G2": "none", "T-G4": "none"}  # threshold actions
    market_scores = {  # market is a champion-board benchmark, not a W/D/L validation expert
        "available": False,
        "reason": "live market source supplies tournament outrights; match validation uses W/D/L-capable experts",
        "champion_sum": float(market_board["p_champion"].sum()),
    }
    selected_forecasts = pool_many(validation_forecasts, weights, pool_fn=pool_fn)  # selected ensemble validation forecasts
    selected_scores = mean_scores(selected_forecasts, validation_outcomes)  # selected validation scores
    socio_index = expert_order.index("socioeconomic")  # socioeconomic expert index
    if weights[socio_index] > 0.0:  # only test if socio contributes
        no_socio = weights.copy()  # no-socio candidate
        no_socio[socio_index] = 0.0  # drop socio
        no_socio = no_socio / no_socio.sum() if no_socio.sum() > 0 else no_socio  # renormalise
        no_socio_scores = mean_scores(pool_many(validation_forecasts, no_socio, pool_fn=pool_fn), validation_outcomes)  # score no-socio
        if no_socio_scores["mean_rps"] < selected_scores["mean_rps"] - 1e-9:  # T-G2 tripped
            weights = no_socio  # ship no-socio weights
            actions["T-G2"] = "socioeconomic_weight_set_to_0_on_validation_guard"  # record action
            selected_scores = no_socio_scores  # selected metrics are no-socio metrics
    return {  # threshold result payload
        "chosen_pool": chosen_name,
        "pool_fn": pool_fn,
        "weights": weights,
        "threshold_actions": actions,
        "market_scores": market_scores,
        "selected_scores": selected_scores,
    }


def _evaluate_t_g4(forecast: pd.DataFrame, market_board: pd.DataFrame) -> dict[str, object]:
    """Evaluate top-team divergence against Goldman and market benchmarks."""
    forecast_lookup = forecast.set_index("team")["p_champion"]  # GYAN champion probabilities
    market_lookup = market_board.set_index("team")["p_champion"]  # market champion probabilities
    rows: list[dict[str, object]] = []  # collect benchmark comparison rows
    for team, goldman_probability in GOLDMAN_2026_TOP_PROBS.items():  # D9 top benchmark teams
        if team not in forecast_lookup.index or team not in market_lookup.index:  # skip absent teams
            continue  # not in the 2026 field
        gyan_probability = float(forecast_lookup.loc[team])  # GYAN p champion
        market_probability = float(market_lookup.loc[team])  # market p champion
        goldman_delta = abs(gyan_probability - goldman_probability)  # absolute Goldman gap
        market_delta = abs(gyan_probability - market_probability)  # absolute market gap
        rows.append(  # comparison row
            {
                "team": team,
                "gyan_p_champion": gyan_probability,
                "goldman_p_champion": goldman_probability,
                "market_p_champion": market_probability,
                "abs_delta_vs_goldman": goldman_delta,
                "abs_delta_vs_market": market_delta,
                "diverges_from_both": goldman_delta > T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD and market_delta > T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD,
            }
        )
    tripped = [row for row in rows if row["diverges_from_both"]]  # rows tripping threshold
    return {  # T-G4 payload
        "threshold": T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD,
        "comparisons": rows,
        "tripped": bool(tripped),
        "tripped_teams": [row["team"] for row in tripped],
        "action": "investigate_driver_before_shipping" if tripped else "none",
    }


def _market_source_divergence(
    proxy_market: pd.DataFrame | None,
    live_market: pd.DataFrame,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare the refreshed live market vector with the pre-refresh proxy vector."""
    if proxy_market is None or proxy_market.empty:  # no prior proxy to compare
        empty = pd.DataFrame(columns=["team", "proxy_p_champion", "live_p_champion", "delta_live_minus_proxy", "abs_delta"])
        empty.to_csv(output_path, index=False)  # still write an audit artifact
        return empty, {  # explicit unavailable status
            "available": False,
            "reason": "no_pre_refresh_market_proxy",
            "threshold": T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD,
            "max_abs_delta": 0.0,
            "limitation_note_required": False,
        }
    proxy = proxy_market.drop_duplicates("team").set_index("team")["p_champion"].astype(float)  # baseline
    live = live_market.drop_duplicates("team").set_index("team")["p_champion"].astype(float)  # refreshed
    teams = sorted(set(proxy.index).union(set(live.index)))  # full comparison universe
    rows = []  # collect per-team differences
    for team in teams:  # compare aligned probabilities
        proxy_probability = float(proxy.get(team, 0.0))  # prior/proxy p
        live_probability = float(live.get(team, 0.0))  # refreshed p
        delta = live_probability - proxy_probability  # signed movement
        rows.append(  # audit row
            {
                "team": team,
                "proxy_p_champion": proxy_probability,
                "live_p_champion": live_probability,
                "delta_live_minus_proxy": delta,
                "abs_delta": abs(delta),
            }
        )
    divergence = pd.DataFrame(rows).sort_values("abs_delta", ascending=False).reset_index(drop=True)  # largest moves first
    divergence.to_csv(output_path, index=False)  # stable latest audit table
    max_abs_delta = float(divergence["abs_delta"].max()) if len(divergence) else 0.0  # headline movement
    return divergence, {  # run-record summary
        "available": True,
        "threshold": T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD,
        "max_abs_delta": max_abs_delta,
        "limitation_note_required": bool(max_abs_delta > T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD),
        "top_moves": divergence.head(6).to_dict(orient="records"),
    }


def _market_stage_engine_shape_audit(
    market_board: pd.DataFrame,
    engine_board: pd.DataFrame,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit whether Market non-champion stages are engine-path scaled."""
    stage_columns = [column for column in STAGE_PROB_COLUMNS if column != "p_champion"]  # non-terminal stages
    merged = market_board.merge(engine_board, on="team", suffixes=("_market", "_engine"))  # align boards
    rows: list[dict[str, object]] = []  # collect one row per team
    for row in merged.itertuples(index=False):  # reconstruct board_from_champion_vector
        engine_champion = max(float(row.p_champion_engine), 1e-9)  # avoid divide-by-zero
        market_champion = float(row.p_champion_market)  # exact market champion probability
        scale = market_champion / engine_champion  # champion replacement scale
        previous = 1.0  # monotone clamp state
        record: dict[str, object] = {  # row audit base
            "team": row.team,
            "engine_p_champion": engine_champion,
            "market_p_champion": market_champion,
            "champion_scale": scale,
        }
        for column in stage_columns:  # each non-champion stage
            engine_value = float(getattr(row, f"{column}_engine"))  # engine path probability
            market_value = float(getattr(row, f"{column}_market"))  # reported market stage probability
            raw_value = min(1.0, max(engine_value * scale, market_champion))  # scaled engine path
            reconstructed = min(previous, max(raw_value, market_champion))  # monotone stage clamp
            record[f"{column}_engine"] = engine_value  # engine source
            record[f"{column}_market"] = market_value  # market report
            record[f"{column}_reconstructed"] = reconstructed  # reconstructed report
            record[f"{column}_abs_reconstruction_error"] = abs(market_value - reconstructed)  # error
            record[f"{column}_ratio_market_to_engine"] = market_value / engine_value if engine_value else np.nan  # ratio
            previous = reconstructed  # next stage cannot exceed current stage
        rows.append(record)  # collect team row
    audit = pd.DataFrame(rows).sort_values("market_p_champion", ascending=False).reset_index(drop=True)  # ranked
    audit.to_csv(output_path, index=False)  # stable latest audit
    max_errors = {column: float(audit[f"{column}_abs_reconstruction_error"].max()) for column in stage_columns}  # error summary
    market_stage_sums = {column: float(market_board[column].sum()) for column in STAGE_PROB_COLUMNS}  # reported sums
    engine_stage_sums = {column: float(engine_board[column].sum()) for column in STAGE_PROB_COLUMNS}  # tournament slot sums
    return audit, {  # run-record payload
        "non_champion_stages_are_scaled_engine_shape": bool(max(max_errors.values()) <= 1e-10),
        "max_abs_reconstruction_error": max_errors,
        "market_stage_sums": market_stage_sums,
        "engine_stage_sums": engine_stage_sums,
        "audit_table": str(output_path),
    }


def _write_expert_correlation_diagnostics(board_map: dict[str, pd.DataFrame], tag: str) -> tuple[pd.DataFrame, Path, dict[str, object]]:
    """Write pairwise expert-board correlation diagnostics across tournament probabilities."""
    rows: list[dict[str, object]] = []  # one row per expert pair and probability column
    expert_names = list(board_map)  # preserve board-map order
    for left_index, expert_a in enumerate(expert_names[:-1]):  # all unique expert pairs
        for expert_b in expert_names[left_index + 1 :]:
            left = board_map[expert_a].set_index("team")  # team-indexed board A
            right = board_map[expert_b].set_index("team")  # team-indexed board B
            common_teams = sorted(set(left.index).intersection(right.index))  # shared field
            top_a = set(left["p_champion"].sort_values(ascending=False).head(12).index)  # champion top tier A
            top_b = set(right["p_champion"].sort_values(ascending=False).head(12).index)  # champion top tier B
            top_union = top_a.union(top_b)  # Jaccard denominator
            for column in STAGE_PROB_COLUMNS:
                a_values = left.loc[common_teams, column].astype(float)  # aligned probabilities A
                b_values = right.loc[common_teams, column].astype(float)  # aligned probabilities B
                rows.append(
                    {
                        "expert_a": expert_a,
                        "expert_b": expert_b,
                        "probability_column": column,
                        "n_teams": int(len(common_teams)),
                        "pearson": float(a_values.corr(b_values, method="pearson")),
                        "spearman": float(a_values.corr(b_values, method="spearman")),
                        "champion_top12_overlap": int(len(top_a.intersection(top_b))) if column == "p_champion" else np.nan,
                        "champion_top12_jaccard": float(len(top_a.intersection(top_b)) / len(top_union)) if column == "p_champion" and top_union else np.nan,
                    }
                )
    diagnostics = pd.DataFrame(rows)  # tabular audit
    timestamped_path = OUTPUTS_TABLES / f"expert_board_correlation_diagnostics_{tag}.csv"  # run-specific artifact
    diagnostics.to_csv(timestamped_path, index=False)  # timestamped output
    diagnostics.to_csv(EXPERT_CORRELATION_DIAGNOSTICS_FILE, index=False)  # stable latest alias
    socio_market = diagnostics[
        (diagnostics["expert_a"] == "socioeconomic")
        & (diagnostics["expert_b"] == "market")
        & (diagnostics["probability_column"] == "p_champion")
    ].iloc[0]  # requested diversity diagnostic
    metrics = {
        "table": str(EXPERT_CORRELATION_DIAGNOSTICS_FILE),
        "socioeconomic_market_champion_pearson": float(socio_market["pearson"]),
        "socioeconomic_market_champion_spearman": float(socio_market["spearman"]),
        "socioeconomic_market_champion_top12_overlap": int(socio_market["champion_top12_overlap"]),
        "socioeconomic_market_champion_top12_jaccard": float(socio_market["champion_top12_jaccard"]),
        "socioeconomic_market_high_correlation_flag": bool(
            abs(float(socio_market["pearson"])) >= 0.80 or abs(float(socio_market["spearman"])) >= 0.80
        ),
    }
    return diagnostics, timestamped_path, metrics


def _write_t_g4_diagnostics(
    t_g4: dict[str, object],
    board_map: dict[str, pd.DataFrame],
    groups: pd.DataFrame,
    elo: pd.DataFrame,
    model: DixonColesModel,
    tag: str,
) -> tuple[pd.DataFrame, Path, Path]:
    """Write top-team divergence diagnostics for the T-G4 gate."""
    rows: list[dict[str, object]] = []  # collect one row per T-G4 comparison team
    elo_lookup = elo.set_index("team")["elo_rating"].to_dict() if "elo_rating" in elo.columns else {}  # Elo
    group_lookup = groups.set_index("team")["group"].to_dict()  # team -> group
    for comparison in t_g4["comparisons"]:  # top-team comparison rows
        team = str(comparison["team"])  # team label
        group = group_lookup.get(team)  # tournament group
        opponents = sorted(groups[(groups["group"] == group) & (groups["team"] != team)]["team"].tolist()) if group else []  # group opponents
        row = {  # core gate values and model drivers
            **comparison,
            "elo_rating": float(elo_lookup.get(team, np.nan)),
            "group": group,
            "group_opponents": ", ".join(opponents),
            "engine_selected": model.selected_engine,
            "score_distribution": model.score_distribution,
            "score_dispersion": model.score_dispersion,
            "attack_parameter": float(model.attack.get(team, np.nan)),
            "defense_parameter": float(model.defense.get(team, np.nan)),
        }
        for expert, board in board_map.items():  # add each expert champion view
            expert_lookup = board.set_index("team")["p_champion"]  # champion vector
            row[f"{expert}_p_champion"] = float(expert_lookup.get(team, np.nan))  # expert p
        rows.append(row)  # collect
    diagnostics = pd.DataFrame(rows)  # diagnostic table
    diagnostics["diagnosis"] = np.where(  # concise gate diagnosis
        diagnostics["diverges_from_both"],
        "confirmed_goal_market_benchmark_divergence",
        "within_t_g4_market_or_benchmark_tolerance",
    )
    timestamped_path = OUTPUTS_TABLES / f"t_g4_divergence_diagnostics_{tag}.csv"  # timestamped audit
    latest_path = OUTPUTS_TABLES / "t_g4_divergence_diagnostics_latest.csv"  # stable audit
    diagnostics.to_csv(timestamped_path, index=False)  # write timestamped table
    diagnostics.to_csv(latest_path, index=False)  # write latest table
    return diagnostics, timestamped_path, latest_path  # diagnostics and paths


def _resolve_t_g4_action(t_g4: dict[str, object], diagnostics: pd.DataFrame) -> str:
    """Return the post-investigation T-G4 action."""
    if not t_g4["tripped"]:  # no threshold breach
        return "none"  # nothing to do
    tripped = diagnostics[diagnostics["diverges_from_both"]].copy()  # breached teams
    if tripped.empty:  # defensive guard
        return "none"  # no breached rows
    goal_is_driver = bool((tripped["goal_p_champion"] < tripped["market_p_champion"]).all())  # market above goal
    if goal_is_driver:  # live-market disagreement is documented, not unresolved
        return "confirmed_goal_market_benchmark_divergence_documented"  # investigated action
    return "investigate_driver_before_shipping"  # unexpected pattern still needs review


def _pool_boards(board_map: dict[str, pd.DataFrame], expert_order: tuple[str, ...], weights: np.ndarray, pool_name: str) -> pd.DataFrame:
    """Pool expert tournament boards into one GYAN stage-probability board."""
    teams = board_map[expert_order[0]]["team"].tolist()  # shared order from goal board
    rows: list[dict[str, object]] = []  # collect pooled rows
    for team in teams:  # one team at a time
        expert_rows = [board_map[expert].set_index("team").loc[team, list(STAGE_PROB_COLUMNS)].to_numpy(dtype=float) for expert in expert_order]  # matrices
        matrix = np.vstack(expert_rows)  # expert x stage probabilities
        if pool_name == "log":  # log pool each stage independently
            pooled = np.exp(np.tensordot(weights, np.log(np.clip(matrix, 1e-15, 1.0)), axes=([0], [0])))  # geometric mean
        else:  # linear pool each stage independently
            pooled = np.tensordot(weights, matrix, axes=([0], [0]))  # arithmetic mean
        rows.append({"team": team, **{column: float(value) for column, value in zip(STAGE_PROB_COLUMNS, pooled, strict=True)}})  # row
    forecast = pd.DataFrame(rows)  # pooled board
    champion = forecast["p_champion"].to_numpy(dtype=float)  # champion probabilities
    forecast["p_champion"] = champion / champion.sum()  # force champion sum to one
    for row_index in forecast.index:  # enforce monotone stage probabilities after champion normalisation
        previous = 1.0  # R32 cannot exceed one
        champion_probability = float(forecast.loc[row_index, "p_champion"])  # terminal probability
        for column in STAGE_PROB_COLUMNS[:-1]:  # pre-champion stages
            value = min(previous, max(float(forecast.loc[row_index, column]), champion_probability))  # monotone clamp
            forecast.loc[row_index, column] = value  # write cleaned value
            previous = value  # next stage cannot exceed previous
    return forecast.sort_values("p_champion", ascending=False).reset_index(drop=True)  # ranked board


def _write_forecast(forecast: pd.DataFrame, tag: str) -> tuple[Path, Path]:
    """Write timestamped and latest GYAN forecast tables."""
    parquet_path = OUTPUTS_TABLES / f"gyan_forecast_2026_{tag}.parquet"  # timestamped parquet
    csv_path = OUTPUTS_TABLES / f"gyan_forecast_2026_{tag}.csv"  # timestamped csv
    forecast.to_parquet(parquet_path, index=False)  # canonical typed output
    forecast.to_csv(csv_path, index=False)  # human-readable output
    forecast.to_csv(GYAN_FORECAST_LATEST, index=False)  # latest alias
    return parquet_path, csv_path  # timestamped paths


def _write_weights(
    weights: np.ndarray,
    pool_result: dict[str, object],
    selected: dict[str, object],
    tag: str,
    fit_expert_order: tuple[str, ...],
    shipped_expert_order: tuple[str, ...],
) -> tuple[Path, Path]:
    """Write ensemble weight diagnostics."""
    rows: list[dict[str, object]] = []  # collect weight rows
    for pool_name, option in pool_result["options"].items():  # each pool candidate
        for label, vector in {"equal": option["equal_weights"], "optimised": option["optimised_weights"], "ship_candidate": option["ship_weights"]}.items():  # equal/opt/selected
            for expert, weight in zip(fit_expert_order, vector, strict=True):  # one weight per fitted expert
                rows.append({"pool": pool_name, "weight_set": label, "expert": expert, "weight": float(weight)})  # row
    for expert, weight in zip(shipped_expert_order, weights, strict=True):  # shipped weights
        rows.append({"pool": selected["chosen_pool"], "weight_set": "shipped", "expert": expert, "weight": float(weight), "reason": selected["historical_weight_fit"]["ship_reason"]})  # shipped row
    frame = pd.DataFrame(rows)  # weight table
    parquet_path = OUTPUTS_TABLES / f"ensemble_weights_{tag}.parquet"  # timestamped parquet
    csv_path = OUTPUTS_TABLES / f"ensemble_weights_{tag}.csv"  # timestamped csv
    frame.to_parquet(parquet_path, index=False)  # canonical typed output
    frame.to_csv(csv_path, index=False)  # human-readable output
    frame.to_csv(ENSEMBLE_WEIGHTS_FILE, index=False)  # latest alias
    return parquet_path, csv_path  # timestamped paths


def _write_yield_delta(named_board: pd.DataFrame, nominal_board: pd.DataFrame) -> pd.DataFrame:
    """Write teams whose forecast moves most from nominal to named Yield."""
    named = named_board.set_index("team")  # named board by team
    nominal = nominal_board.set_index("team")  # nominal board by team
    rows = []  # collect delta rows
    for team in named.index:  # one team at a time
        rows.append(  # delta record
            {
                "team": team,
                "p_champion_nominal": float(nominal.loc[team, "p_champion"]),
                "p_champion_named": float(named.loc[team, "p_champion"]),
                "delta_p_champion_named_minus_nominal": float(named.loc[team, "p_champion"] - nominal.loc[team, "p_champion"]),
                "delta_abs": float(abs(named.loc[team, "p_champion"] - nominal.loc[team, "p_champion"])),
            }
        )
    delta = pd.DataFrame(rows).sort_values("delta_abs", ascending=False).reset_index(drop=True)  # most moved first
    delta.to_csv(YIELD_DELTA_TABLE_FILE, index=False)  # write paper case-study table
    return delta  # return for summary metrics


def _write_reliability_diagram(
    pool_result: dict[str, object],
    split: dict[str, object],
    selected: dict[str, object],
) -> Path:
    """Write reliability diagram comparing linear and log pools."""
    validation_outcomes = split["validation_outcomes"]  # validation outcomes
    validation_forecasts = split["validation_forecasts"]  # validation forecasts
    plt.figure(figsize=(6.2, 5.2))  # compact figure
    for pool_name, option in pool_result["options"].items():  # candidate pools
        pooled = pool_many(validation_forecasts, option["ship_weights"], pool_fn=option["pool_fn"])  # selected candidate forecasts
        predicted = pooled.reshape(-1)  # all class-event probabilities
        observed = validation_outcomes.reshape(-1)  # all class-event indicators
        bins = np.linspace(0.0, 1.0, 8)  # reliability bins
        bin_centres: list[float] = []  # x values
        observed_rates: list[float] = []  # y values
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):  # each bin
            mask = (predicted >= lower) & (predicted < upper if upper < 1.0 else predicted <= upper)  # bin mask
            if mask.sum() == 0:  # skip empty bins
                continue  # no point
            bin_centres.append(float(predicted[mask].mean()))  # mean predicted probability
            observed_rates.append(float(observed[mask].mean()))  # realised frequency
        plt.plot(bin_centres, observed_rates, marker="o", linewidth=1.6, label=pool_name)  # reliability line
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0, label="perfect")  # reference line
    plt.xlabel("Mean predicted probability")  # x label
    plt.ylabel("Observed frequency")  # y label
    plt.title("Stage 3 Validation Reliability")  # title
    plt.legend(fontsize=8)  # legend
    plt.tight_layout()  # avoid clipping
    plt.savefig(RELIABILITY_DIAGRAM_PNG, dpi=300)  # review PNG
    plt.savefig(RELIABILITY_DIAGRAM_PDF)  # vector PDF
    plt.close()  # release figure memory
    caption_path = OUTPUTS_FIGURES / "stage3_reliability_diagram.txt"  # caption sidecar
    caption_path.write_text(  # caption
        f"Reliability diagram on the Stage 3 validation split; shipped pool is {selected['chosen_pool']}. Final paper input freeze timestamp: {FINAL_FREEZE_TIMESTAMP_UTC}.\n",
        encoding="utf-8",
    )
    return caption_path  # caption artifact


def _format_top_table(frame: pd.DataFrame, columns: list[str], n_rows: int = 10) -> str:
    """Return a compact markdown table for summary files."""
    display = frame.head(n_rows)[columns].copy()  # subset rows and columns
    for column in columns:  # format probability columns
        if column.startswith("p_") or column.startswith("delta"):  # numeric probability-like column
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")  # four decimals
    header = "| " + " | ".join(columns) + " |"  # markdown header
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"  # markdown separator
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in display.iterrows()]  # body
    return "\n".join([header, separator, *rows])  # complete markdown table


def _write_summary(
    forecast: pd.DataFrame,
    board_validations: dict[str, dict[str, object]],
    pool_result: dict[str, object],
    selected: dict[str, object],
    weights: np.ndarray,
    market_metadata: dict[str, object],
    yield_delta: pd.DataFrame,
    yield_named_scores: dict[str, float],
    yield_nominal_scores: dict[str, float],
    t_g4: dict[str, object],
    market_source_divergence: dict[str, object],
    market_stage_engine_shape: dict[str, object],
    expert_correlation_metrics: dict[str, object],
    expert_correlation_path: Path,
    t_g4_diagnostics_path: Path,
    validation_rows: int,
    output_paths: dict[str, Path],
    expert_order: tuple[str, ...],
    match_validation_expert_order: tuple[str, ...],
) -> None:
    """Write the Stage 3 summary markdown."""
    weight_summary = ", ".join(f"{expert}={weight:.3f}" for expert, weight in zip(expert_order, weights, strict=True))  # shipped four-expert weights
    chosen_option = selected["historical_weight_fit"]["options"][selected["chosen_pool"]]  # selected pool diagnostics
    unconstrained_weight_summary = ", ".join(  # reference fit without positivity floor
        f"{expert}={weight:.3f}" for expert, weight in zip(expert_order, chosen_option["unconstrained_weights"], strict=True)
    )
    match_weight_summary = ", ".join(  # diagnostic W/D/L validation weights
        f"{expert}={weight:.3f}" for expert, weight in zip(match_validation_expert_order, selected["match_validation_weights"], strict=True)
    )
    top_table = _format_top_table(forecast, ["team", "p_champion", "p_reach_final", "p_reach_SF"], n_rows=10)  # top forecast
    delta_table = _format_top_table(yield_delta, ["team", "delta_p_champion_named_minus_nominal", "p_champion_named", "p_champion_nominal"], n_rows=10)  # yield deltas
    summary = f"""# Stage 3 Summary

## Expert Alignment
- Expert order: `{', '.join(expert_order)}`.
- Match-validation expert order: `{', '.join(match_validation_expert_order)}`.
- Board validations: `{board_validations}`.
- Market snapshot UTC: `{market_metadata['snapshot_time_utc']}`; source note: `{market_metadata['source_note']}`.

## Pool Choice
- Validation rows: {validation_rows}.
- Chosen pool: `{selected['chosen_pool']}`.
- Shipped weights: {weight_summary}.
- Shipped-weight source: `{selected['historical_weight_fit']['ship_reason']}` after leave-one-tournament-out guard over `{selected['historical_weight_fit']['n_matches']}` no-leakage World Cup matches.
- Historical match-objective profile: `{selected['historical_weight_fit']['match_objective_profile']}`.
- In-sample fitted RPS: optimised `{chosen_option['historical_rps_optimised']}` vs equal `{chosen_option['historical_rps_equal']}`.
- Leave-one-tournament-out RPS: optimised `{chosen_option['leave_one_tournament_out']['optimised_oos_rps']}` vs equal `{chosen_option['leave_one_tournament_out']['equal_oos_rps']}`; selected `{chosen_option['leave_one_tournament_out']['selected_weight_set']}`.
- Unconstrained selected-pool reference weights: {unconstrained_weight_summary}; in-sample RPS `{chosen_option['historical_rps_unconstrained']}`.
- Floor policy on optimised candidate: `{selected['historical_weight_fit']['floor_policy']}`; binding experts `{selected['historical_weight_fit']['floor_binding_experts']}`; RPS cost versus unconstrained `{selected['historical_weight_fit']['floor_rps_cost_vs_unconstrained']}`.
- Pool selection reason: `{selected['historical_weight_fit']['selection_reason']}`.
- Historical selected scores: `{selected['selected_scores']}`; equal-weight scores: `{selected['historical_weight_fit']['equal_scores']}`.
- Match-validation diagnostic weights: {match_weight_summary}; scores: `{selected['match_validation_scores']}`.
- Market validation status: `{selected['market_scores']}`.
- Market expert construction transfer: `{selected['historical_weight_fit']['market_expert_construction']}`.
- Live-vs-proxy market divergence: `{market_source_divergence}`.
- Market non-champion stage source audit: `{market_stage_engine_shape}`.
- Expert board correlation diagnostic: `{expert_correlation_metrics}`.
- Expert board correlation table: `{repo_path_str(expert_correlation_path)}`.
- Threshold actions: `{selected['threshold_actions']}`.
- Yield-named validation scores: `{yield_named_scores}`.
- Yield-nominal validation scores: `{yield_nominal_scores}`.
- T-G4 top-team divergence check: `{t_g4}`.
- T-G4 diagnostics: `{repo_path_str(t_g4_diagnostics_path)}`.

## GYAN Champion Top 10

{top_table}

## Yield Named-vs-Nominal Movers

{delta_table}

## Outputs
- GYAN forecast: `{repo_path_str(output_paths['forecast_parquet'])}` and `{repo_path_str(output_paths['forecast_csv'])}`; latest CSV: `{repo_path_str(GYAN_FORECAST_LATEST)}`.
- Expert boards: `{repo_path_str(output_paths['expert_parquet'])}` and `{repo_path_str(output_paths['expert_csv'])}`; latest CSV: `{repo_path_str(EXPERT_BOARDS_2026_FILE)}`.
- Weights: `{repo_path_str(output_paths['weights_parquet'])}` and `{repo_path_str(output_paths['weights_csv'])}`; latest CSV: `{repo_path_str(ENSEMBLE_WEIGHTS_FILE)}`.
- Yield delta table: `{repo_path_str(YIELD_DELTA_TABLE_FILE)}`.
- Market vector: `{repo_path_str(MARKET_IMPLIED_LIVE_FILE)}`.
- Reliability diagram: `{repo_path_str(RELIABILITY_DIAGRAM_PNG)}` and `{repo_path_str(RELIABILITY_DIAGRAM_PDF)}`.

## Human Gate 3
Human confirmation remains required before Stage 4: expert validity, market championship-source audit, shipped weights, calibration, and threshold actions.
"""
    STAGE3_SUMMARY_FILE.write_text(summary, encoding="utf-8")  # write summary file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)  # CLI parser
    parser.add_argument("--n-sims", type=int, default=STAGE3_DEFAULT_N_SIMULATIONS, help="Simulations per feature expert.")
    parser.add_argument("--n-workers", type=int, default=STAGE3_DEFAULT_N_WORKERS, help="Parallel workers.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Master random seed.")
    return parser.parse_args()  # parsed arguments


def main() -> None:
    """Run Stage 3 ensemble construction."""
    args = parse_args()  # CLI arguments
    create_directories()  # ensure project directories exist
    tag = _timestamp()  # shared output timestamp
    logger, log_path = get_run_logger("s3_build_ensemble", stage="stage3", step="build_ensemble")  # run logger
    with RunRecord(  # machine-readable run record
        stage="stage3",
        step="build_ensemble",
        script_path=__file__,
        global_seed=args.seed,
        n_workers=args.n_workers,
        n_simulations=args.n_sims,
        logger=logger,
    ) as record:
        for path in (  # hashed inputs
            DIXON_COLES_PARAMS_LATEST_FILE,
            TEAM_ADVANCEMENT_ENGINEONLY_LATEST,
            SQUAD_FEATURES_2026_FILE,
            SOCIOECONOMIC_FEATURES_FILE,
            MATCHES_WITH_ELO_FILE,
            ELO_CURRENT_RATINGS_FILE,
            SHOOTOUTS_RAW_FILE,
            WORLD_BANK_GDP_PPP_FILE,
            WORLD_BANK_POPULATION_FILE,
            BACKTEST_MARKET_OUTRIGHTS_FILE,
            GROUPS_2026_FILE,
            SCHEDULE_2026_FILE,
            BRACKET_PAIRINGS_2026_FILE,
        ):
            record.add_input(path)  # hash each input artifact
        record.add_output(log_path)  # human-readable log path
        start = time.perf_counter()  # runtime timer
        model = DixonColesModel.load(DIXON_COLES_PARAMS_LATEST_FILE)  # Stage 1 fitted goal model
        engine_board = pd.read_csv(TEAM_ADVANCEMENT_ENGINEONLY_LATEST)  # Stage 2 goal tournament board
        teams = engine_board["team"].tolist()  # shared team order
        squad = pd.read_parquet(SQUAD_FEATURES_2026_FILE)  # squad features
        socioeconomic = pd.read_parquet(SOCIOECONOMIC_FEATURES_FILE)  # socioeconomic features
        matches = pd.read_parquet(MATCHES_WITH_ELO_FILE)  # match-level validation data
        elo = pd.read_csv(ELO_CURRENT_RATINGS_FILE)  # current Elo ratings for penalties/upsets
        structure = load_structure(GROUPS_2026_FILE, SCHEDULE_2026_FILE, BRACKET_PAIRINGS_2026_FILE)  # Stage 2 structure
        yield_named_feature = build_yield_feature_series(squad, named=True)  # named squad signal
        yield_nominal_feature = build_yield_feature_series(squad, named=False)  # nominal value signal
        socioeconomic_feature = build_socioeconomic_feature_series(socioeconomic, teams)  # socio signal
        yield_named_ratings = ratings_from_feature(yield_named_feature, teams)  # named ratings
        yield_nominal_ratings = ratings_from_feature(yield_nominal_feature, teams)  # nominal ratings
        socioeconomic_ratings = ratings_from_feature(socioeconomic_feature, teams)  # socio ratings
        yield_named_model = strength_model_from_ratings(yield_named_ratings, model.home_field)  # named model
        yield_nominal_model = strength_model_from_ratings(yield_nominal_ratings, model.home_field)  # nominal model
        socioeconomic_model = strength_model_from_ratings(socioeconomic_ratings, model.home_field)  # socio model
        logger.info("Simulating Yield-named, Yield-nominal, and Socioeconomic expert boards")  # log start
        yield_named_board = _run_strength_board(yield_named_model, structure, elo, args.n_sims, args.n_workers, args.seed + 301)  # Y named
        yield_nominal_board = _run_strength_board(yield_nominal_model, structure, elo, args.n_sims, args.n_workers, args.seed + 302)  # Y nominal
        socioeconomic_board = _run_strength_board(socioeconomic_model, structure, elo, args.n_sims, args.n_workers, args.seed + 303)  # socio
        proxy_market_table = pd.read_parquet(MARKET_IMPLIED_LIVE_FILE) if MARKET_IMPLIED_LIVE_FILE.exists() else None  # pre-refresh proxy
        market_source_paths = refresh_live_market_source_cache()  # pull current PM/Kalshi/bookmaker raw files
        for market_source_path in market_source_paths.values():  # hash raw market source cache
            record.add_input(market_source_path)  # raw live market input
        market_table, market_metadata = build_live_market_snapshot(  # market champion vector
            teams,
            MARKET_IMPLIED_LIVE_FILE,
            anchor=engine_board.set_index("team")["p_champion"],
            source_paths=market_source_paths,
        )
        market_divergence, market_divergence_metrics = _market_source_divergence(  # live-vs-proxy source audit
            proxy_market_table,
            market_table,
            MARKET_SOURCE_DIVERGENCE_FILE,
        )
        market_champion = market_table.set_index("team")["p_champion"]  # team-indexed market vector
        market_board = board_from_champion_vector(engine_board, market_champion)  # full market stage board
        market_stage_audit, market_stage_metrics = _market_stage_engine_shape_audit(  # stage-source audit
            market_board,
            engine_board,
            MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE,
        )
        board_map = {  # all tournament boards
            "goal": engine_board,
            "yield_named": yield_named_board,
            "yield_nominal": yield_nominal_board,
            "socioeconomic": socioeconomic_board,
            "market": market_board,
        }
        expert_correlation, expert_correlation_path, expert_correlation_metrics = _write_expert_correlation_diagnostics(board_map, tag)  # diversity audit
        board_validations = {expert: validate_expert_board(board) for expert, board in board_map.items()}  # checks
        experts = [  # match-level expert objects for pool fitting
            GoalExpert("goal", engine_board, model),
            StrengthExpert("yield_named", yield_named_board, yield_named_ratings, yield_named_model),
            StrengthExpert("socioeconomic", socioeconomic_board, socioeconomic_ratings, socioeconomic_model),
        ]
        yield_nominal_expert = StrengthExpert("yield_nominal", yield_nominal_board, yield_nominal_ratings, yield_nominal_model)  # nominal expert
        validation_matches = _build_validation_frame(matches, teams)  # held-out matches
        expert_forecasts, outcomes = _expert_forecast_tensor(experts, validation_matches)  # validation tensor
        yield_named_scores = _single_expert_scores(experts[1], validation_matches, outcomes)  # Y named scores
        yield_nominal_scores = _single_expert_scores(yield_nominal_expert, validation_matches, outcomes)  # Y nominal scores
        split = _split_train_validation(expert_forecasts, outcomes, validation_matches)  # weight train/validation split
        pool_result = _evaluate_pool_options(split)  # optimise match-level diagnostic linear/log pools
        match_selected = _apply_thresholds(pool_result, split, MATCH_VALIDATION_EXPERT_ORDER, market_board)  # apply W/D/L guards
        historical_pool = _fit_historical_four_expert_pool(matches, socioeconomic, EXPERT_ORDER)  # fit shipped four-expert pool
        shipped_weights = np.asarray(historical_pool["weights"], dtype=float)  # final four-expert weights
        selected = {  # shipped decision payload
            "chosen_pool": historical_pool["chosen_pool"],
            "pool_fn": historical_pool["pool_fn"],
            "weights": shipped_weights,
            "threshold_actions": {
                "T-G1": "market_included_with_historical_four_expert_rps_weight",
                "T-G2": "socioeconomic_retained_by_historical_four_expert_rps_weight",
                "T-G4": "none",
            },
            "market_scores": {
                "available": True,
                "reason": "market included through D15 historical W/D/L calibration and live 2026 outright board",
                "champion_sum": float(market_board["p_champion"].sum()),
            },
            "selected_scores": historical_pool["selected_scores"],
            "historical_weight_fit": historical_pool,
            "match_validation_scores": match_selected["selected_scores"],
            "match_validation_weights": match_selected["weights"],
            "match_validation_threshold_actions": match_selected["threshold_actions"],
        }
        shipped_board_map = {expert: board_map[expert] for expert in EXPERT_ORDER}  # shipped board map with all four experts
        forecast = _pool_boards(shipped_board_map, EXPERT_ORDER, shipped_weights, selected["chosen_pool"])  # GYAN board
        t_g4 = _evaluate_t_g4(forecast, market_board)  # top-team divergence check
        t_g4_diagnostics, t_g4_diag_csv, t_g4_diag_latest = _write_t_g4_diagnostics(  # driver audit
            t_g4,
            board_map,
            structure["groups"],
            elo,
            model,
            tag,
        )
        t_g4["action"] = _resolve_t_g4_action(t_g4, t_g4_diagnostics)  # investigated action
        selected["threshold_actions"]["T-G4"] = t_g4["action"]  # record concrete T-G4 action
        expert_parquet, expert_csv = _write_expert_board(board_map, tag)  # expert board outputs
        weights_parquet, weights_csv = _write_weights(shipped_weights, historical_pool, selected, tag, EXPERT_ORDER, EXPERT_ORDER)  # weight outputs
        forecast_parquet, forecast_csv = _write_forecast(forecast, tag)  # GYAN outputs
        yield_delta = _write_yield_delta(yield_named_board, yield_nominal_board)  # named-vs-nominal table
        reliability_caption = _write_reliability_diagram(pool_result, split, selected)  # calibration figure
        runtime_seconds = time.perf_counter() - start  # runtime metric
        output_paths = {  # summary path map
            "expert_parquet": expert_parquet,
            "expert_csv": expert_csv,
            "weights_parquet": weights_parquet,
            "weights_csv": weights_csv,
            "forecast_parquet": forecast_parquet,
            "forecast_csv": forecast_csv,
        }
        _write_summary(  # markdown gate summary
            forecast,
            board_validations,
            pool_result,
            selected,
            shipped_weights,
            market_metadata,
            yield_delta,
            yield_named_scores,
            yield_nominal_scores,
            t_g4,
            market_divergence_metrics,
            market_stage_metrics,
            expert_correlation_metrics,
            expert_correlation_path,
            t_g4_diag_latest,
            int(split["validation_rows"]),
            output_paths,
            EXPERT_ORDER,
            MATCH_VALIDATION_EXPERT_ORDER,
        )
        record.add_params(  # key parameters and source notes
            {
                "expert_order": EXPERT_ORDER,
                "match_validation_expert_order": MATCH_VALIDATION_EXPERT_ORDER,
                "n_feature_expert_simulations": args.n_sims,
                "validation_cutoff": STAGE3_VALIDATION_CUTOFF,
                "final_freeze_timestamp_utc": FINAL_FREEZE_TIMESTAMP_UTC,
                "expected_release_tag": FINAL_RELEASE_TAG,
                "no_post_freeze_inputs_statement": f"Final paper run must use no inputs after {FINAL_FREEZE_TIMESTAMP_UTC}.",
                "market_snapshot_time_utc": market_metadata["snapshot_time_utc"],
                "market_source_note": market_metadata["source_note"],
                "market_source_weights": market_metadata["source_weights"],
                "market_source_divergence_table": str(MARKET_SOURCE_DIVERGENCE_FILE),
                "market_stage_engine_shape_audit_table": str(MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE),
                "expert_correlation_diagnostics_table": str(EXPERT_CORRELATION_DIAGNOSTICS_FILE),
                "available_goal_board_simulations": N_SIMULATIONS_HIGH,
            }
        )
        record.add_metrics(  # run metrics
            {
                "runtime_seconds": runtime_seconds,
                "validation_matches_total": int(len(validation_matches)),
                "validation_train_rows": int(split["train_rows"]),
                "validation_holdout_rows": int(split["validation_rows"]),
                "chosen_pool": selected["chosen_pool"],
                "pool_selection_reason": selected["historical_weight_fit"]["selection_reason"],
                "shipped_weights": {expert: float(weight) for expert, weight in zip(EXPERT_ORDER, shipped_weights, strict=True)},
                "match_validation_weights": {expert: float(weight) for expert, weight in zip(MATCH_VALIDATION_EXPERT_ORDER, selected["match_validation_weights"], strict=True)},
                "market_weight_status": "included_in_shipped_four_expert_pool",
                "threshold_actions": selected["threshold_actions"],
                "match_validation_threshold_actions": selected["match_validation_threshold_actions"],
                "selected_scores": selected["selected_scores"],
                "historical_weight_fit": {
                    "n_matches": selected["historical_weight_fit"]["n_matches"],
                    "min_weight": selected["historical_weight_fit"]["min_weight"],
                    "floor_policy": selected["historical_weight_fit"]["floor_policy"],
                    "floor_binding_experts": selected["historical_weight_fit"]["floor_binding_experts"],
                    "floor_rps_cost_vs_unconstrained": selected["historical_weight_fit"]["floor_rps_cost_vs_unconstrained"],
                    "ship_reason": selected["historical_weight_fit"]["ship_reason"],
                    "leave_one_tournament_out": selected["historical_weight_fit"]["leave_one_tournament_out"],
                    "equal_scores": selected["historical_weight_fit"]["equal_scores"],
                    "match_objective_profile": selected["historical_weight_fit"]["match_objective_profile"],
                    "market_expert_construction": selected["historical_weight_fit"]["market_expert_construction"],
                    "leakage_audits": selected["historical_weight_fit"]["leakage_audits"],
                    "market_sources": selected["historical_weight_fit"]["market_sources"],
                },
                "historical_pool_options": {
                    pool_name: {
                        "historical_rps_equal": option["historical_rps_equal"],
                        "historical_rps_unconstrained": option["historical_rps_unconstrained"],
                        "historical_rps_optimised": option["historical_rps_optimised"],
                        "ship_reason": option["ship_reason"],
                        "leave_one_tournament_out": option["leave_one_tournament_out"],
                        "historical_calibration": option["historical_calibration"],
                        "unconstrained_weights": {expert: float(weight) for expert, weight in zip(EXPERT_ORDER, option["unconstrained_weights"], strict=True)},
                        "optimised_weights": {expert: float(weight) for expert, weight in zip(EXPERT_ORDER, option["optimised_weights"], strict=True)},
                        "ship_weights": {expert: float(weight) for expert, weight in zip(EXPERT_ORDER, option["ship_weights"], strict=True)},
                    }
                    for pool_name, option in selected["historical_weight_fit"]["options"].items()
                },
                "match_validation_scores": selected["match_validation_scores"],
                "market_scores": selected["market_scores"],
                "market_source_divergence": market_divergence_metrics,
                "market_stage_engine_shape": market_stage_metrics,
                "expert_correlation": expert_correlation_metrics,
                "yield_named_scores": yield_named_scores,
                "yield_nominal_scores": yield_nominal_scores,
                "yield_named_minus_nominal_rps": yield_named_scores["mean_rps"] - yield_nominal_scores["mean_rps"],
                "t_g4": t_g4,
                "board_validations": board_validations,
                "top10_champion": forecast.head(10)[["team", "p_champion"]].to_dict(orient="records"),
                "max_abs_yield_named_delta": float(yield_delta["delta_abs"].max()),
                "market_source_divergence_rows": int(len(market_divergence)),
                "market_stage_engine_shape_audit_rows": int(len(market_stage_audit)),
                "expert_correlation_diagnostics_rows": int(len(expert_correlation)),
            }
        )
        for path in (  # hashed outputs
            MARKET_IMPLIED_LIVE_FILE,
            expert_parquet,
            expert_csv,
            EXPERT_BOARDS_2026_FILE,
            weights_parquet,
            weights_csv,
            ENSEMBLE_WEIGHTS_FILE,
            forecast_parquet,
            forecast_csv,
            GYAN_FORECAST_LATEST,
            YIELD_DELTA_TABLE_FILE,
            MARKET_SOURCE_DIVERGENCE_FILE,
            MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE,
            expert_correlation_path,
            EXPERT_CORRELATION_DIAGNOSTICS_FILE,
            t_g4_diag_csv,
            t_g4_diag_latest,
            RELIABILITY_DIAGRAM_PNG,
            RELIABILITY_DIAGRAM_PDF,
            reliability_caption,
            STAGE3_SUMMARY_FILE,
        ):
            record.add_output_artifact(path)  # hash output artifact
        logger.info("Stage 3 runtime seconds: %.2f", runtime_seconds)  # log runtime
        logger.info("Chosen pool: %s", selected["chosen_pool"])  # log pool
        logger.info("Shipped weights: %s", record.record["metrics"]["shipped_weights"])  # log weights
        logger.info("Threshold actions: %s", selected["threshold_actions"])  # log threshold actions
        logger.info("GYAN top 10: %s", record.record["metrics"]["top10_champion"])  # log top 10


if __name__ == "__main__":  # direct execution
    main()  # run Stage 3
