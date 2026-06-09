"""Run the Stage 2 engine-only 2026 World Cup Monte-Carlo simulation."""

from __future__ import annotations  # consistent type hints

import argparse  # CLI flags for draw count and workers
import os  # thread caps and plotting backend before numerical imports
import time  # wall-clock simulation timing
from datetime import datetime, timezone  # timestamped output filenames
from pathlib import Path  # typed paths

os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid BLAS oversubscription in workers
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # one BLAS thread per joblib worker
os.environ.setdefault("MKL_NUM_THREADS", "1")  # one MKL thread per joblib worker
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # one Accelerate thread per worker on macOS
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")  # one numexpr thread per worker
os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting backend
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-gyan")  # writable matplotlib cache
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")  # writable font cache root

import matplotlib.pyplot as plt  # noqa: E402  # plotting after env setup
import numpy as np  # noqa: E402  # chunk splitting and convergence arrays
import pandas as pd  # noqa: E402  # output tables
from joblib import Parallel, delayed  # noqa: E402  # cross-process simulation map

from gyan.config import (  # noqa: E402  # central constants and artifact paths
    BRACKET_PAIRINGS_2026_FILE,
    DIXON_COLES_PARAMS_LATEST_FILE,
    ELO_CURRENT_RATINGS_FILE,
    GLOBAL_SEED,
    GROUPS_2026_FILE,
    HISTORICAL_KNOCKOUT_UPSET_RATE,
    MC_CONVERGENCE_FIGURE_PDF,
    MC_CONVERGENCE_FIGURE_PNG,
    N_SIMULATIONS,
    OUTPUTS_FIGURES,
    OUTPUTS_TABLES,
    SCHEDULE_2026_FILE,
    STAGE2_CONVERGENCE_POINTS,
    STAGE2_DEFAULT_N_WORKERS,
    STAGE2_SUMMARY_FILE,
    STAGE_PROBS_HEATMAP_PDF,
    STAGE_PROBS_HEATMAP_PNG,
    TEAM_ADVANCEMENT_ENGINEONLY_LATEST,
    UPSET_RATE_TABLE_FILE,
    create_directories,
    repo_path_str,
)
from gyan.engine.dixon_coles import DixonColesModel  # noqa: E402  # fitted Stage 1 engine
from gyan.simulation.tournament import (  # noqa: E402  # simulation kernel and aggregation
    STAGE_COLUMNS,
    aggregate_chunks,
    load_structure,
    prepare_simulation_inputs,
    run_tournaments_for_indices,
    validate_probabilities,
)
from gyan.utils.logging import RunRecord, get_run_logger  # noqa: E402  # run records/logs


def _timestamp() -> str:
    """Return a compact UTC timestamp for output filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # sortable timestamp


def _split_indices(n_sims: int, n_workers: int) -> list[list[int]]:
    """Split global simulation indices into ordered contiguous worker chunks."""
    chunks = np.array_split(np.arange(n_sims, dtype=int), n_workers)  # contiguous, ordered chunks
    return [chunk.astype(int).tolist() for chunk in chunks if len(chunk)]  # drop empty chunks


def _run_parallel(n_sims: int, n_workers: int, seed: int, prepared: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run tournaments across workers and return probabilities plus metrics."""
    teams = prepared["groups"]["team"].tolist()  # stable team order
    sim_chunks = _split_indices(n_sims, n_workers)  # deterministic global-index chunks
    if len(sim_chunks) == 1:  # avoid joblib overhead for single-worker runs
        chunks = [run_tournaments_for_indices(sim_chunks[0], seed, prepared)]  # direct run
    else:  # process-parallel run
        chunks = Parallel(n_jobs=len(sim_chunks), backend="loky")(  # one process per non-empty chunk
            delayed(run_tournaments_for_indices)(chunk, seed, prepared) for chunk in sim_chunks
        )
    return aggregate_chunks(chunks, teams, n_sims)  # probabilities and aggregate metrics


def _determinism_checks(prepared: dict[str, object], seed: int, n_workers: int, check_sims: int) -> dict[str, object]:
    """Run short exact reproducibility checks on counts and worker-count stability."""
    first, first_metrics = _run_parallel(check_sims, max(1, n_workers), seed, prepared)  # baseline
    second, second_metrics = _run_parallel(check_sims, max(1, n_workers), seed, prepared)  # repeat
    alternate_workers = 1 if n_workers != 1 else 2  # different worker count for seed-stability
    alternate, alternate_metrics = _run_parallel(check_sims, alternate_workers, seed, prepared)  # alternate chunks
    stage_cols = [f"p_reach_{stage}" if stage != "champion" else "p_champion" for stage in STAGE_COLUMNS]  # cols
    same_run = bool(first[["team", *stage_cols]].equals(second[["team", *stage_cols]]))  # exact repeat
    same_workers = bool(first[["team", *stage_cols]].equals(alternate[["team", *stage_cols]]))  # exact by index seed
    same_trace = bool(first_metrics["champion_trace"] == second_metrics["champion_trace"])  # champion sequence
    same_alt_trace = bool(first_metrics["champion_trace"] == alternate_metrics["champion_trace"])  # cross-worker sequence
    return {  # machine-readable check payload
        "check_sims": check_sims,
        "deterministic_same_worker_count": same_run and same_trace,
        "seed_stable_across_worker_counts": same_workers and same_alt_trace,
        "alternate_worker_count": alternate_workers,
    }


def _write_advancement_tables(probabilities: pd.DataFrame, tag: str) -> tuple[Path, Path]:
    """Write timestamped and latest advancement-probability outputs."""
    parquet_path = OUTPUTS_TABLES / f"team_advancement_probs_engineonly_2026_{tag}.parquet"  # required parquet
    csv_path = OUTPUTS_TABLES / f"team_advancement_probs_engineonly_2026_{tag}.csv"  # required csv
    probabilities.to_parquet(parquet_path, index=False)  # write parquet table
    probabilities.to_csv(csv_path, index=False)  # write audit csv
    probabilities.to_csv(TEAM_ADVANCEMENT_ENGINEONLY_LATEST, index=False)  # stable latest alias
    return parquet_path, csv_path  # timestamped paths


def _write_upset_table(metrics: dict[str, object]) -> pd.DataFrame:
    """Write the Stage 2 upset-rate audit table."""
    simulated = float(metrics["knockout_upset_rate"])  # simulated knockout upset rate
    delta = simulated - HISTORICAL_KNOCKOUT_UPSET_RATE  # signed difference from rough base rate
    action = "none" if simulated >= 0.20 else "review_score_dispersion"  # T-G6 trigger guard
    upset = pd.DataFrame(  # compact before/after-style audit table
        [
            {
                "metric": "historical_knockout_upset_rate_reference",
                "rate": HISTORICAL_KNOCKOUT_UPSET_RATE,
                "notes": "Rough one-third base rate from Stage 2 PRD T-G6.",
            },
            {
                "metric": "engine_only_simulated_knockout_upset_rate",
                "rate": simulated,
                "notes": f"Lower current-Elo team beat higher current-Elo team; action={action}; delta={delta:.4f}.",
            },
        ]
    )
    upset.to_csv(UPSET_RATE_TABLE_FILE, index=False)  # write audit table
    metrics["t_g6_action"] = action  # expose in run record
    metrics["upset_rate_delta_vs_reference"] = delta  # expose in run record
    return upset  # table for summary


def _write_convergence_figure(probabilities: pd.DataFrame, champion_trace: list[str], n_sims: int) -> Path:
    """Write champion-probability convergence for the final top four teams."""
    top_teams = probabilities.head(4)["team"].tolist()  # final top four by champion probability
    points = [point for point in STAGE2_CONVERGENCE_POINTS if point <= n_sims]  # available checkpoints
    if n_sims not in points:  # include final run count even if not a standard checkpoint
        points.append(n_sims)  # final point
    rows: list[dict[str, object]] = []  # plotting records
    trace = pd.Series(champion_trace)  # champion sequence
    for point in sorted(set(points)):  # checkpoint draws
        prefix = trace.iloc[:point]  # first n champions
        counts = prefix.value_counts(normalize=True)  # empirical champion probabilities
        for team in top_teams:  # one line per top team
            rows.append({"simulations": point, "team": team, "p_champion": float(counts.get(team, 0.0))})  # record
    plot = pd.DataFrame(rows)  # long plot table
    plt.figure(figsize=(8.2, 5.0))  # compact paper-friendly figure
    for team, team_plot in plot.groupby("team", sort=False):  # draw each team line
        plt.plot(team_plot["simulations"], team_plot["p_champion"], marker="o", linewidth=1.8, label=team)  # line
    plt.xlabel("Simulations")  # x label
    plt.ylabel("Champion probability")  # y label
    plt.title("Monte-Carlo Champion Probability Convergence")  # title
    plt.grid(True, alpha=0.25)  # subtle grid
    plt.legend(fontsize=8)  # top-team legend
    plt.tight_layout()  # avoid clipping
    plt.savefig(MC_CONVERGENCE_FIGURE_PNG, dpi=300)  # review PNG
    plt.savefig(MC_CONVERGENCE_FIGURE_PDF)  # vector PDF
    plt.close()  # release figure memory
    caption_path = OUTPUTS_FIGURES / "mc_convergence.txt"  # caption sidecar
    caption_path.write_text(  # write caption
        f"Champion-probability convergence for the final top four engine-only teams across {n_sims:,} simulations.\n",
        encoding="utf-8",
    )
    return caption_path  # caption artifact


def _convergence_stability(probabilities: pd.DataFrame, champion_trace: list[str], n_sims: int) -> dict[str, object]:
    """Return the top-four 50k-to-100k champion-probability movement."""
    if n_sims < 100_000:  # no 100k checkpoint available
        return {"convergence_50k_100k_available": False}  # explicit unavailable metric
    top_teams = probabilities.head(4)["team"].tolist()  # final top four
    trace = pd.Series(champion_trace)  # champion sequence
    p50 = trace.iloc[:50_000].value_counts(normalize=True)  # 50k empirical probabilities
    p100 = trace.iloc[:100_000].value_counts(normalize=True)  # 100k empirical probabilities
    deltas = {team: float(abs(p100.get(team, 0.0) - p50.get(team, 0.0))) for team in top_teams}  # abs pp move
    return {  # stability metric payload
        "convergence_50k_100k_available": True,
        "top4_champion_abs_delta_50k_100k": deltas,
        "top4_champion_max_abs_delta_50k_100k": max(deltas.values()) if deltas else 0.0,
        "top4_champion_delta_under_0_5pp": (max(deltas.values()) <= 0.005) if deltas else True,
    }


def _write_heatmap(probabilities: pd.DataFrame) -> None:
    """Write a heatmap of per-team stage probabilities."""
    cols = ["p_reach_R32", "p_reach_R16", "p_reach_QF", "p_reach_SF", "p_reach_final", "p_champion"]  # stages
    labels = ["R32", "R16", "QF", "SF", "Final", "Champion"]  # x labels
    plot = probabilities.sort_values("p_champion", ascending=True).reset_index(drop=True)  # strongest at top after invert
    height = max(8.0, len(plot) * 0.22)  # readable team labels
    plt.figure(figsize=(8.5, height))  # tall enough for 48 teams
    image = plt.imshow(plot[cols].to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)  # heatmap
    plt.yticks(np.arange(len(plot)), plot["team"], fontsize=6)  # team labels
    plt.xticks(np.arange(len(cols)), labels, fontsize=8)  # stage labels
    plt.gca().invert_yaxis()  # highest champion probability at top
    plt.colorbar(image, label="Probability")  # probability scale
    plt.title("Engine-only Stage Advancement Probabilities")  # title
    plt.tight_layout()  # avoid clipped labels
    plt.savefig(STAGE_PROBS_HEATMAP_PNG, dpi=300)  # review PNG
    plt.savefig(STAGE_PROBS_HEATMAP_PDF)  # vector PDF
    plt.close()  # release figure memory


def _write_summary(
    probabilities: pd.DataFrame,
    validation: dict[str, object],
    checks: dict[str, object],
    metrics: dict[str, object],
    n_sims: int,
    n_workers: int,
    parquet_path: Path,
    csv_path: Path,
) -> None:
    """Write the Stage 2 markdown gate summary."""
    top10 = probabilities.head(10)[["team", "p_champion", "p_reach_final", "p_reach_SF"]].copy()  # top teams
    top10["p_champion"] = top10["p_champion"].map(lambda value: f"{value:.4f}")  # format
    top10["p_reach_final"] = top10["p_reach_final"].map(lambda value: f"{value:.4f}")  # format
    top10["p_reach_SF"] = top10["p_reach_SF"].map(lambda value: f"{value:.4f}")  # format
    top10_markdown = _markdown_table(top10)  # dependency-free markdown table
    summary = f"""# Stage 2 Summary

## Structure
- Groups: 12; teams: 48; total matches: 104; group matches: 72; Round-of-32 matches: 16.
- Third-place assignment table: 495 Annex C combinations parsed into `data/processed/bracket_pairings_2026.json`.
- Venue flags: host home-field is applied only when Canada, Mexico, or United States is listed first at a home-country venue.
- Source note: the official FIFA DigitalHub match-schedule PDF is required and checked by `s2_build_structure`; Wikipedia remains the parseable source for dates/venues and Annex C.

## Simulation Checks
- Simulations: {n_sims:,}; workers: {n_workers}; draws/sec: {metrics['draws_per_second']:.1f}.
- Deterministic same-worker check ({checks['check_sims']} sims): {checks['deterministic_same_worker_count']}.
- Seed-stable across worker counts ({n_workers} vs {checks['alternate_worker_count']} workers): {checks['seed_stable_across_worker_counts']}.
- Probabilities in [0, 1]: {validation['probabilities_in_range']}; monotone by stage: {validation['probabilities_monotone']}.
- Champion probability sum: {validation['champion_probability_sum']:.12f}.
- Knockout upset rate: {metrics['knockout_upset_rate']:.4f}; T-G6 action: {metrics['t_g6_action']}.
- Group-stage draw rate: {metrics['group_draw_rate']:.4f}; knockout to extra time: {metrics['knockout_to_extra_time_rate']:.4f}; knockout to penalties: {metrics['knockout_to_penalties_rate']:.4f}.
- Top-four 50k-to-100k champion-probability max delta: {metrics.get('top4_champion_max_abs_delta_50k_100k', 'n/a')}.

## Engine-only Champion Top 10

{top10_markdown}

## Outputs
- Advancement probabilities: `{repo_path_str(parquet_path)}` and `{repo_path_str(csv_path)}`; latest CSV: `{repo_path_str(TEAM_ADVANCEMENT_ENGINEONLY_LATEST)}`.
- Convergence: `{repo_path_str(MC_CONVERGENCE_FIGURE_PNG)}` and `{repo_path_str(MC_CONVERGENCE_FIGURE_PDF)}`.
- Stage heatmap: `{repo_path_str(STAGE_PROBS_HEATMAP_PNG)}` and `{repo_path_str(STAGE_PROBS_HEATMAP_PDF)}`.
- Upset table: `{repo_path_str(UPSET_RATE_TABLE_FILE)}`.
- Run records: `outputs/reports/run_stage2_build_structure_*.json` and `outputs/reports/run_stage2_run_simulation_*.json`.

## Human Gate 2
Human confirmation remains required before Stage 3: structure reviewed, deterministic checks accepted, monotone probabilities accepted, and upset-rate calibration accepted.
"""
    STAGE2_SUMMARY_FILE.write_text(summary, encoding="utf-8")  # write markdown summary


def _markdown_table(frame: pd.DataFrame) -> str:
    """Return a compact markdown table without optional pandas dependencies."""
    columns = frame.columns.tolist()  # stable column order
    header = "| " + " | ".join(columns) + " |"  # markdown header row
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"  # separator row
    body = [  # data rows
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *body])  # full table


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)  # CLI description
    parser.add_argument("--n-sims", type=int, default=N_SIMULATIONS, help="Number of tournament simulations.")
    parser.add_argument("--n-workers", type=int, default=STAGE2_DEFAULT_N_WORKERS, help="Parallel workers.")
    parser.add_argument("--check-sims", type=int, default=200, help="Short deterministic-check draw count.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Master random seed.")
    return parser.parse_args()  # parsed flags


def main() -> None:
    """Run Stage 2 simulation and write all required outputs."""
    args = parse_args()  # CLI flags
    if args.n_sims <= 0 or args.n_workers <= 0 or args.check_sims <= 0:  # basic validation
        raise ValueError("n-sims, n-workers, and check-sims must be positive")  # fail fast
    create_directories()  # ensure output directories exist
    tag = _timestamp()  # shared output timestamp
    logger, log_path = get_run_logger("s2_run_simulation", stage="stage2", step="run_simulation")  # log
    with RunRecord(  # machine-readable audit record
        stage="stage2",
        step="run_simulation",
        script_path=__file__,
        global_seed=args.seed,
        n_workers=args.n_workers,
        n_simulations=args.n_sims,
        logger=logger,
    ) as record:
        record.add_input(DIXON_COLES_PARAMS_LATEST_FILE)  # fitted Stage 1 goal engine
        record.add_input(ELO_CURRENT_RATINGS_FILE)  # current Elo for upset and penalty skill terms
        record.add_input(GROUPS_2026_FILE)  # group structure
        record.add_input(SCHEDULE_2026_FILE)  # full schedule
        record.add_input(BRACKET_PAIRINGS_2026_FILE)  # bracket and Annex C
        record.add_output(log_path)  # human-readable log
        model = DixonColesModel.load(DIXON_COLES_PARAMS_LATEST_FILE)  # load fitted engine
        structure = load_structure(GROUPS_2026_FILE, SCHEDULE_2026_FILE, BRACKET_PAIRINGS_2026_FILE)  # load structure
        elo_ratings = pd.read_csv(ELO_CURRENT_RATINGS_FILE)  # load current Elo table
        prepared = prepare_simulation_inputs(model, structure, elo_ratings=elo_ratings)  # shared inputs
        checks = _determinism_checks(prepared, args.seed, args.n_workers, min(args.check_sims, args.n_sims))  # checks
        start = time.perf_counter()  # simulation-only timer
        probabilities, metrics = _run_parallel(args.n_sims, args.n_workers, args.seed, prepared)  # main run
        simulation_seconds = time.perf_counter() - start  # elapsed seconds
        metrics["simulation_seconds"] = simulation_seconds  # record wall time
        metrics["draws_per_second"] = args.n_sims / simulation_seconds if simulation_seconds else 0.0  # throughput
        validation = validate_probabilities(probabilities)  # probability invariants
        metrics.update(_convergence_stability(probabilities, metrics["champion_trace"], args.n_sims))  # convergence
        parquet_path, csv_path = _write_advancement_tables(probabilities, tag)  # write tables
        upset_table = _write_upset_table(metrics)  # write upset-rate audit
        convergence_caption = _write_convergence_figure(probabilities, metrics["champion_trace"], args.n_sims)  # figure
        _write_heatmap(probabilities)  # stage-probability heatmap
        _write_summary(  # gate summary markdown
            probabilities,
            validation,
            checks,
            metrics,
            args.n_sims,
            args.n_workers,
            parquet_path,
            csv_path,
        )
        record.add_params(  # key simulation parameters
            {
                "selected_engine": model.selected_engine,
                "score_distribution": model.score_distribution,
                "score_dispersion": model.score_dispersion,
                "score_matrix_max_goals": model.max_goals,
                "check_sims": checks["check_sims"],
                "thread_caps": {
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                    "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
                },
            }
        )
        record.add_metrics(  # run checks and headline metrics
            {
                **validation,
                **checks,
                "simulation_seconds": metrics["simulation_seconds"],
                "draws_per_second": metrics["draws_per_second"],
                "knockout_upsets": metrics["knockout_upsets"],
                "knockout_decisive_matches": metrics["knockout_decisive_matches"],
                "knockout_upset_rate": metrics["knockout_upset_rate"],
                "group_draws": metrics["group_draws"],
                "group_draw_rate": metrics["group_draw_rate"],
                "knockout_matches": metrics["knockout_matches"],
                "knockout_to_extra_time": metrics["knockout_to_extra_time"],
                "knockout_to_extra_time_rate": metrics["knockout_to_extra_time_rate"],
                "knockout_to_penalties": metrics["knockout_to_penalties"],
                "knockout_to_penalties_rate": metrics["knockout_to_penalties_rate"],
                "upset_rate_delta_vs_reference": metrics["upset_rate_delta_vs_reference"],
                "convergence_50k_100k_available": metrics["convergence_50k_100k_available"],
                "top4_champion_abs_delta_50k_100k": metrics.get("top4_champion_abs_delta_50k_100k", {}),
                "top4_champion_max_abs_delta_50k_100k": metrics.get("top4_champion_max_abs_delta_50k_100k"),
                "top4_champion_delta_under_0_5pp": metrics.get("top4_champion_delta_under_0_5pp"),
                "t_g6_action": metrics["t_g6_action"],
                "top10_champion": probabilities.head(10)[["team", "p_champion"]].to_dict(orient="records"),
                "upset_table_rows": len(upset_table),
            }
        )
        for path in (  # hash all required outputs
            parquet_path,
            csv_path,
            TEAM_ADVANCEMENT_ENGINEONLY_LATEST,
            UPSET_RATE_TABLE_FILE,
            MC_CONVERGENCE_FIGURE_PNG,
            MC_CONVERGENCE_FIGURE_PDF,
            convergence_caption,
            STAGE_PROBS_HEATMAP_PNG,
            STAGE_PROBS_HEATMAP_PDF,
            STAGE2_SUMMARY_FILE,
        ):
            record.add_output_artifact(path)  # hashed output artifact
        logger.info("Wrote advancement probabilities to %s and %s", parquet_path, csv_path)  # log tables
        logger.info("Validation checks: %s", validation)  # log invariants
        logger.info("Determinism checks: %s", checks)  # log deterministic checks
        logger.info("Simulation metrics: %s", {key: value for key, value in metrics.items() if key != "champion_trace"})  # log


if __name__ == "__main__":  # direct execution
    main()  # run script
