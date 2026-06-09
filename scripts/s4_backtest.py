"""Run Stage 4 evaluation, benchmark, final board, and paper asset assembly."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-gyan")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gyan.config import (  # noqa: E402
    BACKTEST_MARKET_OUTRIGHTS_FILE,
    BACKTEST_TOURNAMENTS,
    BRACKET_PAIRINGS_2026_FILE,
    ELO_CURRENT_RATINGS_FILE,
    ENSEMBLE_WEIGHTS_FILE,
    EXPERT_CORRELATION_DIAGNOSTICS_FILE,
    EXPERT_BOARDS_2026_FILE,
    FINAL_FREEZE_TIMESTAMP_UTC,
    FINAL_RELEASE_TAG,
    GLOBAL_SEED,
    GOLDMAN_2026_TOP_PROBS,
    GROUPS_2026_FILE,
    GYAN_FORECAST_LATEST,
    MARKET_IMPLIED_LIVE_FILE,
    MARKET_SOURCE_DIVERGENCE_FILE,
    MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE,
    MATCHES_WITH_ELO_FILE,
    OUTPUTS_FIGURES,
    OUTPUTS_REPORTS,
    OUTPUTS_TABLES,
    PAPER_DIR,
    PROJECT_ROOT,
    SHOOTOUTS_RAW_FILE,
    SOCIOECONOMIC_FEATURES_FILE,
    T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD,
    WORLD_BANK_GDP_PPP_FILE,
    WORLD_BANK_POPULATION_FILE,
    create_directories,
    repo_path_str,
)
from gyan.evaluation.backtest import (  # noqa: E402
    expert_match_forecasts,
    historical_expert_rating_snapshots,
    load_historical_market_outrights,
    load_world_bank_series,
    pool_rating_series,
    prepare_tournament_backtest,
    score_expert,
)
from gyan.evaluation.scoring import mean_scores  # noqa: E402
from gyan.simulation.tournament import _resolve_slot  # noqa: E402
from gyan.utils.logging import RunRecord, get_run_logger, git_commit_short  # noqa: E402


EXPERT_ORDER: tuple[str, ...] = ("goal", "yield_named", "socioeconomic", "market")
PELE_2026_TOP_PROBS: dict[str, float] = {"Spain": 0.185, "France": 0.170, "England": 0.104}


def _timestamp() -> str:
    """Return a compact UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _normalised_weights(weights_file: Path) -> dict[str, float]:
    """Load shipped Stage 3 weights, falling back to goal-only."""
    if not weights_file.exists():
        return {"goal": 1.0, "yield_named": 0.0, "socioeconomic": 0.0, "market": 0.0}
    weights = pd.read_csv(weights_file)
    shipped = weights[weights["weight_set"] == "shipped"]
    result = {expert: 0.0 for expert in EXPERT_ORDER}
    for row in shipped.itertuples(index=False):
        if row.expert in result:
            result[row.expert] = float(row.weight)
    total = sum(result.values())
    if total <= 0.0:
        result["goal"] = 1.0
        total = 1.0
    return {expert: value / total for expert, value in result.items()}


def _benchmark_market_lookup() -> tuple[dict[str, float], list[str], dict[str, str]]:
    """Load the refreshed Stage 3 live market vector for benchmark comparisons."""
    if not MARKET_IMPLIED_LIVE_FILE.exists():
        raise FileNotFoundError(
            f"Stage 4 benchmark requires the refreshed Stage 3 live market vector at {MARKET_IMPLIED_LIVE_FILE}; "
            "rerun scripts/s3_build_ensemble.py before scripts/s4_backtest.py."
        )
    market = pd.read_parquet(MARKET_IMPLIED_LIVE_FILE)
    required = {"team", "p_champion"}
    if not required.issubset(market.columns):
        raise ValueError(f"Live market benchmark file is missing required columns: {required - set(market.columns)}")
    ranked = market.sort_values("p_champion", ascending=False)
    lookup = ranked.set_index("team")["p_champion"].astype(float).to_dict()
    top_teams = ranked.head(12)["team"].astype(str).tolist()
    metadata = {
        "market_benchmark_source": "stage3_refreshed_live_market_vector",
        "market_snapshot_time_utc": str(ranked["snapshot_time_utc"].iloc[0]) if "snapshot_time_utc" in ranked.columns and len(ranked) else "",
        "market_source_note": str(ranked["market_source_note"].iloc[0]) if "market_source_note" in ranked.columns and len(ranked) else "",
    }
    return lookup, top_teams, metadata


def _backtest_ratings(
    tournament,
    matches: pd.DataFrame,
    feature_map: pd.DataFrame,
    gdp: pd.DataFrame,
    population: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """Build all Stage 4 no-leakage expert rating series for one tournament."""
    return historical_expert_rating_snapshots(tournament, matches, feature_map, gdp, population, BACKTEST_MARKET_OUTRIGHTS_FILE)


def _score_pooled_configuration(name: str, included: tuple[str, ...], ratings: dict[str, pd.Series], tournament) -> dict[str, object]:
    """Score an equal-weight pooled expert configuration."""
    weights = {expert: (1.0 if expert in included else 0.0) for expert in ratings if expert != "yield_nominal"}
    pooled = pool_rating_series(ratings, weights)
    row, _board, _forecasts = score_expert(name, pooled, tournament)
    row["configuration"] = "+".join(included)
    return row


def _run_backtests(
    matches: pd.DataFrame,
    shootouts: pd.DataFrame,
    feature_map: pd.DataFrame,
    gdp: pd.DataFrame,
    population: pd.DataFrame,
    shipped_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, dict[str, object]]]:
    """Run all historical backtests and return metrics, ablations, calibration rows."""
    metric_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    audits: dict[int, dict[str, object]] = {}
    for year in BACKTEST_TOURNAMENTS:
        tournament = prepare_tournament_backtest(matches, shootouts, year)
        audits[year] = tournament.leakage_audit
        ratings, champion_overrides, market_statuses = _backtest_ratings(tournament, matches, feature_map, gdp, population)
        forecasts_by_expert: dict[str, np.ndarray] = {}
        outcomes = None
        for expert in EXPERT_ORDER:
            row, _board, forecasts = score_expert(
                expert,
                ratings[expert],
                tournament,
                champion_override=champion_overrides.get(expert),
                market_status=market_statuses.get(expert),
            )
            metric_rows.append(row)
            forecasts_by_expert[expert] = forecasts
            if outcomes is None:
                _forecast, outcomes = expert_match_forecasts(ratings[expert], tournament)
        gyan_ratings = pool_rating_series(ratings, shipped_weights)
        gyan_row, _gyan_board, gyan_forecasts = score_expert("GYAN", gyan_ratings, tournament)
        gyan_row["configuration"] = "stage3_shipped_weights"
        metric_rows.append(gyan_row)
        shipped_ablation_row = gyan_row.copy()
        shipped_ablation_row["model"] = "shipped_gyan"
        shipped_ablation_row["configuration"] = "stage3_shipped_four_expert_weights"
        ablation_rows.append(shipped_ablation_row)
        predicted = gyan_forecasts.reshape(-1)
        observed = outcomes.reshape(-1)
        bins = np.linspace(0.0, 1.0, 7)
        for lower, upper in zip(bins[:-1], bins[1:], strict=True):
            mask = (predicted >= lower) & (predicted < upper if upper < 1.0 else predicted <= upper)
            if mask.any():
                calibration_rows.append(
                    {
                        "tournament": year,
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "mean_predicted": float(predicted[mask].mean()),
                        "observed_rate": float(observed[mask].mean()),
                        "n": int(mask.sum()),
                    }
                )
        configs = {
            "goal_only": ("goal",),
            "goal_yield": ("goal", "yield_named"),
            "goal_yield_socioeconomic": ("goal", "yield_named", "socioeconomic"),
            "goal_yield_market": ("goal", "yield_named", "market"),
            "full_gyan_equal": ("goal", "yield_named", "socioeconomic", "market"),
        }
        for name, included in configs.items():
            ablation_rows.append(_score_pooled_configuration(name, included, ratings, tournament))
        named_scores = mean_scores(*expert_match_forecasts(ratings["yield_named"], tournament))
        nominal_scores = mean_scores(*expert_match_forecasts(ratings["yield_nominal"], tournament))
        ablation_rows.append(
            {
                "tournament": year,
                "model": "yield_named_vs_nominal",
                "configuration": "yield_named_minus_yield_nominal",
                "mean_match_rps": named_scores["mean_rps"],
                "mean_match_brier": named_scores["mean_brier"],
                "mean_match_log_loss": named_scores["mean_log_loss"],
                "rps_delta_named_minus_nominal": named_scores["mean_rps"] - nominal_scores["mean_rps"],
                "leakage_passed": tournament.leakage_audit["passed"],
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(ablation_rows), pd.DataFrame(calibration_rows), audits


def _write_outputs(backtest: pd.DataFrame, ablation: pd.DataFrame, calibration: pd.DataFrame, tag: str) -> dict[str, Path]:
    """Write Stage 4 tables and figures."""
    paths: dict[str, Path] = {}
    paths["backtest_csv"] = OUTPUTS_TABLES / f"backtest_metrics_{tag}.csv"
    paths["ablation_csv"] = OUTPUTS_TABLES / f"ablation_{tag}.csv"
    backtest.to_csv(paths["backtest_csv"], index=False)
    ablation.to_csv(paths["ablation_csv"], index=False)
    paths["calibration_csv"] = OUTPUTS_TABLES / f"backtest_calibration_{tag}.csv"
    calibration.to_csv(paths["calibration_csv"], index=False)
    paths.update(_plot_backtest_calibration(calibration, tag))
    paths.update(_plot_ablation(ablation, tag))
    return paths


def _plot_backtest_calibration(calibration: pd.DataFrame, tag: str) -> dict[str, Path]:
    """Write backtest calibration figure."""
    png = OUTPUTS_FIGURES / f"backtest_calibration_{tag}.png"
    pdf = OUTPUTS_FIGURES / f"backtest_calibration_{tag}.pdf"
    caption = OUTPUTS_FIGURES / f"backtest_calibration_{tag}.txt"
    plt.figure(figsize=(6.2, 5.2))
    for year, frame in calibration.groupby("tournament"):
        plt.plot(frame["mean_predicted"], frame["observed_rate"], marker="o", linewidth=1.5, label=str(year))
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Backtest Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    caption.write_text(
        f"GYAN W/D/L calibration across the 2014, 2018, and 2022 World Cup backtests. Final paper input freeze timestamp: {FINAL_FREEZE_TIMESTAMP_UTC}.\n",
        encoding="utf-8",
    )
    return {"backtest_calibration_png": png, "backtest_calibration_pdf": pdf, "backtest_calibration_caption": caption}


def _plot_ablation(ablation: pd.DataFrame, tag: str) -> dict[str, Path]:
    """Write ablation figure."""
    png = OUTPUTS_FIGURES / f"ablation_{tag}.png"
    pdf = OUTPUTS_FIGURES / f"ablation_{tag}.pdf"
    caption = OUTPUTS_FIGURES / f"ablation_{tag}.txt"
    frame = ablation[ablation["model"].isin(["shipped_gyan", "goal_only", "goal_yield", "goal_yield_socioeconomic", "goal_yield_market", "full_gyan_equal"])].copy()
    pivot = frame.pivot_table(index="model", values="mean_match_rps", aggfunc="mean").sort_values("mean_match_rps")
    plt.figure(figsize=(7.2, 4.8))
    plt.barh(pivot.index, pivot["mean_match_rps"], color="#2F6F73")
    plt.xlabel("Mean match RPS")
    plt.title("Backtest Ablation")
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    caption.write_text(
        f"Lower RPS is better; bars average each ablation configuration over 2014, 2018, and 2022, including the shipped four-expert GYAN pool. Final paper input freeze timestamp: {FINAL_FREEZE_TIMESTAMP_UTC}.\n",
        encoding="utf-8",
    )
    return {"ablation_png": png, "ablation_pdf": pdf, "ablation_caption": caption}


def _write_benchmark_and_final_board(tag: str) -> dict[str, Path]:
    """Write benchmark comparison, final 2026 board, and headline figures."""
    paths: dict[str, Path] = {}
    gyan = pd.read_csv(GYAN_FORECAST_LATEST).sort_values("p_champion", ascending=False)
    market_lookup, market_top_teams, market_metadata = _benchmark_market_lookup()
    benchmark_teams = list(dict.fromkeys(gyan.head(12)["team"].tolist() + list(GOLDMAN_2026_TOP_PROBS) + list(PELE_2026_TOP_PROBS) + market_top_teams))
    rows = []
    gyan_lookup = gyan.set_index("team")["p_champion"].to_dict()
    for team in benchmark_teams:
        rows.append(
            {
                "team": team,
                "gyan_p_champion": float(gyan_lookup.get(team, 0.0)),
                "goldman_p_champion": GOLDMAN_2026_TOP_PROBS.get(team, np.nan),
                "pele_p_champion": PELE_2026_TOP_PROBS.get(team, np.nan),
                "market_p_champion": market_lookup.get(team, np.nan),
                **market_metadata,
                "interpretation": _benchmark_note(team, float(gyan_lookup.get(team, 0.0)), market_lookup.get(team)),
            }
        )
    benchmark = pd.DataFrame(rows).sort_values("gyan_p_champion", ascending=False)
    paths["benchmark_csv"] = OUTPUTS_TABLES / f"benchmark_2026_{tag}.csv"
    benchmark.to_csv(paths["benchmark_csv"], index=False)
    final_board = gyan[["team", "p_reach_R32", "p_reach_R16", "p_reach_QF", "p_reach_SF", "p_reach_final", "p_champion"]].copy()
    paths["final_board_csv"] = OUTPUTS_TABLES / f"gyan_2026_predictions_{tag}.csv"
    final_board.to_csv(paths["final_board_csv"], index=False)
    paths.update(_plot_benchmark(benchmark, tag))
    paths.update(_plot_headline(final_board, tag))
    paths.update(_plot_modal_bracket(final_board, tag))
    return paths


def _top_team_divergence_check(benchmark: pd.DataFrame) -> dict[str, object]:
    """Return T-G4-style divergence diagnostics for the final benchmark table."""
    rows: list[dict[str, object]] = []
    comparable = benchmark[benchmark["goldman_p_champion"].notna() & benchmark["market_p_champion"].notna()].copy()
    for row in comparable.itertuples(index=False):
        goldman_delta = abs(float(row.gyan_p_champion) - float(row.goldman_p_champion))
        market_delta = abs(float(row.gyan_p_champion) - float(row.market_p_champion))
        rows.append(
            {
                "team": row.team,
                "gyan_p_champion": float(row.gyan_p_champion),
                "goldman_p_champion": float(row.goldman_p_champion),
                "market_p_champion": float(row.market_p_champion),
                "abs_delta_vs_goldman": goldman_delta,
                "abs_delta_vs_market": market_delta,
                "diverges_from_both": bool(
                    goldman_delta > T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD
                    and market_delta > T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD
                ),
            }
        )
    tripped = [row for row in rows if row["diverges_from_both"]]
    france = next((row for row in rows if row["team"] == "France"), None)
    return {
        "threshold": T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD,
        "comparisons": rows,
        "tripped": bool(tripped),
        "tripped_teams": [row["team"] for row in tripped],
        "france": france,
        "france_blocker_resolved": bool(france is not None and not france["diverges_from_both"]),
    }


def _benchmark_note(team: str, gyan_probability: float, market_probability: float | None) -> str:
    """Return a concise benchmark interpretation note."""
    external = [
        value
        for value in (GOLDMAN_2026_TOP_PROBS.get(team), PELE_2026_TOP_PROBS.get(team), market_probability)
        if value is not None and not pd.isna(value)
    ]
    if not external:
        return "GYAN top-12 team without published top benchmark in PRD."
    mean_external = float(np.mean(external))
    delta = gyan_probability - mean_external
    if abs(delta) <= 0.04:
        return "Broadly aligned with published benchmark range."
    if delta > 0.0:
        return "GYAN is higher under the shipped four-expert historical-RPS pool."
    return "GYAN is lower under the shipped four-expert historical-RPS pool."


def _plot_benchmark(benchmark: pd.DataFrame, tag: str) -> dict[str, Path]:
    """Write grouped benchmark chart."""
    png = OUTPUTS_FIGURES / f"benchmark_2026_{tag}.png"
    pdf = OUTPUTS_FIGURES / f"benchmark_2026_{tag}.pdf"
    caption = OUTPUTS_FIGURES / f"benchmark_2026_{tag}.txt"
    frame = benchmark.head(12).copy()
    x = np.arange(len(frame))
    width = 0.2
    plt.figure(figsize=(10.5, 5.0))
    for offset, column, label in [(-1.5, "gyan_p_champion", "GYAN"), (-0.5, "goldman_p_champion", "Goldman"), (0.5, "pele_p_champion", "PELE"), (1.5, "market_p_champion", "Market")]:
        plt.bar(x + offset * width, frame[column].fillna(0.0), width=width, label=label)
    plt.xticks(x, frame["team"], rotation=35, ha="right")
    plt.ylabel("Champion probability")
    plt.title("2026 Champion Probability Benchmarks")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    caption.write_text(
        f"Champion-probability comparison for the top GYAN teams, Goldman/PELE benchmarks, and the refreshed Stage 3 live market board. Final paper input freeze timestamp: {FINAL_FREEZE_TIMESTAMP_UTC}.\n",
        encoding="utf-8",
    )
    return {"benchmark_png": png, "benchmark_pdf": pdf, "benchmark_caption": caption}


def _plot_headline(board: pd.DataFrame, tag: str) -> dict[str, Path]:
    """Write headline champion-probability chart."""
    png = OUTPUTS_FIGURES / f"gyan_2026_champion_top12_{tag}.png"
    pdf = OUTPUTS_FIGURES / f"gyan_2026_champion_top12_{tag}.pdf"
    caption = OUTPUTS_FIGURES / f"gyan_2026_champion_top12_{tag}.txt"
    frame = board.head(12).sort_values("p_champion")
    plt.figure(figsize=(8.0, 5.4))
    plt.barh(frame["team"], frame["p_champion"], color="#7A3E3E")
    plt.xlabel("Champion probability")
    plt.title("GYAN 2026 Champion Probabilities")
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    caption.write_text(
        f"Top-12 GYAN champion probabilities; model version Stage 4, seed {GLOBAL_SEED}, source board {GYAN_FORECAST_LATEST.name}, final paper input freeze timestamp {FINAL_FREEZE_TIMESTAMP_UTC}.\n",
        encoding="utf-8",
    )
    return {"headline_png": png, "headline_pdf": pdf, "headline_caption": caption}


def _plot_modal_bracket(board: pd.DataFrame, tag: str) -> dict[str, Path]:
    """Write a slot-resolved deterministic modal-bracket figure and table."""
    png = OUTPUTS_FIGURES / f"modal_bracket_2026_{tag}.png"
    pdf = OUTPUTS_FIGURES / f"modal_bracket_2026_{tag}.pdf"
    caption = OUTPUTS_FIGURES / f"modal_bracket_2026_{tag}.txt"
    csv = OUTPUTS_TABLES / f"modal_bracket_2026_{tag}.csv"
    modal, metadata = _build_modal_bracket(board)
    modal.to_csv(csv, index=False)

    columns = [
        ("R32", "Round of 32", 0.015, 0.052, 5.4),
        ("R16", "Round of 16", 0.285, 0.092, 6.0),
        ("QF", "Quarterfinal", 0.500, 0.150, 6.4),
        ("SF", "Semifinal", 0.675, 0.230, 7.0),
        ("final", "Final", 0.820, 0.320, 7.6),
    ]
    fig, ax = plt.subplots(figsize=(16.0, 9.2))
    ax.axis("off")
    ax.text(0.015, 0.975, "GYAN 2026 Modal Bracket", fontsize=17, fontweight="bold", va="top")
    ax.text(
        0.015,
        0.940,
        "Slot-resolved deterministic chalk path from the current group draw, Annex C third-place assignment, and GYAN champion probabilities.",
        fontsize=8.8,
        color="#444444",
        va="top",
    )
    for stage, title, x, step, fontsize in columns:
        frame = modal[modal["stage"] == stage].copy()
        ax.text(x, 0.900, title, fontsize=9.2, fontweight="bold", va="top", color="#202020")
        for idx, row in enumerate(frame.itertuples(index=False)):
            y = 0.872 - idx * step
            line = f"M{row.match_id}: {row.home_team} vs {row.away_team}"
            winner = f"Winner: {row.winner}"
            ax.text(x, y, line, fontsize=fontsize, va="top", color="#202020")
            ax.text(x, y - step * 0.38, winner, fontsize=fontsize, va="top", color="#5A2F2F")
    champion = metadata["champion"]
    ax.text(0.820, 0.610, "Champion", fontsize=9.2, fontweight="bold", va="top", color="#202020")
    ax.text(0.820, 0.575, champion, fontsize=15, fontweight="bold", va="top", color="#5A2F2F")
    ax.text(
        0.015,
        0.025,
        "All listed matchups are bracket-slot resolutions, not observed fixtures. Winners are selected by higher GYAN champion probability, so this is an illustrative chalk path rather than a sampled tournament trace.",
        fontsize=7.2,
        color="#555555",
        va="bottom",
    )
    plt.tight_layout()
    plt.savefig(png, dpi=300)
    plt.savefig(pdf)
    plt.close()
    caption.write_text(
        f"Slot-resolved deterministic modal bracket. Matchups are defined from the current 2026 group draw and Annex C third-place assignment; winners are selected by higher GYAN champion probability, so the figure is an illustrative chalk path rather than a Monte Carlo path trace or prediction of fact. Final paper input freeze timestamp: {FINAL_FREEZE_TIMESTAMP_UTC}.\n",
        encoding="utf-8",
    )
    return {"modal_bracket_csv": csv, "modal_bracket_png": png, "modal_bracket_pdf": pdf, "modal_bracket_caption": caption}


def _build_modal_bracket(board: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Resolve a coherent deterministic bracket path from the final GYAN board."""
    required = {"team", "p_reach_R32", "p_champion"}
    missing_columns = required - set(board.columns)
    if missing_columns:
        raise ValueError(f"Final board missing columns for modal bracket: {sorted(missing_columns)}")
    groups = pd.read_parquet(GROUPS_2026_FILE)
    bracket = json.loads(BRACKET_PAIRINGS_2026_FILE.read_text(encoding="utf-8"))
    strengths = board.set_index("team")["p_champion"].astype(float)
    r32_probs = board.set_index("team")["p_reach_R32"].astype(float)
    missing_teams = sorted(set(groups["team"]) - set(strengths.index))
    if missing_teams:
        raise ValueError(f"Final board missing tournament teams for modal bracket: {missing_teams}")

    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    thirds: dict[str, str] = {}
    third_rows: list[dict[str, object]] = []
    group_rankings: dict[str, list[str]] = {}
    for group, frame in groups.groupby("group", sort=True):
        ranked = frame.copy()
        ranked["p_champion"] = ranked["team"].map(strengths)
        ranked["p_reach_R32"] = ranked["team"].map(r32_probs)
        ranked = ranked.sort_values(
            ["p_champion", "p_reach_R32", "group_position", "team"],
            ascending=[False, False, True, True],
        )
        ordered = ranked["team"].tolist()
        group_rankings[str(group)] = ordered
        winners[str(group)] = ordered[0]
        runners_up[str(group)] = ordered[1]
        thirds[str(group)] = ordered[2]
        third_rows.append(
            {
                "group": str(group),
                "team": ordered[2],
                "p_champion": float(strengths.loc[ordered[2]]),
                "p_reach_R32": float(r32_probs.loc[ordered[2]]),
            }
        )

    best_thirds = sorted(third_rows, key=lambda row: (-row["p_champion"], -row["p_reach_R32"], row["group"]))[:8]
    qualified_third_groups = sorted(row["group"] for row in best_thirds)
    combination_key = "".join(qualified_third_groups)
    if combination_key not in bracket["third_place_combinations"]:
        raise KeyError(f"Missing Annex C third-place assignment for groups {combination_key}")
    qualified_thirds = {group: thirds[group] for group in qualified_third_groups}
    qualifiers = {
        "winners": winners,
        "runners_up": runners_up,
        "thirds": thirds,
        "qualified_third_groups": qualified_third_groups,
        "qualified_thirds": qualified_thirds,
        "third_assignments": bracket["third_place_combinations"][combination_key],
    }

    match_winners: dict[int, str] = {}
    match_losers: dict[int, str] = {}
    rows: list[dict[str, object]] = []
    pairings = list(bracket["round_of_32"]) + list(bracket["knockout_tree"])
    for pairing in pairings:
        match_id = int(pairing["match_id"])
        home = _resolve_slot(pairing["home_slot"], qualifiers, match_winners, match_losers)
        away = _resolve_slot(pairing["away_slot"], qualifiers, match_winners, match_losers)
        home_strength = float(strengths.loc[home])
        away_strength = float(strengths.loc[away])
        if home_strength == away_strength:
            winner, loser = sorted([home, away])[0], sorted([home, away])[1]
        elif home_strength > away_strength:
            winner, loser = home, away
        else:
            winner, loser = away, home
        match_winners[match_id] = winner
        match_losers[match_id] = loser
        rows.append(
            {
                "match_id": match_id,
                "stage": pairing["stage"],
                "home_slot": pairing["home_slot"],
                "away_slot": pairing["away_slot"],
                "home_team": home,
                "away_team": away,
                "winner": winner,
                "loser": loser,
                "home_p_champion": home_strength,
                "away_p_champion": away_strength,
                "winner_p_champion": float(strengths.loc[winner]),
                "selection_rule": "higher_gyan_p_champion",
            }
        )
    modal = pd.DataFrame(rows)
    validation = _validate_modal_bracket(modal)
    metadata = {
        "champion": str(modal.loc[modal["match_id"] == 104, "winner"].iloc[0]),
        "qualified_third_groups": qualified_third_groups,
        "third_assignments": qualifiers["third_assignments"],
        "group_rankings": group_rankings,
        "validation": validation,
    }
    if not validation["coherent"]:
        raise ValueError(f"Modal bracket failed coherence checks: {validation}")
    return modal, metadata


def _validate_modal_bracket(modal: pd.DataFrame) -> dict[str, object]:
    """Validate that a modal bracket table is internally coherent."""
    required_ids = set(range(73, 105))
    match_ids = set(modal["match_id"].astype(int))
    final = modal[modal["match_id"] == 104].iloc[0]
    semifinal_winners = set(modal[modal["match_id"].isin([101, 102])]["winner"])
    finalists = {final["home_team"], final["away_team"]}
    qf_winners = set(modal[modal["match_id"].isin([97, 98, 99, 100])]["winner"])
    semifinalists = set(modal[modal["match_id"].isin([101, 102])]["home_team"]) | set(modal[modal["match_id"].isin([101, 102])]["away_team"])
    winners_in_match = bool(((modal["winner"] == modal["home_team"]) | (modal["winner"] == modal["away_team"])).all())
    checks = {
        "all_match_ids_present": match_ids == required_ids,
        "winners_in_own_match": winners_in_match,
        "qf_winners_are_semifinalists": qf_winners == semifinalists,
        "semifinal_winners_are_finalists": semifinal_winners == finalists,
        "champion_is_finalist": final["winner"] in finalists,
        "unresolved_matchups": int(modal[["home_team", "away_team", "winner"]].isna().sum().sum()),
    }
    checks["coherent"] = bool(
        checks["all_match_ids_present"]
        and checks["winners_in_own_match"]
        and checks["qf_winners_are_semifinalists"]
        and checks["semifinal_winners_are_finalists"]
        and checks["champion_is_finalist"]
        and checks["unresolved_matchups"] == 0
    )
    return checks


def _write_stage4_validation(paths: dict[str, Path], backtest: pd.DataFrame, ablation: pd.DataFrame, tag: str) -> tuple[Path, dict[str, object]]:
    """Write and enforce cross-artifact validation checks for Stage 4 outputs."""
    final_board = pd.read_csv(paths["final_board_csv"])
    latest_forecast = pd.read_csv(GYAN_FORECAST_LATEST)
    benchmark = pd.read_csv(paths["benchmark_csv"])
    modal = pd.read_csv(paths["modal_bracket_csv"])
    stage_columns = ["p_reach_R32", "p_reach_R16", "p_reach_QF", "p_reach_SF", "p_reach_final", "p_champion"]

    final_latest = final_board.merge(
        latest_forecast[["team", *stage_columns]],
        on="team",
        suffixes=("_final", "_latest"),
        validate="one_to_one",
    )
    max_final_latest_diff = max(
        float((final_latest[f"{column}_final"] - final_latest[f"{column}_latest"]).abs().max())
        for column in stage_columns
    )
    benchmark_join = benchmark.merge(
        final_board[["team", "p_champion"]],
        on="team",
        how="left",
        validate="many_to_one",
    )
    benchmark_diff = float((benchmark_join["gyan_p_champion"] - benchmark_join["p_champion"]).abs().fillna(0.0).max())
    expected_ablation_models = {
        "shipped_gyan",
        "goal_only",
        "goal_yield",
        "goal_yield_socioeconomic",
        "goal_yield_market",
        "full_gyan_equal",
        "yield_named_vs_nominal",
    }
    expected_backtest_models = set(EXPERT_ORDER) | {"GYAN"}
    backtest_model_coverage = {
        str(year): sorted(backtest.loc[backtest["tournament"] == year, "model"].dropna().unique().tolist())
        for year in BACKTEST_TOURNAMENTS
    }
    ablation_model_coverage = {
        str(year): sorted(ablation.loc[ablation["tournament"] == year, "model"].dropna().unique().tolist())
        for year in BACKTEST_TOURNAMENTS
    }
    modal_validation = _validate_modal_bracket(modal)
    checks = {
        "final_board_has_48_unique_teams": bool(len(final_board) == 48 and final_board["team"].is_unique),
        "final_board_probabilities_in_range": bool(((final_board[stage_columns] >= 0.0) & (final_board[stage_columns] <= 1.0)).all().all()),
        "final_board_probabilities_monotone": bool((final_board[stage_columns].diff(axis=1).iloc[:, 1:] <= 1e-12).all().all()),
        "champion_probability_sum_close_to_one": bool(np.isclose(final_board["p_champion"].sum(), 1.0, atol=1e-9)),
        "final_board_matches_latest_forecast": bool(max_final_latest_diff <= 1e-12),
        "benchmark_matches_final_board": bool(benchmark_diff <= 1e-12),
        "backtest_tournament_coverage": bool(set(backtest["tournament"].unique()) == set(BACKTEST_TOURNAMENTS)),
        "backtest_model_coverage": bool(all(expected_backtest_models.issubset(set(models)) for models in backtest_model_coverage.values())),
        "ablation_model_coverage": bool(all(expected_ablation_models.issubset(set(models)) for models in ablation_model_coverage.values())),
        "modal_bracket_coherent": bool(modal_validation["coherent"]),
        "output_files_exist": bool(all(path.exists() for path in paths.values() if isinstance(path, Path) and path.suffix)),
    }
    validation = {
        "tag": tag,
        "checks": checks,
        "all_passed": bool(all(checks.values())),
        "details": {
            "champion_probability_sum": float(final_board["p_champion"].sum()),
            "max_final_board_vs_latest_forecast_abs_diff": max_final_latest_diff,
            "benchmark_max_abs_diff": benchmark_diff,
            "backtest_model_coverage": backtest_model_coverage,
            "ablation_model_coverage": ablation_model_coverage,
            "modal_bracket": modal_validation,
            "modal_champion": str(modal.loc[modal["match_id"] == 104, "winner"].iloc[0]),
        },
    }
    path = OUTPUTS_REPORTS / f"stage4_output_validation_{tag}.json"
    path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["all_passed"]:
        raise ValueError(f"Stage 4 validation failed: {validation}")
    return path, validation


def _assemble_paper(tag: str, paths: dict[str, Path], summary: dict[str, object]) -> dict[str, Path]:
    """Assemble manuscript, reproducibility appendix, and vector figures."""
    paper_figures = PAPER_DIR / "figures"
    paper_figures.mkdir(parents=True, exist_ok=True)
    for stale_pdf in paper_figures.glob("*.pdf"):
        stale_pdf.unlink()
    for figure in _paper_figure_candidates(tag):
        shutil.copy2(figure, paper_figures / figure.name)
    manuscript = PAPER_DIR / "manuscript.md"
    reproducibility = PAPER_DIR / "reproducibility.md"
    manuscript.write_text(_manuscript_text(tag, paths, summary), encoding="utf-8")
    reproducibility.write_text(_reproducibility_text(paths), encoding="utf-8")
    return {"manuscript": manuscript, "reproducibility": reproducibility, "paper_figures": paper_figures}


def _paper_figure_candidates(tag: str) -> list[Path]:
    """Return current PDFs that should be copied into the paper figures folder."""
    timestamped_pdf = re.compile(r"_20[0-9]{6}_[0-9]{6}\.pdf$")
    candidates: list[Path] = []
    for figure in sorted(OUTPUTS_FIGURES.glob("*.pdf")):
        if f"_{tag}.pdf" in figure.name or not timestamped_pdf.search(figure.name):
            candidates.append(figure)
    return candidates


def _manuscript_text(tag: str, paths: dict[str, Path], summary: dict[str, object]) -> str:
    """Return the Stage 4 manuscript draft."""
    expert_correlation_note = _expert_correlation_note()  # Socioeconomic-vs-Market diversity caveat
    floor_policy_note = _floor_policy_note(summary["shipped_weights"])  # positive-floor framing
    match_objective_note = _match_objective_profile_note()  # group-heavy match objective caveat
    squad_window_limitation = _squad_window_limitation_text()  # post-freeze squad replacement caveat
    market_construction_limitation = _market_construction_limitation_text()  # market match/stage construction transfer
    market_source_limitation = _market_source_limitation_text()  # live-vs-proxy source-transfer caveat
    market_stage_limitation = _market_stage_limitation_text()  # market stage-source caveat
    return f"""# GYAN World Cup Model

## Introduction
GYAN is an ensemble forecasting model for the 2026 FIFA World Cup. It combines a goal model, a squad-yield signal, socioeconomic structure, and market information where reproducible sources are available.

## Data
The project follows the D1..D15 registry in `PRD/CONVENTIONS.md`: international results, Elo references, SPI/forecast archives, Transfermarkt squad values, FIFA ranking data, World Bank macro data, climate proxies, published benchmark PDFs, and market sources.

## Methods
The Goal expert uses the Stage 1 Poisson/Dixon-Coles mean engine with a draw-calibrated correlated negative-binomial score matrix inside the Stage 2 tournament simulator. Yield uses team-value and named-squad adjustments for 2026; in historical backtests it is represented by pre-tournament form proxies because historical named squads and values are not cached. The socioeconomic expert uses the Hoffmann specification with year-available World Bank GDP/population data. The Market expert uses live de-vigged 2026 outrights for the current board and cached D15 bookmaker outrights for historical calibration. Expert pooling uses the Stage 3 shipped four-expert weights `{summary['shipped_weights']}`, fit by constrained historical RPS with a positive minimum weight for every expert. {match_objective_note} The final paper input freeze timestamp is `{FINAL_FREEZE_TIMESTAMP_UTC}`; the final paper run should use no inputs after that cutoff.

## Results
Backtest metrics are in `{repo_path_str(paths['backtest_csv'])}`. The ablation table is `{repo_path_str(paths['ablation_csv'])}`. The benchmark table is `{repo_path_str(paths['benchmark_csv'])}`. The final 2026 board is `{repo_path_str(paths['final_board_csv'])}`.

Top 2026 teams: {summary['top10']}.

Shipped-vs-ablation check: {summary['shipped_vs_best_ablation']}.

Top-team divergence check: {summary['top_team_divergence']}.

## Discussion
The Stage 3 shipped model is now a four-expert GYAN pool rather than an engine-only forecast. Match-level validation without market outrights still favours the Goal expert, so that diagnostic remains reported separately; the shipped tournament board uses the historical no-leakage four-expert calibration because it can include the market expert.

{floor_policy_note}

The named-squad and injury-aware Yield expert is a methodological contribution, but it should not be described as the main driver of the current board. Its value is evaluated as a marginal design improvement through the yield-named versus yield-nominal ablation table, not as the dominant source of forecast mass.

Because the Market expert receives most of the shipped weight, agreement between GYAN and the live market benchmark is expected and should not be presented as independent validation. The independent-value evidence comes from the Goldman comparison, historical ablations, and the documented source/weight limitations.

{expert_correlation_note}

## Limitations
Forecasts are probabilities, not claims of fact. Stage 4 uses cached D15 historical outright boards for the backtest market expert: 2014 and 2018 use published bookmaker-consensus probabilities, and 2022 uses a pre-tournament William Hill decimal-odds board dated 2022-09-01. Polymarket's 2022 World Cup winner event is cached for audit, but its public CLOB/trade APIs did not expose a complete pre-kickoff price vector.
{squad_window_limitation}
{market_construction_limitation}
{market_source_limitation}
{market_stage_limitation}

## Reproducibility Appendix
See `paper/reproducibility.md`.
"""


def _latest_stage3_historical_fit() -> dict[str, object]:
    """Return historical weight-fit metrics from the latest Stage 3 run record."""
    latest_stage3 = sorted(OUTPUTS_REPORTS.glob("run_stage3_build_ensemble_*.json"))
    if not latest_stage3:
        return {}
    try:
        run_record = json.loads(latest_stage3[-1].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return run_record.get("metrics", {}).get("historical_weight_fit", {})


def _match_objective_profile_note() -> str:
    """Return the paper note for match-level fit composition versus tournament deployment."""
    profile = _latest_stage3_historical_fit().get("match_objective_profile", {})
    if not profile:
        return (
            "The weight objective is match-level mean RPS over the historical final tournaments, "
            "which is group-stage-heavy, and the resulting weights are then applied to a knockout-sensitive tournament board."
        )
    total = int(profile.get("matches_total", 0))
    group = int(profile.get("group_stage_matches", 0))
    group_share = float(profile.get("group_stage_share", 0.0))
    tournaments = ", ".join(str(year) for year in profile.get("historical_tournaments", []))
    return (
        f"The weight objective is match-level mean RPS over the {tournaments} final tournaments; "
        f"in the 32-team historical format, {group} of {total} matches ({group_share:.0%}) are group-stage games. "
        "Those weights are then applied to a 2026 tournament board whose champion probabilities depend heavily on "
        "knockout dynamics, bracket paths, and variance, so the high Market/strength weight can partly reflect the "
        "group-stage-heavy calibration target."
    )


def _floor_policy_note(shipped_weights: dict[str, float]) -> str:
    """Return the paper note for the Stage 3 positive-weight floor."""
    floor_payload = _latest_stage3_historical_fit()
    floor_cost = floor_payload.get("floor_rps_cost_vs_unconstrained", "not available")
    binding_experts = floor_payload.get("floor_binding_experts", ["goal", "yield_named"])
    ship_reason = str(floor_payload.get("ship_reason", "unknown"))
    loto = floor_payload.get("leave_one_tournament_out", {})
    loto_text = ""
    if isinstance(loto, dict) and "optimised_oos_rps" in loto and "equal_oos_rps" in loto:
        loto_text = (
            f" Leave-one-tournament-out RPS is {float(loto['optimised_oos_rps']):.6f} for the optimized weights "
            f"versus {float(loto['equal_oos_rps']):.6f} for equal weights."
        )
    goal_weight = float(shipped_weights.get("goal", 0.0))
    yield_weight = float(shipped_weights.get("yield_named", 0.0))
    socio_weight = float(shipped_weights.get("socioeconomic", 0.0))
    market_weight = float(shipped_weights.get("market", 0.0))
    if ship_reason == "equal_weights_loto_guard":
        return (
            "The 5% minimum expert weight is a deliberate implementation addition to keep all four "
            "experts represented in the optimized candidate; it was not part of the original PRD optimization spec. "
            f"In the optimized candidate it binds for {binding_experts}, and the in-sample RPS cost versus the "
            f"unconstrained selected-pool optimum is {floor_cost}.{loto_text} "
            "Because the optimized candidate does not beat equal weights out of sample, the shipped board uses "
            f"equal weights instead: Goal {goal_weight:.1%}, Yield {yield_weight:.1%}, Socioeconomic {socio_weight:.1%}, Market {market_weight:.1%}."
        )
    return (
        "The 5% minimum expert weight is a deliberate implementation addition to keep all four "
        f"experts represented for structure and interpretability; it was not part of the original PRD optimization spec. "
        f"In the current fit it binds for {binding_experts}, with Goal at {goal_weight:.1%} and Yield at {yield_weight:.1%}. "
        f"The board is therefore primarily a Socioeconomic-plus-Market blend ({socio_weight + market_weight:.1%} combined), "
        f"and the in-sample RPS cost versus the unconstrained selected-pool optimum is {floor_cost}.{loto_text}"
    )


def _market_construction_limitation_text() -> str:
    """Return the market construction-transfer limitation from Stage 3 diagnostics."""
    construction = _latest_stage3_historical_fit().get("market_expert_construction", {})
    if not construction:
        return (
            "\nThe historical Market match forecasts are derived from outright champion prices through "
            "a structural rating-to-match-probability transform; this construction-transfer diagnostic "
            "was not available in the latest Stage 3 run record."
        )
    return (
        "\nMarket construction differs by evaluation surface. Historically, the Market expert has no raw "
        "match W/D/L prices: bookmaker champion outrights are normalized, converted into log-probability "
        "ratings, and then passed through the calibrated rating-to-W/D/L score model for match-level RPS. "
        "For champion scoring, those historical outright probabilities are used directly. For 2026, the "
        "live Market expert uses the blended Polymarket/Kalshi/bookmaker champion vector directly, while "
        "non-champion stages are scaled from the Stage 2 engine path. Thus the Market weight transfers "
        "cleanly at the champion-probability primitive but should be described as a market-strength hybrid "
        "for match-level fitting, not as weight on raw match prices."
    )


def _squad_window_limitation_text() -> str:
    """Return the squad-freeze limitation for post-cutoff replacements."""
    return (
        f"\nThe final input freeze at {FINAL_FREEZE_TIMESTAMP_UTC} locks the squad and injury "
        "snapshot for reproducibility. That cutoff captures teams playing on June 11 and June 12, "
        "but teams whose first match is June 13 or later may still make permitted replacement changes "
        "inside their own 24-hour pre-match window. Those post-freeze squad or injury changes are an "
        "unavoidable limitation of a reproducible pre-opening forecast."
    )


def _expert_correlation_note() -> str:
    """Return a paper note about Socioeconomic-vs-Market board correlation."""
    if not EXPERT_CORRELATION_DIAGNOSTICS_FILE.exists():  # Stage 3 may not have produced this yet
        return "The Socioeconomic-versus-Market board-correlation diagnostic was not available for this run."
    diagnostics = pd.read_csv(EXPERT_CORRELATION_DIAGNOSTICS_FILE)  # Stage 3 expert diversity audit
    row = diagnostics[
        (diagnostics["expert_a"] == "socioeconomic")
        & (diagnostics["expert_b"] == "market")
        & (diagnostics["probability_column"] == "p_champion")
    ]
    if row.empty:
        return "The Socioeconomic-versus-Market champion-correlation diagnostic was not available for this run."
    record = row.iloc[0]
    pearson = float(record["pearson"])
    spearman = float(record["spearman"])
    top_overlap = int(record["champion_top12_overlap"])
    if abs(pearson) >= 0.80 or abs(spearman) >= 0.80:
        return (
            "The Socioeconomic and Market champion boards are highly correlated "
            f"(Pearson {pearson:.3f}, Spearman {spearman:.3f}, top-12 overlap {top_overlap}/12), "
            "so their combined weight should not be interpreted as fully independent information."
        )
    return (
        "The Socioeconomic and Market champion boards are not highly correlated in the Stage 3 diagnostic "
        f"(Pearson {pearson:.3f}, Spearman {spearman:.3f}, top-12 overlap {top_overlap}/12), "
        "which gives some support to expert diversity while still leaving Market as the dominant signal."
    )


def _market_source_limitation_text() -> str:
    """Return a source-transfer limitation based on the Stage 3 market audit."""
    if not MARKET_SOURCE_DIVERGENCE_FILE.exists():  # Stage 3 may not have produced the audit yet
        return ""
    divergence = pd.read_csv(MARKET_SOURCE_DIVERGENCE_FILE)  # live-vs-proxy audit
    if divergence.empty or "abs_delta" not in divergence.columns:  # no comparable proxy
        return "\nThe final live-vs-proxy market-source divergence check was unavailable because no pre-refresh market proxy vector was present."
    max_abs_delta = float(divergence["abs_delta"].max())  # largest team movement
    top_team = str(divergence.sort_values("abs_delta", ascending=False).iloc[0]["team"])  # headline team
    if max_abs_delta > T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD:  # material source-transfer warning
        return (
            "\nThe live 2026 Market expert differs materially from the pre-refresh proxy "
            f"(max champion-probability movement {max_abs_delta:.3f} for {top_team}). "
            "The market weight was earned on historical bookmaker outrights, while the live "
            "forecast blends Polymarket, Kalshi, and bookmaker sources; this source-transfer "
            "difference should be interpreted as a limitation rather than a refitted 2026 edge."
        )
    return (
        "\nThe live-vs-proxy market-source divergence check did not exceed the sharp-divergence "
        f"threshold; the largest champion-probability movement was {max_abs_delta:.3f} for {top_team}."
    )


def _market_stage_limitation_text() -> str:
    """Return a limitation for Market non-champion stages if they are engine-shaped."""
    if not MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE.exists():  # Stage 3 may not have produced the audit yet
        return ""
    audit = pd.read_csv(MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE)  # market stage-source audit
    error_columns = [column for column in audit.columns if column.endswith("_abs_reconstruction_error")]  # stage errors
    if not error_columns:  # malformed or legacy audit
        return ""
    max_error = float(audit[error_columns].max().max())  # reconstruction error over stages/teams
    if max_error <= 1e-10:  # Market stages match scaled engine path shape
        return (
            "\nThe Market expert directly prices champion probabilities, but its non-champion "
            "stage columns are reconstructed from the Stage 2 engine path shape. Therefore "
            "R32/R16/QF/SF/final Market probabilities are partly engine-derived, and the "
            "Market expert's independence is strongest only at the champion stage."
        )
    return (
        "\nThe Market non-champion stage audit did not exactly match the scaled engine path "
        f"shape; maximum reconstruction error was {max_error:.3g}."
    )


def _reproducibility_text(paths: dict[str, Path]) -> str:
    """Return reproducibility appendix text."""
    requirements = PROJECT_ROOT / "requirements.txt"
    return f"""# Reproducibility

- `GLOBAL_SEED`: `{GLOBAL_SEED}`
- Git commit at Stage 4 run: `{git_commit_short()}`
- Expected release tag for the final paper run: `{FINAL_RELEASE_TAG}`
- Final input freeze timestamp UTC: `{FINAL_FREEZE_TIMESTAMP_UTC}`
- No-post-freeze-input statement: final paper records should use no inputs after `{FINAL_FREEZE_TIMESTAMP_UTC}`.
- Requirements file: `{repo_path_str(requirements)}`
- Run order: `scripts/s1_download_data.py`, `scripts/s1_clean_matches.py`, `scripts/s1_build_elo.py`, `scripts/s1_build_socioeconomic.py`, `scripts/s1_build_squad_value.py`, `scripts/s1_fit_dixon_coles.py`, `scripts/s1_validate_engine.py`, `scripts/s2_build_structure.py`, `scripts/s2_run_simulation.py`, `scripts/s3_build_ensemble.py`, `scripts/s4_backtest.py`
- Final board: `{repo_path_str(paths['final_board_csv'])}`
- Backtest metrics: `{repo_path_str(paths['backtest_csv'])}`
- Ablation: `{repo_path_str(paths['ablation_csv'])}`
- Benchmark: `{repo_path_str(paths['benchmark_csv'])}`

Re-run Stage 4 with a fixed seed to regenerate the deterministic tables and figures. Earlier Monte Carlo stages should match within their documented simulation tolerance, or exactly when the same seed and worker partition are used.
"""


def _write_summary(paths: dict[str, Path], backtest: pd.DataFrame, ablation: pd.DataFrame, audits: dict[int, dict[str, object]], tag: str, validation: dict[str, object]) -> tuple[Path, dict[str, object]]:
    """Write Stage 4 summary markdown."""
    final_board = pd.read_csv(paths["final_board_csv"])
    top10 = final_board.head(10)[["team", "p_champion"]].to_dict(orient="records")
    gyan_metrics = backtest[backtest["model"] == "GYAN"][["tournament", "mean_match_rps", "champion_log_loss", "finalist_hit_rate", "semifinalist_hit_rate"]]
    ablation_mean = ablation.groupby("model", dropna=False)["mean_match_rps"].mean(numeric_only=True).sort_values().to_dict()
    non_shipped_ablation = {model: value for model, value in ablation_mean.items() if model not in {"shipped_gyan", "yield_named_vs_nominal"}}
    best_non_shipped_model = min(non_shipped_ablation, key=non_shipped_ablation.get)
    shipped_mean_rps = float(ablation_mean["shipped_gyan"])
    best_non_shipped_mean_rps = float(non_shipped_ablation[best_non_shipped_model])
    shipped_vs_best_ablation = {
        "shipped_gyan_mean_rps": shipped_mean_rps,
        "best_non_shipped_ablation": best_non_shipped_model,
        "best_non_shipped_mean_rps": best_non_shipped_mean_rps,
        "delta_shipped_minus_best_non_shipped": shipped_mean_rps - best_non_shipped_mean_rps,
        "blocker_resolved": bool(shipped_mean_rps <= best_non_shipped_mean_rps),
    }
    market_sources = backtest[backtest["model"] == "market"][["tournament", "market_status"]].to_dict(orient="records")
    shipped_weights = _normalised_weights(ENSEMBLE_WEIGHTS_FILE)
    benchmark = pd.read_csv(paths["benchmark_csv"])
    top_team_divergence = _top_team_divergence_check(benchmark)
    summary = {
        "top10": top10,
        "gyan_metrics": gyan_metrics.to_dict(orient="records"),
        "ablation_mean_rps": ablation_mean,
        "audits": audits,
        "market_sources": market_sources,
        "shipped_weights": shipped_weights,
        "shipped_vs_best_ablation": shipped_vs_best_ablation,
        "top_team_divergence": top_team_divergence,
        "output_validation": validation,
    }
    text = f"""# Stage 4 Summary

## Leakage Audit
`{audits}`

## Backtest Metrics
{gyan_metrics.to_markdown(index=False)}

## Shipped Weights
`{shipped_weights}`

## Ablation Mean RPS
`{ablation_mean}`

## Shipped vs Best Ablation
`{shipped_vs_best_ablation}`

## Top-Team Divergence
`{top_team_divergence}`

## Historical Market Sources
{pd.DataFrame(market_sources).to_markdown(index=False)}

## Output Validation
`{validation}`

## 2026 GYAN Top 10
{pd.DataFrame(top10).to_markdown(index=False)}

## Outputs
- Backtest metrics: `{repo_path_str(paths['backtest_csv'])}`
- Ablation: `{repo_path_str(paths['ablation_csv'])}`
- Benchmark: `{repo_path_str(paths['benchmark_csv'])}`
- Final board: `{repo_path_str(paths['final_board_csv'])}`
- Validation: `{repo_path_str(paths['validation_json'])}`
- Paper draft: `{repo_path_str(paths['manuscript'])}`
- Reproducibility appendix: `{repo_path_str(paths['reproducibility'])}`

## Human Gates
Human confirmation remains required for Gate 4 and final Gate 5.
"""
    path = OUTPUTS_REPORTS / "stage4_summary.md"
    path.write_text(text, encoding="utf-8")
    return path, summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Recorded seed for reproducibility metadata.")
    return parser.parse_args()


def main() -> None:
    """Run Stage 4."""
    args = parse_args()
    create_directories()
    tag = _timestamp()
    logger, log_path = get_run_logger("s4_backtest", stage="stage4", step="evaluation")
    with RunRecord(stage="stage4", step="evaluation", script_path=__file__, global_seed=args.seed, logger=logger) as record:
        start = time.perf_counter()
        inputs = [
            MATCHES_WITH_ELO_FILE,
            SOCIOECONOMIC_FEATURES_FILE,
            WORLD_BANK_GDP_PPP_FILE,
            WORLD_BANK_POPULATION_FILE,
            GYAN_FORECAST_LATEST,
            MARKET_IMPLIED_LIVE_FILE,
            EXPERT_CORRELATION_DIAGNOSTICS_FILE,
            EXPERT_BOARDS_2026_FILE,
            ENSEMBLE_WEIGHTS_FILE,
            ELO_CURRENT_RATINGS_FILE,
            SHOOTOUTS_RAW_FILE,
            BACKTEST_MARKET_OUTRIGHTS_FILE,
        ]
        for path in inputs:
            record.add_input(path)
        record.add_output(log_path)
        matches = pd.read_parquet(MATCHES_WITH_ELO_FILE)
        shootouts = pd.read_csv(SHOOTOUTS_RAW_FILE)
        feature_map = pd.read_parquet(SOCIOECONOMIC_FEATURES_FILE)
        gdp = load_world_bank_series(WORLD_BANK_GDP_PPP_FILE)
        population = load_world_bank_series(WORLD_BANK_POPULATION_FILE)
        historical_markets = load_historical_market_outrights(BACKTEST_MARKET_OUTRIGHTS_FILE)
        shipped_weights = _normalised_weights(ENSEMBLE_WEIGHTS_FILE)
        backtest, ablation, calibration, audits = _run_backtests(matches, shootouts, feature_map, gdp, population, shipped_weights)
        paths = _write_outputs(backtest, ablation, calibration, tag)
        paths.update(_write_benchmark_and_final_board(tag))
        validation_path, validation = _write_stage4_validation(paths, backtest, ablation, tag)
        paths["validation_json"] = validation_path
        summary_path, summary = _write_summary({**paths, "manuscript": PAPER_DIR / "manuscript.md", "reproducibility": PAPER_DIR / "reproducibility.md"}, backtest, ablation, audits, tag, validation)
        paths.update(_assemble_paper(tag, paths, summary))
        paths["summary"] = summary_path
        runtime_seconds = time.perf_counter() - start
        record.add_params(
            {
                "backtest_tournaments": BACKTEST_TOURNAMENTS,
                "shipped_weights": shipped_weights,
                "final_freeze_timestamp_utc": FINAL_FREEZE_TIMESTAMP_UTC,
                "expected_release_tag": FINAL_RELEASE_TAG,
                "no_post_freeze_inputs_statement": f"Final paper run must use no inputs after {FINAL_FREEZE_TIMESTAMP_UTC}.",
                "historical_market_sources": historical_markets.groupby("year")["source"].first().to_dict(),
                "historical_market_note": "Backtest market uses cached D15 historical outright boards; 2022 Polymarket exists but lacks a complete public pre-kickoff price vector.",
            }
        )
        record.add_metrics(
            {
                "runtime_seconds": runtime_seconds,
                "leakage_audits": audits,
                "gyan_backtest": summary["gyan_metrics"],
                "ablation_mean_rps": summary["ablation_mean_rps"],
                "shipped_vs_best_ablation": summary["shipped_vs_best_ablation"],
                "top_team_divergence": summary["top_team_divergence"],
                "output_validation": summary["output_validation"],
                "top10_champion": summary["top10"],
                "champion_probability_sum": float(pd.read_csv(paths["final_board_csv"])["p_champion"].sum()),
            }
        )
        for path in paths.values():
            if isinstance(path, Path) and path.is_file():
                record.add_output_artifact(path)
        logger.info("Stage 4 runtime seconds: %.2f", runtime_seconds)
        logger.info("Leakage audits: %s", audits)
        logger.info("GYAN backtest metrics: %s", summary["gyan_metrics"])
        logger.info("Top 10: %s", summary["top10"])


if __name__ == "__main__":
    main()
