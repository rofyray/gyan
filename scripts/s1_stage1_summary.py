"""Write the Stage 1 gate summary and the Elo trajectory figure."""

from __future__ import annotations  # consistent type hints

import json  # read run records
import os  # configure plotting cache
from pathlib import Path  # filesystem paths

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting backend
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-gyan")  # writable matplotlib cache
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")  # writable font cache root

import matplotlib.pyplot as plt  # noqa: E402  # plotting after env setup
import pandas as pd  # noqa: E402  # load output tables

from gyan.config import (  # central paths
    ELO_CURRENT_RATINGS_FILE,
    ELO_SPOTCHECK_FILE,
    ENGINE_VALIDATION_FILE,
    ENGINE_VALIDATION_SPI_FILE,
    MATCHES_WITH_ELO_FILE,
    OUTPUTS_FIGURES,
    OUTPUTS_REPORTS,
    SOCIOECONOMIC_FEATURES_FILE,
    SQUAD_FEATURES_2026_FILE,
    STAGE1_SUMMARY_FILE,
    create_directories,
    repo_path_str,
)


def _latest_record(step: str) -> dict[str, object]:
    """Return the newest run record for a Stage 1 step."""
    paths = sorted(OUTPUTS_REPORTS.glob(f"run_stage1_{step}_*.json"))  # matching records
    if not paths:  # step has not been run
        return {}  # empty record keeps summary generation tolerant
    return json.loads(paths[-1].read_text(encoding="utf-8"))  # newest by timestamped filename


def _write_elo_trajectory_figure() -> tuple[Path, Path, Path]:
    """Write the required Elo trajectory figure for 2026 favourites."""
    matches = pd.read_parquet(MATCHES_WITH_ELO_FILE)  # load match-level Elo features
    matches["date"] = pd.to_datetime(matches["date"])  # ensure datetime x-axis
    favourites = pd.read_csv(ELO_CURRENT_RATINGS_FILE).sort_values("elo_rating", ascending=False).head(8)  # top 8
    figure_rows: list[dict[str, object]] = []  # collect team-date-rating rows
    for team in favourites["team"]:  # one line per favourite
        home_rows = matches[matches["home_team"] == team][["date", "home_elo_post"]].rename(columns={"home_elo_post": "elo"})  # home
        away_rows = matches[matches["away_team"] == team][["date", "away_elo_post"]].rename(columns={"away_elo_post": "elo"})  # away
        team_rows = pd.concat([home_rows, away_rows], ignore_index=True).sort_values("date")  # full trajectory
        team_rows = team_rows[team_rows["date"] >= pd.Timestamp("2014-01-01")]  # since 2014
        for row in team_rows.itertuples(index=False):  # collect plotting rows
            figure_rows.append({"team": team, "date": row.date, "elo": row.elo})  # long row
    plot = pd.DataFrame(figure_rows)  # long-format plotting table
    png_path = OUTPUTS_FIGURES / "elo_trajectory_favourites_since_2014.png"  # PNG path
    pdf_path = OUTPUTS_FIGURES / "elo_trajectory_favourites_since_2014.pdf"  # vector path
    caption_path = OUTPUTS_FIGURES / "elo_trajectory_favourites_since_2014.txt"  # caption path
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)  # ensure figure directory
    plt.figure(figsize=(9, 5.5))  # compact paper-friendly figure
    for team, team_plot in plot.groupby("team"):  # draw one line per team
        plt.plot(team_plot["date"], team_plot["elo"], linewidth=1.6, label=team)  # trajectory
    plt.title("Elo Trajectory of 2026 Favourites Since 2014")  # title
    plt.xlabel("Date")  # x-axis label
    plt.ylabel("Elo rating")  # y-axis label
    plt.legend(fontsize=8, ncol=2)  # readable legend
    plt.tight_layout()  # avoid clipped labels
    plt.savefig(png_path, dpi=300)  # review PNG
    plt.savefig(pdf_path)  # vector PDF
    plt.close()  # release figure memory
    caption_path.write_text(  # required caption sidecar
        "Current top-eight Elo teams, with post-match Elo ratings tracked from January 2014 through the latest D1 result.\n",
        encoding="utf-8",
    )
    return png_path, pdf_path, caption_path  # written artifacts


def main() -> None:
    """Write the Stage 1 markdown summary and final required figure."""
    create_directories()  # ensure output directories exist
    socioeconomic_record = _latest_record("build_socioeconomic")  # latest Task 1.5 record
    squad_record = _latest_record("build_squad_value")  # latest Task 1.6 record
    fit_record = _latest_record("fit_dixon_coles")  # latest Task 1.7 record
    validation_record = _latest_record("validate_engine")  # latest Task 1.8 record
    matches = pd.read_parquet(MATCHES_WITH_ELO_FILE)  # match-feature output for counts/span
    squad = pd.read_parquet(SQUAD_FEATURES_2026_FILE)  # squad features for sanity values
    socioeconomic = pd.read_parquet(SOCIOECONOMIC_FEATURES_FILE)  # socioeconomic features
    spotcheck = pd.read_csv(ELO_SPOTCHECK_FILE)  # Elo deltas
    validation = pd.read_csv(ENGINE_VALIDATION_FILE)  # engine validation per-match table
    spi_validation = pd.read_csv(ENGINE_VALIDATION_SPI_FILE) if ENGINE_VALIDATION_SPI_FILE.exists() else pd.DataFrame()  # D3 audit
    png_path, pdf_path, caption_path = _write_elo_trajectory_figure()  # required Elo figure
    selected_spotcheck = spotcheck[spotcheck["selected_for_check"]]  # formal Elo subset
    summary = f"""# Stage 1 Summary

## Data and Elo
- Match feature rows: {len(matches):,}; date span: {pd.to_datetime(matches['date']).min().date()} to {pd.to_datetime(matches['date']).max().date()}.
- Elo selected-team max abs delta: {selected_spotcheck['abs_rating_delta'].max():.1f}; selected failures: {(~selected_spotcheck['within_tolerance']).sum()}.

## Socioeconomic
- Rows: {len(socioeconomic):,}; complete core-feature rows: {socioeconomic_record.get('metrics', {}).get('complete_feature_rows', 'n/a')}.
- Pure Hoffmann R^2: {socioeconomic_record.get('metrics', {}).get('pure_hoffmann_r_squared', 'n/a')}.
- Augmented FIFA-points R^2: {socioeconomic_record.get('metrics', {}).get('augmented_fifa_points_r_squared', 'n/a')}.
- Caveat: current Elo is used as the OLS target, so the published R^2 references are diagnostics rather than strict comparables.

## Squad Value
- Teams: {squad['team'].nunique()}; named players: {squad_record.get('metrics', {}).get('named_players', 'n/a')}.
- Most valuable squad: {squad.iloc[0]['team']} ({squad.iloc[0]['selected_squad_value_eur']:,.0f} EUR).
- Named-squad injury adjustment: {squad['injury_adjustment_eur'].sum():,.0f} EUR.
- Pre-lock lost-value what-if: {squad['pre_lock_lost_value_what_if_eur'].sum():,.0f} EUR.
- D4 player-value source: individual Transfermarkt national-team/profile market values; direct coverage {squad_record.get('metrics', {}).get('direct_player_market_value_rate', 'n/a')}; missing values are audited in `outputs/tables/squad_players_2026.csv`.

## Goal Engine
- Fit rows: {fit_record.get('metrics', {}).get('n_train_matches', 'n/a')}; teams fit: {fit_record.get('metrics', {}).get('n_teams', 'n/a')}.
- Raw optimiser success: {fit_record.get('metrics', {}).get('optimizer_success', 'n/a')}; accepted finite fit: {fit_record.get('metrics', {}).get('success', 'n/a')}.
- Heldout rows: {len(validation):,}; mean Dixon-Coles RPS: {validation_record.get('metrics', {}).get('mean_rps_dixon_coles', 'n/a')}; mean same-decay plain-Poisson RPS: {validation_record.get('metrics', {}).get('mean_rps_plain_poisson', 'n/a')}.
- Selected engine: {validation_record.get('metrics', {}).get('selected_engine', 'n/a')}; score distribution: {validation_record.get('metrics', {}).get('score_distribution', 'n/a')}; T-G3 actioned: {validation_record.get('metrics', {}).get('t_g3_actioned', 'n/a')}.
- Draw calibration: observed {validation_record.get('metrics', {}).get('observed_draw_rate', 'n/a')}; plain Poisson predicted {validation_record.get('metrics', {}).get('plain_poisson_predicted_draw_rate', 'n/a')} (pass={validation_record.get('metrics', {}).get('plain_poisson_draw_frequency_passed', 'n/a')}); calibrated score matrix predicted {validation_record.get('metrics', {}).get('calibrated_predicted_draw_rate', 'n/a')} (pass={validation_record.get('metrics', {}).get('calibrated_draw_frequency_passed', 'n/a')}).
- D3 SPI benchmark rows: {validation_record.get('metrics', {}).get('spi_international_rows', len(spi_validation))}; benchmark cutoff: {validation_record.get('metrics', {}).get('spi_benchmark_cutoff', 'n/a')}.
- D3 SPI benchmark RPS: model {validation_record.get('metrics', {}).get('mean_rps_model_on_spi_benchmark', 'n/a')}; SPI {validation_record.get('metrics', {}).get('mean_rps_spi_benchmark', 'n/a')}; model-minus-SPI {validation_record.get('metrics', {}).get('mean_rps_delta_model_minus_spi', 'n/a')}.

## Outputs
- Squad features: `data/processed/squad_features_2026.parquet`.
- Dixon-Coles/latest goal parameters: `data/processed/dixon_coles_params_latest.json`.
- Engine validation: `outputs/tables/engine_validation_latest.csv`.
- Draw calibration: `outputs/tables/engine_draw_calibration_latest.csv`.
- SPI benchmark validation: `outputs/tables/engine_validation_spi_benchmark_latest.csv`.
- Attack-defense figure: `outputs/figures/attack_defense_top20.png` and `.pdf`.
- Elo trajectory figure: `{repo_path_str(png_path)}` and `{repo_path_str(pdf_path)}`; caption: `{repo_path_str(caption_path)}`.

## Human Gate 1
Human confirmation remains required before Stage 2: data hashes recorded; Elo sane; R^2 diagnostics understood; T-G3 plain-Poisson selection actioned; no validation leakage by split date.
"""
    STAGE1_SUMMARY_FILE.write_text(summary, encoding="utf-8")  # write final markdown summary
    print(STAGE1_SUMMARY_FILE)  # small CLI confirmation for users/scripts


if __name__ == "__main__":  # allow direct script execution
    main()  # write summary
