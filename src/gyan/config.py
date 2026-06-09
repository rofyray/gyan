"""GYAN model configuration: the single source of truth for the whole project.

Every path, random seed, and modelling constant lives here so that no other
module ever hard-codes a "magic number". Each constant carries a comment naming
its source (see PRD/CONVENTIONS.md Section 7 for the data-source registry).

This is a STARTER file. As stages are built, add new constants here rather than
inline in code. Keep it all-Python and all-commented (project rule).
"""

from __future__ import annotations  # allow modern type-hint syntax on all runtimes

from pathlib import Path  # filesystem paths that work cross-platform

# ---------------------------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT resolves to the gyan-wc-model/ directory (two levels above this
# file: src/gyan/config.py -> src/gyan -> src -> ROOT). Using parents[2] keeps
# the path correct no matter where the interpreter is launched from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]  # repo root directory

# Data directories (see CONVENTIONS Section 1). raw is immutable once written.
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"            # downloaded source data
DATA_INTERIM: Path = PROJECT_ROOT / "data" / "interim"    # cleaned intermediate data
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed" # model-ready feature tables

# Output directories. Tables, figures, and machine-readable run reports.
OUTPUTS_TABLES: Path = PROJECT_ROOT / "outputs" / "tables"    # .parquet and .csv results
OUTPUTS_FIGURES: Path = PROJECT_ROOT / "outputs" / "figures"  # .png plus vector figures
OUTPUTS_REPORTS: Path = PROJECT_ROOT / "outputs" / "reports"  # JSON run records, summaries

# Logs and the paper workspace.
LOGS_DIR: Path = PROJECT_ROOT / "logs"     # timestamped run logs (git-ignored)
PAPER_DIR: Path = PROJECT_ROOT / "paper"   # manuscript draft and paper-ready assets

# Specific named input files that are hand-maintained or built early.
INJURIES_FILE: Path = DATA_RAW / "injuries_2026.csv"          # editable injury tracker
TEAM_NAME_MAP_FILE: Path = DATA_INTERIM / "team_name_map.csv" # canonical team-name map
MATCHES_CLEAN_FILE: Path = DATA_INTERIM / "matches_clean.parquet" # canonical match table
SHOOTOUTS_RAW_FILE: Path = DATA_RAW / "d1_martj42_shootouts.csv" # D1 penalty-shootout winners
MATCHES_WITH_ELO_FILE: Path = DATA_PROCESSED / "matches_with_elo.parquet" # Elo feature table
ELO_REFERENCE_WORLD_FILE: Path = DATA_RAW / "d2_eloratings_world.tsv" # D2 current Elo table
ELO_REFERENCE_TEAM_LABELS_FILE: Path = DATA_RAW / "d2_eloratings_en_teams.tsv" # D2 code labels
ELO_REFERENCE_TEAMS_FILE: Path = DATA_RAW / "d2_eloratings_teams.tsv" # D2 successor-code table
ELO_CURRENT_RATINGS_FILE: Path = OUTPUTS_TABLES / "elo_current_ratings_from_d1.csv" # Elo audit
ELO_SPOTCHECK_FILE: Path = OUTPUTS_TABLES / "elo_reference_spotcheck.csv" # D2 comparison audit
FIFA_RANKINGS_API_FILE: Path = DATA_RAW / "d5_fifa_rankings_api.json" # official D5 ranking JSON
WORLD_BANK_GDP_PPP_FILE: Path = DATA_RAW / "d11_world_bank_gdp_per_capita_ppp.json" # D11 GDP
WORLD_BANK_POPULATION_FILE: Path = DATA_RAW / "d11_world_bank_population.json" # D11 population
COUNTRY_TEMPERATURE_FILE: Path = DATA_RAW / "d12_country_average_yearly_temperature_wikipedia.html" # D12 proxy
SOCIOECONOMIC_FEATURES_FILE: Path = DATA_PROCESSED / "socioeconomic_features.parquet" # Stage 1.5
SOCIOECONOMIC_FEATURES_CSV: Path = OUTPUTS_TABLES / "socioeconomic_features.csv" # audit CSV
TRANSFERMARKT_VALUES_FILE: Path = DATA_RAW / "d4_transfermarkt_national_team_values.html" # legacy D4 club values
TRANSFERMARKT_PLAYERS_FILE: Path = DATA_RAW / "d4_transfermarkt_players.csv.gz" # D4 individual player values
TRANSFERMARKT_NATIONAL_TEAMS_FILE: Path = DATA_RAW / "d4_transfermarkt_national_teams.csv.gz" # D4 national-team pages
TRANSFERMARKT_TEAM_PAGES_DIR: Path = DATA_RAW / "d4_transfermarkt_national_team_pages" # D4 squad pages
WIKIPEDIA_2026_SQUADS_FILE: Path = DATA_RAW / "d6_wikipedia_2026_world_cup_squads.html" # D6 squads
ESPN_2026_SQUADS_FILE: Path = DATA_RAW / "d6_espn_2026_world_cup_squad_lists.html" # D6 cross-check
SQUAD_FEATURES_2026_FILE: Path = DATA_PROCESSED / "squad_features_2026.parquet" # Stage 1.6
SQUAD_FEATURES_2026_CSV: Path = OUTPUTS_TABLES / "squad_features_2026.csv" # squad audit CSV
SQUAD_PLAYERS_2026_FILE: Path = OUTPUTS_TABLES / "squad_players_2026.csv" # player audit CSV
SPI_MATCHES_FILE: Path = DATA_RAW / "d3_spi_matches_intl.csv" # D3 international SPI validation archive
ENGINE_VALIDATION_FILE: Path = OUTPUTS_TABLES / "engine_validation_latest.csv" # latest validation audit
ENGINE_VALIDATION_SPI_FILE: Path = OUTPUTS_TABLES / "engine_validation_spi_benchmark_latest.csv" # D3 SPI benchmark audit
ENGINE_DRAW_CALIBRATION_FILE: Path = OUTPUTS_TABLES / "engine_draw_calibration_latest.csv" # draw-rate audit
DIXON_COLES_PARAMS_LATEST_FILE: Path = DATA_PROCESSED / "dixon_coles_params_latest.json" # Stage 1 engine params
DIXON_COLES_PARAMS_DC_FILE: Path = DATA_PROCESSED / "dixon_coles_params_dixon_coles_latest.json" # DC candidate
DIXON_COLES_PARAMS_PLAIN_FILE: Path = DATA_PROCESSED / "dixon_coles_params_plain_poisson_latest.json" # no-rho candidate
STAGE1_SUMMARY_FILE: Path = OUTPUTS_REPORTS / "stage1_summary.md" # Stage 1 gate summary
WIKIPEDIA_2026_WORLD_CUP_FILE: Path = DATA_RAW / "d7_wikipedia_2026_world_cup.html" # D7 page
WIKIPEDIA_2026_KNOCKOUT_FILE: Path = DATA_RAW / "d7_wikipedia_2026_world_cup_knockout_stage.html" # Annex C
FIFA_OFFICIAL_2026_WORLD_CUP_FILE: Path = DATA_RAW / "d7_fifa_official_2026_world_cup.html" # D7 official shell
FIFA_OFFICIAL_2026_MATCH_SCHEDULE_PDF: Path = DATA_RAW / "d7_fifa_official_2026_match_schedule.pdf" # D7 official schedule PDF
GROUPS_2026_FILE: Path = DATA_PROCESSED / "groups_2026.parquet" # Stage 2 group table
SCHEDULE_2026_FILE: Path = DATA_PROCESSED / "schedule_2026.parquet" # Stage 2 fixture table
BRACKET_PAIRINGS_2026_FILE: Path = DATA_PROCESSED / "bracket_pairings_2026.json" # Stage 2 bracket
TEAM_ADVANCEMENT_ENGINEONLY_LATEST: Path = OUTPUTS_TABLES / "team_advancement_probs_engineonly_2026_latest.csv" # latest Stage 2 table
MC_CONVERGENCE_FIGURE_PNG: Path = OUTPUTS_FIGURES / "mc_convergence.png" # Stage 2 convergence PNG
MC_CONVERGENCE_FIGURE_PDF: Path = OUTPUTS_FIGURES / "mc_convergence.pdf" # Stage 2 convergence PDF
STAGE_PROBS_HEATMAP_PNG: Path = OUTPUTS_FIGURES / "engine_stage_probability_heatmap.png" # Stage 2 heatmap PNG
STAGE_PROBS_HEATMAP_PDF: Path = OUTPUTS_FIGURES / "engine_stage_probability_heatmap.pdf" # Stage 2 heatmap PDF
UPSET_RATE_TABLE_FILE: Path = OUTPUTS_TABLES / "upset_rate_engineonly_2026.csv" # Stage 2 upset audit
STAGE2_SUMMARY_FILE: Path = OUTPUTS_REPORTS / "stage2_summary.md" # Stage 2 gate summary
MARKET_IMPLIED_LIVE_FILE: Path = DATA_PROCESSED / "market_implied_live.parquet" # Stage 3 market vector
MARKET_SOURCE_DIVERGENCE_FILE: Path = OUTPUTS_TABLES / "market_source_divergence_latest.csv" # live-vs-proxy market audit
MARKET_STAGE_ENGINE_SHAPE_AUDIT_FILE: Path = OUTPUTS_TABLES / "market_stage_engine_shape_audit_latest.csv" # market stage-source audit
EXPERT_CORRELATION_DIAGNOSTICS_FILE: Path = OUTPUTS_TABLES / "expert_board_correlation_diagnostics_latest.csv" # expert diversity audit
POLYMARKET_WORLD_CUP_WINNER_FILE: Path = DATA_RAW / "d13_polymarket_world_cup_winner_event.json" # D13 PM
KALSHI_WORLD_CUP_WINNER_FILE: Path = DATA_RAW / "d14_kalshi_mens_world_cup_winner_markets.json" # D14 Kalshi
BOOKMAKER_WORLD_CUP_WINNER_FILE: Path = DATA_RAW / "d15_bookmakersreview_world_cup.html" # D15 books
BACKTEST_MARKET_OUTRIGHTS_FILE: Path = DATA_RAW / "d15_historical_world_cup_outrights.csv" # D15 historical books
EXPERT_BOARDS_2026_FILE: Path = OUTPUTS_TABLES / "expert_boards_2026_latest.csv" # Stage 3 experts
ENSEMBLE_WEIGHTS_FILE: Path = OUTPUTS_TABLES / "ensemble_weights_latest.csv" # Stage 3 weights
YIELD_DELTA_TABLE_FILE: Path = OUTPUTS_TABLES / "yield_named_vs_nominal_delta_2026.csv" # Stage 3 Y delta
GYAN_FORECAST_LATEST: Path = OUTPUTS_TABLES / "gyan_forecast_2026_latest.csv" # latest GYAN board
RELIABILITY_DIAGRAM_PNG: Path = OUTPUTS_FIGURES / "stage3_reliability_diagram.png" # Stage 3 calib PNG
RELIABILITY_DIAGRAM_PDF: Path = OUTPUTS_FIGURES / "stage3_reliability_diagram.pdf" # Stage 3 calib PDF
STAGE3_SUMMARY_FILE: Path = OUTPUTS_REPORTS / "stage3_summary.md" # Stage 3 gate summary

# Every output directory we may write to; create_directories() makes them all.
_ALL_OUTPUT_DIRS: tuple[Path, ...] = (
    DATA_RAW, DATA_INTERIM, DATA_PROCESSED,        # data tiers
    OUTPUTS_TABLES, OUTPUTS_FIGURES, OUTPUTS_REPORTS,  # outputs
    LOGS_DIR, PAPER_DIR,                            # logs and paper
)


def create_directories() -> None:
    """Create every project directory if it does not already exist.

    Call once at the start of any entry-point script so writes never fail on a
    missing folder. Idempotent: safe to call repeatedly.
    """
    for directory in _ALL_OUTPUT_DIRS:          # iterate over every needed folder
        directory.mkdir(parents=True, exist_ok=True)  # create it, parents included


def repo_path_str(path: Path | str) -> str:
    """Return a repository-relative path string for shareable reports."""
    raw_path = Path(path)  # normalise once
    try:
        if raw_path.is_absolute():
            relative = raw_path.resolve().relative_to(PROJECT_ROOT.resolve())
        else:
            relative = raw_path
    except ValueError:
        return str(path)
    if str(relative) in {"", "."}:
        return "."
    return f"./{relative.as_posix()}"


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------
# Master random seed. Set to the 2026 World Cup opening date (11 June 2026) so it
# is memorable and documented. Every RNG stream is spawned from this (see
# utils/rng.py and CONVENTIONS Section 3.4). Log it in every run record.
GLOBAL_SEED: int = 20260611  # YYYYMMDD of the WC 2026 opening match
FINAL_FREEZE_TIMESTAMP_UTC: str = "2026-06-10T19:00:00Z"  # final pre-opening input freeze
FINAL_RELEASE_TAG: str = "gyan-v1.0-final"  # expected git tag for reproducible paper run

# ---------------------------------------------------------------------------
# 3. Monte-Carlo simulation
# ---------------------------------------------------------------------------
N_SIMULATIONS: int = 50_000        # default tournament draws; Goldman uses 50k (D9)
N_SIMULATIONS_HIGH: int = 100_000  # high-precision setting; Silver's PELE uses 100k
MAX_GOALS: int = 10                # truncate scoreline matrices at 10-10 (prob mass ~1)

# ---------------------------------------------------------------------------
# 4. Elo ratings (eloratings.net formula, source D2)
# ---------------------------------------------------------------------------
ELO_INITIAL_RATING: float = 1500.0   # rating assigned to a team with no history
ELO_HOME_ADVANTAGE: float = 100.0    # rating points added to the genuine home side
ELO_DIVISOR: float = 400.0           # logistic divisor in the win-expectancy formula

# K-factor by match-importance label. Values are exactly those of eloratings.net.
ELO_K_BY_IMPORTANCE: dict[str, int] = {
    "world_cup_finals": 60,          # the World Cup final tournament itself
    "continental_finals": 50,        # Euros, Copa America finals, major intercontinental
    "wc_continental_qualifier": 40,  # WC and continental qualifiers, other major tournaments
    "other_tournament": 30,          # minor tournaments
    "friendly": 20,                  # friendlies
}

# ---------------------------------------------------------------------------
# 5. FIFA/Coca-Cola World Ranking (SUM method, source D5)
# ---------------------------------------------------------------------------
# IMPORTANT: the FIFA ranking is NOT the eloratings Elo. It uses a different
# divisor, no home advantage, and ignores goal difference. Keep them separate
# (CONVENTIONS Section 9 data hazard).
FIFA_DIVISOR: float = 600.0  # logistic divisor in the FIFA SUM expectancy formula

# ---------------------------------------------------------------------------
# 6. Dixon-Coles goal model (Stage 1 Task 1.7)
# ---------------------------------------------------------------------------
DIXON_COLES_XI: float = 0.0018          # time-decay rate; START value, tuned in Stage 1
DIXON_COLES_MAX_GOALS: int = MAX_GOALS  # scoreline matrix size, reuse the global cap
DIXON_COLES_RHO_BOUNDS: tuple[float, float] = (-0.20, 0.20)  # stable DC dependence bounds
DIXON_COLES_RECENT_MIN_DATE: str = "2014-01-01"  # fit on the modern international era
DIXON_COLES_VALIDATION_CUTOFF: str = "2024-01-01"  # heldout split for Stage 1.8
DIXON_COLES_MAX_TRAIN_MATCHES: int = 12_000  # cap optimisation rows for quick reproducibility
DIXON_COLES_MAXITER: int = 150  # L-BFGS-B iteration budget for Stage 1 local fitting
DIXON_COLES_RIDGE: float = 0.002  # weak ridge penalty to stabilise small-sample teams
SCORE_DISTRIBUTION_POISSON: str = "poisson"  # independent Poisson score matrix
SCORE_DISTRIBUTION_CORRELATED_NB: str = "correlated_negative_binomial"  # shared-frailty NB score matrix
DEFAULT_SCORE_DISTRIBUTION: str = SCORE_DISTRIBUTION_CORRELATED_NB  # calibrated Stage 2 score matrix
NEGATIVE_BINOMIAL_DISPERSION: float = 8.0  # heldout draw-calibrated shared-frailty shape k
DRAW_RATE_TOLERANCE: float = 0.01  # draw-rate gate tolerance in absolute probability points

# ---------------------------------------------------------------------------
# 7. Squad value / Yield expert (Stage 1 Task 1.6, Stage 3 Task 3.4)
# ---------------------------------------------------------------------------
UEFA_VALUE_DISCOUNT: float = 0.30  # Transfermarkt pro-Europe correction (PELE-style)
# Weight multipliers applied to a named player's market value based on injury
# status from injuries_2026.csv. Editable; "doubtful" is a judgement call.
SQUAD_STATUS_WEIGHT: dict[str, float] = {
    "available": 1.0,  # fully fit, counts at full value
    "doubtful": 0.5,   # uncertain to play, counts at half value (editable)
    "out": 0.0,        # ruled out, contributes nothing
}

# ---------------------------------------------------------------------------
# 8. Socioeconomic expert (Hoffmann, Ging & Ramasamy 2002, source D8)
# ---------------------------------------------------------------------------
TEMP_OPTIMUM_C: float = 14.0  # ideal mean annual temperature; term is (temp - 14)^2
# Published OLS coefficients from the 2002 paper, used as priors / starting values.
HOFFMANN_2002_COEFFICIENTS: dict[str, float] = {
    "constant": 492.59,        # intercept a
    "gnp_per_capita": 0.0107,  # b1 on GNP/capita
    "gnp_per_capita_sq": -2.45e-07,  # b2 on GNP/capita squared (inverted-U)
    "temp_dev_sq": -0.4895,    # eta on (temperature - 14)^2
    "host_dummy": 81.05,       # k on the host-country dummy
    "latin_x_pop_share": 8587.46,  # phi on Latin-dummy times population share
}
HOFFMANN_2002_R_SQUARED: float = 0.318  # the paper's reported R^2 (academic base)
KLEMENT_TARGET_R_SQUARED: float = 0.55  # Klement's augmented model target (with FIFA points)
LATIN_FOOTBALL_NATIONS: tuple[str, ...] = (  # Luso-Hispanic Americas plus Spain/Portugal
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica",
    "Cuba", "Dominican Republic", "Ecuador", "El Salvador", "Guatemala",
    "Honduras", "Mexico", "Nicaragua", "Panama", "Paraguay", "Peru",
    "Portugal", "Spain", "Uruguay", "Venezuela",
)

# ---------------------------------------------------------------------------
# 9. Market expert (Stage 3 Task 3.1)
# ---------------------------------------------------------------------------
# Live 2026: blend Polymarket + Kalshi + bookmaker consensus. Backtest
# 2014/18/22: bookmaker outrights only. Polymarket has a 2022 event but no
# complete public pre-kickoff price vector in the cached APIs; Kalshi's World Cup
# market history starts with 2026 markets in 2025. Weights are the within-market
# expert pooling weights; start equal, optionally liquidity-tilt.
MARKET_SOURCE_WEIGHTS_LIVE: dict[str, float] = {
    "polymarket": 0.40,  # highest volume / liquidity prediction market (D13)
    "kalshi": 0.30,      # CFTC-regulated prediction market (D14)
    "bookmaker": 0.30,   # de-vigged sportsbook consensus (D15) for cross-period consistency
}
MARKET_SOURCE_WEIGHTS_BACKTEST: dict[str, float] = {
    "bookmaker": 1.0,    # cached D15 historical outrights drive Stage 4 backtests
}
# De-vig method for converting raw odds/prices to a normalised probability vector.
# "proportional" divides each implied probability by their sum (simple, robust).
DEVIG_METHOD: str = "proportional"
STAGE3_DEFAULT_N_SIMULATIONS: int = 50_000  # expert-board simulation draws for Stage 3
STAGE3_DEFAULT_N_WORKERS: int = 4  # conservative local default for expert simulations
STAGE3_MIN_SHIPPED_EXPERT_WEIGHT: float = 0.05  # keep all four GYAN experts active in shipped pool
STRENGTH_RATING_DIVISOR: float = 1_000.0  # Elo-like rating gap -> log-goal ability scale
STRENGTH_MODEL_DEFENSE_SHARE: float = 0.45  # stronger teams attack more and concede less
STAGE3_VALIDATION_CUTOFF: str = "2024-01-01"  # heldout split for Stage 3 pool fitting
T_G4_TOP_TEAM_DIVERGENCE_THRESHOLD: float = 0.08  # sharp top-team divergence threshold
GOLDMAN_2026_TOP_PROBS: dict[str, float] = {  # D9 top-team benchmark from main PRD
    "Spain": 0.257,  # Goldman 2026 champion probability
    "France": 0.189,  # Goldman 2026 champion probability
    "Argentina": 0.143,  # Goldman 2026 champion probability
    "Brazil": 0.076,  # Goldman 2026 champion probability
    "England": 0.050,  # Goldman 2026 champion probability
}

# ---------------------------------------------------------------------------
# 10. 2026 tournament structure (source D7)
# ---------------------------------------------------------------------------
N_TEAMS: int = 48        # expanded field
N_GROUPS: int = 12       # groups A through L
GROUP_SIZE: int = 4      # four teams per group
N_BEST_THIRDS: int = 8   # eight best third-placed teams advance to the Round of 32
N_MATCHES_TOTAL: int = 104  # total matches in the 2026 schedule
GROUP_LABELS: tuple[str, ...] = tuple("ABCDEFGHIJKL")  # the 12 group letters
HOST_NATIONS_2026: tuple[str, ...] = ("Canada", "Mexico", "United States")  # co-hosts
STAGE2_DEFAULT_N_WORKERS: int = 4  # conservative local default; joblib can override
STAGE2_CONVERGENCE_POINTS: tuple[int, ...] = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000) # PRD
PENALTY_SKILL_ELO_DIVISOR: float = 800.0  # small shootout skill tilt from Elo gap
EXTRA_TIME_SCALE: float = 30.0 / 90.0  # extra-time Poisson means are 30/90 of regulation
HISTORICAL_KNOCKOUT_UPSET_RATE: float = 1.0 / 3.0  # T-G6 rough historical base-rate target

# Points awarded in the group stage (standard football scoring).
POINTS_WIN: int = 3   # points for a win
POINTS_DRAW: int = 1  # points for a draw
POINTS_LOSS: int = 0  # points for a loss

# ---------------------------------------------------------------------------
# 11. Evaluation (Stage 4)
# ---------------------------------------------------------------------------
BACKTEST_TOURNAMENTS: tuple[int, ...] = (2014, 2018, 2022)  # held-out backtest years
RPS_N_CATEGORIES_WDL: int = 3  # win/draw/loss ordered categories for match-level RPS
