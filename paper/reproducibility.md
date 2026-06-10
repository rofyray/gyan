# Reproducibility

## Final Run Identity

| Field | Value |
|---|---|
| Release tag | `gyan-v1.0-final` |
| Git commit | `eee0a29` |
| Global seed | `20260611` |
| Configured freeze timestamp | `2026-06-10T19:00:00Z` |
| Stage 3 market snapshot | `2026-06-10T19:12:08.427758+00:00` |
| Final artifact tag | `20260610_191222` |
| Stage 2 simulations | 100,000 |
| Stage 3 feature-expert simulations | 100,000 |

The run records contain the configured no-post-freeze field from `src/gyan/config.py`.
The actual final live market vector was refreshed at
`2026-06-10T19:12:08.427758+00:00`. Final reproducibility is therefore tied to
the run records, source hashes, and output hashes rather than to the nominal
freeze field alone.

## Run Records

| Stage | Run record |
|---|---|
| Stage 1 download | `artifacts/reports/run_stage1_download_data_20260610_190556.json` |
| Stage 1 clean matches | `artifacts/reports/run_stage1_clean_matches_20260610_190601.json` |
| Stage 1 Elo | `artifacts/reports/run_stage1_build_elo_20260610_190607.json` |
| Stage 1 socioeconomic | `artifacts/reports/run_stage1_build_socioeconomic_20260610_190614.json` |
| Stage 1 squad value | `artifacts/reports/run_stage1_build_squad_value_20260610_190619.json` |
| Stage 1 goal fit | `artifacts/reports/run_stage1_fit_dixon_coles_20260610_190809.json` |
| Stage 1 engine validation | `artifacts/reports/run_stage1_validate_engine_20260610_190918.json` |
| Stage 2 structure | `artifacts/reports/run_stage2_build_structure_20260610_190929.json` |
| Stage 2 100k simulation | `artifacts/reports/run_stage2_run_simulation_20260610_191007.json` |
| Stage 3 100k ensemble | `artifacts/reports/run_stage3_build_ensemble_20260610_191212.json` |
| Stage 4 evaluation | `artifacts/reports/run_stage4_evaluation_20260610_191224.json` |

Each run record includes package versions, git metadata, seed, recorded inputs,
recorded outputs, and SHA-256 artifact descriptors where the script registers
them.

## Final Outputs

| Output | Path |
|---|---|
| Final board | `artifacts/tables/gyan_2026_predictions_20260610_191222.csv` |
| Modal bracket | `artifacts/tables/modal_bracket_2026_20260610_191222.csv` |
| Backtest metrics | `artifacts/tables/backtest_metrics_20260610_191222.csv` |
| Ablation table | `artifacts/tables/ablation_20260610_191222.csv` |
| Benchmark table | `artifacts/tables/benchmark_2026_20260610_191222.csv` |
| Stage 4 validation | `artifacts/reports/stage4_output_validation_20260610_191222.json` |
| Paper figures | `paper/figures/` |
| Manuscript draft | `paper/manuscript.md` |

Stage 4 validation reports `all_passed: true`.

## Key Final Inputs

These key inputs are recorded in the final Stage 3 and Stage 4 run records.

| Input | SHA-256 prefix | Bytes |
|---|---:|---:|
| `data/processed/dixon_coles_params_latest.json` | `9d3b10846f25` | 29,626 |
| `outputs/tables/team_advancement_probs_engineonly_2026_latest.csv` | `b4a3855843c5` | 2,730 |
| `data/processed/squad_features_2026.parquet` | `0e0a5ae6d47f` | 13,310 |
| `data/processed/socioeconomic_features.parquet` | `df5fd5492aec` | 45,079 |
| `data/processed/matches_with_elo.parquet` | `803ae808e76c` | 850,364 |
| `outputs/tables/elo_current_ratings_from_d1.csv` | `80a83aeb482e` | 9,576 |
| `data/raw/d1_martj42_shootouts.csv` | `e52e503badc1` | 28,809 |
| `data/raw/d11_world_bank_gdp_per_capita_ppp.json` | `235cfaea47a6` | 4,167,217 |
| `data/raw/d11_world_bank_population.json` | `2bb1839c1404` | 3,526,530 |
| `data/raw/d15_historical_world_cup_outrights.csv` | `d20037fb79fd` | 7,526 |
| `data/processed/groups_2026.parquet` | `8574cb89a2c5` | 3,472 |
| `data/processed/schedule_2026.parquet` | `56d443ed49c8` | 8,987 |
| `data/processed/bracket_pairings_2026.json` | `5f0bf7afa3c0` | 79,889 |
| `data/raw/d13_polymarket_world_cup_winner_event.json` | `614f55ef8e6c` | 218,334 |
| `data/raw/d14_kalshi_mens_world_cup_winner_markets.json` | `943a2dd420ea` | 85,969 |
| `data/raw/d15_bookmakersreview_world_cup.html` | `f61be0e8da63` | 317,357 |
| `data/processed/market_implied_live.parquet` | `6f7aff480faa` | 4,003 |
| `outputs/tables/gyan_forecast_2026_latest.csv` | `5d7814edd951` | 6,241 |
| `outputs/tables/expert_boards_2026_latest.csv` | `ec16b5cc49a2` | 18,992 |
| `outputs/tables/ensemble_weights_latest.csv` | `c7f3760372bf` | 1,297 |

## Source Inventory

| ID | Source | Local cache / file group |
|---|---|---|
| D1 | martj42 international results | `d1_martj42_results.csv`, `d1_martj42_shootouts.csv`, `d1_martj42_former_names.csv` |
| D2 | World Football Elo Ratings | `d2_eloratings_about.html`, `d2_eloratings_current_ratings.js`, `d2_eloratings_world.tsv`, `d2_eloratings_teams.tsv`, `d2_eloratings_en_teams.tsv` |
| D2b | Elo GitHub mirrors | `d2b_demetriodor_football_elo_ratings.html`, `d2b_jgravier_soccer_elo.html` |
| D3 | FiveThirtyEight SPI archive | `d3_spi_matches_intl.csv`, `d3_spi_global_rankings.csv` |
| D3b | FiveThirtyEight World Cup predictions archive | `d3b_world_cup_predictions_datapackage.json`, `d3b_wc_20140609_140000.csv` |
| D4 | Transfermarkt-derived data and Transfermarkt pages | `d4_transfermarkt_players.csv.gz`, `d4_transfermarkt_national_teams.csv.gz`, `d4_transfermarkt_national_team_values.html`, `d4_transfermarkt_national_team_pages/*.html` |
| D5 | FIFA men's rankings | `d5_fifa_mens_ranking.html`, `d5_fifa_rankings_api.json` |
| D6 | 2026 squad lists | `d6_wikipedia_2026_world_cup_squads.html`, `d6_espn_2026_world_cup_squad_lists.html` |
| D7 | 2026 groups, schedule, and knockout assignments | `d7_wikipedia_2026_world_cup.html`, `d7_fifa_official_2026_world_cup.html`, `d7_fifa_official_2026_match_schedule.pdf`, `d7_wikipedia_2026_world_cup_knockout_stage.html` |
| D8 | Hoffmann, Ging, and Ramasamy 2002 | `d8_hoffmann_ging_ramasamy_2002.pdf` |
| D9 | Goldman Sachs 2026 report | `d9_goldman_2026_world_cup_report.pdf` |
| D10 | Klement / Panmure Liberum report | `d10_klement_panmure_liberum_2026.pdf` |
| D11 | World Bank GDP per capita PPP and population APIs | `d11_world_bank_gdp_per_capita_ppp.json`, `d11_world_bank_population.json` |
| D12 | Climate Knowledge Portal and country-average temperature table | `d12_world_bank_climate_data_collections_readme.html`, `d12_country_average_yearly_temperature_wikipedia.html` |
| D13 | Polymarket Gamma API | `d13_polymarket_world_cup_winner_event.json` |
| D14 | Kalshi public markets API | `d14_kalshi_mens_world_cup_winner_markets.json` |
| D15 | Bookmaker live and historical outright sources | `d15_bookmakersreview_world_cup.html`, `d15_historical_world_cup_outrights.csv`, historical D15 PDF/HTML caches |
| Manual | Injury/absence tracker | `injuries_2026.csv` |

The Stage 1 source URLs are defined in `src/gyan/data/*.py`. Live market URLs are
defined in `src/gyan/ensemble/market.py`.

## Executed Run Order

```bash
.venv/bin/python scripts/s1_download_data.py
.venv/bin/python scripts/s1_clean_matches.py
.venv/bin/python scripts/s1_build_elo.py
.venv/bin/python scripts/s1_build_socioeconomic.py
.venv/bin/python scripts/s1_build_squad_value.py
.venv/bin/python scripts/s1_fit_dixon_coles.py
.venv/bin/python scripts/s1_validate_engine.py
.venv/bin/python scripts/s2_build_structure.py
.venv/bin/python scripts/s2_run_simulation.py --n-sims 100000
.venv/bin/python scripts/s3_build_ensemble.py --n-sims 100000
.venv/bin/python scripts/s4_backtest.py
```

## Environment

- Requirements file: `requirements.txt`
- Python and package versions: recorded in each run record
- Output validation: `artifacts/reports/stage4_output_validation_20260610_191222.json`
