# GYAN World Cup Model

GYAN is the final pre-opening forecast pipeline for the 2026 FIFA World Cup. It
combines four expert signals into a 48-team tournament probability board:

- **Goal:** an Elo-anchored international goal model. Dixon-Coles tuning was
  tested, but the final engine uses plain-Poisson means with a draw-calibrated
  correlated negative-binomial score matrix.
- **Yield named:** a named-squad value signal using 2026 squad lists,
  Transfermarkt player values, UEFA value discounting, age weighting, and
  injury/absence adjustments.
- **Socioeconomic:** a Hoffmann/Klement-style macro, climate, FIFA-points, and
  population prior.
- **Market:** de-vigged 2026 outright prices from Polymarket, Kalshi, and a
  bookmaker consensus source, with historical bookmaker outrights used for
  2014/2018/2022 backtests.

The shipped forecast is a constrained four-expert linear opinion pool fit by
historical match-level Ranked Probability Score (RPS), then applied to a
100,000-draw 2026 tournament simulation and evaluated with leakage-guarded
World Cup backtests.

## Final Run

The final run artifacts in this repository are from the June 10, 2026 run tagged
`20260610_191222`.

| Field | Value |
|---|---|
| Release tag | `gyan-v1.0-final` |
| Git commit in run records | `eee0a29` |
| Global seed | `20260611` |
| Configured input freeze | `2026-06-10T19:00:00Z` |
| Stage 3 market snapshot | `2026-06-10T19:12:08.427758+00:00` |
| Stage 2 simulations | 100,000 |
| Stage 3 feature-expert simulations | 100,000 |
| Stage 4 validation | pass |

The final run was started after the configured 19:00 UTC freeze time. The run
records therefore preserve the configured freeze field and the actual refreshed
source hashes/timestamps. The Stage 3 market vector is timestamped at
`2026-06-10T19:12:08.427758+00:00`; that timestamp and the raw-source SHA-256
hashes define the market state used by the final board.

## Final Artifacts

| Artifact | Path |
|---|---|
| Final prediction board | `artifacts/tables/gyan_2026_predictions_20260610_191222.csv` |
| Latest forecast alias | `artifacts/tables/gyan_forecast_2026_latest.csv` |
| Full modal bracket table | `artifacts/tables/modal_bracket_2026_20260610_191222.csv` |
| Backtest metrics | `artifacts/tables/backtest_metrics_20260610_191222.csv` |
| Ablation table | `artifacts/tables/ablation_20260610_191222.csv` |
| Benchmark comparison | `artifacts/tables/benchmark_2026_20260610_191222.csv` |
| Stage 4 validation | `artifacts/reports/stage4_output_validation_20260610_191222.json` |
| Stage 3 run record | `artifacts/reports/run_stage3_build_ensemble_20260610_191212.json` |
| Stage 4 run record | `artifacts/reports/run_stage4_evaluation_20260610_191224.json` |
| Paper draft | `paper/manuscript.md` |
| Reproducibility appendix | `paper/reproducibility.md` |

Paper-ready figures are in `paper/figures/`; committed final tables and reports
are in `artifacts/`.

## Final Champion Board

| Rank | Team | Champion probability |
|---:|---|---:|
| 1 | Spain | 13.20% |
| 2 | France | 11.22% |
| 3 | England | 9.21% |
| 4 | Argentina | 8.64% |
| 5 | Portugal | 7.71% |
| 6 | Brazil | 7.68% |
| 7 | Germany | 4.98% |
| 8 | Netherlands | 3.67% |
| 9 | Colombia | 1.95% |
| 10 | Morocco | 1.92% |
| 11 | Mexico | 1.90% |
| 12 | Belgium | 1.78% |

The final board has 48 unique teams, probabilities in `[0, 1]`, monotone stage
probabilities, and champion probability sum `0.9999999999999978`.

## Modal Bracket

The modal bracket is a deterministic slot-resolved chalk path. It is not one
sampled Monte Carlo tournament and it does not imply exact match scorelines.
Group placements are ranked by final `p_champion` first and `p_reach_R32` as a
tie-breaker; knockout winners are selected by higher `p_champion`.

Final modal path:

| Stage | Match | Winner | Loser |
|---|---|---|---|
| SF | France vs Spain | Spain | France |
| SF | England vs Argentina | England | Argentina |
| Third place | France vs Argentina | France | Argentina |
| Final | Spain vs England | Spain | England |

The full Round-of-32 through final table is
`artifacts/tables/modal_bracket_2026_20260610_191222.csv`. It includes each
match's two teams, winner, loser, and both teams' final champion probabilities.

The deterministic group placements implied by that bracket are:

| Group | Winner | Runner-up | Third in modal ranking |
|---|---|---|---|
| A | Mexico | South Korea | Czech Republic* |
| B | Switzerland | Canada | Bosnia and Herzegovina* |
| C | Brazil | Morocco | Scotland |
| D | United States | Turkey | Paraguay* |
| E | Germany | Ecuador | Ivory Coast* |
| F | Netherlands | Japan | Sweden |
| G | Belgium | Iran | Egypt* |
| H | Spain | Uruguay | Cape Verde |
| I | France | Norway | Senegal* |
| J | Argentina | Austria | Algeria* |
| K | Portugal | Colombia | Democratic Republic of the Congo* |
| L | England | Croatia | Panama |

`*` marks a third-place team assigned to the modal Round of 32. These placements
are bracket construction choices, not saved probabilities of finishing first,
second, or third.

## Model Results

Final shipped weights:

| Expert | Weight |
|---|---:|
| Goal | 0.050 |
| Yield named | 0.050 |
| Socioeconomic | 0.244 |
| Market | 0.656 |

Stage 2 simulation diagnostics:

| Metric | Value |
|---|---:|
| Knockout upset rate | 35.23% |
| Group draw rate | 24.58% |
| Knockouts to extra time | 29.00% |
| Knockouts to penalties | 16.30% |
| Top-four max champion-probability movement from 50k to 100k | 0.165 percentage points |

Stage 1 engine validation:

| Metric | Value |
|---|---:|
| Held-out rows | 2,538 |
| Dixon-Coles RPS | 0.166905 |
| Plain-Poisson RPS | 0.166780 |
| Calibrated score-matrix RPS | 0.166704 |
| Observed draw rate | 23.72% |
| Calibrated predicted draw rate | 23.84% |

Stage 4 backtest summary:

| Tournament | GYAN mean match RPS | Champion log loss | Finalist hit rate | Semifinalist hit rate |
|---:|---:|---:|---:|---:|
| 2014 | 0.184989 | 2.235515 | 0.50 | 0.75 |
| 2018 | 0.192493 | 2.338752 | 0.00 | 0.25 |
| 2022 | 0.209626 | 2.221327 | 0.50 | 0.50 |

The shipped GYAN mean RPS across the three backtests is `0.195703`. The best
non-shipped ablation is `goal_yield_market` at `0.198709`, so the shipped
four-expert pool is better by `0.003006` RPS.

## Interpretation

The final board is market anchored by design. Market receives 65.6% of the
shipped weight, so agreement with the live market benchmark is an input-alignment
check rather than independent validation. The non-market evidence comes from
historical ablations, benchmark comparison to Goldman/PELE where available, and
the documented leakage and source audits.

The 5% minimum expert weight is a structural constraint that keeps all four
experts represented. In the final constrained fit, Goal and Yield named bind at
that floor. Their 5% weights are retained structural components, not evidence
that the unconstrained historical RPS objective selected exactly 5% for those
experts.

The Market expert differs by evaluation surface. Historical market backtests use
bookmaker champion outrights converted to strength ratings and match W/D/L
probabilities. The 2026 live Market expert uses the blended Polymarket, Kalshi,
and bookmaker champion vector directly. Non-champion Market stage probabilities
are scaled from the Stage 2 engine path shape, so market independence is
strongest at the champion-probability level.

## Data Sources Used

Raw source files are cached under `data/raw/` and hashed in run records. The
table below lists the sources used or cached by the final pipeline.

| ID | Source | Source location | Local cache | Role in final pipeline |
|---|---|---|---|---|
| D1 | martj42 international results | `raw.githubusercontent.com/martj42/international_results` | `d1_martj42_results.csv`, `d1_martj42_shootouts.csv`, `d1_martj42_former_names.csv` | Historical match table, shootouts, team-name cleaning, backtests |
| D2 | World Football Elo Ratings | `eloratings.net` | `d2_eloratings_about.html`, `d2_eloratings_current_ratings.js`, `d2_eloratings_world.tsv`, `d2_eloratings_teams.tsv`, `d2_eloratings_en_teams.tsv` | Elo formula/reference data and rating spot checks |
| D2b | Elo GitHub mirrors | `github.com/demetriodor/Footbal-Elo-Ratings`, `github.com/JGravier/soccer-elo` | `d2b_demetriodor_football_elo_ratings.html`, `d2b_jgravier_soccer_elo.html` | Elo source audit and fallback context |
| D3 | FiveThirtyEight SPI international archive | Internet Archive copy of `projects.fivethirtyeight.com/soccer-api/international/spi_matches_intl.csv`; DataHub SPI rankings | `d3_spi_matches_intl.csv`, `d3_spi_global_rankings.csv` | External SPI benchmark and validation context |
| D3b | FiveThirtyEight World Cup prediction archive | `datahub.io/fivethirtyeight/world-cup-predictions` | `d3b_world_cup_predictions_datapackage.json`, `d3b_wc_20140609_140000.csv` | Historical forecast archive context |
| D4 | Transfermarkt-derived player and national-team data plus Transfermarkt page cache | R2-hosted Transfermarkt-derived CSVs and `transfermarkt.com` national-team pages | `d4_transfermarkt_players.csv.gz`, `d4_transfermarkt_national_teams.csv.gz`, `d4_transfermarkt_national_team_values.html`, `d4_transfermarkt_national_team_pages/*.html` | Squad player values and national-team squad page value audit |
| D5 | FIFA/Coca-Cola men's ranking | `inside.fifa.com/fifa-world-ranking/men`, `api.fifa.com/api/v3/rankings` | `d5_fifa_mens_ranking.html`, `d5_fifa_rankings_api.json` | FIFA-points feature for socioeconomic modeling |
| D6 | 2026 squad lists | Wikipedia 2026 squads page and ESPN all-team squad list | `d6_wikipedia_2026_world_cup_squads.html`, `d6_espn_2026_world_cup_squad_lists.html` | Named 2026 squad parsing and cross-check |
| D7 | 2026 World Cup groups, schedule, venues, and knockout assignments | Wikipedia tournament/knockout pages, FIFA tournament page, FIFA schedule PDF | `d7_wikipedia_2026_world_cup.html`, `d7_fifa_official_2026_world_cup.html`, `d7_fifa_official_2026_match_schedule.pdf`, `d7_wikipedia_2026_world_cup_knockout_stage.html` | Groups, schedule, venues, Round-of-32 bracket, Annex C third-place assignments |
| D8 | Hoffmann, Ging, and Ramasamy 2002 | `redalyc.org/pdf/103/10305205.pdf` | `d8_hoffmann_ging_ramasamy_2002.pdf` | Socioeconomic model specification and coefficients |
| D9 | Goldman Sachs 2026 World Cup report | cached PDF from `static.poder360.com.br` | `d9_goldman_2026_world_cup_report.pdf` | Published benchmark champion probabilities |
| D10 | Klement / Panmure Liberum 2026 report | `panmureliberum.com/media/3179/strs_1031724.pdf` | `d10_klement_panmure_liberum_2026.pdf` | Published socioeconomic/benchmark context |
| D11 | World Bank GDP per capita PPP and population APIs | `api.worldbank.org` indicators `NY.GDP.PCAP.PP.CD` and `SP.POP.TOTL` | `d11_world_bank_gdp_per_capita_ppp.json`, `d11_world_bank_population.json` | Macro features for socioeconomic model and historical snapshots |
| D12 | Climate Knowledge Portal and country-average temperature table | `worldbank.github.io/climateknowledgeportal`, Wikipedia country-average temperature table | `d12_world_bank_climate_data_collections_readme.html`, `d12_country_average_yearly_temperature_wikipedia.html` | Climate/temperature deviation feature |
| D13 | Polymarket Gamma API, World Cup winner event | `gamma-api.polymarket.com/events/slug/world-cup-winner` | `d13_polymarket_world_cup_winner_event.json` | Live 2026 Market expert champion vector |
| D14 | Kalshi public markets API, men's World Cup winner markets | `api.elections.kalshi.com/trade-api/v2/markets?event_ticker=KXMENWORLDCUP-26` | `d14_kalshi_mens_world_cup_winner_markets.json` | Live 2026 Market expert champion vector |
| D15 | Bookmaker odds and historical bookmaker outright boards | `bookmakersreview.com/fifa-world-cup/`; cached historical Zeileis/Leitner/Hornik and SportStatist/William Hill sources | `d15_bookmakersreview_world_cup.html`, `d15_historical_world_cup_outrights.csv`, historical PDF/HTML caches in `data/raw/` | Live bookmaker component, historical Market backtests, benchmark context |
| Manual | Injury/absence tracker | local curated CSV | `injuries_2026.csv` | Injury and absence adjustments for Yield named |

Stage 3 used raw live market files with these SHA-256 prefixes:

| Source | Path | SHA-256 prefix |
|---|---|---:|
| Polymarket | `data/raw/d13_polymarket_world_cup_winner_event.json` | `614f55ef8e6c` |
| Kalshi | `data/raw/d14_kalshi_mens_world_cup_winner_markets.json` | `943a2dd420ea` |
| Bookmaker live | `data/raw/d15_bookmakersreview_world_cup.html` | `f61be0e8da63` |
| Historical outrights | `data/raw/d15_historical_world_cup_outrights.csv` | `d20037fb79fd` |

## Run Records

| Stage | Script | Final run record |
|---|---|---|
| Stage 1 download | `scripts/s1_download_data.py` | `artifacts/reports/run_stage1_download_data_20260610_190556.json` |
| Stage 1 clean matches | `scripts/s1_clean_matches.py` | `artifacts/reports/run_stage1_clean_matches_20260610_190601.json` |
| Stage 1 Elo | `scripts/s1_build_elo.py` | `artifacts/reports/run_stage1_build_elo_20260610_190607.json` |
| Stage 1 socioeconomic | `scripts/s1_build_socioeconomic.py` | `artifacts/reports/run_stage1_build_socioeconomic_20260610_190614.json` |
| Stage 1 squad value | `scripts/s1_build_squad_value.py` | `artifacts/reports/run_stage1_build_squad_value_20260610_190619.json` |
| Stage 1 goal fit | `scripts/s1_fit_dixon_coles.py` | `artifacts/reports/run_stage1_fit_dixon_coles_20260610_190809.json` |
| Stage 1 validation | `scripts/s1_validate_engine.py` | `artifacts/reports/run_stage1_validate_engine_20260610_190918.json` |
| Stage 2 structure | `scripts/s2_build_structure.py` | `artifacts/reports/run_stage2_build_structure_20260610_190929.json` |
| Stage 2 simulation | `scripts/s2_run_simulation.py --n-sims 100000` | `artifacts/reports/run_stage2_run_simulation_20260610_191007.json` |
| Stage 3 ensemble | `scripts/s3_build_ensemble.py --n-sims 100000` | `artifacts/reports/run_stage3_build_ensemble_20260610_191212.json` |
| Stage 4 evaluation | `scripts/s4_backtest.py` | `artifacts/reports/run_stage4_evaluation_20260610_191224.json` |

## Repository Layout

```text
gyan-wc-model/
|-- data/raw/          # raw source cache
|-- data/interim/      # cleaned intermediate data
|-- data/processed/    # model-ready feature and structure tables
|-- artifacts/tables/  # committed final CSV artifacts
|-- artifacts/reports/ # committed final run records and validation
|-- paper/             # manuscript, reproducibility appendix, paper figures
|-- scripts/           # stage entry points
|-- src/gyan/          # model package
`-- tests/             # pytest suite
```

## Environment

- Python 3.12+
- Dependencies pinned in `requirements.txt`
- Main runtime libraries: NumPy, pandas, pyarrow, scipy, statsmodels, joblib,
  httpx, beautifulsoup4, lxml, matplotlib, seaborn
- Tests: pytest and pytest-cov

The final run records include package versions, input hashes, output hashes,
`GLOBAL_SEED`, git commit, and release tag metadata.

## Disclaimer

**This project is for educational and entertainment purposes only.**

GYAN outputs are probabilistic model estimates, not statements of fact, betting
advice, financial advice, or guarantees about future sporting outcomes. The
model, data, assumptions, and market inputs can be incomplete, stale, biased, or
wrong. Do not use this repository or its forecasts as the basis for wagering,
investment decisions, or any other high-stakes decision.
