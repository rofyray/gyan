# GYAN World Cup Model

## Introduction
GYAN is an ensemble forecasting model for the 2026 FIFA World Cup. It combines a goal model, a squad-yield signal, socioeconomic structure, and market information where reproducible sources are available.

## Data
The project follows the D1..D15 registry in `PRD/CONVENTIONS.md`: international results, Elo references, SPI/forecast archives, Transfermarkt squad values, FIFA ranking data, World Bank macro data, climate proxies, published benchmark PDFs, and market sources.

## Methods
The Goal expert uses the Stage 1 Poisson/Dixon-Coles mean engine with a draw-calibrated correlated negative-binomial score matrix inside the Stage 2 tournament simulator. Yield uses team-value and named-squad adjustments for 2026; in historical backtests it is represented by pre-tournament form proxies because historical named squads and values are not cached. The socioeconomic expert uses the Hoffmann specification with year-available World Bank GDP/population data. The Market expert uses live de-vigged 2026 outrights for the current board and cached D15 bookmaker outrights for historical calibration. Expert pooling uses the Stage 3 shipped four-expert weights `{'goal': 0.05, 'yield_named': 0.05, 'socioeconomic': 0.243701504486925, 'market': 0.656298495513075}`, fit by constrained historical RPS with a positive minimum weight for every expert. The weight objective is match-level mean RPS over the 2014, 2018, 2022 final tournaments; in the 32-team historical format, 144 of 192 matches (75%) are group-stage games. Those weights are then applied to a 2026 tournament board whose champion probabilities depend heavily on knockout dynamics, bracket paths, and variance, so the high Market/strength weight can partly reflect the group-stage-heavy calibration target. The final paper input freeze timestamp is `2026-06-10T19:00:00Z`; the final paper run should use no inputs after that cutoff.

## Results
Backtest metrics are in `./outputs/tables/backtest_metrics_20260609_022619.csv`. The ablation table is `./outputs/tables/ablation_20260609_022619.csv`. The benchmark table is `./outputs/tables/benchmark_2026_20260609_022619.csv`. The final 2026 board is `./outputs/tables/gyan_2026_predictions_20260609_022619.csv`.

Top 2026 teams: [{'team': 'Spain', 'p_champion': 0.1302863044219064}, {'team': 'France', 'p_champion': 0.113191818175089}, {'team': 'England', 'p_champion': 0.0917387252596562}, {'team': 'Argentina', 'p_champion': 0.08505051317103}, {'team': 'Brazil', 'p_champion': 0.0759299322797848}, {'team': 'Portugal', 'p_champion': 0.0751249793092282}, {'team': 'Germany', 'p_champion': 0.0507743015637632}, {'team': 'Netherlands', 'p_champion': 0.0371319484047309}, {'team': 'Morocco', 'p_champion': 0.0200692532321625}, {'team': 'Mexico', 'p_champion': 0.0192294085684982}].

Shipped-vs-ablation check: {'shipped_gyan_mean_rps': 0.1957014978400681, 'best_non_shipped_ablation': 'goal_yield_market', 'best_non_shipped_mean_rps': 0.198702991193038, 'delta_shipped_minus_best_non_shipped': -0.0030014933529698973, 'blocker_resolved': True}.

Top-team divergence check: {'threshold': 0.08, 'comparisons': [{'team': 'Spain', 'gyan_p_champion': 0.1302863044219064, 'goldman_p_champion': 0.257, 'market_p_champion': 0.1703834667500977, 'abs_delta_vs_goldman': 0.12671369557809362, 'abs_delta_vs_market': 0.0400971623281913, 'diverges_from_both': False}, {'team': 'France', 'gyan_p_champion': 0.113191818175089, 'goldman_p_champion': 0.189, 'market_p_champion': 0.1511945444582387, 'abs_delta_vs_goldman': 0.075808181824911, 'abs_delta_vs_market': 0.03800272628314971, 'diverges_from_both': False}, {'team': 'England', 'gyan_p_champion': 0.0917387252596562, 'goldman_p_champion': 0.05, 'market_p_champion': 0.119757634973977, 'abs_delta_vs_goldman': 0.041738725259656204, 'abs_delta_vs_market': 0.02801890971432079, 'diverges_from_both': False}, {'team': 'Argentina', 'gyan_p_champion': 0.08505051317103, 'goldman_p_champion': 0.143, 'market_p_champion': 0.0956260229374502, 'abs_delta_vs_goldman': 0.057949486828969984, 'abs_delta_vs_market': 0.010575509766420202, 'diverges_from_both': False}, {'team': 'Brazil', 'gyan_p_champion': 0.0759299322797848, 'goldman_p_champion': 0.076, 'market_p_champion': 0.0926556559287976, 'abs_delta_vs_goldman': 7.006772021520002e-05, 'abs_delta_vs_market': 0.0167257236490128, 'diverges_from_both': False}], 'tripped': False, 'tripped_teams': [], 'france': {'team': 'France', 'gyan_p_champion': 0.113191818175089, 'goldman_p_champion': 0.189, 'market_p_champion': 0.1511945444582387, 'abs_delta_vs_goldman': 0.075808181824911, 'abs_delta_vs_market': 0.03800272628314971, 'diverges_from_both': False}, 'france_blocker_resolved': True}.

## Discussion
The Stage 3 shipped model is now a four-expert GYAN pool rather than an engine-only forecast. Match-level validation without market outrights still favours the Goal expert, so that diagnostic remains reported separately; the shipped tournament board uses the historical no-leakage four-expert calibration because it can include the market expert.

The 5% minimum expert weight is a deliberate implementation addition to keep all four experts represented for structure and interpretability; it was not part of the original PRD optimization spec. In the current fit it binds for ['goal', 'yield_named'], with Goal at 5.0% and Yield at 5.0%. The board is therefore primarily a Socioeconomic-plus-Market blend (90.0% combined), and the in-sample RPS cost versus the unconstrained selected-pool optimum is 0.000183806508698231. Leave-one-tournament-out RPS is 0.198501 for the optimized weights versus 0.202661 for equal weights.

The named-squad and injury-aware Yield expert is a methodological contribution, but it should not be described as the main driver of the current board. Its value is evaluated as a marginal design improvement through the yield-named versus yield-nominal ablation table, not as the dominant source of forecast mass.

Because the Market expert receives most of the shipped weight, agreement between GYAN and the live market benchmark is expected and should not be presented as independent validation. The independent-value evidence comes from the Goldman comparison, historical ablations, and the documented source/weight limitations.

The Socioeconomic and Market champion boards are not highly correlated in the Stage 3 diagnostic (Pearson 0.642, Spearman 0.680, top-12 overlap 6/12), which gives some support to expert diversity while still leaving Market as the dominant signal.

## Limitations
Forecasts are probabilities, not claims of fact. Stage 4 uses cached D15 historical outright boards for the backtest market expert: 2014 and 2018 use published bookmaker-consensus probabilities, and 2022 uses a pre-tournament William Hill decimal-odds board dated 2022-09-01. Polymarket's 2022 World Cup winner event is cached for audit, but its public CLOB/trade APIs did not expose a complete pre-kickoff price vector.

The final input freeze at 2026-06-10T19:00:00Z locks the squad and injury snapshot for reproducibility. That cutoff captures teams playing on June 11 and June 12, but teams whose first match is June 13 or later may still make permitted replacement changes inside their own 24-hour pre-match window. Those post-freeze squad or injury changes are an unavoidable limitation of a reproducible pre-opening forecast.

Market construction differs by evaluation surface. Historically, the Market expert has no raw match W/D/L prices: bookmaker champion outrights are normalized, converted into log-probability ratings, and then passed through the calibrated rating-to-W/D/L score model for match-level RPS. For champion scoring, those historical outright probabilities are used directly. For 2026, the live Market expert uses the blended Polymarket/Kalshi/bookmaker champion vector directly, while non-champion stages are scaled from the Stage 2 engine path. Thus the Market weight transfers cleanly at the champion-probability primitive but should be described as a market-strength hybrid for match-level fitting, not as weight on raw match prices.

The live-vs-proxy market-source divergence check did not exceed the sharp-divergence threshold; the largest champion-probability movement was 0.000 for Netherlands.

The Market expert directly prices champion probabilities, but its non-champion stage columns are reconstructed from the Stage 2 engine path shape. Therefore R32/R16/QF/SF/final Market probabilities are partly engine-derived, and the Market expert's independence is strongest only at the champion stage.

## Reproducibility Appendix
See `paper/reproducibility.md`.
