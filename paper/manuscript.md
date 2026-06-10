# GYAN World Cup Model

## Introduction
GYAN is an ensemble forecasting model for the 2026 FIFA World Cup. It combines a goal model, a squad-yield signal, socioeconomic structure, and market information where reproducible sources are available.

## Data
The project follows the D1..D15 registry in `PRD/CONVENTIONS.md`: international results, Elo references, SPI/forecast archives, Transfermarkt squad values, FIFA ranking data, World Bank macro data, climate proxies, published benchmark PDFs, and market sources.

## Methods
The Goal expert uses the Stage 1 Poisson/Dixon-Coles mean engine with a draw-calibrated correlated negative-binomial score matrix inside the Stage 2 tournament simulator. Yield uses team-value and named-squad adjustments for 2026; in historical backtests it is represented by pre-tournament form proxies because historical named squads and values are not cached. The socioeconomic expert uses the Hoffmann specification with year-available World Bank GDP/population data. The Market expert uses live de-vigged 2026 outrights for the current board and cached D15 bookmaker outrights for historical calibration. Expert pooling uses the Stage 3 shipped four-expert weights (Goal 0.050, Yield named 0.050, Socioeconomic 0.244, Market 0.656), fit by constrained historical RPS with a positive minimum weight for every expert. The weight objective is match-level mean RPS over the 2014, 2018, 2022 final tournaments; in the 32-team historical format, 144 of 192 matches (75%) are group-stage games. Those weights are then applied to a 2026 tournament board whose champion probabilities depend heavily on knockout dynamics, bracket paths, and variance, so the high Market/strength weight can partly reflect the group-stage-heavy calibration target. The refreshed final artifact set is `20260610_223240`; it force-pulled public raw inputs on June 10, 2026 and uses the live market snapshot timestamp `2026-06-10T22:32:31.425764+00:00`.

## Results
Backtest metrics are in `./outputs/tables/backtest_metrics_20260610_223240.csv`. The ablation table is `./outputs/tables/ablation_20260610_223240.csv`. The benchmark table is `./outputs/tables/benchmark_2026_20260610_223240.csv`. The final 2026 board is `./outputs/tables/gyan_2026_predictions_20260610_223240.csv`.

The top 2026 champion probabilities are:

| Rank | Team | Champion probability |
|---:|---|---:|
| 1 | Spain | 13.23% |
| 2 | France | 11.20% |
| 3 | England | 9.19% |
| 4 | Argentina | 8.67% |
| 5 | Portugal | 7.69% |
| 6 | Brazil | 7.67% |
| 7 | Germany | 4.97% |
| 8 | Netherlands | 3.74% |
| 9 | Colombia | 1.95% |
| 10 | Mexico | 1.92% |

The shipped GYAN board has mean historical RPS 0.195703. The best non-shipped ablation is `goal_yield_market` at 0.198709, so the shipped specification improves RPS by 0.003006.

The top-team divergence check does not trip the 0.08 threshold against both Goldman and market comparators. Spain differs from Goldman by 0.1247 but from the market by 0.0424; France differs from Goldman by 0.0770 and from the market by 0.0376. England, Argentina, and Brazil are also within the two-comparator threshold.

## Discussion
The Stage 3 shipped model is now a four-expert GYAN pool rather than an engine-only forecast. Match-level validation without market outrights still favours the Goal expert, so that diagnostic remains reported separately; the shipped tournament board uses the historical no-leakage four-expert calibration because it can include the market expert.

The 5% minimum expert weight is a deliberate implementation addition to keep all four experts represented for structure and interpretability; it was not part of the original PRD optimization spec. In the current fit it binds for Goal and Yield, both at 5.0%. The board is therefore primarily a Socioeconomic-plus-Market blend (90.0% combined), and the in-sample RPS cost versus the unconstrained selected-pool optimum is 0.000184. Leave-one-tournament-out RPS is 0.198503 for the optimized weights versus 0.202667 for equal weights.

The named-squad and injury-aware Yield expert is a methodological contribution, but it should not be described as the main driver of the current board. Its value is evaluated as a marginal design improvement through the yield-named versus yield-nominal ablation table, not as the dominant source of forecast mass.

Because the Market expert receives most of the shipped weight, agreement between GYAN and the live market benchmark is expected and should not be presented as independent validation. The independent-value evidence comes from the Goldman comparison, historical ablations, and the documented source/weight limitations.

The Socioeconomic and Market champion boards are not highly correlated in the Stage 3 diagnostic (Pearson 0.639, Spearman 0.697, top-12 overlap 7/12), which gives some support to expert diversity while still leaving Market as the dominant signal.

## Limitations
Forecasts are probabilities, not claims of fact. Stage 4 uses cached D15 historical outright boards for the backtest market expert: 2014 and 2018 use published bookmaker-consensus probabilities, and 2022 uses a pre-tournament William Hill decimal-odds board dated 2022-09-01. Polymarket's 2022 World Cup winner event is cached for audit, but its public CLOB/trade APIs did not expose a complete pre-kickoff price vector.

The refreshed final run locks the squad, injury, Transfermarkt, market, and other raw-source hashes as of the June 10, 2026 force-refresh pass. Teams whose first match is June 13 or later may still make permitted replacement changes inside their own 24-hour pre-match window. Those post-refresh squad or injury changes are an unavoidable limitation of a reproducible pre-opening forecast.

Market construction differs by evaluation surface. Historically, the Market expert has no raw match W/D/L prices: bookmaker champion outrights are normalized, converted into log-probability ratings, and then passed through the calibrated rating-to-W/D/L score model for match-level RPS. For champion scoring, those historical outright probabilities are used directly. For 2026, the live Market expert uses the blended Polymarket/Kalshi/bookmaker champion vector directly, while non-champion stages are scaled from the Stage 2 engine path. Thus the Market weight transfers cleanly at the champion-probability primitive but should be described as a market-strength hybrid for match-level fitting, not as weight on raw match prices.

The live-vs-proxy market-source divergence check did not exceed the sharp-divergence threshold; the largest champion-probability movement was 0.001 for Netherlands.

The Market expert directly prices champion probabilities, but its non-champion stage columns are reconstructed from the Stage 2 engine path shape. Therefore R32/R16/QF/SF/final Market probabilities are partly engine-derived, and the Market expert's independence is strongest only at the champion stage.

## Reproducibility Appendix
See `paper/reproducibility.md`.
