# GYAN World Cup Model

## Introduction

GYAN is a four-expert ensemble forecast for the 2026 FIFA World Cup. The final
forecast combines a goal model, named-squad value information, socioeconomic
structure, and live market-implied champion probabilities into a 48-team
tournament board.

The final board is the June 10, 2026 run tagged `20260610_191222`. It was run at
git commit `eee0a29` with release tag `gyan-v1.0-final` and `GLOBAL_SEED`
`20260611`. The configured freeze timestamp is `2026-06-10T19:00:00Z`; the
actual Stage 3 market snapshot in the final run is
`2026-06-10T19:12:08.427758+00:00`.

## Data

The model uses the D1-D15 data registry implemented in `src/gyan/data/*.py` and
the live market loaders in `src/gyan/ensemble/market.py`. Historical match
results come from martj42 international results. Elo reference material comes
from World Football Elo Ratings and cached mirror pages. SPI and archived World
Cup forecast data come from FiveThirtyEight/DataHub/Internet Archive caches.
Squads and market values come from Wikipedia, ESPN, Transfermarkt-derived player
and national-team datasets, cached Transfermarkt squad pages, and the local
injury tracker `data/raw/injuries_2026.csv`. Socioeconomic inputs use FIFA
ranking points, World Bank GDP per capita PPP and population, country-average
temperature data, and the Hoffmann/Ging/Ramasamy specification. Benchmark inputs
include Goldman Sachs and Klement/Panmure Liberum PDFs. The live 2026 Market
expert blends Polymarket, Kalshi, and bookmaker outright sources; historical
market calibration uses cached D15 bookmaker outright boards for 2014, 2018, and
2022.

The final source inventory and run-record hashes are reported in
`paper/reproducibility.md` and `README.md`.

## Methods

The Goal expert uses a fitted international score model. The final validation
selected plain-Poisson means rather than Dixon-Coles dependence because
Dixon-Coles tuning did not improve held-out RPS. The shipped score matrix is a
draw-calibrated correlated negative-binomial distribution with dispersion `8.0`;
this calibration brought the predicted draw rate to 23.84% against an observed
23.72% in the held-out validation slice.

The Yield named expert converts 2026 named squads into team strength through
player market values, UEFA value discounting, age weighting, and injury/absence
adjustments. In historical backtests, where comparable named-squad and player
value snapshots are not cached, the Yield family is represented by pre-tournament
form proxies.

The Socioeconomic expert uses macroeconomic and football-development structure:
GDP per capita PPP, population, temperature deviation, FIFA ranking points, and
the Hoffmann-style prior specification.

The Market expert uses different raw data by task. For the 2026 board, it uses a
de-vigged blend of Polymarket, Kalshi, and bookmaker champion probabilities. For
historical match-level backtests, bookmaker champion outrights are converted into
market-strength ratings and then into W/D/L probabilities through the calibrated
rating-to-score model. Historical champion scoring uses the outright champion
probabilities directly.

The shipped ensemble is a constrained linear opinion pool fit on 192 historical
World Cup matches from 2014, 2018, and 2022. The objective is mean match-level
RPS. The selected weights are:

| Expert | Weight |
|---|---:|
| Goal | 0.050 |
| Yield named | 0.050 |
| Socioeconomic | 0.244 |
| Market | 0.656 |

The 5% lower bound is an explicit structural floor to keep all four experts
represented. In the final fit, Goal and Yield named bind at that floor. The
optimized pool's leave-one-tournament-out RPS is `0.198503`, compared with
`0.202667` for equal weights.

## Results

The committed copy of the final 2026 board is
`artifacts/tables/gyan_2026_predictions_20260610_191222.csv`.

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

The committed copy of the final modal bracket is
`artifacts/tables/modal_bracket_2026_20260610_191222.csv`. It is a deterministic
chalk path that selects group placements and knockout winners by final champion
probability, not a sampled tournament trace. Its final is Spain over England,
with France over Argentina in the third-place match.

Stage 2 aggregate simulation diagnostics from 100,000 draws:

| Metric | Value |
|---|---:|
| Knockout upset rate | 35.23% |
| Group draw rate | 24.58% |
| Knockouts to extra time | 29.00% |
| Knockouts to penalties | 16.30% |

Backtest metrics for the shipped GYAN pool:

| Tournament | Mean match RPS | Champion log loss | Finalist hit rate | Semifinalist hit rate |
|---:|---:|---:|---:|---:|
| 2014 | 0.184989 | 2.235515 | 0.50 | 0.75 |
| 2018 | 0.192493 | 2.338752 | 0.00 | 0.25 |
| 2022 | 0.209626 | 2.221327 | 0.50 | 0.50 |

Across the three historical backtests, shipped GYAN has mean RPS `0.195703`.
The best non-shipped ablation is `goal_yield_market` at `0.198709`, so the
shipped four-expert pool improves on that ablation by `0.003006` RPS.

Top-team divergence did not trip the sharp T-G4 threshold. Spain is materially
lower than Goldman Sachs' 25.7% benchmark but remains within the live market
comparison threshold; France, England, Argentina, and Brazil also do not diverge
sharply from both Goldman and market at the same time.

## Discussion

The final board is best interpreted as a market-anchored, historically calibrated
four-expert pool. Because Market receives most of the shipped weight, agreement
with the live market benchmark is expected and is not independent validation.
The independent evidence for the model is the
leakage-guarded historical evaluation, ablation performance, source transparency,
and benchmark divergence diagnostics.

The named-squad and injury-aware Yield expert is a methodological contribution,
but it is not the dominant driver of the final board. Its shipped 5% weight is
the positive expert floor. Its substantive value is evaluated through the
named-versus-nominal Yield diagnostics and team-level movement tables.

The historical weight objective is match-level RPS over 2014/2018/2022. In those
32-team tournaments, 144 of 192 calibration matches are group-stage games. The
2026 deployment target is a 48-team tournament board whose champion probabilities
are sensitive to knockout paths, extra time, penalties, and bracket variance.
This calibration/deployment mismatch is a central limitation and helps explain
why broad market/strength signals receive large weight.

## Limitations

Forecasts are probabilities, not claims of fact. The model does not persist exact
group-stage scoreline predictions or sampled upset storylines in the final
artifacts. The saved modal bracket is deterministic and is a chalk path.

Historical market data are bookmaker-outright based. Polymarket's 2022 World Cup
winner event is cached for audit context, but the public APIs did not expose a
complete pre-kickoff team vector suitable for the same historical calibration.
Kalshi World Cup winner markets begin with the 2026 cycle and therefore cannot
support historical World Cup backtests.

The 2026 live Market expert prices champion probability directly. Its R32, R16,
QF, SF, and final probabilities are reconstructed by scaling the Stage 2 engine
path shape, so those intermediate stage probabilities are partly engine-derived.
The Market expert's strongest independent information is at the champion stage.

The configured input freeze is `2026-06-10T19:00:00Z`, while the final Stage 3
market snapshot is `2026-06-10T19:12:08.427758+00:00`. The final run is therefore
best identified by its run-record hashes and market snapshot time. Later squad
or injury replacements after the captured raw files are not reflected.

## Reproducibility

The reproducibility appendix is `paper/reproducibility.md`. It lists the final
run records, source inventory, key input hashes, simulation counts, seed, commit,
release tag, and output artifact paths.
