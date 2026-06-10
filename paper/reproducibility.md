# Reproducibility

- `GLOBAL_SEED`: `20260611`
- Git commit at Stage 4 run: `0deecdd`
- Expected release tag for the final paper run: `gyan-v1.0-final`
- Configured input freeze field in run records: `2026-06-10T19:00:00Z`
- Refreshed final run note: the `20260610_223240` artifact set intentionally supersedes the earlier freeze by force-pulling the public raw inputs on June 10, 2026; the live market snapshot is `2026-06-10T22:32:31.425764+00:00`.
- Requirements file: `./requirements.txt`
- Run order: `scripts/s1_download_data.py --force`, `scripts/s1_clean_matches.py`, `scripts/s1_build_elo.py`, `scripts/s1_build_socioeconomic.py`, `scripts/s1_build_squad_value.py --force-team-pages`, `scripts/s1_fit_dixon_coles.py`, `scripts/s1_validate_engine.py`, `scripts/s2_build_structure.py`, `scripts/s2_run_simulation.py --n-sims 100000`, `scripts/s3_build_ensemble.py --n-sims 100000`, `scripts/s4_backtest.py`
- Final board: `./outputs/tables/gyan_2026_predictions_20260610_223240.csv`
- Backtest metrics: `./outputs/tables/backtest_metrics_20260610_223240.csv`
- Ablation: `./outputs/tables/ablation_20260610_223240.csv`
- Benchmark: `./outputs/tables/benchmark_2026_20260610_223240.csv`

Re-run Stage 4 with a fixed seed to regenerate the deterministic tables and figures. Earlier Monte Carlo stages should match within their documented simulation tolerance, or exactly when the same seed and worker partition are used.
