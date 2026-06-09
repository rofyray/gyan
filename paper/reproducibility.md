# Reproducibility

- `GLOBAL_SEED`: `20260611`
- Git commit at Stage 4 run: `unknown`
- Expected release tag for the final paper run: `gyan-v1.0-final`
- Final input freeze timestamp UTC: `2026-06-10T19:00:00Z`
- No-post-freeze-input statement: final paper records should use no inputs after `2026-06-10T19:00:00Z`.
- Requirements file: `./requirements.txt`
- Run order: `scripts/s1_download_data.py`, `scripts/s1_clean_matches.py`, `scripts/s1_build_elo.py`, `scripts/s1_build_socioeconomic.py`, `scripts/s1_build_squad_value.py`, `scripts/s1_fit_dixon_coles.py`, `scripts/s1_validate_engine.py`, `scripts/s2_build_structure.py`, `scripts/s2_run_simulation.py`, `scripts/s3_build_ensemble.py`, `scripts/s4_backtest.py`
- Final board: `./outputs/tables/gyan_2026_predictions_20260609_022619.csv`
- Backtest metrics: `./outputs/tables/backtest_metrics_20260609_022619.csv`
- Ablation: `./outputs/tables/ablation_20260609_022619.csv`
- Benchmark: `./outputs/tables/benchmark_2026_20260609_022619.csv`

Re-run Stage 4 with a fixed seed to regenerate the deterministic tables and figures. Earlier Monte Carlo stages should match within their documented simulation tolerance, or exactly when the same seed and worker partition are used.
