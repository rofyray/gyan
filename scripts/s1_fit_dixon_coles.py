"""Fit the Stage 1.7 Dixon-Coles goal model and write parameter artifacts."""

from __future__ import annotations  # consistent modern type hints

import os  # set matplotlib cache path before importing pyplot
from datetime import datetime, timezone  # timestamp parameter filenames
from pathlib import Path  # file paths for artifacts

os.environ.setdefault("MPLBACKEND", "Agg")  # headless non-GUI plotting backend
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-gyan")  # writable matplotlib cache
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")  # writable fontconfig cache root

import matplotlib.pyplot as plt  # noqa: E402  # plotting after MPLCONFIGDIR is set
import pandas as pd  # noqa: E402  # load feature and audit tables

from gyan.config import (  # central paths and model constants
    DATA_PROCESSED,
    DIXON_COLES_PARAMS_DC_FILE,
    DIXON_COLES_MAX_TRAIN_MATCHES,
    DIXON_COLES_MAXITER,
    DIXON_COLES_PARAMS_PLAIN_FILE,
    DIXON_COLES_RECENT_MIN_DATE,
    DIXON_COLES_RIDGE,
    DIXON_COLES_XI,
    ELO_CURRENT_RATINGS_FILE,
    GLOBAL_SEED,
    MATCHES_WITH_ELO_FILE,
    OUTPUTS_FIGURES,
    create_directories,
)
from gyan.engine.dixon_coles import fit_dixon_coles  # maximum-likelihood fit helper
from gyan.utils.logging import RunRecord, get_run_logger  # required logs and run records


def _timestamp() -> str:
    """Return a compact UTC timestamp for output filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")  # sortable timestamp


def _training_subset(matches: pd.DataFrame) -> pd.DataFrame:
    """Return the modern-era capped training subset for Stage 1.7 fitting."""
    recent = matches[pd.to_datetime(matches["date"]) >= pd.Timestamp(DIXON_COLES_RECENT_MIN_DATE)].copy()  # modern
    if len(recent) > DIXON_COLES_MAX_TRAIN_MATCHES:  # cap very large optimisation tables
        recent = recent.tail(DIXON_COLES_MAX_TRAIN_MATCHES).copy()  # keep the most recent rows
    return recent.reset_index(drop=True)  # stable index for fitting


def _plot_attack_defense(params_path: Path, elo_path: Path, figure_png: Path, figure_pdf: Path, caption_path: Path) -> None:
    """Write the top-20 attack-vs-defense scatter required by Stage 1 outputs."""
    from gyan.engine.dixon_coles import DixonColesModel  # local import avoids circular script scope

    model = DixonColesModel.load(params_path)  # load fitted model parameters
    elo = pd.read_csv(elo_path).sort_values("elo_rating", ascending=False).head(20)  # top 20 by Elo
    plot = pd.DataFrame(  # prepare plotting coordinates
        {
            "team": elo["team"],
            "attack": elo["team"].map(model.attack),
            "defense": elo["team"].map(model.defense),
        }
    ).dropna()  # drop teams absent from fitted subset
    figure_png.parent.mkdir(parents=True, exist_ok=True)  # ensure figure directory exists
    plt.figure(figsize=(8, 6))  # paper-friendly aspect
    plt.axhline(0.0, color="#999999", linewidth=0.8)  # reference attack/defense axes
    plt.axvline(0.0, color="#999999", linewidth=0.8)  # reference attack/defense axes
    plt.scatter(plot["attack"], plot["defense"], color="#2563eb", s=38)  # team points
    for row in plot.itertuples(index=False):  # annotate each plotted team
        plt.text(row.attack + 0.005, row.defense + 0.005, row.team, fontsize=8)  # label
    plt.title("Dixon-Coles Attack vs Defense, Top 20 Elo Teams")  # figure title
    plt.xlabel("Attack parameter (higher = more goals)")  # x-axis label
    plt.ylabel("Defense parameter (higher = suppresses goals)")  # y-axis label
    plt.tight_layout()  # avoid clipped labels
    plt.savefig(figure_png, dpi=300)  # review PNG
    plt.savefig(figure_pdf)  # vector PDF for paper
    plt.close()  # release figure memory
    caption_path.write_text(  # write required caption sidecar
        "Top-20 current Elo teams plotted by fitted Dixon-Coles attack and defense parameters. "
        "Defense is oriented so larger values reduce opponent goal means.\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run the Stage 1.7 Dixon-Coles fitting step."""
    create_directories()  # ensure output directories exist
    tag = _timestamp()  # shared output timestamp
    params_path = DATA_PROCESSED / f"dixon_coles_params_{tag}.json"  # parameter JSON path
    dc_params_path = DATA_PROCESSED / f"dixon_coles_params_dixon_coles_{tag}.json"  # DC candidate
    plain_params_path = DATA_PROCESSED / f"dixon_coles_params_plain_poisson_{tag}.json"  # no-rho candidate
    latest_params_path = DATA_PROCESSED / "dixon_coles_params_latest.json"  # stable latest alias
    figure_png = OUTPUTS_FIGURES / "attack_defense_top20.png"  # required PNG figure
    figure_pdf = OUTPUTS_FIGURES / "attack_defense_top20.pdf"  # required vector figure
    caption_path = OUTPUTS_FIGURES / "attack_defense_top20.txt"  # required caption
    logger, log_path = get_run_logger("s1_fit_dixon_coles", stage="stage1", step="fit_dixon_coles")  # log
    with RunRecord(  # create machine-readable audit record
        stage="stage1",
        step="fit_dixon_coles",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_input(MATCHES_WITH_ELO_FILE)  # hash Elo-enriched match table
        record.add_input(ELO_CURRENT_RATINGS_FILE)  # hash current Elo table used for figure
        record.add_output(log_path)  # record human-readable log path
        matches = pd.read_parquet(MATCHES_WITH_ELO_FILE)  # load Stage 1.4 feature table
        training = _training_subset(matches)  # select modern-era fit rows
        model, diagnostics = fit_dixon_coles(  # fit weighted Dixon-Coles candidate
            training,
            xi=DIXON_COLES_XI,
            fit_rho=True,
            maxiter=DIXON_COLES_MAXITER,
            ridge=DIXON_COLES_RIDGE,
        )
        plain_model, plain_diagnostics = fit_dixon_coles(  # fit matching no-rho Poisson candidate
            training,
            xi=DIXON_COLES_XI,
            fit_rho=False,
            maxiter=DIXON_COLES_MAXITER,
            ridge=DIXON_COLES_RIDGE,
        )
        metadata = {  # parameter-file metadata for reproducibility
            "fit_metadata": diagnostics,
            "training_min_date": str(pd.to_datetime(training["date"]).min().date()),
            "training_max_date": str(pd.to_datetime(training["date"]).max().date()),
        }
        plain_metadata = {  # no-rho candidate metadata
            "fit_metadata": plain_diagnostics,
            "training_min_date": str(pd.to_datetime(training["date"]).min().date()),
            "training_max_date": str(pd.to_datetime(training["date"]).max().date()),
        }
        model.save(params_path, extra=metadata)  # legacy timestamped DC parameter file
        model.save(dc_params_path, extra=metadata)  # timestamped DC candidate
        plain_model.save(plain_params_path, extra=plain_metadata)  # timestamped no-rho candidate
        model.save(DIXON_COLES_PARAMS_DC_FILE, extra=metadata)  # stable DC candidate
        plain_model.save(DIXON_COLES_PARAMS_PLAIN_FILE, extra=plain_metadata)  # stable no-rho candidate
        model.save(latest_params_path, extra=metadata)  # stable latest defaults to DC until validation selects
        _plot_attack_defense(params_path, ELO_CURRENT_RATINGS_FILE, figure_png, figure_pdf, caption_path)  # figure
        record.add_params(  # record key fitting constants
            {
                "xi": DIXON_COLES_XI,
                "recent_min_date": DIXON_COLES_RECENT_MIN_DATE,
                "max_train_matches": DIXON_COLES_MAX_TRAIN_MATCHES,
                "maxiter": DIXON_COLES_MAXITER,
                "ridge": DIXON_COLES_RIDGE,
            }
        )
        record.add_metrics({**diagnostics, "plain_poisson_fit": plain_diagnostics})  # record optimiser diagnostics
        record.add_output_artifact(params_path)  # hashed timestamped params
        record.add_output_artifact(dc_params_path)  # hashed timestamped DC candidate
        record.add_output_artifact(plain_params_path)  # hashed timestamped no-rho candidate
        record.add_output_artifact(DIXON_COLES_PARAMS_DC_FILE)  # hashed latest DC candidate
        record.add_output_artifact(DIXON_COLES_PARAMS_PLAIN_FILE)  # hashed latest no-rho candidate
        record.add_output_artifact(latest_params_path)  # hashed latest params
        record.add_output_artifact(figure_png)  # hashed PNG figure
        record.add_output_artifact(figure_pdf)  # hashed PDF figure
        record.add_output_artifact(caption_path)  # hashed caption
        logger.info("Wrote Dixon-Coles parameters to %s", params_path)  # log params path
        logger.info("Fit diagnostics: %s", diagnostics)  # log optimiser diagnostics
        if not diagnostics["success"] or not plain_diagnostics["success"]:  # fail loudly on optimiser failure
            raise RuntimeError("One or more Stage 1 engine candidate fits failed")  # error


if __name__ == "__main__":  # allow direct execution
    main()  # run script
