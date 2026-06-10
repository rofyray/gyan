"""Download Stage 1 raw data artifacts once and record their hashes."""

from __future__ import annotations  # use modern type hints consistently

import argparse  # CLI switch for forced source refreshes
import json  # write a machine-readable download manifest
from dataclasses import asdict  # convert DownloadOutcome records to dictionaries

import httpx  # HTTP client used by the shared download helper

from gyan.config import GLOBAL_SEED, OUTPUTS_REPORTS, create_directories, repo_path_str  # project constants
from gyan.data.download_common import download_source_file, user_agent_headers  # downloader
from gyan.data.source_registry import stage1_source_files  # ordered Stage 1 sources
from gyan.utils.logging import RunRecord, get_run_logger  # required logging/run record helpers


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the downloader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw artifacts instead of using cache.")
    return parser.parse_args()


def main() -> None:
    """Run the Stage 1.2 raw-data download step."""
    args = parse_args()
    create_directories()  # ensure data/raw, logs, and report folders exist
    logger, log_path = get_run_logger(  # create the required human-readable log
        "s1_download_data",
        stage="stage1",
        step="download_data",
    )
    with RunRecord(  # create the required machine-readable run record
        stage="stage1",
        step="download_data",
        script_path=__file__,
        global_seed=GLOBAL_SEED,
        logger=logger,
    ) as record:
        record.add_output(log_path)  # include the human-readable log as an output
        source_files = stage1_source_files()  # gather every configured source artifact
        record.add_param("n_source_files", len(source_files))  # record configured source count
        record.add_param("force_refresh", args.force)  # record whether existing raw cache was overwritten
        logger.info("Configured %s source artifacts", len(source_files))  # log source count
        outcomes = []  # collect every download/cached/failed result
        with httpx.Client(  # reuse connections and follow redirects for public sources
            headers=user_agent_headers(),
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
        ) as client:
            for source_file in source_files:  # process each source in registry order
                logger.info("Fetching %s %s", source_file.source_id, source_file.label)
                outcome = download_source_file(source_file, client=client, force=args.force)  # fetch/cache artifact
                outcomes.append(outcome)  # keep the outcome for the manifest
                if outcome.status == "failed":  # failed artifact gets explicit log detail
                    logger.warning(  # warn but continue so fallbacks/optional sources can work
                        "Failed %s %s from %s: %s",
                        outcome.source_id,
                        outcome.label,
                        outcome.url,
                        outcome.error,
                    )
                else:  # successful or cached artifact gets size/hash logged
                    logger.info(  # log the raw artifact path, size, and SHA-256
                        "%s %s %s bytes sha256=%s path=%s",
                        outcome.status,
                        outcome.label,
                        outcome.bytes,
                        outcome.sha256,
                        outcome.path,
                    )
                    record.add_input(outcome.path)  # raw artifacts are inputs to later steps
                    record.add_output(outcome.path)  # and outputs of this download step
        manifest_rows = [asdict(outcome) for outcome in outcomes]  # JSON-safe manifest rows
        manifest_path = OUTPUTS_REPORTS / "stage1_download_manifest.json"  # stable manifest path
        manifest_path.write_text(  # write all statuses, errors, sizes, and hashes
            json.dumps(manifest_rows, indent=2),
            encoding="utf-8",
        )
        record.add_output(manifest_path)  # include manifest in the run record
        failed_required = [  # required failures should fail Task 1.2
            outcome for outcome, source_file in zip(outcomes, source_files)
            if source_file.required and outcome.status == "failed"
        ]
        record.add_metric("n_downloaded", sum(row.status == "downloaded" for row in outcomes))
        record.add_metric("n_cached", sum(row.status == "cached" for row in outcomes))
        record.add_metric("n_failed", sum(row.status == "failed" for row in outcomes))
        record.add_metric("n_failed_required", len(failed_required))
        logger.info("Download manifest written to %s", repo_path_str(manifest_path))  # log manifest path
        if failed_required:  # required artifacts must be present before proceeding
            labels = ", ".join(outcome.label for outcome in failed_required)  # readable labels
            raise RuntimeError(f"Required downloads failed: {labels}")  # fail the script


if __name__ == "__main__":  # allow direct execution as a stage script
    main()  # run the entry point
