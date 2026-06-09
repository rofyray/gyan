"""Shared helpers for Stage 1 raw-data downloads."""

from __future__ import annotations  # use modern type hints consistently

from dataclasses import dataclass  # small immutable records for source metadata
from pathlib import Path  # filesystem paths for raw artifact targets

import httpx  # HTTP client used for all source downloads

from gyan.config import repo_path_str
from gyan.utils.logging import sha256_file  # shared SHA-256 helper for run records


@dataclass(frozen=True)
class SourceFile:
    """Describe one raw artifact to download.

    Parameters
    ----------
    source_id : str
        Registry identifier such as "D1" or "D6".
    label : str
        Human-readable artifact label.
    url : str
        Source URL to request.
    raw_path : pathlib.Path
        Destination under data/raw/.
    required : bool
        Whether a failed download should fail the whole script.
    min_bytes : int
        Minimum acceptable byte size for a non-empty artifact.
    """

    source_id: str  # registry identifier from PRD/CONVENTIONS.md Section 7
    label: str  # short artifact label for logs and manifests
    url: str  # network location of the raw artifact
    raw_path: Path  # destination path in data/raw
    required: bool = True  # required artifacts fail the step if unavailable
    min_bytes: int = 1  # every artifact must be non-empty by default


@dataclass(frozen=True)
class DownloadOutcome:
    """Record the result of one attempted source download."""

    source_id: str  # registry identifier copied from SourceFile
    label: str  # artifact label copied from SourceFile
    url: str  # requested source URL
    path: str  # destination path as a string for JSON serialisation
    status: str  # "downloaded", "cached", or "failed"
    bytes: int  # file size in bytes, or 0 on failure
    sha256: str | None  # SHA-256 for downloaded/cached files, None on failure
    error: str | None  # error message for failed downloads


def user_agent_headers() -> dict[str, str]:
    """Return polite HTTP headers for public-data downloads."""
    return {  # use a real contact-style user agent instead of the httpx default
        "User-Agent": "GYAN-WorldCupModel/0.1 (+https://github.com/openai/codex)",
        "Accept": "*/*",  # accept CSV, JSON, PDF, and HTML artifacts
    }


def download_source_file(
    source_file: SourceFile,
    client: httpx.Client,
    force: bool = False,
) -> DownloadOutcome:
    """Download one SourceFile and return a structured outcome.

    Parameters
    ----------
    source_file : SourceFile
        The source artifact to fetch.
    client : httpx.Client
        Reused HTTP client with timeout and redirect handling.
    force : bool
        If True, overwrite an existing raw artifact.

    Returns
    -------
    DownloadOutcome
        Status, file size, SHA-256, and any error message.
    """
    raw_path = source_file.raw_path  # destination path for this artifact
    if raw_path.exists() and not force:  # raw artifacts are immutable once written
        file_size = raw_path.stat().st_size  # read the cached artifact size
        file_hash = sha256_file(raw_path)  # hash the cached artifact
        return DownloadOutcome(  # report a cache hit without touching the file
            source_id=source_file.source_id,
            label=source_file.label,
            url=source_file.url,
            path=repo_path_str(raw_path),
            status="cached",
            bytes=file_size,
            sha256=file_hash,
            error=None,
        )

    try:  # keep network errors isolated to this artifact
        response = client.get(source_file.url)  # request the source URL
        response.raise_for_status()  # convert HTTP 4xx/5xx into an exception
        content = response.content  # keep bytes so PDFs and CSVs are both safe
        if len(content) < source_file.min_bytes:  # reject empty/truncated responses
            raise ValueError(  # raise a clear validation error
                f"downloaded {len(content)} bytes, expected at least {source_file.min_bytes}"
            )
        raw_path.parent.mkdir(parents=True, exist_ok=True)  # ensure target folder exists
        raw_path.write_bytes(content)  # persist the raw response exactly as received
        file_hash = sha256_file(raw_path)  # hash the persisted raw artifact
        return DownloadOutcome(  # report the successful download
            source_id=source_file.source_id,
            label=source_file.label,
            url=source_file.url,
            path=repo_path_str(raw_path),
            status="downloaded",
            bytes=len(content),
            sha256=file_hash,
            error=None,
        )
    except Exception as exc:  # record failures so fallback/later stages can inspect them
        return DownloadOutcome(  # failed optional sources do not crash here
            source_id=source_file.source_id,
            label=source_file.label,
            url=source_file.url,
            path=repo_path_str(raw_path),
            status="failed",
            bytes=0,
            sha256=None,
            error=str(exc),
        )
