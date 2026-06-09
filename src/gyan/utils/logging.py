"""Run logging and machine-readable run records for GYAN.

Two responsibilities (CONVENTIONS Section 4):

1. get_run_logger(...): a stdlib logger that writes a human-readable, timestamped
   log file under logs/ and optionally echoes to the console.
2. RunRecord / write_run_record(...): a JSON "run record" written under
   outputs/reports/ that captures everything needed to reproduce and audit a run:
   timestamps, git commit, python and package versions, the seed, input file
   hashes, parameters, metrics, output paths, and wall-clock duration.

Every number that matters must end up in BOTH the log and the run record; never
print results only to stdout.

Note: a module named gyan.utils.logging does NOT shadow the stdlib `logging`
module, because Python 3 uses absolute imports; `import logging` below is stdlib.
"""

from __future__ import annotations  # modern type-hint syntax on all runtimes

import json         # serialise the run record to JSON
import logging      # stdlib logging (not shadowed by this module's name)
import platform     # python version string
import subprocess   # capture the git commit hash
import hashlib      # SHA-256 of input files
from datetime import datetime, timezone  # timezone-aware UTC timestamps
from importlib import metadata           # read installed package versions
from pathlib import Path                 # filesystem paths

# Project paths come from the single source of truth, config.py.
from gyan.config import LOGS_DIR, OUTPUTS_REPORTS, PROJECT_ROOT, repo_path_str

# Core libraries whose versions we record in every run (for the paper's
# reproducibility appendix). "gyan" is included so the package version is captured.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "gyan", "numpy", "pandas", "pyarrow", "scipy", "statsmodels",
    "penaltyblog", "numba", "joblib", "scikit-learn", "matplotlib",
    "seaborn", "pydantic", "httpx", "requests", "beautifulsoup4", "lxml",
    "soccerdata", "python-json-logger", "pytest", "pytest-cov", "tqdm",
)


def _utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)  # aware datetime, never naive


def _timestamp_tag(moment: datetime) -> str:
    """Format a datetime as a filename-safe YYYYMMDD_HHMMSS tag."""
    return moment.strftime("%Y%m%d_%H%M%S")  # compact, sortable, path-safe


def git_commit_short() -> str:
    """Return the short git commit hash of the repo, or 'unknown' if unavailable.

    Gracefully handles git not being installed or the project not being a git
    repository, so logging never crashes a run.
    """
    try:                                              # guard any subprocess failure
        result = subprocess.run(                      # invoke git in the repo directory
            ["git", "rev-parse", "--short", "HEAD"],  # command: short HEAD hash
            cwd=PROJECT_ROOT,                         # run from the project root
            capture_output=True,                      # capture stdout and stderr
            text=True,                                # decode bytes to str
            check=True,                               # raise on a non-zero exit code
            timeout=10,                               # never hang the run
        )
        return result.stdout.strip()                  # the hash, whitespace-trimmed
    except Exception:                                 # any failure -> sentinel value
        return "unknown"                              # do not crash on missing git


def git_tags_at_head() -> list[str]:
    """Return git tags pointing at HEAD, or an empty list if unavailable."""
    try:                                              # guard git/subprocess failures
        result = subprocess.run(                      # list tags on the current commit
            ["git", "tag", "--points-at", "HEAD"],    # exact tags only
            cwd=PROJECT_ROOT,                         # run from the project root
            capture_output=True,                      # capture stdout/stderr
            text=True,                                # decode bytes to str
            check=True,                               # raise on non-zero exit
            timeout=10,                               # avoid hanging a run
        )
        return [tag for tag in result.stdout.splitlines() if tag]  # stable list
    except Exception:                                 # any failure -> empty list
        return []                                     # do not crash on missing git


def capture_package_versions(
    packages: tuple[str, ...] = _TRACKED_PACKAGES,
) -> dict[str, str]:
    """Return a mapping of package name to installed version.

    Missing packages are recorded as 'not_installed' rather than raising, so a
    partial environment still produces a complete record.
    """
    versions: dict[str, str] = {}                     # accumulate name -> version
    for package_name in packages:                     # iterate over tracked packages
        try:                                          # the version lookup may fail
            versions[package_name] = metadata.version(package_name)  # installed version
        except metadata.PackageNotFoundError:         # package not installed here
            versions[package_name] = "not_installed"  # record the absence explicitly
    return versions                                   # complete version map


def sha256_file(path: Path | str, chunk_bytes: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB chunks (memory-safe).

    Parameters
    ----------
    path : Path | str
        File to hash.
    chunk_bytes : int
        Read buffer size in bytes (default 1 MiB) so large files are not loaded
        into memory all at once.
    """
    file_path = Path(path)                            # normalise the argument to a Path
    hasher = hashlib.sha256()                         # incremental SHA-256 state
    with file_path.open("rb") as handle:              # open the file in binary mode
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):  # read until EOF
            hasher.update(chunk)                      # fold each chunk into the hash
    return hasher.hexdigest()                         # return the final hex digest


def artifact_descriptor(path: Path | str) -> dict[str, object]:
    """Return a reproducibility descriptor for a file artifact.

    Parameters
    ----------
    path : Path | str
        File to describe.

    Returns
    -------
    dict[str, object]
        Path, SHA-256 content hash, and file size in bytes.
    """
    file_path = Path(path)                            # normalise the argument to a Path
    return {                                          # build a JSON-serialisable descriptor
        "path": repo_path_str(file_path),             # shareable repo-relative path
        "sha256": sha256_file(file_path),             # content hash for auditability
        "bytes": file_path.stat().st_size,            # size in bytes for sanity checks
    }


def get_run_logger(
    name: str,
    stage: str | None = None,
    step: str | None = None,
    to_console: bool = True,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """Create a logger that writes a timestamped file under logs/ (CONVENTIONS S4).

    Parameters
    ----------
    name : str
        Logger name (usually the module or script name).
    stage, step : str | None
        Optional labels used in the log filename. If both are given the file is
        logs/{stage}_{step}_{timestamp}.log, else logs/{name}_{timestamp}.log.
    to_console : bool
        Also echo log lines to the console if True.
    level : int
        Logging threshold (default INFO).

    Returns
    -------
    (logging.Logger, Path)
        The configured logger and the path of the log file it writes to.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)       # ensure the logs directory exists
    timestamp = _timestamp_tag(_utc_now())            # filename timestamp tag
    if stage and step:                                # prefer stage/step naming
        log_filename = f"{stage}_{step}_{timestamp}.log"  # e.g. stage1_elo_2026...log
    else:                                             # otherwise fall back to the name
        log_filename = f"{name}_{timestamp}.log"      # e.g. mymodule_2026...log
    log_path = LOGS_DIR / log_filename                # full path of the log file

    logger = logging.getLogger(name)                  # fetch or create the named logger
    logger.setLevel(level)                            # set its level threshold
    logger.propagate = False                          # do not also log to the root logger
    logger.handlers.clear()                           # avoid duplicate handlers on re-call

    log_format = logging.Formatter(                   # a consistent, parseable line format
        "%(asctime)s %(levelname)s %(name)s: %(message)s",  # time level name: message
        datefmt="%Y-%m-%dT%H:%M:%S%z",                # ISO-8601 timestamps
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")  # write to the log file
    file_handler.setFormatter(log_format)             # apply the format to file output
    file_handler.setLevel(level)                      # file captures at the given level
    logger.addHandler(file_handler)                   # attach the file handler

    if to_console:                                    # optionally also echo to console
        console_handler = logging.StreamHandler()     # writes to stderr by default
        console_handler.setFormatter(log_format)      # same format as the file
        console_handler.setLevel(level)               # same threshold
        logger.addHandler(console_handler)            # attach the console handler

    logger.info("Run logger started; writing to %s", repo_path_str(log_path))  # first log line
    return logger, log_path                           # hand back the logger and its path


def _json_default(value: object) -> object:
    """JSON serialiser fallback for types json cannot handle natively.

    Converts Path to str and numpy scalars (objects exposing .item()) to native
    Python scalars; otherwise falls back to str() so a record never fails to write.
    """
    if isinstance(value, Path):                       # Path objects -> string
        return repo_path_str(value)
    if hasattr(value, "item"):                        # numpy scalar -> python scalar
        return value.item()
    return str(value)                                 # last-resort string form


def write_run_record(record: dict, stage: str, step: str) -> Path:
    """Write a run-record dict to outputs/reports as timestamped JSON.

    Parameters
    ----------
    record : dict
        The fully assembled run record.
    stage, step : str
        Labels used in the filename run_{stage}_{step}_{timestamp}.json.

    Returns
    -------
    Path
        The path of the written JSON file.
    """
    OUTPUTS_REPORTS.mkdir(parents=True, exist_ok=True)  # ensure the reports dir exists
    timestamp = _timestamp_tag(_utc_now())            # filename timestamp tag
    report_path = OUTPUTS_REPORTS / f"run_{stage}_{step}_{timestamp}.json"  # target path
    with report_path.open("w", encoding="utf-8") as handle:  # open the file for writing
        json.dump(record, handle, indent=2, default=_json_default)  # write pretty JSON
    return report_path                                # hand back the written path


class RunRecord:
    """Context manager that assembles and writes a run record (CONVENTIONS S4).

    Usage
    -----
    >>> with RunRecord(stage="stage1", step="elo", global_seed=GLOBAL_SEED) as rec:
    ...     rec.add_input(some_path)          # auto-hashes the input file
    ...     rec.add_param("xi", 0.0018)       # record a parameter
    ...     rec.add_metric("mean_rps", 0.21)  # record a metric
    ...     rec.add_output(out_path)          # record an output artefact
    ... # on exit, duration is computed and the JSON record is written automatically

    The record auto-captures the start timestamp, git commit, python version, and
    core package versions.
    """

    def __init__(
        self,
        stage: str,
        step: str,
        script_path: str | Path | None = None,
        global_seed: int | None = None,
        n_workers: int | None = None,
        n_simulations: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the record skeleton and capture environment metadata."""
        self.stage = stage                            # stage label, used in filename
        self.step = step                              # step label, used in filename
        self.logger = logger                          # optional logger for messages
        self._start = _utc_now()                      # wall-clock start time
        self.record: dict = {                         # the record dict being assembled
            "timestamp_utc": self._start.isoformat(),         # ISO-8601 start time
            "stage": stage,                                   # stage label
            "step": step,                                     # step label
            "script_path": repo_path_str(script_path) if script_path else None,  # entry script
            "git_commit": git_commit_short(),                 # repo commit hash
            "git_tags": git_tags_at_head(),                    # release tags on HEAD
            "python_version": platform.python_version(),      # e.g. 3.12.x
            "package_versions": capture_package_versions(),   # core library versions
            "global_seed": global_seed,                       # the master seed
            "n_workers": n_workers,                           # parallel worker count
            "n_simulations": n_simulations,                   # MC draws (if relevant)
            "inputs": [],                                     # list of input descriptors
            "params": {},                                     # key parameters
            "metrics": {},                                    # key output metrics
            "outputs": [],                                    # output artefact paths
            "output_artifacts": [],                           # hashed output descriptors
            "status": "running",                              # set to ok/error on exit
        }

    def add_input(self, path: str | Path) -> None:
        """Record an input file with its SHA-256 and size (for reproducibility)."""
        self.record["inputs"].append(artifact_descriptor(path))  # append hashed descriptor

    def add_param(self, key: str, value: object) -> None:
        """Record a single key parameter."""
        self.record["params"][key] = value           # store under params

    def add_params(self, params: dict) -> None:
        """Record several parameters at once."""
        self.record["params"].update(params)         # merge into params

    def add_metric(self, key: str, value: object) -> None:
        """Record a single output metric (RPS, Brier, etc.)."""
        self.record["metrics"][key] = value           # store under metrics

    def add_metrics(self, metrics: dict) -> None:
        """Record several metrics at once."""
        self.record["metrics"].update(metrics)        # merge into metrics

    def add_output(self, path: str | Path) -> None:
        """Record an output artefact path that the run produced."""
        self.record["outputs"].append(repo_path_str(path))  # store shareable path string

    def add_output_artifact(self, path: str | Path) -> None:
        """Record an output artifact path plus SHA-256 and size metadata."""
        descriptor = artifact_descriptor(path)          # hash and size the output file
        self.record["outputs"].append(descriptor["path"])  # preserve legacy path list
        self.record["output_artifacts"].append(descriptor)  # add full audit descriptor

    def write(self) -> Path:
        """Finalise duration and write the JSON record; return its path."""
        end = _utc_now()                              # wall-clock end time
        self.record["duration_seconds"] = (end - self._start).total_seconds()  # elapsed
        report_path = write_run_record(self.record, self.stage, self.step)  # persist JSON
        if self.logger is not None:                   # if a logger was supplied
            self.logger.info("Run record written to %s", repo_path_str(report_path))  # log the path
        return report_path                            # hand back the path

    def __enter__(self) -> "RunRecord":
        """Enter the context; return self so callers can add inputs/params/metrics."""
        return self                                   # the record object itself

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Exit the context: mark status, write the record, and re-raise any error."""
        self.record["status"] = "error" if exc_type else "ok"  # success vs failure flag
        if exc_type and self.logger is not None:      # if an exception occurred and we log
            self.logger.error("Run failed: %s", exc_value)  # record the exception message
        self.write()                                  # always write the record, even on error
        return False                                  # return False so exceptions propagate
