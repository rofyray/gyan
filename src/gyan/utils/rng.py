"""Parallel-safe, reproducible random number generation for GYAN.

All randomness in the project flows from GLOBAL_SEED (config.py) through NumPy's
modern SeedSequence / Generator API. This guarantees the two properties a
publishable result needs (CONVENTIONS Section 3.4):

1. Reproducibility: the same seed reproduces the same numbers every run.
2. Independence across parallel workers: each worker gets a statistically
   independent stream, so a parallel Monte-Carlo gives the SAME aggregate result
   regardless of how many cores it runs on.

Never call numpy.random.<func> (the legacy global RNG) in modelling code. Always
take a Generator from here and call methods on it (rng.poisson, rng.choice, ...).
"""

from __future__ import annotations  # allow modern type-hint syntax on all runtimes

import hashlib  # stable, process-independent hashing for named sub-streams

# Import the modern NumPy RNG primitives. SeedSequence turns a seed into high
# quality entropy; default_rng builds a Generator; Generator is the return type.
from numpy.random import SeedSequence, default_rng, Generator


def _validate_n_workers(n_workers: int) -> None:
    """Raise ValueError if a requested worker count cannot produce RNG streams."""
    if n_workers < 1:                                       # guard against bad input
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")  # fail loudly


def get_rng(global_seed: int) -> Generator:
    """Return one reproducible NumPy Generator seeded from global_seed.

    Use this for single-threaded code. For parallel work use spawn_rngs instead.

    Parameters
    ----------
    global_seed : int
        The project master seed (config.GLOBAL_SEED).

    Returns
    -------
    numpy.random.Generator
        A fresh, reproducible generator.
    """
    seed_sequence = SeedSequence(global_seed)  # deterministic entropy from the seed
    return default_rng(seed_sequence)          # build and return a Generator from it


def spawn_rngs(n_workers: int, global_seed: int) -> list[Generator]:
    """Spawn n_workers independent, reproducible Generators from one master seed.

    This is the parallel-safe pattern (CONVENTIONS Section 3.4): a master
    SeedSequence is split into n_workers non-overlapping child sequences, each of
    which seeds its own Generator. Worker w always receives the same stream for a
    given (global_seed, n_workers), so parallel results are bit-reproducible.

    Parameters
    ----------
    n_workers : int
        Number of independent streams to create (one per parallel worker). Must
        be >= 1.
    global_seed : int
        The project master seed (config.GLOBAL_SEED).

    Returns
    -------
    list[numpy.random.Generator]
        A list of n_workers independent generators.

    Raises
    ------
    ValueError
        If n_workers is less than 1.
    """
    child_sequences = spawn_seed_sequences(n_workers, global_seed)  # deterministic child seeds
    return [default_rng(child) for child in child_sequences]  # one Generator per child


def spawn_seed_sequences(n_workers: int, global_seed: int) -> list[SeedSequence]:
    """Spawn n_workers independent SeedSequence objects from one master seed.

    Use this when a downstream library or worker initialiser needs seed objects
    rather than ready-made NumPy Generators. Keeping the split here prevents each
    caller from reimplementing the parallel-safe seeding pattern.

    Parameters
    ----------
    n_workers : int
        Number of independent seed sequences to create. Must be >= 1.
    global_seed : int
        The project master seed (config.GLOBAL_SEED).

    Returns
    -------
    list[numpy.random.SeedSequence]
        A list of n_workers independent child seed sequences.

    Raises
    ------
    ValueError
        If n_workers is less than 1.
    """
    _validate_n_workers(n_workers)                         # enforce a usable worker count
    master_sequence = SeedSequence(global_seed)            # single root of all entropy
    return master_sequence.spawn(n_workers)                # n independent child sequences


def named_rng(global_seed: int, name: str) -> Generator:
    """Return a reproducible Generator for a NAMED, independent sub-stream.

    Useful when different components need their own independent randomness from
    the same master seed (for example the goal-model bootstrap vs the market
    sampler) without interfering with each other or with worker streams.

    The name is hashed with SHA-256 (stable across processes, unlike Python's
    built-in hash(), which is salted per process) and mixed into the SeedSequence
    as extra entropy alongside the global seed.

    Parameters
    ----------
    global_seed : int
        The project master seed (config.GLOBAL_SEED).
    name : str
        A label identifying the sub-stream, e.g. "market_sampler".

    Returns
    -------
    numpy.random.Generator
        A reproducible generator unique to the (global_seed, name) pair.
    """
    name_digest = hashlib.sha256(name.encode("utf-8")).digest()  # stable 32-byte hash
    name_entropy = int.from_bytes(name_digest[:8], "big")        # take 64 bits as an int
    seed_sequence = SeedSequence([global_seed, name_entropy])    # mix seed + name entropy
    return default_rng(seed_sequence)                            # build and return it
