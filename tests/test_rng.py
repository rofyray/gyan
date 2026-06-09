"""Unit tests for reproducible RNG helper functions."""

import numpy as np  # compare deterministic draws across generators
import pytest  # exception assertions for invalid worker counts

from gyan.utils.rng import (  # public RNG helpers under test
    named_rng,
    spawn_rngs,
    spawn_seed_sequences,
)


def test_spawn_rngs_are_reproducible_and_independent() -> None:
    """Check spawned worker streams reproduce while remaining distinct."""
    first_run = spawn_rngs(2, 20260611)  # split the master seed into two streams
    second_run = spawn_rngs(2, 20260611)  # repeat the same split for comparison
    first_draws = [rng.integers(0, 1_000_000, size=5) for rng in first_run]  # sample streams
    second_draws = [rng.integers(0, 1_000_000, size=5) for rng in second_run]  # resample
    assert np.array_equal(first_draws[0], second_draws[0])  # worker 0 is reproducible
    assert np.array_equal(first_draws[1], second_draws[1])  # worker 1 is reproducible
    assert not np.array_equal(first_draws[0], first_draws[1])  # streams are independent


def test_spawn_seed_sequences_validates_worker_count() -> None:
    """Check invalid parallel worker counts fail before any RNG is created."""
    with pytest.raises(ValueError, match="n_workers must be >= 1"):  # invalid count
        spawn_seed_sequences(0, 20260611)  # zero workers cannot produce streams


def test_named_rng_is_stable_for_name() -> None:
    """Check named streams are stable for the same seed/name pair."""
    alpha = named_rng(20260611, "alpha").normal(size=4)  # deterministic named draw
    alpha_repeat = named_rng(20260611, "alpha").normal(size=4)  # same name and seed
    beta = named_rng(20260611, "beta").normal(size=4)  # different named stream
    assert np.array_equal(alpha, alpha_repeat)  # same stream reproduces exactly
    assert not np.array_equal(alpha, beta)  # distinct stream names split entropy
