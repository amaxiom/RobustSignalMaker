"""Milestone 1: reproducibility contract (RMM/RPM-style suite).

Seeds must be pure functions of (base, index, repeat), the documented offsets
must not collide, and derived generators must be exactly reproducible.
"""
from __future__ import annotations

import numpy as np
import pytest

from robustsignalmaker import (
    BOOTSTRAP_OFFSET,
    INNER_OFFSET,
    REPEAT_STRIDE,
    REPRESENTATION_OFFSET,
    Seeds,
    set_global_seed,
)


def test_seed_derivation_is_pure_and_documented():
    s = Seeds(base=42, n_bootstrap=100, n_representations=8)
    # pure: same inputs, same outputs, independent of call order
    a = s.bootstrap(7)
    _ = s.representation(3)
    assert s.bootstrap(7) == a
    # documented arithmetic
    assert s.outer(2, repeat=1) == 42 + REPEAT_STRIDE + 2
    assert s.inner(4) == 42 + INNER_OFFSET + 4
    assert s.bootstrap(9) == 42 + BOOTSTRAP_OFFSET + 9
    assert s.representation(5) == 42 + REPRESENTATION_OFFSET + 5


def test_seed_streams_do_not_collide_for_realistic_sizes():
    s = Seeds(base=0, n_bootstrap=9_999, n_representations=9_999)
    boots = {s.bootstrap(b) for b in range(s.n_bootstrap)}
    reps = {s.representation(r) for r in range(s.n_representations)}
    inner = {s.inner(f) for f in range(100)}
    outer = {s.outer(f) for f in range(100)}
    all_seeds = boots | reps | inner | outer
    assert len(all_seeds) == len(boots) + len(reps) + len(inner) + len(outer)


def test_collision_guard_raises_loudly():
    with pytest.raises(ValueError, match="collide"):
        Seeds(base=0, n_bootstrap=REPRESENTATION_OFFSET - BOOTSTRAP_OFFSET)
    with pytest.raises(ValueError, match="collide"):
        Seeds(base=0, n_representations=INNER_OFFSET - REPRESENTATION_OFFSET)


def test_derived_generators_reproduce_exactly():
    s = Seeds(base=7)
    draws_a = s.rng(s.bootstrap(3)).standard_normal(16)
    draws_b = s.rng(s.bootstrap(3)).standard_normal(16)
    assert np.array_equal(draws_a, draws_b)
    draws_c = s.rng(s.bootstrap(4)).standard_normal(16)
    assert not np.array_equal(draws_a, draws_c)


def test_set_global_seed_pins_ambient_channels():
    set_global_seed(123)
    a = np.random.rand(4)
    set_global_seed(123)
    b = np.random.rand(4)
    assert np.array_equal(a, b)
