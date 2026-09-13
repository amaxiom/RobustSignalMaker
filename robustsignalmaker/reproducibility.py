"""Reproducibility: deterministic seeding and per-stream seed derivation.

RobustSignalMaker (RSM) inherits the reproducibility contract shared by
RobustModelMaker (RMM) and RobustPixelMaker (RPM): a single integer
``random_state`` deterministically derives every downstream seed via
documented, transparent offsets. Two runs with the same data, parameters and
seed return identical selected regions, fold scores, stability frequencies
and predictions.

Seed-derivation scheme (kept additive and transparent, per RMM/RPM):
    outer fold      : base + fold_idx                       (+ repeat * REPEAT_STRIDE)
    bootstrap       : base + BOOTSTRAP_OFFSET      + b_idx
    representation  : base + REPRESENTATION_OFFSET + r_idx
    inner search    : base + INNER_OFFSET          + fold_idx

The offsets are spaced so realistic loop sizes cannot collide; ``Seeds``
asserts this at construction time rather than trusting it silently. All
derivations are pure functions of ``(base, index, repeat)`` so results never
depend on call order or execution schedule.

Scope of the contract (measured, 2026-09-01 bug sweep): reproducibility is
"same seed AND same row order". Permuting the input rows leaves the learned
mask and the selected set unchanged, but fitted coefficients can differ at
the last few floating-point bits, because BLAS reductions are not
associative; resampling-based selectors additionally draw resamples by row
INDEX, so their frequencies are defined relative to the given ordering.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

# --- documented offsets (transparent, RMM/RPM-style) ------------------------
BOOTSTRAP_OFFSET = 10_000
REPRESENTATION_OFFSET = 20_000
INNER_OFFSET = 30_000
REPEAT_STRIDE = 100_000


def set_global_seed(seed: int) -> None:
    """Seed every global RNG channel RSM can reach.

    Sets ``PYTHONHASHSEED``, the ``random`` module, and NumPy's legacy global
    RNG. Prefer the per-stream generators from :class:`Seeds` for actual work;
    this closes ambient channels. RSM is numpy-only, so unlike RPM there is no
    torch branch here.
    """
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class Seeds:
    """Deterministic per-stream seed derivation from one base seed.

    All methods are pure functions of ``(base, index, repeat)`` so results
    never depend on call order or execution schedule, a requirement for
    resuming and for reproducibility under parallelism.
    """

    base: int
    n_bootstrap: int = 0
    n_representations: int = 0

    def __post_init__(self) -> None:
        # Guard the offset spacing so loops cannot alias onto one another.
        if not (0 <= self.n_bootstrap < REPRESENTATION_OFFSET - BOOTSTRAP_OFFSET):
            raise ValueError(
                f"n_bootstrap={self.n_bootstrap} would collide with representation seeds; "
                f"must be < {REPRESENTATION_OFFSET - BOOTSTRAP_OFFSET}"
            )
        if not (0 <= self.n_representations < INNER_OFFSET - REPRESENTATION_OFFSET):
            raise ValueError(
                f"n_representations={self.n_representations} would collide with inner seeds; "
                f"must be < {INNER_OFFSET - REPRESENTATION_OFFSET}"
            )

    def outer(self, fold_idx: int, repeat: int = 0) -> int:
        return self.base + repeat * REPEAT_STRIDE + fold_idx

    def inner(self, fold_idx: int, repeat: int = 0) -> int:
        return self.base + repeat * REPEAT_STRIDE + INNER_OFFSET + fold_idx

    def bootstrap(self, b_idx: int) -> int:
        return self.base + BOOTSTRAP_OFFSET + b_idx

    def representation(self, r_idx: int) -> int:
        return self.base + REPRESENTATION_OFFSET + r_idx

    def rng(self, seed: int) -> np.random.Generator:
        """A fresh NumPy Generator for a derived seed (preferred over global RNG)."""
        return np.random.default_rng(seed)
