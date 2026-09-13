"""Milestone 8: benchmark-harness smoke tests.

Guards the harness lessons (registries agree, variants are named, loaders
honour their contract) without re-running the benchmark; the benchmark
itself runs via ``py benchmarks/run_synthetic_benchmark.py`` and its
findings live in benchmarks/FINDINGS.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from competitors import METHOD_ORDER, SELECTORS, interp_fill, mean_fill  # noqa: E402
from datasets import (  # noqa: E402
    DATA_DIR,
    LOADERS,
    REAL_LOADERS,
    SYNTHETIC_LOADERS,
    truth_segments,
)

from robustsignalmaker import SegmentGrid  # noqa: E402


def test_method_order_derives_from_the_registry():
    assert METHOD_ORDER == list(SELECTORS)
    assert "rsm_masked" in SELECTORS and "full_signal" in SELECTORS
    # every fill variant names its fill explicitly (the RPM harness lesson)
    for name in SELECTORS:
        if name not in ("full_signal", "rsm_masked"):
            assert name.endswith(("_mean_fill", "_interp_fill")), name


@pytest.mark.parametrize("name", list(SYNTHETIC_LOADERS))
def test_synthetic_loaders_honour_the_contract(name):
    X, V, y, meta = SYNTHETIC_LOADERS[name](n=60, seed=1)
    assert X.ndim == 3 and V.shape == X.shape and V.dtype == bool
    assert len(y) == X.shape[0]
    assert meta["name"] == name and meta["task"] == "binary"
    assert meta["truth_points"].shape == (X.shape[-1],)
    # observed values are finite; the loader never fabricates
    assert np.isfinite(X[V]).all()
    grid = SegmentGrid(X.shape[-1], segment=meta["segment"])
    truth = truth_segments(meta["truth_points"], grid)
    assert len(truth) >= 1


@pytest.mark.parametrize("name", list(REAL_LOADERS))
def test_real_loaders_honour_the_contract_when_cached(name):
    """Real loaders need downloaded files; without the cache this skips
    rather than failing (and rather than downloading inside the test run)."""
    caches = {"tecator": DATA_DIR / "openml",
              "corn": DATA_DIR / "corn.mat",
              "ovarian": DATA_DIR / "OvarianCD_PostQAQC.zip",
              "rruff": DATA_DIR / "rruff_excellent_unoriented.zip",
              "basicmotions": DATA_DIR / "BasicMotions.zip"}
    if not caches[name].exists():
        pytest.skip(f"{name} data not cached under benchmarks/data/")
    X, V, y, meta = REAL_LOADERS[name]()
    assert X.ndim == 3 and V.shape == X.shape and V.dtype == bool
    assert len(y) == X.shape[0] and X.shape[0] >= 80
    assert meta["name"] == name
    assert meta["task"] in ("binary", "multiclass", "regression")
    assert "truth_points" not in meta and meta["k"] >= 1  # no fake ground truth
    assert np.isfinite(X[V]).all()


def test_truth_segments_handles_bands_spikes_and_their_mixture():
    grid = SegmentGrid(64, segment=8)
    band = np.zeros(64, dtype=bool)
    band[8:24] = True  # segments 1 and 2 fully
    assert truth_segments(band, grid).tolist() == [1, 2]
    spikes = np.zeros(64, dtype=bool)
    spikes[[3, 40]] = True  # single points: still truth under the same rule
    assert truth_segments(spikes, grid).tolist() == [0, 5]
    assert truth_segments(np.zeros(64, dtype=bool), grid).size == 0
    # the 2026-09-01 sweep case: a spike coexisting with a band must not be
    # silently dropped (the old rule compared against the fullest segment)
    mixed = band.copy()
    mixed[40] = True
    assert truth_segments(mixed, grid).tolist() == [1, 2, 5]
    # a band's thin spill into a neighbouring segment still does not qualify
    spill = np.zeros(64, dtype=bool)
    spill[5:24] = True  # 3 points into segment 0, full segments 1 and 2
    assert truth_segments(spill, grid).tolist() == [1, 2]


def test_fills_only_write_into_gaps():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 1, 40))
    V = np.ones_like(X, dtype=bool)
    V[:, :, 10:20] = False
    for fill in (mean_fill, interp_fill):
        Xf = fill(X, V)
        assert np.array_equal(Xf[V], X[V])          # observed values untouched
        assert np.isfinite(Xf).all()                # gaps filled with something


def test_selectors_run_and_respect_k_on_a_tiny_control():
    X, V, y, meta = LOADERS["raman"](n=80, n_points=256, seed=3)
    grid = SegmentGrid(256, segment=16)
    for name in ("anova_mean_fill", "mcuve_mean_fill", "full_signal"):
        sel = SELECTORS[name](X, V, y, grid, "binary", 0, 3)
        assert len(sel) == (grid.n_segments if name == "full_signal" else 3)
        assert all(0 <= s < grid.n_segments for s in sel)
