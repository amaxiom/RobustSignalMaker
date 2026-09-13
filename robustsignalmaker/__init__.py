"""RobustSignalMaker: NaN-aware, leakage-free stability selection for signals.

Third sibling of RobustModelMaker (tabular columns) and RobustPixelMaker
(image patches): RSM identifies the important parts of scientific signals and
spectra (time-, space-, mass-series) to retain and removes the rest, with
missing data treated as absent evidence rather than something to fabricate.
See docs/IMPLEMENTATION_GUIDE.md for the design and docs/API_REFERENCE.md
for the public surface.
"""
from __future__ import annotations

from .config import RSMConfig, fit_count
from .metrics import (
    BaselineComparison,
    adjusted_jaccard,
    expected_jaccard,
    jaccard,
    mean_pairwise_jaccard,
    paired_comparison,
    rmse_from_score,
    score_predictions,
)
from .masking import (
    SoftMaskSelector,
    loss_and_dz,
    mask_data_gradient,
    renormalised_gated_forward,
)
from .nested_cv import (
    BootstrapMaskFoldEstimator,
    EnsembleMaskFoldEstimator,
    FoldEstimator,
    FullSignalFoldEstimator,
    NestedCV,
    SoftMaskFoldEstimator,
    infer_task,
    make_inner_splitter,
    make_outer_splitter,
)
from .representations import (
    GaussianScale1D,
    RandomConv1D,
    SavGolDerivative,
    SegmentMean,
    SegmentStats,
    default_ensemble,
)
from .reproducibility import (
    BOOTSTRAP_OFFSET,
    INNER_OFFSET,
    REPEAT_STRIDE,
    REPRESENTATION_OFFSET,
    Seeds,
    set_global_seed,
)
from .models import (
    ALGORITHMS,
    ExternalRefit,
    PLSDA,
    make_refit_model,
    refit_features,
)
from .results import RSMResult
from .segments import SegmentGrid
from .selection import (
    BootstrapMaskSelector,
    RepresentationEnsembleSelector,
    complementary_pair,
    observation_groups,
    resample_indices,
)
from .tuning import lam_for_coverage, lam_frontier
from .synthetic import (
    SignalControl,
    drop_labels,
    make_signal_control,
    mask_band_by_group,
    mask_dropout_stretches,
    mask_saturation_censor,
    mask_scattered,
)
from .validity import (
    InsufficientEvidenceError,
    MissingnessInformativeWarning,
    StandardisationStats,
    as_validity,
    fabricated_edge,
    fill_mean,
    fill_zero,
    is_contrast_filter,
    masked_mean,
    masked_standardise,
    masked_standardise_fit,
    masked_std,
    missingness_association,
    observation_support,
    renormalised_convolve1d,
    renormalised_dot,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # config
    "RSMConfig",
    "fit_count",
    # reproducibility
    "Seeds",
    "set_global_seed",
    "BOOTSTRAP_OFFSET",
    "REPRESENTATION_OFFSET",
    "INNER_OFFSET",
    "REPEAT_STRIDE",
    # validity
    "InsufficientEvidenceError",
    "MissingnessInformativeWarning",
    "StandardisationStats",
    "as_validity",
    "masked_mean",
    "masked_std",
    "masked_standardise_fit",
    "masked_standardise",
    "renormalised_convolve1d",
    "renormalised_dot",
    "is_contrast_filter",
    "observation_support",
    "missingness_association",
    "fill_zero",
    "fill_mean",
    "fabricated_edge",
    # segments
    "SegmentGrid",
    # masking
    "SoftMaskSelector",
    "renormalised_gated_forward",
    "mask_data_gradient",
    "loss_and_dz",
    # representations
    "SegmentMean",
    "SegmentStats",
    "SavGolDerivative",
    "GaussianScale1D",
    "RandomConv1D",
    "default_ensemble",
    # nested CV and results
    "NestedCV",
    "FoldEstimator",
    "SoftMaskFoldEstimator",
    "BootstrapMaskFoldEstimator",
    "FullSignalFoldEstimator",
    "infer_task",
    "make_outer_splitter",
    "make_inner_splitter",
    "RSMResult",
    # refit model zoo
    "ALGORITHMS",
    "make_refit_model",
    "PLSDA",
    "ExternalRefit",
    "refit_features",
    # selection
    "BootstrapMaskSelector",
    "RepresentationEnsembleSelector",
    "EnsembleMaskFoldEstimator",
    "resample_indices",
    "complementary_pair",
    "observation_groups",
    # tuning
    "lam_frontier",
    "lam_for_coverage",
    # synthetic
    "SignalControl",
    "make_signal_control",
    "mask_scattered",
    "mask_dropout_stretches",
    "mask_band_by_group",
    "mask_saturation_censor",
    "drop_labels",
    # metrics
    "score_predictions",
    "rmse_from_score",
    "jaccard",
    "expected_jaccard",
    "adjusted_jaccard",
    "mean_pairwise_jaccard",
    "BaselineComparison",
    "paired_comparison",
]
