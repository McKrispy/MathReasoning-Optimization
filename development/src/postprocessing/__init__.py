"""Postprocessing namespace: calibration, aggregation, masking and fusion (POST-P01 ~ POST-P07)."""
from .calibration import TemperatureScaler
from .aggregation import WeightedSelfConsistencyAggregator, apply_disagreement_gate
from .masking import (
    SoftmaxClusterAggregator,
    apply_conflict_penalty,
    apply_validity_mask,
    detect_parser_heading_conflict,
)

from .soft_stacking import (
    SoftStackingEnsemble,
    fit_convex_combination_weights,
    standardize_within_problem,
)

__all__ = [
    "TemperatureScaler",
    "WeightedSelfConsistencyAggregator",
    "apply_disagreement_gate",
    "SoftmaxClusterAggregator",
    "apply_conflict_penalty",
    "apply_validity_mask",
    "detect_parser_heading_conflict",
    "standardize_within_problem",
    "fit_convex_combination_weights",
    "SoftStackingEnsemble",
]

