"""Hard gates for stacking cuffless BP models on BHI 2026 Track 1 OOF predictions.

The weighted MAE implemented here is a local stub of the published sample-level
formula. It is not the competition scorer.
"""

from .gates import (
    DELTA_WMAE_MAX,
    OOF_COLUMNS,
    PRED_CORR_MAX,
    RESID_CORR_MAX,
    hard_pass,
    hard_pass_oof,
)
from .metrics import (
    SBP_WEIGHT,
    DBP_WEIGHT,
    pearson_corr,
    residual_corr,
    sbp140_mae_bias,
    weighted_mae,
    worst_quartile_subject_mae,
)

__all__ = [
    "DBP_WEIGHT",
    "DELTA_WMAE_MAX",
    "OOF_COLUMNS",
    "PRED_CORR_MAX",
    "RESID_CORR_MAX",
    "SBP_WEIGHT",
    "hard_pass",
    "hard_pass_oof",
    "pearson_corr",
    "residual_corr",
    "sbp140_mae_bias",
    "weighted_mae",
    "worst_quartile_subject_mae",
]

__version__ = "0.1.0"
