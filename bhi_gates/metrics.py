"""Sample and subject metrics for BHI 2026 Track 1 stacking checks.

``weighted_mae`` is an official-style stub of the published Track 1 loss.
Assumptions (read before treating a number as a leaderboard score):

* The published sample-level formula is used as stated for the BHI 2026
  Data Challenge, Track 1 (snapshot cuffless BP):

      MAE = (1 / N) * sum_i (0.6 * |SBP_i - SBP_hat_i| + 0.4 * |DBP_i - DBP_hat_i|)

  Weights are 0.6 on systolic error and 0.4 on diastolic error. They already
  sum to 1 and are applied directly. Lower is better. Units are mmHg.

* Each row is one scored sample and every row has equal weight. The stub does
  not reweight subjects, windows, or recordings inside this mean.

* Column 0 is SBP and column 1 is DBP. A single sample must be shaped
  ``(1, 2)``. Inputs must be finite; nothing is imputed or clipped.

* The leaderboard map ``1 / (1 + MAE)`` is not applied. Deltas elsewhere in
  this package are raw mmHg on this MAE.

* No calibration term, auscultatory-reference adjustment, or physiological
  exclusion rule is applied. Public versus private split membership is unknown
  at OOF time and is not modeled.

* This is not the organizer scoring binary. Rounding, file format checks, and
  any undisclosed tie-break are outside the stub.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.stats import ConstantInputWarning, NearConstantInputWarning, pearsonr

SBP_WEIGHT = 0.6
DBP_WEIGHT = 0.4
SBP_SUBSET_MMHG = 140.0

ArrayLike = Any


def weighted_mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sbp_weight: float = SBP_WEIGHT,
    dbp_weight: float = DBP_WEIGHT,
) -> float:
    """Return the sample-level weighted MAE in mmHg.

    Parameters
    ----------
    y_true, y_pred:
        Shape ``(n, 2)`` arrays of ``[SBP, DBP]`` in mmHg.
    sbp_weight, dbp_weight:
        Channel weights. The defaults are the published 0.6 and 0.4 and must
        sum to 1 so the result stays on the published mmHg scale.
    """
    true = _as_bp(y_true, "y_true")
    pred = _as_bp(y_pred, "y_pred")
    _same_rows(true, pred, "y_true", "y_pred")
    if true.shape[0] == 0:
        raise ValueError("weighted_mae requires at least one sample")
    _check_weights(sbp_weight, dbp_weight)
    abs_err = np.abs(true - pred)
    per_sample = sbp_weight * abs_err[:, 0] + dbp_weight * abs_err[:, 1]
    return float(np.mean(per_sample))


def pearson_corr(a: ArrayLike, b: ArrayLike) -> float:
    """Pearson product-moment correlation between two paired arrays.

    One-dimensional arrays are compared entrywise. Shape ``(n, 2)`` BP arrays
    are raveled in C order first, so each SBP pair and each DBP pair enters a
    single correlation. That one coefficient is what the stacking gate
    thresholds. At least two finite pairs are required. A constant input has
    an undefined correlation; this returns NaN rather than raising, and the
    hard gate treats NaN as a failed check.
    """
    left = _paired_vector(a, "a")
    right = _paired_vector(b, "b")
    if left.shape != right.shape:
        raise ValueError(
            f"pearson_corr inputs must have the same shape after pairing, "
            f"got {left.shape} and {right.shape}"
        )
    if left.size < 2:
        raise ValueError("pearson_corr requires at least two paired values")
    if np.min(left) == np.max(left) or np.min(right) == np.max(right):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        warnings.simplefilter("ignore", category=NearConstantInputWarning)
        result = pearsonr(left, right)
    statistic = result.statistic if hasattr(result, "statistic") else result[0]
    return float(statistic)


def residual_corr(y: ArrayLike, pred_a: ArrayLike, pred_b: ArrayLike) -> float:
    """Pearson correlation of residuals ``y - pred_a`` and ``y - pred_b``.

    Shapes follow ``pearson_corr``. For BP arrays the residual is taken in
    mmHg on both SBP and DBP, then concatenated the same way as predictions.
    """
    truth = np.asarray(y, dtype=float)
    first = np.asarray(pred_a, dtype=float)
    second = np.asarray(pred_b, dtype=float)
    if truth.shape != first.shape or truth.shape != second.shape:
        raise ValueError(
            "residual_corr requires y, pred_a, and pred_b to share a shape, "
            f"got {truth.shape}, {first.shape}, {second.shape}"
        )
    return pearson_corr(truth - first, truth - second)


def sbp140_mae_bias(
    sbp_true: ArrayLike,
    sbp_pred: ArrayLike,
    *,
    threshold: float = SBP_SUBSET_MMHG,
) -> dict[str, float | int]:
    """MAE and bias on the reference-SBP subset ``sbp_true >= threshold``.

    Bias is ``mean(pred - true)`` in mmHg on that subset (negative means the
    model under-reads hypertensive systolic pressure). MAE is
    ``mean(|pred - true|)`` in mmHg. Rows below ``threshold`` are ignored.
    When the subset is empty, ``mae`` and ``bias`` are NaN and ``n`` is 0.
    """
    truth = _finite_vector(sbp_true, "sbp_true")
    pred = _finite_vector(sbp_pred, "sbp_pred")
    if truth.shape != pred.shape:
        raise ValueError(
            f"sbp_true and sbp_pred must have the same length, "
            f"got {truth.shape[0]} and {pred.shape[0]}"
        )
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    mask = truth >= float(threshold)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return {"mae": float("nan"), "bias": float("nan"), "n": 0}
    signed = pred[mask] - truth[mask]
    return {
        "mae": float(np.mean(np.abs(signed))),
        "bias": float(np.mean(signed)),
        "n": n,
    }


def worst_quartile_subject_mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    subject_ids: ArrayLike,
    *,
    sbp_weight: float = SBP_WEIGHT,
    dbp_weight: float = DBP_WEIGHT,
) -> float:
    """Mean per-subject WMAE among the worst ``ceil(n_subjects / 4)`` subjects.

    The per-subject score is ``weighted_mae`` on that subject's rows (published
    SBP/DBP weights, equal weight per row inside the subject). Subjects are
    then ranked by that score. The returned value is the mean of the worst
    ``k = max(1, ceil(0.25 * n_subjects))`` scores. One subject therefore
    contributes its own WMAE. Lower is better.

    This is a tail summary of the same loss as ``weighted_mae``, not a
    paired-subject contrast and not an SBP-only MAE.
    """
    true = _as_bp(y_true, "y_true")
    pred = _as_bp(y_pred, "y_pred")
    _same_rows(true, pred, "y_true", "y_pred")
    if true.shape[0] == 0:
        raise ValueError("worst_quartile_subject_mae requires at least one sample")
    ids = _subject_ids(subject_ids, true.shape[0])
    _check_weights(sbp_weight, dbp_weight)
    unique_ids = np.unique(ids)
    scores = np.empty(unique_ids.size, dtype=float)
    for i, sid in enumerate(unique_ids):
        mask = ids == sid
        scores[i] = weighted_mae(
            true[mask],
            pred[mask],
            sbp_weight=sbp_weight,
            dbp_weight=dbp_weight,
        )
    k = max(1, int(np.ceil(scores.size * 0.25)))
    worst = np.sort(scores)[-k:]
    return float(np.mean(worst))


def _as_bp(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2) [SBP, DBP], got {arr.shape}")
    if arr.shape[0] > 0 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _paired_vector(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2 and arr.shape[1] == 2:
        vec = np.reshape(arr, -1)
    else:
        raise ValueError(f"{name} must have shape (n,) or (n, 2), got {arr.shape}")
    if vec.size > 0 and not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must contain only finite values")
    return vec


def _finite_vector(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-d array, got shape {arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _subject_ids(subject_ids: ArrayLike, n: int) -> np.ndarray:
    ids = np.asarray(subject_ids)
    if ids.ndim != 1:
        raise ValueError(f"subject_ids must be 1-d, got shape {ids.shape}")
    if ids.shape[0] != n:
        raise ValueError(
            f"subject_ids length {ids.shape[0]} does not match n={n}"
        )
    if ids.dtype.kind == "f" and np.any(np.isnan(ids)):
        raise ValueError("subject_ids must not contain NaN")
    return ids


def _same_rows(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> None:
    if left.shape[0] != right.shape[0]:
        raise ValueError(
            f"{left_name} and {right_name} must have the same number of rows, "
            f"got {left.shape[0]} and {right.shape[0]}"
        )


def _check_weights(sbp_weight: float, dbp_weight: float) -> None:
    if sbp_weight < 0 or dbp_weight < 0:
        raise ValueError("SBP and DBP weights must be non-negative")
    if not np.isclose(sbp_weight + dbp_weight, 1.0):
        raise ValueError(
            "SBP and DBP weights must sum to 1 so WMAE stays on the published mmHg scale; "
            f"got {sbp_weight} + {dbp_weight}"
        )
