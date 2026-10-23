"""Promotion gates for adding a candidate model to a Track 1 stack.

A candidate is promoted only when every hard check passes. Checks are diversity
(prediction correlation and residual correlation), a minimum gain on the
official-style weighted MAE, and no degradation on the SBP >= 140 mmHg subset
or on the worst-quartile subject WMAE. All five checks run even after one has
failed, so ``reasons`` lists every blocker.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    SBP_SUBSET_MMHG,
    pearson_corr,
    residual_corr,
    sbp140_mae_bias,
    weighted_mae,
    worst_quartile_subject_mae,
)

PRED_CORR_MAX = 0.98
RESID_CORR_MAX = 0.7
DELTA_WMAE_MAX = -0.01
_ATOL = 1e-8

OOF_COLUMNS = (
    "subject_id",
    "sbp_true",
    "dbp_true",
    "sbp_pred",
    "dbp_pred",
)

ArrayLike = Any


def hard_pass(
    champ: ArrayLike,
    candidate: ArrayLike,
    y: ArrayLike,
    subject_ids: ArrayLike,
    sbp: ArrayLike,
    *,
    pred_corr_max: float = PRED_CORR_MAX,
    resid_corr_max: float = RESID_CORR_MAX,
    delta_wmae_max: float = DELTA_WMAE_MAX,
    sbp_threshold: float = SBP_SUBSET_MMHG,
) -> dict[str, Any]:
    """Decide whether ``candidate`` clears the hard stacking gates.

    Parameters
    ----------
    champ, candidate:
        Out-of-fold predictions, shape ``(n, 2)``, columns ``[SBP, DBP]`` in
        mmHg, row-aligned with ``y``.
    y:
        Reference BP, shape ``(n, 2)``, columns ``[SBP, DBP]`` in mmHg.
    subject_ids:
        Length-``n`` subject labels. Repeated ids mark rows from one subject.
    sbp:
        Reference systolic pressure in mmHg, length ``n``. This is the series
        the hypertensive subset is cut from. It must match ``y[:, 0]`` within
        1e-4 mmHg.

    Returns
    -------
    dict
        ``promote`` is True only when every check passes. ``reasons`` is empty
        in that case, and otherwise lists one string per failed check.
        ``checks`` stores the underlying numbers.

    Thresholds, all of which must hold:

    * ``pred_corr <= 0.98`` — champion and candidate predictions are not
      near-duplicates. Correlation is Pearson r on concatenated SBP and DBP
      pairs.
    * ``resid_corr <= 0.7`` — residuals ``y - pred`` are not the same errors.
    * ``delta_wmae <= -0.01`` — candidate WMAE minus champion WMAE, in mmHg.
      The candidate must improve the official-style loss by at least 0.01.
    * SBP >= 140 MAE is not worse (candidate MAE <= champion MAE). Bias is
      reported and is not an extra blocker; the challenge loss is absolute
      error, and "not worse" is applied to that MAE.
    * Worst-quartile subject WMAE is not worse (candidate <= champion).

    An undefined correlation, or an empty SBP >= 140 subset, fails closed.
    "Not worse" and the numeric inequalities allow an absolute tolerance of
    1e-8 so binary floating-point noise does not flip a true tie.
    """
    champ_bp = _as_bp(champ, "champ")
    cand_bp = _as_bp(candidate, "candidate")
    truth = _as_bp(y, "y")
    if champ_bp.shape != cand_bp.shape or champ_bp.shape != truth.shape:
        raise ValueError(
            "champ, candidate, and y must share shape (n, 2), "
            f"got {champ_bp.shape}, {cand_bp.shape}, {truth.shape}"
        )
    n = truth.shape[0]
    if n < 2:
        raise ValueError("hard_pass requires at least two samples")
    ids = np.asarray(subject_ids)
    if ids.ndim != 1 or ids.shape[0] != n:
        raise ValueError(f"subject_ids must have shape ({n},), got {ids.shape}")
    sbp_ref = np.asarray(sbp, dtype=float)
    if sbp_ref.ndim != 1 or sbp_ref.shape[0] != n:
        raise ValueError(f"sbp must have shape ({n},), got {getattr(sbp_ref, 'shape', None)}")
    if not np.all(np.isfinite(sbp_ref)):
        raise ValueError("sbp must contain only finite values")
    if not np.allclose(sbp_ref, truth[:, 0], rtol=0.0, atol=1e-4):
        raise ValueError("sbp must match y[:, 0] (reference systolic pressure, mmHg)")

    pred_c = pearson_corr(champ_bp, cand_bp)
    resid_c = residual_corr(truth, champ_bp, cand_bp)
    wmae_champ = weighted_mae(truth, champ_bp)
    wmae_cand = weighted_mae(truth, cand_bp)
    delta = float(wmae_cand - wmae_champ)
    high_champ = sbp140_mae_bias(sbp_ref, champ_bp[:, 0], threshold=sbp_threshold)
    high_cand = sbp140_mae_bias(sbp_ref, cand_bp[:, 0], threshold=sbp_threshold)
    worst_champ = worst_quartile_subject_mae(truth, champ_bp, ids)
    worst_cand = worst_quartile_subject_mae(truth, cand_bp, ids)

    pred_ok = _le(pred_c, pred_corr_max)
    resid_ok = _le(resid_c, resid_corr_max)
    delta_ok = _le(delta, delta_wmae_max)
    if high_champ["n"] == 0 or high_cand["n"] == 0:
        sbp_ok = False
    else:
        sbp_ok = _not_worse(float(high_cand["mae"]), float(high_champ["mae"]))
    worst_ok = _not_worse(worst_cand, worst_champ)

    reasons: list[str] = []
    if not pred_ok:
        reasons.append(
            f"pred_corr {_fmt(pred_c)} > {pred_corr_max:g} "
            "(candidate predictions are too similar to the champion)"
        )
    if not resid_ok:
        reasons.append(
            f"resid_corr {_fmt(resid_c)} > {resid_corr_max:g} "
            "(candidate residuals track the champion residuals)"
        )
    if not delta_ok:
        reasons.append(
            f"delta_wmae {_fmt(delta)} > {delta_wmae_max:g} "
            "(candidate WMAE improvement is short of the required margin)"
        )
    if not sbp_ok:
        if high_champ["n"] == 0:
            reasons.append(
                f"sbp140 subset is empty at reference SBP >= {sbp_threshold:g}; "
                "refusing to treat it as not worse"
            )
        else:
            reasons.append(
                "sbp140 MAE worsened from "
                f"{_fmt(high_champ['mae'])} to {_fmt(high_cand['mae'])} mmHg"
            )
    if not worst_ok:
        reasons.append(
            "worst_quartile subject MAE worsened from "
            f"{_fmt(worst_champ)} to {_fmt(worst_cand)} mmHg"
        )

    return {
        "promote": len(reasons) == 0,
        "reasons": reasons,
        "checks": {
            "pred_corr": {
                "value": pred_c,
                "op": "<=",
                "threshold": float(pred_corr_max),
                "passed": pred_ok,
            },
            "resid_corr": {
                "value": resid_c,
                "op": "<=",
                "threshold": float(resid_corr_max),
                "passed": resid_ok,
            },
            "delta_wmae": {
                "value": delta,
                "op": "<=",
                "threshold": float(delta_wmae_max),
                "passed": delta_ok,
                "wmae_champ": wmae_champ,
                "wmae_candidate": wmae_cand,
            },
            "sbp140": {
                "mae_champ": high_champ["mae"],
                "mae_candidate": high_cand["mae"],
                "bias_champ": high_champ["bias"],
                "bias_candidate": high_cand["bias"],
                "n": int(high_champ["n"]),
                "passed": sbp_ok,
            },
            "worst_quartile": {
                "champ": worst_champ,
                "candidate": worst_cand,
                "passed": worst_ok,
            },
            "n_samples": n,
            "n_subjects": int(np.unique(ids).size),
        },
    }


def hard_pass_oof(
    champ_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ``hard_pass`` on two row-aligned OOF tables.

    Both frames need the columns ``subject_id``, ``sbp_true``, ``dbp_true``,
    ``sbp_pred``, and ``dbp_pred``. Reference columns must match row for row.
    Extra columns are ignored. ``kwargs`` are forwarded to ``hard_pass``.
    """
    champ_df = _require_oof(champ_df, "champ_df")
    candidate_df = _require_oof(candidate_df, "candidate_df")
    if len(champ_df) != len(candidate_df):
        raise ValueError(
            "OOF frames must have the same number of rows, "
            f"got {len(champ_df)} and {len(candidate_df)}"
        )
    if not np.array_equal(
        champ_df["subject_id"].to_numpy(),
        candidate_df["subject_id"].to_numpy(),
    ):
        raise ValueError("subject_id is not row-aligned between champion and candidate")
    for column in ("sbp_true", "dbp_true"):
        left = champ_df[column].to_numpy(dtype=float)
        right = candidate_df[column].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=0.0, atol=1e-4, equal_nan=False):
            raise ValueError(
                f"{column} is not row-aligned between champion and candidate"
            )
    y = champ_df[["sbp_true", "dbp_true"]].to_numpy(dtype=float)
    return hard_pass(
        champ_df[["sbp_pred", "dbp_pred"]].to_numpy(dtype=float),
        candidate_df[["sbp_pred", "dbp_pred"]].to_numpy(dtype=float),
        y,
        champ_df["subject_id"].to_numpy(),
        y[:, 0],
        **kwargs,
    )


def _as_bp(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2) [SBP, DBP], got {arr.shape}")
    if arr.size > 0 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _le(value: float, limit: float) -> bool:
    return bool(np.isfinite(value) and value <= float(limit) + _ATOL)


def _not_worse(candidate: float, champion: float) -> bool:
    """True when the candidate error summary is finite and not higher."""
    return bool(
        np.isfinite(candidate)
        and np.isfinite(champion)
        and candidate <= champion + _ATOL
    )


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "undefined"
    if not np.isfinite(number):
        return "undefined"
    return f"{number:.6f}"


def _require_oof(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = [column for column in OOF_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; expected {list(OOF_COLUMNS)}")
    return df
