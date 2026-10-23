"""Synthetic fail/pass cases for the Track 1 stacking gates.

Rows are constructed in this file. Nothing here is competition data.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from bhi_gates import (
    DELTA_WMAE_MAX,
    PRED_CORR_MAX,
    RESID_CORR_MAX,
    hard_pass,
    hard_pass_oof,
    pearson_corr,
    residual_corr,
    sbp140_mae_bias,
    weighted_mae,
    worst_quartile_subject_mae,
)


def _panel(n_subjects: int = 8, n_per: int = 25, n_high: int = 5, high_level: float = 155.0):
    """Deterministic BP panel with a hypertensive block at the end of each subject."""
    ids = np.repeat(np.arange(n_subjects), n_per)
    sbp = np.empty(n_subjects * n_per, dtype=float)
    n_normal = n_per - n_high
    for subject in range(n_subjects):
        start = subject * n_per
        normal = 120.0 + np.linspace(-3.0, 3.0, n_normal)
        high = high_level + np.linspace(-2.0, 2.0, n_high)
        sbp[start : start + n_per] = np.concatenate([normal, high])
    dbp = 75.0 + 0.25 * (sbp - 120.0)
    y = np.column_stack([sbp, dbp])
    return y, ids, sbp


def _wave(n: int, amp: float, freq: float, phase: float) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    return np.column_stack(
        [
            amp * np.sin(freq * idx + phase),
            amp * np.cos(freq * 0.83 * idx + phase * 1.3),
        ]
    )


def _failed(result: dict) -> set[str]:
    names = set()
    for name, check in result["checks"].items():
        if isinstance(check, dict) and check.get("passed") is False:
            names.add(name)
    return names


class MetricSmokeTest(unittest.TestCase):
    def test_weighted_mae_matches_published_formula(self) -> None:
        y = np.array([[120.0, 80.0], [140.0, 90.0]])
        pred = np.array([[122.0, 81.0], [145.0, 88.0]])
        # 0.6*|2| + 0.4*|1| = 1.6 and 0.6*|5| + 0.4*|2| = 3.8; mean is 2.7.
        self.assertEqual(weighted_mae(y, pred), 2.7)

    def test_sbp140_includes_boundary_and_reports_signed_bias(self) -> None:
        truth = np.array([120.0, 139.0, 140.0, 150.0])
        pred = np.array([118.0, 130.0, 136.0, 140.0])
        out = sbp140_mae_bias(truth, pred)
        # 139 is excluded. Errors on 140 and 150 are 4 and 10 mmHg, both low.
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["mae"], 7.0)
        self.assertEqual(out["bias"], -7.0)
        empty = sbp140_mae_bias(np.array([120.0, 130.0]), np.array([121.0, 129.0]))
        self.assertEqual(empty["n"], 0)
        self.assertTrue(math.isnan(empty["mae"]))
        self.assertTrue(math.isnan(empty["bias"]))

    def test_worst_quartile_means_the_worst_ceil_quarter(self) -> None:
        # Equal SBP and DBP absolute errors make WMAE equal to that error.
        four = np.array([1.0, 2.0, 3.0, 4.0])
        y4 = np.column_stack([np.full(4, 120.0), np.full(4, 80.0)])
        p4 = np.column_stack([120.0 + four, 80.0 + four])
        self.assertAlmostEqual(
            worst_quartile_subject_mae(y4, p4, np.arange(4)),
            4.0,
            places=9,
        )
        eight = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 8.0])
        y8 = np.column_stack([np.full(8, 120.0), np.full(8, 80.0)])
        p8 = np.column_stack([120.0 + eight, 80.0 + eight])
        # ceil(8/4) = 2, mean of 4 and 8.
        self.assertAlmostEqual(
            worst_quartile_subject_mae(y8, p8, np.arange(8)),
            6.0,
            places=9,
        )

    def test_subject_mae_averages_rows_inside_the_subject(self) -> None:
        y = np.array([[120.0, 80.0], [120.0, 80.0]])
        pred = np.array([[130.0, 80.0], [120.0, 90.0]])
        # Row losses are 0.6*10 and 0.4*10. One subject, so the tail is their mean.
        self.assertAlmostEqual(worst_quartile_subject_mae(y, pred, np.array(["a", "a"])), 5.0)

    def test_pearson_and_residual_correlation(self) -> None:
        left = np.array([1.0, 2.0, 3.0, 4.0])
        right = np.array([2.0, 4.0, 6.0, 8.0])
        self.assertAlmostEqual(pearson_corr(left, right), 1.0)
        y = np.column_stack([np.arange(10.0), np.arange(10.0)])
        error = np.column_stack([np.sin(np.arange(10.0)), np.cos(np.arange(10.0))])
        self.assertAlmostEqual(residual_corr(y, y + error, y - error), -1.0, places=9)
        self.assertAlmostEqual(residual_corr(y, y + error, y + error), 1.0, places=9)

    def test_constant_input_correlation_is_undefined(self) -> None:
        self.assertTrue(math.isnan(pearson_corr(np.ones(5), np.arange(5.0))))

    def test_weights_must_sum_to_one(self) -> None:
        y = np.array([[120.0, 80.0], [130.0, 85.0]])
        pred = np.array([[121.0, 81.0], [128.0, 86.0]])
        with self.assertRaises(ValueError):
            weighted_mae(y, pred, sbp_weight=0.6, dbp_weight=0.6)


class GateSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.y, self.ids, self.sbp = _panel()
        self.n = len(self.y)
        self.assertGreaterEqual(int(np.count_nonzero(self.sbp >= 140.0)), 1)

    def test_published_thresholds(self) -> None:
        self.assertEqual(PRED_CORR_MAX, 0.98)
        self.assertEqual(RESID_CORR_MAX, 0.7)
        self.assertEqual(DELTA_WMAE_MAX, -0.01)

    def test_promotes_when_every_gate_passes(self) -> None:
        champ = self.y + _wave(self.n, 8.0, 0.37, 0.2)
        candidate = self.y + _wave(self.n, 2.5, 0.91, 1.7)
        result = hard_pass(champ, candidate, self.y, self.ids, self.sbp)
        self.assertTrue(result["promote"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(_failed(result), set())
        self.assertLess(result["checks"]["pred_corr"]["value"], 0.98)
        self.assertLess(result["checks"]["resid_corr"]["value"], 0.7)
        self.assertLess(result["checks"]["delta_wmae"]["value"], -1.0)
        self.assertIsInstance(result["promote"], bool)

    def test_rejects_near_duplicate_predictions(self) -> None:
        champ = self.y + _wave(self.n, 1.5, 0.37, 0.1)
        candidate = self.y + _wave(self.n, 0.4, 0.37, 1.2)
        result = hard_pass(champ, candidate, self.y, self.ids, self.sbp)
        self.assertEqual(_failed(result), {"pred_corr"})
        self.assertGreater(result["checks"]["pred_corr"]["value"], 0.99)
        self.assertTrue(any("pred_corr" in reason for reason in result["reasons"]))

    def test_rejects_shared_residuals(self) -> None:
        error = _wave(self.n, 30.0, 0.37, 0.4)
        result = hard_pass(self.y + error, self.y + 0.45 * error, self.y, self.ids, self.sbp)
        self.assertEqual(_failed(result), {"resid_corr"})
        self.assertGreater(result["checks"]["resid_corr"]["value"], 0.7)
        self.assertTrue(any("resid_corr" in reason for reason in result["reasons"]))

    def test_rejects_wmae_gain_smaller_than_one_hundredth(self) -> None:
        error = _wave(self.n, 6.0, 0.37, 0.0)
        # Opposite residuals keep the models diverse, and 0.2% smaller errors
        # improve WMAE by less than the required 0.01 mmHg.
        result = hard_pass(self.y + error, self.y - 0.998 * error, self.y, self.ids, self.sbp)
        self.assertEqual(_failed(result), {"delta_wmae"})
        delta = result["checks"]["delta_wmae"]["value"]
        self.assertLess(delta, 0.0)
        self.assertGreater(delta, -0.01)
        self.assertTrue(any("delta_wmae" in reason for reason in result["reasons"]))

    def test_accepts_wmae_gain_of_exactly_one_hundredth(self) -> None:
        error = _wave(self.n, 6.0, 0.37, 0.0)
        champ = self.y + error
        scale = 1.0 - 0.01 / weighted_mae(self.y, champ)
        result = hard_pass(champ, self.y - scale * error, self.y, self.ids, self.sbp)
        self.assertTrue(result["promote"])
        self.assertAlmostEqual(result["checks"]["delta_wmae"]["value"], -0.01, places=9)

    def test_rejects_worse_hypertensive_sbp(self) -> None:
        champ = self.y + _wave(self.n, 8.0, 0.37, 0.2)
        candidate = self.y + _wave(self.n, 2.0, 0.91, 1.7)
        candidate = candidate.copy()
        high = self.sbp >= 140.0
        candidate[high, 0] = self.sbp[high] - 14.0
        result = hard_pass(champ, candidate, self.y, self.ids, self.sbp)
        self.assertEqual(_failed(result), {"sbp140"})
        self.assertGreater(
            result["checks"]["sbp140"]["mae_candidate"],
            result["checks"]["sbp140"]["mae_champ"],
        )
        self.assertEqual(result["checks"]["sbp140"]["bias_candidate"], -14.0)
        self.assertTrue(any("sbp140" in reason for reason in result["reasons"]))

    def test_rejects_worse_subject_tail(self) -> None:
        champ = self.y + _wave(self.n, 8.0, 0.37, 0.2)
        candidate = self.y + _wave(self.n, 2.0, 0.91, 1.7)
        candidate = candidate.copy()
        sacrificed = self.ids == 0
        candidate[sacrificed] = self.y[sacrificed] + _wave(self.n, 18.0, 1.31, 2.4)[sacrificed]
        result = hard_pass(champ, candidate, self.y, self.ids, self.sbp)
        self.assertEqual(_failed(result), {"worst_quartile"})
        self.assertGreater(
            result["checks"]["worst_quartile"]["candidate"],
            result["checks"]["worst_quartile"]["champ"],
        )
        self.assertTrue(any("worst_quartile" in reason for reason in result["reasons"]))

    def test_identical_champion_fails_similarity_and_gain_only(self) -> None:
        champ = self.y + _wave(self.n, 8.0, 0.37, 0.2)
        result = hard_pass(champ, champ.copy(), self.y, self.ids, self.sbp)
        self.assertEqual(_failed(result), {"pred_corr", "resid_corr", "delta_wmae"})
        self.assertEqual(len(result["reasons"]), 3)
        # Matching subset errors satisfy "not worse".
        self.assertTrue(result["checks"]["sbp140"]["passed"])
        self.assertTrue(result["checks"]["worst_quartile"]["passed"])

    def test_empty_hypertensive_subset_fails_closed(self) -> None:
        y, ids, sbp = _panel(high_level=125.0)
        self.assertLess(float(sbp.max()), 140.0)
        n = len(y)
        champ = y + _wave(n, 8.0, 0.37, 0.2)
        candidate = y + _wave(n, 2.5, 0.91, 1.7)
        result = hard_pass(champ, candidate, y, ids, sbp)
        self.assertEqual(_failed(result), {"sbp140"})
        self.assertEqual(result["checks"]["sbp140"]["n"], 0)
        self.assertTrue(any("empty" in reason for reason in result["reasons"]))

    def test_oof_frames_match_the_array_decision(self) -> None:
        champ = self.y + _wave(self.n, 8.0, 0.37, 0.2)
        candidate = self.y + _wave(self.n, 2.5, 0.91, 1.7)
        labels = np.array([f"sub-{i:02d}" for i in self.ids])
        champ_df = pd.DataFrame(
            {
                "subject_id": labels,
                "sbp_true": self.y[:, 0],
                "dbp_true": self.y[:, 1],
                "sbp_pred": champ[:, 0],
                "dbp_pred": champ[:, 1],
                "note": "ignored",
            }
        )
        candidate_df = pd.DataFrame(
            {
                "subject_id": labels,
                "sbp_true": self.y[:, 0],
                "dbp_true": self.y[:, 1],
                "sbp_pred": candidate[:, 0],
                "dbp_pred": candidate[:, 1],
            }
        )
        from_frames = hard_pass_oof(champ_df, candidate_df)
        from_arrays = hard_pass(champ, candidate, self.y, labels, self.sbp)
        self.assertEqual(from_frames["promote"], from_arrays["promote"])
        self.assertEqual(from_frames["reasons"], from_arrays["reasons"])
        self.assertTrue(from_frames["promote"])

    def test_oof_rejects_misaligned_reference_and_missing_columns(self) -> None:
        champ = self.y + _wave(self.n, 8.0, 0.37, 0.2)
        frame = pd.DataFrame(
            {
                "subject_id": self.ids,
                "sbp_true": self.y[:, 0],
                "dbp_true": self.y[:, 1],
                "sbp_pred": champ[:, 0],
                "dbp_pred": champ[:, 1],
            }
        )
        shifted = frame.copy()
        shifted.loc[0, "sbp_true"] = float(shifted.loc[0, "sbp_true"]) + 1.0
        with self.assertRaises(ValueError):
            hard_pass_oof(frame, shifted)
        with self.assertRaises(ValueError):
            hard_pass_oof(frame.drop(columns=["dbp_pred"]), frame)

    def test_sbp_argument_must_match_reference_column(self) -> None:
        champ = self.y + _wave(self.n, 8.0, 0.37, 0.2)
        candidate = self.y + _wave(self.n, 2.5, 0.91, 1.7)
        with self.assertRaises(ValueError):
            hard_pass(champ, candidate, self.y, self.ids, self.sbp + 1.0)
        with self.assertRaises(ValueError):
            hard_pass(champ[:1], candidate[:1], self.y[:1], self.ids[:1], self.sbp[:1])


if __name__ == "__main__":
    unittest.main()
