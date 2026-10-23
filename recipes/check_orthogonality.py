#!/usr/bin/env python3
"""Print residual-orthogonality metrics for two OOF prediction columns.

Reads a CSV path supplied by the caller. This stub does not embed, download,
or synthesize a competition dataset.

Convention: --yhat-a is the candidate and --yhat-b is the champion.
Printed lines:

    corr_yhat    Pearson correlation of the two predictions
    corr_resid   Pearson correlation of (y - yhat_a) and (y - yhat_b)
    delta_wmae   WMAE(yhat_a) - WMAE(yhat_b)

WMAE is sum(w * |y - yhat|) / sum(w). With --weight, w is that column and
must be finite and positive. Otherwise every kept row has weight 1, so
delta_wmae is an unweighted MAE difference. A negative delta means the
candidate is closer to y than the champion. A constant vector prints nan
for its correlation (a zero residual has undefined Pearson correlation).

Usage:
  python recipes/check_orthogonality.py PATH_TO_YOUR_OOF.csv \\
      --yhat-a yhat_new --yhat-b yhat_champ --y y_true
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def wmae(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    denom = float(np.sum(w))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(w * np.abs(y - yhat)) / denom)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print corr_yhat, corr_resid, and delta_wmae for two OOF columns."
    )
    parser.add_argument("csv", help="Caller-supplied OOF CSV path.")
    parser.add_argument("--yhat-a", default="yhat_a", help="Candidate prediction column.")
    parser.add_argument("--yhat-b", default="yhat_b", help="Champion prediction column.")
    parser.add_argument("--y", default="y_true", help="Label column.")
    parser.add_argument("--weight", default=None, help="Optional positive row-weight column.")
    args = parser.parse_args(argv)

    needed = [args.yhat_a, args.yhat_b, args.y]
    if args.weight:
        needed.append(args.weight)
    try:
        frame = pd.read_csv(args.csv, usecols=needed)
    except ValueError as exc:
        print(f"column error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"read error: {exc}", file=sys.stderr)
        return 2

    preds = frame[[args.yhat_a, args.yhat_b, args.y]].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(preds.to_numpy(dtype=float)).all(axis=1)
    if args.weight:
        weights = pd.to_numeric(frame[args.weight], errors="coerce").to_numpy(dtype=float)
        mask = mask & np.isfinite(weights) & (weights > 0.0)
    else:
        weights = np.ones(len(frame), dtype=float)
    if int(mask.sum()) < 3:
        print("fewer than 3 finite rows", file=sys.stderr)
        return 2

    yhat_a = preds.loc[mask, args.yhat_a].to_numpy(dtype=float)
    yhat_b = preds.loc[mask, args.yhat_b].to_numpy(dtype=float)
    y_true = preds.loc[mask, args.y].to_numpy(dtype=float)
    w = weights[mask]

    corr_yhat = pearson(yhat_a, yhat_b)
    corr_resid = pearson(y_true - yhat_a, y_true - yhat_b)
    delta = wmae(y_true, yhat_a, w) - wmae(y_true, yhat_b, w)
    print(f"corr_yhat {corr_yhat:.6f}")
    print(f"corr_resid {corr_resid:.6f}")
    print(f"delta_wmae {delta:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
