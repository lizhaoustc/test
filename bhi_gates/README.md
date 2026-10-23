# bhi_gates

Local promotion checks for stacking a candidate cuffless BP model on top of a
champion, using out-of-fold (OOF) predictions from BHI 2026 Track 1 research.

The package does not ship competition data, does not train models, and does
not call the Kaggle scorer. Inputs are tables you already computed.

## Install

From the repository root, with Python 3.10+:

```bash
python -m pip install numpy pandas scipy
```

Those are the only runtime dependencies. Run the tests from the repository
root so `bhi_gates` is importable:

```bash
python -m unittest bhi_gates.tests.test_gates_smoke
```

## OOF columns

Each model is one CSV, one row per sample, champion and candidate row-aligned:

| column | meaning |
| --- | --- |
| `subject_id` | subject label; repeated rows belong to one subject |
| `sbp_true` | reference systolic BP, mmHg |
| `dbp_true` | reference diastolic BP, mmHg |
| `sbp_pred` | model systolic prediction, mmHg |
| `dbp_pred` | model diastolic prediction, mmHg |

```python
import pandas as pd
from bhi_gates import hard_pass_oof

champ = pd.read_csv("oof_champion.csv")
candidate = pd.read_csv("oof_candidate.csv")
result = hard_pass_oof(champ, candidate)

if result["promote"]:
    print("candidate cleared every hard gate")
else:
    for reason in result["reasons"]:
        print(reason)
```

`hard_pass_oof` checks that `subject_id`, `sbp_true`, and `dbp_true` match
row for row, then calls `hard_pass`. Extra columns are ignored.

The same decision from arrays:

```python
import numpy as np
from bhi_gates import hard_pass

y = champ[["sbp_true", "dbp_true"]].to_numpy(dtype=float)
result = hard_pass(
    champ=champ[["sbp_pred", "dbp_pred"]].to_numpy(dtype=float),
    candidate=candidate[["sbp_pred", "dbp_pred"]].to_numpy(dtype=float),
    y=y,
    subject_ids=champ["subject_id"].to_numpy(),
    sbp=y[:, 0],
)
```

`result["promote"]` is a bool. `result["reasons"]` is a list of strings: empty
when the candidate is promoted, otherwise one entry per failed check.
`result["checks"]` holds the numbers behind each check.

## Hard gates

All of these must hold. A miss on any one of them blocks promotion.

| check | rule | why it is here |
| --- | --- | --- |
| `pred_corr` | Pearson r of champion vs candidate predictions `<= 0.98` | rejects a near-copy of the champion |
| `resid_corr` | Pearson r of residuals `<= 0.7` | rejects a model that makes the same errors |
| `delta_wmae` | candidate WMAE − champion WMAE `<= -0.01` mmHg | requires a real gain on the Track 1 loss |
| `sbp140` | SBP MAE on reference SBP ≥ 140 is not worse | the hypertensive subset must not pay for the average gain |
| `worst_quartile` | worst-quartile subject WMAE is not worse | the gain must not come from abandoning the worst subjects |

"Not worse" means the candidate summary is less than or equal to the champion
summary, with an absolute tolerance of `1e-8` mmHg. An undefined correlation
or an empty SBP ≥ 140 subset fails closed.

SBP ≥ 140 bias (`mean(pred - true)`, mmHg) is computed and returned in
`checks["sbp140"]` so under-reading is visible. It is not an additional
blocker. The published ranking loss is absolute error, and the gate applies
"not worse" to that subset MAE.

## Official-style WMAE

`weighted_mae` reconstructs the published Track 1 sample-level loss:

```text
MAE = (1 / N) * sum_i (0.6 * |SBP_i - SBP_hat_i| + 0.4 * |DBP_i - DBP_hat_i|)
```

Assumptions, also spelled out in `metrics.py`:

* Weights are the published 0.6 (SBP) and 0.4 (DBP). They sum to 1 and are not
  rescaled. The number is in mmHg. Lower is better.
* Each row is one sample and rows are weighted equally. Subject aggregation is
  a separate metric (`worst_quartile_subject_mae`).
* Column 0 is SBP and column 1 is DBP. Values must be finite. Nothing is
  imputed, calibrated, or clipped.
* The leaderboard transform `1 / (1 + MAE)` is not applied. `delta_wmae` is
  the raw mmHg difference.
* Public/private split handling and any organizer-side rounding are not
  reproduced. Do not quote this stub as an official leaderboard score.

```python
from bhi_gates import weighted_mae
import numpy as np

y = np.array([[120.0, 80.0], [140.0, 90.0]])
pred = np.array([[122.0, 81.0], [145.0, 88.0]])
weighted_mae(y, pred)  # 2.7 mmHg
```

Pearson correlations use `scipy.stats.pearsonr`. For two-column BP arrays, SBP
pairs and DBP pairs are concatenated into one correlation so each gate has a
single threshold. Residual correlation is that same correlation applied to
`y - prediction`.

`sbp140_mae_bias` restricts to rows with reference SBP ≥ 140 (the boundary
row is included). MAE is `mean(|pred - true|)`. Bias is `mean(pred - true)`.

`worst_quartile_subject_mae` computes the official-style WMAE inside each
subject, then returns the mean of the worst `ceil(n_subjects / 4)` subjects
(at least one). It is a tail summary of the same loss, not an SBP-only MAE
and not a paired test of which subject got worse.
