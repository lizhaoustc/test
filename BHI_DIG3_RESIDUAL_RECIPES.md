# BHI 2026 Track 1 — DIG3 residual recipes

Subject-independent cuffless BP. Person-group 5-fold, no per-subject calibration.
Current OOF champion is the RegResid v21 router (~6.567 WMAE). Physics and morph
packs stacked on the pre-v21 combine baseline stayed off this shortlist.

Every recipe trains an expert on the champion residual and is scored only on
outer-fold OOF. `upload=false` on every run. Stack: `HistGradientBoostingRegressor`
plus numpy/pandas. Set `early_stopping=False` so person groups stay intact.

Shared contract:

- Columns: `person_id`, `y_true`, `yhat_champ`, plus whatever feature columns the recipe names. The champion prediction is an offset, never a feature.
- Target `r = y_true - yhat_champ`. Emit `yhat = yhat_champ + δ`.
- `corr_yhat` = Pearson(`yhat`, `yhat_champ`) ≤ 0.98.
- `corr_resid` = Pearson(`y_true - yhat`, `y_true - yhat_champ`) ≤ 0.7.
- `ΔWMAE` = WMAE(`yhat`) − WMAE(`yhat_champ`) ≤ −0.01. Use the same weighted MAE that produced the 6.567 figure. If that helper is absent, weight every row by 1, or by a caller-supplied `w` column. Print the three numbers with `recipes/check_orthogonality.py` (`--yhat-a` candidate, `--yhat-b` champion).
- SBP≥140: MAE on rows with `y_true ≥ 140` stays ≤ the champion on those same rows. On a multi-target table, apply the mask only where the target is SBP.
- worstQ: lock the people first — the quartile of `person_id`s with the largest per-person mean `|y_true - yhat_champ|`. MAE of `yhat` on that frozen set stays ≤ the champion. Do not reselect the quartile from the new errors.
- Promote only when all five checks pass on the pooled OOF and on each fold. One retry lives inside that recipe’s Failure WRAP; then leave v21 unchanged.

When `δ` is uncorrelated with `yhat_champ` and the champion prediction SD is 15–20 mmHg, an OOF SD of `δ` under ~4 mmHg stays above the 0.98 ceiling. Keep the correction large enough to move errors and small enough to hold SBP≥140 and worstQ.

---

## 1. WGR-Resid — worst-group reweight on the champ residual

- **Features / loss.** Target `r` inside each outer train fold. Row weight `w = 1 + a·1[y_true≥140] + b·1[|r| ≥ Q75_train(|r|)]`, then cap `w` at the train-fold 95th percentile. Pre-register `(a,b,c) ∈ {(1,1,8),(2,1,8),(1,2,4)}`. Fit `HistGradientBoostingRegressor(loss="absolute_error", learning_rate=0.05, max_iter=400, max_depth=6, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=21)` on the numeric columns already used by v21. `δ = clip(r̂, ±c)` in mmHg.
- **Inputs assumed.** The v21 person-group OOF frame and its numeric feature matrix. Q75 and the weight cap use outer-train persons only. `upload=false`.
- **Hard gates.** `corr_yhat ≤ 0.98`, `corr_resid ≤ 0.7`, `ΔWMAE ≤ -0.01`, SBP≥140 MAE ≤ champ, worstQ MAE ≤ champ.
- **Promote.** Among triples that pass every gate on the pool and on all 5 folds, keep the lowest pooled WMAE and store `(a,b,c)` beside v21.
- **Failure WRAP.** If `corr_yhat > 0.98`, replace that triple’s `c` with `c/2` and rescore once. If SBP≥140 or worstQ is worse than v21, drop the triple. When none remain, leave v21 unchanged and log `rejected WGR-Resid`, the best `ΔWMAE`, and the failing gate.

## 2. Conf-Select — split-conformal selective override

- **Features / loss.** Inside each outer train fold, hold out 20% of persons as calibration (`random_state=21`). On the fit persons, train two HGBs with `loss="absolute_error"`, `learning_rate=0.05`, `max_iter=300`, `max_depth=4`, `min_samples_leaf=40`, `early_stopping=False`: one predicts `r`, one predicts `|r|` (difficulty `d̂`). Features are the v21 numeric columns. Mondrian bins use predicted champ SBP: `<120`, `120–139`, `≥140`. On calibration persons, `q_bin` is the 0.80 quantile of true `|r|` in that bin; a bin with fewer than 30 calibration rows uses the global 0.80 quantile. `δ = clip(r̂, ±8)` when `d̂ ≥ q_bin`, else `δ = 0`.
- **Inputs assumed.** Same OOF frame as v21, including `yhat_champ` so the bin exists at score time. Thresholds come only from calibration persons. `upload=false`. The quantile caps the override rate per bin; it is an override switch, not a coverage certificate for corrected BP.
- **Hard gates.** The five shared checks. Also log the OOF override fraction.
- **Promote.** Pre-registered quantile level is 0.80. Promote only if all five gates pass on the pool and on every fold and the override fraction is between 0.05 and 0.35.
- **Failure WRAP.** If the override fraction is above 0.35, or SBP≥140 or worstQ worsens, refit once with quantile level 0.90. If a gate still fails, emit `δ = 0`, leave v21 unchanged, and log `rejected Conf-Select` plus the override fraction.

## 3. Inv-Resid — demographics-free invariant residual

- **Features / loss.** From the v21 numeric columns, drop any name matching `age|sex|gender|bmi|height|weight|site|device|race|ethnic` (case-insensitive). On outer-train persons with at least two rows, ICC = `var_between / (var_between + var_within)` using person means and an unweighted mean of within-person variances (`ddof=1`). A feature with both variances 0 is dropped as constant. A feature with within variance 0 and between variance > 0 has ICC 1. Drop ICC ≥ 0.25. If that removes more than half the remaining columns, keep the lowest-ICC half instead. Need at least 8 columns. Row weight is `1/n_person` so each train person sums to the same mass. Fit HGB `loss="absolute_error"`, `learning_rate=0.05`, `max_iter=300`, `max_depth=4`, `min_samples_leaf=50`, `l2_regularization=1.0`, `early_stopping=False`, `random_state=21`. `δ = clip(r̂, ±8)`.
- **Inputs assumed.** v21 numeric matrix, `person_id`, `y_true`, `yhat_champ`. ICC and weights are fit on outer-train persons only. No within-person z-score. `upload=false`.
- **Hard gates.** The five shared checks. At least one column must have been removed by the denylist or the ICC cut.
- **Promote.** Pass all five gates on the pool and on every fold. Record the kept column list and the median dropped ICC with the OOF metrics.
- **Failure WRAP.** If fewer than 8 columns remain, stop and log `rejected Inv-Resid, too few invariant columns`. If `corr_yhat > 0.98`, refit once at `max_depth=2`. If a gate still fails, leave v21 unchanged. Dropped columns stay dropped.

## 4. Stiff-Resid — arterial stiffness proxy residual

- **Features / loss.** Use a column when it is present (case-insensitive): `ptt_ms`, `pat_ms`, `hr_bpm`, `crest_time_ms`, `pulse_width_50_ms`, `reflection_index`, `aug_idx`, `notch_delay_ms`. Add ratios only from columns that exist: `ptt_or_pat * hr_bpm`, `crest_time_ms / pulse_width_50_ms`. Require at least 3 raw proxy columns. Standardize with train-fold mean and std. Two low-capacity fits on `r`, inner person holdout 20% (`random_state=21`) picks which one is refit on all outer-train persons: (i) numpy `lstsq` of `r` on `[1 | proxies]`; (ii) HGB `loss="absolute_error"`, `max_iter=200`, `max_depth=2`, `min_samples_leaf=80`, `early_stopping=False`, `random_state=21`. `δ = clip(r̂, ±10)`.
- **Inputs assumed.** v21 OOF labels and champion column, plus whichever proxy columns already sit in the feature table. Missing proxies are skipped. `upload=false`.
- **Hard gates.** The five shared checks, applied to the inner-split winner only.
- **Promote.** Winner is the model with lower inner-holdout WMAE. Promote it when all five gates pass on pooled OOF and on every fold. If both pass and their pooled WMAE differs by less than 0.005, promote the linear fit. Record coefficients or the depth-2 tree dump.
- **Failure WRAP.** If fewer than 3 raw proxy columns exist, log `rejected Stiff-Resid, proxies absent` and stop. If the winner misses a gate, score the other model once. If that also misses, leave v21 unchanged. Do not fill the gap with the physics or morph packs.

## 5. AntiCorr-Pool — anti-correlated probe-pool fusion

- **Features / loss.** Eight residual probes, each a person-group inner 4-fold OOF on outer-train persons, features from the v21 numeric columns, `early_stopping=False`, `random_state` as below. Probes 1–4: `loss="absolute_error"`, `max_depth=3`, `max_iter=200`, `min_samples_leaf=40`, a fresh 40% column subset, seeds 0–3. Probe 5: all columns, `max_depth=2`, absolute error, seed 5. Probe 6: all columns, `max_depth=5`, absolute error, seed 6. Probe 7: all columns, `loss="squared_error"`, `max_depth=3`, seed 7. Probe 8: the half of columns with the smallest `|Pearson(col, r)|` on the outer train fold, absolute error, `max_depth=3`, seed 8. Learning rate 0.05 for all.
- **Fusion.** Drop probes with inner-OOF Pearson(`r̂_k`, `r`) ≤ 0. Rank the rest by Pearson(`r̂_k`, `yhat_champ`) ascending, so the probe most anti-correlated with the champion is tried first. Start at `δ = 0` and accept at most three. After the first acceptance, a probe is eligible only when Pearson(`r̂_k`, current `δ`) ≤ 0. For `α ∈ {0.25, 0.5, 1}` keep the step `δ ← δ + α·r̂_k` with the best inner WMAE among steps that improve inner WMAE by ≥ 0.01 versus the current `δ` and keep inner SBP≥140 and worstQ MAE ≤ v21. Refit each accepted probe on all outer-train persons, rebuild `δ` with the stored alphas, and score the outer test fold. `yhat = yhat_champ + δ`.
- **Inputs assumed.** v21 OOF frame and numeric matrix. Inner OOF uses outer-train persons only. `upload=false`.
- **Hard gates.** The five shared checks on the finished fused outer OOF.
- **Promote.** Promote when at least one probe is accepted and all five gates pass on the pool and on every fold. Log accepted probe ids and alphas.
- **Failure WRAP.** If no probe is accepted, or the finished fusion misses a gate, log `rejected AntiCorr-Pool` and the lowest inner Pearson(`r̂_k`, `yhat_champ`) among probes with positive residual correlation. Leave v21 unchanged. Keep the eight probes unaveraged. A second pass is not allowed.

---

Checker (no dataset is bundled):

```bash
python recipes/check_orthogonality.py PATH_TO_YOUR_OOF.csv \
  --yhat-a yhat_new --yhat-b yhat_champ --y y_true
```
