# BHI paper scout: champion residual + hard orthogonality gates

Shortlist of eight papers from 2024–2026 on cuffless or wearable blood pressure (BP). Each one is chosen because a piece of it transfers to this training recipe:

**Champion residual.** Freeze a population model \(f_c\) (the champion) trained only on other people. Train a second head \(r\) to predict the residual \(y - f_c(x)\), then emit \(\hat{y} = f_c(x) + r(x)\). The residual head is not a per-person cuff offset.

**Hard orthogonality gates.** Before \(r\) sees a representation, drop every direction whose cosine with the champion embedding, or with a subject-identity subspace fit on training people only, exceeds a fixed threshold. The gate is a projection or a binary mask. A soft penalty is not a substitute. A second gate abstains when a conformal interval on the residual is wider than a preset width.

**Track1 constraints used in every sketch below.**

- Inputs are 16 PPG channels plus ECG. No age, sex, BMI, or other subject covariates, and no cuff value at test time.
- Split is person 5-fold: every window from one person stays in one fold. Train on four folds, test on the held-out people. No fine-tune on the test person.
- Inside the four training folds, keep a person-disjoint slice for residual fitting and another for conformal calibration. The test fold is never used to fit \(f_c\), \(r\), the identity subspace, or the conformal quantile.

Checked on 23 October 2026. A DOI is listed only when it appeared on a publisher page, in the paper PDF, or in a publisher-indexed record opened for this note. Anything not opened that way is marked **need-verify**. Code links are included only when the page was fetched or the paper itself prints the URL. A search-index hit that returned HTTP 404 is not treated as a working repository.

## 1. PPG-IDR: identity-orthogonal medical features

- **Title:** PPG-IDR: Leveraging User Identity for Robust Cross-user PPG Sensing via Disentangled Representations
- **Authors:** Hung Manh Pham, Xiao Ma, Changshuo Hu, Xiaoyu Xu, Thivya Kandappu, Yuezhong Wu, Tarek Abdelzaher, Archan Misra, Dong Ma
- **Year:** 2026 (Proc. ACM IMWUT 10(3), Article 147, September 2026)
- **DOI:** [10.1145/3832019](https://doi.org/10.1145/3832019) (printed in the ACM PDF)
- **Code:** The PDF says the code will be released upon acceptance. No public repository was confirmed in the paper text. **need-verify.** The backbone they freeze is PaPaGei ([Nokia-Bell-Labs/papagei-foundation-model](https://github.com/Nokia-Bell-Labs/papagei-foundation-model)); that repository is the foundation model, not PPG-IDR.
- **One-line method:** Dual encoders on PPG; a cosine push loss \(\mathcal{L}_{\text{PUSH}} = \frac{1}{N}\sum_i \langle z^{\text{med}}_i, z^{\text{id}}_i \rangle\) plus an identity adversary, with inference using only the medical branch.
- **Why it maps:** This is the closest published orthogonality gate. The identity encoder is the nuisance champion; the medical encoder is forced off that subspace; BP is one of the downstream targets (unseen-user MIMIC-BP MAE 13.69 mmHg SBP and 8.32 mmHg DBP in their Table 3, versus 16.48 / 11.17 for the single Papagei encoder). Their push loss is soft. Track1 hardens it: zero the medical coordinates whose cosine with the identity subspace exceeds a fixed threshold, and train \(r\) only on the champion’s residual in the surviving coordinates.

**Track1 adaptation**

- Freeze a 16-channel PPG + ECG champion \(f_c\) trained with person 5-fold ERM. Its penultimate vector is \(z_c\).
- On training people only, fit an identity subspace (linear identity probe, or the top principal components of per-person mean embeddings). Do not fit it on the test fold.
- Gate: \(z_\perp = z - \sum_k \mathbf{1}[|\cos(z, u_k)| > \tau] \langle z, u_k \rangle u_k\), with \(\tau\) fixed before the test fold (start at 0.2). \(u_k\) spans both \(z_c\) and the identity subspace.
- Train \(r(z_\perp)\) with loss \(|r(z_\perp) - (y - f_c(x))|\) for SBP and DBP. At test time add \(f_c(x) + r(z_\perp)\). No identity head is run on test people.
- Report per-fold MAE and the fraction of coordinates the gate zeros. If the gate zeros almost everything, \(\tau\) was too low; retune \(\tau\) on the training-fold validation people only.

## 2. Worst-group BP via CD-REx

- **Title:** Robust optimization for PPG-based blood pressure estimation
- **Authors:** Sungjun Lim, Taero Kim, Hyeonjeong Lee, Yewon Kim, Minhoi Park, Kwang-Yong Kim, Minseong Kim, Kyu Hyung Kim, Jiyoung Jung, Kyungwoo Song
- **Year:** 2025 (Biomedical Signal Processing and Control, vol. 105, July 2025, article 107585; ScienceDirect record dated 2 February 2025)
- **DOI:** [10.1016/j.bspc.2025.107585](https://doi.org/10.1016/j.bspc.2025.107585)
- **Code:** Search indexes still describe `https://github.com/sungjun98/ppg_bp_estimation`, and that URL returned HTTP 404 on 23 October 2026. The first author’s publication page lists the paper and does not link code. **need-verify.**
- **One-line method:** PPG-only ConvTransformer on MIMIC-III, with in-group Time-CutMix and CD-REx (count-weighted risk-variance plus a Tukey-transformed distribution penalty) so hypotension and hypertensive-crisis groups are not ignored by average risk.
- **Why it maps:** The ERM model is the champion: it spends its capacity on the crowded normal and prehypertension bands. The residual that matters is the error inside hypo and crisis. CD-REx is a soft variance penalty across BP groups. Track1 keeps the champion for the majority band and trains \(r\) on group-wise residuals, with a hard gate that refuses residual features aligned with the majority-band champion direction.

**Track1 adaptation**

- Define groups on reference SBP only, using the paper’s bands (hypo, normal, prehypertension, hypertension, crisis). Assign groups from the label, never from the prediction, and only for training losses and reporting.
- Fit \(f_c\) by plain ERM on the four training folds. Freeze it.
- Residual target is \(y - f_c(x)\). Up-weight hypo and crisis inside \(r\)’s loss by the inverse training-fold count. Time-CutMix only mixes two training-fold windows from the same BP group.
- Hard gate: estimate the majority-band (normal + prehypertension) mean of \(z_c\) on training people, and zero residual-head coordinates aligned with that mean above \(\tau\).
- On each test fold, report mean MAE and the worst of the five group MAEs for SBP and DBP. A gain in mean MAE with a worse crisis MAE is a failed run.

## 3. Conformal intervals as the abstain gate

- **Title:** Conformal prediction quantifies wearable cuffless blood pressure with certainty
- **Authors:** Zhan Shen, Tapabrata Chakraborti, Christopher R. S. Banerji, Xiaorong Ding
- **Year:** 2025 (Scientific Reports, published 23 July 2025; CityUHK record lists article 26697)
- **DOI:** [10.1038/s41598-025-09580-0](https://doi.org/10.1038/s41598-025-09580-0)
- **Code:** No method repository found. The paper uses the public Aurora-BP features ([microsoft/aurorabp-sample-data](https://github.com/microsoft/aurorabp-sample-data)).
- **One-line method:** Quantile-loss gradient boosting on wearable ECG and PPG features from 483 Aurora-BP subjects, then split-conformal calibration so the interval covers reference BP at a set rate (about 14.3 / 9.8 mmHg MAD for 24 h SBP / DBP; daytime coverage 91.67% and nighttime 95.89% at \(\epsilon = 0.05\) in the companion conference abstract, with the journal giving the same protocol).
- **Why it maps:** Conformal width is the hard selective gate on top of a point predictor. Their own limitation section says the interval has one global width because calibration was pooled over subjects. Track1 puts that gate on the residual \(y - f_c(x)\), calibrates it on people the residual head did not train on, and abstains when the residual interval exceeds a width chosen on training folds.

**Track1 adaptation**

- Champion \(f_c\): 16-channel PPG + ECG network, person 5-fold, no cuff calibration.
- From the four training folds, split people 60/40 into residual-train and conformal-calibration, matching their subject split. Refit this inner split inside every outer fold.
- Fit \(r\) on residual-train people. On calibration people, store nonconformity scores \(|y - f_c(x) - r(x)|\) separately for SBP and DBP, and separately for day vs night if a time flag exists without identifying the person.
- Test interval is \([f_c + r - q,\; f_c + r + q]\) with \(q\) the conformal quantile of the calibration scores. Emit the point only when \(2q\) is below a width fixed on training folds (a practical start is 10 mmHg for DBP and 16 mmHg for SBP).
- Report coverage and MAE on emitted windows, plus the abstention rate, per test fold. Do not tune \(q\) on the test fold. Their paper notes that a single global \(q\) is not conditional on the person; keep \(q\) marginal over calibration people and say so.

## 4. Evidential residual plus conformal width

- **Title:** Uncertainty Quantification of Cuffless Blood Pressure with Deep Evidential Regression Model
- **Authors:** Zhan Shen, Shuaiting Yao, Yifan Chen, Xiaorong Ding
- **Year:** 2024 (EMBC, Orlando, 15–19 July 2024; IEEE Xplore document 10782943)
- **DOI:** [10.1109/EMBC53108.2024.10782943](https://doi.org/10.1109/EMBC53108.2024.10782943) (indexed by IEEE Xplore and Europe PMC; the PDF was not opened, so page-level quotes are **need-verify**)
- **Code:** None found.
- **One-line method:** Deep evidential regression on five PPG features predicts a Normal-Inverse-Gamma distribution over BP; conformal prediction then wraps the epistemic uncertainty into an interval (reported MAD 5.56 / 3.18 mmHg SBP / DBP and coverage 94.8% / 95.9% on 37 Finapres subjects).
- **Why it maps:** Evidential regression is a residual distribution around a mean, which is the right target for \(r\), and the conformal step is the hard gate. The cohort is small and the features are five scalars, so the coverage numbers will not carry over to 16-channel person 5-fold. Steal the head, not the headline error.

**Track1 adaptation**

- Keep \(f_c\) as a point model. Let \(r\) emit evidential parameters \((\gamma, \nu, \alpha, \beta)\) for the residual of SBP and of DBP, with \(\gamma\) the residual mean.
- Loss is the evidential negative log-likelihood on \(y - f_c(x)\), plus the usual evidence regularizer so the head cannot shrink uncertainty by memorizing training people.
- Hard feature gate from paper 1 runs before this head, so evidential evidence cannot sit on identity directions.
- Conformal step from paper 3 calibrates \(|y - f_c - \gamma|\) on calibration people. Abstain when the conformal width is large or when epistemic uncertainty \(\beta / (\nu(\alpha-1))\) exceeds a training-fold threshold.
- Because \(n = 37\) in the source paper, do not copy their coverage as a target. The acceptance test is coverage at the nominal rate on held-out people, with MAE reported only on accepted windows.

## 5. Subject-adversarial multi-wavelength PPG

- **Title:** Generalizable Blood Pressure Estimation from Multi-Wavelength PPG Using Curriculum-Adversarial Learning
- **Authors:** Zequan Liang, Ruoyu Zhang, Wei Shao, Mahdi Pirayesh Shirazi Nejad, Ehsan Kourkchi, Setareh Rafatirad, Houman Homayoun
- **Year:** 2025 (arXiv:2509.12518, read as HTML; IEEE BSN version indexed 15 September 2025)
- **DOI:** [10.1109/BSN66969.2025.11337701](https://doi.org/10.1109/BSN66969.2025.11337701) (**need-verify**; IEEE PDF not opened). Open version: [arXiv:2509.12518](https://arxiv.org/abs/2509.12518)
- **Code:** None found in the arXiv HTML.
- **One-line method:** Per-wavelength CNNs on four fingertip PPG channels (660, 730, 850, 940 nm; Cui et al. public set, 180 subjects), a gradient-reversal subject discriminator, and a curriculum from hypertension classification to SBP/DBP regression, under a subject-disjoint 4:1 split (MAE 14.2 mmHg SBP, 6.4 mmHg DBP).
- **Why it maps:** Subject identity is treated as the domain to remove, which is the nuisance half of the orthogonality gate. Gradient reversal is a soft fight with a discriminator that still needs identity labels. Track1 replaces the reversal layer with the hard projection, and uses the curriculum as the champion: stage-1 classifies hypertension, stage-2’s residual corrects the mmHg error. Their 4:1 subject split is one outer fold of person 5-fold, not five folds; rerun it as five.

**Track1 adaptation**

- One temporal tower per PPG channel (16 towers) plus one ECG tower. Each tower consumes the waveform, its derivative, and its envelope, following their six-stream idea, then a fusion layer.
- Stage 1 (champion pretrain): hypertension vs not, labels from training-fold reference BP only (their cut was SBP \(\ge 130\) or DBP \(\ge 90\)). Freeze the towers.
- Stage 2: hard-orthogonalize the fused vector against a training-person identity subspace (paper 1). Train \(r\) on \(y - f_c(x)\). Do not attach a subject discriminator at test time, and do not train one on test IDs.
- Inner 5-fold on training people for \(\tau\) and the curriculum length; the outer test fold stays untouched. Their paper already uses 5-fold inside the training pool.
- Ablate three test-fold numbers: all 16 PPG channels, PPG with the ECG tower, and PPG with the ECG tower after the gate. The ECG tower earns its keep only if gated fusion beats PPG-only on held-out people.

## 6. Causal-invariance feature gate

- **Title:** Multi-Source Causal Invariance for Cuffless Blood Pressure Estimation Based on Photoplethysmography Signal Features
- **Authors:** Yiliu Xu, Zhaoming He, Hao Wang
- **Year:** 2025 (Sensors 25(11), 3254, published 22 May 2025)
- **DOI:** [10.3390/s25113254](https://doi.org/10.3390/s25113254)
- **Code:** No author repository found. Do not treat general Markov-blanket libraries as this paper’s code.
- **One-line method:** MDSFS-EMB runs PPFS and HITON-MB with Gaussian-copula mutual information on 218 PPG features from three heterogeneous sources, keeps the stable Markov blanket (about 20–21 features), and tests it on an external PPG-BP set with record-level (person-disjoint) splits.
- **Why it maps:** A Markov blanket that survives several sources is a hard keep-or-drop gate. Features that move with device or cohort are the directions a champion overfits. Track1 runs the same intersection on training folds only, then lets the residual head see only the surviving coordinates, which are the ones orthogonal to source-specific confounders by construction of the blanket.

**Track1 adaptation**

- Build a fixed dictionary of morphology features per PPG channel and ECG timing features (intervals inside the window, not a cuff-derived PAT offset). Cap the dictionary; do not add demographics.
- On each training fold, treat the other three training folds as separate sources. Run a Markov-blanket selector per source and keep the intersection. Freeze that binary mask.
- Champion \(f_c\) may use the full waveform. Residual head \(r\) receives only masked features, and is trained on \(y - f_c(x)\).
- Person 5-fold already matches their record-level split. Refit the mask inside each outer training pool. Never refit it after seeing the test fold.
- External check, if a second dataset exists: apply the frozen mask with no reselection. Report the overlap of the five fold-wise masks; a feature that survives only one fold is not invariant.

## 7. Calibration-free champion, then a residual on the shift

- **Title:** Generalizable deep learning for photoplethysmography-based blood pressure estimation — A benchmarking study
- **Authors:** Mohammad Moulaeifard, Peter H. Charlton, Nils Strodthoff
- **Year:** 2025 (Machine Learning: Health 1 010501, published 15 September 2025)
- **DOI:** [10.1088/3049-477X/ae01a8](https://doi.org/10.1088/3049-477X/ae01a8)
- **Code:** [AI4HealthUOL/ppg-ood-generalization](https://github.com/AI4HealthUOL/ppg-ood-generalization), also archived by the authors as [10.5281/zenodo.15797452](https://doi.org/10.5281/zenodo.15797452) (Zenodo DOI printed in the paper’s reference 55). Preprint: [arXiv:2502.19167](https://arxiv.org/abs/2502.19167)
- **One-line method:** Five waveform networks on PulseDB, scored both with subject overlap (Calib) and without (CalibFree, AAMI). Best calibration-free in-distribution MAE is 13.9 / 8.5 mmHg SBP / DBP (XResNet1d101); external calibration-free MAE spans 10.0–18.6 / 5.9–10.3. Importance weights from the BP-label histogram are the only adaptation they apply.
- **Why it maps:** CalibFree XResNet1d101 is a concrete champion: it is the strong population model once subject overlap is forbidden. The Calib-minus-CalibFree gap is exactly the residual a calibration-free system still owes. Their importance weights rebalance the label histogram; Track1 instead trains \(r\) on that gap and gates \(r\) so it cannot reload the subject directions that make Calib look good.

**Track1 adaptation**

- Implement the split the way this repository does: no person in both train and test. Use five such folds rather than one CalibFree cut. Ignore their Calib protocol for the headline number.
- Champion candidates: their XResNet1d101 and a smaller net of the same family, on 16 PPG channels stacked with ECG as extra input channels. Pick \(f_c\) by mean test-fold MAE on training-pool validation people.
- Freeze \(f_c\). Train \(r\) on the residual. Apply the hard orthogonality gate against \(z_c\) so \(r\) has to use directions the champion did not.
- Their importance weighting is a useful extra only as a training-fold label reweight for hypo and crisis (paper 2). Do not estimate weights from the test label histogram; that leaks the test BP distribution, which their discussion already flags as a limit of weight-from-target-histogram adaptation.
- Table for each fold: CalibFree MAE of \(f_c\), MAE of \(f_c+r\), worst-group MAE, and abstention rate from paper 3. Copy their external-set evaluation if those sets are available, with the mask and the gate frozen.

## 8. Calibration-free dynamic tracking with channel gates

- **Title:** UTransBPNet for cuffless and calibration-free blood pressure estimation under dynamic conditions
- **Authors:** Yali Zheng, Hongda Huang, Jiasheng Gao, Jingyuan Hong, Shenghao Wu, Yuanting Zhang, Qing Liu
- **Year:** 2025 (Scientific Reports, published 21 May 2025)
- **DOI:** [10.1038/s41598-025-02963-3](https://doi.org/10.1038/s41598-025-02963-3)
- **Code:** None found.
- **One-line method:** Squeeze-and-excitation U-Net for short-range features plus a transformer and cross-attention for long-range context, on ECG, PPG, and PPG derivatives, trained without a personal cuff offset. Leave-one-subject-out on drink (\(n=25\)) and exercise sets; instance 10-fold, not person-out, on their MIMIC subset. They also train across scenarios and show the drop from distribution shift and from the tails of BP.
- **Why it maps:** The population model with no personal cuff is the champion. SE blocks are channel gates, and the U-Net residual path is the natural place to add \(r\). Their cross-scenario drop is the residual worth fitting. The MIMIC split in the paper mixes windows from the same person, so Track1 must not copy it. SE gates are soft sigmoid masks; Track1 thresholds them.

**Track1 adaptation**

- Input tensor: 16 PPG channels, the first difference of each, and one ECG channel. No personal baseline BP. Their population-mean baseline is a reporting baseline, not an input feature.
- Champion: SE-U-Net + transformer trained on four person folds. Freeze it.
- Residual head: a second SE block on the U-Net skip features, trained to predict \(y - f_c(x)\). Binarize each SE weight at 0.5 and, on top of that, apply the orthogonality gate against \(z_c\) and the training-person identity subspace.
- Outer loop is person 5-fold for every dataset, including any ICU subset. Leave-one-subject-out on a 25-person set is a diagnostic, not the replacement for the five-fold table.
- Cross-condition check inside the training people (rest vs exercise or day vs night, if those tags exist): fit \(r\) on one condition’s residual and test on the other condition’s held-out people. Report that shift MAE next to the pooled person-fold MAE.

## Read order for Track1

1. Paper 7 for the champion and the person-disjoint split, because the code is public and the CalibFree protocol matches the constraint.
2. Paper 1 for the hard orthogonality gate. Use the formula above; their released code was not confirmed.
3. Paper 2 for worst-group metrics and in-group mixing. Reimplement CD-REx from the ScienceDirect description if the repository is still dark.
4. Papers 3 and 4 for the abstain gate. Paper 3 is the wearable ECG+PPG protocol; paper 4 is the evidential residual head.
5. Papers 5, 6, and 8 are architecture donors: per-channel towers, a frozen feature mask, and a U-Net residual path. None of the three had a confirmed method repository.

## Near misses

These are real 2024–2025 cuffless or wearable BP papers that did not make the eight because the transfer to residual-plus-hard-gate is thin.

- Seyedmohsen Dehghanojamahalleh et al., “Subject-Independent Cuff-Less Blood Pressure Monitoring via Multivariate Analysis of Finger/Toe Photoplethysmography and Electrocardiogram Data,” BioMedInformatics 2025, 5(2), 24. DOI [10.3390/biomedinformatics5020024](https://doi.org/10.3390/biomedinformatics5020024). Subject-independent and no initial cuff, finger plus toe PPG with ECG, multivariate regression on 80 readings. No residual head and no orthogonality constraint. No code found.
- Suril Mehta, Nipun Kwatra, Mohit Jain, Daniel McDuff, “Examining the challenges of blood pressure estimation via photoplethysmogram,” Scientific Reports, 7 August 2024, [nature.com/articles/s41598-024-68862-1](https://www.nature.com/articles/s41598-024-68862-1). DOI **need-verify** (inferred from the article URL, not read off a DOI line). Useful leakage audit; code at [lirus7/PPG-BP-Analysis](https://github.com/lirus7/PPG-BP-Analysis) was named in the abstract. It does not propose a residual or an orthogonality gate.

## Verification log

| Item | Status |
| --- | --- |
| PPG-IDR DOI 10.1145/3832019 | Printed in the ACM PDF |
| PPG-IDR code | Not confirmed. **need-verify** |
| Lim et al. DOI 10.1016/j.bspc.2025.107585, vol. 105, article 107585 | ScienceDirect record |
| Lim et al. GitHub | Indexed URL 404 on 23 Oct 2026. **need-verify** |
| Shen et al. Sci. Rep. DOI 10.1038/s41598-025-09580-0 | Nature article page |
| Shen et al. Sci. Rep. method code | Not found |
| Shen et al. EMBC DOI 10.1109/EMBC53108.2024.10782943 | IEEE Xplore document 10782943 and Europe PMC. PDF not opened |
| Liang et al. arXiv:2509.12518 | HTML read |
| Liang et al. IEEE DOI 10.1109/BSN66969.2025.11337701 | Indexed only. **need-verify** |
| Xu et al. DOI 10.3390/s25113254 | Printed in the Sensors PDF / PMC |
| Moulaeifard et al. DOI 10.1088/3049-477X/ae01a8 | IOP page and Cambridge author PDF |
| Moulaeifard et al. code and Zenodo 10.5281/zenodo.15797452 | GitHub README fetched; Zenodo DOI printed in the paper |
| Zheng et al. DOI 10.1038/s41598-025-02963-3 | Nature article page and PMC |
| Zheng et al. code | Not found |
