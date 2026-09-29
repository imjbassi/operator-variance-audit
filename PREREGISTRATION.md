# Preregistration: Operator-Composition Variance in robomimic Multi-Human Benchmarks

**Date registered:** 2026-09-29 (written before any experimental training run)
**Author:** Jaiveer Bassi
**Status:** frozen. Any later change is recorded as a dated amendment in the
"Amendments" section at the bottom and never edited in place.

## 1. Claims under test

**Primary.** On robomimic Multi-Human (MH) datasets, the variance in
final-checkpoint success rate attributable to *operator composition* of the
training set exceeds the variance attributable to *training seed*. If so, the
conventional 3-seed uncertainty understates true uncertainty, and published MH
comparisons separated by a few points are not resolvable from the reported
statistic.

**Secondary.** The proficiency tier label (Better / Okay / Worse) does not
capture the operator effect: within-tier operator variation is comparable to
between-tier variation.

## 2. Predictions (stated before data)

P1. On Square-MH, the operator-partition SD (Arm B) exceeds the seed SD
    (Arm A), with a ratio interval excluding 1.
P2. On Square-MH, the operator-partition variance component from Arm B
    exceeds the matched random-partition variance component from Arm C
    (excess positive, interval excluding 0).
P3. Lift-MH is near ceiling for BC-RNN and will show little variance of any
    kind; Can-MH is expected to fall between Lift and Square. These are
    expectations, not tested hypotheses; the tests are run on all three tasks
    and corrected jointly (Section 6).
P4. On the secondary claim, the between-tier share of partition variance will
    not be distinguishable from what random tier labellings produce
    (permutation p > 0.05 on at least Square).

## 3. Falsification

- If, on a task, the Arm B and Arm C partition variance components are
  indistinguishable (excess interval includes 0), the operator effect is not
  separable from demonstration resampling at this sample size. The paper for
  that task becomes a negative result with an explicit power statement
  (Section 6.6). This outcome is acceptable and will be reported as such,
  not reframed.
- If the operator-partition SD does not exceed the seed SD with an interval
  excluding 1, the primary claim fails for that task and is reported as failed.
- A partition or seed variance estimate of exactly zero is a *boundary*
  estimate, not evidence of zero variance. It will be labelled as such in
  every table and in the text.

## 4. Data

- robomimic v1.5 Multi-Human low-dim datasets: `lift/mh`, `can/mh`,
  `square/mh` (300 demos each, 6 operators x 50 demos).
- Operator identity is read directly from the stock filter keys
  `{better,okay,worse}_operator_{1,2}` (verified in `STEP0_operator_identity.md`;
  each key = 50 contiguous demos, disjoint, union = all 300).
- Pristine downloads are stored read-only with SHA-256 sums. Training reads a
  working copy that adds only the partition filter keys defined in
  `partitions/*.json`; a verification script checks the working copy's
  partition masks equal the JSON before every run.
- The operator index is within-tier. Whether `better_operator_1` on one task is
  the same person as on another task is not documented; no cross-task
  same-operator claim will be made.

## 5. Experimental design

**Policy.** BC-RNN, low-dim observations, robomimic v0.5.0 paper
hyperparameters for low-dim BC-RNN: GRU/LSTM hidden 400, 2 layers, no MLP head,
GMM head (5 modes), sequence length 10, Adam lr 1e-4, batch 100,
2000 epochs x 100 gradient steps. Observation keys: `object`,
`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`. No image
observations. No diffusion policy. No other architectures.

**Checkpoint rule.** The FINAL checkpoint (epoch 2000) is evaluated. No
intermediate rollouts are run during training and no best-of-N selection is
performed. Validation is disabled; all demos in a partition are used for
training.

**Evaluation.** 50 rollouts per checkpoint, horizon 500, terminate on success.
The 50 initial states per task are generated once from seeded environment
resets, stored in `initial_states/<task>.npz`, committed, and reused
identically for every checkpoint of that task. The rollout RNG seed is fixed
per rollout index (same across conditions). Success = `env.is_success()["task"]`
reached within the horizon. Per-rollout outcomes are written to CSV.

**Arms.**

| Arm | Question | Partitions | Demos / partition | Seeds | Runs / task | Runs total |
|-----|----------|-----------:|------------------:|------:|------------:|-----------:|
| A | seed variance at fixed data | 1 (all 300) | 300 | 10 (seeds 1-10) | 10 | 30 |
| B | operator-composition variance | 6 leave-one-operator-out | 250 | 3 (seeds 1-3) | 18 | 54 |
| C | matched random control | 6 random draws | 250 | 3 (seeds 1-3) | 18 | 54 |

- Arm B partitions: drop one operator's 50 demos; keep the other 250. Every
  Arm B partition therefore has identical size (250) with no subsampling
  randomness; partitions differ only by which operator is absent.
- Arm C partitions: 250 demos drawn uniformly without replacement from all 300,
  ignoring operator identity, six independent draws from
  `numpy.random.default_rng(20260929)`. Sizes are identical to Arm B by
  construction. Arm C isolates the effect of *which specific demonstrations*
  are present from *which operators* are present.
- Partition assignments are generated once, written to `partitions/*.json`,
  and committed before any run. They are immutable thereafter.
- Total: 138 training runs, 6,900 evaluation rollouts.

**Tasks fixed.** Lift, Can, Square only. No tasks, architectures, or arms are
added without a dated amendment.

## 6. Statistical analysis plan

Unit of replication: the operator (equivalently, the leave-one-operator-out
partition). Never the rollout, never the demonstration.

6.1 **Per-run statistic.** Success rate over the 50 fixed-state rollouts of
    the final checkpoint.

6.2 **Seed variance (Arm A).** Per task, sample SD of success rate across the
    10 seeds, with a seed-bootstrap percentile interval (2,000 resamples).

6.3 **Two-way random-effects decomposition (Arms B and C separately).** Model
    the 6 x 3 partition-by-seed table as
    y_ps = mu + a_p + b_s + e_ps, a_p ~ (0, sigma2_p), b_s ~ (0, sigma2_s),
    e_ps ~ (0, sigma2_e). Two estimators, both reported:
    (i) method of moments (ANOVA): sigma2_p = (MS_p - MS_e)/3,
        sigma2_s = (MS_s - MS_e)/6, truncated at 0 with a boundary flag;
    (ii) REML via a crossed-random-effects mixed model
        (statsmodels MixedLM with variance components for partition and seed).
    Where the two estimators disagree materially (either differs from the
    other by more than a factor of 2 in SD, or one is at the boundary and the
    other is not), this is stated explicitly in the results.

6.4 **Headline statistic.** Per task, the ratio
    R = SD_partition(Arm B, MoM) / SD_seed(Arm A). Interval by joint
    bootstrap: resample the 6 operators (partitions) with replacement and the
    10 Arm A seeds with replacement, 2,000 resamples, percentile 95% interval.
    The claim holds for a task if R > 1 and the interval excludes 1. The same
    ratio computed with the REML SD is reported alongside.
    A within-Arm-B ratio SD_partition / SD_seed (both from the Arm B table) is
    reported as a secondary view.

6.5 **Control contrast (B vs C).** Per task, D = sigma2_p(B) - sigma2_p(C),
    with an operator/draw-bootstrap percentile interval (resample the 6
    partitions within each arm, 2,000 resamples). The operator effect is
    declared separable only if D > 0 and the interval excludes 0.

6.6 **Power statement (mandatory if any test is negative).** Simulation under
    the fitted seed and residual variances of the 6 x 3 design: the minimum
    partition SD detectable with 80% power at the corrected alpha. Reported
    per task regardless of outcome.

6.7 **Secondary claim (tier).** Using Arm B per-partition means (averaged over
    seeds), decompose the between-partition sum of squares into between-tier
    and within-tier parts. Test whether the between-tier share exceeds what is
    obtained under random assignment of the tier labels to the six
    partitions: exact permutation test over all 90 distinct labellings.
    Report the within-tier mean absolute pairwise difference alongside the
    between-tier one.

6.8 **Multiple comparisons.** Primary family: 6 contrasts (3 tasks x {R > 1,
    D > 0}). Holm-Bonferroni within the family. Secondary family: 3 tier
    permutation tests, Holm-Bonferroni within that family. Raw and corrected
    p-values (or interval-based decisions with equivalent alpha adjustment)
    are both reported.

6.9 **Descriptive tables that will always be reported.** The full 6 x 3
    partition-by-seed tables for Arms B and C, the 10-seed Arm A vector, and
    per-partition trajectory-length statistics, for every task.

## 7. Known limitations (stated up front)

- Six operators per task. The partition variance component rests on six
  groups; intervals will be wide and boundary estimates are plausible.
- One architecture (BC-RNN), one observation modality (low-dim), one
  training-length setting.
- Simulation only, one benchmark family (robomimic / robosuite).
- Operator labels are the robomimic authors' tier-within-index labels; no
  operator-level covariates exist.
- Success rate over 50 rollouts is itself a binomial estimate; its sampling
  noise appears in the residual term and is not separately modelled.

## 8. Compute and reproducibility

- Single RTX 4070 (12 GB), WSL2 Ubuntu. Runs execute under nohup via a
  resumable lock-based queue (`scripts/run_queue.py`); a run is complete only
  when its `metrics.json` exists.
- Pinned environment: `env/setup_wsl_env.sh`, resolved lock in
  `env/requirements-lock.txt`, robomimic at tag v0.5.0, robosuite 1.5.1,
  mujoco 3.2.6, torch 2.8.0+cu128, numpy 1.26.4.
- Every number in the paper is regenerated from `results/*.csv` by
  `analysis/*.py`. Per-run metrics and per-rollout outcomes are committed.
- A short smoke test (a few epochs, not evaluated for the paper) is run before
  Arm A solely to time a run and validate the pipeline end to end. If the
  measured budget makes 2000 epochs infeasible for 138 runs, the change will
  be recorded as a dated amendment here *before* Arm A starts.

## 9. Reporting gates

Results are reported at fixed gates: after Arm A; after Arm B but before
Arm C is interpreted; after Arm C. No prose about findings is written before
all three arms and the analysis scripts are complete.

## Amendments

(none)
