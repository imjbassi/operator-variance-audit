# Gate notes after Arm B, written before any Arm C result existed

Date: 2026-09-30 (Arm C training had started; 0 of 54 Arm C runs complete when written)

These notes record what was observed at the Arm B gate and which analyses are
preregistered versus exploratory, so that nothing added later can be mistaken
for a pre-planned test. Numbers are from `analysis_summary.md` at commit time.

## Preregistered outcomes (Section 6 of PREREGISTRATION.md)

- Headline R = SD_partition(B, MoM) / SD_seed(A):
  Lift 0.66 [0.00, 1.35]; Can 0.18 [0.00, 1.71]; Square 1.08 [0.00, 1.94].
  No task excludes 1. Under the preregistered decision rule the primary claim
  is NOT established on any task at this gate. Holm-adjusted one-sided p = 1.0
  for all three.
- MoM and REML agree on every component on all three tasks (no material
  disagreement after the boundary-flag fix; Lift's seed component is at the
  boundary under both estimators).
- Boundary estimates: Lift seed component = 0 (both estimators). Reported as a
  boundary estimate, not as zero seed variance.
- Secondary (tier): exact permutation p = 0.20 (Can, Square), 0.47 (Lift);
  floor is 0.067 by design (Amendment 1a). Effect sizes: on Square the
  between-tier share is 0.94 against a null mean of 0.40, and the within-tier
  mean absolute difference is 0.038 versus 0.124 between tiers. This is the
  OPPOSITE of prediction P4: on Square the tier label captures most of the
  operator effect, driven by the two "better" operators (dropping either one
  lowers success from about 0.65 to about 0.47).
- Power of the preregistered headline procedure at the fitted nuisance SDs is
  low: on Square, 14% at a true R of 2 and 42% at a true R of 3. A negative
  headline result is therefore weak evidence about the size of the effect.

## Observations that are NOT preregistered tests (exploratory, flagged as such)

1. The preregistered denominator SD_seed(A) is the raw spread across seeds,
   which includes the binomial noise of a 50-rollout estimate (0.069 on
   Square at a 0.60 success rate). The numerator SD_partition is a residual-
   corrected variance component. The headline ratio therefore compares a
   corrected quantity against an uncorrected one and is conservative against
   the claim. This was noted at the Arm A gate before any Arm B data. A
   corrected-denominator ratio (Arm A seed SD after removing the binomial
   floor in quadrature: 0.039 on Square, giving R about 2.2) will be reported
   only as an exploratory sensitivity analysis, clearly labelled.
2. The within-table ratio SD_partition / SD_seed from the Arm B 6 x 3 table
   (a preregistered secondary view) is 3.13 on Square, 0.17 on Can, and
   undefined on Lift (seed component at boundary).
3. The exact F-test for the partition component (implemented in `vc.py`
   before any data, but not named in the preregistration text) gives
   p = 0.003 on Square, 0.45 on Can, 0.09 on Lift.
4. The REML bootstrap interval for the Can headline ratio is degenerate
   (upper end about 1.9e9) because many resamples put the Arm A seed SD near
   zero. It is reported as degenerate rather than as an informative interval.

## What Arm C decides

If the Arm C partition component on Square is as large as Arm B's 0.086,
the operator effect is not separable from demonstration resampling at 250
demos and the paper is a negative result with the power statement above.
If Arm C's partition component is materially smaller (D > 0 with interval
excluding 0 per 6.5), the operator-composition effect is supported on Square
even though the preregistered headline ratio did not clear 1. Either way the
headline outcome is reported as it stands.
