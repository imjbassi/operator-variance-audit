# Post hoc state-level analyses (Amendment 4; NOT preregistered tests)

## Seed effect on shared initial states (Arm A)

| case | mean | seed SD | binomial floor | per-state floor | chi2 vs binomial p | Cochran's Q (df) | Cochran p | discriminating states |
|---|---|---|---|---|---|---|---|---|
| stage1_lift | 0.988 | 0.017 | 0.015 | 0.015 | 0.290 | 12.00 (9) | 0.2133 | 4 of 50 |
| stage1_can | 0.946 | 0.023 | 0.032 | 0.032 | 0.857 | 4.76 (9) | 0.8551 | 21 of 50 |
| stage1_square | 0.600 | 0.079 | 0.069 | 0.064 | 0.223 | 13.71 (9) | 0.1329 | 46 of 50 |
| stage2_square | 0.639 | 0.031 | 0.021 | 0.020 | 0.024 | 21.97 (9) | 0.0090 | 466 of 500 |

## Smallest resolvable difference between two three-seed means (points, 95%)

Paired uses max(seed SD, per-state floor) (Amendment 5); the raw-seed-SD value is shown for reference.

| case | paired SD used | paired, normal | paired, t(4) | paired with raw seed SD | unpaired, normal | unpaired, t(4) |
|---|---|---|---|---|---|---|
| stage1_lift | 0.017 | 2.7 | 3.8 | 2.7 | 2.8 | 4.0 |
| stage1_can | 0.032 (floor) | 5.1 | 7.2 | 3.7 | 5.1 | 7.2 |
| stage1_square | 0.079 | 12.7 | 18.0 | 12.7 | 13.4 | 18.9 |
| stage2_square | 0.031 | 5.0 | 7.1 | 5.0 | 5.2 | 7.3 |

## Tier pairings (Arm B partition means, Square)

- stage1_square: observed tier share 0.936; 3 of 15 pairings reach it; tier pairing ranks 3; all of those pair the two better operators: True
- stage2_square: observed tier share 0.432; 7 of 15 pairings reach it; tier pairing ranks 7; all of those pair the two better operators: False

## Stage 2 contrast D: Monte Carlo error of the preregistered computation

- resampled (2000): raw p 0.0245 (MC SE 0.0035), Holm p 0.0490 (MC SE 0.0069), interval excludes 0: True
- exact enumeration: raw p 0.0251, Holm p 0.0501

## Power at a fixed ratio across stages (Square)

- stage1_square: seed SD 0.079, residual SD 0.057; at R = 2 the partition SD is 0.159, 2.78 x the residual; power 0.18 (MC SE 0.027); power at R = 1: 0.01
- stage2_square: seed SD 0.031, residual SD 0.045; at R = 2 the partition SD is 0.063, 1.40 x the residual; power 0.13 (MC SE 0.024); power at R = 1: 0.00
- false positives at R = 1: at most 1 of 200 simulations per task and stage (exact 95% upper bound 2.8%); too few runs to show a rate below the nominal 2.5%
