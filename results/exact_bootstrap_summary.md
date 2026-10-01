# Exact enumeration of the preregistered bootstraps

Same statistics and resampling scheme as Sections 6.4 and 6.5; all 462 x 462 (D) and
462 x 92,378 (R) resamples enumerated with multinomial weights, so there is no Monte Carlo error.

| stage | task | D [95%] | p(D<=0) | R [95%] | p(R<=1) |
|---|---|---|---|---|---|
| stage1 | lift | 0.00012 [-0.00015, 0.00025] | 0.1937 | 0.66 [0.00, 1.52] | 0.8932 |
| stage1 | can | -0.00058 [-0.00170, 0.00044] | 0.7933 | 0.18 [0.00, 1.63] | 0.8802 |
| stage1 | square | 0.00622 [-0.00135, 0.01049] | 0.0844 | 1.08 [0.00, 1.93] | 0.4273 |
| stage2 | square | 0.00269 [-0.00000, 0.00523] | 0.0251 | 1.71 [0.52, 3.18] | 0.1260 |

## Holm-Bonferroni with exact p-values

**stage1**

| contrast | p raw | p Holm |
|---|---|---|
| lift:R>1 | 0.8932 | 1.0000 |
| lift:D>0 | 0.1937 | 0.9684 |
| can:R>1 | 0.8802 | 1.0000 |
| can:D>0 | 0.7933 | 1.0000 |
| square:R>1 | 0.4273 | 1.0000 |
| square:D>0 | 0.0844 | 0.5062 |

**stage2**

| contrast | p raw | p Holm |
|---|---|---|
| square:R>1 | 0.1260 | 0.1260 |
| square:D>0 | 0.0251 | 0.0501 |

