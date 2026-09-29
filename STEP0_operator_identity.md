# Step 0 — Is per-operator identity recoverable from robomimic MH HDF5 files?

Date: 2026-09-29
Files inspected (WSL, `~/robomimic/datasets/`):
- `can/mh/low_dim_v141.hdf5` (robosuite env_version 1.4.1, 300 demos)
- `square/mh/low_dim_v15.hdf5` (robosuite env_version 1.5.1, 300 demos)
- Lift MH: not present locally (not yet downloaded)

## Verdict

**Operator identity is read directly from the files. No inference is required.**

Each MH file ships six filter keys under `mask/`:

| filter key          | n demos | demo index block |
|---------------------|--------:|------------------|
| better_operator_1   | 50      | 0–49             |
| worse_operator_2    | 50      | 50–99            |
| okay_operator_2     | 50      | 100–149          |
| better_operator_2   | 50      | 150–199          |
| worse_operator_1    | 50      | 200–249          |
| okay_operator_1     | 50      | 250–299          |

Programmatic checks (both files):
- Six operator keys are pairwise disjoint. Union equals the full set of 300 demos.
- `{tier}_operator_1 ∪ {tier}_operator_2 == {tier}` for better / okay / worse.
- Per-operator `_train` / `_valid` keys ship too and equal `operator ∩ global train/valid`.
- Demo-index → operator layout is identical in Can and Square (fingerprint `a1072f96389d7149`).
- Every demo ends in success (final reward 1.0). Trajectory-length ordering by tier is as expected
  (worse operators longest and most variable) in both tasks.

Per-demo attributes are only `model_file` and `num_samples` (plus empty `camera_info` in v1.5).
There is no per-demo operator, session, or timestamp attribute; the filter keys are the only
carrier of operator identity. The contiguous 50-demo block structure also recovers the same
grouping from ordering alone, but the analysis will use the filter keys, not ordering.

## Provenance

The two local files were modified by a prior project (smoothness audit): ten extra keys
(`ted_top50`, `sal_top50`, `jerk_top50`, `path_len_top50`, `oracle_top50`, `random_top50`,
`nocuration_top50`, and `_norm` variants) were written by `~/robomimic/score.py`.
No local script references `operator_` or writes those keys, so the operator keys are stock.
Recommendation: re-download all three MH files fresh for this project, verify the operator-key
fingerprint above, and keep the files read-only.

## Caveats that shape what the paper can claim

1. `operator_k` is an index **within a tier**. Nothing in the files or docs states whether
   `better_operator_1` on Can is the same human as `better_operator_1` on Square. Per-task
   analysis (as designed) is unaffected. Any cross-task "same operator" statement is unsupported.
2. Tier labels were assigned by the robomimic authors; the files carry no proficiency metric.
3. Six groups per task. Labels are exact, but the design is still thin for variance components
   (already addressed in the design: MoM + REML, boundary estimates flagged).

## Environment findings relevant to Step 1 (decision needed)

- Installed venv (`~/robomimic/.venv`): Python 3.11, robomimic 0.5.0, robosuite **1.4.1**,
  mujoco 2.3.2, torch 2.13, numpy 2.4, h5py 3.16. GPU: RTX 4070 12 GB. Disk: 643 GB free.
- robomimic 0.5.0's dataset registry serves **v1.5 datasets only** (lift/can/square MH low_dim).
- The local Square file is v1.5 (robosuite 1.5 composite controller); the local Can file is
  v1.4.1. Rollouts require the simulator version to match the dataset version. The Square v1.5
  file cannot be rolled out under the installed robosuite 1.4.1.
- Recommendation: standardize on **v1.5 datasets for all three tasks** in a fresh, pinned venv
  with robosuite 1.5.x (matches robomimic 0.5.0). Fresh downloads: ~120 MB each.
