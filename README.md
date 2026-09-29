# Operator-Composition Variance in robomimic Multi-Human Benchmarks

Audit of how much of the reported uncertainty on robomimic Multi-Human (MH)
benchmarks is explained by *which operators' demonstrations are in the training
set* versus *training seed*. Preregistered design: see
[PREREGISTRATION.md](PREREGISTRATION.md). Gating check on whether operator
identity is recoverable from the released files: see
[STEP0_operator_identity.md](STEP0_operator_identity.md).

## Layout

| Path | Contents |
|------|----------|
| `PREREGISTRATION.md` | Frozen predictions, falsification criteria, analysis plan, amendments log |
| `STEP0_operator_identity.md` | Verification that per-operator filter keys ship in the MH files |
| `env/` | Pinned environment build (`setup_wsl_env.sh`), dataset download, resolved lock file |
| `partitions/` | Immutable partition assignments (JSON) for Arms A, B, C |
| `initial_states/` | The 50 fixed evaluation initial states per task |
| `configs/` | Base BC-RNN low-dim config and the run manifest |
| `scripts/` | Partition generation, dataset verification, training/eval queue |
| `analysis/` | Variance-component estimators, bootstrap, permutation, tables |
| `results/` | Per-run metrics and per-rollout outcomes (CSV); everything in the paper regenerates from these |

## Compute layout (WSL2)

Code and small artifacts live in this repository. Datasets, checkpoints and
per-run logs live on native WSL storage under `~/ova/` and are not committed.
Aggregated CSVs are written back into `results/`.
