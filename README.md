# Operator-Composition Variance in robomimic Multi-Human Benchmarks

A preregistered audit of how the uncertainty reported on the robomimic Multi-Human (MH)
benchmarks decomposes: training seed, which demonstrations are in the training set, which
operators are in the training set, and evaluation (rollout) noise.

**Outcome in one paragraph.** The preregistered claim, that operator-composition variance exceeds
seed variance, was **not supported on any task** (Lift, Can, Square; 138 BC-RNN runs). With six
operators the preregistered test has about 18% power at a true ratio of two, so that negative
result bounds little. What the data do establish is about the reporting convention: on Lift and
Can the across-seed spread of a 50-rollout success rate is indistinguishable from binomial rollout
noise; on Square two 3-seed means must differ by about 13 points to be distinguishable; and when
the same 46 Square checkpoints are re-evaluated with 500 rollouts, the 50-rollout numbers correlate
only r = 0.56 with the 500-rollout ones and an apparent proficiency-tier pattern disappears. At
500 rollouts the operator-partition SD is 0.054 against 0.014 for size-matched random partitions,
a contrast that sits exactly on the decision boundary and is not claimed.

- Preregistration and dated amendments: [PREREGISTRATION.md](PREREGISTRATION.md)
- Gating check that operator identity ships in the files: [STEP0_operator_identity.md](STEP0_operator_identity.md)
- Draft paper: [paper/paper.pdf](paper/paper.pdf) (source `paper/paper.tex`)
- Result summaries: `results/analysis_summary.md` (Stage 1), `results/analysis_stage2_summary.md`
  (Stage 2), `results/exact_bootstrap_summary.md`, `results/exploratory_summary.md`,
  `results/GATE_B_notes.md` (notes written before any Arm C result existed)

## Layout

| Path | Contents |
|------|----------|
| `PREREGISTRATION.md` | Frozen predictions, falsification criteria, analysis plan, Amendments 1-3 |
| `env/` | Pinned environment build, dataset download and checksums, resolved lock file, worker launchers, Windows watchdog |
| `partitions/` | Immutable partition assignments (JSON) for Arms A, B, C |
| `initial_states/` | Fixed evaluation initial conditions: 50 per task (Stage 1), 500 for Square (Stage 2) |
| `configs/` | Base BC-RNN low-dim config |
| `scripts/` | Dataset verification, partition generation, training/eval queue, Stage 2 evaluation queue, collectors |
| `analysis/` | Estimators (`vc.py`), preregistered analysis, exact bootstrap, exploratory analyses, figures, paper numbers |
| `results/` | Per-run and per-rollout CSVs for both stages and all analysis outputs |
| `figures/`, `paper/` | Generated figures; paper source, generated macros and tables, PDF |

## Regenerating everything from the committed CSVs

Inside the pinned environment (see `env/setup_wsl_env.sh`):

```bash
python analysis/run_analysis.py                                   # Stage 1, results/analysis*.{json,md}
python analysis/run_analysis.py --runs results/runs_stage2.csv --tasks square --out_prefix analysis_stage2
python analysis/exact_bootstrap.py
python analysis/exploratory.py
bash paper/build.sh                                               # numbers, tables, figures, PDF
```

Bootstrap seeds are fixed, so these reproduce the committed outputs.

## Re-running the experiments

```bash
bash env/setup_wsl_env.sh            # pinned venv: torch 2.8.0+cu128, robosuite 1.5.1, mujoco 3.2.6, robomimic v0.5.0
bash env/download_datasets.sh        # pristine v1.5 MH low-dim datasets, read-only, with SHA-256 sums
python scripts/verify_datasets.py ~/ova/datasets/pristine/*/mh/low_dim_v15.hdf5
python scripts/prepare_working_datasets.py --task square --pristine ... --working ...   # per task
ARMS="A B C" NW=3 bash env/launch_workers.sh      # 138 runs, resumable
NW=6 bash env/launch_stage2.sh                    # Stage 2: 46 Square checkpoints x 500 rollouts
python scripts/collect_results.py && python scripts/collect_stage2.py
```

Datasets, checkpoints and per-run logs live on native WSL storage under `~/ova/` and are not
committed. Hardware used: one RTX 4070 (12 GB), WSL2.

## Citing and third-party files

Citation metadata is in `CITATION.cff`; `.zenodo.json` carries the same metadata for a Zenodo
deposit. `paper/IEEEtran.cls` and `paper/IEEEtran.bst` (Michael Shell, V1.8b / 1.14) are
redistributed unmodified from CTAN under the LaTeX Project Public License so the paper builds
without a TeX package install. `python analysis/audit_claims.py` mechanically re-checks the
hand-written quantitative statements in the paper against the result files.

## Compute notes

During the grid the Microsoft Store auto-updated the WSL package twice and terminated the VM. No
result was lost: three runs that had finished training were evaluated from their saved final
checkpoint, and three partially trained runs were retrained from scratch. `env/keepawake.ps1` is
the watchdog that was added to relaunch the idempotent queues after such a kill. One Stage 2
evaluation chunk failed under memory pressure with eight workers and was re-run with six.
