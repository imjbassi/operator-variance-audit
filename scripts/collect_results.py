"""Aggregate completed runs into the committed CSVs that every paper number regenerates from.

  results/runs.csv      one row per completed run (task, arm, partition, held-out operator, seed,
                        success_rate, n_rollouts, epochs, timings, checkpoint sha, git commit)
  results/rollouts.csv  one row per rollout (run_id, rollout index, success, steps)

Usage:  python scripts/collect_results.py [--runs_root ~/ova/runs]
"""
import argparse, csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

RUN_FIELDS = ["run_id", "task", "arm", "partition_key", "held_out_operator", "n_demos", "seed",
              "success_rate", "n_rollouts", "epochs", "train_seconds", "eval_seconds",
              "ckpt_sha256", "git_commit", "finished"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=os.path.expanduser("~/ova/runs"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "results"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    infos, rollouts = [], []
    for info_path in sorted(glob.glob(os.path.join(a.runs_root, "*", "*", "run_info.json"))):
        with open(info_path) as fh:
            info = json.load(fh)
        infos.append({k: info.get(k) for k in RUN_FIELDS})
        with open(os.path.join(os.path.dirname(info_path), "eval", "rollouts.csv")) as fh:
            for row in csv.DictReader(fh):
                rollouts.append({"run_id": info["run_id"], **row})
    infos.sort(key=lambda r: (r["arm"], r["task"], r["partition_key"], int(r["seed"])))
    with open(os.path.join(a.out_dir, "runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS); w.writeheader(); w.writerows(infos)
    with open(os.path.join(a.out_dir, "rollouts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "rollout", "success", "steps"]); w.writeheader(); w.writerows(rollouts)
    by = {}
    for r in infos:
        by.setdefault((r["arm"], r["task"]), []).append(r)
    print(f"{len(infos)} runs, {len(rollouts)} rollouts")
    for (arm, task), rs in sorted(by.items()):
        print(f"  arm {arm} {task:7s} n={len(rs)}")


if __name__ == "__main__":
    main()
