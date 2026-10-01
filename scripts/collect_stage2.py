"""Aggregate Stage 2 (Amendment 2) chunked evaluations into committed CSVs.

  results/runs_stage2.csv      one row per Square checkpoint: success rate over 500 rollouts
  results/rollouts_stage2.csv  one row per rollout (run_id, rollout index, success, steps)

A run is included only when all 10 chunks (rollouts 0..499) are present and its checkpoint
sha256 equals the one evaluated in Stage 1.

Usage:  python scripts/collect_stage2.py [--runs_root ~/ova/runs]
"""
import argparse, csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RUN_FIELDS = ["run_id", "task", "arm", "partition_key", "held_out_operator", "n_demos", "seed",
              "success_rate", "n_rollouts", "epochs", "eval_seconds", "ckpt_sha256", "stage1_success_rate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=os.path.expanduser("~/ova/runs"))
    ap.add_argument("--out_dir", default=os.path.join(REPO, "results"))
    a = ap.parse_args()
    infos, rollouts, incomplete = [], [], []
    for info_path in sorted(glob.glob(os.path.join(a.runs_root, "*", "square__*", "run_info.json"))):
        run_dir = os.path.dirname(info_path)
        with open(info_path) as fh:
            info = json.load(fh)
        rows, secs = [], 0.0
        for start in range(0, 500, 50):
            rp = os.path.join(run_dir, "eval_stage2", f"rollouts_{start:04d}.csv")
            mp = os.path.join(run_dir, "eval_stage2", f"metrics_{start:04d}.json")
            if not (os.path.exists(rp) and os.path.exists(mp)):
                break
            with open(mp) as fh:
                m = json.load(fh)
            assert m["ckpt_sha256"] == info["ckpt_sha256"], f"checkpoint mismatch {info['run_id']}"
            secs += m["eval_seconds"]
            with open(rp) as fh:
                rows += list(csv.DictReader(fh))
        if len(rows) != 500 or sorted(int(r["rollout"]) for r in rows) != list(range(500)):
            incomplete.append((info["run_id"], len(rows)))
            continue
        sr = sum(int(r["success"]) for r in rows) / 500.0
        infos.append({"run_id": info["run_id"], "task": info["task"], "arm": info["arm"],
                      "partition_key": info["partition_key"], "held_out_operator": info["held_out_operator"],
                      "n_demos": info["n_demos"], "seed": info["seed"], "success_rate": sr, "n_rollouts": 500,
                      "epochs": info["epochs"], "eval_seconds": secs, "ckpt_sha256": info["ckpt_sha256"],
                      "stage1_success_rate": info["success_rate"]})
        rollouts += [{"run_id": info["run_id"], **r} for r in sorted(rows, key=lambda r: int(r["rollout"]))]
    infos.sort(key=lambda r: (r["arm"], r["task"], r["partition_key"], int(r["seed"])))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "runs_stage2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RUN_FIELDS); w.writeheader(); w.writerows(infos)
    with open(os.path.join(a.out_dir, "rollouts_stage2.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id", "rollout", "success", "steps"]); w.writeheader(); w.writerows(rollouts)
    print(f"{len(infos)} complete Square checkpoints, {len(rollouts)} rollouts; incomplete: {incomplete}")


if __name__ == "__main__":
    main()
