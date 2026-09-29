"""Resumable, lock-based run queue: train BC-RNN on a partition, then evaluate the FINAL checkpoint.

Manifest (deterministic, from partitions/*.json):
  Arm A : task x A_all            x seeds 1..10
  Arm B : task x B_drop_<op>      x seeds 1..3
  Arm C : task x C_rand_<i>       x seeds 1..3
Run id: <task>__<partition_key>__s<seed>

Each worker loops over the manifest (filtered by --arms/--tasks), claims a run by atomically
creating <run_dir>/.lock, trains, evaluates, writes <run_dir>/eval/metrics.json (completion marker)
and <run_dir>/run_info.json. Runs whose metrics.json exists are skipped; stale locks from dead
workers are removed after --stale_hours. Several workers can run concurrently.

Usage (inside the pinned venv, from the repo root):
  python scripts/run_queue.py --arms A --worker 0 [--tasks lift can square] [--epochs 2000]
  python scripts/run_queue.py --smoke            # 1 tiny run to time the pipeline (not part of the experiment)
"""
import argparse, copy, glob, json, os, shutil, socket, subprocess, sys, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OVA = os.path.expanduser("~/ova")
ROBOMIMIC_TRAIN = os.path.join(OVA, "src", "robomimic", "robomimic", "scripts", "train.py")
TASKS = ["lift", "can", "square"]
SEEDS = {"A": list(range(1, 11)), "B": [1, 2, 3], "C": [1, 2, 3]}


def working_hdf5(task):
    return os.path.join(OVA, "datasets", "working", task, "mh", "low_dim_v15.hdf5")


def build_manifest(arms, tasks):
    runs = []
    for task in tasks:
        with open(os.path.join(REPO, "partitions", f"{task}.json")) as fh:
            P = json.load(fh)
        for key, meta in P["partition_meta"].items():
            arm = meta["arm"]
            if arm not in arms:
                continue
            for seed in SEEDS[arm]:
                runs.append({"run_id": f"{task}__{key}__s{seed}", "task": task, "arm": arm,
                             "partition_key": key, "held_out_operator": meta["held_out_operator"],
                             "n_demos": meta["n"], "seed": seed})
    # interleave tasks so partial progress covers all tasks; order within task: partition then seed
    order = {"A": 0, "B": 1, "C": 2}
    runs.sort(key=lambda r: (order[r["arm"]], r["seed"], r["task"], r["partition_key"]))
    return runs


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def make_config(run, run_dir, epochs):
    with open(os.path.join(REPO, "configs", "base_bc_rnn_lowdim.json")) as fh:
        cfg = json.load(fh)
    cfg["experiment"]["name"] = run["run_id"]
    cfg["train"]["data"] = [{"path": working_hdf5(run["task"]), "filter_key": run["partition_key"]}]
    cfg["train"]["hdf5_filter_key"] = run["partition_key"]
    cfg["train"]["output_dir"] = os.path.join(run_dir, "train")
    cfg["train"]["seed"] = run["seed"]
    cfg["train"]["num_epochs"] = epochs
    cfg["experiment"]["save"]["epochs"] = [epochs]
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=1)
    return path


def try_claim(run_dir, stale_hours):
    lock = os.path.join(run_dir, ".lock")
    os.makedirs(run_dir, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age_h = (time.time() - os.path.getmtime(lock)) / 3600
        if age_h < stale_hours:
            return False
        print(f"removing stale lock ({age_h:.1f} h) in {run_dir}")
        os.remove(lock)
        return try_claim(run_dir, stale_hours)
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"host": socket.gethostname(), "pid": os.getpid(),
                             "time": datetime.datetime.now().isoformat()}))
    return True


def touch(path):
    os.utime(path, None)


def execute(run, run_dir, epochs, n_eval):
    t0 = time.time()
    train_dir = os.path.join(run_dir, "train")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)  # never resume a half-finished run; retrain deterministically from scratch
    cfg_path = make_config(run, run_dir, epochs)
    lock = os.path.join(run_dir, ".lock")
    with open(os.path.join(run_dir, "train_stdout.txt"), "w") as log:
        proc = subprocess.Popen([sys.executable, ROBOMIMIC_TRAIN, "--config", cfg_path],
                                stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        while proc.poll() is None:
            time.sleep(60); touch(lock)  # heartbeat so the lock never looks stale while training
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited {proc.returncode} for {run['run_id']}")
    t_train = time.time() - t0
    ckpts = glob.glob(os.path.join(train_dir, run["run_id"], "*", "models", f"model_epoch_{epochs}.pth"))
    if len(ckpts) != 1:
        raise RuntimeError(f"expected exactly one final checkpoint, found {ckpts}")
    ckpt = ckpts[0]
    t1 = time.time()
    eval_dir = os.path.join(run_dir, "eval")
    cmd = [sys.executable, os.path.join(HERE, "eval_final.py"), "--ckpt", ckpt, "--task", run["task"], "--out_dir", eval_dir]
    if n_eval:
        cmd += ["--n", str(n_eval)]
    with open(os.path.join(run_dir, "eval_stdout.txt"), "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        while proc.poll() is None:
            time.sleep(30); touch(lock)
    if proc.returncode != 0:
        raise RuntimeError(f"eval_final.py exited {proc.returncode} for {run['run_id']}")
    t_eval = time.time() - t1
    with open(os.path.join(eval_dir, "metrics.json")) as fh:
        m = json.load(fh)
    info = dict(run)
    info.update({"epochs": epochs, "train_seconds": t_train, "eval_seconds": t_eval,
                 "success_rate": m["success_rate"], "n_rollouts": m["n_rollouts"],
                 "ckpt_sha256": m["ckpt_sha256"], "git_commit": git_commit(),
                 "finished": datetime.datetime.now().isoformat()})
    with open(os.path.join(run_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    # keep the checkpoint (small, ~10 MB) but drop nothing else; logs stay for audit
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A"], choices=["A", "B", "C"])
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--runs_root", default=os.path.join(OVA, "runs"))
    ap.add_argument("--stale_hours", type=float, default=6.0)
    ap.add_argument("--smoke", action="store_true", help="one 3-epoch run on lift into ~/ova/runs_smoke (timing only)")
    ap.add_argument("--smoke_epochs", type=int, default=3)
    ap.add_argument("--smoke_task", default="lift")
    ap.add_argument("--smoke_n_eval", type=int, default=5)
    a = ap.parse_args()

    if a.smoke:
        run = {"run_id": f"smoke__{a.smoke_task}__A_all__s1", "task": a.smoke_task, "arm": "smoke",
               "partition_key": "A_all", "held_out_operator": None, "n_demos": 300, "seed": 1}
        run_dir = os.path.join(OVA, "runs_smoke", run["run_id"])
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir)
        info = execute(run, run_dir, a.smoke_epochs, a.smoke_n_eval)
        print(json.dumps(info, indent=1))
        return

    manifest = build_manifest(a.arms, a.tasks)
    with open(os.path.join(a.runs_root, f"manifest_{'_'.join(a.arms)}.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[worker {a.worker}] manifest: {len(manifest)} runs for arms {a.arms}")
    done = skipped = 0
    for run in manifest:
        run_dir = os.path.join(a.runs_root, run["arm"], run["run_id"])
        if os.path.exists(os.path.join(run_dir, "eval", "metrics.json")):
            skipped += 1; continue
        if not try_claim(run_dir, a.stale_hours):
            continue
        print(f"[worker {a.worker}] {datetime.datetime.now():%Y-%m-%d %H:%M} START {run['run_id']}", flush=True)
        try:
            info = execute(run, run_dir, a.epochs, None)
            print(f"[worker {a.worker}] {datetime.datetime.now():%Y-%m-%d %H:%M} DONE  {run['run_id']} "
                  f"sr={info['success_rate']:.2f} train={info['train_seconds']/60:.0f}min eval={info['eval_seconds']/60:.0f}min", flush=True)
            done += 1
        except Exception as e:
            print(f"[worker {a.worker}] FAILED {run['run_id']}: {e}", flush=True)
            with open(os.path.join(run_dir, "FAILED.txt"), "a") as fh:
                fh.write(f"{datetime.datetime.now().isoformat()} {e}\n")
        finally:
            lock = os.path.join(run_dir, ".lock")
            if os.path.exists(lock):
                os.remove(lock)
    print(f"[worker {a.worker}] finished: {done} completed here, {skipped} already done")


if __name__ == "__main__":
    main()
