"""Stage 2 (Amendment 2): re-evaluate the existing Square final checkpoints at 500 rollouts.

Unit of work = (run, chunk): 46 runs x 10 chunks of 50 rollouts. A chunk is complete when
<run_dir>/eval_stage2/rollouts_<start>.csv exists (written atomically). Workers claim a chunk
with an O_EXCL lock file; the launcher clears stale locks when no worker is alive. No training,
no checkpoint selection: only model_epoch_2000.pth of each completed Stage 1 run is used, and
its sha256 must equal the one recorded in Stage 1's run_info.json.

Usage:  python scripts/eval_stage2_queue.py --worker 0
"""
import argparse, datetime, glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OVA = os.path.expanduser("~/ova")
TASK = "square"
STATES_NAME = "square_stage2"
EVAL_SEED_BASE = 400_000
N_TOTAL, CHUNK = 500, 50


def units(runs_root):
    out = []
    for info_path in sorted(glob.glob(os.path.join(runs_root, "*", f"{TASK}__*", "run_info.json"))):
        run_dir = os.path.dirname(info_path)
        with open(info_path) as fh:
            info = json.load(fh)
        ck = glob.glob(os.path.join(run_dir, "train", info["run_id"], "*", "models", f"model_epoch_{info['epochs']}.pth"))
        assert len(ck) == 1, f"final checkpoint missing for {info['run_id']}: {ck}"
        for start in range(0, N_TOTAL, CHUNK):
            out.append({"run_id": info["run_id"], "run_dir": run_dir, "ckpt": ck[0],
                        "ckpt_sha256": info["ckpt_sha256"], "start": start})
    # chunk-major order so every checkpoint accumulates rollouts evenly
    out.sort(key=lambda u: (u["start"], u["run_id"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--runs_root", default=os.path.join(OVA, "runs"))
    a = ap.parse_args()
    U = units(a.runs_root)
    print(f"[s2 worker {a.worker}] {len(U)} units", flush=True)
    done = 0
    for u in U:
        out_dir = os.path.join(u["run_dir"], "eval_stage2")
        os.makedirs(out_dir, exist_ok=True)
        marker = os.path.join(out_dir, f"rollouts_{u['start']:04d}.csv")
        lock = os.path.join(out_dir, f".lock_{u['start']:04d}")
        if os.path.exists(marker):
            continue
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        try:
            t0 = time.time()
            cmd = [sys.executable, os.path.join(HERE, "eval_final.py"), "--ckpt", u["ckpt"], "--task", TASK,
                   "--states_name", STATES_NAME, "--eval_seed_base", str(EVAL_SEED_BASE),
                   "--start", str(u["start"]), "--n", str(CHUNK), "--chunk", "--out_dir", out_dir]
            with open(os.path.join(out_dir, f"stdout_{u['start']:04d}.txt"), "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            if rc != 0 or not os.path.exists(marker):
                raise RuntimeError(f"eval_final exit {rc}")
            with open(os.path.join(out_dir, f"metrics_{u['start']:04d}.json")) as fh:
                m = json.load(fh)
            assert m["ckpt_sha256"] == u["ckpt_sha256"], "checkpoint differs from the one evaluated in Stage 1"
            done += 1
            print(f"[s2 worker {a.worker}] {datetime.datetime.now():%m-%d %H:%M} {u['run_id']} start={u['start']} "
                  f"sr={m['success_rate']:.2f} {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"[s2 worker {a.worker}] FAILED {u['run_id']} start={u['start']}: {e}", flush=True)
            with open(os.path.join(out_dir, "FAILED_stage2.txt"), "a") as fh:
                fh.write(f"{datetime.datetime.now().isoformat()} start={u['start']} {e}\n")
        finally:
            if os.path.exists(lock):
                os.remove(lock)
    print(f"[s2 worker {a.worker}] finished, {done} chunks here", flush=True)


if __name__ == "__main__":
    main()
