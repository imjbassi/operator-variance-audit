"""Evaluate one FINAL checkpoint on a task's fixed initial states.

No checkpoint selection happens here: the caller passes the epoch-2000 checkpoint. For rollout i:
  torch / numpy seeds  <- eval_seed_base + i   (identical across all checkpoints)
  env.reset(); env.reset_to({"model": XML[i], "states": S[i]})  (identical initial condition)
  run policy for up to HORIZON steps, terminate on success.

Stage 1 (preregistered primary): 50 states in initial_states/<task>.npz, seed base 200000.
  Writes <out_dir>/rollouts.csv and <out_dir>/metrics.json.
Stage 2 (Amendment 2): 500 states in initial_states/<task>_stage2.npz, seed base 400000,
  evaluated in resumable chunks via --start/--n/--chunk; each chunk writes
  <out_dir>/rollouts_<start>.csv and <out_dir>/metrics_<start>.json.

Usage:
  python scripts/eval_final.py --ckpt <model_epoch_2000.pth> --task square --out_dir <run_dir>/eval
  python scripts/eval_final.py --ckpt <...> --task square --states_name square_stage2 \
      --eval_seed_base 400000 --start 100 --n 50 --chunk --out_dir <run_dir>/eval_stage2
"""
import argparse, csv, hashlib, json, os, re, time
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SEED_BASE = 200_000
HORIZON = 500  # robomimic registry horizon for MH lift/can/square


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--states_name", default=None, help="basename in initial_states/ (default: task)")
    ap.add_argument("--eval_seed_base", type=int, default=EVAL_SEED_BASE)
    ap.add_argument("--start", type=int, default=0, help="first rollout index")
    ap.add_argument("--n", type=int, default=None, help="number of rollouts from --start (default: all)")
    ap.add_argument("--chunk", action="store_true", help="write rollouts_<start>.csv / metrics_<start>.json")
    a = ap.parse_args()

    name = a.states_name or a.task
    npz = np.load(os.path.join(HERE, "..", "initial_states", f"{name}.npz"))
    states, models = npz["states"], [str(m) for m in npz["models"]]
    with open(os.path.join(HERE, "..", "initial_states", f"{name}.json")) as fh:
        smeta = json.load(fh)
    assert hashlib.sha256(states.tobytes()).hexdigest() == smeta["states_sha256"], "initial states file altered"
    assert [hashlib.sha256(m.encode()).hexdigest()[:16] for m in models] == smeta["model_xml_sha256_16"], "model xmls altered"
    n = a.n if a.n is not None else states.shape[0] - a.start
    idxs = list(range(a.start, a.start + n))
    assert idxs[-1] < states.shape[0], "rollout index beyond stored initial states"

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=a.ckpt, device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False)
    m_ep = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(a.ckpt))
    ckpt_epoch = int(m_ep.group(1)) if m_ep else ckpt_dict.get("epoch", None)

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    for i in idxs:
        np.random.seed(a.eval_seed_base + i)
        torch.manual_seed(a.eval_seed_base + i)
        policy.start_episode()
        env.reset()
        # restore the exact model XML (per-episode geometry) AND the simulator state
        obs = env.reset_to({"model": models[i], "states": states[i]})
        success, steps = False, 0
        for t in range(a.horizon):
            act = policy(ob=obs)
            obs, r, done, _ = env.step(act)
            steps = t + 1
            if env.is_success()["task"]:
                success = True
                break
        rows.append({"rollout": i, "success": int(success), "steps": steps})
    elapsed = time.time() - t0

    suffix = f"_{a.start:04d}" if a.chunk else ""
    tmp = os.path.join(a.out_dir, f"rollouts{suffix}.csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rollout", "success", "steps"])
        w.writeheader(); w.writerows(rows)
    sr = float(np.mean([r["success"] for r in rows]))
    metrics = {
        "task": a.task, "n_rollouts": n, "start": a.start, "horizon": a.horizon, "success_rate": sr,
        "n_success": int(sum(r["success"] for r in rows)),
        "mean_steps_success": float(np.mean([r["steps"] for r in rows if r["success"]])) if sr > 0 else None,
        "ckpt": os.path.abspath(a.ckpt), "ckpt_sha256": sha256_file(a.ckpt), "ckpt_epoch": ckpt_epoch,
        "eval_seed_base": a.eval_seed_base, "states_name": name, "init_states_sha256": smeta["states_sha256"],
        "eval_seconds": elapsed,
    }
    with open(os.path.join(a.out_dir, f"metrics{suffix}.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)
    os.replace(tmp, os.path.join(a.out_dir, f"rollouts{suffix}.csv"))  # atomic completion marker
    print(f"success_rate={sr:.3f} ({metrics['n_success']}/{n}) epoch={ckpt_epoch} start={a.start} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
