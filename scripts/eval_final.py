"""Evaluate one FINAL checkpoint on the task's 50 fixed initial states.

No checkpoint selection happens here: the caller passes the epoch-2000 checkpoint. For rollout i:
  torch / numpy seeds  <- EVAL_SEED_BASE + i   (identical across all checkpoints)
  env.reset(); env.reset_to({"states": S[i]})  (identical initial state across all checkpoints)
  run policy for up to HORIZON steps, terminate on success.
Writes  <out_dir>/rollouts.csv  (one row per rollout)  and  <out_dir>/metrics.json.

Usage:
  python scripts/eval_final.py --ckpt <model_epoch_2000.pth> --task square --out_dir <run_dir>/eval
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
    ap.add_argument("--n", type=int, default=None, help="override number of rollouts (smoke tests only)")
    a = ap.parse_args()

    npz = np.load(os.path.join(HERE, "..", "initial_states", f"{a.task}.npz"))
    states, models = npz["states"], [str(m) for m in npz["models"]]
    with open(os.path.join(HERE, "..", "initial_states", f"{a.task}.json")) as fh:
        smeta = json.load(fh)
    assert hashlib.sha256(states.tobytes()).hexdigest() == smeta["states_sha256"], "initial states file altered"
    assert [hashlib.sha256(m.encode()).hexdigest()[:16] for m in models] == smeta["model_xml_sha256_16"], "model xmls altered"
    n = a.n or states.shape[0]

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=a.ckpt, device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False)
    m_ep = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(a.ckpt))
    ckpt_epoch = int(m_ep.group(1)) if m_ep else ckpt_dict.get("epoch", None)

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    for i in range(n):
        np.random.seed(EVAL_SEED_BASE + i)
        torch.manual_seed(EVAL_SEED_BASE + i)
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

    with open(os.path.join(a.out_dir, "rollouts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rollout", "success", "steps"])
        w.writeheader(); w.writerows(rows)
    sr = float(np.mean([r["success"] for r in rows]))
    metrics = {
        "task": a.task, "n_rollouts": n, "horizon": a.horizon, "success_rate": sr,
        "n_success": int(sum(r["success"] for r in rows)),
        "mean_steps_success": float(np.mean([r["steps"] for r in rows if r["success"]])) if sr > 0 else None,
        "ckpt": os.path.abspath(a.ckpt), "ckpt_sha256": sha256_file(a.ckpt), "ckpt_epoch": ckpt_epoch,
        "eval_seed_base": EVAL_SEED_BASE, "init_states_sha256": smeta["states_sha256"],
        "eval_seconds": elapsed,
    }
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)
    print(f"success_rate={sr:.3f} ({metrics['n_success']}/{n}) epoch={ckpt_epoch} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
