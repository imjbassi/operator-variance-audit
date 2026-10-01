"""Generate fixed evaluation initial states for a task, once, from seeded resets.

The environment is built from the dataset's own env metadata (same robosuite env, version,
controller config). For state i, numpy's global RNG (which robosuite placement samplers use)
is seeded with seed_base + i before env.reset(); the flattened MuJoCo state AND the model XML
are stored. Evaluation later calls env.reset() followed by
env.reset_to({"model": xml_i, "states": s_i}) so every checkpoint of a task sees exactly the
same initial conditions.

NOTE: robosuite >= 1.4 regenerates the MJCF on every hard reset and some tasks randomise
geometry per episode (Lift: cube size), hence the per-state model XML.

Stage 1 (preregistered): 50 states, seed base 100000  -> initial_states/<task>.npz
Stage 2 (Amendment 2):  500 states, seed base 300000  -> initial_states/<task>_stage2.npz

Refuses to overwrite (initial states are part of the frozen design).

Usage:
  python scripts/gen_initial_states.py --task square --hdf5 ~/ova/datasets/pristine/square/mh/low_dim_v15.hdf5
  python scripts/gen_initial_states.py --task square --hdf5 <...> --n 500 --seed_base 300000 --name square_stage2
"""
import argparse, hashlib, json, os, sys
import numpy as np

import robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

HERE = os.path.dirname(os.path.abspath(__file__))
N_STATES = 50
INIT_SEED_BASE = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--n", type=int, default=N_STATES)
    ap.add_argument("--seed_base", type=int, default=INIT_SEED_BASE)
    ap.add_argument("--name", default=None, help="output basename (default: task)")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "initial_states"))
    a = ap.parse_args()
    name = a.name or a.task
    out = os.path.join(a.out_dir, f"{name}.npz")
    meta_out = os.path.join(a.out_dir, f"{name}.json")
    if os.path.exists(out):
        print(f"REFUSING to overwrite {out}"); sys.exit(2)

    # low-dim obs spec so the env wrapper knows which keys to produce
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": [
        "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"], "rgb": []}})
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=os.path.expanduser(a.hdf5))
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)

    states, models = [], []
    for i in range(a.n):
        np.random.seed(a.seed_base + i)
        env.reset()
        st = env.get_state()
        states.append(np.array(st["states"], dtype=np.float64))
        models.append(st["model"])
    states = np.stack(states)
    model_hashes = [hashlib.sha256(m.encode()).hexdigest()[:16] for m in models]
    assert len({hashlib.sha256(s.tobytes()).hexdigest() for s in states}) == a.n, "duplicate initial states"
    print(f"{a.n} states, state dim {states.shape[1]}, distinct model xml hashes: {len(set(model_hashes))}")

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(out, states=states, models=np.array(models))  # unicode array, no pickle
    meta = {
        "task": a.task, "name": name, "n_states": a.n, "init_seed_base": a.seed_base,
        "env_name": env_meta["env_name"], "env_version": env_meta.get("env_version"),
        "source_hdf5": os.path.basename(a.hdf5),
        "states_sha256": hashlib.sha256(states.tobytes()).hexdigest(),
        "model_xml_sha256_16": model_hashes,
        "n_distinct_model_xml": len(set(model_hashes)),
        "robomimic_version": robomimic.__version__,
    }
    try:
        import robosuite, mujoco
        meta["robosuite_version"] = robosuite.__version__
        meta["mujoco_version"] = mujoco.__version__
    except Exception:
        pass
    with open(meta_out, "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {out} and {meta_out}")


if __name__ == "__main__":
    main()
