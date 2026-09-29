"""Generate the immutable partition assignments for Arms A, B and C.

Arm A : one partition, all 300 demos.
Arm B : six leave-one-operator-out partitions (250 demos each; drop one operator's 50).
Arm C : six random draws of 250 demos from all 300, ignoring operator identity,
        from numpy.random.default_rng(20260929 + task_offset). Sizes match Arm B.

Operator membership is read from the stock filter keys of the pristine HDF5 file and the
operator-key fingerprint is asserted, so the partitions are tied to a verified grouping.

Output: partitions/<task>.json with
  {"task", "source_sha256", "operator_key_fingerprint", "rng_seed",
   "operators": {op: [demos]}, "partitions": {partition_key: [demos]},
   "partition_meta": {partition_key: {"arm", "held_out_operator"|null, "n"}}}

Usage:
  python scripts/make_partitions.py --task square --hdf5 ~/ova/datasets/pristine/square/mh/low_dim_v15.hdf5
Refuses to overwrite an existing partitions file (immutability).
"""
import argparse, hashlib, json, os, re, sys
import h5py
import numpy as np

OPS = ["better_operator_1", "better_operator_2", "okay_operator_1",
       "okay_operator_2", "worse_operator_1", "worse_operator_2"]
EXPECTED_FINGERPRINT = "a1072f96389d7149"
BASE_SEED = 20260929
TASK_OFFSET = {"lift": 0, "can": 1, "square": 2}
N_SUB = 250


def num(s):
    return int(re.search(r"(\d+)$", s).group(1))


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_OFFSET))
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "partitions"))
    a = ap.parse_args()

    out = os.path.join(a.out_dir, f"{a.task}.json")
    if os.path.exists(out):
        print(f"REFUSING to overwrite existing {out} (partitions are immutable)")
        sys.exit(2)

    with h5py.File(a.hdf5, "r") as f:
        all_demos = sorted(f["data"].keys(), key=num)
        operators = {k: sorted((x.decode() for x in f["mask"][k][()]), key=num) for k in OPS}
    fp = hashlib.sha256("|".join(f"{k}:{','.join(operators[k])}" for k in OPS).encode()).hexdigest()[:16]
    assert fp == EXPECTED_FINGERPRINT, f"fingerprint {fp} != {EXPECTED_FINGERPRINT}"
    assert len(all_demos) == 300 and all(len(v) == 50 for v in operators.values())

    partitions, meta = {}, {}
    # Arm A
    partitions["A_all"] = list(all_demos)
    meta["A_all"] = {"arm": "A", "held_out_operator": None, "n": 300}
    # Arm B
    for k in OPS:
        key = f"B_drop_{k}"
        partitions[key] = [d for d in all_demos if d not in set(operators[k])]
        meta[key] = {"arm": "B", "held_out_operator": k, "n": len(partitions[key])}
        assert meta[key]["n"] == N_SUB
    # Arm C
    seed = BASE_SEED + TASK_OFFSET[a.task]
    rng = np.random.default_rng(seed)
    for i in range(6):
        idx = np.sort(rng.choice(300, size=N_SUB, replace=False))
        key = f"C_rand_{i+1}"
        partitions[key] = [all_demos[j] for j in idx]
        meta[key] = {"arm": "C", "held_out_operator": None, "n": N_SUB,
                     "operator_counts": {k: int(sum(1 for d in partitions[key] if d in set(operators[k]))) for k in OPS}}

    doc = {
        "task": a.task,
        "source_file": os.path.basename(a.hdf5),
        "source_sha256": sha256_file(a.hdf5),
        "operator_key_fingerprint": fp,
        "rng_seed": seed,
        "n_subsample": N_SUB,
        "operators": operators,
        "partitions": partitions,
        "partition_meta": meta,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"wrote {out}")
    for k, m in meta.items():
        print(f"  {k:28s} arm={m['arm']} n={m['n']} " + (str(m.get("operator_counts", "")) if m["arm"] == "C" else ""))


if __name__ == "__main__":
    main()
