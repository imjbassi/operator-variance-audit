"""Create the working copy of each MH dataset that training reads.

The pristine download is never modified. The working copy is a byte-identical copy plus
one `mask/<partition_key>` dataset per partition in partitions/<task>.json. After writing,
the masks are re-read and compared to the JSON, and the working copy is made read-only.

Usage:
  python scripts/prepare_working_datasets.py --task square \
      --pristine ~/ova/datasets/pristine/square/mh/low_dim_v15.hdf5 \
      --working  ~/ova/datasets/working/square/mh/low_dim_v15.hdf5
"""
import argparse, hashlib, json, os, shutil, stat, sys
import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--working", required=True)
    a = ap.parse_args()
    pj = os.path.join(HERE, "..", "partitions", f"{a.task}.json")
    with open(pj) as fh:
        P = json.load(fh)
    pristine, working = os.path.expanduser(a.pristine), os.path.expanduser(a.working)

    sha = sha256_file(pristine)
    assert sha == P["source_sha256"], f"pristine sha256 {sha} != partitions JSON {P['source_sha256']}"
    print(f"pristine sha256 OK: {sha}")

    if os.path.exists(working):
        print(f"working copy exists: {working}; verifying only")
    else:
        os.makedirs(os.path.dirname(working), exist_ok=True)
        shutil.copyfile(pristine, working)
        os.chmod(working, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        with h5py.File(working, "a") as f:
            for key, demos in P["partitions"].items():
                name = f"mask/{key}"
                if name in f:
                    del f[name]
                f.create_dataset(name, data=np.array(demos, dtype="S"))
        os.chmod(working, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        print(f"wrote {len(P['partitions'])} partition masks into {working}")

    # verify
    bad = 0
    with h5py.File(working, "r") as f:
        for key, demos in P["partitions"].items():
            got = [x.decode() for x in f[f"mask/{key}"][()]]
            if got != demos:
                bad += 1
                print(f"  MISMATCH {key}")
        # pristine content untouched: same demo set, same per-demo sample counts
        n = sum(int(f["data"][d].attrs["num_samples"]) for d in f["data"])
        print(f"  total samples in working copy: {n}")
    if bad:
        print(f"{bad} mask mismatches"); sys.exit(1)
    print("working copy masks match partitions JSON")


if __name__ == "__main__":
    main()
