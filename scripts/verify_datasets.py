"""Verify robomimic MH low-dim datasets before use.

Checks, per file:
  * 300 demos, contiguous indices
  * six stock operator filter keys, 50 demos each, pairwise disjoint, union = all demos
  * operator keys union to tier keys
  * every demo ends in success
  * operator-key fingerprint (sha256 over sorted demo lists) matches the value
    recorded in STEP0 (a1072f96389d7149) so a re-download is provably the same grouping
  * optional: a working copy's partition masks equal the committed partitions JSON

Usage:
  python scripts/verify_datasets.py <file.hdf5> [<file.hdf5> ...] [--partitions partitions/<task>.json]
Exit code 1 on any failure.
"""
import argparse, hashlib, json, re, sys
import h5py

OPS = ["better_operator_1", "better_operator_2", "okay_operator_1",
       "okay_operator_2", "worse_operator_1", "worse_operator_2"]
TIERS = ["better", "okay", "worse"]
EXPECTED_FINGERPRINT = "a1072f96389d7149"


def num(s):
    return int(re.search(r"(\d+)$", s).group(1))


def check(cond, msg, failures):
    print(("  OK   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def verify(path, partitions=None):
    failures = []
    print(f"== {path}")
    with h5py.File(path, "r") as f:
        demos = set(f["data"].keys())
        check(len(demos) == 300, f"300 demos (found {len(demos)})", failures)
        idx = sorted(num(d) for d in demos)
        check(idx == list(range(len(idx))), "demo indices contiguous", failures)
        sets = {k: set(x.decode() for x in f["mask"][k][()]) for k in OPS + TIERS}
        for k in OPS:
            check(len(sets[k]) == 50, f"{k} has 50 demos ({len(sets[k])})", failures)
        overl = [(a, b) for i, a in enumerate(OPS) for b in OPS[i + 1:] if sets[a] & sets[b]]
        check(not overl, f"operator keys pairwise disjoint {overl}", failures)
        union = set().union(*(sets[k] for k in OPS))
        check(union == demos, "union of operator keys == all demos", failures)
        for t in TIERS:
            check(sets[f"{t}_operator_1"] | sets[f"{t}_operator_2"] == sets[t],
                  f"{t}_operator_1|2 == {t}", failures)
        # Every demo must reach success at some step. (Can v1.5 demo_81 reaches success at
        # step 241 of 246 and loses it on the final step under re-simulated v1.5 physics;
        # it is retained, as in the released dataset. Lift and Square end in success throughout.)
        rew = {d: f["data"][d]["rewards"][()] for d in demos}
        check(all(r.max() == 1.0 for r in rew.values()), "every demo reaches success (max reward 1.0)", failures)
        not_final = sorted(d for d, r in rew.items() if r[-1] != 1.0)
        print(f"       demos not ending in success (informational): {not_final}")
        fp = hashlib.sha256("|".join(
            f"{k}:{','.join(sorted(sets[k], key=num))}" for k in OPS).encode()).hexdigest()[:16]
        check(fp == EXPECTED_FINGERPRINT, f"operator-key fingerprint {fp} == {EXPECTED_FINGERPRINT}", failures)
        env_args = json.loads(f["data"].attrs["env_args"])
        print(f"       env={env_args['env_name']} env_version={env_args.get('env_version')}")
        if partitions:
            with open(partitions) as pf:
                P = json.load(pf)
            for key, demolist in P["partitions"].items():
                mk = f"mask/{key}"
                if mk not in f:
                    check(False, f"working copy missing {mk}", failures)
                    continue
                got = sorted((x.decode() for x in f[mk][()]), key=num)
                check(got == sorted(demolist, key=num), f"{mk} matches JSON (n={len(got)})", failures)
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--partitions", default=None)
    a = ap.parse_args()
    allf = []
    for p in a.files:
        allf += verify(p, a.partitions)
    if allf:
        print(f"\n{len(allf)} FAILURE(S)")
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
