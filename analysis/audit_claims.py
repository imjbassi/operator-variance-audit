"""Mechanical audit of the hand-written quantitative statements in paper/paper.tex that are not
macros (caption claims, rounded figures in the recommendations, structural facts). Each check
recomputes the statement from the committed result files and prints PASS/FAIL.

Usage:  python analysis/audit_claims.py      (exit code 1 if any check fails)
"""
import json, os, sys
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
fails = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)


def load(n):
    with open(os.path.join(RES, n)) as fh:
        return json.load(fh)


a1, a2, ex, expl = load("analysis.json"), load("analysis_stage2.json"), load("exact_bootstrap.json"), load("exploratory.json")
s1 = pd.read_csv(os.path.join(RES, "runs.csv")); s2 = pd.read_csv(os.path.join(RES, "runs_stage2.csv"))
r1 = pd.read_csv(os.path.join(RES, "rollouts.csv")); r2 = pd.read_csv(os.path.join(RES, "rollouts_stage2.csv"))

# counts stated in the text
check(len(s1) == 138 and len(r1) == 6900, f"138 training runs and 6,900 Stage 1 rollouts (found {len(s1)}, {len(r1)})")
check(len(s2) == 46 and len(r2) == 23000, f"46 Square checkpoints and 23,000 Stage 2 rollouts (found {len(s2)}, {len(r2)})")
check((s1.groupby(["task", "arm"]).size().unstack().values == np.array([[10, 18, 18]] * 3)).all(), "10 / 18 / 18 runs per task in Arms A / B / C")
check((s1.epochs == 2000).all(), "every run evaluated at epoch 2000 (final checkpoint)")
check(set(s2.ckpt_sha256) <= set(s1.ckpt_sha256), "Stage 2 evaluated exactly the Stage 1 Square checkpoints (sha256 match)")

# per-run success rates equal the per-rollout means
m1 = r1.groupby("run_id").success.mean()
check(np.allclose(s1.set_index("run_id").success_rate.sort_index(), m1.sort_index()), "Stage 1 run success rates equal per-rollout means")
m2 = r2.groupby("run_id").success.mean()
check(np.allclose(s2.set_index("run_id").success_rate.sort_index(), m2.sort_index()), "Stage 2 run success rates equal per-rollout means")

# Table I caption: REML agrees with MoM to within 0.003 on every entry
mx = 0.0
for T in list(a1["tasks"].values()) + list(a2["tasks"].values()):
    for arm in ("B", "C"):
        for k in ("sd_p", "sd_s", "sd_e"):
            mx = max(mx, abs(T[arm]["mom"][k] - T[arm]["reml"][k]))
check(mx <= 0.003 + 1e-9, f"REML and MoM agree to within 0.003 on every component (max |diff| = {mx:.4f})")
check(not any(T[arm]["disagree"][c] for T in list(a1["tasks"].values()) + list(a2["tasks"].values())
              for arm in ("B", "C") for c in ("p", "s")), "no material estimator disagreement flagged on any component")

# Section IV-A: Can spread below floor, Lift marginally above, Square above
A = {t: a1["tasks"][t]["A"] for t in ("lift", "can", "square")}
check(A["can"]["sd_seed"] < A["can"]["binomial_sd_50_rollouts"], "Can: seed SD below the binomial floor")
check(0 < A["lift"]["sd_seed"] - A["lift"]["binomial_sd_50_rollouts"] < 0.005, "Lift: seed SD marginally above the floor (< 0.005)")
check(A["square"]["sd_seed"] > A["square"]["binomial_sd_50_rollouts"], "Square: seed SD above the floor")

# Section IV-B statements
for t in ("lift", "can", "square"):
    h, d = a1["tasks"][t]["headline_ratio_mom"], a1["tasks"][t]["B_minus_C"]
    check(h["ci_low"] <= 1.0, f"Stage 1 {t}: R interval does not exclude 1")
    check(d["ci_low"] <= 0.0, f"Stage 1 {t}: D interval does not exclude 0")
check(a1["tasks"]["can"]["C"]["mom"]["sd_p"] > a1["tasks"]["can"]["B"]["mom"]["sd_p"], "Can: random partitions vary more than operator partitions")
lb = a1["tasks"]["lift"]
check(sum([lb["B"]["mom"]["boundary_s"], lb["C"]["mom"]["boundary_p"], lb["C"]["mom"]["boundary_s"]]) >= 2, "Lift: several boundary estimates")
pw = {t: {round(v["true_R"], 1): v["power"] for v in a1["tasks"][t]["power_headline_procedure"].values()} for t in ("lift", "can", "square")}
check(all(pw[t][r] <= pw["square"][r] + 0.02 for t in ("lift", "can") for r in (2.0, 3.0, 4.0)), "power on Lift and Can is no better than on Square")

# Section IV-C: Stage 2 statements
T2 = a2["tasks"]["square"]
check(T2["headline_ratio_mom"]["ci_low"] <= 1.0, "Stage 2: R interval does not exclude 1")
p_mc = [e for e in a2["primary_family"] if e["contrast"] == "square:D>0"][0]["p_holm"]
p_ex = [e for e in ex["stage2"]["holm"] if e["contrast"] == "square:D>0"][0]["p_holm"]
check(p_mc < 0.05 < p_ex and abs(p_mc - p_ex) < 0.002, f"Stage 2 D: resampled Holm p {p_mc:.4f} and exact {p_ex:.4f} straddle 0.05 by < 0.002")
check(abs(T2["B"]["mom"]["sd_e"] - T2["C"]["mom"]["sd_e"]) < 0.001, "Stage 2: equal residual SDs in Arms B and C (to 3 decimals)")
check(ex["stage2"]["square"]["D"]["ci_low"] <= 0.0, "Stage 2: exact D interval lower end is not above 0")
g = expl["stage_agreement"]
check(g["mean_diff_50_minus_500"] < 0, "50-state set gave lower success on average than the 500-state set")
check(abs(g["sd_diff"] - g["expected_sd_diff_binomial"]) < 0.01, "50-vs-500 differences match the binomial expectation (within 0.01)")

# Section IV-D: tier statements
b1, b2 = a1["tasks"]["square"]["B"], T2["B"]
pm1, pm2 = b1["partition_means"], b2["partition_means"]
moved = {k: abs(pm2[k] - pm1[k]) for k in pm1}
check(max(moved, key=moved.get) == "better_operator_1", "partition whose mean moved most between stages: better_operator_1 held out")
others = [v for k, v in pm1.items() if not k.startswith("better")]
check(max(pm1["better_operator_1"], pm1["better_operator_2"]) < min(others), "Stage 1: both drop-better partitions below all others")
check(b2["tier_pairwise"]["within_tier_mean_absdiff"] >= b2["tier_pairwise"]["between_tier_mean_absdiff"], "Stage 2: within-tier |diff| as large as between-tier")

# Section V: rounded figures in the recommendations
check(round(100 * np.sqrt(0.6 * 0.4 / 50)) == 7 and round(100 * np.sqrt(0.6 * 0.4 / 500)) == 2, "binomial SE at 60%: 7 points (n=50), 2 points (n=500)")

# Section III: partition structure
P = json.load(open(os.path.join(REPO, "partitions", "square.json")))["partitions"]
Bk = [k for k in P if k.startswith("B_")]
check(all(len(set(P[a]) & set(P[b])) == 200 for i, a in enumerate(Bk) for b in Bk[i + 1:]), "any two Arm B partitions share exactly 200 demonstrations")
Ck = [k for k in P if k.startswith("C_")]
ov = [len(set(P[a]) & set(P[b])) for i, a in enumerate(Ck) for b in Ck[i + 1:]]
check(195 <= np.mean(ov) <= 220, f"Arm C partitions have comparable overlap (mean {np.mean(ov):.0f} of 250)")

print(f"\n{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
