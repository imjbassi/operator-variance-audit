"""EXPLORATORY and descriptive analyses. Nothing here is a preregistered test.

These were declared as exploratory in PREREGISTRATION.md Amendment 2 before Stage 2 was run:
  (i)   headline ratio with the Arm A seed SD corrected for the binomial floor in quadrature
  (ii)  exact F-tests for the partition component (reported by run_analysis.py; repeated here)
  (iii) tier-level contrast: partitions that drop a "better" operator versus the rest
Plus descriptive quantities used in the paper's framing:
  (iv)  resolvability: smallest difference between two methods that a mean-of-3-seeds comparison
        can resolve, given the measured seed-level spread at fixed data
  (v)   Stage 1 (50 rollouts) versus Stage 2 (500 rollouts) agreement on the 46 Square checkpoints

Outputs: results/exploratory.json and results/exploratory_summary.md
Usage:   python analysis/exploratory.py
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPS = ["better_operator_1", "better_operator_2", "okay_operator_1",
       "okay_operator_2", "worse_operator_1", "worse_operator_2"]
B_KEYS = [f"B_drop_{op}" for op in OPS]
C_KEYS = [f"C_rand_{i}" for i in range(1, 7)]


def table(df, arm, task, keys):
    sub = df[(df.arm == arm) & (df.task == task)]
    return np.array([[float(sub[(sub.partition_key == k) & (sub.seed == s)].success_rate.iloc[0])
                      for s in (1, 2, 3)] for k in keys])


def corrected_ratio(B, A, n_roll, n_boot, rng):
    """R_corr = SD_p(B, MoM) / sqrt(max(SD(A)^2 - mean p(1-p)/n, 0)); joint bootstrap as in 6.4."""
    def stat(Bt, At):
        sp = vc.mom_two_way(Bt)["sd_p"]
        p = At.mean()
        den2 = At.var(ddof=1) - p * (1 - p) / n_roll
        return sp / np.sqrt(den2) if den2 > 0 else np.inf
    point = stat(B, A)
    draws = np.array([stat(B[rng.integers(0, 6, 6)], A[rng.integers(0, A.size, A.size)]) for _ in range(n_boot)])
    fin = np.isfinite(draws)
    return {"point": float(point) if np.isfinite(point) else None,
            "ci_low": float(np.percentile(draws[fin], 2.5)) if fin.any() else None,
            "ci_high": float(np.percentile(draws[fin], 97.5)) if fin.any() else None,
            "frac_denominator_at_or_below_floor": float(1 - fin.mean()),
            "p_R_le_1": float(np.mean(draws <= 1.0))}


def better_vs_rest(Bt):
    """Mean of the two partitions that drop a 'better' operator minus mean of the other four.
    Exact permutation over the 15 ways of choosing which two partitions are 'better'."""
    rm = Bt.mean(axis=1)
    obs = rm[:2].mean() - rm[2:].mean()
    diffs = []
    for pair in itertools.combinations(range(6), 2):
        mask = np.zeros(6, bool); mask[list(pair)] = True
        diffs.append(rm[mask].mean() - rm[~mask].mean())
    diffs = np.array(diffs)
    return {"diff": float(obs), "mean_drop_better": float(rm[:2].mean()), "mean_rest": float(rm[2:].mean()),
            "perm_p_one_sided": float(np.mean(diffs <= obs + 1e-12)), "perm_floor": 1 / 15}


def resolvability(A):
    """Smallest true difference between two methods, each reported as the mean of 3 seeds, that is
    distinguishable at two-sided 95%: normal approximation and Welch-style t with 4 df."""
    sd = A.std(ddof=1)
    se = sd * np.sqrt(2 / 3)
    return {"sd_seed_level": float(sd), "se_diff_of_3seed_means": float(se),
            "mdd_normal_95": float(1.96 * se), "mdd_t4_95": float(stats.t.ppf(0.975, 4) * se),
            "sd_of_3seed_sd_relative": float(np.sqrt(1 / (2 * 2)))}  # rel. SE of an SD from 3 values ~ 50%


def analyse(df, tasks, n_boot, rng, label):
    out, lines = {}, [f"## {label}", ""]
    for task in tasks:
        A = df[(df.arm == "A") & (df.task == task)].sort_values("seed").success_rate.to_numpy(float)
        n_roll = int(df[(df.arm == "A") & (df.task == task)].n_rollouts.iloc[0])
        Bt, Ct = table(df, "B", task, B_KEYS), table(df, "C", task, C_KEYS)
        T = {"corrected_ratio": corrected_ratio(Bt, A, n_roll, n_boot, rng),
             "f_test_B": vc.f_test_p(Bt), "f_test_C": vc.f_test_p(Ct),
             "better_vs_rest": better_vs_rest(Bt), "resolvability": resolvability(A),
             "sd_p_B": float(vc.mom_two_way(Bt)["sd_p"]), "sd_p_C": float(vc.mom_two_way(Ct)["sd_p"]),
             "range_A": float(A.max() - A.min()),
             "range_B_partition_means": float(np.ptp(Bt.mean(axis=1))), "range_C_partition_means": float(np.ptp(Ct.mean(axis=1)))}
        out[task] = T
        cr, bv, rs = T["corrected_ratio"], T["better_vs_rest"], T["resolvability"]
        pt = "undefined (Arm A spread at or below the binomial floor)" if cr["point"] is None else f"{cr['point']:.2f}"
        hi = "unbounded" if cr["ci_high"] is None or cr["frac_denominator_at_or_below_floor"] > 0.025 else f"{cr['ci_high']:.2f}"
        lo = "undefined" if cr["ci_low"] is None else f"{cr['ci_low']:.2f}"
        lines += [f"### {task}",
                  f"- (i) Corrected-denominator ratio: {pt} [{lo}, {hi}]; "
                  f"{cr['frac_denominator_at_or_below_floor']*100:.0f}% of resamples have the Arm A spread at or below the binomial floor",
                  f"- (ii) F-test partition component: Arm B p = {T['f_test_B']['p_partition']:.4f}; Arm C p = {T['f_test_C']['p_partition']:.4f}",
                  f"- (iii) Drop-a-better-operator mean {bv['mean_drop_better']:.3f} vs rest {bv['mean_rest']:.3f} "
                  f"(difference {bv['diff']:+.3f}; exact permutation p = {bv['perm_p_one_sided']:.3f}, floor 0.067)",
                  f"- (iv) Resolvability of a 3-seed comparison: seed-level SD {rs['sd_seed_level']:.3f}; smallest resolvable "
                  f"difference between two 3-seed means {rs['mdd_normal_95']*100:.1f} points (normal) / {rs['mdd_t4_95']*100:.1f} points (t, 4 df)",
                  f"- Ranges: Arm A seeds {T['range_A']*100:.0f} points; Arm B partition means {T['range_B_partition_means']*100:.0f}; "
                  f"Arm C partition means {T['range_C_partition_means']*100:.0f}", ""]
    return out, lines


def main():
    rng = np.random.default_rng(0)
    R, lines = {}, ["# Exploratory and descriptive analyses (NOT preregistered tests)", "",
                    "Declared exploratory in PREREGISTRATION.md Amendment 2. Auto-generated by analysis/exploratory.py.", ""]
    s1 = pd.read_csv(os.path.join(REPO, "results", "runs.csv")); s1["seed"] = s1.seed.astype(int)
    R["stage1"], l1 = analyse(s1, ["lift", "can", "square"], 2000, rng, "Stage 1 (50 rollouts per checkpoint)")
    lines += l1
    p2 = os.path.join(REPO, "results", "runs_stage2.csv")
    if os.path.exists(p2):
        s2 = pd.read_csv(p2); s2["seed"] = s2.seed.astype(int)
        if len(s2) == 46:
            R["stage2"], l2 = analyse(s2, ["square"], 2000, rng, "Stage 2 (Square, 500 rollouts per checkpoint)")
            lines += l2
            # (v) agreement between stages on the same 46 checkpoints
            m = s2.merge(s1[["run_id", "success_rate"]], on="run_id", suffixes=("_500", "_50"))
            d = m.success_rate_50 - m.success_rate_500
            pb = m.success_rate_500
            expected_sd = float(np.sqrt(np.mean(pb * (1 - pb) * (1 / 50 + 1 / 500))))
            R["stage_agreement"] = {"n": int(len(m)), "pearson_r": float(np.corrcoef(m.success_rate_50, m.success_rate_500)[0, 1]),
                                    "mean_diff_50_minus_500": float(d.mean()), "sd_diff": float(d.std(ddof=1)),
                                    "expected_sd_diff_binomial": expected_sd, "max_abs_diff": float(d.abs().max())}
            g = R["stage_agreement"]
            lines += ["## (v) Stage 1 versus Stage 2 on the same 46 Square checkpoints", "",
                      f"- Pearson r between 50-rollout and 500-rollout success rates: {g['pearson_r']:.2f}",
                      f"- 50-rollout minus 500-rollout: mean {g['mean_diff_50_minus_500']:+.3f}, SD {g['sd_diff']:.3f} "
                      f"(binomial expectation {g['expected_sd_diff_binomial']:.3f}), max |diff| {g['max_abs_diff']:.2f}", ""]
    with open(os.path.join(REPO, "results", "exploratory.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    with open(os.path.join(REPO, "results", "exploratory_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
