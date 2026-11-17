"""Post hoc analyses added after a third review (PREREGISTRATION.md Amendment 6). No new data;
nothing here is a test of a preregistered claim.

  (a) 80%-power resolvability thresholds: Eq. (1) of the paper is the difference detected about
      half the time; the 80%-power threshold is (z_.975 + z_.80) / z_.975 times larger.
  (b) Measurement-error attenuation: the correlation between 50- and 500-rollout success rates
      that binomial rollout noise alone predicts, against the observed one.
  (c) Jackknife scaling of the leave-one-operator-out spread to a fresh pool of six operators.
  (d) Upward bias of the maximum of m independent n-rollout evaluations of one policy
      (winner's curse in best-checkpoint reporting).

Inputs:  results/analysis.json, results/analysis_stage2.json, results/state_level.json,
         results/runs_stage2.csv
Outputs: results/posthoc_review.json, results/posthoc_review_summary.md
Usage:   python analysis/posthoc_review.py
"""
import json, os
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
N_BOOT, BOOT_SEED = 10000, 20261116


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def power80(sl):
    f = float((stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) / stats.norm.ppf(0.975))
    out = {"factor": f}
    for key, v in sl["seed_effect"].items():
        out[key] = {"mdd_paired_power80": f * v["mdd_paired_normal"],
                    "mdd_unpaired_power80": f * v["mdd_unpaired_normal"],
                    "mdd_paired_power50": v["mdd_paired_normal"]}
    return out


def predicted_r(x500, n_lo=50, n_hi=500):
    """Correlation between an n_lo- and an n_hi-rollout success rate of the same checkpoints if
    they differ only by independent binomial rollout noise. The variance of true success rates
    is the n_hi-rollout variance minus its binomial noise."""
    pq = float(np.mean(x500 * (1 - x500)))
    e_lo, e_hi = pq / n_lo, pq / n_hi
    v_hi = float(np.var(x500, ddof=1))
    v_true = max(v_hi - e_hi, 0.0)
    r = v_true / np.sqrt((v_true + e_lo) * (v_true + e_hi)) if v_true > 0 else 0.0
    return {"var_true": v_true, "sd_true": float(np.sqrt(v_true)), "noise_var_lo": e_lo, "noise_var_hi": e_hi,
            "reliability_lo": v_true / (v_true + e_lo), "reliability_hi": v_true / (v_true + e_hi),
            "predicted_r": float(r)}


def attenuation(s2):
    x50, x500 = s2.stage1_success_rate.values, s2.success_rate.values
    out = predicted_r(x500)
    out["n"] = int(len(s2))
    out["observed_r"] = float(np.corrcoef(x50, x500)[0, 1])
    out["observed_var_50"] = float(np.var(x50, ddof=1))
    out["model_var_50"] = out["var_true"] + out["noise_var_lo"]
    out["disattenuated_r"] = out["observed_r"] / out["predicted_r"]
    rng = np.random.default_rng(BOOT_SEED)
    pr, ob = [], []
    for _ in range(N_BOOT):                      # checkpoint bootstrap
        i = rng.integers(0, len(s2), len(s2))
        pr.append(predicted_r(x500[i])["predicted_r"]); ob.append(np.corrcoef(x50[i], x500[i])[0, 1])
    out["predicted_r_ci"] = [float(q) for q in np.percentile(pr, [2.5, 97.5])]
    out["observed_r_ci"] = [float(q) for q in np.percentile(ob, [2.5, 97.5])]
    out["n_boot"] = N_BOOT
    return out


def jackknife(tasks):
    """Delete-one jackknife: SE^2 = (n-1)/n * sum_i (theta_(i) - mean)^2 = ((n-1)/sqrt(n))^2 * SD^2,
    with theta_(i) the leave-one-operator-out value. Applied to the residual-corrected partition
    SD (and, for reference, to the raw SD of the partition means, which still contains seed and
    rollout noise)."""
    out = {}
    for key, T in tasks.items():
        B = T["B"]
        pm = np.array(list(B["partition_means"].values()))
        n = len(pm)
        f = (n - 1) / np.sqrt(n)
        out[key] = {"n_operators": n, "factor": float(f), "sd_p_B": B["mom"]["sd_p"],
                    "jackknife_sd_fresh_pool": float(f * B["mom"]["sd_p"]),
                    "raw_sd_partition_means": float(pm.std(ddof=1)),
                    "jackknife_sd_fresh_pool_raw": float(f * pm.std(ddof=1))}
    return out


def max_bias(p, n, m):
    """E[max of m independent Binomial(n, p) / n] - p, exact."""
    k = np.arange(n + 1)
    cdf_max = stats.binom.cdf(k, n, p) ** m
    pmf_max = np.diff(np.concatenate([[0.0], cdf_max]))
    return float((pmf_max * k).sum() / n - p)


def main():
    a1, a2, sl = load("analysis.json"), load("analysis_stage2.json"), load("state_level.json")
    s2 = pd.read_csv(os.path.join(RES, "runs_stage2.csv"))
    tasks = {f"stage1_{t}": a1["tasks"][t] for t in ("lift", "can", "square")}
    tasks["stage2_square"] = a2["tasks"]["square"]
    p_sq = a1["tasks"]["square"]["A"]["mean"]
    out = {"note": "Post hoc (Amendment 6). Not tests of the preregistered claims.",
           "power80": power80(sl), "attenuation": attenuation(s2), "jackknife": jackknife(tasks),
           "max_selection": {"p": p_sq, "n_rollouts": 50, "binomial_sd": float(np.sqrt(p_sq * (1 - p_sq) / 50)),
                             "bias_max_of_40": max_bias(p_sq, 50, 40), "bias_max_of_8": max_bias(p_sq, 50, 8)}}
    with open(os.path.join(RES, "posthoc_review.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    P, a, J, mx = out["power80"], out["attenuation"], out["jackknife"], out["max_selection"]
    L = ["# Post hoc analyses after a third review (NOT preregistered tests)", "",
         "Declared in PREREGISTRATION.md Amendment 6. Auto-generated by analysis/posthoc_review.py.", "",
         "## (a) Resolvability at 80% power", "",
         f"Eq. (1) is the difference detected with about 50% power. Factor for 80% power: {P['factor']:.3f}.", "",
         "| task, stage | paired, 50% power | paired, 80% power | unpaired, 80% power |", "|---|---|---|---|"]
    for key in ("stage1_lift", "stage1_can", "stage1_square", "stage2_square"):
        v = P[key]
        L.append(f"| {key} | {100 * v['mdd_paired_power50']:.1f} | {100 * v['mdd_paired_power80']:.1f} | {100 * v['mdd_unpaired_power80']:.1f} |")
    L += ["", "## (b) Attenuation of the 50- versus 500-rollout correlation (46 Square checkpoints)", "",
          f"- SD of true success rates across checkpoints (500-rollout variance minus binomial noise): {a['sd_true']:.3f}",
          f"- Reliability of a success rate: {a['reliability_lo']:.2f} at 50 rollouts, {a['reliability_hi']:.2f} at 500",
          f"- Correlation predicted by rollout noise alone: {a['predicted_r']:.2f} "
          f"[{a['predicted_r_ci'][0]:.2f}, {a['predicted_r_ci'][1]:.2f}] (checkpoint bootstrap)",
          f"- Observed: {a['observed_r']:.2f} [{a['observed_r_ci'][0]:.2f}, {a['observed_r_ci'][1]:.2f}]; "
          f"disattenuated {a['disattenuated_r']:.2f}",
          f"- 50-rollout variance across checkpoints: observed {a['observed_var_50']:.5f}, model {a['model_var_50']:.5f}", "",
          "## (c) Jackknife scaling of the leave-one-operator-out spread", "",
          "Leave-one-out partitions share 200 of 250 demonstrations, so their spread understates the spread across",
          "fresh operator pools. Delete-one jackknife factor (n-1)/sqrt(n) = "
          f"{J['stage1_square']['factor']:.2f} for n = 6. Extrapolation from six points; exploratory.", "",
          "| task, stage | Arm B partition SD | jackknife SD, fresh pool of six | same from raw partition means |", "|---|---|---|---|"]
    for key, v in J.items():
        L.append(f"| {key} | {v['sd_p_B']:.3f} | {v['jackknife_sd_fresh_pool']:.3f} | {v['jackknife_sd_fresh_pool_raw']:.3f} |")
    L += ["", "## (d) Bias of the maximum of noisy evaluations", "",
          f"At p = {mx['p']:.2f} with 50 rollouts (binomial SD {mx['binomial_sd']:.3f}), the expected maximum of independent",
          f"evaluations of one policy exceeds p by {100 * mx['bias_max_of_40']:.1f} points for 40 evaluations and "
          f"{100 * mx['bias_max_of_8']:.1f} for 8.", ""]
    with open(os.path.join(RES, "posthoc_review_summary.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
