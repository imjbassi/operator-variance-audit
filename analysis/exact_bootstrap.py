"""Exact enumeration of the preregistered bootstraps (no Monte Carlo error).

The preregistered procedures (6.4, 6.5) resample 6 partitions (and 10 Arm A seeds) with
replacement and use 2,000 random resamples. Because the number of distinct resamples is small
(462 multisets of 6 rows; 92,378 multisets of 10 seeds), the full bootstrap distribution can be
enumerated exactly with multinomial weights. This removes the dependence of a borderline
decision on the bootstrap random seed. It is the same statistic and the same resampling scheme;
the 2,000-resample numbers in analysis*.json remain the preregistered ones and are reported
alongside.

Outputs: results/exact_bootstrap.json and results/exact_bootstrap_summary.md
Usage:   python analysis/exact_bootstrap.py
"""
import itertools, json, math, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OPS = ["better_operator_1", "better_operator_2", "okay_operator_1",
       "okay_operator_2", "worse_operator_1", "worse_operator_2"]
B_KEYS = [f"B_drop_{op}" for op in OPS]
C_KEYS = [f"C_rand_{i}" for i in range(1, 7)]


def multisets(n):
    """All count vectors (c_1..c_n) with sum n, and their multinomial bootstrap probabilities."""
    counts, probs = [], []
    for comb in itertools.combinations_with_replacement(range(n), n):
        c = np.bincount(comb, minlength=n)
        w = math.factorial(n)
        for k in c:
            w //= math.factorial(int(k))
        counts.append(c); probs.append(w / n ** n)
    return np.array(counts), np.array(probs)


def wquantile(vals, w, qs):
    o = np.argsort(vals, kind="stable")
    v, cw = vals[o], np.cumsum(w[o])
    return [float(v[min(np.searchsorted(cw, q, side="left"), v.size - 1)]) for q in qs]


def table(df, arm, task, keys):
    sub = df[(df.arm == arm) & (df.task == task)]
    return np.array([[float(sub[(sub.partition_key == k) & (sub.seed == s)].success_rate.iloc[0])
                      for s in (1, 2, 3)] for k in keys])


def analyse(df, task, C6, P6, C10, P10):
    A = df[(df.arm == "A") & (df.task == task)].sort_values("seed").success_rate.to_numpy(float)
    Bt, Ct = table(df, "B", task, B_KEYS), table(df, "C", task, C_KEYS)
    s2B = np.array([vc.mom_two_way(np.repeat(Bt, c, axis=0))["s2_p"] for c in C6])
    s2C = np.array([vc.mom_two_way(np.repeat(Ct, c, axis=0))["s2_p"] for c in C6])
    # ---- D = s2_p(B) - s2_p(C): 462 x 462 outcomes
    D = (s2B[:, None] - s2C[None, :]).ravel()
    W = (P6[:, None] * P6[None, :]).ravel()
    dlo, dhi = wquantile(D, W, [0.025, 0.975])
    out = {"D": {"point": float(vc.mom_two_way(Bt)["s2_p"] - vc.mom_two_way(Ct)["s2_p"]),
                 "ci_low": dlo, "ci_high": dhi, "p_D_le_0": float(W[D <= 0].sum()),
                 "p_D_lt_0": float(W[D < 0].sum()), "n_outcomes": int(D.size)}}
    # ---- R = sd_p(B) / sd(A): 462 x 92378 outcomes
    n = A.size
    m1 = C10 @ A / n
    m2 = C10 @ (A ** 2) / n
    sdA = np.sqrt(np.maximum((m2 - m1 ** 2) * n / (n - 1), 0.0))
    sdB = np.sqrt(s2B)
    o = np.argsort(sdA); sdA_s, PA_s = sdA[o], P10[o]; cumA = np.cumsum(PA_s)
    # P(R <= 1) = P(sdB <= sdA) = sum_b P6[b] * P(sdA >= sdB[b]); sdA == 0 with sdB > 0 gives R = inf (> 1)
    p_le_1 = 0.0
    for b, pb in zip(sdB, P6):
        idx = np.searchsorted(sdA_s, b, side="left")
        p_ge = 1.0 - (cumA[idx - 1] if idx > 0 else 0.0)
        if b == 0:  # R = 0/sdA <= 1 whenever sdA > 0; 0/0 is undefined and, as in vc.bootstrap_ratio, not counted
            p_ge = float(PA_s[sdA_s > 0].sum())
        p_le_1 += pb * p_ge
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (sdB[:, None] / sdA[None, :])
    R[np.isnan(R)] = 0.0
    Wr = (P6[:, None] * P10[None, :]).ravel()
    rlo, rhi = wquantile(R.ravel(), Wr, [0.025, 0.975])
    sA = A.std(ddof=1)
    out["R"] = {"point": float(vc.mom_two_way(Bt)["sd_p"] / sA) if sA > 0 else None, "ci_low": rlo,
                "ci_high": rhi if np.isfinite(rhi) else None, "p_R_le_1": float(p_le_1), "n_outcomes": int(R.size)}
    return out


def main():
    C6, P6 = multisets(6)
    C10, P10 = multisets(10)
    assert abs(P6.sum() - 1) < 1e-12 and abs(P10.sum() - 1) < 1e-9
    R, lines = {}, ["# Exact enumeration of the preregistered bootstraps", "",
                    "Same statistics and resampling scheme as Sections 6.4 and 6.5; all 462 x 462 (D) and",
                    "462 x 92,378 (R) resamples enumerated with multinomial weights, so there is no Monte Carlo error.", "",
                    "| stage | task | D [95%] | p(D<=0) | R [95%] | p(R<=1) |", "|---|---|---|---|---|---|"]
    stages = [("stage1", "runs.csv", ["lift", "can", "square"]), ("stage2", "runs_stage2.csv", ["square"])]
    for stage, fn, tasks in stages:
        p = os.path.join(REPO, "results", fn)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p); df["seed"] = df.seed.astype(int)
        R[stage] = {}
        for task in tasks:
            r = analyse(df, task, C6, P6, C10, P10)
            R[stage][task] = r
            d, q = r["D"], r["R"]
            hi = "inf" if q["ci_high"] is None else f"{q['ci_high']:.2f}"
            pt = "n/a" if q["point"] is None else f"{q['point']:.2f}"
            lines.append(f"| {stage} | {task} | {d['point']:.5f} [{d['ci_low']:.5f}, {d['ci_high']:.5f}] | {d['p_D_le_0']:.4f} | "
                         f"{pt} [{q['ci_low']:.2f}, {hi}] | {q['p_R_le_1']:.4f} |")
    # Holm within each preregistered family, using the exact p-values
    lines += ["", "## Holm-Bonferroni with exact p-values", ""]
    for stage, fam in (("stage1", ["lift", "can", "square"]), ("stage2", ["square"])):
        if stage not in R:
            continue
        labels, ps = [], []
        for t in fam:
            labels += [f"{t}:R>1", f"{t}:D>0"]; ps += [R[stage][t]["R"]["p_R_le_1"], R[stage][t]["D"]["p_D_le_0"]]
        adj = vc.holm(ps)
        R[stage]["holm"] = [{"contrast": l, "p_raw": float(p), "p_holm": float(a)} for l, p, a in zip(labels, ps, adj)]
        lines += [f"**{stage}**", "", "| contrast | p raw | p Holm |", "|---|---|---|"]
        lines += [f"| {l} | {p:.4f} | {a:.4f} |" for l, p, a in zip(labels, ps, adj)] + [""]
    with open(os.path.join(REPO, "results", "exact_bootstrap.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    with open(os.path.join(REPO, "results", "exact_bootstrap_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
