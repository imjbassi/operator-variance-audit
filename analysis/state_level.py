"""POST HOC state-level analyses added after informal review (PREREGISTRATION.md Amendments 4 and 5).
Not preregistered tests. Everything is computed from the committed per-rollout CSVs and JSON.

  (a) Seed effect on shared initial states. Evaluation on stored states is repeatable, so the
      Arm A checkpoints share states and their outcomes are not independent Bernoulli draws.
      For the checkpoints x states success matrix we report:
        - binomial floor        sqrt(p(1-p)/n)               (sampling error vs fresh states)
        - per-state floor       sqrt(mean_i q_i(1-q_i)/n)    (null of no seed effect on shared states;
                                q_i(1-q_i) bias-corrected by m/(m-1), m = number of checkpoints)
        - Cochran's Q           exact paired test that all checkpoints share one success rate
        - chi-square of the seed SD against the binomial floor, for comparison with the draft
  (b) Resolvability of two three-seed means: paired (both methods on the same stored states) and
      unpaired (independent draws of initial conditions, as with unseeded environment resets).
      The paired run-to-run SD is max(s_A, per-state floor) (Amendment 5): s_A is itself a noisy
      ten-seed estimate, and where it falls below the floor (Can) it understates the noise that
      shared-state evaluation alone produces. The raw-s_A threshold is kept as mdd_paired_raw_*.
  (c) Tier pairings: all 15 pairings of the six Arm B partitions, ranked by between-pair share.
  (d) Monte Carlo error of the preregistered 2,000-resample p-value for the Stage 2 contrast D.
  (e) Why power at a fixed ratio falls from Stage 1 to Stage 2.

Outputs: results/state_level.json and results/state_level_summary.md
Usage:   python analysis/state_level.py
"""
import itertools, json, os
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
OPS = ["better_operator_1", "better_operator_2", "okay_operator_1",
       "okay_operator_2", "worse_operator_1", "worse_operator_2"]
N_POWER_SIM = 200   # as used by analysis/run_analysis.py (default --n_power_sim)
N_BOOT = 2000       # preregistered number of bootstrap resamples


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def success_matrix(runs, rolls, task, arm="A"):
    ids = runs[(runs.task == task) & (runs.arm == arm)].sort_values("seed").run_id.tolist()
    sub = rolls[rolls.run_id.isin(ids)]
    return sub.pivot(index="run_id", columns="rollout", values="success").loc[ids].to_numpy(float)


def cochran_q(M):
    m = M.shape[0]
    C, Rw, N = M.sum(axis=1), M.sum(axis=0), M.sum()
    den = m * N - (Rw ** 2).sum()
    Q = (m - 1) * (m * (C ** 2).sum() - N ** 2) / den
    return float(Q), int(m - 1), float(stats.chi2.sf(Q, m - 1))


def seed_effect(M):
    m, n = M.shape
    y = M.mean(axis=1)
    p, s = float(y.mean()), float(y.std(ddof=1))
    q = M.mean(axis=0)
    binom2 = p * (1 - p) / n
    state2 = float(np.mean(q * (1 - q)) * m / (m - 1) / n)
    ckpt2 = max(s ** 2 - state2, 0.0)
    chi = (m - 1) * s ** 2 / binom2
    Q, df, pQ = cochran_q(M)
    z, t4 = 1.96, float(stats.t.ppf(0.975, 4))
    unp = np.sqrt(ckpt2 + binom2)
    s_pair = max(s, float(np.sqrt(state2)))   # Amendment 5
    return {"n_checkpoints": m, "n_states": n, "mean": p, "seed_sd": s,
            "binomial_floor": float(np.sqrt(binom2)), "per_state_floor": float(np.sqrt(state2)),
            "seed_sd_above_per_state_floor": float(np.sqrt(ckpt2)),
            "chi2_vs_binomial": float(chi), "chi2_vs_binomial_p": float(stats.chi2.sf(chi, m - 1)),
            "cochran_q": Q, "cochran_df": df, "cochran_p": pQ,
            "states_always_solved": int((q == 1).sum()), "states_never_solved": int((q == 0).sum()),
            "states_discriminating": int(((q > 0) & (q < 1)).sum()),
            "paired_sd_used": s_pair, "paired_sd_is_floor": bool(s_pair > s),
            "mdd_paired_normal": float(z * s_pair * np.sqrt(2 / 3)), "mdd_paired_t4": float(t4 * s_pair * np.sqrt(2 / 3)),
            "mdd_paired_raw_normal": float(z * s * np.sqrt(2 / 3)), "mdd_paired_raw_t4": float(t4 * s * np.sqrt(2 / 3)),
            "mdd_unpaired_normal": float(z * unp * np.sqrt(2 / 3)), "mdd_unpaired_t4": float(t4 * unp * np.sqrt(2 / 3))}


def pairings(row_means):
    y = np.asarray(row_means, float)
    gm, sst = y.mean(), float(((y - y.mean()) ** 2).sum())
    seen, out = set(), []
    for perm in itertools.permutations(range(6)):
        pr = tuple(sorted(tuple(sorted((perm[2 * k], perm[2 * k + 1]))) for k in range(3)))
        if pr in seen:
            continue
        seen.add(pr)
        out.append((float(sum(2 * (y[list(p)].mean() - gm) ** 2 for p in pr) / sst), pr))
    out.sort(key=lambda r: -r[0])
    obs_pr = ((0, 1), (2, 3), (4, 5))
    obs = [s for s, pr in out if pr == obs_pr][0]
    at_least = [(s, pr) for s, pr in out if s >= obs - 1e-12]
    return {"observed_share": obs, "n_pairings": len(out), "n_at_least_observed": len(at_least),
            "rank_of_tier_pairing": 1 + [pr for _, pr in out].index(obs_pr),
            "all_at_least_observed_pair_the_two_better": all((0, 1) in pr for _, pr in at_least),
            "top": [{"share": s, "pairs": [[OPS[a], OPS[b]] for a, b in pr]} for s, pr in out[:5]]}


def main():
    s1, r1 = pd.read_csv(os.path.join(RES, "runs.csv")), pd.read_csv(os.path.join(RES, "rollouts.csv"))
    s2, r2 = pd.read_csv(os.path.join(RES, "runs_stage2.csv")), pd.read_csv(os.path.join(RES, "rollouts_stage2.csv"))
    a1, a2, ex = load("analysis.json"), load("analysis_stage2.json"), load("exact_bootstrap.json")
    R = {"seed_effect": {}, "tier_pairings": {}, "note": "POST HOC (Amendment 4); not preregistered tests"}
    for t in ("lift", "can", "square"):
        R["seed_effect"][f"stage1_{t}"] = seed_effect(success_matrix(s1, r1, t))
    R["seed_effect"]["stage2_square"] = seed_effect(success_matrix(s2, r2, "square"))
    for lab, a in (("stage1_square", a1), ("stage2_square", a2)):
        pm = a["tasks"]["square"]["B"]["partition_means"]
        R["tier_pairings"][lab] = pairings([pm[o] for o in OPS])
    # (d) Monte Carlo error of the preregistered p for D at Stage 2
    d = a2["tasks"]["square"]["B_minus_C"]
    hm = [e for e in a2["primary_family"] if e["contrast"] == "square:D>0"][0]
    he = [e for e in ex["stage2"]["holm"] if e["contrast"] == "square:D>0"][0]
    p = d["p_D_le_0"]; se = float(np.sqrt(p * (1 - p) / N_BOOT))
    R["stage2_D"] = {"p_raw_resampled": p, "p_holm_resampled": hm["p_holm"], "mc_se_raw": se,
                     "mc_se_holm": 2 * se, "ci_resampled": [d["ci_low"], d["ci_high"]],
                     "ci_excludes_0_resampled": bool(d["ci_low"] > 0),
                     "p_raw_exact": ex["stage2"]["square"]["D"]["p_D_le_0"], "p_holm_exact": he["p_holm"],
                     "ci_exact": [ex["stage2"]["square"]["D"]["ci_low"], ex["stage2"]["square"]["D"]["ci_high"]]}
    # (e) power at a fixed ratio across stages
    R["power"] = {}
    for lab, T in (("stage1_square", a1["tasks"]["square"]), ("stage2_square", a2["tasks"]["square"])):
        P = {round(v["true_R"], 1): v["power"] for v in T["power_headline_procedure"].values()}
        sA, se_ = T["A"]["sd_seed"], T["B"]["mom"]["sd_e"]
        R["power"][lab] = {"seed_sd_A": sA, "resid_sd_B": se_, "sd_p_at_R2": 2 * sA, "sd_p_over_resid_at_R2": 2 * sA / se_,
                           "power_R1": P[1.0], "power_R2": P[2.0], "power_R3": P[3.0],
                           "mc_se_power_R2": float(np.sqrt(P[2.0] * (1 - P[2.0]) / N_POWER_SIM))}
    R["power"]["false_positive_max_R1"] = max(T["power_headline_procedure"][k]["power"]
                                              for T in list(a1["tasks"].values()) + [a2["tasks"]["square"]]
                                              for k in T["power_headline_procedure"]
                                              if round(T["power_headline_procedure"][k]["true_R"], 1) == 1.0)

    with open(os.path.join(RES, "state_level.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    L = ["# Post hoc state-level analyses (Amendment 4; NOT preregistered tests)", "",
         "## Seed effect on shared initial states (Arm A)", "",
         "| case | mean | seed SD | binomial floor | per-state floor | chi2 vs binomial p | Cochran's Q (df) | Cochran p | discriminating states |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, v in R["seed_effect"].items():
        L.append(f"| {k} | {v['mean']:.3f} | {v['seed_sd']:.3f} | {v['binomial_floor']:.3f} | {v['per_state_floor']:.3f} | "
                 f"{v['chi2_vs_binomial_p']:.3f} | {v['cochran_q']:.2f} ({v['cochran_df']}) | {v['cochran_p']:.4f} | "
                 f"{v['states_discriminating']} of {v['n_states']} |")
    L += ["", "## Smallest resolvable difference between two three-seed means (points, 95%)", "",
          "Paired uses max(seed SD, per-state floor) (Amendment 5); the raw-seed-SD value is shown for reference.", "",
          "| case | paired SD used | paired, normal | paired, t(4) | paired with raw seed SD | unpaired, normal | unpaired, t(4) |",
          "|---|---|---|---|---|---|---|"]
    for k, v in R["seed_effect"].items():
        L.append(f"| {k} | {v['paired_sd_used']:.3f}{' (floor)' if v['paired_sd_is_floor'] else ''} | "
                 f"{100*v['mdd_paired_normal']:.1f} | {100*v['mdd_paired_t4']:.1f} | {100*v['mdd_paired_raw_normal']:.1f} | "
                 f"{100*v['mdd_unpaired_normal']:.1f} | {100*v['mdd_unpaired_t4']:.1f} |")
    L += ["", "## Tier pairings (Arm B partition means, Square)", ""]
    for k, v in R["tier_pairings"].items():
        L.append(f"- {k}: observed tier share {v['observed_share']:.3f}; {v['n_at_least_observed']} of {v['n_pairings']} pairings "
                 f"reach it; tier pairing ranks {v['rank_of_tier_pairing']}; all of those pair the two better operators: "
                 f"{v['all_at_least_observed_pair_the_two_better']}")
    g = R["stage2_D"]
    L += ["", "## Stage 2 contrast D: Monte Carlo error of the preregistered computation", "",
          f"- resampled (2000): raw p {g['p_raw_resampled']:.4f} (MC SE {g['mc_se_raw']:.4f}), Holm p {g['p_holm_resampled']:.4f} "
          f"(MC SE {g['mc_se_holm']:.4f}), interval excludes 0: {g['ci_excludes_0_resampled']}",
          f"- exact enumeration: raw p {g['p_raw_exact']:.4f}, Holm p {g['p_holm_exact']:.4f}", "",
          "## Power at a fixed ratio across stages (Square)", ""]
    for k in ("stage1_square", "stage2_square"):
        v = R["power"][k]
        L.append(f"- {k}: seed SD {v['seed_sd_A']:.3f}, residual SD {v['resid_sd_B']:.3f}; at R = 2 the partition SD is "
                 f"{v['sd_p_at_R2']:.3f}, {v['sd_p_over_resid_at_R2']:.2f} x the residual; power {v['power_R2']:.2f} "
                 f"(MC SE {v['mc_se_power_R2']:.3f}); power at R = 1: {v['power_R1']:.2f}")
    fp_k = round(R["power"]["false_positive_max_R1"] * N_POWER_SIM)
    R["power"]["false_positive_max_count"] = int(fp_k)
    R["power"]["false_positive_exact_upper95"] = float(stats.beta.ppf(0.975, fp_k + 1, N_POWER_SIM - fp_k))
    L.append(f"- false positives at R = 1: at most {fp_k} of {N_POWER_SIM} simulations per task and stage "
             f"(exact 95% upper bound {100*R['power']['false_positive_exact_upper95']:.1f}%); too few runs to show a rate below the nominal 2.5%")
    with open(os.path.join(RES, "state_level.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    with open(os.path.join(RES, "state_level_summary.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
