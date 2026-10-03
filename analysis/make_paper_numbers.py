"""Generate every number and table used in the paper from the committed result files.

  paper/numbers.tex            one \\newcommand per reported quantity (prose uses only these macros)
  paper/tables/tab_components.tex
  paper/tables/tab_square_partitions.tex
  paper/tables/tab_resolvability.tex

Inputs: results/analysis.json (Stage 1), results/analysis_stage2.json (Stage 2, Square),
        results/exact_bootstrap.json, results/exploratory.json
Usage:  python analysis/make_paper_numbers.py
"""
import json, os
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(REPO, "results")
OUT = os.path.join(REPO, "paper")
TN = {"lift": "Lift", "can": "Can", "square": "Sq"}
OPS = ["better_operator_1", "better_operator_2", "okay_operator_1",
       "okay_operator_2", "worse_operator_1", "worse_operator_2"]
OP_TEX = ["Better 1", "Better 2", "Okay 1", "Okay 2", "Worse 1", "Worse 2"]


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def pts(x, d=0): return f"{100 * x:.{d}f}"
def pv(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
def pv4(p): return f"{p:.4f}"   # D contrast: four decimals, because Stage 2 sits at the 0.05 boundary
def sd(m, key, bkey): return f3(m[key]) + ("$^{\\dagger}$" if m.get(bkey) else "")


def holm_lookup(fam, contrast):
    for e in fam:
        if e["contrast"] == contrast:
            return e
    raise KeyError(contrast)


def task_macros(M, pre, T, fam, ex, exfam, expl, task):
    A, B, C = T["A"], T["B"], T["C"]
    n = A.get("n_rollouts", 50)
    M[pre + "AMean"] = f3(A["mean"]); M[pre + "ASD"] = f3(A["sd_seed"])
    M[pre + "ASDlo"] = f3(A["sd_seed_ci"][0]); M[pre + "ASDhi"] = f3(A["sd_seed_ci"][1])
    M[pre + "AMin"] = f2(A["min"]); M[pre + "AMax"] = f2(A["max"]); M[pre + "ARange"] = pts(A["max"] - A["min"])
    M[pre + "Binom"] = f3(A["binomial_sd_50_rollouts"]); M[pre + "ASDcorr"] = f3(A["sd_seed_minus_binomial"])
    for arm, X in (("B", B), ("C", C)):
        M[pre + arm + "SDp"] = f3(X["mom"]["sd_p"]); M[pre + arm + "SDs"] = f3(X["mom"]["sd_s"])
        M[pre + arm + "SDe"] = f3(X["mom"]["sd_e"]); M[pre + arm + "Fp"] = pv(X["f_test"]["p_partition"])
        M[pre + arm + "RemlSDp"] = f3(X["reml"]["sd_p"])
        rm = np.mean(X["table"], axis=1)
        M[pre + arm + "Range"] = pts(rm.max() - rm.min())
        M[pre + arm + "MeanLo"] = f2(rm.min()); M[pre + arm + "MeanHi"] = f2(rm.max())
    h, d = T["headline_ratio_mom"], T["B_minus_C"]
    hr = T["headline_ratio_reml"]
    M[pre + "R"] = f2(h["point"]); M[pre + "Rlo"] = f2(h["ci_low"]); M[pre + "Rhi"] = f2(h["ci_high"])
    M[pre + "Rp"] = pv(h["p_R_le_1"]); M[pre + "RpHolm"] = pv(holm_lookup(fam, f"{task}:R>1")["p_holm"])
    M[pre + "RReml"] = f2(hr["point"])
    M[pre + "D"] = f"{1e3 * d['point']:.2f}"; M[pre + "Dlo"] = f"{1e3 * d['ci_low']:.2f}"; M[pre + "Dhi"] = f"{1e3 * d['ci_high']:.2f}"
    M[pre + "Dp"] = pv4(d["p_D_le_0"]); M[pre + "DpHolm"] = pv4(holm_lookup(fam, f"{task}:D>0")["p_holm"])
    e = ex[task]
    M[pre + "RExLo"] = f2(e["R"]["ci_low"]); M[pre + "RExHi"] = f2(e["R"]["ci_high"]) if e["R"]["ci_high"] is not None else "\\infty"
    M[pre + "RExp"] = pv(e["R"]["p_R_le_1"]); M[pre + "RExpHolm"] = pv(holm_lookup(exfam, f"{task}:R>1")["p_holm"])
    M[pre + "DExLo"] = f"{1e3 * e['D']['ci_low']:.2f}"; M[pre + "DExHi"] = f"{1e3 * e['D']['ci_high']:.2f}"
    M[pre + "DExp"] = pv4(e["D"]["p_D_le_0"]); M[pre + "DExpHolm"] = pv4(holm_lookup(exfam, f"{task}:D>0")["p_holm"])
    P = list(T["power_headline_procedure"].values())
    byR = {round(p["true_R"], 1): p["power"] for p in P}
    for r, nm in ((2.0, "Two"), (3.0, "Three"), (4.0, "Four")):
        M[pre + "Pow" + nm] = pts(byR[r])
    tp, tw = B["tier_permutation"], B["tier_pairwise"]
    M[pre + "TierShare"] = f2(tp["observed_share"]); M[pre + "TierNull"] = f2(tp["null_mean_share"]); M[pre + "TierP"] = pv(tp["p_value"])
    M[pre + "TierWithin"] = f3(tw["within_tier_mean_absdiff"]); M[pre + "TierBetween"] = f3(tw["between_tier_mean_absdiff"])
    M[pre + "MinDetF"] = f3(B["min_detectable_sd_p_alpha05"])
    x = expl[task]
    M[pre + "MDD"] = pts(x["resolvability"]["mdd_normal_95"], 1); M[pre + "MDDt"] = pts(x["resolvability"]["mdd_t4_95"], 1)
    cr = x["corrected_ratio"]
    M[pre + "RCorr"] = "undefined" if cr["point"] is None else f2(cr["point"])
    M[pre + "RCorrLo"] = "undefined" if cr["ci_low"] is None else f2(cr["ci_low"])
    M[pre + "RCorrFloorFrac"] = pts(cr["frac_denominator_at_or_below_floor"])
    bv = x["better_vs_rest"]
    M[pre + "DropBetter"] = f3(bv["mean_drop_better"]); M[pre + "DropRest"] = f3(bv["mean_rest"])
    M[pre + "DropDiff"] = pts(abs(bv["diff"]), 1); M[pre + "DropP"] = pv(bv["perm_p_one_sided"])


def state_macros(M, pre, v):
    """Post hoc (Amendment 4) seed-effect and resolvability quantities from results/state_level.json."""
    M[pre + "StateFloor"] = f3(v["per_state_floor"]); M[pre + "CochQ"] = f"{v['cochran_q']:.1f}"
    M[pre + "CochDf"] = str(v["cochran_df"]); M[pre + "CochP"] = pv(v["cochran_p"])
    M[pre + "ChiB"] = f"{v['chi2_vs_binomial']:.1f}"; M[pre + "ChiBp"] = pv(v["chi2_vs_binomial_p"])
    M[pre + "Discrim"] = str(v["states_discriminating"]); M[pre + "NStates"] = str(v["n_states"])
    M[pre + "MDDu"] = pts(v["mdd_unpaired_normal"], 1); M[pre + "MDDut"] = pts(v["mdd_unpaired_t4"], 1)
    # Paired thresholds quoted in the text use max(s_A, per-state floor) (Amendment 5); this
    # overrides the raw-s_A values that task_macros() takes from exploratory.json.
    M[pre + "MDD"] = pts(v["mdd_paired_normal"], 1); M[pre + "MDDt"] = pts(v["mdd_paired_t4"], 1)
    M[pre + "MDDraw"] = pts(v["mdd_paired_raw_normal"], 1)
    M[pre + "SDck"] = f3(v["seed_sd_above_per_state_floor"])


def components_row(label, T, fam, ex, exfam, task):
    A, B, C = T["A"], T["B"], T["C"]
    h, d, e = T["headline_ratio_mom"], T["B_minus_C"], ex[task]
    return (f"{label} & {f3(A['mean'])} & {f3(A['sd_seed'])} [{f3(A['sd_seed_ci'][0])}, {f3(A['sd_seed_ci'][1])}] & "
            f"{f3(A['binomial_sd_50_rollouts'])} & "
            f"{sd(B['mom'], 'sd_p', 'boundary_p')} & {sd(B['mom'], 'sd_s', 'boundary_s')} & {f3(B['mom']['sd_e'])} & "
            f"{sd(C['mom'], 'sd_p', 'boundary_p')} & {sd(C['mom'], 'sd_s', 'boundary_s')} & {f3(C['mom']['sd_e'])} & "
            f"{f2(h['point'])} [{f2(h['ci_low'])}, {f2(h['ci_high'])}] & {pv(holm_lookup(fam, task + ':R>1')['p_holm'])} & "
            f"{1e3 * d['point']:.2f} [{1e3 * d['ci_low']:.2f}, {1e3 * d['ci_high']:.2f}] & "
            f"{pv4(d['p_D_le_0'])} / {pv4(e['D']['p_D_le_0'])} & "
            f"{pv4(holm_lookup(fam, task + ':D>0')['p_holm'])} / {pv4(holm_lookup(exfam, task + ':D>0')['p_holm'])} \\\\")


def main():
    a1, a2 = load("analysis.json"), load("analysis_stage2.json")
    ex, expl = load("exact_bootstrap.json"), load("exploratory.json")
    sl = load("state_level.json")
    M = {}
    for key, pre in (("stage1_square", "OneSq"), ("stage2_square", "TwoSq")):
        tp = sl["tier_pairings"][key]
        M[pre + "PairsReach"] = str(tp["n_at_least_observed"]); M[pre + "PairsRank"] = str(tp["rank_of_tier_pairing"])
        pw = sl["power"][key]
        M[pre + "PowRatio"] = f2(pw["sd_p_over_resid_at_R2"]); M[pre + "PowSE"] = pts(pw["mc_se_power_R2"], 1)
        M[pre + "PowSdp"] = f3(pw["sd_p_at_R2"])
    M["FPmax"] = pts(sl["power"]["false_positive_max_R1"], 1)
    M["FPmaxCount"] = str(sl["power"]["false_positive_max_count"])
    M["FPupper"] = pts(sl["power"]["false_positive_exact_upper95"], 1)
    g = sl["stage2_D"]
    M["TwoSqDmcSE"] = f"{g['mc_se_raw']:.4f}"; M["TwoSqDmcSEHolm"] = f"{g['mc_se_holm']:.4f}"
    for task in ("lift", "can", "square"):
        task_macros(M, "One" + TN[task], a1["tasks"][task], a1["primary_family"], ex["stage1"], ex["stage1"]["holm"],
                    expl["stage1"], task)
    task_macros(M, "TwoSq", a2["tasks"]["square"], a2["primary_family"], ex["stage2"], ex["stage2"]["holm"],
                expl["stage2"], "square")
    # state-level (post hoc) macros last: their paired thresholds must override task_macros()
    for key, pre in (("stage1_lift", "OneLift"), ("stage1_can", "OneCan"), ("stage1_square", "OneSq"),
                     ("stage2_square", "TwoSq")):
        state_macros(M, pre, sl["seed_effect"][key])
    g = expl["stage_agreement"]
    M["AgreeR"] = f2(g["pearson_r"]); M["AgreeRsq"] = pts(g["pearson_r"] ** 2)
    M["AgreeSD"] = f3(g["sd_diff"]); M["AgreeExp"] = f3(g["expected_sd_diff_binomial"])
    M["AgreeMean"] = pts(abs(g["mean_diff_50_minus_500"]), 1); M["AgreeMax"] = pts(g["max_abs_diff"])
    # Square Arm B partition means, both stages
    pm1 = a1["tasks"]["square"]["B"]["partition_means"]; pm2 = a2["tasks"]["square"]["B"]["partition_means"]
    for op, nm in zip(OPS, ["BOne", "BTwo", "OOne", "OTwo", "WOne", "WTwo"]):
        M["OneSqDrop" + nm] = f3(pm1[op]); M["TwoSqDrop" + nm] = f3(pm2[op])

    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by analysis/make_paper_numbers.py from results/*.json. Do not edit.\n")
        for k in sorted(M):
            fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")

    # ---- Table: variance components and preregistered contrasts
    rows = [components_row(t.capitalize(), a1["tasks"][t], a1["primary_family"], ex["stage1"], ex["stage1"]["holm"], t)
            for t in ("lift", "can", "square")]
    row2 = components_row("Square", a2["tasks"]["square"], a2["primary_family"], ex["stage2"], ex["stage2"]["holm"], "square")
    with open(os.path.join(OUT, "tables", "tab_components.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by analysis/make_paper_numbers.py. Do not edit.\n"
                 "\\begin{tabular}{@{}l ccc ccc ccc cc ccc@{}}\n\\toprule\n"
                 " & \\multicolumn{3}{c}{Arm A (10 seeds)} & \\multicolumn{3}{c}{Arm B (operator held out)} & "
                 "\\multicolumn{3}{c}{Arm C (random 250)} & \\multicolumn{2}{c}{$R$ (Sec.~\\ref{sec:stats})} & "
                 "\\multicolumn{3}{c}{$D \\times 10^{3}$ (Sec.~\\ref{sec:stats})} \\\\\n"
                 "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-12}\\cmidrule(l){13-15}\n"
                 "Task & mean & SD & floor & $\\hat\\sigma_p$ & $\\hat\\sigma_s$ & $\\hat\\sigma_e$ & "
                 "$\\hat\\sigma_p$ & $\\hat\\sigma_s$ & $\\hat\\sigma_e$ & est.\\ [95\\%] & $p_{\\mathrm{Holm}}$ & "
                 "est.\\ [95\\%] & $p$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
                 "\\multicolumn{15}{@{}l}{\\emph{Stage 1 (preregistered primary): 50 rollouts per checkpoint}} \\\\\n"
                 + "\n".join(rows) + "\n\\midrule\n"
                 "\\multicolumn{15}{@{}l}{\\emph{Stage 2 (follow-up, Amendment 2): same checkpoints, 500 rollouts}} \\\\\n"
                 + row2 + "\n\\bottomrule\n\\end{tabular}\n")

    # ---- Table: Square Arm B partition means by stage
    t1 = np.array(a1["tasks"]["square"]["B"]["table"]); t2 = np.array(a2["tasks"]["square"]["B"]["table"])
    with open(os.path.join(OUT, "tables", "tab_square_partitions.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by analysis/make_paper_numbers.py. Do not edit.\n"
                 "\\begin{tabular}{@{}l cc cc@{}}\n\\toprule\n"
                 " & \\multicolumn{2}{c}{50 rollouts} & \\multicolumn{2}{c}{500 rollouts} \\\\\n"
                 "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
                 "Held-out operator & mean & seed range & mean & seed range \\\\\n\\midrule\n")
        for i, op in enumerate(OP_TEX):
            fh.write(f"{op} & {t1[i].mean():.3f} & {t1[i].min():.2f}--{t1[i].max():.2f} & "
                     f"{t2[i].mean():.3f} & {t2[i].min():.2f}--{t2[i].max():.2f} \\\\\n")
        b1, b2 = a1["tasks"]["square"]["B"], a2["tasks"]["square"]["B"]
        fh.write("\\midrule\n"
                 f"Between-tier share & \\multicolumn{{2}}{{c}}{{{b1['tier_permutation']['observed_share']:.2f}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{b2['tier_permutation']['observed_share']:.2f}}} \\\\\n"
                 f"Null mean share & \\multicolumn{{2}}{{c}}{{{b1['tier_permutation']['null_mean_share']:.2f}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{b2['tier_permutation']['null_mean_share']:.2f}}} \\\\\n"
                 f"Within-tier $|\\Delta|$ & \\multicolumn{{2}}{{c}}{{{b1['tier_pairwise']['within_tier_mean_absdiff']:.3f}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{b2['tier_pairwise']['within_tier_mean_absdiff']:.3f}}} \\\\\n"
                 f"Between-tier $|\\Delta|$ & \\multicolumn{{2}}{{c}}{{{b1['tier_pairwise']['between_tier_mean_absdiff']:.3f}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{b2['tier_pairwise']['between_tier_mean_absdiff']:.3f}}} \\\\\n"
                 "\\bottomrule\n\\end{tabular}\n")

    # ---- Table: seed effect on shared states and resolvability (post hoc, Amendment 4)
    with open(os.path.join(OUT, "tables", "tab_resolvability.tex"), "w") as fh:
        fh.write("% AUTO-GENERATED by analysis/make_paper_numbers.py. Do not edit.\n"
                 "\\begin{tabular}{@{}l c ccc c cc@{}}\n\\toprule\n"
                 " & & & \\multicolumn{2}{c}{floor} & & \\multicolumn{2}{c}{min.\\ resolvable diff.\\ (points)} \\\\\n"
                 "\\cmidrule(lr){4-5}\\cmidrule(l){7-8}\n"
                 "Task & $n$ & seed SD & binomial & per-state & Cochran $p$ & paired & unpaired \\\\\n\\midrule\n")
        for key, lab in (("stage1_lift", "Lift"), ("stage1_can", "Can"), ("stage1_square", "Square"),
                         ("stage2_square", "Square")):
            v = sl["seed_effect"][key]
            mark = "$^{*}$" if v["paired_sd_is_floor"] else ""
            fh.write(f"{lab} & {v['n_states']} & {v['seed_sd']:.3f} & {v['binomial_floor']:.3f} & {v['per_state_floor']:.3f} & "
                     f"{pv(v['cochran_p'])} & {100 * v['mdd_paired_normal']:.1f} ({100 * v['mdd_paired_t4']:.1f}){mark} & "
                     f"{100 * v['mdd_unpaired_normal']:.1f} ({100 * v['mdd_unpaired_t4']:.1f}) \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {len(M)} macros and 3 tables")


if __name__ == "__main__":
    main()
