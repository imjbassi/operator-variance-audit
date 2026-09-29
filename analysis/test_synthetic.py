"""Sanity checks of the estimators in vc.py on synthetic 6 x 3 tables with known components.

Run:  python analysis/test_synthetic.py
Prints recovery of sigma_p / sigma_s by MoM and REML (averaged over simulations), bootstrap
interval coverage for the headline ratio, permutation-test calibration, and Holm behaviour.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc


def sim_table(rng, P, S, sd_p, sd_s, sd_e, mu=0.5):
    a = rng.normal(0, sd_p, P)[:, None]
    b = rng.normal(0, sd_s, S)[None, :]
    e = rng.normal(0, sd_e, (P, S))
    return mu + a + b + e


def main():
    rng = np.random.default_rng(0)
    P, S = 6, 3
    print("== recovery (mean over 300 sims), truth sd_p=0.10 sd_s=0.04 sd_e=0.05")
    mom_p, mom_s, reml_p, reml_s, bp, br = [], [], [], [], 0, 0
    for _ in range(300):
        t = sim_table(rng, P, S, 0.10, 0.04, 0.05)
        m = vc.mom_two_way(t); r = vc.reml_two_way(t)
        mom_p.append(m["s2_p"]); mom_s.append(m["s2_s"]); reml_p.append(r["s2_p"]); reml_s.append(r["s2_s"])
        bp += m["boundary_p"]; br += r["boundary_p"]
    print(f"  MoM  E[s2_p]^0.5={np.sqrt(np.mean(mom_p)):.3f}  E[s2_s]^0.5={np.sqrt(np.mean(mom_s)):.3f}  boundary_p frac={bp/300:.2f}")
    print(f"  REML E[s2_p]^0.5={np.sqrt(np.mean(reml_p)):.3f}  E[s2_s]^0.5={np.sqrt(np.mean(reml_s)):.3f}  boundary_p frac={br/300:.2f}")

    print("== null case: sd_p = 0 (expect ~50% boundary for MoM, F-test p uniform)")
    ps, bnd = [], 0
    for _ in range(300):
        t = sim_table(rng, P, S, 0.0, 0.04, 0.05)
        ps.append(vc.f_test_p(t)["p_partition"]); bnd += vc.mom_two_way(t)["boundary_p"]
    print(f"  boundary frac={bnd/300:.2f}  F-test rejection rate at 0.05={np.mean(np.array(ps) < 0.05):.3f}")

    print("== headline ratio bootstrap coverage (truth R = 0.10/0.04 = 2.5), 100 sims x 500 boots")
    cover, excl1 = 0, 0
    for _ in range(100):
        B = sim_table(rng, P, S, 0.10, 0.04, 0.05)
        A = rng.normal(0.5, np.sqrt(0.04 ** 2 + 0.05 ** 2), 10)  # Arm A seed spread includes residual noise
        truthR = 0.10 / np.sqrt(0.04 ** 2 + 0.05 ** 2)
        out = vc.bootstrap_ratio(B, A, n_boot=500, rng=rng)
        cover += out["ci_low"] <= truthR <= out["ci_high"]; excl1 += out["ci_low"] > 1
    print(f"  truth R={truthR:.2f}  CI coverage={cover/100:.2f}  P(CI excludes 1)={excl1/100:.2f}")

    print("== ratio bootstrap under R = 1 (false-positive rate of 'CI excludes 1')")
    fp = 0
    for _ in range(100):
        B = sim_table(rng, P, S, 0.05, 0.04, 0.05)
        A = rng.normal(0.5, np.sqrt(0.04 ** 2 + 0.05 ** 2), 10)
        truthR = 0.05 / np.sqrt(0.04 ** 2 + 0.05 ** 2)
        out = vc.bootstrap_ratio(B, A, n_boot=500, rng=rng)
        fp += out["ci_low"] > 1
    print(f"  truth R={truthR:.2f}  P(CI low > 1)={fp/100:.2f}")

    print("== tier permutation calibration under no tier effect (expect p<0.05 rate ~0.05)")
    tiers = ["better", "better", "okay", "okay", "worse", "worse"]
    rej = 0
    for _ in range(500):
        rm = rng.normal(0.5, 0.1, 6)
        rej += vc.tier_permutation_test(rm, tiers)["p_value"] < 0.05
    print(f"  rejection rate={rej/500:.3f}")

    print("== Holm")
    print("  ", vc.holm([0.01, 0.04, 0.03, 0.20, np.nan, 0.001]))
    print("== min detectable sd_p (s2_e=0.05^2, P=6, S=3, alpha=0.05, power=0.8):",
          f"{vc.min_detectable_sd_p(0.05**2):.3f}")


if __name__ == "__main__":
    main()
