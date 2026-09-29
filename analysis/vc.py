"""Variance-component estimators and resampling procedures (preregistered, Section 6).

All functions operate on plain numpy arrays so they can be unit-tested on synthetic data.

  table : (P, S) array of success rates, rows = partitions (operators / random draws), cols = seeds.
"""
import itertools, warnings
import numpy as np


# ---------------------------------------------------------------- 6.3 (i) method of moments
def mom_two_way(table):
    """ANOVA / method-of-moments two-way random effects, one observation per cell.
    Returns dict with raw (untruncated) and truncated components plus boundary flags."""
    y = np.asarray(table, dtype=float)
    P, S = y.shape
    gm = y.mean()
    rm = y.mean(axis=1, keepdims=True)
    cm = y.mean(axis=0, keepdims=True)
    ss_p = S * float(((rm - gm) ** 2).sum())
    ss_s = P * float(((cm - gm) ** 2).sum())
    ss_e = float(((y - rm - cm + gm) ** 2).sum())
    df_p, df_s, df_e = P - 1, S - 1, (P - 1) * (S - 1)
    ms_p, ms_s, ms_e = ss_p / df_p, ss_s / df_s, ss_e / df_e
    s2p_raw = (ms_p - ms_e) / S
    s2s_raw = (ms_s - ms_e) / P
    return {
        "P": P, "S": S, "grand_mean": gm,
        "ms_p": ms_p, "ms_s": ms_s, "ms_e": ms_e,
        "s2_p_raw": s2p_raw, "s2_s_raw": s2s_raw, "s2_e": ms_e,
        "s2_p": max(s2p_raw, 0.0), "s2_s": max(s2s_raw, 0.0),
        "sd_p": np.sqrt(max(s2p_raw, 0.0)), "sd_s": np.sqrt(max(s2s_raw, 0.0)), "sd_e": np.sqrt(ms_e),
        "boundary_p": s2p_raw <= 0.0, "boundary_s": s2s_raw <= 0.0,
        "F_p": ms_p / ms_e if ms_e > 0 else np.inf, "F_s": ms_s / ms_e if ms_e > 0 else np.inf,
        "df_p": df_p, "df_s": df_s, "df_e": df_e,
    }


def f_test_p(table):
    """Exact F-test p-values for H0: sigma2_p = 0 and H0: sigma2_s = 0 (balanced two-way, no interaction)."""
    from scipy import stats
    m = mom_two_way(table)
    return {"p_partition": float(stats.f.sf(m["F_p"], m["df_p"], m["df_e"])),
            "p_seed": float(stats.f.sf(m["F_s"], m["df_s"], m["df_e"]))}


# ---------------------------------------------------------------- 6.3 (ii) REML
def reml_two_way(table, tol=1e-10):
    """REML via statsmodels MixedLM with crossed variance components (partition, seed)
    inside a single dummy group. Returns dict with variances, SDs, boundary flags, convergence."""
    import pandas as pd
    import statsmodels.formula.api as smf
    y = np.asarray(table, dtype=float)
    P, S = y.shape
    df = pd.DataFrame({"y": y.ravel(),
                       "p": np.repeat(np.arange(P), S).astype(str),
                       "s": np.tile(np.arange(S), P).astype(str),
                       "g": "all"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("y ~ 1", df, groups="g", re_formula="0",
                            vc_formula={"p": "0 + C(p)", "s": "0 + C(s)"})
        best = None
        for method in ("lbfgs", "powell", "nm", "bfgs", "cg"):
            try:
                res = model.fit(reml=True, method=method, maxiter=2000)
            except Exception:
                continue
            if best is None or (np.isfinite(res.llf) and res.llf > best.llf + 1e-9):
                best = res
    if best is None:
        return {"s2_p": np.nan, "s2_s": np.nan, "s2_e": np.nan, "sd_p": np.nan, "sd_s": np.nan, "sd_e": np.nan,
                "boundary_p": None, "boundary_s": None, "converged": False, "llf": np.nan}
    names = list(best.model.exog_vc.names)
    vc = dict(zip(names, np.asarray(best.vcomp, dtype=float) * best.scale))  # vcomp is relative to scale
    s2p, s2s, s2e = float(vc["p"]), float(vc["s"]), float(best.scale)
    return {"s2_p": s2p, "s2_s": s2s, "s2_e": s2e,
            "sd_p": np.sqrt(max(s2p, 0)), "sd_s": np.sqrt(max(s2s, 0)), "sd_e": np.sqrt(s2e),
            "boundary_p": s2p <= tol * max(s2e, 1e-12), "boundary_s": s2s <= tol * max(s2e, 1e-12),
            "converged": bool(best.converged), "llf": float(best.llf)}


def estimators_disagree(mom, reml, factor=2.0):
    """Material disagreement per prereg 6.3: SDs differ by > factor, or exactly one is at the boundary."""
    out = {}
    for c in ("p", "s"):
        a, b = mom[f"sd_{c}"], reml[f"sd_{c}"]
        ba, bb = bool(mom[f"boundary_{c}"]), bool(reml[f"boundary_{c}"]) if reml[f"boundary_{c}"] is not None else False
        if ba != bb:
            out[c] = True
        elif ba and bb:
            out[c] = False
        else:
            lo, hi = min(a, b), max(a, b)
            out[c] = bool(hi > factor * lo) if lo > 0 else bool(hi > 0)
    return out


# ---------------------------------------------------------------- 6.2 seed SD (Arm A)
def seed_sd(vec):
    v = np.asarray(vec, dtype=float)
    return float(v.std(ddof=1)) if v.size > 1 else np.nan


def bootstrap_seed_sd(vec, n_boot=2000, rng=None):
    rng = np.random.default_rng(rng)
    v = np.asarray(vec, dtype=float)
    out = np.empty(n_boot)
    for b in range(n_boot):
        out[b] = seed_sd(v[rng.integers(0, v.size, v.size)])
    return out


# ---------------------------------------------------------------- 6.4 headline ratio
def bootstrap_ratio(B_table, A_vec, n_boot=2000, rng=None, estimator="mom"):
    """R = SD_partition(B) / SD_seed(A). Joint bootstrap: rows of B (operators) with replacement,
    entries of A (seeds) with replacement. Returns point estimate, bootstrap draws, percentile CI,
    one-sided bootstrap p-value for H0: R <= 1, and the fraction of resamples at the partition boundary."""
    rng = np.random.default_rng(rng)
    B = np.asarray(B_table, dtype=float)
    A = np.asarray(A_vec, dtype=float)
    est = mom_two_way if estimator == "mom" else reml_two_way

    def ratio(Bt, At):
        sp = est(Bt)["sd_p"]
        ss = seed_sd(At)
        return sp / ss if ss > 0 else (np.inf if sp > 0 else np.nan)

    point = ratio(B, A)
    draws = np.empty(n_boot)
    nb = 0
    for b in range(n_boot):
        Bt = B[rng.integers(0, B.shape[0], B.shape[0])]
        At = A[rng.integers(0, A.size, A.size)]
        draws[b] = ratio(Bt, At)
        nb += est(Bt)["boundary_p"]
    finite = draws[np.isfinite(draws)]
    ci = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
    p_one_sided = float(np.mean(draws <= 1.0))  # H0: R <= 1
    return {"point": float(point), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p_R_le_1": p_one_sided, "frac_boundary": nb / n_boot, "n_boot": n_boot,
            "draws": draws}


def within_table_ratio(table):
    """Secondary view: SD_partition / SD_seed both from the same 6 x 3 table (MoM)."""
    m = mom_two_way(table)
    return m["sd_p"] / m["sd_s"] if m["sd_s"] > 0 else (np.inf if m["sd_p"] > 0 else np.nan)


# ---------------------------------------------------------------- 6.5 B vs C
def bootstrap_diff(B_table, C_table, n_boot=2000, rng=None):
    """D = s2_p(B) - s2_p(C) (MoM, truncated). Resample rows within each arm."""
    rng = np.random.default_rng(rng)
    B, C = np.asarray(B_table, float), np.asarray(C_table, float)
    point = mom_two_way(B)["s2_p"] - mom_two_way(C)["s2_p"]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        Bt = B[rng.integers(0, B.shape[0], B.shape[0])]
        Ct = C[rng.integers(0, C.shape[0], C.shape[0])]
        draws[b] = mom_two_way(Bt)["s2_p"] - mom_two_way(Ct)["s2_p"]
    ci = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p_D_le_0": float(np.mean(draws <= 0.0)), "n_boot": n_boot, "draws": draws,
            "sd_p_B": mom_two_way(B)["sd_p"], "sd_p_C": mom_two_way(C)["sd_p"]}


# ---------------------------------------------------------------- 6.7 tier permutation
def tier_between_share(row_means, tiers):
    """Fraction of between-partition sum of squares explained by tier (3 tiers x 2 partitions)."""
    y = np.asarray(row_means, float)
    tiers = np.asarray(tiers)
    gm = y.mean()
    ss_tot = float(((y - gm) ** 2).sum())
    ss_between = 0.0
    for t in np.unique(tiers):
        yt = y[tiers == t]
        ss_between += yt.size * (yt.mean() - gm) ** 2
    return ss_between / ss_tot if ss_tot > 0 else np.nan


def tier_permutation_test(row_means, tiers):
    """Exact permutation test over all distinct assignments of the tier multiset to the 6 partitions.
    p = fraction of labellings whose between-tier share >= observed (observed included)."""
    y = np.asarray(row_means, float)
    tiers = list(tiers)
    obs = tier_between_share(y, tiers)
    seen, shares = set(), []
    for perm in itertools.permutations(tiers):
        if perm in seen:
            continue
        seen.add(perm)
        shares.append(tier_between_share(y, perm))
    shares = np.asarray(shares)
    return {"observed_share": float(obs), "n_labellings": int(len(shares)),
            "p_value": float(np.mean(shares >= obs - 1e-12)), "null_mean_share": float(shares.mean())}


def pairwise_tier_spread(row_means, tiers):
    """Mean |difference| between same-tier pairs vs cross-tier pairs of partition means."""
    y = np.asarray(row_means, float)
    tiers = np.asarray(tiers)
    within, between = [], []
    for i in range(y.size):
        for j in range(i + 1, y.size):
            (within if tiers[i] == tiers[j] else between).append(abs(y[i] - y[j]))
    return {"within_tier_mean_absdiff": float(np.mean(within)), "between_tier_mean_absdiff": float(np.mean(between)),
            "n_within_pairs": len(within), "n_between_pairs": len(between)}


# ---------------------------------------------------------------- 6.8 Holm-Bonferroni
def holm(pvals):
    """Holm step-down adjusted p-values (monotone), same order as input. NaNs are passed through."""
    p = np.asarray(pvals, float)
    idx = np.where(np.isfinite(p))[0]
    adj = np.full_like(p, np.nan)
    if idx.size == 0:
        return adj
    m = idx.size
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


# ---------------------------------------------------------------- 6.6 power
def min_detectable_sd_p(s2_e, P=6, S=3, alpha=0.05, power=0.8):
    """Smallest partition SD detectable by the exact F-test for sigma2_p = 0 in a balanced P x S
    two-way random-effects design with residual variance s2_e (which absorbs seed-by-partition
    interaction and rollout binomial noise). Under the alternative, MS_p/MS_e ~ (1 + S s2_p/s2_e) F."""
    from scipy import stats, optimize
    df1, df2 = P - 1, (P - 1) * (S - 1)
    fcrit = stats.f.isf(alpha, df1, df2)

    def pw(sd_p):
        lam = 1.0 + S * sd_p ** 2 / s2_e
        return stats.f.sf(fcrit / lam, df1, df2)

    if s2_e <= 0:
        return 0.0
    hi = 1.0
    while pw(hi) < power and hi < 1e3:
        hi *= 2
    return float(optimize.brentq(lambda x: pw(x) - power, 0.0, hi))
