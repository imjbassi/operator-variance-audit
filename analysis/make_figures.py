"""Regenerate every figure in the paper from the committed CSV / JSON results.

  figures/fig_arms.{pdf,png}        Stage 1: per-task success rates for Arms A, B, C with the
                                    binomial noise band of one 50-rollout estimate
  figures/fig_square_stages.{pdf,png}  Square: Stage 1 (50 rollouts) vs Stage 2 (500 rollouts)
  figures/fig_power.{pdf,png}       Power of the preregistered headline procedure vs true ratio

Usage:  python analysis/make_figures.py
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "figures")
OPS = ["better_operator_1", "better_operator_2", "okay_operator_1",
       "okay_operator_2", "worse_operator_1", "worse_operator_2"]
OP_SHORT = ["B1", "B2", "O1", "O2", "W1", "W2"]
B_KEYS = [f"B_drop_{op}" for op in OPS]
C_KEYS = [f"C_rand_{i}" for i in range(1, 7)]
TASK_TITLE = {"lift": "Lift", "can": "Can", "square": "Square"}

# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface);
# marker shape is the secondary encoding so identity never rests on colour alone (print / CVD).
ARM = {"A": {"c": "#2a78d6", "m": "o", "label": "Arm A: all 300 demos, 10 seeds"},
       "B": {"c": "#eb6834", "m": "s", "label": "Arm B: one operator held out"},
       "C": {"c": "#1baf7a", "m": "^", "label": "Arm C: random 250 demos"}}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)


def tab(df, arm, task, keys):
    sub = df[(df.arm == arm) & (df.task == task)]
    return np.array([[float(sub[(sub.partition_key == k) & (sub.seed == s)].success_rate.iloc[0])
                      for s in (1, 2, 3)] for k in keys])


def arm_panel(ax, df, task, n_roll, show_xlabels=True):
    """One small multiple: Arm A seeds, Arm B partitions, Arm C partitions on one success-rate axis."""
    A = df[(df.arm == "A") & (df.task == task)].sort_values("seed").success_rate.to_numpy(float)
    Bt, Ct = tab(df, "B", task, B_KEYS), tab(df, "C", task, C_KEYS)
    pbar = A.mean()
    bsd = np.sqrt(pbar * (1 - pbar) / n_roll)
    xA, xB, xC = 0.0, 1.7 + 1.55 * np.arange(6), 11.2 + 0.9 * np.arange(6)
    # noise band: Arm A mean +/- 1 binomial SD of a single n-rollout estimate
    ax.axhspan(pbar - bsd, pbar + bsd, color="#f0efec", zorder=0, linewidth=0)
    ax.axhline(pbar, color=AXIS, linewidth=0.7, zorder=1)
    kw = dict(s=20, linewidths=0.6, edgecolors=SURFACE, zorder=3)
    jit = np.linspace(-0.3, 0.3, A.size)
    ax.scatter(xA + jit, A, color=ARM["A"]["c"], marker=ARM["A"]["m"], **kw)
    for xs, T, arm in ((xB, Bt, "B"), (xC, Ct, "C")):
        for x, row in zip(xs, T):
            ax.scatter(x + np.array([-0.2, 0.0, 0.2]), row, color=ARM[arm]["c"], marker=ARM[arm]["m"], **kw)
            ax.plot([x - 0.36, x + 0.36], [row.mean()] * 2, color=INK, linewidth=1.1, zorder=4, solid_capstyle="butt")
    ax.set_xlim(-0.8, 16.4)
    ax.set_xticks(list(xB))   # only Arm B partitions are named; Arm C draws are exchangeable
    ax.set_xticklabels(OP_SHORT if show_xlabels else [""] * 6, fontsize=5.2)
    ax.tick_params(axis="x", length=0, pad=2)
    for x in (0.85, 10.3):
        ax.axvline(x, color=GRID, linewidth=0.6, zorder=0)
    style(ax)
    return pbar, bsd


def fig_arms(df):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), gridspec_kw={"wspace": 0.26})
    lims = {"lift": (0.78, 1.015), "can": (0.74, 1.015), "square": (0.36, 0.82)}
    for ax, task in zip(axes, ["lift", "can", "square"]):
        n = int(df[(df.task == task)].n_rollouts.iloc[0])
        arm_panel(ax, df, task, n)
        ax.set_ylim(*lims[task])
        ax.set_title(TASK_TITLE[task], loc="left", color=INK, pad=12)
        group_labels(ax)
    axes[0].set_ylabel("Success rate (final checkpoint, 50 rollouts)")
    legend(fig, "Arm A mean ± 1 binomial SD of one 50-rollout estimate")
    save(fig, "fig_arms")


def group_labels(ax):
    # group captions sit above the plot area, inside each group's own column
    for xc, txt in ((0.0, "A"), (5.6, "B"), (13.45, "C")):
        ax.text(xc, 1.0, txt, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.text(5.6, -0.105, "held-out operator", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.3, color=INK2)


def legend(fig, band_label):
    handles = [plt.Line2D([], [], linestyle="", marker=ARM[a]["m"], color=ARM[a]["c"], markersize=4.5,
                          markeredgecolor=SURFACE, markeredgewidth=0.5, label=ARM[a]["label"]) for a in "ABC"]
    handles += [plt.Line2D([], [], color=INK, linewidth=1.1, label="mean over 3 seeds"),
                plt.Rectangle((0, 0), 1, 1, color="#f0efec", label=band_label)]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06),
               handletextpad=0.5, columnspacing=1.4, labelcolor=INK2)


def fig_square_stages(s1, s2):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.15), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, df, n, ttl in ((axes[0], s1, 50, "Square, Stage 1: 50 rollouts per checkpoint"),
                           (axes[1], s2, 500, "Square, Stage 2: same checkpoints, 500 rollouts")):
        arm_panel(ax, df, "square", n)
        ax.set_ylim(0.36, 0.82)
        ax.set_title(ttl, loc="left", color=INK, pad=12)
        group_labels(ax)
    axes[0].set_ylabel("Success rate (final checkpoint)")
    legend(fig, "Arm A mean ± 1 binomial SD of one estimate")
    save(fig, "fig_square_stages")


def fig_power(analysis, analysis2=None):
    """Power of the headline procedure. Colour = task; line style = stage (solid: 50 rollouts,
    dashed: Square at 500 rollouts). Every line is direct-labelled, so identity never rests on colour."""
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    cols = {"lift": "#2a78d6", "can": "#eb6834", "square": "#1baf7a"}
    mk = {"lift": "o", "can": "s", "square": "^"}
    series = [(analysis["tasks"][t].get("power_headline_procedure"), t, "-", TASK_TITLE[t] + ", 50")
              for t in ["lift", "can", "square"]]
    if analysis2 is not None:
        series.append((analysis2["tasks"]["square"].get("power_headline_procedure"), "square", (0, (3, 1.6)), "Square, 500"))
    ends = []
    for P, task, ls, label in series:
        if not P:
            continue
        R = [v["true_R"] for v in P.values()]; pw = [v["power"] for v in P.values()]
        solid = ls == "-"
        ax.plot(R, pw, color=cols[task], linewidth=1.5, linestyle=ls, marker=mk[task], markersize=4,
                markeredgecolor=SURFACE if solid else cols[task], markeredgewidth=0.6 if solid else 1.1, zorder=3,
                markerfacecolor=cols[task] if solid else SURFACE)
        ends.append((pw[-1], label))
    # direct labels at the line ends, nudged apart so they never collide
    ends.sort()
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < 0.075:
            ys[i] = ys[i - 1] + 0.075
    for y, (_, label) in zip(ys, ends):
        ax.text(4.08, y, label, va="center", ha="left", fontsize=6.6, color=INK2)
    ax.axhline(0.8, color=AXIS, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(1.0, 0.815, "80% power", fontsize=6.5, color=MUTED, va="bottom")
    ax.set_xlim(0.9, 5.15); ax.set_ylim(0, 1.0)
    ax.set_xticks([1, 1.5, 2, 3, 4])
    ax.set_xlabel("True ratio of operator-partition SD to seed SD")
    ax.set_ylabel("P(95% interval excludes 1)")
    style(ax)
    save(fig, "fig_power")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.03, dpi=220)
    plt.close(fig)
    print("wrote", name)


def main():
    s1 = pd.read_csv(os.path.join(REPO, "results", "runs.csv")); s1["seed"] = s1.seed.astype(int)
    fig_arms(s1)
    p2 = os.path.join(REPO, "results", "runs_stage2.csv")
    if os.path.exists(p2):
        s2 = pd.read_csv(p2); s2["seed"] = s2.seed.astype(int)
        if len(s2) == 46:
            fig_square_stages(s1, s2)
    with open(os.path.join(REPO, "results", "analysis.json")) as fh:
        a1 = json.load(fh)
    p2a = os.path.join(REPO, "results", "analysis_stage2.json")
    a2 = None
    if os.path.exists(p2a):
        with open(p2a) as fh:
            a2 = json.load(fh)
    fig_power(a1, a2)


if __name__ == "__main__":
    main()
