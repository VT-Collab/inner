"""Plot the four figures of Section VII from results.csv (mean and standard deviation over seeds): figures/fig_data.png
(E1: success, Jacobian norm and null-space ratio versus demonstrations), figures/fig_irrelevant.png (E2: success versus
state dimension), figures/fig_shift.png (E3) and figures/fig_scramble.png (E4: success per method). The same four figures
with the mean goal error (final distance to the goal, averaged over the 1000 test starts) in place of the success rate are
figures/fig_data_error.png, fig_irrelevant_error.png, fig_shift_error.png and fig_scramble_error.png. Run collect_results.py
first. The figure titles show the seed counts, so partial results are plotted as they land.
"""
import os, csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
ALGOS = [("bc", "BC", "#7f7f7f"), ("inner", "InneR", "#1f77b4"), ("frobenius", "Frobenius", "#ff7f0e"), ("nuclear", "Nuclear", "#2ca02c"), ("inner_c", "InneR-C", "#9467bd")]
CELLS = [("mlp", "static"), ("mlp", "dynamic"), ("diffusion", "static"), ("diffusion", "dynamic")]
TITLE = {"mlp": "MLP", "diffusion": "Diffusion"}

rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))
for r in rows:
    for k in ("state_dim", "n_demos", "seed"):
        r[k] = int(r[k])
    for k in ("success", "err_mean", "jacobian_norm", "null_ratio"):
        r[k] = float(r[k]) if r[k] else np.nan


def series(exp, policy, env, algo, x_key, metric):
    """x values, means and standard deviations over seeds of a metric, and the seed counts"""
    g = defaultdict(list)
    for r in rows:
        if (r["experiment"], r["policy"], r["env"], r["algo"]) == (exp, policy, env, algo):
            g[r[x_key]].append(r[metric])
    xs = sorted(g)
    return xs, np.array([np.mean(g[x]) for x in xs]), np.array([np.std(g[x], ddof=1) if len(g[x]) > 1 else 0 for x in xs]), [len(g[x]) for x in xs]


def seeds_label(exp, policy, env):
    n = defaultdict(set)
    for r in rows:
        if (r["experiment"], r["policy"], r["env"]) == (exp, policy, env):
            n[r["algo"]].add(r["seed"])
    c = sorted({len(v) for v in n.values()})
    return f"{c[0]} seeds" if len(c) == 1 else f"{c[0]} to {c[-1]} seeds"


def line_figure(exp, x_key, x_label, metrics, fname, log_x=True):
    fig, axes = plt.subplots(len(metrics), 4, figsize=(16, 3.4 * len(metrics)), squeeze=False)
    for j, (policy, env) in enumerate(CELLS):
        for i, (metric, y_label, log_y) in enumerate(metrics):
            ax = axes[i, j]
            for algo, name, color in ALGOS:
                xs, m, s, _ = series(exp, policy, env, algo, x_key, metric)
                if not xs:
                    continue
                ax.plot(xs, m, "-o", color=color, label=name, ms=4, lw=1.5)
                ax.fill_between(xs, m - s, m + s, color=color, alpha=0.15, lw=0)
            if log_x:
                ax.set_xscale("log")
                xs_all = sorted({r[x_key] for r in rows if (r["experiment"], r["policy"], r["env"]) == (exp, policy, env)})
                ax.set_xticks(xs_all); ax.set_xticklabels([str(x) for x in xs_all]); ax.minorticks_off()
            if log_y:
                ax.set_yscale("log")
            elif metric == "success":
                ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(f"{TITLE[policy]}, {env} ({seeds_label(exp, policy, env)})", fontsize=11)
            if i == len(metrics) - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(y_label)
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


def bar_figure(exp, fname, title, metric="success"):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for j, (policy, env) in enumerate(CELLS):
        ax = axes[j]
        demos = sorted({r["n_demos"] for r in rows if (r["experiment"], r["policy"], r["env"]) == (exp, policy, env)})
        tops = []
        for k, (algo, name, color) in enumerate(ALGOS):
            _, m, s, n = series(exp, policy, env, algo, "n_demos", metric)
            if len(m):
                ax.bar(k, m[0], yerr=s[0], color=color, width=0.7, capsize=3, label=name)
                if metric == "success":
                    ax.text(k, min(m[0] + s[0] + 0.03, 0.95), f"{100*m[0]:.0f}%", ha="center", fontsize=8)
                else:
                    ax.text(k, m[0] + s[0], f"{m[0]:.2f}", ha="center", va="bottom", fontsize=8)
                tops.append(m[0] + s[0])
        ax.set_xticks(range(len(ALGOS))); ax.set_xticklabels([a[1] for a in ALGOS], fontsize=9)
        ax.set_ylim(0, 1.05 if metric == "success" else 1.15 * max(tops, default=1)); ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{TITLE[policy]}, {env}, {demos[0] if demos else '?'} demos ({seeds_label(exp, policy, env)})", fontsize=10)
        if j == 0:
            ax.set_ylabel("success rate" if metric == "success" else "goal error")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


line_figure("E1", "n_demos", "demonstrations",
            [("success", "success rate", False), ("jacobian_norm", "squared Frobenius norm of J (trace of F)", True), ("null_ratio", "InneR null-space ratio", True)],
            "fig_data.png")
line_figure("E2", "state_dim", "state dimension n", [("success", "success rate", False)], "fig_irrelevant.png")
bar_figure("E3", "fig_shift.png", "E3: training noise correlated with the goal (rho = 0.9), test uncorrelated")
bar_figure("E4", "fig_scramble.png", "E4: nonlinear scramble of the policy input (condition 3, bend 0.5)")

# the same figures with the mean goal error in place of the success rate
line_figure("E1", "n_demos", "demonstrations", [("err_mean", "goal error", False)], "fig_data_error.png")
line_figure("E2", "state_dim", "state dimension n", [("err_mean", "goal error", False)], "fig_irrelevant_error.png")
bar_figure("E3", "fig_shift_error.png", "E3: training noise correlated with the goal (rho = 0.9), test uncorrelated", metric="err_mean")
bar_figure("E4", "fig_scramble_error.png", "E4: nonlinear scramble of the policy input (condition 3, bend 0.5)", metric="err_mean")
print("wrote", sorted(os.listdir(OUT)))
