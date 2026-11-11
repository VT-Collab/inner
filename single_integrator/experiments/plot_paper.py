"""Paper versions of the Section VII figures, written as SVG to figures-svg/ (run collect_results.py first): the goal error
of E1 to E4 and the regularized quantities of E1 (the success-rate versions were dropped from the paper on 7 Nov 2026).
Formatting: Palatino Linotype with text kept as editable text (svg.fonttype none), no grid, no markers, mean and standard
error of the mean over seeds (shaded for lines, error bars for bars), methods in the order BC, Frobenius, Nuclear, InneR-C,
InneR with InneR in orange and the baselines in muted grays, browns and purples, state dimensions evenly spaced, four panels
(MLP static, MLP dynamic, Diffusion static, Diffusion dynamic) per figure at the IEEE double-column width.
"""
import os, csv, glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures-svg")
os.makedirs(OUT, exist_ok=True)
PREVIEW = os.environ.get("PREVIEW_DIR")  # optional folder for PNG previews of the same figures

# Palatino Linotype: registered from the Windows fonts folder when running under WSL; the SVG references the font by
# name, so the text stays editable and renders with the real font wherever it is installed
for f in glob.glob("/mnt/c/Windows/Fonts/pala*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Palatino Linotype", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})

ALGOS = [("bc", "BC", "#808080"), ("frobenius", "Frobenius", "#4d4d4d"), ("nuclear", "Nuclear", "#9b7653"),
         ("inner_c", "InneR-C", "#8b6fae"), ("inner", "InneR", "#ff9900")]
# error bars of the bar plots: a black SEM line without caps, drawn behind its bar so only the top half shows
ERR = dict(capsize=0, error_kw={"lw": 0.8, "ecolor": "black", "zorder": 1}, zorder=2)
CELLS = [("mlp", "static"), ("mlp", "dynamic"), ("diffusion", "static"), ("diffusion", "dynamic")]
TITLE = {"mlp": "MLP", "diffusion": "Diffusion"}
WIDTH = 7.16   # IEEE double-column text width in inches
ROW = 1.75     # height of one row of panels

rows = list(csv.DictReader(open(os.path.join(HERE, "results.csv"))))
for r in rows:
    for k in ("state_dim", "n_demos", "seed"):
        r[k] = int(r[k])
    for k in ("success", "err_mean", "jacobian_norm", "null_ratio"):
        r[k] = float(r[k]) if r[k] else np.nan


def series(exp, policy, env, algo, x_key, metric):
    """x values, means and standard errors of the mean over seeds"""
    g = defaultdict(list)
    for r in rows:
        if (r["experiment"], r["policy"], r["env"], r["algo"]) == (exp, policy, env, algo):
            g[r[x_key]].append(r[metric])
    xs = sorted(g)
    m = np.array([np.mean(g[x]) for x in xs])
    se = np.array([np.std(g[x], ddof=1) / np.sqrt(len(g[x])) if len(g[x]) > 1 else 0 for x in xs])
    return xs, m, se


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".svg"), format="svg")
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=200)
    plt.close(fig)


def line_figure(exp, x_key, x_label, metrics, name, even_x):
    """one row of four panels per metric; even_x places the x values at equal spacing (state dimension), otherwise log x"""
    fig, axes = plt.subplots(len(metrics), 4, figsize=(WIDTH, ROW * len(metrics) + 0.3), squeeze=False)
    for j, (policy, env) in enumerate(CELLS):
        xs_all = sorted({r[x_key] for r in rows if (r["experiment"], r["policy"], r["env"]) == (exp, policy, env)})
        pos = {x: (i if even_x else x) for i, x in enumerate(xs_all)}
        for i, (metric, y_label, log_y) in enumerate(metrics):
            ax = axes[i, j]
            for algo, label, color in ALGOS:
                xs, m, se = series(exp, policy, env, algo, x_key, metric)
                if not xs:
                    continue
                p = [pos[x] for x in xs]
                ax.plot(p, m, "-", color=color, label=label, lw=1.3)
                ax.fill_between(p, m - se, m + se, color=color, alpha=0.2, lw=0)
            if not even_x:
                ax.set_xscale("log"); ax.minorticks_off()
            ax.set_xticks([pos[x] for x in xs_all]); ax.set_xticklabels([str(x) for x in xs_all])
            if log_y:
                ax.set_yscale("log")
            elif metric == "success":
                ax.set_ylim(0, 1.02)
            else:
                ax.set_ylim(bottom=0)
            if i == 0:
                ax.set_title(f"{TITLE[policy]}, {env}")
            if i == len(metrics) - 1:
                ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel(y_label)
    axes[0, 0].legend(frameon=False, handlelength=1.5)
    fig.tight_layout(pad=0.4)
    save(fig, name)


def bar_figure(exp, metric, name):
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, ROW + 0.3))
    for j, (policy, env) in enumerate(CELLS):
        ax = axes[j]
        for k, (algo, label, color) in enumerate(ALGOS):
            _, m, se = series(exp, policy, env, algo, "n_demos", metric)
            if len(m):
                ax.bar(k, m[0], yerr=se[0], color=color, width=0.7, label=label, **ERR)
        ax.set_xticks(range(len(ALGOS))); ax.set_xticklabels([a[1] for a in ALGOS], rotation=30, ha="right")
        if metric == "success":
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(f"{TITLE[policy]}, {env}")
        if j == 0:
            ax.set_ylabel("success rate" if metric == "success" else "goal error")
    fig.tight_layout(pad=0.4)
    save(fig, name)


def grouped_bar_figure(exp, policy, env, demos, metrics, name, algos=ALGOS, width=3.5):
    """what each method regularizes: one panel per metric, bars grouped by demonstrations, methods within each group
    (the layout of Figure 9 of the November 2026 report), legend in one row below the panels"""
    fig, axes = plt.subplots(1, len(metrics), figsize=(width, 2.0))
    bar_w = 0.8 / len(algos)
    for ax, (metric, y_label, log_y) in zip(axes, metrics):
        for k, (algo, label, color) in enumerate(algos):
            xs, m, se = series(exp, policy, env, algo, "n_demos", metric)
            sel = [i for i, x in enumerate(xs) if x in demos]
            pos = [demos.index(xs[i]) + (k - (len(algos) - 1) / 2) * bar_w for i in sel]
            ax.bar(pos, m[sel], yerr=se[sel], width=bar_w, color=color, label=label, **ERR)
        ax.set_xticks(range(len(demos))); ax.set_xticklabels([str(d) for d in demos]); ax.set_xlabel("demonstrations")
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(y_label)
    fig.suptitle(f"{TITLE[policy]}, {env}", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(algos), frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(pad=0.4, rect=(0, 0.1, 1, 1))
    save(fig, name)


# E1 dataset size: the goal error
line_figure("E1", "n_demos", "demonstrations", [("err_mean", "goal error", False)], "fig_data_error", even_x=False)
# what each method regularizes, MLP on the static goal (Figure 9 of the November 2026 report, with InneR-C added)
grouped_bar_figure("E1", "mlp", "static", [100, 200, 400],
                   [("jacobian_norm", "squared Frobenius norm of J", True), ("null_ratio", "InneR null-space ratio", False)],
                   "fig_data_regularizers")
# E2 irrelevant state dimension (state dimensions evenly spaced)
line_figure("E2", "state_dim", "state dimension n", [("err_mean", "goal error", False)], "fig_irrelevant_error", even_x=True)
# E3 shifted distribution and E4 dispersed information: one bar per method
bar_figure("E3", "err_mean", "fig_shift_error")
bar_figure("E4", "err_mean", "fig_scramble_error")
print("wrote", sorted(os.listdir(OUT)))
