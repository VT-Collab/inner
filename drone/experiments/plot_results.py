import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Plot merged MLP/DP drone results using mean and standard error across seeds.
Manuscript colors, success-rate panels without legends.
Error lines extend upward by one SEM without caps; line bands show mean +/- SEM.
SEM is sample standard deviation (ddof=1) divided by sqrt(valid seed count).
Success uncertainty bands are clipped to the valid 0--100% range for display.
"""
import os, csv, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, PercentFormatter

# Match the reference sizes, with Times New Roman requested for drone figures.
from matplotlib import font_manager
font_override = os.environ.get('DRONE_FONT_PATH')
legacy_font = Path.home()/'.ros/webotsR2025a/webots/resources/fonts/Times New Roman.ttf'
if font_override:
    font_manager.fontManager.addfont(str(Path(font_override).expanduser()))
elif legacy_font.is_file():
    font_manager.fontManager.addfont(str(legacy_font))
else:
    try:
        font_manager.findfont('Times New Roman', fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError('Install Times New Roman or set DRONE_FONT_PATH to its .ttf file before plotting.') from exc
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
                     'font.size': 10, 'axes.labelsize': 10,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
                     'mathtext.it': 'Liberation Serif:italic',
                     'mathtext.bf': 'Liberation Serif:bold', 'pdf.fonttype': 42})
TITLE_FONT = {'fontfamily': 'Liberation Serif', 'fontweight': 'bold', 'fontsize': 10}
from paths import DP_RESULTS
parser = argparse.ArgumentParser(description="Reference-style drone plots from results.csv")
parser.add_argument('--csv-dir',type=Path,default=DP_RESULTS)
args = parser.parse_args()
OUT = str(args.csv_dir / 'csv_figures' / 'results')
os.makedirs(OUT, exist_ok=True)
# Method colors of the paper figures.
ALGOS = [("bc", "DP", "#808080"), ("inner", "InneR", "#ff9900"), ("frobenius", "Frobenius", "#4d4d4d"), ("nuclear", "Nuclear", "#9b7653"), ("inner_c", "InneR-C", "#8b6fae")]
CELLS = [("mlp", "drone"), ("diffusion", "drone")]
TITLE = {"mlp": "MLP", "diffusion": "Diffusion"}

rows = list(csv.DictReader(open(args.csv_dir / "results.csv")))
for r in rows:
    for k in ("state_dim", "n_demos", "seed"):
        r[k] = int(r[k])
    for k in ("success", "err_mean", "jacobian_norm", "null_ratio"):
        r[k] = float(r[k]) if r[k] else np.nan


summary_rows = list(csv.DictReader(open(args.csv_dir / 'summary.csv')))
# Verify the two exports agree before drawing.
keys = ['experiment','policy','env','state_dim','n_demos','rho','cond','algo']
for item in summary_rows:
    matched = [r for r in rows if all(str(r[k]) == item[k] for k in keys)]
    if len(matched) != int(item['n_seeds']):
        raise ValueError('CSV snapshots disagree on seed coverage; refresh both exports')
    for metric in ['success','err_mean','jacobian_norm','null_ratio']:
        values = [r[metric] for r in matched if np.isfinite(r[metric])]
        if not values:
            continue
        mean, std = np.mean(values), np.std(values,ddof=1) if len(values)>1 else 0.
        if not np.allclose([mean,std],[float(item[metric+'_mean']),float(item[metric+'_std'])],rtol=1e-10,atol=1e-12):
            raise ValueError(f'CSV aggregate mismatch: {metric}, {item}')

def series(exp, policy, env, algo, x_key, metric):
    groups = defaultdict(list)
    for row in rows:
        if (row['experiment'], row['policy'], row['env'], row['algo']) == (exp, policy, env, algo):
            if np.isfinite(row[metric]):
                groups[row[x_key]].append(row[metric])
    xs = sorted(groups)
    return xs, np.array([np.mean(groups[x]) for x in xs]), np.array([
        np.std(groups[x], ddof=1) / np.sqrt(len(groups[x])) if len(groups[x])>1 else 0. for x in xs]), [len(groups[x]) for x in xs]


def style_axes(ax):
    ax.set_box_aspect(1)  # Square plotting area, independent of data units.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    if ax.get_yscale() == 'log':
        ax.yaxis.set_minor_formatter(NullFormatter())


# Common height and width per panel keep typography consistent across figures.
FIGURE_POINTS = {'E2': (616.591, 137.943), 'E1': (308.296, 137.943),
                 'E3': (308.296, 137.943), 'E4': (308.296, 137.943)}

def figure_size(exp):
    return tuple(v/72 for v in FIGURE_POINTS[exp])


ORDERED = sorted(ALGOS, key=lambda a: ['bc','frobenius','nuclear','inner_c','inner'].index(a[0]))


def line_figure(exp, x_key, x_label, fname):
    show_diagnostics = exp == 'E2'
    fig = plt.figure(figsize=figure_size(exp), layout='compressed')
    fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.01, hspace=.01)
    outer = fig.add_gridspec(1, 4 if show_diagnostics else 2, wspace=.01)
    for col, (policy, env) in enumerate(CELLS):
        ax = fig.add_subplot(outer[0, col])
        diagnostics = [fig.add_subplot(outer[0, i]) for i in (2,3)] if show_diagnostics and policy == 'mlp' else []
        data = [(algo, color, *series(exp,policy,env,algo,x_key,'success')[:3]) for algo,_,color in ORDERED]
        for algo,color,xs,mean,sem in data:
            ax.plot(xs,mean,'-o',color=color,ms=3,lw=1.3)
            ax.fill_between(xs,np.clip(mean-sem,0,1),np.clip(mean+sem,0,1),color=color,alpha=.15,lw=0)
        xs_all = sorted({r[x_key] for r in rows if (r['experiment'],r['policy'],r['env'])==(exp,policy,env)})
        ax.set_xticks(xs_all)
        ax.set_ylim(0,1.02)
        ax.set_yticks([0,.25,.5,.75,1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel(x_label); ax.set_ylabel('success rate')
        ax.set_title(TITLE[policy],**TITLE_FONT)
        width=.8/len(ORDERED)
        for panel,metric,label in zip(diagnostics,['jacobian_norm','null_ratio'],
                [r'Jacobian Norm $\|J\|_F^2$' if policy=='mlp' else 'trace of Fisher information', 'Null Space Ratio']):
            for j,(algo,_,color) in enumerate(ORDERED):
                xs,mean,sem,_=series(exp,policy,env,algo,x_key,metric)
                positions=[xs_all.index(x)+(j-(len(ORDERED)-1)/2)*width for x in xs]
                if len(mean):
                    panel.bar(positions,mean,width=width,color=color,yerr=np.vstack((np.zeros_like(sem),sem)),
                              capsize=0,error_kw={'elinewidth':.8})
            panel.set_xticks(range(len(xs_all)),[str(x) for x in xs_all])
            panel.set_xlabel(x_label); panel.set_ylabel(label)
            panel.set_title('MLP',**TITLE_FONT)
            if metric=='jacobian_norm': panel.set_yscale('log')
            else: panel.set_ylim(bottom=0)
        for panel in [ax,*diagnostics]:
            style_axes(panel)
    for extension in ('png', 'pdf', 'svg'):
        export_options = {'bbox_inches': 'tight', 'pad_inches': .02}
        if extension == 'svg':
            export_options['transparent'] = True
        fig.savefig(Path(OUT) / (Path(fname).stem + '.' + extension), dpi=200, **export_options)
    plt.close(fig)


def bar_figure(exp,fname):
    fig,axes=plt.subplots(1,len(CELLS),figsize=figure_size(exp),layout='compressed')
    fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.01, hspace=.01)
    for ax,(policy,env) in zip(axes,CELLS):
        for k,(algo,name,color) in enumerate(ORDERED):
            _,mean,sem,_=series(exp,policy,env,algo,'n_demos','success')
            if len(mean)!=1: raise ValueError(f'Expected one condition: {exp}, {policy}, {algo}')
            ax.bar(k,mean[0],yerr=[[0.],[min(sem[0],1-mean[0])]],color=color,width=.7,capsize=0)
        ax.set_xticks([])  # Manuscript identifies methods by color, without bar labels.
        ax.set_ylim(0,1.02)
        ax.set_yticks([0,.25,.5,.75,1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel('success rate')
        ax.set_title(TITLE[policy],**TITLE_FONT)
        style_axes(ax)
    for extension in ('png', 'pdf', 'svg'):
        export_options = {'bbox_inches': 'tight', 'pad_inches': .02}
        if extension == 'svg':
            export_options['transparent'] = True
        fig.savefig(Path(OUT) / (Path(fname).stem + '.' + extension), dpi=200, **export_options)
    plt.close(fig)


line_figure('E1','state_dim',r'State Dimension ($n$)','drone_fig_irrelevant.png')
line_figure('E2','n_demos','demonstrations','drone_fig_data.png')
bar_figure('E3','drone_fig_shift.png')
bar_figure('E4','drone_fig_scramble.png')
print('Wrote combined MLP/Diffusion figures to',OUT)
