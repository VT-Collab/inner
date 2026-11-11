"""Write jobs_cpu.txt (MLP) and jobs_gpu.txt (diffusion): every run of the four experiments (cells.py x ALGOS x SEEDS),
seed-major so that an interrupted campaign still has complete seeds, and within a seed in experiment order. The queue skips
runs already in results.jsonl, so the lists can be regenerated at any time.
"""
import os
from cells import all_cells, ALGOS, SEEDS, BEND, tag, experiments_of
HERE = os.path.dirname(os.path.abspath(__file__))


def job(c, algo, seed):
    s = f"--policy {c['policy']} --algo {algo} --env {c['env']} --state_dim {c['state_dim']} --n_demos {c['n_demos']} --seed {seed}"
    if c["rho"] > 0:
        s += f" --noise_correlation {c['rho']:g}"
    if c["cond"] > 0:
        s += f" --scramble_condition {c['cond']:g} --scramble_bend {BEND:g}"
    return s


for policy, fname in (("mlp", "jobs_cpu.txt"), ("diffusion", "jobs_gpu.txt")):
    lines, n = [], 0
    for seed in SEEDS:
        lines.append(f"# seed {seed}")
        for c in all_cells():
            if c["policy"] != policy:
                continue
            lines.append(f"# {tag(c)} ({'+'.join(experiments_of(c))})")
            for algo in ALGOS:
                lines.append(job(c, algo, seed)); n += 1
    open(os.path.join(HERE, fname), "w").write("\n".join(lines) + "\n")
    print(f"{fname}: {n} runs")
