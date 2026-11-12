import os,json,csv,math
from collections import defaultdict
from experiments.mlp_cells import NAMES
from paths import MLP_RESULTS
COLS = ["experiment", "policy", "algo", "env", "state_dim", "n_demos", "seed", "rho", "cond",
        "success", "err_mean", "err_median", "jacobian_norm", "null_ratio", "nuclear_norm", "null_ratio_batch", "tag", "model"]
METRICS = ["success", "err_mean", "err_median", "jacobian_norm", "null_ratio", "nuclear_norm", "null_ratio_batch"]


def collect_mlp(root=MLP_RESULTS):
    HERE=str(root)
    rows = []
    for line in open(os.path.join(HERE, "drone_mlp_results.jsonl")):
        r = json.loads(line)
        for e in r["experiments"]:
            rows.append({**{k: r.get(k) for k in COLS if k not in ("experiment", "model")}, "experiment": e, "model": f"models/{r['tag']}.pt"})
    rows.sort(key=lambda r: (r["experiment"], r["policy"], r["env"], r["state_dim"], r["n_demos"], r["algo"], r["seed"]))
    with open(os.path.join(HERE, "drone_mlp_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    
    groups = defaultdict(list)
    for r in rows:
        groups[(r["experiment"], r["policy"], r["env"], r["state_dim"], r["n_demos"], r["rho"], r["cond"], r["algo"])].append(r)
    with open(os.path.join(HERE, "drone_mlp_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "policy", "env", "state_dim", "n_demos", "rho", "cond", "algo", "n_seeds"] + sum([[m + "_mean", m + "_std"] for m in METRICS], []))
        for k, g in sorted(groups.items()):
            row = list(k) + [len(g)]
            for m in METRICS:
                v = [r[m] for r in g if r[m] is not None]
                mean = sum(v) / len(v) if v else float("nan")
                std = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
                row += [f"{mean:.4f}", f"{std:.4f}"]
            w.writerow(row)
    counts = defaultdict(int)
    for r in rows:
        counts[r["experiment"]] += 1
    print("rows per experiment:", {e: counts[e] for e in NAMES}, "| cells summarized:", len(groups))
