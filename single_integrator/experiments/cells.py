"""The four experiments of Section VII of the paper, enumerated once and shared by make_jobs.py,
run_experiment.py and collect_results.py. A cell is one (policy, env, state_dim, n_demos, rho, cond) setting; every cell is
run for the five algorithms and SEEDS. The n = 16 stage-2 cells of E2 are the same models as E1 (same tag), run once.
"""
ALGOS = ["bc", "inner", "frobenius", "nuclear", "inner_c"]
SEEDS = range(10)
RHO, COND, BEND = 0.9, 3.0, 0.5
DEMOS_E1 = {"static": [25, 50, 100, 200, 400], "dynamic": [50, 100, 200, 400, 800]}
DIMS_E2 = {"static": [8, 12, 16, 24, 32], "dynamic": [13, 16, 24, 32]}       # n = 8 is MLP only (r = 8 needs a null space)
DEMOS_E2 = {("mlp", "static"): 100, ("mlp", "dynamic"): 400, ("diffusion", "static"): 50, ("diffusion", "dynamic"): 200}
DEMOS_E3 = {("mlp", "static"): 200, ("mlp", "dynamic"): 400, ("diffusion", "static"): 50, ("diffusion", "dynamic"): 300}
DEMOS_E4 = {("mlp", "static"): 400, ("mlp", "dynamic"): 800, ("diffusion", "static"): 100, ("diffusion", "dynamic"): 600}
NAMES = {"E1": "dataset size", "E2": "irrelevant state dimension", "E3": "shifted train/test distribution", "E4": "dispersed information"}


def cell(policy, env, state_dim, n_demos, rho=0.0, cond=0.0):
    return dict(policy=policy, env=env, state_dim=state_dim, n_demos=n_demos, rho=rho, cond=cond)


def all_cells():
    """every distinct cell of the four experiments (deduplicated), in experiment order"""
    out = []
    for policy in ("mlp", "diffusion"):
        for env in ("static", "dynamic"):
            out += [cell(policy, env, 16, n) for n in DEMOS_E1[env]]                                       # E1
            out += [cell(policy, env, d, DEMOS_E2[policy, env]) for d in DIMS_E2[env] if not (policy == "diffusion" and d == 8)]  # E2
            out.append(cell(policy, env, 16, DEMOS_E3[policy, env], rho=RHO))                               # E3
            out.append(cell(policy, env, 16, DEMOS_E4[policy, env], cond=COND))                             # E4
    seen, uniq = set(), []
    for c in out:
        if tag(c) not in seen:
            seen.add(tag(c)); uniq.append(c)
    return uniq


def experiments_of(c):
    """the experiments a cell belongs to (the n = 16 stage-2 cells belong to E1 and E2)"""
    if c["rho"] > 0:
        return ["E3"]
    if c["cond"] > 0:
        return ["E4"]
    exps = []
    if c["state_dim"] == 16 and c["n_demos"] in DEMOS_E1[c["env"]]:
        exps.append("E1")
    if c["n_demos"] == DEMOS_E2[c["policy"], c["env"]] and c["state_dim"] in DIMS_E2[c["env"]]:
        exps.append("E2")
    return exps


def tag(c, algo=None, seed=None):
    """<policy>[_<algo>]_<env>_d<n>_n<demos>[_rho0.9|_cond3][_s<seed>]"""
    t = c["policy"] + (f"_{algo}" if algo else "") + f"_{c['env']}_d{c['state_dim']}_n{c['n_demos']}"
    if c["rho"] > 0:
        t += f"_rho{c['rho']:g}"
    if c["cond"] > 0:
        t += f"_cond{c['cond']:g}"
    return t + (f"_s{seed}" if seed is not None else "")


if __name__ == "__main__":
    cells = all_cells()
    for e in NAMES:
        n = sum(1 for c in cells if e in experiments_of(c))
        print(f"{e} ({NAMES[e]}): {n} cells")
    print(f"{len(cells)} distinct cells x {len(ALGOS)} algorithms x {len(SEEDS)} seeds = {len(cells) * len(ALGOS) * len(SEEDS)} runs")
    for c in cells:
        print(f"  {tag(c):42s} {'+'.join(experiments_of(c))}")
