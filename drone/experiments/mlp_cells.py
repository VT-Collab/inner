from policy.mlp.config import input_dim, e3_rho, e4_condition

ALGOS = ["bc", "inner", "frobenius", "nuclear", "inner_c"]
SEEDS = range(2, 93, 10)                                        # 2, 12, ..., 92
DEMOS_E2 = [100, 300, 500, 700]                                 # E2: the e2 input (22D)
EXPS_E1 = ["clean", "e1_1", "e1_2", "e1_3"]                     # E1: 19D, and 1, 2, 3 other drones (26D, 33D, 40D)
DEMOS = 500                                                     # the number of demos of E1, E3 and E4
NAMES = {"E1": "irrelevant state dimension", "E2": "dataset size", "E3": "shifted train/test distribution",
         "E4": "dispersed information"}


def cell(exp, n_demos):
    return dict(policy="mlp", env="drone", exp=exp, state_dim=input_dim(exp), n_demos=n_demos,
                rho=e3_rho if exp == "e3" else 0.0, cond=e4_condition if exp == "e4" else 0.0)


def all_cells():
    """every cell of the four experiments, in experiment order"""
    return ([cell(e, DEMOS) for e in EXPS_E1] + [cell("e2", n) for n in DEMOS_E2]
            + [cell("e3", DEMOS), cell("e4", DEMOS)])


def experiments_of(c):
    """the experiment a cell belongs to"""
    return [{"e2": "E2", "e3": "E3", "e4": "E4"}.get(c["exp"], "E1")]


def tag(c, algo=None, seed=None):
    """mlp[_<algo>]_drone_d<n>_n<demos>[_rho0.9|_cond3][_s<seed>], the tag format of single_integrator/experiments/cells.py"""
    t = c["policy"] + (f"_{algo}" if algo else "") + f"_{c['env']}_d{c['state_dim']}_n{c['n_demos']}"
    if c["rho"] > 0:
        t += f"_rho{c['rho']:g}"
    if c["cond"] > 0:
        t += f"_cond{c['cond']:g}"
    return t + (f"_s{seed}" if seed is not None else "")

