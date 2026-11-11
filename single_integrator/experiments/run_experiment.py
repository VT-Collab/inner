"""Train and evaluate one run of the Section VII experiments: one cell (policy, env, state_dim, n_demos, rho, cond), one
algorithm, one seed. The training loop is the one in toy_environment/train_{mlp,diffusion}_policy.py with the paper's
hyperparameters (PARAMS), the dataset generated in memory from the seed as in get_dataset.py, and the test rollout of
test_*_policy.py (1000 seeded starts, 40 steps). Writes models/<tag>.pt and appends one line to results.jsonl.

  toy_environment/venv/bin/python experiments/run_experiment.py --policy mlp --algo inner --env static --state_dim 16 --n_demos 100 --seed 0
  ... --noise_correlation 0.9  (E3)      ... --scramble_condition 3 --scramble_bend 0.5  (E4)
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "toy_environment"))
sys.path.insert(0, HERE)
import numpy as np
import torch
from models import MLPPolicy, DiffusionPolicy
from train_diffusion_policy import get_fisher_information
from get_dataset import get_action, correlate_noise
from config import action_dim, hidden_dim, n_steps
from cells import ALGOS, cell, tag, experiments_of

# hyperparameters of Section VII (identical to train_mlp_policy.py and train_diffusion_policy.py)
PARAMS = {
    "mlp": dict(EPOCH=200, LR=1e-3, BATCH_SIZE=200, K_NEIGHBORS=8, GAMMA=0.2, GAMMA_FROBENIUS=0.01, GAMMA_NUCLEAR=0.01),
    "diffusion": dict(EPOCH=1000, LR=3e-3, BATCH_SIZE=200, K_NEIGHBORS=8, GAMMA=0.3, GAMMA_FROBENIUS=1e-4, GAMMA_NUCLEAR=1e-3,
                      FISHER_RANK=2 * action_dim, N_ACTION_SAMPLES=2 * action_dim, N_NOISE_SAMPLES=4, N_SAMPLE_STEPS=1),
}
N_TESTS = 1000
RANGE = 10.0


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["mlp", "diffusion"], required=True)
    p.add_argument("--algo", choices=ALGOS, required=True)
    p.add_argument("--env", choices=["static", "dynamic"], required=True)
    p.add_argument("--state_dim", type=int, required=True)
    p.add_argument("--n_demos", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise_correlation", type=float, default=0.0, help="E3: Pearson correlation of each noise entry with a goal coordinate (training data only)")
    p.add_argument("--scramble_condition", type=float, default=0.0, help="E4: condition number bound of the scramble's first layer (0 = no scramble)")
    p.add_argument("--scramble_bend", type=float, default=0.5, help="E4: bend of the scramble (used only with --scramble_condition)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads (2 is best when many runs share the machine)")
    p.add_argument("--out_dir", default=HERE)
    p.add_argument("--tag_only", action="store_true")
    return p.parse_args()


def make_dataset(state_dim, n_demos, dynamic_goal, seed, noise_correlation):
    """n_demos expert demonstrations of n_steps (state, action) pairs, as get_dataset.get_dataset with a seeded RNG"""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n_demos):
        state = rng.uniform(-RANGE, RANGE, state_dim)
        if noise_correlation > 0:
            correlate_noise(state, dynamic_goal, noise_correlation)
        toggle = dynamic_goal and state[-1] < 0.0
        for _ in range(n_steps):
            action = get_action(state, toggle)
            data.append(state.tolist() + action.tolist())
            state[0:action_dim] += action
    return torch.tensor(data, dtype=torch.float32)


def evaluate(model, policy, state_dim, dynamic_goal, device, seed):
    """the test rollout of test_*_policy.py: mean and median final distance to the goal, and the success rate"""
    states = np.random.default_rng(seed).uniform(-RANGE, RANGE, (N_TESTS, state_dim))
    goals = states[:, action_dim:2 * action_dim].copy()
    if dynamic_goal:
        goals = np.where(states[:, -1:] < 0.0, states[:, 2 * action_dim:3 * action_dim], goals)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(n_steps):
            st = torch.tensor(states, dtype=torch.float32, device=device)
            if policy == "mlp":
                acts = model(st).cpu().numpy()
            else:
                noise = torch.randn(len(states), action_dim, generator=generator).to(device)
                acts = model.sample(st, 10, noise).cpu().numpy()
            states[:, 0:action_dim] += acts
    model.train()
    err = np.linalg.norm(goals - states[:, 0:action_dim], axis=1)
    return {"success": float((err < 1.0).mean()), "err_mean": float(err.mean()), "err_median": float(np.median(err))}


# ---- regularizers (the formulas of the training scripts; "individual" holds A(s) or F(s) per state, "basis" J or the scores g)
def inner_loss(individual, basis, rank, state_dim, k):
    """InneR (Algorithm 1): cluster by active-subspace alignment, average over each region, penalize the shared null space"""
    B = individual.shape[0]
    with torch.no_grad():
        _, _, Vh = torch.linalg.svd(basis, full_matrices=False)
        P = torch.bmm(Vh[:, :rank].mT, Vh[:, :rank]).view(B, -1).double()
        _, knn = torch.topk(torch.cdist(P, P), min(k, B), dim=1, largest=False)
    return null_ratio(individual[knn].mean(dim=1), rank, state_dim)


def null_ratio(avg, rank, state_dim):
    """ratio of the n - r smallest eigenvalues of each averaged matrix to its detached trace (Equation 26), mean over the batch"""
    evals = torch.linalg.eigvalsh(avg).clamp(min=0.0)
    return (evals[:, :state_dim - rank].sum(dim=-1) / (evals.sum(dim=-1).detach() + 1e-8)).mean()


def regularizers(model, policy, states, P, state_dim, which=None, actions_hat=None):
    """the regularizers of a batch: InneR, squared Frobenius norm (trace of F), nuclear norm, InneR-C (k = B); only the one
    named by `which` during training (all four for the final diagnostics). Same operations as the training scripts."""
    if policy == "mlp":
        if actions_hat is None:
            actions_hat = model(states)
        J = torch.zeros((states.shape[0], action_dim, state_dim), device=states.device, dtype=states.dtype)
        for i in range(action_dim):
            J[:, i] = torch.autograd.grad(actions_hat[:, i], states, grad_outputs=torch.ones_like(actions_hat[:, i]), create_graph=True)[0]
        J = model.scramble.jacobian(J, states)       # with respect to the (scrambled) state the policy sees; identity otherwise
        individual, basis, rank = J.mT @ J, J, action_dim
        frob = lambda: J.pow(2).sum(dim=(-2, -1)).mean()
        nuc = lambda: torch.linalg.svdvals(J).sum(dim=-1).mean()
    else:
        F, g = get_fisher_information(model, states, P["N_ACTION_SAMPLES"], P["N_NOISE_SAMPLES"], P["N_SAMPLE_STEPS"])
        individual, basis, rank = F, g, P["FISHER_RANK"]
        frob = lambda: torch.einsum("bii->b", F).mean()
        nuc = lambda: (torch.linalg.svdvals(g).sum(dim=-1) / P["N_ACTION_SAMPLES"] ** 0.5).mean()
    fns = {"inner": lambda: inner_loss(individual, basis, rank, state_dim, P["K_NEIGHBORS"]), "frobenius": frob, "nuclear": nuc,
           "inner_c": lambda: null_ratio(individual.mean(dim=0, keepdim=True), rank, state_dim)}
    return {k: f() for k, f in fns.items() if which is None or k == which}


def main():
    a = parse()
    c = cell(a.policy, a.env, a.state_dim, a.n_demos, a.noise_correlation, a.scramble_condition)
    run_tag = tag(c, a.algo, a.seed)
    if a.tag_only:
        print(run_tag); return
    P = PARAMS[a.policy]
    dynamic_goal = a.env == "dynamic"
    assert a.state_dim >= 2 * action_dim and (not dynamic_goal or a.state_dim > 3 * action_dim)
    assert a.policy == "mlp" or P["FISHER_RANK"] < a.state_dim, "the Fisher rank must leave a null space (n = 8 is MLP only)"
    assert a.policy == "mlp" or a.state_dim <= 32, "the batched CUDA SVD is only fast up to 32 state entries"
    device = "cpu" if a.policy == "mlp" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(device == "cpu")

    data = make_dataset(a.state_dim, a.n_demos, dynamic_goal, a.seed, a.noise_correlation)
    model = (MLPPolicy if a.policy == "mlp" else DiffusionPolicy)(a.state_dim, hidden_dim, action_dim).to(device)
    if a.scramble_condition > 0:    # drawn from the seed, the same for every algorithm of the seed, saved with the weights
        model.scramble.randomize(a.scramble_condition, 5.0, a.scramble_bend, torch.Generator().manual_seed(a.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=P["LR"])
    loader = torch.utils.data.DataLoader(data, batch_size=P["BATCH_SIZE"], shuffle=True, generator=torch.Generator().manual_seed(a.seed))
    weight = {"bc": 0.0, "inner": P["GAMMA"], "frobenius": P["GAMMA_FROBENIUS"], "nuclear": P["GAMMA_NUCLEAR"], "inner_c": P["GAMMA"]}[a.algo]

    t0 = time.time()
    model.train()
    for epoch in range(P["EPOCH"] + 1):
        for x in loader:
            x = x.to(device)
            states, actions = x[:, :a.state_dim], x[:, a.state_dim:]
            actions_hat = None
            if a.policy == "mlp":
                states = states.requires_grad_(True)
                actions_hat = model(states)
                loss = model.mse_loss(actions, actions_hat)
            else:
                loss = model.diffusion_loss(actions, states)
            if a.algo != "bc":
                loss = loss + weight * regularizers(model, a.policy, states, P, a.state_dim, a.algo, actions_hat)[a.algo]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_s = time.time() - t0

    # evaluation, and the regularizer values of the final model on the first 200 training states (E1 plots them)
    metrics = evaluate(model, a.policy, a.state_dim, dynamic_goal, device, a.seed)
    states = data[:P["BATCH_SIZE"], :a.state_dim].to(device).requires_grad_(a.policy == "mlp")
    diag = {k: float(v.detach()) for k, v in regularizers(model, a.policy, states, P, a.state_dim).items()}
    print(f"[{run_tag}] {metrics} regularizers {diag} ({train_s:.0f}s)", flush=True)

    os.makedirs(os.path.join(a.out_dir, "models"), exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, os.path.join(a.out_dir, "models", run_tag + ".pt"))
    rec = {"tag": run_tag, "experiments": experiments_of(c), "policy": a.policy, "algo": a.algo, "env": a.env,
           "state_dim": a.state_dim, "n_demos": a.n_demos, "seed": a.seed, "rho": a.noise_correlation,
           "cond": a.scramble_condition, "bend": a.scramble_bend if a.scramble_condition > 0 else 0.0,
           **metrics, "jacobian_norm": diag["frobenius"], "null_ratio": diag["inner"], "nuclear_norm": diag["nuclear"],
           "null_ratio_batch": diag["inner_c"], "params": P, "train_s": train_s, "device": device,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "run_experiment.py"}
    with open(os.path.join(a.out_dir, "results.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
