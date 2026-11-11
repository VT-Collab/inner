import os
import torch
import argparse
from models import DiffusionPolicy
from config import dims
from train_mlp_policy import MyData
from torch.utils.data import DataLoader


def get_fisher_information(model, states, n_action_samples, n_noise_samples, n_sample_steps):
    """Fisher information F(s) of the policy wrt state, and the scores g (Equations 13, 17, 21)"""
    batch_size, state_dim = states.shape

    with torch.no_grad():
        states_a = states.repeat_interleave(n_action_samples, dim=0)
        actions_a = model.sample(states_a, n_sample_steps)

    states_rep = states_a.repeat_interleave(n_noise_samples, dim=0).detach().requires_grad_(True)
    actions_rep = actions_a.repeat_interleave(n_noise_samples, dim=0)
    t = torch.multinomial(model.score_t_probs, len(actions_rep), replacement=True)
    noise = torch.randn_like(actions_rep)

    weighted_error = model.elbo_weights[t] / model.score_t_probs[t] * model.denoising_error(actions_rep, states_rep, t, noise)

    # score of each sampled action with respect to state (Equation 21), with t drawn from q(t)
    grad_s = torch.autograd.grad(weighted_error.sum(), states_rep, create_graph=True)[0]
    grad_s = grad_s.view(batch_size, n_action_samples, n_noise_samples, state_dim)
    g = -grad_s.mean(dim=2)

    # Fisher information: average outer product of the score over the sampled actions (Equation 17)
    F = g.mT @ g / n_action_samples
    return F, g


def train_model(savenumber, args, device):
    """train one model: 0 diffusion policy, 1 InneR, 2 trace of F, 3 nuclear norm of the scores"""

    print(f"[-] training diffusion model {savenumber} on device: {device}")
    LR = 1e-3

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(device == "cpu")

    train_data = MyData(args.dataset)
    state_dim, action_dim = dims(train_data.stats)

    model = DiffusionPolicy(state_dim, args.hidden_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # off by default: cosine learning-rate decay, and an EMA of the weights (what gets saved)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps) if args.cosine else None
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema)) if args.ema else None
    model.train()

    run = None
    if args.wandb:
        import wandb
        # <set>/seed<N>/diffusion_weights<k>, grouped by set
        set_name, seed_dir = os.path.normpath(args.save_dir).split(os.sep)[-2:]
        run = wandb.init(project=args.wandb, name=f"{set_name}/{seed_dir}/diffusion_weights{savenumber}", group=set_name,
                         config={**vars(args), "model": savenumber, **{k: train_data.stats.get(k) for k in
                                 ("state_space", "action_space", "target_steps", "labels")}, "n_pairs": len(train_data)})
    recent = []

    # (the shuffle has its own random generator, so the random draws of InneR do not change the batch order)
    BATCH_SIZE = min(args.batch_size, len(train_data))
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                           generator=torch.Generator().manual_seed(args.seed))

    # active-space rank of InneR (--fisher_rank, default 2m); the action samples stay 2m for every model
    FISHER_RANK = args.fisher_rank or 2 * action_dim
    # with no null space (e.g. the pot alone in a cartesian state) only the baseline trains, and InneR is not logged
    has_null_space = FISHER_RANK < state_dim
    assert has_null_space or savenumber != 1, "the Fisher rank must leave at least one null space direction"
    N_ACTION_SAMPLES = 2 * action_dim
    N_NOISE_SAMPLES = 4
    N_SAMPLE_STEPS = 3
    K_NEIGHBORS = 8
    GAMMA = args.gamma

    GAMMA_FROBENIUS = 1e-4
    GAMMA_NUCLEAR = 1e-3

    null_space_dim = state_dim - FISHER_RANK
    step = 0
    while step < args.steps:
        loss_diffusion_avg = 0
        n_batches = 0
        for batch, x in enumerate(train_set):
            log = (step + 1) % args.log_every == 0 or step + 1 == args.steps

            x = x.to(device)

            states = x[:, :state_dim]
            actions = x[:, state_dim:]
            batch_size = states.shape[0]

            loss_diffusion = model.diffusion_loss(actions, states)

            loss_inner = torch.zeros((), device=device)
            loss_frobenius = torch.zeros((), device=device)
            loss_nuclear = torch.zeros((), device=device)
            if savenumber != 0 or log:

                F_individual, g = get_fisher_information(
                        model, states, N_ACTION_SAMPLES, N_NOISE_SAMPLES, N_SAMPLE_STEPS)

                loss_frobenius = torch.einsum("bii->b", F_individual).mean()
                loss_nuclear = (torch.linalg.svdvals(g).sum(dim=-1) / N_ACTION_SAMPLES ** 0.5).mean()

            if savenumber == 1 or (log and has_null_space):

                with torch.no_grad():
                    _, _, Vh = torch.linalg.svd(g, full_matrices=False)
                    Vh = Vh[:, :FISHER_RANK]
                    P = torch.bmm(Vh.mT, Vh)
                    P_flat = P.view(batch_size, -1).double()
                    dists = torch.cdist(P_flat, P_flat)
                    current_k = min(K_NEIGHBORS, batch_size)
                    _, knn_indices = torch.topk(dists, current_k, dim=1, largest=False)

                F_neighborhood = F_individual[knn_indices].mean(dim=1)

                # InneR loss, ratio of the n - r smallest eigenvalues to the detached total
                evals = torch.linalg.eigvalsh(F_neighborhood).clamp(min=0.0)
                bottom = evals[:, :null_space_dim].sum(dim=-1)
                total = evals.sum(dim=-1)
                loss_inner = (bottom / (total.detach() + 1e-8)).mean()

            if savenumber == 0:
                loss = loss_diffusion
            if savenumber == 1:
                loss = loss_diffusion + GAMMA * loss_inner
            if savenumber == 2:
                loss = loss_diffusion + GAMMA_FROBENIUS * loss_frobenius
            if savenumber == 3:
                loss = loss_diffusion + GAMMA_NUCLEAR * loss_nuclear

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1

            loss_diffusion_avg += loss_diffusion.item()
            n_batches += 1
            if run is not None:
                recent.append(loss_diffusion.item())
                if step % 100 == 0:
                    run.log({"diffusion": sum(recent) / len(recent), "loss": loss.item()}, step=step)
                    recent = []

            if log:
                print(f"Step: {step} | Diffusion: {loss_diffusion_avg/n_batches:.4f} | InneR: {loss_inner.item():.4f} | "
                      f"Frobenius: {loss_frobenius.item():.4f} | Nuclear: {loss_nuclear.item():.4f}")
                loss_diffusion_avg, n_batches = 0, 0
                save(ema.module if ema is not None else model, train_data.stats, args, f"{args.save_dir}/diffusion_weights{savenumber}")
                if run is not None:
                    run.log({"inner": loss_inner.item(), "frobenius": loss_frobenius.item(), "nuclear": loss_nuclear.item()}, step=step)
            if step == args.steps:
                break
    if run is not None:
        run.finish()


def save(model, stats, args, path):
    """save the weights with the normalization and model size, so deploy.py can rebuild the policy"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"kind": "diffusion", "hidden_dim": args.hidden_dim, "stats": stats,
                "model": {k: v.cpu() for k, v in model.state_dict().items()}}, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/dataset.pkl")
    parser.add_argument("--save_dir", default="models")
    # 0: diffusion policy, 1: InneR with the Fisher, 2: trace of F, 3: nuclear norm of the scores
    parser.add_argument("--models", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--fisher_rank", type=int, default=None, help="InneR's active-space rank (default 2 * action dim)")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA decay of the saved weights, e.g. 0.999 (0: off)")
    parser.add_argument("--cosine", action="store_true", help="cosine learning-rate decay over --steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", default=None, help="wandb project for the loss curves (off by default)")
    args = parser.parse_args()

    for savenumber in args.models:
        train_model(savenumber, args, args.device)
