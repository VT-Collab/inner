import os
import torch
import pickle
import argparse
from models import MLPPolicy
from config import dims
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    """dataset of normalized (state, action) rows"""

    def __init__(self, path):
        dataset = pickle.load(open(path, "rb"))
        self.data = torch.tensor(dataset["data"], dtype=torch.float32)
        self.stats = dataset["stats"]
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


def train_model(savenumber, args, device):
    """train one model: 0 BC, 1 InneR, 2 Frobenius norm of J, 3 nuclear norm of J"""

    print(f"[-] training MLP model {savenumber} on device: {device}")
    LR = 1e-3

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(device == "cpu")

    train_data = MyData(args.dataset)
    state_dim, action_dim = dims(train_data.stats)

    model = MLPPolicy(state_dim, args.hidden_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    BATCH_SIZE = 200
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                           generator=torch.Generator().manual_seed(args.seed))

    # inner product in action space
    M = torch.eye(action_dim, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    L = torch.linalg.cholesky(M)
    K_NEIGHBORS = 8
    GAMMA = args.gamma

    ## Jacobian regularization baselines (Section IV of the paper)
    GAMMA_FROBENIUS = 0.01
    GAMMA_NUCLEAR = 0.01

    null_space_dim = state_dim - action_dim
    step = 0
    while step < args.steps:
        loss_mse_avg = 0
        loss_inner_avg = 0
        loss_frobenius_avg = 0
        loss_nuclear_avg = 0
        n_batches = 0
        for batch, x in enumerate(train_set):
            log = (step + 1) % args.log_every == 0 or step + 1 == args.steps

            x = x.to(device)

            states = x[:, :state_dim].requires_grad_(True)
            actions = x[:, state_dim:]
            batch_size = states.shape[0]

            actions_hat = model(states)
            loss_mse = model.mse_loss(actions, actions_hat)

            loss_inner = torch.zeros((), device=device)
            loss_frobenius = torch.zeros((), device=device)
            loss_nuclear = torch.zeros((), device=device)
            if savenumber != 0 or log:

                J = torch.zeros((batch_size, action_dim, state_dim), device=states.device, dtype=states.dtype)
                for i in range (action_dim):
                    J[:, i] = torch.autograd.grad(
                            actions_hat[:, i],
                            states,
                            grad_outputs=torch.ones_like(actions_hat[:, i]),
                            create_graph=True)[0]

                # A = J^T * M * J = J_tilde^T * J_tilde
                L_batch = L[:batch_size]
                J_tilde = torch.bmm(L_batch.transpose(-1, -2), J)
                A_individual = J_tilde.mT @ J_tilde

                loss_frobenius = J_tilde.pow(2).sum(dim=(-2, -1)).mean()
                loss_nuclear = torch.linalg.svdvals(J_tilde).sum(dim=-1).mean()

            if savenumber == 1 or log:

                with torch.no_grad():
                    _, _, Vh = torch.linalg.svd(J_tilde, full_matrices=False)
                    P = torch.bmm(Vh.mT, Vh)
                    P_flat = P.view(batch_size, -1).double()
                    dists = torch.cdist(P_flat, P_flat)
                    current_k = min(K_NEIGHBORS, batch_size)
                    _, knn_indices = torch.topk(dists, current_k, dim=1, largest=False)

                A_neighborhood = A_individual[knn_indices].mean(dim=1)

                # InneR loss, ratio of the n - m smallest eigenvalues to the detached total
                evals = torch.linalg.eigvalsh(A_neighborhood).clamp(min=0.0)
                bottom = evals[:, :null_space_dim].sum(dim=-1)
                total = evals.sum(dim=-1)
                loss_inner = (bottom / (total.detach() + 1e-8)).mean()

            if savenumber == 0:
                loss = loss_mse
            if savenumber == 1:
                loss = loss_mse + GAMMA * loss_inner
            if savenumber == 2:
                loss = loss_mse + GAMMA_FROBENIUS * loss_frobenius
            if savenumber == 3:
                loss = loss_mse + GAMMA_NUCLEAR * loss_nuclear

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            loss_mse_avg += loss_mse.item()
            loss_inner_avg += loss_inner.item()
            loss_frobenius_avg += loss_frobenius.item()
            loss_nuclear_avg += loss_nuclear.item()
            n_batches += 1

            if log:
                print(f"Step: {step} | MSE: {loss_mse_avg/n_batches:.4f} | InneR: {loss_inner.item():.4f} | "
                      f"Frobenius: {loss_frobenius.item():.4f} | Nuclear: {loss_nuclear.item():.4f}")
                loss_mse_avg, n_batches = 0, 0
                save(model, train_data.stats, args, f"{args.save_dir}/mlp_weights{savenumber}")
            if step == args.steps:
                break


def save(model, stats, args, path):
    """save the weights with the normalization and model size, so deploy.py can rebuild the policy"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"kind": "mlp", "hidden_dim": args.hidden_dim, "stats": stats,
                "model": {k: v.cpu() for k, v in model.state_dict().items()}}, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/dataset.pkl")
    parser.add_argument("--save_dir", default="models")
    # 0: BC, 1: InneR, 2: Frobenius norm of J, 3: nuclear norm of J
    parser.add_argument("--models", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    for savenumber in args.models:
        train_model(savenumber, args, args.device)
