# MLP training on the datasets/data trajectories.
# Called by experiments/run_experiment.py; experiments/mlp_cells.py owns output tags.
# Model 0: BC; 1: InneR; 2: squared Jacobian Frobenius norm; 3: nuclear norm; 4: InneR-C.
# E4 uses a fixed seed-specific scramble saved with the weights.
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import (state_dim, action_dim, experiments, n_train_demos, default_seed, default_experiment, default_n_demos,
                    e4_condition, e4_bend, e4_shift, hidden_dim, batch_size, learning_rate, epochs, k_neighbors, gamma,
                    gamma_frobenius, gamma_nuclear, eps, input_dim, data_key, run_name)
from .models import MLPPolicy, StateScramble
from .data import training_inputs, provenance

MODEL_NAMES = {0: "BC", 1: "InneR", 2: "Frobenius", 3: "nuclear", 4: "InneR-C"}
WEIGHTS = {0: 0.0, 1: gamma, 2: gamma_frobenius, 3: gamma_nuclear, 4: gamma}
# CPU threads of this process
TORCH_THREADS = 2
# training states used for the diagnostics of each final model
DIAG_STATES = 200


## data
# the first n_demos demonstrations of the dataset, with the experiment's extra entries appended to the 19D state;
# states and actions are z-scored with the statistics of these demos (the e2 noise entries are not normalized)
def load_training_data(experiment, n_demos):
    inputs, actions = training_inputs(experiment, n_demos)
    group = data_key(experiment)
    rows = len(inputs)

    # population mean and std of each entry over the n_demos demos (std below 1e-6 -> 1), in float64
    stats = {}
    for key, values in (("state", inputs[:rows]), ("action", actions[:rows])):
        mean, std = values.mean(0), values.std(0)
        std = np.where(std < 1e-6, 1.0, std)
        if key == "state" and group == "e2":
            mean[state_dim:], std[state_dim:] = 0.0, 1.0
        stats[f"{key}_mean"], stats[f"{key}_std"] = mean.astype(np.float32), std.astype(np.float32)
    # e1 with fewer than three other drones keeps the first entries of the three
    d = input_dim(experiment)
    stats["state_mean"], stats["state_std"] = stats["state_mean"][:d], stats["state_std"][:d]

    x = inputs[:rows, :d].astype(np.float32)
    y = actions[:rows].astype(np.float32)
    x = torch.from_numpy(((x - stats["state_mean"]) / stats["state_std"]).astype(np.float32))
    y = torch.from_numpy(((y - stats["action_mean"]) / stats["action_std"]).astype(np.float32))
    return x, y, stats


## inner products
# policy Jacobian J [B, m, n] of the normalized action with respect to the normalized state
def policy_jacobian(states, predictions, create_graph=True):
    return torch.stack([
        torch.autograd.grad(predictions[:, i].sum(), states, create_graph=create_graph)[0]
        for i in range(action_dim)
    ], dim=1)


# the batched GPU SVD of J [B, m, n] is fast only up to n = 32; from 33 entries on, the row space comes from the m x m
# Gram matrix J J^T in float64 instead
GRAM_PATH_MIN_DIM = 33


# orthogonal projection V V^T onto the row space of each J, from its m right singular vectors (Equation 22)
def row_space_projector(J):
    if J.shape[-1] < GRAM_PATH_MIN_DIM:
        _, _, vh = torch.linalg.svd(J, full_matrices=False)
        return vh.transpose(-1, -2) @ vh
    Jd = J.double()
    _, U = torch.linalg.eigh(Jd @ Jd.transpose(-1, -2))
    vh = U.transpose(-1, -2) @ Jd
    vh = vh / torch.linalg.vector_norm(vh, dim=-1, keepdim=True).clamp_min(1e-300)
    return (vh.transpose(-1, -2) @ vh).to(J.dtype)


# sum of the singular values of each J (differentiable)
def nuclear_norms(J):
    if J.shape[-1] < GRAM_PATH_MIN_DIM:
        return torch.linalg.svdvals(J).sum(dim=1)
    Jd = J.double()
    eig = torch.linalg.eigvalsh(Jd @ Jd.transpose(-1, -2)).clamp_min(1e-30)
    return eig.sqrt().sum(dim=1).to(J.dtype)


# eigenvalues in ascending order; float32 eigh can fail to converge when most eigenvalues are close to zero (the shared null
# space that InneR creates), and only then is it repeated in float64
def eigvalsh(matrices):
    try:
        return torch.linalg.eigvalsh(matrices)
    except torch.linalg.LinAlgError:
        return torch.linalg.eigvalsh(matrices.double()).to(matrices.dtype)


# InneR loss of a batch of averaged inner products (Equation 26 in the paper): for each averaged matrix, the ratio of its
# n - m smallest eigenvalues to its trace, then the mean over the batch; the trace is detached, so the loss cannot be
# lowered by growing the largest eigenvalues
def inner_loss(A_avg):
    A_avg = 0.5 * (A_avg + A_avg.transpose(-1, -2))
    null_eigs = eigvalsh(A_avg)[:, :A_avg.shape[-1] - action_dim].clamp_min(0.0)
    trace = A_avg.diagonal(dim1=-2, dim2=-1).sum(dim=-1).detach().clamp_min(eps)
    return (null_eigs.sum(dim=-1) / trace).mean()


# InneR (Algorithm 1): average A = J^T M J (M = I) over each state's k nearest neighbors by the chordal distance between
# the row spaces of J (Equation 23), then the loss of Equation 26
def inner_regularizer(states, predictions, k):
    J = policy_jacobian(states, predictions)
    with torch.no_grad():
        P = row_space_projector(J.detach())
        distances = torch.cdist(P.flatten(1), P.flatten(1))
        nearest = distances.topk(min(k, states.shape[0]), largest=False).indices
    A = J.transpose(-1, -2) @ J
    return inner_loss(A[nearest].mean(dim=1))


# InneR-C: the same loss on the average of A over the whole batch
def inner_c_regularizer(states, predictions):
    J = policy_jacobian(states, predictions)
    return inner_loss((J.transpose(-1, -2) @ J).mean(dim=0, keepdim=True))


def regularizer(savenumber, states, predictions):
    if savenumber == 1:
        return inner_regularizer(states, predictions, k_neighbors)
    if savenumber == 4:
        return inner_c_regularizer(states, predictions)
    J = policy_jacobian(states, predictions)
    if savenumber == 2:
        return J.square().sum(dim=(1, 2)).mean()
    return nuclear_norms(J).mean()


# e4: the scramble of the normalized input, drawn from the training seed; None for the other experiments
def make_scramble(experiment, seed):
    if experiment != "e4":
        return None
    generator = torch.Generator().manual_seed(seed)
    return StateScramble(input_dim(experiment)).randomize(e4_condition, e4_shift, e4_bend, generator)


## training
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(savenumber, x, y, seed, device):
    print(f"[-] training MLP model {savenumber} ({MODEL_NAMES[savenumber]}) on device: {device}")
    # the same initial weights and batch order for every model of a seed
    seed_everything(seed)
    model = MLPPolicy(x.shape[1], hidden_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weight = WEIGHTS[savenumber]

    # batches of a shuffled DataLoader(batch_size, shuffle=True, generator=manual_seed(seed)), gathered on the device:
    # every epoch draws the iterator's base seed, the permutation of the rows, and the sampler's unused next permutation
    generator = torch.Generator().manual_seed(seed)
    xd, yd = x.to(device), y.to(device)
    rows = len(x)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        perm = torch.randperm(rows, generator=generator)
        torch.randperm(rows, generator=generator)
        mse_sum = torch.zeros((), device=device)
        reg_sum = torch.zeros((), device=device)
        for idx in perm.split(batch_size):
            idx = idx.to(device)
            xb, yb = xd[idx], yd[idx]
            if savenumber != 0:
                xb = xb.requires_grad_(True)
            pred = model(xb)
            mse = F.mse_loss(pred, yb)
            if savenumber == 0:
                reg, loss = torch.zeros((), device=device), mse
            else:
                reg = regularizer(savenumber, xb, pred)
                loss = mse + weight * reg
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            mse_sum += mse.detach() * len(idx)
            reg_sum += reg.detach() * len(idx)
        if not (torch.isfinite(mse_sum) and torch.isfinite(reg_sum)):
            raise FloatingPointError(f"non-finite loss of model {savenumber} at epoch {epoch}")
        history.append({"epoch": epoch, "mse": mse_sum.item() / rows, "regularizer": reg_sum.item() / rows})
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch: {epoch} | MSE: {history[-1]['mse']:.6f} | regularizer: {history[-1]['regularizer']:.6f}")
    return model, history


# the regularized quantities of the final model on DIAG_STATES training states, in the coordinates the policy sees (after
# the e4 scramble): the squared Frobenius and nuclear norms of J, and the InneR loss (null-space ratio) with the training
# k and over the whole set (InneR-C)
def diagnostics(model, x, seed, n_demos, device, scramble=None):
    idx = np.sort(np.random.default_rng([seed, n_demos, 7]).choice(len(x), DIAG_STATES, replace=False))
    s = x[idx].to(device)
    if scramble is not None:
        with torch.no_grad():
            s = scramble.to(device)(s)
    s = s.requires_grad_(True)
    model.eval()
    pred = model(s)
    J = policy_jacobian(s, pred)
    return {"jacobian_norm": float(J.square().sum(dim=(1, 2)).mean()),
            "nuclear_norm": float(nuclear_norms(J.detach()).mean()),
            "null_ratio": float(inner_regularizer(s, pred, k_neighbors).detach()),
            "null_ratio_batch": float(inner_c_regularizer(s, pred).detach())}


# the weights with everything the test needs: the normalization statistics and the e4 scramble
def checkpoint(model, stats, scramble, savenumber, experiment, n_demos, seed, history, diag):
    return {"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "normalization": {k: torch.from_numpy(v) for k, v in stats.items()},
            "scramble_state_dict": None if scramble is None else {k: v.cpu() for k, v in scramble.state_dict().items()},
            "name": MODEL_NAMES[savenumber], "weight": WEIGHTS[savenumber], "experiment": experiment,
            "n_demos": n_demos, "seed": seed, "input_dim": len(stats["state_mean"]), "hidden_dim": hidden_dim,
            "epochs": epochs, "history": history, "diagnostics": diag,
            "dataset": provenance(experiment)}

