import sys
import torch
import pickle
from models import DiffusionPolicy
from config import state_dim, action_dim, hidden_dim, scramble_state, scramble_condition, scramble_shift, scramble_bend
from torch.utils.data import Dataset, DataLoader


# import dataset for training
class MyData(Dataset):

    def __init__(self):
        self.data = pickle.load(open("datasets/dataset.pkl", "rb"))
        self.data = torch.tensor(self.data, dtype=torch.float32)
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


# estimate the Fisher information matrix F(s) of the policy with respect to state
# this follows Equations (13), (17), and (21) in the paper
def get_fisher_information(model, states, n_action_samples, n_noise_samples, n_sample_steps):
    batch_size, state_dim = states.shape

    # sample actions from the current policy at each state, a ~ pi(. | s)
    # we never differentiate through the sampling process
    with torch.no_grad():
        states_a = states.repeat_interleave(n_action_samples, dim=0)
        actions_a = model.sample(states_a, n_sample_steps)

    # make one copy of each (state, action) pair for every (t, eps) sample
    # the state copies are the inputs that we differentiate with respect to
    states_rep = states_a.repeat_interleave(n_noise_samples, dim=0).detach().requires_grad_(True)
    actions_rep = actions_a.repeat_interleave(n_noise_samples, dim=0)
    t = torch.multinomial(model.score_t_probs, len(actions_rep), replacement=True)
    noise = torch.randn_like(actions_rep)

    # ELBO-weighted denoising loss of each copy, lambda_t * l(a, s; t, eps) in Equation (20),
    # divided by the probability q(t) of drawing step t (importance weight, see score_t_probs in models.py)
    weighted_error = model.elbo_weights[t] / model.score_t_probs[t] * model.denoising_error(actions_rep, states_rep, t, noise)

    # score of each sampled action with respect to state (Equation 21), with t drawn from q(t) instead of uniformly
    # g(a, s) = -T * grad_s E_{t ~ U, eps}[lambda_t * l(a, s; t, eps)] = -grad_s E_{t ~ q, eps}[lambda_t * l(a, s; t, eps) / q(t)]
    # (with q(t) = 1 / T this is exactly the formula in the paper)
    grad_s = torch.autograd.grad(weighted_error.sum(), states_rep, create_graph=True)[0]
    grad_s = grad_s.view(batch_size, n_action_samples, n_noise_samples, state_dim)
    g = -grad_s.mean(dim=2)
    # the policy sees the scrambled state z (models.py), so convert the score to the derivative with respect to z
    # (chain rule; the identity without scrambling)
    g = model.scramble.jacobian(g, states)

    # Fisher information: average outer product of the score over the sampled actions (Equation 17)
    F = g.mT @ g / n_action_samples
    return F, g


# InneR loss of a batch of averaged inner products (Equation 26 in the paper)
# for each averaged matrix, the ratio of its n - r smallest eigenvalues to the total of all eigenvalues, then the mean over the batch
# F is symmetric, so we use eigvalsh (cheaper than an SVD, eigenvalues in ascending order)
# the total is detached, so the loss cannot be lowered by growing the largest eigenvalues
def inner_loss(F_avg, null_space_dim):
    evals = torch.linalg.eigvalsh(F_avg).clamp(min=0.0)
    bottom = evals[:, :null_space_dim].sum(dim=-1)
    total = evals.sum(dim=-1)
    return (bottom / (total.detach() + 1e-8)).mean()


# train model
def train_model(savenumber, state_dim, hidden_dim, action_dim, device, seed=0):

    ## training parameters
    print(f"[-] training diffusion model {savenumber} on device: {device}")
    EPOCH = 1000
    LR = 3e-3

    # fix the seed so every model gets the same initial weights and batch order
    torch.manual_seed(seed)
    # on the CPU, also use deterministic algorithms so repeated runs give the same model
    # (the backward pass of the neighborhood average otherwise uses atomic adds across threads;
    # on the GPU it is already deterministic)
    torch.use_deterministic_algorithms(device == "cpu")

    # initialize model and optimizer
    model = DiffusionPolicy(state_dim, hidden_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    # scramble the state that the policy sees (config.py) with a random transformation drawn from the seed of the run:
    # the same for every model of the run, and saved with the weights so that the test script applies it too
    if scramble_state:
        model.scramble.randomize(scramble_condition, scramble_shift, scramble_bend, torch.Generator().manual_seed(seed))
        print(f"[-] scrambling the state ({model.scramble.describe()})")

    # initialize dataset
    # (the shuffle has its own random generator, so the random draws of InneR do not change the batch order)
    train_data = MyData()
    # fixed batch size, so that the number of gradient steps grows with the dataset (as in train_mlp_policy.py)
    BATCH_SIZE = 200
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,
                           generator=torch.Generator().manual_seed(seed))

    ## inner product parameters
    # rank r of the Fisher information: an upper estimate of the number of state directions the action distribution depends on
    # the action dimension m is a lower bound (exact for a deterministic policy), but a stochastic policy can also change the
    # shape of its distribution, so r >= m; overestimating r only weakens the regularizer (fewer eigenvalues are penalized),
    # while underestimating it penalizes directions the policy needs, so we use the conservative choice r = 2m
    FISHER_RANK = 2 * action_dim
    assert FISHER_RANK < state_dim, "the Fisher rank must leave at least one null space direction"
    # number of actions sampled from the policy at each state to estimate the Fisher (at least r, so F(s) can have rank r)
    N_ACTION_SAMPLES = FISHER_RANK
    assert N_ACTION_SAMPLES >= FISHER_RANK, "need at least r sampled actions for F(s) to have rank r"
    # number of (t, eps) samples used to estimate the score of each sampled action
    N_NOISE_SAMPLES = 4
    # number of DDIM steps used to sample those actions
    # (they only need to be plausible draws from pi(. | s); 1 step matched or beat 3 steps in the 30 Oct 2026 sweeps and is 15% faster)
    N_SAMPLE_STEPS = 1
    # number of nearest neighbors to average across
    K_NEIGHBORS = 8
    # weight of the InneR regularization loss, shared by InneR-C, which is InneR with k = B
    GAMMA = 0.3

    ## Fisher regularization baselines (the stochastic analogues of the Jacobian norms in Section IV of the paper)
    # both penalize each F(s) on its own: the trace of F(s) is the sum of its eigenvalues (like the squared Frobenius
    # norm of J, which is the trace of A), and the nuclear norm of the scores g / sqrt(N_a) is the sum of the square
    # roots of the eigenvalues of F(s) (like the nuclear norm of J)
    GAMMA_FROBENIUS = 1e-4
    GAMMA_NUCLEAR = 1e-3

    # main training loop
    null_space_dim = state_dim - FISHER_RANK
    for epoch in range(EPOCH+1):
        loss_diffusion_avg = 0
        loss_inner_avg = 0
        loss_frobenius_avg = 0
        loss_nuclear_avg = 0
        loss_inner_c_avg = 0
        for batch, x in enumerate(train_set):

            # send batch data to the target device
            x = x.to(device)

            # collect the demonstrated states and actions
            states = x[:, :state_dim]
            actions = x[:, state_dim:]
            batch_size = states.shape[0]

            # default diffusion policy loss
            loss_diffusion = model.diffusion_loss(actions, states)

            # regularizers: InneR (model 1), Fisher trace (model 2), Fisher nuclear norm (model 3), and InneR-C (model 4)
            # every model computes all four on logging epochs, to report them; the Fisher estimate is shared, and on the
            # other epochs each model only computes its own loss
            loss_inner = torch.zeros((), device=device)
            loss_frobenius = torch.zeros((), device=device)
            loss_nuclear = torch.zeros((), device=device)
            loss_inner_c = torch.zeros((), device=device)
            if savenumber != 0 or epoch % 100 == 0:

                # get the Fisher information matrix F(s) and the scores g for all states
                F_individual, g = get_fisher_information(
                        model, states, N_ACTION_SAMPLES, N_NOISE_SAMPLES, N_SAMPLE_STEPS)

                # Fisher regularization baselines, averaged over the batch
                # trace of F(s), i.e. the sum of the eigenvalues of F(s)
                if savenumber == 2 or epoch % 100 == 0:
                    loss_frobenius = torch.einsum("bii->b", F_individual).mean()
                # nuclear norm of g / sqrt(N_a), i.e. the sum of the square roots of the eigenvalues of F(s) = g^T g / N_a
                if savenumber == 3 or epoch % 100 == 0:
                    loss_nuclear = (torch.linalg.svdvals(g).sum(dim=-1) / N_ACTION_SAMPLES ** 0.5).mean()

                # InneR without clustering (InneR-C): Algorithm 1 with k = B, so the one region is the whole batch
                # and the loss penalizes the shared null space of its average Fisher information
                if savenumber == 4 or epoch % 100 == 0:
                    loss_inner_c = inner_loss(F_individual.mean(dim=0, keepdim=True), null_space_dim)

            # inner product regularization (InneR) with the Fisher information
            if savenumber == 1 or epoch % 100 == 0:

                # for each F matrix, find the K-nearest neighbors in active space
                with torch.no_grad():
                    # leading eigenvectors of F (rows of Vh), from the SVD of the scores
                    _, _, Vh = torch.linalg.svd(g, full_matrices=False)
                    Vh = Vh[:, :FISHER_RANK]
                    # compute projection matrices directly (P = Vh^T * Vh)
                    P = torch.bmm(Vh.mT, Vh)
                    # flatten to compute pairwise Grassmann distances
                    # (float64 because float32 cdist has ~2e-3 cancellation error)
                    P_flat = P.view(batch_size, -1).double()
                    dists = torch.cdist(P_flat, P_flat)
                    # find K-Nearest Neighbors based on active subspace alignment
                    current_k = min(K_NEIGHBORS, batch_size)
                    _, knn_indices = torch.topk(dists, current_k, dim=1, largest=False)

                # active subspace regularization on each aligned neighborhood
                F_neighborhood = F_individual[knn_indices].mean(dim=1)

                # InneR loss: ratio of the n - r smallest eigenvalues of each region to the total of all eigenvalues
                loss_inner = inner_loss(F_neighborhood, null_space_dim)

            # get overall loss
            if savenumber == 0:
                loss = loss_diffusion
            if savenumber == 1:
                loss = loss_diffusion + GAMMA * loss_inner
            if savenumber == 2:
                loss = loss_diffusion + GAMMA_FROBENIUS * loss_frobenius
            if savenumber == 3:
                loss = loss_diffusion + GAMMA_NUCLEAR * loss_nuclear
            if savenumber == 4:
                loss = loss_diffusion + GAMMA * loss_inner_c

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss averages (kept on the device: .item() would make the CPU wait for the GPU at every step)
            loss_diffusion_avg += loss_diffusion.detach()
            loss_inner_avg += loss_inner.detach()
            loss_frobenius_avg += loss_frobenius.detach()
            loss_nuclear_avg += loss_nuclear.detach()
            loss_inner_c_avg += loss_inner_c.detach()

        # print average loss and save model
        if epoch % 100 == 0:
            n_batches = len(train_set)
            print(f"Epoch: {epoch} | Diffusion: {loss_diffusion_avg/n_batches:.4f} | InneR: {loss_inner_avg/n_batches:.4f} | "
                  f"Frobenius: {loss_frobenius_avg/n_batches:.4f} | Nuclear: {loss_nuclear_avg/n_batches:.4f} | "
                  f"InneR-C: {loss_inner_c_avg/n_batches:.4f}")
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, "models/diffusion_weights" + str(savenumber))


# train models
if __name__ == "__main__":

    # CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # seed of this run, from the command line (default 0); bash.sh passes the same seed to every script
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # train baseline (standard diffusion policy, 0) and ours (InneR with the Fisher information, 1)
    train_model(0, state_dim, hidden_dim, action_dim, device, seed)
    train_model(1, state_dim, hidden_dim, action_dim, device, seed)

    # # Fisher regularization baselines, trace (Frobenius, 2) and nuclear norm (3), and InneR without clustering (InneR-C, 4)
    # # we typically do not test these for spot checks, but they should be used for comprehensive evaluations
    # # (to roll them out, add their names to MODEL_NAMES in test_diffusion_policy.py)
    # train_model(2, state_dim, hidden_dim, action_dim, device, seed)
    # train_model(3, state_dim, hidden_dim, action_dim, device, seed)
    # train_model(4, state_dim, hidden_dim, action_dim, device, seed)
