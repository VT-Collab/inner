import sys
import torch
import pickle
from models import MLPPolicy
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


# InneR loss of a batch of averaged inner products (Equation 26 in the paper)
# for each averaged matrix, the ratio of its n - m smallest eigenvalues to the total of all eigenvalues, then the mean over the batch
# A is symmetric, so we use eigvalsh (cheaper than an SVD, eigenvalues in ascending order)
# the total is detached, so the loss cannot be lowered by growing the largest eigenvalues
def inner_loss(A_avg, null_space_dim):
    evals = torch.linalg.eigvalsh(A_avg).clamp(min=0.0)
    bottom = evals[:, :null_space_dim].sum(dim=-1)
    total = evals.sum(dim=-1)
    return (bottom / (total.detach() + 1e-8)).mean()


# train model
def train_model(savenumber, state_dim, hidden_dim, action_dim, device, seed=0):

    ## training parameters
    print(f"[-] training MLP model {savenumber} on device: {device}")
    EPOCH = 200
    LR = 1e-3

    # fix the seed so every model gets the same initial weights and batch order
    torch.manual_seed(seed)
    # on the CPU, also use deterministic algorithms so repeated runs give the same model
    # (the backward pass of the neighborhood average otherwise uses atomic adds across threads;
    # on the GPU it is already deterministic)
    torch.use_deterministic_algorithms(device == "cpu")

    # initialize model and optimizer
    model = MLPPolicy(state_dim, hidden_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()

    # scramble the state that the policy sees (config.py) with a random transformation drawn from the seed of the run:
    # the same for every model of the run, and saved with the weights so that the test script applies it too
    if scramble_state:
        model.scramble.randomize(scramble_condition, scramble_shift, scramble_bend, torch.Generator().manual_seed(seed))
        print(f"[-] scrambling the state ({model.scramble.describe()})")

    # initialize dataset
    train_data = MyData()
    # fixed batch size, so that the number of gradient steps grows with the dataset and the ratio of K_NEIGHBORS to the batch stays the same
    BATCH_SIZE = 200
    train_set = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    ## inner product parameters
    # inner product in action space
    M = torch.eye(action_dim, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    L = torch.linalg.cholesky(M)
    # number of nearest neighbors to average across
    K_NEIGHBORS = 8
    # weight of the InneR regularization loss (shared by InneR-C, which is InneR with k = B)
    GAMMA = 0.2

    ## Jacobian regularization baselines (Section IV of the paper)
    # both penalize each J(s) on its own: the squared Frobenius norm shrinks every singular value in proportion,
    # and the nuclear norm pushes the smallest singular values to zero (reducing the rank of J)
    GAMMA_FROBENIUS = 0.01
    GAMMA_NUCLEAR = 0.01

    # main training loop
    null_space_dim = state_dim - action_dim
    for epoch in range(EPOCH+1):
        loss_mse_avg = 0
        loss_inner_avg = 0
        loss_frobenius_avg = 0
        loss_nuclear_avg = 0
        loss_inner_c_avg = 0
        for batch, x in enumerate(train_set):

            # send batch data to the target device
            x = x.to(device)
        
            # collect the demonstrated states and actions
            states = x[:, :state_dim].requires_grad_(True)
            actions = x[:, state_dim:]
            batch_size = states.shape[0]

            # predict actions using policy
            actions_hat = model(states)
            loss_mse = model.mse_loss(actions, actions_hat)

            # regularizers: InneR (model 1), Frobenius norm (model 2), nuclear norm (model 3), and InneR-C (model 4)
            # every model computes all four on logging epochs, to report them; the policy Jacobian is shared, and on the
            # other epochs each model only computes its own loss
            loss_inner = torch.zeros((), device=device)
            loss_frobenius = torch.zeros((), device=device)
            loss_nuclear = torch.zeros((), device=device)
            loss_inner_c = torch.zeros((), device=device)
            if savenumber != 0 or epoch % 100 == 0:

                # get the policy Jacobian wrt state
                J = torch.zeros((batch_size, action_dim, state_dim), device=states.device, dtype=states.dtype)
                for i in range (action_dim):
                    J[:, i] = torch.autograd.grad(
                            actions_hat[:, i],
                            states,
                            grad_outputs=torch.ones_like(actions_hat[:, i]),
                            create_graph=True)[0]
                # the policy sees the scrambled state z (models.py), so regularize its Jacobian with respect to z
                # (chain rule; the identity without scrambling)
                J = model.scramble.jacobian(J, states)

                # compute individual J_tilde and A matrices for all states
                # note that A = J^T * M * J = J_tilde^T * J_tilde, so norms of J_tilde are the norms of J induced by M
                L_batch = L[:batch_size]
                J_tilde = torch.bmm(L_batch.transpose(-1, -2), J)
                A_individual = J_tilde.mT @ J_tilde

                # Jacobian regularization baselines, averaged over the batch
                # squared Frobenius norm of J_tilde, i.e. the sum of the eigenvalues of A(s)
                if savenumber == 2 or epoch % 100 == 0:
                    loss_frobenius = J_tilde.pow(2).sum(dim=(-2, -1)).mean()
                # nuclear norm of J_tilde, i.e. the sum of the square roots of the eigenvalues of A(s)
                if savenumber == 3 or epoch % 100 == 0:
                    loss_nuclear = torch.linalg.svdvals(J_tilde).sum(dim=-1).mean()

                # InneR without clustering (InneR-C): Algorithm 1 with k = B, so the one region is the whole batch
                # and the loss penalizes the shared null space of its average inner product (Equation 9)
                if savenumber == 4 or epoch % 100 == 0:
                    loss_inner_c = inner_loss(A_individual.mean(dim=0, keepdim=True), null_space_dim)

            # inner product regularization (InneR)
            if savenumber == 1 or epoch % 100 == 0:

                # for each A matrix, find the K-nearest neighbors in active space
                with torch.no_grad():
                    # get right singular vectors of J_tilde (rows of Vh span its row space)
                    _, _, Vh = torch.linalg.svd(J_tilde, full_matrices=False)
                    # compute projection matrices directly (P = Vh^T * Vh)
                    P = torch.bmm(Vh.mT, Vh)
                    # flatten to compute pairwise Grassmann distances
                    P_flat = P.view(batch_size, -1).double()
                    dists = torch.cdist(P_flat, P_flat)
                    # find K-Nearest Neighbors based on active subspace alignment
                    current_k = min(K_NEIGHBORS, batch_size)
                    _, knn_indices = torch.topk(dists, current_k, dim=1, largest=False)

                # active subspace regularization on each aligned neighborhood
                A_neighborhood = A_individual[knn_indices].mean(dim=1)

                # InneR loss: ratio of the n - m smallest eigenvalues of each region to the total of all eigenvalues
                loss_inner = inner_loss(A_neighborhood, null_space_dim)

            # get overall loss
            if savenumber == 0:
                loss = loss_mse
            if savenumber == 1:
                loss = loss_mse + GAMMA * loss_inner
            if savenumber == 2:
                loss = loss_mse + GAMMA_FROBENIUS * loss_frobenius
            if savenumber == 3:
                loss = loss_mse + GAMMA_NUCLEAR * loss_nuclear
            if savenumber == 4:
                loss = loss_mse + GAMMA * loss_inner_c

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update loss averages
            loss_mse_avg += loss_mse.item()
            loss_inner_avg += loss_inner.item()
            loss_frobenius_avg += loss_frobenius.item()
            loss_nuclear_avg += loss_nuclear.item()
            loss_inner_c_avg += loss_inner_c.item()
            
        # print average loss and save model
        if epoch % 100 == 0:
            n_batches = len(train_set)
            print(f"Epoch: {epoch} | MSE: {loss_mse_avg/n_batches:.4f} | InneR: {loss_inner_avg/n_batches:.4f} | "
                  f"Frobenius: {loss_frobenius_avg/n_batches:.4f} | Nuclear: {loss_nuclear_avg/n_batches:.4f} | "
                  f"InneR-C: {loss_inner_c_avg/n_batches:.4f}")
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, "models/mlp_weights" + str(savenumber))


# train models
if __name__ == "__main__":

    # CPU, because this small MLP trains faster on the CPU than on a GPU (where kernel launches dominate)
    device = "cpu"

    # seed of this run, from the command line (default 0); bash.sh passes the same seed to every script
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # train baseline (standard BC, 0) and ours (InneR, 1)
    train_model(0, state_dim, hidden_dim, action_dim, device, seed)
    train_model(1, state_dim, hidden_dim, action_dim, device, seed)

    # # Jacobian regularization baselines, Frobenius norm (2) and nuclear norm (3), and InneR without clustering (InneR-C, 4)
    # # we typically do not test these for spot checks, but they should be used for comprehensive evaluations
    # # (to roll them out, add their names to MODEL_NAMES in test_mlp_policy.py)
    # train_model(2, state_dim, hidden_dim, action_dim, device, seed)
    # train_model(3, state_dim, hidden_dim, action_dim, device, seed)
    # train_model(4, state_dim, hidden_dim, action_dim, device, seed)
