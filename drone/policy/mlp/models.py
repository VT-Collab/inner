import math
import torch
import torch.nn as nn


# initialize the weights of each linear layer
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        torch.nn.init.constant_(m.bias, 0)


# deterministic control policy
# input: normalized state (19D state + the experiment's extra entries), output: normalized motor RPM
# two hidden layers with ReLU and a linear output
class MLPPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLPPolicy, self).__init__()

        self.pi_1 = nn.Linear(state_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)
        # small final layer, so the first predicted RPM is close to the mean RPM of the training data
        torch.nn.init.uniform_(self.pi_3.weight, -3e-3, 3e-3)
        torch.nn.init.constant_(self.pi_3.bias, 0)

        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.pi_1(state))
        x = self.relu(self.pi_2(x))
        return self.pi_3(x)


## e4: the scramble of single_integrator/toy_environment/models.py (random_affine and StateScramble, the same draws and the same forward pass;
## the chain rule and the loading of older weights are left out, since the drone MLP is trained on the scrambled input)

# random affine transformation z = A s + b, the building block of StateScramble
# A = U diag(sigma) V^T with random orthogonal U and V, and log-uniform sigma with ratio at most condition (1 is a rotation)
# b is uniform in [-shift, shift]; a seeded generator makes the draw reproducible
def random_affine(state_dim, condition=10.0, shift=5.0, generator=None):
    def random_orthogonal():
        Q, R = torch.linalg.qr(torch.randn(state_dim, state_dim, generator=generator, dtype=torch.float64))
        return Q * torch.sign(torch.diagonal(R))
    log_sigma = (torch.rand(state_dim, generator=generator, dtype=torch.float64) - 0.5) * math.log(condition)
    A = random_orthogonal() @ torch.diag(torch.exp(log_sigma)) @ random_orthogonal().T
    b = (torch.rand(state_dim, generator=generator, dtype=torch.float64) * 2.0 - 1.0) * shift
    return A.float(), b.float()


# fixed invertible scramble z = A2 phi(A1 s + b1) + b2: a random affine layer, a bend of each entry
# phi_i(x) = x + c_i tau tanh(x / tau), and a random rotation plus offset. the slope of phi_i is between 1 - |c_i| and
# 1 + |c_i|, so phi is strictly increasing for |c_i| < 1 and no information is lost
# the identity until randomize() is called; all parameters are buffers, so they are saved with the weights
class StateScramble(nn.Module):
    def __init__(self, state_dim, tau=5.0):
        super(StateScramble, self).__init__()
        self.register_buffer("A1", torch.eye(state_dim))
        self.register_buffer("A1_inv", torch.eye(state_dim))
        self.register_buffer("b1", torch.zeros(state_dim))
        self.register_buffer("bend", torch.zeros(state_dim))
        self.register_buffer("A2", torch.eye(state_dim))
        self.register_buffer("A2_inv", torch.eye(state_dim))
        self.register_buffer("b2", torch.zeros(state_dim))
        # width of the bend (fixed, so not a buffer)
        self.tau = tau

    ## draw a random transformation from a seeded generator
    # the affine layer is drawn first, so for the same seed it is the same with and without the bend; bend 0 gives z = A1 s + b1
    def randomize(self, condition, shift, bend, generator):
        n = len(self.A1)
        A1, b1 = random_affine(n, condition, shift, generator)
        if bend > 0.0:
            c = (torch.rand(n, generator=generator) * 2.0 - 1.0) * bend
            A2, b2 = random_affine(n, 1.0, shift, generator)
            self.set_transform(A1, b1, c, A2, b2)
        else:
            self.set_transform(A1, b1)
        return self

    ## set the transformation (A1 and A2 must be invertible, |bend| < 1); without bend, A2, and b2 it is affine, z = A1 s + b1
    def set_transform(self, A1, b1, bend=None, A2=None, b2=None):
        if bend is not None and bend.abs().max() >= 1.0:
            raise ValueError("the bend must be below 1 for phi to be increasing")
        eye = torch.eye(len(self.A1), device=self.A1.device)
        self.A1.copy_(A1)
        self.A1_inv.copy_(torch.linalg.inv(self.A1.double()).float())
        self.b1.copy_(b1)
        self.bend.copy_(torch.zeros_like(self.bend) if bend is None else bend)
        self.A2.copy_(eye if A2 is None else A2)
        self.A2_inv.copy_(torch.linalg.inv(self.A2.double()).float())
        self.b2.copy_(torch.zeros_like(self.b2) if b2 is None else b2)

    ## describe the transformation in one line, for the training log
    def describe(self):
        kind = f"nonlinear, bend up to {self.bend.abs().max():.2f}" if self.bend.any() else "affine"
        return f"{kind}, condition number {torch.linalg.cond(self.A1):.1f}"

    ## scramble the state
    # input state s output z = A2 phi(A1 s + b1) + b2
    def forward(self, state):
        h = state @ self.A1.T + self.b1
        h = h + self.bend * self.tau * torch.tanh(h / self.tau)
        return h @ self.A2.T + self.b2
