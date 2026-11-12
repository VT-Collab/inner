"""Fixed invertible observation map, from single_integrator/toy_environment/models.py."""
import math
import torch
import torch.nn as nn


# randomize initial weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)


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


# fixed scramble of the state seen by the policies (scramble_state in config.py), the first step of every policy
# z = A2 phi(A1 s + b1) + b2: a random affine layer, a bend of each entry phi_i(x) = x + c_i tau tanh(x / tau), and a random
# rotation plus offset. the slope of phi_i is 1 + c_i sech^2(x / tau), between 1 - |c_i| and 1 + |c_i|, so phi is strictly
# increasing for |c_i| < 1 and the whole map is invertible: no information is lost (the dataset and the tests keep s)
# the identity until randomize() or set_transform() is called; all parameters are buffers, so they are saved with the weights
# and the tests apply the transformation used in training
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

    ## draw a random transformation from a seeded generator (the arguments are the scramble settings of config.py)
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

    ## check whether the transformation is the identity
    def is_identity(self):
        eye = torch.eye(len(self.A1), device=self.A1.device)
        return torch.equal(self.A1, eye) and torch.equal(self.A2, eye) and not (self.b1.any() or self.b2.any() or self.bend.any())

    ## describe the transformation in one line, for the training log
    def describe(self):
        if self.is_identity():
            return "identity"
        kind = f"nonlinear, bend up to {self.bend.abs().max():.2f}" if self.bend.any() else "affine"
        return f"{kind}, condition number {torch.linalg.cond(self.A1):.1f}"

    ## the first affine layer, h = A1 s + b1 (the input of the bend)
    def affine(self, state):
        return state @ self.A1.T + self.b1

    ## scramble the state
    # input state s output z = A2 phi(A1 s + b1) + b2
    def forward(self, state):
        h = self.affine(state)
        h = h + self.bend * self.tau * torch.tanh(h / self.tau)
        return h @ self.A2.T + self.b2

    ## convert a derivative with respect to s into the derivative with respect to z, for a Jacobian J(s) or a score g(a, s)
    # chain rule d/dz = (d/ds) A1^-1 diag(1 / phi'(A1 s + b1)) A2^-1 along the last dimension; the slope of phi depends on
    # the state, so the states are needed, and it is a constant (the scramble has no parameters)
    def jacobian(self, jacobian_s, state):
        with torch.no_grad():
            slope = 1.0 + self.bend * (1.0 - torch.tanh(self.affine(state) / self.tau) ** 2)
        return (jacobian_s @ self.A1_inv) / slope.unsqueeze(-2) @ self.A2_inv

    ## load the identity for weights saved before the scramble existed, and the affine scramble for weights saved before the bend
    # (the inverses are recomputed if only A1 and A2 are saved; weights saved before 5 Nov 2026 name the first layer A, A_inv, and b)
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for old, new in (("A", "A1"), ("A_inv", "A1_inv"), ("b", "b1")):
            if prefix + old in state_dict:
                state_dict[prefix + new] = state_dict.pop(prefix + old)
        eye = torch.eye(len(self.A1), device=self.A1.device)
        A1 = state_dict.setdefault(prefix + "A1", eye)
        state_dict.setdefault(prefix + "A1_inv", torch.linalg.inv(A1.double()).float())
        state_dict.setdefault(prefix + "b1", torch.zeros_like(self.b1))
        state_dict.setdefault(prefix + "bend", torch.zeros_like(self.bend))
        A2 = state_dict.setdefault(prefix + "A2", eye)
        state_dict.setdefault(prefix + "A2_inv", torch.linalg.inv(A2.double()).float())
        state_dict.setdefault(prefix + "b2", torch.zeros_like(self.b2))
        super(StateScramble, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)


