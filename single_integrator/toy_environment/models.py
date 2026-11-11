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


# control policy
class MLPPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLPPolicy, self).__init__()

        ## define policy
        # scramble of the state (identity unless randomized, see StateScramble)
        self.scramble = StateScramble(state_dim)
        # fully connected multi-layer perceptron (MLP)
        self.pi_1 = nn.Linear(state_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, action_dim)

        ## helper functions
        self.apply(weights_init_)
        # final layer initialization (because tanh, not relu)
        torch.nn.init.uniform_(self.pi_3.weight, -3e-3, 3e-3)
        torch.nn.init.constant_(self.pi_3.bias, 0)
        # relu activation function
        self.relu = nn.ReLU()
        # loss function
        self.mse_loss = nn.MSELoss()

    ## execute robot policy
    # input state output action
    def forward(self, state):
        x = self.relu(self.pi_1(self.scramble(state)))
        x = self.relu(self.pi_2(x))
        return torch.tanh(self.pi_3(x))


# sinusoidal embedding of the diffusion step
# maps each integer step t to sines and cosines at several frequencies
def step_embedding(t, embed_dim):
    half_dim = embed_dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half_dim, device=t.device) / half_dim)
    angles = t.float().unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


# diffusion control policy
# trained as a denoising diffusion model (DDPM) and sampled with DDIM
class DiffusionPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, n_diffusion_steps=100, embed_dim=16):
        super(DiffusionPolicy, self).__init__()
        self.action_dim = action_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.embed_dim = embed_dim

        ## noise schedule
        # linear schedule of betas (Ho et al. 2020), with a larger final beta because T = 100 steps
        betas = torch.linspace(1e-4, 0.1, n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        # weight of each step in the ELBO, lambda_t in Equation (20)
        # this is the DDPM weight with sigma_t^2 = beta_t (Ho et al. 2020, Equation 12)
        elbo_weights = betas / (2.0 * alphas * (1.0 - alpha_bars))
        # proposal q(t) over diffusion steps for the score in Equation (21), proportional to lambda_t * sqrt(alpha_bar_t / (1 - alpha_bar_t))
        # low-noise steps carry most of the score, so drawing them more often (and reweighting by 1 / q(t)) lowers its variance
        # 10% of q(t) is uniform so 1 / q(t) stays bounded; a uniform q(t) = 1 / T gives the original estimator
        score_t_probs = elbo_weights * torch.sqrt(alpha_bars / (1.0 - alpha_bars))
        score_t_probs = 0.9 * score_t_probs / score_t_probs.sum() + 0.1 / n_diffusion_steps
        # buffers are saved with the model weights and move with .to(device)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("elbo_weights", elbo_weights)
        # (except q(t), so that models saved before it was added still load)
        self.register_buffer("score_t_probs", score_t_probs, persistent=False)

        ## define noise prediction network
        # scramble of the state (identity unless randomized, see StateScramble)
        self.scramble = StateScramble(state_dim)
        # fully connected multi-layer perceptron (MLP)
        # input: noisy action, embedding of the diffusion step, and state
        self.pi_1 = nn.Linear(action_dim + embed_dim + state_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_4 = nn.Linear(hidden_dim, action_dim)

        ## helper functions
        self.apply(weights_init_)
        # relu activation function
        self.relu = nn.ReLU()
        # loss function
        self.mse_loss = nn.MSELoss()

    ## predict the noise that was added to the action
    # input noisy action, diffusion step, and state, output predicted noise
    def forward(self, noisy_action, t, state):
        x = torch.cat([noisy_action, step_embedding(t, self.embed_dim), self.scramble(state)], dim=-1)
        x = self.relu(self.pi_1(x))
        x = self.relu(self.pi_2(x))
        x = self.relu(self.pi_3(x))
        return self.pi_4(x)

    ## forward process: add noise to the action at diffusion step t
    def add_noise(self, action, t, noise):
        alpha_bar = self.alpha_bars[t].unsqueeze(-1)
        return torch.sqrt(alpha_bar) * action + torch.sqrt(1.0 - alpha_bar) * noise

    ## denoising loss l(a, s; t, eps) from the paper
    # squared error between the added noise and the predicted noise, one value per row
    def denoising_error(self, action, state, t, noise):
        noise_hat = self.forward(self.add_noise(action, t, noise), t, state)
        return (noise - noise_hat).pow(2).sum(dim=-1)

    ## default diffusion policy loss
    # predict the noise at a random diffusion step for every (state, action) pair
    def diffusion_loss(self, action, state):
        t = torch.randint(0, self.n_diffusion_steps, (len(action),), device=action.device)
        noise = torch.randn_like(action)
        noise_hat = self.forward(self.add_noise(action, t, noise), t, state)
        return self.mse_loss(noise_hat, noise)

    ## execute robot policy
    # input state output action, sampled with DDIM starting from gaussian noise
    def sample(self, state, n_sample_steps=10, noise=None):
        if noise is None:
            noise = torch.randn(len(state), self.action_dim, device=state.device)
        action = noise
        # evenly spaced diffusion steps from T-1 down to 0
        steps = torch.linspace(self.n_diffusion_steps - 1, 0, n_sample_steps).round().long().tolist()
        for i, t in enumerate(steps):
            alpha_bar = self.alpha_bars[t]
            alpha_bar_next = self.alpha_bars[steps[i + 1]] if i + 1 < len(steps) else torch.ones_like(alpha_bar)
            # predict the noise, and from it the clean action (clipped to the action range)
            t_batch = torch.full((len(state),), t, dtype=torch.long, device=state.device)
            noise_hat = self.forward(action, t_batch, state)
            action_0 = (action - torch.sqrt(1.0 - alpha_bar) * noise_hat) / torch.sqrt(alpha_bar)
            action_0 = action_0.clamp(-1.0, 1.0)
            # recompute the noise from the clipped action so the update stays consistent
            noise_hat = (action - torch.sqrt(alpha_bar) * action_0) / torch.sqrt(1.0 - alpha_bar)
            # deterministic DDIM update to the next (less noisy) step
            action = torch.sqrt(alpha_bar_next) * action_0 + torch.sqrt(1.0 - alpha_bar_next) * noise_hat
        return action
