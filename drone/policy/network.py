import math
import torch
import torch.nn as nn


# randomize initial weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)


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
        x = torch.cat([noisy_action, step_embedding(t, self.embed_dim), state], dim=-1)
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
