import math
import torch
import torch.nn as nn


def weights_init_(m):
    """initialize linear layers"""
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.constant_(m.bias, 0)


class MLPPolicy(nn.Module):
    """state to action MLP"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(MLPPolicy, self).__init__()

        self.pi_1 = nn.Linear(state_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)
        # final layer initialization (because tanh, not relu)
        torch.nn.init.uniform_(self.pi_3.weight, -3e-3, 3e-3)
        torch.nn.init.constant_(self.pi_3.bias, 0)
        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()

    def forward(self, state):
        x = self.relu(self.pi_1(state))
        x = self.relu(self.pi_2(x))
        return torch.tanh(self.pi_3(x))


def step_embedding(t, embed_dim):
    """sinusoidal embedding of the diffusion step"""
    half_dim = embed_dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half_dim, device=t.device) / half_dim)
    angles = t.float().unsqueeze(-1) * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DiffusionPolicy(nn.Module):
    """noise-prediction MLP, trained with DDPM and sampled with DDIM"""
    def __init__(self, state_dim, hidden_dim, action_dim, n_diffusion_steps=100, embed_dim=16):
        super(DiffusionPolicy, self).__init__()
        self.action_dim = action_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.embed_dim = embed_dim

        # linear betas (Ho et al. 2020), larger final beta because T = 100
        betas = torch.linspace(1e-4, 0.1, n_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        # lambda_t in Equation (20), the DDPM weight (Ho et al. 2020, Equation 12)
        elbo_weights = betas / (2.0 * alphas * (1.0 - alpha_bars))
        # proposal q(t) for the score in Equation (21), proportional to lambda_t * sqrt(alpha_bar_t / (1 - alpha_bar_t))
        # 10% uniform so 1 / q(t) stays bounded
        score_t_probs = elbo_weights * torch.sqrt(alpha_bars / (1.0 - alpha_bars))
        score_t_probs = 0.9 * score_t_probs / score_t_probs.sum() + 0.1 / n_diffusion_steps
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("elbo_weights", elbo_weights)
        # not persistent, so older checkpoints still load
        self.register_buffer("score_t_probs", score_t_probs, persistent=False)

        self.pi_1 = nn.Linear(action_dim + embed_dim + state_dim, hidden_dim)
        self.pi_2 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_3 = nn.Linear(hidden_dim, hidden_dim)
        self.pi_4 = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)
        self.relu = nn.ReLU()
        self.mse_loss = nn.MSELoss()

    def forward(self, noisy_action, t, state):
        """predict the noise added to the action"""
        x = torch.cat([noisy_action, step_embedding(t, self.embed_dim), state], dim=-1)
        x = self.relu(self.pi_1(x))
        x = self.relu(self.pi_2(x))
        x = self.relu(self.pi_3(x))
        return self.pi_4(x)

    def add_noise(self, action, t, noise):
        """forward process, noise the action to step t"""
        alpha_bar = self.alpha_bars[t].unsqueeze(-1)
        return torch.sqrt(alpha_bar) * action + torch.sqrt(1.0 - alpha_bar) * noise

    def denoising_error(self, action, state, t, noise):
        """denoising loss l(a, s; t, eps) from the paper, one value per row"""
        noise_hat = self.forward(self.add_noise(action, t, noise), t, state)
        return (noise - noise_hat).pow(2).sum(dim=-1)

    def diffusion_loss(self, action, state):
        """mean noise-prediction error at random diffusion steps"""
        t = torch.randint(0, self.n_diffusion_steps, (len(action),), device=action.device)
        noise = torch.randn_like(action)
        noise_hat = self.forward(self.add_noise(action, t, noise), t, state)
        return self.mse_loss(noise_hat, noise)

    def sample(self, state, n_sample_steps=10, noise=None):
        """sample actions with DDIM, starting from gaussian noise"""
        if noise is None:
            noise = torch.randn(len(state), self.action_dim, device=state.device)
        action = noise
        steps = torch.linspace(self.n_diffusion_steps - 1, 0, n_sample_steps).round().long().tolist()
        for i, t in enumerate(steps):
            alpha_bar = self.alpha_bars[t]
            alpha_bar_next = self.alpha_bars[steps[i + 1]] if i + 1 < len(steps) else torch.ones_like(alpha_bar)
            t_batch = torch.full((len(state),), t, dtype=torch.long, device=state.device)
            noise_hat = self.forward(action, t_batch, state)
            action_0 = (action - torch.sqrt(1.0 - alpha_bar) * noise_hat) / torch.sqrt(alpha_bar)
            action_0 = action_0.clamp(-1.0, 1.0)
            noise_hat = (action - torch.sqrt(alpha_bar) * action_0) / torch.sqrt(1.0 - alpha_bar)
            action = torch.sqrt(alpha_bar_next) * action_0 + torch.sqrt(1.0 - alpha_bar_next) * noise_hat
        return action
