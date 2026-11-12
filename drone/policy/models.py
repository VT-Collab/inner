import math
from collections import deque
import numpy as np
import torch
from torch import nn
from .network import DiffusionPolicy as BackupPolicy
from .state_scramble import StateScramble
class DiffusionPolicy(BackupPolicy):
    """Diffusion policy with drone checkpoint and action-window adapters."""
    def __init__(self, state_dim=23, obs_horizon=1, pred_horizon=1,
                 action_horizon=1, hidden_dim=128, diffusion_steps=100,
                 beta_start=1e-4, beta_end=0.1):
        if (obs_horizon, pred_horizon, action_horizon) != (1, 1, 1):
            raise ValueError("Backup policy uses one observation and one action")
        if (beta_start, beta_end) != (1e-4, 0.1):
            raise ValueError("Backup policy fixes beta endpoints to 1e-4 and 0.1")
        super().__init__(state_dim, hidden_dim, 4, diffusion_steps)
        self.config = dict(state_dim=state_dim, obs_horizon=obs_horizon,
            pred_horizon=pred_horizon, action_horizon=action_horizon,
            hidden_dim=hidden_dim, diffusion_steps=diffusion_steps,
            beta_start=beta_start, beta_end=beta_end)
        self.state_dim = self.input_dim = state_dim
        self.obs_horizon = self.pred_horizon = self.action_horizon = 1
        self.diffusion_steps = diffusion_steps

    def forward(self, actions, timesteps, states):
        result = super().forward(actions.flatten(1), timesteps, states)
        return result[:, None] if actions.ndim == 3 else result

    def corrupt(self, actions, timesteps, noise):
        return super().add_noise(actions.flatten(1), timesteps, noise.flatten(1)).reshape_as(actions)

    @torch.no_grad()
    def sample(self, states, generator=None, n_sample_steps=10):
        noise = torch.randn(len(states), 4, device=states.device,
                            dtype=states.dtype, generator=generator)
        return super().sample(states, n_sample_steps, noise)[:, None]


class DiffusionController:
    """CPU rollout adapter: history ending now, future chunk beginning now."""
    def __init__(self, policy):
        self.policy = policy.eval().cpu()
        self.state_dim = policy.state_dim
        self.history = deque(maxlen=policy.obs_horizon)
        self.pending = deque()
        self.reset(0)

    def reset(self, seed):
        self.history.clear()
        self.pending.clear()
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def predict_normalized(self, state):
        observation = torch.as_tensor(np.asarray(state), dtype=torch.float32).clone()
        if not self.history:
            for _ in range(self.policy.obs_horizon):
                self.history.append(observation)
        else:
            self.history.append(observation)
        if not self.pending:
            condition = torch.cat(tuple(self.history))[None]
            chunk = self.policy.sample(condition, self.generator)[0]
            self.pending.extend(chunk[:self.policy.action_horizon].numpy())
        return self.pending.popleft()


class ScrambledPolicy(DiffusionPolicy):
    def __init__(self, *args, scramble_seed=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.scramble = StateScramble(self.state_dim)
        self.scramble.randomize(3., 1., .5, torch.Generator().manual_seed(scramble_seed))
        self.config['scramble_seed'] = scramble_seed

    def forward(self, actions, timesteps, states):
        return super().forward(actions, timesteps, self.scramble(states))



def make_policy(cfg, policy_config=None):
    kwargs = dict(policy_config) if policy_config is not None else dict(state_dim=cfg['state_dim'], hidden_dim=128)
    if cfg['experiment'] == 'e4':
        kwargs.setdefault('scramble_seed', cfg['seed'])
        return ScrambledPolicy(**kwargs)
    return DiffusionPolicy(**kwargs)
