import torch
import numpy as np
from models import MLPPolicy, DiffusionPolicy
from config import dims


class Policy:
    """trained policy: raw state in, raw action out"""

    def __init__(self, path, device="cpu", n_sample_steps=10):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.kind = ckpt["kind"]
        self.stats = ckpt["stats"]
        self.device = device
        self.n_sample_steps = n_sample_steps
        state_dim, action_dim = dims(self.stats)
        if self.kind == "mlp":
            self.model = MLPPolicy(state_dim, ckpt["hidden_dim"], action_dim)
        else:
            self.model = DiffusionPolicy(state_dim, ckpt["hidden_dim"], action_dim)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, state):
        s = self.stats
        x = (np.asarray(state, dtype=np.float32) - s["state_min"]) / (s["state_max"] - s["state_min"]) * 2 - 1
        x = torch.tensor(x, dtype=torch.float32, device=self.device)[None]
        if self.kind == "mlp":
            a = self.model(x)
        else:
            a = self.model.sample(x, self.n_sample_steps)
        a = a[0].cpu().numpy()
        return (a + 1) / 2 * (s["action_max"] - s["action_min"]) + s["action_min"]
