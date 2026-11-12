from naming import checkpoint_path
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .data import select_inputs, normalization
from .io import atomic_json
from .models import make_policy
from .regularizers import inner_regularizer, regularizer as fisher_regularizer
def denoising_loss(policy, states, actions, rng):
    t = torch.randint(policy.diffusion_steps, (len(states),),
                      device=states.device, generator=rng)
    noise = torch.randn(actions.shape, device=actions.device,
                        dtype=actions.dtype, generator=rng)
    prediction = policy(policy.corrupt(actions, t, noise), t, states)
    return (prediction - noise).square().mean()


def train(folder, cfg, rows, lengths):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'config.json').exists():
        if json.loads((folder / 'config.json').read_text()) != cfg:
            raise ValueError(f'Configuration changed: {folder}')
    else:
        atomic_json(folder / 'config.json', cfg)
    count = int(lengths[:cfg['demos']].sum())
    x = select_inputs(rows[:count], cfg['state_dim'])
    y = torch.from_numpy(rows[:count, -4:].copy())
    norm = normalization(x, y, 'e2' if cfg['experiment']=='e4' else cfg['experiment'])
    x = (x - norm['state_mean']) / norm['state_std']
    y = ((y - norm['action_mean']) / norm['action_std'])[:, None]
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    policy = make_policy(cfg).cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg['lr'])
    shuffle = torch.Generator().manual_seed(cfg['seed'] + 10)
    base_rng = torch.Generator(device='cuda').manual_seed(cfg['seed'] + 20)
    reg_rng = torch.Generator(device='cuda').manual_seed(cfg['seed'] + 30)
    loader = DataLoader(TensorDataset(x, y), batch_size=200, shuffle=True, generator=shuffle)
    start_epoch, elapsed, history = 0, 0., []
    cp = checkpoint_path(folder,cfg)
    cp.parent.mkdir(parents=True,exist_ok=True)
    if cp.exists():
        saved = torch.load(cp, map_location='cpu', weights_only=True)
        if saved['config'] != cfg:
            raise ValueError('Checkpoint configuration mismatch')
        policy.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        shuffle.set_state(saved['shuffle_rng'])
        base_rng.set_state(saved['base_rng'])
        reg_rng.set_state(saved['reg_rng'])
        start_epoch, elapsed, history = saved['epoch'], saved['elapsed_seconds'], saved['history']
        if start_epoch == cfg['epochs']:
            return
    start = time.monotonic()
    for epoch in range(start_epoch + 1, cfg['epochs'] + 1):
        policy.train()
        totals = np.zeros(3)
        for batch, (states, actions) in enumerate(loader):
            if cfg['smoke'] and batch >= 2:
                break
            states, actions = states.cuda(), actions.cuda()
            mse = denoising_loss(policy, states, actions, base_rng)
            reg = mse.new_zeros(())
            if cfg['method'] in ['dp_inner', 'dp_inner_c']:
                reg = inner_regularizer(policy, states, reg_rng, rank=8, k=32,
                                        action_samples=8, score_samples=4,
                                        batch_average=cfg['method']=='dp_inner_c')
            elif cfg['method'] in ['dp_frobenius', 'dp_nuclear']:
                reg = fisher_regularizer(policy, states, reg_rng, cfg['method'],
                                         action_samples=8, score_samples=4)
            loss = mse + cfg['weight'] * reg
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss: {folder}, epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            totals += [float(mse.detach()) * len(states), float(reg.detach()) * len(states), len(states)]
        row = dict(epoch=epoch, noise_mse=float(totals[0]/totals[2]), regularizer=float(totals[1]/totals[2]),
                   total_loss=float((totals[0]+cfg['weight']*totals[1])/totals[2]),
                   elapsed_seconds=elapsed + time.monotonic()-start)
        history.append(row)
        saved = dict(config=cfg, epoch=epoch, policy_config=policy.config,
                     model_state_dict={k:v.detach().cpu() for k,v in policy.state_dict().items()},
                     optimizer_state_dict=optimizer.state_dict(), normalization=norm,
                     shuffle_rng=shuffle.get_state(), base_rng=base_rng.get_state(), reg_rng=reg_rng.get_state(),
                     elapsed_seconds=row['elapsed_seconds'], history=history)
        temp = cp.with_suffix('.pt.tmp')
        torch.save(saved, temp)
        temp.replace(cp)
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg['epochs']:
            print(f'{folder} {row}', flush=True)
    atomic_json(folder / 'training_complete.json', dict(epochs=cfg['epochs'], rows=count,
                                                      checkpoint=cfg['checkpoint'], elapsed_seconds=history[-1]['elapsed_seconds']))

