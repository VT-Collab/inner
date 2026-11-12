from naming import checkpoint_path
import json
import numpy as np
import torch
from .models import DiffusionController, make_policy
from .data import select_inputs
from .io import atomic_json, digest
from .regularizers import elbo_scores, fisher_from_scores
from .rollout import evaluate_case, make_rollout_args, summarize
class PackagedController:
    """Feed the packaged inputs (16D state, goal, extras) to the obs20+goal evaluator, with their normalization."""
    state_dim = 23

    def __init__(self, policy, norm, extras):
        self.controller = DiffusionController(policy)
        self.norm = norm
        self.extras = extras
        self.step = 0

    def reset(self, seed):
        self.controller.reset(seed)
        self.step = 0

    def predict_normalized(self, obs_goal):
        extra = self.extras[self.step] if self.extras.ndim == 2 else self.extras
        state = np.concatenate([obs_goal[:16], obs_goal[20:23], extra]).astype(np.float32)
        self.step += 1
        return self.controller.predict_normalized((state-self.norm['state_mean'])/self.norm['state_std'])


def evaluate(folder, cfg, test_rows, test_lengths, extras):
    if (folder / 'evaluation.json').exists():
        return
    ck = torch.load(checkpoint_path(folder,cfg), map_location='cpu', weights_only=True)
    if ck['epoch'] != cfg['epochs']:
        raise ValueError('Evaluation requires final epoch')
    policy = make_policy(cfg, ck['policy_config'])
    policy.load_state_dict(ck['model_state_dict'])
    norm = {k:v.numpy() for k,v in ck['normalization'].items()}
    outer_norm = dict(norm, state_mean=np.zeros(23, np.float32), state_std=np.ones(23, np.float32))
    starts = np.r_[0, np.cumsum(test_lengths)[:-1]]
    args = make_rollout_args(max_seconds=cfg['evaluation_steps']/48, sampling_seed=2026)
    records = []
    records_path = folder / 'episodes.json'
    if records_path.exists():
        records = json.loads(records_path.read_text())
        assert [r['case_id'] for r in records] == list(range(len(records)))
    for i in range(len(records), cfg['evaluation_cases']):
        first = test_rows[starts[i]]
        case = dict(case_id=i, initial_observation=np.r_[first[:16], np.zeros(4)], target=first[16:19])
        extra = extras[i, ..., :cfg['state_dim']-19]
        model = PackagedController(policy, norm, extra)
        record = evaluate_case(model, outer_norm, case, args,
                               'dp_inner' if cfg['method']=='dp_inner_c' else cfg['method'])
        records.append(record)
        atomic_json(records_path, records)
        if (i+1) % 25 == 0 or i+1 == cfg['evaluation_cases']:
            print(f'{folder} test {i+1}/{cfg["evaluation_cases"]}: {summarize(records)["success_rate"]:.3f}', flush=True)
    atomic_json(folder / 'evaluation.json', dict(summarize(records), checkpoint_epoch=ck['epoch'],
                                               settings=vars(args), checkpoint_sha256=digest(checkpoint_path(folder,cfg))))


def geometry(folder, cfg, test_rows):
    if cfg['experiment'] == 'e3' or (folder / 'geometry.json').exists():
        return
    ck = torch.load(checkpoint_path(folder,cfg), map_location='cpu', weights_only=True)
    policy = make_policy(cfg, ck['policy_config']).cuda()
    policy.load_state_dict(ck['model_state_dict']); policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    count = 40 if cfg['smoke'] else 800
    indices = np.random.default_rng(2026).choice(len(test_rows), count, replace=False)
    norm = ck['normalization']
    states = select_inputs(test_rows[indices], cfg['state_dim'])
    states = ((states-norm['state_mean'])/norm['state_std']).cuda()
    rng = torch.Generator(device='cuda').manual_seed(2026)
    results = []
    for s in states.split(200):
        b = len(s)
        actions = policy.sample(s[:,None].expand(b,8,-1).reshape(b*8,-1), rng, n_sample_steps=3).reshape(b,8,1,4)
        t = torch.multinomial(policy.score_t_probs,b*8*4,replacement=True,generator=rng).reshape(b,8,4)
        noise = torch.randn(b,8,4,1,4,device='cuda',generator=rng)
        scores = elbo_scores(policy,s,actions,t,noise,create_graph=False)
        with torch.no_grad():
            _,_,vh = torch.linalg.svd(scores,full_matrices=False)
            proj = vh[:,:8].mT @ vh[:,:8]
            neighbors = torch.cdist(proj.flatten(1).double(),proj.flatten(1).double()).topk(min(32,b),largest=False).indices
            fisher = fisher_from_scores(scores)
            average = fisher[neighbors].mean(1).double()
            eigen = torch.linalg.eigvalsh((average+average.mT)/2).clamp_min(0)
            ratio = eigen[:,:cfg['state_dim']-8].sum(-1)/(eigen.sum(-1)+1e-8)
            trace = fisher.diagonal(dim1=-2,dim2=-1).sum(-1)
            results.append(torch.stack([trace,ratio],-1).cpu().numpy())
    values = np.concatenate(results)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite geometry')
    np.savez_compressed(folder/'geometry_states.npz',indices=indices,values=values)
    atomic_json(folder/'geometry.json',dict(fisher_trace=float(values[:,0].mean()),
                                           null_ratio=float(values[:,1].mean()), states=count,
                                           rank=8,k=32,action_samples=8,score_samples=4,
                                           coordinates='normalized inputs; ratio is a fraction, not percent'))

