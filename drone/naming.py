"""Checkpoint names: models/<policy>_<algo>_<env>_d<D>_n<N>[_rho<R>|_cond<C>]_s<S>.pt."""
from pathlib import Path

ALGOS = dict(dp='bc',dp_inner='inner',dp_frobenius='frobenius',dp_nuclear='nuclear',dp_inner_c='inner_c')

def tag(policy, algo, dimension, demos, seed, rho=0., cond=0., env='drone'):
    name=f'{policy}_{algo}_{env}_d{dimension}_n{demos}'
    if rho>0:name+=f'_rho{rho:g}'
    if cond>0:name+=f'_cond{cond:g}'
    return name+f'_s{seed}'

def diffusion_tag(cfg):
    return tag('diffusion',ALGOS[cfg['method']],cfg['state_dim'],cfg['demos'],cfg['seed'],
               rho=.9 if cfg['experiment']=='e3' else 0.,
               cond=cfg.get('scramble_condition',3.) if cfg['experiment']=='e4' else 0.)

def checkpoint_relative(cfg):
    return 'models/'+diffusion_tag(cfg)+'.pt'

def checkpoint_path(folder, cfg):
    # Per-run metrics remain at <root>/eX/seedS/nN_dD/method/.
    return Path(folder).parents[3]/checkpoint_relative(cfg)
