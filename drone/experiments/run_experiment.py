"""Train and evaluate one manuscript condition for MLP or Diffusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from experiments import cells, mlp_cells
from naming import checkpoint_relative
from paths import DP_RESULTS, MLP_RESULTS

METHODS = mlp_cells.ALGOS
POLICIES = ['mlp', 'diffusion']
DP_METHODS = dict(zip(METHODS, cells.METHODS))


def run_definition(policy, experiment, seed, method, demos, state_dim, root=None):
    """Validate one condition and resolve its checkpoint path."""
    if policy not in POLICIES or experiment not in cells.EXPERIMENTS or seed not in cells.SEEDS or method not in METHODS:
        raise ValueError('Unsupported manuscript policy, experiment, seed or method')
    root = Path(root) if root is not None else (DP_RESULTS if policy == 'diffusion' else MLP_RESULTS)
    other = MLP_RESULTS if policy == 'diffusion' else DP_RESULTS
    if root.resolve() == other.resolve():
        raise ValueError('Root belongs to the other policy family')
    run = dict(policy=policy, experiment=experiment, seed=seed, method=method, demos=demos, state_dim=state_dim)
    if policy == 'diffusion':
        if (demos, state_dim) not in cells.conditions(experiment):
            raise ValueError('Unsupported manuscript condition')
        cfg = cells.config(experiment, demos, state_dim, seed, DP_METHODS[method])
        return run, cfg, root / checkpoint_relative(cfg)
    matches = [c for c in mlp_cells.all_cells()
               if mlp_cells.experiments_of(c) == [experiment.upper()]
               and (c['n_demos'], c['state_dim']) == (demos, state_dim)]
    if len(matches) != 1:
        raise ValueError('Unsupported manuscript condition')
    run['exp'] = matches[0]['exp']
    return run, None, root / 'models' / (mlp_cells.tag(matches[0], method, seed) + '.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    parser.add_argument('--experiment', choices=cells.EXPERIMENTS, required=True)
    parser.add_argument('--seed', type=int, choices=cells.SEEDS, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--demos', type=int, required=True)
    parser.add_argument('--state-dim', type=int, required=True)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    try:
        run, cfg, checkpoint = run_definition(args.policy, args.experiment, args.seed,
                                              args.method, args.demos, args.state_dim, args.root)
    except ValueError as exc:
        parser.error(str(exc))
    print(checkpoint, flush=True)
    root = checkpoint.parent.parent
    import torch
    torch.set_num_threads(2)
    if args.policy == 'diffusion':
        from experiments.diffusion_run import execute
        execute(cfg, root)
    else:
        from experiments.mlp_run import execute
        execute(run, root)


if __name__ == '__main__':
    main()
