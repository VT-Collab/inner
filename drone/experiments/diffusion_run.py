"""Train, evaluate and diagnose one manuscript run; reuse matching completed runs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import fcntl
import json
import numpy as np
import torch
from experiments.cells import ROOT, SEEDS, METHODS, EXPERIMENTS, conditions, config, run_folder
from naming import checkpoint_path
from policy.data import DATA, load_pickle, read_data
from policy.io import atomic_json, digest
from policy.training import train
from policy.evaluation import evaluate, geometry


def validate_sources(root):
    manifest = root/'workflow_manifest.json'
    if not manifest.exists():
        manifest = root/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        for name, expected in previous.get('files', {}).items():
            # Resolve against the current dataset root.
            path = DATA/Path(name).parent.name/Path(name).name
            if digest(path) != expected:
                raise ValueError(f'Dataset changed: {path}')


def completed(root, cfg):
    folder = run_folder(root, cfg)
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text()) != cfg:
        raise ValueError(f'Configuration changed: {folder}')
    if not (folder/'evaluation.json').exists():
        return False
    if not (folder/'config.json').exists():
        raise ValueError(f'Missing configuration: {folder}')
    result = json.loads((folder/'evaluation.json').read_text())
    if result['episodes'] != cfg['evaluation_cases'] or result['checkpoint_epoch'] != cfg['epochs']:
        raise ValueError(f'Incomplete evaluation: {folder}')
    if not (checkpoint_path(folder,cfg)).exists() or digest(checkpoint_path(folder,cfg)) != result['checkpoint_sha256']:
        raise ValueError(f'Evaluation/checkpoint mismatch: {folder}')
    return cfg['experiment'] in ['e3','e4'] or (folder/'geometry.json').exists()


def execute(cfg, root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_sources(root)
        if completed(root, cfg):
            print('Reusing', run_folder(root, cfg), flush=True)
            return
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; refusing CPU fallback')
        exp = 'e2' if cfg['experiment']=='e4' else cfg['experiment']
        rows, lengths = read_data(exp, 'train')
        test, test_lengths = read_data(exp, 'test')
        extras = np.asarray(load_pickle(DATA/exp/'test_extras.pkl'), dtype=np.float32)
        if len(test_lengths)!=250 or not np.isfinite(extras).all():
            raise ValueError('Invalid test cases/extras')
        aligned = (np.concatenate([extras[i,:n] for i,n in enumerate(test_lengths)])
                   if exp=='e1' else np.repeat(extras,test_lengths,axis=0))
        if not np.array_equal(aligned,test[:,19:-4]):
            raise ValueError('Offline/rollout extras misaligned')
        folder = run_folder(root, cfg)
        def status(phase, **kw):
            atomic_json(root/'status.json', dict(phase=phase, **cfg, **kw))
        # Record code and data hashes.
        source_root = Path(__file__).resolve().parents[1]
        atomic_json(root/'workflow_manifest.json',dict(
            code={str(p.relative_to(source_root)):digest(p) for sub in ['policy','experiments']
                  for p in sorted((source_root/sub).rglob('*.py'))},
            files={str(p):digest(p) for e in ['e2','e1','e3'] for p in sorted((DATA/e).glob('*.pkl'))}))
        try:
            status('training'); train(folder,cfg,rows,lengths)
            status('evaluating'); evaluate(folder,cfg,test,test_lengths,extras)
            if cfg['experiment'] in ['e2','e1']:
                status('geometry'); geometry(folder,cfg,test)
            from experiments.collect_diffusion import collect
            collect(root)
            status('complete')
        except BaseException as exc:
            status('failed',error=repr(exc)); raise

