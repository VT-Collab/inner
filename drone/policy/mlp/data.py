"""Read the packaged trajectories and test cases, the same ones the diffusion policy uses."""
from pathlib import Path
import hashlib
import pickle
import sys
import numpy as np
from .config import input_dim

from paths import DATA, SIMULATOR
if str(SIMULATOR) not in sys.path:
    sys.path.insert(0,str(SIMULATOR))


def dataset_key(experiment):
    if experiment in ('e2','e4'):return 'e2'
    if experiment=='clean' or experiment in ('e1_1','e1_2','e1_3'):return 'e1'
    if experiment=='e3':return 'e3'
    raise ValueError(f'Unknown experiment: {experiment}')


from policy.data import load_pickle, read_data


def load_rows(experiment, split):
    # Same source and episode boundaries; retain MLP float64 normalization.
    return read_data(dataset_key(experiment), split, dtype=np.float64)


def training_inputs(experiment,n_demos):
    rows,lengths=load_rows(experiment,'train')
    if not 1<=n_demos<=len(lengths):raise ValueError('Invalid demonstration count')
    count=int(lengths[:n_demos].sum())
    return rows[:count,:input_dim(experiment)],rows[:count,-4:]


def test_cases(experiment):
    rows,lengths=load_rows(experiment,'test')
    key=dataset_key(experiment)
    extras=np.asarray(load_pickle(DATA/key/'test_extras.pkl'),dtype=np.float32)
    if len(lengths)!=250 or not np.isfinite(extras).all():raise ValueError('Invalid test cases')
    aligned=np.concatenate([extras[i,:n] for i,n in enumerate(lengths)]) if key=='e1' else np.repeat(extras,lengths,axis=0)
    if not np.array_equal(aligned,rows[:,19:-4].astype(np.float32)):
        raise ValueError('Recorded test extras do not match offline rows')
    starts=np.r_[0,np.cumsum(lengths)[:-1]]
    first=rows[starts].astype(np.float32)
    initial=np.concatenate((first[:,:16],np.zeros((len(first),4),np.float32)),axis=1)
    dimension=input_dim(experiment)-19
    values=extras[...,:dimension]
    extra=(lambda case,step:values[case,step]) if key=='e1' else (lambda case,step:values[case])
    return initial,first[:,16:19],extra


def provenance(experiment):
    folder=DATA/dataset_key(experiment)
    files={}
    for name in ['train.pkl','train_lengths.pkl','test.pkl','test_lengths.pkl','test_extras.pkl','norm_stats.pkl']:
        h=hashlib.sha256()
        with (folder/name).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        files[f'{folder.name}/{name}']=h.hexdigest()
    return {'source':'drone-policy/datasets/data','files':files,'selection':'first N complete demonstrations; fixed test order'}
