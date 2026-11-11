# Real-robot experiments

Diffusion policies for three tasks on a Franka arm (close a coffee maker, stack two cubes, push a pot), trained with
plain behavior cloning, InneR, or a Fisher trace / nuclear norm penalty.

## Install

```bash
# python 3.11
pip install -r requirements.txt
```

## Train

Each script tracks the objects in the recorded demonstrations, adds rotated copies, builds one dataset per number of
distractors, and trains the policies.

```bash
bash train_close.sh   # or train_stack.sh, train_push.sh
```

The variables at the top of each script (`DEMOS`, `PRIV`) point it at the recorded demonstrations, which are not in
this repo. To train on an existing dataset:

```bash
python train_diffusion_policy.py --dataset datasets/<name>.pkl --save_dir models/<name> --models 0 1 3 \
    --steps 100000 --ema 0.999 --cosine
```

`--models`: 0 behavior cloning, 1 InneR, 2 Fisher trace, 3 Fisher nuclear norm. `--gamma` is InneR's weight and
`--fisher_rank` its rank (default: twice the action dimension).

## Run on the robot

```bash
python deploy.py --model models/<name>/seed0/diffusion_weights1 --perception close --video_dir videos/<run>
```

`deploy.py` talks to our lab's Franka controller and camera publisher (`robot.py`, `cameras.py`), so it will need
adapting to another setup.

## Files

- `models.py`, `train_diffusion_policy.py`: the diffusion policy and the InneR loss (`get_fisher_information`)
- `track_*.py`, `perception.py`: object tracking with SAM 3 and EdgeTAM
- `lowdim_demos.py`, `process_demos.py`: rotated copies of the demonstrations and the state/action datasets
- `deploy.py`, `policy.py`, `viewer.py`: rollouts on the robot
