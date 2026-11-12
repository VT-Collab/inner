"""Train and evaluate one MLP run."""
import os, json, time, fcntl
from types import SimpleNamespace
import torch
from policy.mlp import config
from policy.mlp.training import load_training_data, make_scramble, train_model, diagnostics, checkpoint
from policy.mlp.evaluation import load_test_cases, load_policy, mlp_policy, evaluate_case, summarize
from experiments.mlp_cells import cell, tag, experiments_of
from paths import MLP_RESULTS
SAVENUMBER = {"bc": 0, "inner": 1, "frobenius": 2, "nuclear": 3, "inner_c": 4}
PARAMS = dict(HIDDEN=config.hidden_dim, EPOCH=config.epochs, LR=config.learning_rate, BATCH_SIZE=config.batch_size,
              K_NEIGHBORS=config.k_neighbors, GAMMA=config.gamma, GAMMA_FROBENIUS=config.gamma_frobenius,
              GAMMA_NUCLEAR=config.gamma_nuclear)


def recorded(root, run_tag):
    path=root/'drone_mlp_results.jsonl'
    if not path.exists():return False
    matches=[json.loads(line) for line in path.read_text().splitlines() if line.strip() and json.loads(line)['tag']==run_tag]
    if len(matches)>1:raise ValueError(f'Duplicate recorded MLP tag: {run_tag}')
    return bool(matches)


def execute(job, root=MLP_RESULTS):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _execute(job,root)


def _execute(job, root):
    a=SimpleNamespace(exp=job['exp'], n_demos=job['demos'], algo=job['method'], seed=job['seed'])
    c=cell(a.exp,a.n_demos)
    run_tag=tag(c,a.algo,a.seed)
    if recorded(root,run_tag):
        print('Reusing recorded MLP result',run_tag,flush=True)
        return
    if (root/'models'/f'{run_tag}.pt').exists():
        raise FileExistsError(f'Unrecorded checkpoint exists for {run_tag}; refusing to overwrite it')
    out_dir=str(root.resolve())
    torch.set_num_threads(2)
    device="cuda" if torch.cuda.is_available() else "cpu"
    # training: the input of e4 is scrambled with the scramble of the seed
    x, y, stats = load_training_data(a.exp, a.n_demos)
    scramble = make_scramble(a.exp, a.seed)
    with torch.no_grad():
        z = x if scramble is None else scramble(x)
    t0 = time.time()
    model, history = train_model(SAVENUMBER[a.algo], z, y, a.seed, device)
    train_s = time.time() - t0
    diag = diagnostics(model, x, a.seed, a.n_demos, device, scramble)
    ckpt = checkpoint(model, stats, scramble, SAVENUMBER[a.algo], a.exp, a.n_demos, a.seed, history, diag)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    torch.save(ckpt, os.path.join(out_dir, "models", run_tag + ".pt"))

    # test: the 250 test cases for at most 20 s
    t1 = time.time()
    policy, st = load_policy(ckpt)
    initial_obs20, targets, extra = load_test_cases(a.exp)
    records = [evaluate_case(initial_obs20[i], targets[i], mlp_policy(policy, st, lambda step, i=i: extra(i, step)))
               for i in range(len(targets))]
    metrics = summarize(records)
    test_s = time.time() - t1
    print(f"[{run_tag}] success {metrics['success_at_20']:.3f} | goal error mean {metrics['err_mean']:.3f} m, median "
          f"{metrics['err_median']:.3f} m | regularizers {diag} ({train_s:.0f}s + {test_s:.0f}s)", flush=True)

    rec = {"tag": run_tag, "experiments": experiments_of(c), "policy": "mlp", "algo": a.algo, "env": "drone",
           "exp": a.exp, "state_dim": c["state_dim"], "n_demos": a.n_demos, "seed": a.seed, "rho": c["rho"],
           "cond": c["cond"], "bend": config.e4_bend if c["cond"] > 0 else 0.0, "success": metrics["success_at_20"],
           **metrics, **diag, "params": PARAMS, "train_s": train_s, "test_s": test_s, "device": device,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "run_experiment.py", "dataset": ckpt["dataset"]}
    with open(os.path.join(out_dir, "drone_mlp_results.jsonl"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

