"""Run the job lines of a file with N concurrent workers. Each line is the argument string of run_experiment.py (lines
starting with # are comments). The file is re-read whenever a worker is free, so it can be edited while the queue runs;
a job is skipped when its tag is already in results.jsonl or failed.txt. Exit on a STOP_<name> file in the out_dir, or
after --idle_minutes without a running job.

  toy_environment/venv/bin/python experiments/run_queue.py --name cpu --jobs experiments/jobs_cpu.txt --workers 10
  toy_environment/venv/bin/python experiments/run_queue.py --name gpu --jobs experiments/jobs_gpu.txt --workers 3
"""
import os, json, time, argparse, subprocess, shlex
HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), "toy_environment", "venv", "bin", "python")
RUN = os.path.join(HERE, "run_experiment.py")

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--jobs", required=True)
p.add_argument("--workers", type=int, required=True)
p.add_argument("--idle_minutes", type=float, default=10)
p.add_argument("--out_dir", default=HERE)
a = p.parse_args()
OUT = os.path.abspath(a.out_dir)
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)


def done_tags():
    tags = set()
    if os.path.exists(os.path.join(OUT, "results.jsonl")):
        tags |= {json.loads(l)["tag"] for l in open(os.path.join(OUT, "results.jsonl")) if l.strip()}
    if os.path.exists(os.path.join(OUT, "failed.txt")):
        tags |= {l.split()[0] for l in open(os.path.join(OUT, "failed.txt")) if l.strip()}
    return tags


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


running, tag_cache, last_active = {}, {}, time.time()
while True:
    for tag, (proc, start) in list(running.items()):
        if proc.poll() is not None:
            dt = time.time() - start
            if proc.returncode == 0:
                log(f"done   {tag} ({dt:.0f}s)")
            else:
                log(f"FAILED {tag} (exit {proc.returncode}, {dt:.0f}s), see logs/{tag}.log")
                with open(os.path.join(OUT, "failed.txt"), "a") as f:
                    f.write(f"{tag} exit={proc.returncode} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            del running[tag]
            last_active = time.time()
    if os.path.exists(os.path.join(OUT, f"STOP_{a.name}")):
        if not running:
            log("stop file found, exiting"); break
    elif len(running) < a.workers:
        skip = done_tags() | set(running)
        for line in open(a.jobs):
            args = line.strip()
            if not args or args.startswith("#"):
                continue
            if args not in tag_cache:
                tag_cache[args] = subprocess.check_output([PY, RUN] + shlex.split(args) + ["--tag_only"], text=True).strip()
            tag = tag_cache[args]
            if tag in skip:
                continue
            logf = open(os.path.join(OUT, "logs", tag + ".log"), "a")
            running[tag] = (subprocess.Popen([PY, RUN] + shlex.split(args) + ["--out_dir", OUT], stdout=logf, stderr=subprocess.STDOUT), time.time())
            last_active = time.time()
            log(f"start  {tag}")
            if len(running) >= a.workers:
                break
    if not running and time.time() - last_active > a.idle_minutes * 60:
        log("idle, exiting"); break
    time.sleep(5)
