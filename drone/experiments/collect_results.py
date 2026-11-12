"""Refresh both family exports, then the combined manuscript sheets."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv
from experiments.merge_policy_results import write, METRICS
from paths import DP_RESULTS, MLP_RESULTS
from experiments.collect_diffusion import collect
from experiments.collect_mlp import collect_mlp


def collect_all(dp_root=DP_RESULTS,mlp_root=MLP_RESULTS):
    if dp_root.resolve()==mlp_root.resolve():raise ValueError('Family output roots must differ')
    if not (mlp_root/'drone_mlp_results.jsonl').exists():
        dp_root.mkdir(parents=True,exist_ok=True)
        return collect(dp_root)
    collect_mlp(mlp_root)
    dp_root.mkdir(parents=True,exist_ok=True)
    source=mlp_root/'drone_mlp_results.csv'
    target=dp_root/'drone_mlp_results.csv'
    with source.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    # Match recorded metrics by condition.
    key=lambda row:tuple(row[k] for k in ['experiment','policy','algo','env','state_dim','n_demos','seed'])
    if len({key(row) for row in rows})!=len(rows):raise ValueError('Duplicate MLP conditions')
    if target.exists():
        with target.open(newline='') as f:previous=list(csv.DictReader(f))
        if len({key(row) for row in previous})!=len(previous):raise ValueError('Duplicate imported MLP conditions')
        old={key(row):row for row in previous}
        for row in rows:
            prior=old.pop(key(row),None)
            if prior is not None:
                if any(row[m]!=prior[m] and (not row[m] or not prior[m] or float(row[m])!=float(prior[m])) for m in METRICS):
                    raise ValueError(f'Conflicting historical MLP metrics: {key(row)}')
        rows.extend(old.values())
    write(target,fields,rows)
    return collect(dp_root)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dp-root',type=Path,default=DP_RESULTS)
    p.add_argument('--mlp-root',type=Path,default=MLP_RESULTS)
    a=p.parse_args();print(collect_all(a.dp_root,a.mlp_root))
