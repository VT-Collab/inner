import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Merge per-family drone CSV exports and compute sample mean/std across seeds."""
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from paths import DP_RESULTS
ROOT = DP_RESULTS
GROUP = ['experiment','policy','env','state_dim','n_demos','rho','cond','algo']
METRICS = ['success','err_mean','err_median','jacobian_norm','null_ratio','nuclear_norm','null_ratio_batch']


def write(path, fields, rows):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def merge(root=ROOT):
    root = Path(root)
    rows, seen, fields = [], set(), None
    for family, name in [('diffusion', 'drone_dp_results.csv'), ('mlp', 'drone_mlp_results.csv')]:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if fields is None: fields = reader.fieldnames
            if reader.fieldnames != fields: raise ValueError(f'Header mismatch: {path}')
            for row in reader:
                if row['policy'] != family: raise ValueError(f'Wrong policy in {path}')
                key = tuple(row[k] for k in GROUP + ['seed'])
                if key in seen: raise ValueError(f'Duplicate run: {key}')
                seen.add(key)
                for metric in METRICS:
                    if row[metric] and not math.isfinite(float(row[metric])):
                        raise ValueError(f'Nonfinite {metric}: {key}')
                rows.append(row)
    rows.sort(key=lambda r: (r['experiment'], r['policy'], int(r['state_dim']), int(r['n_demos']), r['algo'], int(r['seed'])))
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[k] for k in GROUP)].append(row)
    summaries, counts = [], {}
    for key, group in sorted(groups.items()):
        row = dict(zip(GROUP, key)); row['n_seeds'] = len(group)
        counts['|'.join(key)] = {}
        for metric in METRICS:
            values = [float(r[metric]) for r in group if r[metric] != '']
            row[metric+'_mean'] = statistics.mean(values) if values else None
            row[metric+'_std'] = statistics.stdev(values) if len(values)>1 else (0. if values else None)
            counts['|'.join(key)][metric] = len(values)
        summaries.append(row)
    write(root/'results.csv', fields, rows)
    write(root/'summary.csv', GROUP+['n_seeds']+[c for m in METRICS for c in [m+'_mean',m+'_std']], summaries)
    summary_fields = GROUP+['n_seeds']+[c for m in METRICS for c in [m+'_mean',m+'_std']]
    for family, prefix in [('diffusion', 'drone_dp'), ('mlp', 'drone_mlp')]:
        write(root/(prefix+'_summary.csv'), summary_fields, [r for r in summaries if r['policy']==family])
    metadata = dict(generated_at=datetime.now(timezone.utc).isoformat(), rows=len(rows), groups=len(groups),
        policies=dict(Counter(r['policy'] for r in rows)), metric_counts=counts,
        aggregation='Mean and sample standard deviation (ddof=1) across available seeds, separately per policy and condition. Missing metrics remain blank.',
        sources=['drone_dp_results.csv','drone_mlp_results.csv'])
    (root/'merged_csv_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    m=merge(); print(f"Merged {m['rows']} runs, {m['groups']} groups: {m['policies']}")
