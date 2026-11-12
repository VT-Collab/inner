import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Export completed drone diffusion evaluations to CSV.

Missing diagnostics are blank, never zero. Summary std is sample standard
deviation, not SEM. Raw exports include every completed run, even when a seed
is incomplete across methods; manuscript plots retain their paired-seed rule.
"""
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from paths import DP_RESULTS
from naming import diffusion_tag, checkpoint_relative
ROOT = DP_RESULTS
COLS = ['experiment','policy','algo','env','state_dim','n_demos','seed','rho','cond',
        'success','err_mean','err_median','jacobian_norm','null_ratio','nuclear_norm',
        'null_ratio_batch','tag','model']
METRICS = ['success','err_mean','err_median','jacobian_norm','null_ratio','nuclear_norm','null_ratio_batch']
GROUP = ['experiment','policy','env','state_dim','n_demos','rho','cond','algo']
ALGOS = dict(dp='bc',dp_inner='inner',dp_frobenius='frobenius',dp_nuclear='nuclear',dp_inner_c='inner_c')

def write_csv(path, fields, rows):
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)

def collect(root=ROOT):
    root=Path(root);rows=[];seen=set()
    for path in sorted(root.glob('e[1234]/seed*/n*/*/evaluation.json')):
        folder=path.parent
        cfg=json.loads((folder/'config.json').read_text())
        result=json.loads(path.read_text())
        if cfg.get('smoke') or result.get('episodes')!=250 or result.get('checkpoint_epoch')!=100:
            continue
        if cfg['method'] not in ALGOS:continue
        episodes=json.loads((folder/'episodes.json').read_text())
        assert len(episodes)==250 and [v['case_id'] for v in episodes]==list(range(250)),path
        errors=[float(v['final_xyz_error_m']) for v in episodes]
        assert all(math.isfinite(v) for v in errors),path
        assert math.isclose(statistics.mean(errors),result['mean_final_xyz_error_m_all'],rel_tol=1e-6,abs_tol=1e-8),path
        assert math.isclose(sum(v['success'] for v in episodes)/250,result['success_rate'],abs_tol=1e-12),path
        geometry=json.loads((folder/'geometry.json').read_text()) if (folder/'geometry.json').exists() else {}
        row=dict(experiment=cfg['experiment'].upper(),policy='diffusion',algo=ALGOS[cfg['method']],env='drone',
                 state_dim=cfg['state_dim'],n_demos=cfg['demos'],seed=cfg['seed'],
                 rho=cfg.get('rho',''),cond=cfg.get('scramble_condition',0.),
                 success=result['success_rate'],err_mean=result['mean_final_xyz_error_m_all'],
                 err_median=statistics.median(errors),jacobian_norm=geometry.get('fisher_trace'),
                 null_ratio=geometry.get('null_ratio'),nuclear_norm=geometry.get('nuclear_norm'),
                 null_ratio_batch=geometry.get('null_ratio_batch'),
                 tag=diffusion_tag(cfg),model=checkpoint_relative(cfg))
        key=tuple(row[k] for k in GROUP+['seed']);assert key not in seen,key;seen.add(key)
        rows.append(row)
    rows.sort(key=lambda r:tuple(r[k] for k in ['experiment','policy','env','state_dim','n_demos','algo','seed']))
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in GROUP)].append(row)
    summaries=[];metric_counts={}
    for key,group in sorted(groups.items()):
        row=dict(zip(GROUP,key));row['n_seeds']=len(group)
        counts={}
        for metric in METRICS:
            values=[r[metric] for r in group if r[metric] is not None]
            counts[metric]=len(values)
            row[metric+'_mean']=statistics.mean(values) if values else None
            row[metric+'_std']=statistics.stdev(values) if len(values)>1 else (0. if values else None)
        summaries.append(row);metric_counts['|'.join(map(str,key))]=counts
    write_csv(root/'drone_dp_results.csv',COLS,rows)
    write_csv(root/'drone_dp_summary.csv',GROUP+['n_seeds']+[c for m in METRICS for c in [m+'_mean',m+'_std']],summaries)
    metadata=dict(generated_at=datetime.now(timezone.utc).isoformat(),rows=len(rows),groups=len(summaries),
                  completed_runs_by_experiment=dict(Counter(r['experiment'] for r in rows)),
                  metric_counts=metric_counts,
                  notes=['Only completed final-epoch 250-case evaluations are exported.',
                         'algo=bc denotes unregularized DP, matching the reference convention; env=drone.',
                         'success is a fraction; err_mean and err_median are final XYZ goal distances in meters, including failures.',
                         'err_median is computed from episode XYZ errors, not the saved XY median.',
                         'jacobian_norm is the estimated Fisher trace; null_ratio uses the existing 800 held-out-state diagnostic.',
                         'Absent metrics are blank. Nuclear norm and full-batch null-space diagnostics were not recorded.',
                         'rho is blank when not recorded in config; cond=0 for no scramble, 3 for E4.',
                         'Summary uses all available completed runs per cell, sample std (ddof=1), and full numeric precision.',
                         'Metric-specific sample counts are recorded here because diagnostics may lag evaluation.',
                         'Current manuscript figures read merged CSVs and show mean and SEM across available seeds.'])
    tmp=root/'csv_export_metadata.json.tmp';tmp.write_text(json.dumps(metadata,indent=2)+'\n');tmp.replace(root/'csv_export_metadata.json')
    if (root/'drone_mlp_results.csv').exists():
        from experiments.merge_policy_results import merge
        merge(root)
    else:
        write_csv(root/'results.csv',COLS,rows)
        write_csv(root/'summary.csv',GROUP+['n_seeds']+[c for m in METRICS for c in [m+'_mean',m+'_std']],summaries)
    return metadata

if __name__=='__main__':
    m=collect();print(f"Exported {m['rows']} completed runs, {m['groups']} summary rows: {m['completed_runs_by_experiment']}")
