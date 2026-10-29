"""Explicit aggregation/export of paired diagnostics; no imputation or causal fit."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .reporting import flatten, write_csv


METRICS = ('conflict_before','conflict_after','conflict_improvement',
           'update_conflict_before','update_conflict_after','update_conflict_improvement',
           'gradient_cosine_before','gradient_cosine_after','task_gradient_cosine',
           'update_cosine_before','update_cosine_after',
           'task_loss_before','task_loss_after','post_task_loss_before','post_task_loss_after',
           'post_task_loss_improvement','post_prediction_discrepancy_before',
           'post_prediction_discrepancy_after','post_prediction_discrepancy_improvement',
           'actual_gradient_conflict','actual_update_conflict','actual_post_task_loss',
           'actual_post_prediction_discrepancy','diagnostic_probe_norm','diagnostic_reference_norm')


def summarize_diagnostics(rows):
    counts, means = {}, {}
    for key in METRICS:
        values=[r[key] for r in rows if r.get(key) is not None]
        counts[key]=len(values)
        means[key]=sum(values)/len(values) if values else None
    return dict(schema_version=2,sampled_steps=len(rows),
                aggregation='unweighted mean over valid sampled local steps; not independent seeds',
                client_sample_counts=dict(Counter(str(r['client']) for r in rows)),
                means=means,valid_counts=counts)


def export(folder):
    folder=Path(folder)
    cfg=json.loads((folder/'config.json').read_text())
    rounds=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines() if s.strip()]
    complete={r['round'] for r in rounds}
    samples=[];seen=set()
    for line in (folder/'steps.jsonl').open(encoding='utf-8'):
        row=json.loads(line)
        if row.get('diagnostic_schema_version')!=2 or row['round'] not in complete:continue
        identity=row['diagnostic_sample_id']
        if identity in seen:raise ValueError(f'Duplicate diagnostic sample: {identity}')
        seen.add(identity)
        samples.append({'method':cfg['method'],'seed':cfg['seed'],**flatten(row)})
    if not samples:raise ValueError('No schema-v2 diagnostics in complete rounds; historical logs cannot be reconstructed')
    write_csv(folder/'diagnostic_samples.csv',samples)
    output=[];by_round={r['round']:r for r in rounds}
    for row in rounds:
        if not row.get('diagnostics',{}).get('sampled_steps'):continue
        previous=by_round.get(row['round']-1);following=by_round.get(row['round']+1)
        current=set(row['clients']);old=set(previous['clients']) if previous else set()
        record={'method':cfg['method'],'seed':cfg['seed'],'round':row['round']+1,
                'local_steps':row['steps'],'participant_jaccard':len(current&old)/len(current|old) if previous else None,
                **flatten(row['diagnostics'],'diagnostics')}
        for metric in ('dispersion','direction_cosine','server_lr'):
            record[f'current_{metric}']=row.get(metric)
            record[f'next_{metric}']=following.get(metric) if following else None
        if following:
            nxt=set(following['clients'])
            record['next_local_steps']=following['steps']
            record['next_participant_jaccard']=len(current&nxt)/len(current|nxt)
        output.append(record)
    write_csv(folder/'diagnostic_trajectory_pairs.csv',output)
    return len(samples),len(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();print('Exported samples, rounds:',export(a.run_dir))


if __name__=='__main__':main()
