"""Aggregate completed final evaluations; never fabricate missing cells."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import statistics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='runs'); p.add_argument('--output',default='results_summary.csv')
    a=p.parse_args(); groups=defaultdict(list)
    for file in Path(a.root).rglob('*.final.json'):
        r=json.loads(file.read_text(encoding='utf8'))
        # Parent experiment tag distinguishes sensitivity configs, not only method.
        groups[(str(file.parent.parent.parent),r['method'],r['selection'],r['manifest_sha256'])].append(r)
    fields=['experiment','method','selection','manifest','n_seeds','seeds','AP50_mean','AP50_std','AP50_95_mean','AP50_95_std','AP_small_mean','AP_small_std']
    with open(a.output,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for (experiment,method,selection,manifest),rows in sorted(groups.items()):
            seeds=[r['seed'] for r in rows]
            if len(seeds)!=len(set(seeds)): raise ValueError('Duplicate seed evaluations in one group')
            out=dict(experiment=experiment,method=method,selection=selection,manifest=manifest,n_seeds=len(rows),seeds=str(sorted(seeds)))
            for metric in ['AP50','AP50_95','AP_small']:
                values=[r.get(metric) for r in rows]
                if all(v is not None for v in values):
                    out[metric+'_mean']=statistics.mean(values)
                    out[metric+'_std']=statistics.stdev(values) if len(values)>1 else ''
                else: out[metric+'_mean']=out[metric+'_std']=''
            writer.writerow(out)
    print(a.output)

if __name__=='__main__': main()
