"""Generate reviewable jobs first; execute only with --execute."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feddecouple.config import Config, METHODS

MAIN=['fedavg','fedprox','fedsol','fedhyper_g','fedhyper_server','fedhyper_full','fedsol_fedhyper','feddecouple']
ABLATIONS=['fedavg','pgd_only','hga_only','feddecouple','probe_l2','probe_cls','probe_box','direct_joint','same_query','cosine_hga']

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/base.json')
    p.add_argument('--suite',choices=['main','ablations','extra','sensitivity','all'],default='main')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--output',default='experiment_jobs'); p.add_argument('--execute',action='store_true')
    a=p.parse_args(); base=Config.load(a.config)
    jobs=[]
    names=MAIN if a.suite=='main' else ABLATIONS if a.suite=='ablations' else [n for n in METHODS if n not in MAIN+ABLATIONS] if a.suite=='extra' else list(METHODS) if a.suite=='all' else []
    for name in names:
        cfg=replace(base,method=name)
        if name=='scaffold':
            cfg=replace(cfg,optimizer='sgd',momentum=0.,weight_decay=0.,clip_grad=0.)
            # Add matched SGD FedAvg; never silently compare a different local optimizer.
            jobs.append(('scaffold_matched_fedavg',replace(cfg,method='fedavg')))
        jobs.append((name,cfg))
    if a.suite in ('sensitivity','all'):
        for rho in [0.,.01,.05,.1]: jobs.append((f'rho_{rho}',replace(base,rho=rho)))
        for beta in [.0001,.001,.01,.1,1.]: jobs.append((f'beta_{beta}',replace(base,hga_beta=beta)))
        for lr in [.1,.5,1.,1.5,2.]: jobs.append((f'fixed_{lr}',replace(base,method='pgd_only',server_lr=lr)))
        for ratio in [.25,.5,1.,2.,4.]: jobs.append((f'ratio_{ratio}',replace(base,method='pgd_only',lambda_cls=2*ratio/(1+ratio),lambda_box=2/(1+ratio))))
        for temperature in [1.,2.,4.]: jobs.append((f'temperature_{temperature}',replace(base,temperature=temperature)))
        for beta in [.05,.1,.2]: jobs.append((f'smoothl1_{beta}',replace(base,smooth_l1_beta=beta)))
        for rate in [.2,.5,1.]: jobs.append((f'participation_{rate}',replace(base,participation=rate)))
        for method in ['fedavg','pgd_only','feddecouple']:
            jobs.append((f'{method}_persistent',replace(base,method=method,persistent_optimizer=True)))
            jobs.append((f'{method}_localbn',replace(base,method=method,bn_mode='local')))
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    commands=[]
    for tag,cfg in jobs:
        for seed in a.seeds:
            cfg=replace(cfg,seed=seed,output=str(Path(base.output)/tag))
            cfg.validate(); path=output/f'{tag}_seed{seed}.json'; cfg.dump(path)
            commands.append([sys.executable,'-m','feddecouple.train','--config',str(path)])
    (output/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf8')
    print(f'{len(commands)} jobs. Configs: {output}. No training started unless --execute is specified.')
    if a.execute:
        for cmd in commands: subprocess.run(cmd,check=True)

if __name__=='__main__': main()
