"""Explicit final evaluation, separate from dev-set tuning."""
import argparse
import json
import hashlib
from pathlib import Path
import torch
from .config import Config
from .backend import build
from .data import load_manifest
from .evaluate import evaluate
from .reporting import export_final_table

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--device')
    a=p.parse_args()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=Config(**checkpoint['config'])
    if a.device: cfg.device=a.device
    manifest=load_manifest(cfg.manifest,cfg.clients)
    if hashlib.sha256(Path(cfg.manifest).read_bytes()).hexdigest()!=checkpoint['manifest_hash']:
        raise ValueError('Evaluation manifest differs from training manifest')
    model,_=build(cfg); model.load_state_dict(checkpoint['model']); model.to(cfg.device)
    out=Path(a.checkpoint).with_suffix('.final.json')
    if cfg.method=='fedbn' or cfg.bn_mode=='local':
        scores=[]
        for k in range(cfg.clients):
            state=dict(checkpoint['model']); state.update(checkpoint['bn_states'].get(k,{})); model.load_state_dict(state)
            scores.append(evaluate(model,manifest['test'],cfg))
        result={'client_bn_models':scores,'scope':'client-specific BN models on common final set',
                **{key:sum(s[key] for s in scores)/len(scores) if all(s[key] is not None for s in scores) else None for key in ['AP50_95','AP50','AP_small']}}
    else:
        result=evaluate(model,manifest['test'],cfg,out.with_suffix('.predictions.json'))
    if 'test_domains' in manifest and cfg.method!='fedbn' and cfg.bn_mode!='local':
        domains={name:evaluate(model,records,cfg) for name,records in manifest['test_domains'].items()}
        valid=[m['AP50_95'] for m in domains.values() if m['AP50_95'] is not None]
        result['domain_metrics']=domains
        result['client_mean_AP50_95']=sum(valid)/len(valid) if valid else None
    else:
        result['client_mean_AP50_95']=None
        result['client_metric_note']='No verified domain mapping for the evaluated model; local BN also requires a domain-to-client-model mapping.'
    result.update(method=cfg.method,seed=cfg.seed,round=checkpoint['round'],selection=Path(a.checkpoint).stem,
                  manifest_sha256=checkpoint['manifest_hash'],dataset_protocol=manifest['protocol'])
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    export_final_table(out, result)
    print(out)

if __name__=='__main__': main()
