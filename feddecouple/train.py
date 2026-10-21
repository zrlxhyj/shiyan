import argparse
import copy
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .backend import build, optimizer_for, UPSTREAM_COMMIT
from .client import one_step, HyperLocal
from .config import Config, METHODS
from .data import VisDrone, collate, load_manifest
from .evaluate import evaluate
from .server import Server
from .state import seed_all, snapshot, params, bn_names
from .reporting import export_training_tables

def sync(cfg):
    if cfg.device.startswith('cuda'): torch.cuda.synchronize()

def append_json(path, row):
    with open(path,'a',encoding='utf8') as f: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def run(cfg, resume=False):
    cfg.validate()
    import importlib.util
    if importlib.util.find_spec('pycocotools') is None:
        raise RuntimeError('Missing pycocotools: install requirements.txt in the training environment before starting a run')
    manifest=load_manifest(cfg.manifest,cfg.clients)
    manifest_hash=hashlib.sha256(Path(cfg.manifest).read_bytes()).hexdigest()
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; set device=cpu only for debugging')
    folder=Path(cfg.output)/cfg.method/f'seed_{cfg.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint_path=folder/'last.pt'
    if (folder/'rounds.jsonl').exists() and not resume:
        raise FileExistsError(f'{folder} already contains a run; choose another output or --resume')
    if resume:
        existing = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if existing['manifest_hash'] != manifest_hash or existing['config'] != asdict(cfg):
            raise ValueError('Resume requires identical configuration and manifest')
        del existing
    seed_all(cfg.seed)
    model,criterion=build(cfg)
    model.to(cfg.device); criterion.to(cfg.device)
    cfg.dump(folder/'config.json')
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
         'device':torch.cuda.get_device_name() if cfg.device.startswith('cuda') else 'cpu',
         'upstream_expected_commit':UPSTREAM_COMMIT,'manifest_sha256':manifest_hash,
         'actual_participants':cfg.selected_count,'actual_participation':cfg.selected_count/cfg.clients,
         'precision':'FP32','simulation':'sequential clients on one device',
         'pretrained':cfg.pretrained or 'random initialization (pilot only; provide same pretrained checkpoint for paper runs)'}
    (folder/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf8')
    selection=random.Random(cfg.seed+200000)
    schedule=[selection.sample(range(cfg.clients),cfg.selected_count) for _ in range(cfg.rounds)]
    (folder/'participation.json').write_text(json.dumps(schedule),encoding='utf8')
    server=Server(cfg)
    local_bn=cfg.bn_mode=='local' or cfg.method=='fedbn'
    excluded=bn_names(model) if local_bn else set()
    bn_states={}; opt_states={}; controls={}
    global_c={n:torch.zeros_like(p,device='cpu') for n,p in params(model).items()} if cfg.method=='scaffold' else {}
    start_round=0; best=-1.; communication=0; elapsed_total=0.
    if resume:
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        if ck['manifest_hash']!=manifest_hash or ck['config']!=asdict(cfg):
            raise ValueError('Resume requires identical configuration and manifest')
        model.load_state_dict(ck['model']); server.__dict__.update(ck['server'])
        bn_states, opt_states, controls, global_c=ck['bn_states'],ck['opt_states'],ck['controls'],ck['global_c']
        start_round=ck['round']+1; best=ck['best']; communication=ck['communication']; elapsed_total=ck['elapsed_total']
        # Recover from a crash after log append but before checkpoint commit.
        for file in ('rounds.jsonl','steps.jsonl'):
            path=folder/file
            if path.exists():
                rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
                path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows if r['round']<start_round),encoding='utf8')
    export_training_tables(folder)
    for round_id in range(start_round,cfg.rounds):
        sync(cfg); started=time.perf_counter()
        print(f'\n[{cfg.method}] Round {round_id+1}/{cfg.rounds} | '
              f'clients: {[i+1 for i in schedule[round_id]]} (1-based)',flush=True)
        if cfg.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        global_state=snapshot(model,cpu=True)
        states=[]; sizes=[]; losses=[]; steps_total=0; diagnostics_seconds=0.
        updates_c=[]
        for client_id in schedule[round_id]:
            # Identical per-client data, augmentation and dropout RNG across methods.
            seed_all(cfg.seed+100003*round_id+997*client_id)
            student=copy.deepcopy(model)
            if local_bn and client_id in bn_states:
                own=student.state_dict(); own.update(bn_states[client_id]); student.load_state_dict(own)
            for name,b in student.named_buffers():
                if name.endswith('num_batches_tracked'): b.zero_()
            rule=METHODS[cfg.method][0]
            teacher=None
            if rule not in ('plain','hyper_local','scaffold','sam'):
                teacher=copy.deepcopy(student).eval()
                teacher.requires_grad_(False)
            student.train()
            optimizer=optimizer_for(student,cfg)
            if cfg.persistent_optimizer and client_id in opt_states:
                optimizer.load_state_dict(opt_states[client_id])
            dataset=VisDrone(manifest['clients'][client_id],cfg.image_size,train=True)
            loader=DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.workers,collate_fn=collate,
                              generator=torch.Generator().manual_seed(cfg.seed+100003*round_id+997*client_id))
            prev={n:torch.zeros_like(p,device='cpu') for n,p in params(student).items()}
            if server.previous is not None:
                prev.update({n:-d for n,d in server.previous.items()})
            hyper=HyperLocal(optimizer,prev,cfg.selected_count) if METHODS[cfg.method][0]=='hyper_local' else None
            ci=controls.get(client_id,{n:torch.zeros_like(p,device='cpu') for n,p in params(student).items()})
            lr_sums={n:0. for n in params(student)}
            name_by_id={id(p):n for n,p in params(student).items()}
            steps=0
            total_steps=len(loader)*cfg.local_epochs
            if cfg.max_steps: total_steps=min(total_steps,cfg.max_steps)
            client_started=time.perf_counter()
            last_progress=client_started
            print(f'  Client {client_id+1}/{cfg.clients} | images={len(dataset)} | '
                  f'training steps={total_steps}',flush=True)
            for epoch in range(cfg.local_epochs):
                for batch_index,(images,targets) in enumerate(loader):
                    images=images.to(cfg.device)
                    targets=[{k:v.to(cfg.device) for k,v in t.items()} for t in targets]
                    before={n:p.detach().cpu().clone() for n,p in params(student).items()} if hyper else None
                    for group in optimizer.param_groups:
                        for p in group['params']: lr_sums[name_by_id[id(p)]]+=group['lr']
                    diagnostic=bool(cfg.diagnostic_every and steps%cfg.diagnostic_every==0)
                    row=one_step(student,teacher,criterion,optimizer,images,targets,cfg,(global_c,ci),diagnostic)
                    row.update(round=round_id,client=client_id,epoch=epoch,step=steps,local_lr=optimizer.param_groups[0]['lr'])
                    append_json(folder/'steps.jsonl',row)
                    diagnostics_seconds+=row.get('diagnostic_seconds',0)
                    if hyper:
                        movement={n:p.detach().cpu()-before[n] for n,p in params(student).items()}
                        hyper.step(movement,optimizer,batch_index)
                    losses.append(row['task_loss']); steps+=1
                    now=time.perf_counter()
                    if steps==1 or steps%10==0 or steps==total_steps or now-last_progress>=10:
                        elapsed=now-client_started
                        eta=elapsed/steps*(total_steps-steps)
                        print(f'    Epoch {epoch+1}/{cfg.local_epochs} | '
                              f'step {steps}/{total_steps} ({100*steps/total_steps:.1f}%) | '
                              f'loss={row["task_loss"]:.4f} | '
                              f'elapsed={elapsed:.1f}s | ETA={eta:.1f}s',flush=True)
                        last_progress=now
                    if cfg.max_steps and steps>=cfg.max_steps: break
                if cfg.max_steps and steps>=cfg.max_steps: break
            state=snapshot(student,cpu=True)
            if cfg.method=='scaffold':
                new_c={n:ci[n]-global_c[n]+(global_state[n]-state[n])/lr_sums[n] for n in ci}
                updates_c.append({n:new_c[n]-ci[n] for n in ci}); controls[client_id]=new_c
            if local_bn: bn_states[client_id]={n:state[n] for n in excluded}
            if cfg.persistent_optimizer:
                opt_states[client_id]=copy.deepcopy(optimizer.state_dict())
                for values in opt_states[client_id]['state'].values():
                    for k,v in values.items():
                        if torch.is_tensor(v): values[k]=v.cpu()
            states.append(state); sizes.append(len(dataset)); steps_total+=steps
            del student,teacher,optimizer
        print('  Aggregating client updates...',flush=True)
        trace=server.update(model,states,sizes,excluded)
        if cfg.method=='scaffold':
            for n in global_c: global_c[n]+=sum(u[n] for u in updates_c)/cfg.clients
        # Model payload, no protocol overhead; fixed frozen tensors also transmitted.
        payload=sum(t.numel()*t.element_size() for n,t in global_state.items() if n not in excluded and t.is_floating_point())
        communication+=2*len(states)*payload
        if cfg.method=='scaffold': communication+=2*len(states)*sum(t.numel()*t.element_size() for t in global_c.values())
        sync(cfg); train_seconds=time.perf_counter()-started
        elapsed_total+=train_seconds
        evaluation_started=time.perf_counter()
        metrics=None
        if (round_id+1)%cfg.eval_every==0 or round_id==cfg.rounds-1:
            print(f'  Evaluating dev set: {len(manifest["dev"])} images',flush=True)
            if local_bn:
                # FedBN has no single shared BN model: all client BN states evaluated on SAME dev.
                per=[]; base=snapshot(model,cpu=True)
                for k in range(cfg.clients):
                    merged=dict(base); merged.update(bn_states.get(k,{})); model.load_state_dict(merged)
                    per.append(evaluate(model,manifest['dev'],cfg))
                model.load_state_dict(base)
                metrics={'AP50_95':sum(m['AP50_95'] for m in per)/len(per),'client_bn_models':per,
                         'scope':'mean of client-specific BN models on common dev; not shared-model AP'}
            else: metrics=evaluate(model,manifest['dev'],cfg)
        sync(cfg)
        eval_seconds=time.perf_counter()-evaluation_started
        is_best=metrics is not None and metrics['AP50_95'] is not None and metrics['AP50_95']>best
        if is_best: best=metrics['AP50_95']
        trace.update(round=round_id,clients=schedule[round_id],steps=steps_total,loss=sum(losses)/len(losses),
                     train_seconds=train_seconds,diagnostic_seconds=diagnostics_seconds,eval_seconds=eval_seconds,
                     cumulative_train_seconds=elapsed_total,communication_bytes=communication,
                     peak_memory_gb=torch.cuda.max_memory_allocated()/1e9 if cfg.device.startswith('cuda') else None,dev=metrics)
        append_json(folder/'rounds.jsonl',trace)
        ck={'model':snapshot(model,cpu=True),'server':server.__dict__,'round':round_id,'best':best,
            'communication':communication,'elapsed_total':elapsed_total,'config':asdict(cfg),'manifest_hash':manifest_hash,
            'bn_states':bn_states,'opt_states':opt_states,'controls':controls,'global_c':global_c}
        temporary=folder/'checkpoint.tmp'
        torch.save(ck,temporary); temporary.replace(checkpoint_path)
        if is_best: torch.save(ck,folder/'best_dev.pt')
        export_training_tables(folder)
        print(f'  Round {round_id+1}/{cfg.rounds} saved to {folder} | '
              f'train={train_seconds:.1f}s | eval={eval_seconds:.1f}s | '
              f'mean loss={trace["loss"]:.4f}',flush=True)
        print(json.dumps({'method':cfg.method,'round':round_id+1,'dev':metrics,'server_lr':trace['server_lr']},ensure_ascii=False),flush=True)
    return folder

def main(default_method=None, argv=None):
    p=argparse.ArgumentParser(description='Run one federated method; results/<method>/seed_<seed> by default.')
    p.add_argument('--config', help='Optional JSON config; otherwise use configs/base.json')
    p.add_argument('--method',choices=list(METHODS))
    p.add_argument('--seed',type=int)
    p.add_argument('--rounds',type=int)
    p.add_argument('--batch-size',type=int)
    p.add_argument('--max-steps',type=int,help='Maximum steps per client per round; 0 means all batches')
    p.add_argument('--participation',type=float)
    p.add_argument('--output',help='Results root; method and seed subdirectories are added automatically')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if default_method and a.method and a.method != default_method:
        p.error(f'This entry runs {default_method}; use python -m {a.method}.train instead')
    cfg=Config.load(a.config or 'configs/base.json')
    cfg.method=default_method or a.method or ('feddecouple' if a.config is None else cfg.method)
    if a.config is None:
        cfg.output='results'
        if cfg.method=='scaffold':
            cfg.optimizer='sgd'; cfg.momentum=0.; cfg.weight_decay=0.; cfg.clip_grad=0.
            print('SCAFFOLD requires SGD: momentum=0, weight_decay=0, clip_grad=0. Use matched SGD for comparisons.',flush=True)
    for name in ('seed','rounds','batch_size','max_steps','participation','output'):
        value=getattr(a,name)
        if value is not None: setattr(cfg,name,value)
    cfg.validate()
    print(f'Method: {cfg.method} | rounds: {cfg.rounds} | batch: {cfg.batch_size} | '
          f'clients/round: {cfg.selected_count}/{cfg.clients} | '
          f'results: {Path(cfg.output)/cfg.method/f"seed_{cfg.seed}"}',flush=True)
    run(cfg,a.resume)

if __name__=='__main__': main()
