"""Use verified sequence IDs; do not fabricate scene/domain metadata."""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from feddecouple.data import read_labels, CLASSES

def partition_groups(groups, clients, alpha, seed):
    rng = np.random.default_rng(seed)
    keys = list(groups)
    if len(keys) < clients:
        raise ValueError('Need at least as many verified sequence groups as clients')
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)
    counts = {g: np.bincount([o['category_id']-1 for r in rs for o in r['objects']], minlength=10).astype(float) for g,rs in groups.items()}
    target = (rng.dirichlet([alpha]*clients,size=10).T if alpha else np.full((clients,10),1/clients))
    target *= np.sum(list(counts.values()),axis=0)[None,:]
    assigned = np.zeros((clients,10)); sizes = np.zeros(clients)
    result = [[] for _ in range(clients)]
    total_images = sum(map(len,groups.values()))
    for idx,g in enumerate(keys):
        if idx < clients:
            chosen = idx  # ensure non-empty clients; this constraint is reported
        else:
            # Minimum increase of standardized squared deviation from target counts.
            denom = np.maximum(target,1.)
            costs = (((assigned+counts[g]-target)**2-(assigned-target)**2)/denom).sum(1)
            costs += .05*((sizes+len(groups[g]))**2-sizes**2)/(total_images/clients)**2
            chosen = int(np.argmin(costs))
        result[chosen].extend(groups[g]); assigned[chosen] += counts[g]; sizes[chosen] += len(groups[g])
    return result, assigned.astype(int).tolist(), target.tolist()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',required=True, help='VisDrone2019-DET-train directory')
    p.add_argument('--val',required=True, help='Official labelled val directory; split into dev/final by sequence')
    p.add_argument('--groups',required=True, help='CSV columns split,file,group. Split=train or val. Group IDs must be globally consistent.')
    p.add_argument('--clients',type=int,default=5)
    p.add_argument('--alpha',type=float,default=.5,help='0=balanced group reference; positive=group count constrained Dirichlet')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--dev-fraction',type=float,default=.5)
    p.add_argument('--output',default='data/partition.json')
    a=p.parse_args()
    if a.alpha < 0 or not 0 < a.dev_fraction < 1: p.error('Invalid alpha/dev-fraction')
    with open(a.groups,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    mapping={(r['split'],r['file']):r['group'] for r in rows}
    if len(mapping)!=len(rows) or any(not g.strip() for g in mapping.values()):
        raise ValueError('Duplicate image rows or empty group IDs')
    splits={}
    image_id=0
    for split,folder in [('train',a.train),('val',a.val)]:
        groups=defaultdict(list)
        for image in sorted((Path(folder)/'images').glob('*')):
            if image.suffix.lower() not in ('.jpg','.jpeg','.png'): continue
            key=(split,image.name)
            if key not in mapping: raise ValueError(f'Missing verified sequence ID: {key}')
            with Image.open(image) as im: width,height=im.size
            objects,ignored=read_labels(Path(folder)/'annotations'/f'{image.stem}.txt',width,height)
            image_id+=1
            groups[mapping[key]].append({'id':image_id,'image':str(image.resolve()),'group':mapping[key],
                 'width':width,'height':height,'objects':objects,'ignored':ignored})
        splits[split]=groups
    if set(splits['train']) & set(splits['val']):
        raise ValueError('Sequence leakage across official train/val. Rebuild group-disjoint source directories first.')
    clients,hist,target=partition_groups(splits['train'],a.clients,a.alpha,a.seed)
    keys=sorted(splits['val']); np.random.default_rng(a.seed).shuffle(keys)
    if len(keys)<2: raise ValueError('Need >=2 validation sequence groups for independent dev/final sets')
    n=min(len(keys)-1,max(1,round(len(keys)*a.dev_fraction)))
    dev=[r for k in keys[:n] for r in splits['val'][k]]
    test=[r for k in keys[n:] for r in splits['val'][k]]
    output={'protocol':'custom sequence-disjoint split; official train retained; labelled val split into dev and final',
            'evaluation':'COCO-style 100 maxDets; ignored-region IoA filtering (not official VisDrone metric)',
            'seed':a.seed,'alpha':a.alpha,'classes':CLASSES,'clients':clients,'dev':dev,'test':test,
            'statistics':{'class_histograms':hist,'dirichlet_targets':target,
             'images':[len(c) for c in clients], 'missing_class_fraction':[(np.array(h)==0).mean().item() for h in hist],
             'small_instances':[sum(o['bbox'][2]*o['bbox'][3]<32**2 for r in c for o in r['objects']) for c in clients]}}
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(output['statistics'],indent=2))

if __name__=='__main__': main()
