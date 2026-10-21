import contextlib
import io
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import VisDrone, collate, CLASSES

def coco_ground_truth(records):
    from pycocotools.coco import COCO
    annotations=[]
    for r in records:
        for o in r['objects']:
            annotations.append({'id':len(annotations)+1,'image_id':r['id'],'category_id':o['category_id'],
                                'bbox':o['bbox'],'area':o['bbox'][2]*o['bbox'][3],'iscrowd':0})
    c=COCO()
    c.dataset={'info':{},'images':[{'id':r['id'],'width':r['width'],'height':r['height']} for r in records],
               'annotations':annotations,'categories':[{'id':i+1,'name':n} for i,n in enumerate(CLASSES)]}
    c.createIndex()
    return c

@torch.no_grad()
def evaluate(model, records, cfg, prediction_path=None):
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    import json
    from pathlib import Path
    prior=model.training
    model.eval()
    predictions=[]
    by_id={r['id']:r for r in records}
    try:
        loader=DataLoader(VisDrone(records,cfg.image_size),batch_size=cfg.batch_size,collate_fn=collate,num_workers=cfg.workers)
        evaluated=0
        started=time.perf_counter()
        last_progress=started
        for batch_index,(images,targets) in enumerate(loader):
            out=model(images.to(cfg.device))
            for i,t in enumerate(targets):
                scores,flat=out['pred_logits'][i].sigmoid().flatten().topk(min(300,out['pred_logits'][i].numel()))
                labels=flat%10+1; query=flat//10
                boxes=out['pred_boxes'][i,query].clone()
                boxes[:,:2]-=boxes[:,2:]/2
                h,w=t['orig_size'].tolist()
                boxes*=boxes.new_tensor([w,h,w,h])
                image_id=t['image_id'].item()
                ignored=by_id[image_id]['ignored']
                for score,label,box in zip(scores.tolist(),labels.tolist(),boxes.tolist()):
                    x,y,bw,bh=box
                    # Declared approximation: excluded regions use intersection over detection area.
                    if any(max(0,min(x+bw,ix+iw)-max(x,ix))*max(0,min(y+bh,iy+ih)-max(y,iy))/max(bw*bh,1e-12)>.5 for ix,iy,iw,ih in ignored):
                        continue
                    predictions.append({'image_id':image_id,'category_id':label,'bbox':box,'score':score})
            evaluated+=len(targets)
            now=time.perf_counter()
            if batch_index==0 or (batch_index+1)%10==0 or evaluated==len(records) or now-last_progress>=10:
                elapsed=now-started
                eta=elapsed/evaluated*(len(records)-evaluated)
                print(f'    Eval {evaluated}/{len(records)} images | '
                      f'batch {batch_index+1}/{len(loader)} | '
                      f'elapsed={elapsed:.1f}s | ETA={eta:.1f}s',flush=True)
                last_progress=now
    finally:
        model.train(prior)
    if prediction_path:
        Path(prediction_path).write_text(json.dumps(predictions),encoding='utf8')
    print('    Computing COCO metrics...',flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gt=coco_ground_truth(records)
        if predictions:
            result=gt.loadRes(predictions)
        else:
            result=COCO(); result.dataset={**gt.dataset,'annotations':[]}; result.createIndex()
        evaluator=COCOeval(gt,result,'bbox')
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    stats=evaluator.stats
    def valid(value): return None if value<0 else float(value*100)
    per_class={}
    precision=evaluator.eval['precision']
    for i,name in enumerate(CLASSES):
        v=precision[:,:,i,0,-1]; v=v[v>=0]
        per_class[name]=None if not len(v) else float(v.mean()*100)
    return {'AP50_95':valid(stats[0]),'AP50':valid(stats[1]),'AP_small':valid(stats[3]),
            'AP_medium':valid(stats[4]),'AP_large':valid(stats[5]),'per_class':per_class,
            'protocol':'COCO-style, original-pixel areas, maxDets=100, ignored-region IoA filter; not official VisDrone AP'}
