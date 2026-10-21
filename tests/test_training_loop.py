"""Synthetic integration tests; the fake evaluator is NOT a detector benchmark."""
import copy
import json
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from PIL import Image
import torch
from feddecouple.config import Config
from feddecouple.train import run
from tests.test_algorithms import Criterion

class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.bn=torch.nn.BatchNorm1d(3); self.head=torch.nn.Linear(3,14)
    def forward(self,x,targets=None):
        out=self.head(self.bn(x.mean((2,3)))).unsqueeze(1)
        return {'pred_logits':out[:,:,:10],'pred_boxes':out[:,:,10:].sigmoid()}

def make_fixture(tmp_path,monkeypatch):
    import feddecouple.train as training
    real_find=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name,*a,**kw: object() if name=='pycocotools' else real_find(name,*a,**kw))
    monkeypatch.setattr(training,'build',lambda cfg:(TinyDetector(),CriterionModule()))
    monkeypatch.setattr(training,'evaluate',lambda model,records,cfg: {'AP50_95':float(model.head.weight.square().mean())})
    clients=[]; index=0
    for c in range(7):
        records=[]
        for j in range(4):
            index+=1; path=tmp_path/f'{index}.png'
            Image.fromarray(np.full((8,8,3),30+index,dtype=np.uint8)).save(path)
            records.append({'id':index,'image':str(path),'width':8,'height':8,'group':str(c),
                            'objects':[{'bbox':[1,1,2,2],'category_id':1}],'ignored':[]})
        clients.append(records)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'clients':clients[:5],'dev':clients[5],'test':clients[6]}))
    return manifest

class CriterionModule(torch.nn.Module,Criterion):
    def forward(self,out,targets): return Criterion.__call__(self,out,targets)

@pytest.mark.parametrize('method',['feddecouple','fedhyper_full','scaffold','fedbn'])
def test_five_clients_two_rounds(tmp_path,monkeypatch,method):
    manifest=make_fixture(tmp_path,monkeypatch)
    cfg=Config(method=method,device='cpu',image_size=8,batch_size=2,rounds=2,local_epochs=1,
               manifest=str(manifest),output=str(tmp_path/'runs'),optimizer='sgd',momentum=0,weight_decay=0,clip_grad=0,
               persistent_optimizer=True)
    folder=run(cfg)
    rows=[json.loads(x) for x in (folder/'rounds.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(len(row['clients'])==3 for row in rows)
    checkpoint=torch.load(folder/'last.pt',weights_only=False)
    assert checkpoint['round']==1
    assert checkpoint['communication']>0
    assert len(checkpoint['opt_states'])>=3
    assert len(checkpoint['bn_states'])>=3 if method=='fedbn' else True
    run(cfg,resume=True)
    assert len((folder/'rounds.jsonl').read_text().splitlines())==2

def test_interrupted_resume_is_identical(tmp_path,monkeypatch):
    import feddecouple.train as training
    manifest=make_fixture(tmp_path,monkeypatch)
    cfg=Config(device='cpu',image_size=8,batch_size=2,rounds=3,local_epochs=1,
               manifest=str(manifest),output=str(tmp_path/'full'),persistent_optimizer=True)
    full=run(cfg)
    expected=torch.load(full/'last.pt',weights_only=False)['model']
    cfg.output=str(tmp_path/'resumed')
    evaluator=training.evaluate
    calls=0
    def interrupt(model,records,config):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('Simulated power loss after local updates')
        return evaluator(model,records,config)
    monkeypatch.setattr(training,'evaluate',interrupt)
    with pytest.raises(RuntimeError): run(cfg)
    monkeypatch.setattr(training,'evaluate',evaluator)
    folder=run(cfg,resume=True)
    actual=torch.load(folder/'last.pt',weights_only=False)['model']
    for n,v in expected.items(): torch.testing.assert_close(v,actual[n],rtol=0,atol=0)
    assert len((folder/'rounds.jsonl').read_text().splitlines())==3
