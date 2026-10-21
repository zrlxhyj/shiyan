"""Real RT-DETR forward/backward on synthetic images. NOT accuracy data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy
import json
import torch
from feddecouple.config import Config
from feddecouple.backend import build,optimizer_for
from feddecouple.client import one_step
from feddecouple.state import seed_all,snapshot

def main():
    torch.set_num_threads(4)
    seed_all(7)
    cfg=Config(device='cpu',image_size=160,batch_size=2)
    model,criterion=build(cfg)
    model.train(); teacher=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=optimizer_for(model,cfg)
    images=torch.rand(2,3,160,160)
    targets=[{'boxes':torch.tensor([[.5,.5,.2,.2],[.2,.3,.1,.1]]),'labels':torch.tensor([3,0])} for _ in range(2)]
    old=snapshot(teacher)
    result=[]
    for i in range(2):
        row=one_step(model,teacher,criterion,optimizer,images,targets,cfg)
        result.append(row)
        assert row['pairs']==4
        assert row['perturb_norm']<=cfg.rho+1e-5
    empty=[{'boxes':torch.empty(0,4),'labels':torch.empty(0,dtype=torch.long)} for _ in range(2)]
    row=one_step(model,teacher,criterion,optimizer,images,empty,cfg)
    result.append(row)
    assert row['pairs']==0
    assert all(torch.equal(v,old[n]) for n,v in teacher.state_dict().items())
    assert all(b.item()==3 for n,b in model.named_buffers() if n.endswith('num_batches_tracked'))
    Path('checks').mkdir(exist_ok=True)
    Path('checks/native_smoke.json').write_text(json.dumps({'synthetic_only':True,'not_experimental_results':True,'steps':result},indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
