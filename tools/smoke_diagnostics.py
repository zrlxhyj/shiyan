"""Real RT-DETR synthetic smoke: diagnostic ON/OFF must preserve the formal update."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from feddecouple.backend import build,optimizer_for
from feddecouple.client import one_step
from feddecouple.config import Config
from feddecouple.state import seed_all,snapshot


def main():
    torch.set_num_threads(4);seed_all(7)
    cfg=Config(device='cpu',image_size=160,batch_size=2)
    base,criterion=build(cfg)
    base.train();teacher=copy.deepcopy(base).eval().requires_grad_(False)
    images=torch.rand(2,3,160,160)
    targets=[{'boxes':torch.tensor([[.5,.5,.2,.2],[.2,.3,.1,.1]]),
              'labels':torch.tensor([3,0])} for _ in range(2)]
    results=[]
    for method in ('fedavg','feddecouple'):
        c=replace(cfg,method=method);states=[]
        for diagnose in (False,True):
            model=copy.deepcopy(base);crit=copy.deepcopy(criterion)
            opt=optimizer_for(model,c);seed_all(44)
            row=one_step(model,teacher,crit,opt,images,targets,c,diagnose=diagnose)
            states.append(snapshot(model,cpu=True))
            if diagnose:results.append({'method':method,**row})
            del model,opt,crit
        for name,tensor in states[0].items():
            torch.testing.assert_close(tensor,states[1][name],rtol=0,atol=0)
        print(method,': exact formal model/buffer equality with diagnostics ON/OFF',flush=True)
    Path('checks').mkdir(exist_ok=True)
    Path('checks/diagnostic_smoke.json').write_text(json.dumps({'synthetic_only':True,
        'not_accuracy_results':True,'on_off_exact_equality':True,'diagnostics':results},indent=2),encoding='utf-8')


if __name__=='__main__':main()
