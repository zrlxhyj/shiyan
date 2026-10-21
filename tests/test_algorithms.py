import copy
from dataclasses import replace
import torch
import pytest
from feddecouple.config import Config, METHODS
from feddecouple.client import one_step, HyperLocal
from feddecouple.server import Server
from feddecouple.proximal import make_pairs, detection_discrepancy
from feddecouple.state import snapshot, seed_all

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn=torch.nn.BatchNorm1d(4)
        self.head=torch.nn.Linear(4,14)
    def forward(self,x,targets=None):
        o=self.head(self.bn(x)).unsqueeze(1).repeat(1,2,1)
        return {'pred_logits':o[:,:,:10],'pred_boxes':o[:,:,10:].sigmoid()}

class Criterion:
    def matcher(self,out,targets):
        return [(torch.arange(min(2,len(t['labels']))),torch.arange(min(2,len(t['labels'])))) for t in targets]
    def __call__(self,out,targets):
        return {'loss':out['pred_logits'].square().mean()+out['pred_boxes'].square().mean()}

def batch():
    return torch.randn(3,4),[{'labels':torch.tensor([1]),'boxes':torch.tensor([[.5,.5,.2,.2]])} for _ in range(3)]

@pytest.mark.parametrize('method',list(METHODS))
def test_methods_update_once(method):
    seed_all(9); model=Toy().train(); teacher=copy.deepcopy(model).eval().requires_grad_(False)
    teacher_state=snapshot(teacher)
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    cfg=Config(method=method,device='cpu',clip_grad=0.)
    zeros={n:torch.zeros_like(p) for n,p in model.named_parameters()}
    old=snapshot(model)
    row=one_step(model,teacher,Criterion(),opt,*batch(),cfg,(zeros,zeros))
    assert model.bn.num_batches_tracked.item()==1
    assert row['perturb_norm']<=cfg.rho+1e-6
    assert any(not torch.equal(old[n],v) for n,v in model.state_dict().items() if v.is_floating_point())
    assert all(torch.equal(teacher_state[n],v) for n,v in teacher.state_dict().items())

def test_zero_radius_exact_fedavg():
    seed_all(1); a=Toy(); b=copy.deepcopy(a); teacher=copy.deepcopy(a).eval().requires_grad_(False)
    x,t=batch()
    for model,method in [(a,'fedavg'),(b,'feddecouple')]:
        seed_all(55)
        one_step(model,teacher,Criterion(),torch.optim.SGD(model.parameters(),lr=.01),x,t,Config(method=method,rho=0,clip_grad=0))
    for n,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[n],rtol=0,atol=0)

def test_empty_pairs_zero_and_grad():
    out={'pred_logits':torch.randn(1,2,10,requires_grad=True),'pred_boxes':torch.rand(1,2,4,requires_grad=True)}
    loss,stats=detection_discrepancy(out,out,[],Config())
    loss.backward()
    assert loss.item()==0 and stats['pairs']==0
    assert out['pred_logits'].grad.abs().sum()==0

def test_pairing_uses_target_identity_not_query_index():
    calls=iter([[(torch.tensor([0,1]),torch.tensor([1,0]))],[(torch.tensor([0,1]),torch.tensor([0,1]))]])
    pairs=make_pairs({}, {}, [],lambda *args:next(calls))
    assert pairs==[(0,1,0,0),(0,0,1,1)]

def test_identical_prediction_discrepancy():
    out={'pred_logits':torch.randn(1,2,10,requires_grad=True),'pred_boxes':torch.rand(1,2,4,requires_grad=True)}
    loss,_=detection_discrepancy(out,out,[(0,0,0,0),(0,1,1,1)],Config())
    assert abs(loss.item())<1e-7

def test_weighted_fedavg_and_hga_sign():
    model=torch.nn.Linear(1,1,bias=False)
    model.weight.data.fill_(10.)
    server=Server(Config(method='hga_only',hga_beta=.1))
    server.update(model,[{'weight':torch.tensor([[8.]])},{'weight':torch.tensor([[6.]])}],[1,3])
    torch.testing.assert_close(model.weight,torch.tensor([[6.5]]))
    row=server.update(model,[{'weight':torch.tensor([[5.5]])}],[1])
    assert abs(row['server_lr']-1.35)<1e-6
    assert abs(model.weight.item()-5.15)<1e-5

def test_hyper_gm_reference_formula():
    model=torch.nn.Linear(1,1,bias=False); model.weight.data.fill_(0)
    server=Server(Config(method='fedhyper_server'))
    server.update(model,[{'weight':torch.tensor([[.1]])}],[1])
    server.update(model,[{'weight':torch.tensor([[.3]])}],[1])
    # Source agg=.02, movement=.2*1.02 + .9*.02*.1 = .2058
    assert abs(model.weight.item()-.3058)<1e-6

def test_failure_restores_parameters_buffers():
    model=Toy(); teacher=copy.deepcopy(model).eval().requires_grad_(False); old=snapshot(model)
    class Bad(Criterion):
        def __call__(self,*args): raise RuntimeError('injected')
    with pytest.raises(RuntimeError):
        one_step(model,teacher,Bad(),torch.optim.SGD(model.parameters(),lr=.01),*batch(),Config())
    for n,v in model.state_dict().items(): torch.testing.assert_close(v,old[n],rtol=0,atol=0)

def test_diagnostics_do_not_change_training():
    from feddecouple.backend import optimizer_for
    cfg=Config(method='feddecouple',optimizer='sgd',momentum=.9,weight_decay=0,clip_grad=0)
    seed_all(3); a=Toy(); b=copy.deepcopy(a); teacher=copy.deepcopy(a).eval().requires_grad_(False)
    x,t=batch()
    for model,diagnose in [(a,False),(b,True)]:
        seed_all(44)
        one_step(model,teacher,Criterion(),optimizer_for(model,cfg),x,t,cfg,diagnose=diagnose)
    for n,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[n],rtol=0,atol=0)
