import copy
import json
import random

import numpy as np
import pytest
import torch

from feddecouple.backend import optimizer_for
from feddecouple.client import one_step
from feddecouple.config import Config
from feddecouple.diagnostics import paired_diagnostics
from feddecouple.state import snapshot, buffers, seed_all, rng_state
from tests.test_algorithms import Toy, Criterion, batch


class StochasticToy(Toy):
    def __init__(self):
        super().__init__()
        self.dropout=torch.nn.Dropout(.35)
        self.register_buffer('cache',torch.tensor([9.]),persistent=False)

    def forward(self,x,targets=None):
        return super().forward(self.dropout(x),targets)


class StatefulCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('calls',torch.tensor(0))

    def matcher(self,out,targets):
        self.calls.add_(1)
        return Criterion().matcher(out,targets)

    def forward(self,out,targets):
        self.calls.add_(1)
        # Exercise all RNG families without changing the objective.
        random.random();np.random.rand();torch.rand((),device=out['pred_logits'].device)
        return Criterion()(out,targets)


def assert_nested(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_nested(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_nested(x,y)
    else:assert a==b


@pytest.mark.parametrize('method',['fedavg','feddecouple'])
@pytest.mark.parametrize('optimizer',['adamw','sgd'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_paired_diagnostics_preserve_entire_training_state(method,optimizer,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    seed_all(8);model=StochasticToy().to(device).train()
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():teacher.head.weight.add_(.1)
    x,t=batch();x=x.to(device);t=[{k:v.to(device) for k,v in target.items()} for target in t]
    cfg=Config(method=method,optimizer=optimizer,device=device,clip_grad=.1,weight_decay=.01,momentum=.9)
    histories=[]
    for enabled in (False,True):
        student=copy.deepcopy(model);reference=copy.deepcopy(teacher);crit=StatefulCriterion().to(device)
        opt=optimizer_for(student,cfg)
        for i in range(3):
            seed_all(90+i)
            old={n:p.detach().clone() for n,p in student.named_parameters()}
            row=one_step(student,reference,crit,opt,x,t,cfg,diagnose=enabled)
            if enabled:
                assert row['diagnostic_actual_branch']==('original' if method=='fedavg' else 'probed')
                assert row['conflict_before'] is not None
                assert row['update_conflict_after'] is not None
                key='update_norm_before' if method=='fedavg' else 'update_norm_after'
                actual=sum((old[n]-p).double().square().sum().item() for n,p in student.named_parameters())**.5
                assert actual==pytest.approx(row[key],rel=1e-5,abs=1e-8)
                json.dumps(row,allow_nan=False)
        histories.append((snapshot(student),buffers(student),copy.deepcopy(opt.state_dict()),
                          snapshot(crit),snapshot(reference),rng_state(),
                          {n:p.grad.detach().clone() for n,p in student.named_parameters()}))
    assert_nested(histories[0],histories[1])


def test_zero_radius_produces_identical_counterfactual_arms():
    seed_all(3);model=StochasticToy().train();teacher=copy.deepcopy(model).eval().requires_grad_(False)
    cfg=Config(rho=0);opt=optimizer_for(model,cfg)
    row=paired_diagnostics(model,teacher,StatefulCriterion(),opt,*batch(),cfg)
    for metric in ('conflict','update_conflict','post_task_loss','post_prediction_discrepancy'):
        assert row[metric+'_improvement']==0


def test_zero_reference_is_missing_not_false_zero_conflict():
    model=Toy().eval();teacher=copy.deepcopy(model).eval().requires_grad_(False)
    cfg=Config(lambda_cls=0,lambda_box=0)
    row=paired_diagnostics(model,teacher,Criterion(),optimizer_for(model,cfg),*batch(),cfg)
    assert row['diagnostic_status']=='zero_reference_gradient'
    assert row['conflict_before'] is None and row['update_conflict_after'] is None
    assert row['conflict_improvement'] is None
    json.dumps(row,allow_nan=False)


def test_diagnostic_failure_restores_rng_and_leaves_live_state_untouched():
    class Bad(StatefulCriterion):
        def forward(self,*args):
            self.calls.add_(1);torch.rand(3);random.random();np.random.rand()
            raise RuntimeError('diagnostic injected failure')
    seed_all(23);model=StochasticToy().train();teacher=copy.deepcopy(model).eval().requires_grad_(False)
    cfg=Config();opt=optimizer_for(model,cfg);crit=Bad();x,t=batch()
    before=(snapshot(model),buffers(model),copy.deepcopy(opt.state_dict()),snapshot(teacher),snapshot(crit),rng_state(),x.clone(),copy.deepcopy(t))
    with pytest.raises(RuntimeError,match='injected failure'):
        paired_diagnostics(model,teacher,crit,opt,x,t,cfg)
    after=(snapshot(model),buffers(model),opt.state_dict(),snapshot(teacher),snapshot(crit),rng_state(),x,t)
    assert_nested(before,after)
