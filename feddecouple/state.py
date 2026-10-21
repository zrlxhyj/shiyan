import random
import numpy as np
import torch

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])

def buffers(model):
    return {n: b.detach().clone() for n, b in model.named_buffers()}

@torch.no_grad()
def restore_buffers(model, saved):
    for n, b in model.named_buffers():
        b.copy_(saved[n])

def params(model):
    return {n: p for n, p in model.named_parameters() if p.requires_grad}

def snapshot(model, cpu=False):
    return {n: t.detach().to('cpu' if cpu else t.device).clone() for n, t in model.state_dict().items()}

def bn_names(model):
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            names.update(f'{n}.{k}' if n else k for k in m.state_dict())
    return names

def dot(a, b):
    return sum((a[n].double() * b[n].double()).sum().item() for n in a)

def conflict(a, b, eps=1e-12):
    return max(0., -dot(a, b) / ((dot(a, a)*dot(b, b))**.5 + eps))

def grads(loss, model):
    ps = params(model)
    gs = torch.autograd.grad(loss, tuple(ps.values()), allow_unused=True)
    return {n: torch.zeros_like(p) if g is None else g.detach() for (n, p), g in zip(ps.items(), gs)}
