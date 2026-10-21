import math
import torch
from .state import params, dot
from .config import METHODS

class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lr = cfg.server_lr
        self.previous = None
        self.hyper_history = None
        self.agg = 0.
        self.m = None
        self.v = None

    @torch.no_grad()
    def update(self, model, local_states, sizes, excluded=()):
        cfg = self.cfg
        rule = METHODS[cfg.method][1]
        ps = params(model)
        keys = [n for n in ps if n not in excluded]
        weights = [s/sum(sizes) for s in sizes]
        # CPU accumulators avoid retaining all clients on GPU.
        old = {n: ps[n].detach().cpu() for n in keys}
        delta = {n: sum(w*(old[n]-state[n]) for w,state in zip(weights,local_states)) for n in keys}
        energy = sum(w*sum((old[n]-s[n]).double().square().sum().item() for n in keys) for w,s in zip(weights,local_states))
        norm2 = dot(delta, delta)
        inner, cosine = None, None
        if self.previous is not None:
            inner = dot(delta, self.previous)
            prev2 = dot(self.previous, self.previous)
            if norm2 > cfg.eps and prev2 > cfg.eps:
                cosine = inner/(math.sqrt(norm2*prev2)+cfg.eps)
        if rule in ('hga', 'cosine') and self.previous is not None:
            signal = inner if rule == 'hga' else (cosine or 0.)
            self.lr = min(cfg.server_max, max(cfg.server_min, self.lr+cfg.hga_beta*signal))
        movement = {n: self.lr*d for n,d in delta.items()}
        if rule in ('hyper_g', 'hyper_gm'):
            if self.hyper_history is not None:
                self.agg += cfg.hyper_rate*dot(delta, self.hyper_history)
                self.agg = min(4., max(-.8 if rule == 'hyper_g' else -.2, self.agg))
                movement = {n: (1+self.agg)*d + (.9*self.agg*self.hyper_history[n] if rule == 'hyper_gm' else 0) for n,d in delta.items()}
            else:
                movement = {n: d.clone() for n,d in delta.items()}
            self.lr = 1+self.agg
            self.hyper_history = {n: d.clone() + (.9*self.hyper_history[n] if self.hyper_history is not None and rule == 'hyper_gm' else 0) for n,d in delta.items()}
        elif rule in ('momentum', 'adam', 'adagrad'):
            if self.m is None:
                self.m = {n: torch.zeros_like(d) for n,d in delta.items()}
                self.v = {n: torch.zeros_like(d) for n,d in delta.items()}
            for n,d in delta.items():
                if rule == 'momentum':
                    self.m[n] = cfg.server_beta1*self.m[n]+d
                    movement[n] = self.lr*self.m[n]
                else:
                    self.m[n] = cfg.server_beta1*self.m[n]+(1-cfg.server_beta1)*d
                    self.v[n] = (self.v[n]+d.square() if rule == 'adagrad' else cfg.server_beta2*self.v[n]+(1-cfg.server_beta2)*d.square())
                    movement[n] = self.lr*self.m[n]/(self.v[n].sqrt()+cfg.server_tau)
        for n in keys:
            ps[n].sub_(movement[n].to(ps[n]))
        # Client snapshots contain only persistent buffers from state_dict().
        # Fixed caches such as RT-DETR position embeddings are not aggregated.
        persistent_keys = model.state_dict().keys()
        for n,b in model.named_buffers():
            if n in excluded or n not in persistent_keys:
                continue
            if b.is_floating_point():
                b.copy_(sum(w*s[n] for w,s in zip(weights,local_states)).to(b))
            elif n.endswith('num_batches_tracked'):
                b.zero_()
        self.previous = {n:d.clone() for n,d in delta.items()}
        return {'aggregate_norm': math.sqrt(norm2), 'dispersion': max(0., (energy-norm2)/(energy+cfg.eps)),
                'direction_cosine': cosine, 'trajectory_inner': inner, 'server_lr': self.lr,
                'global_move_norm': math.sqrt(dot(movement,movement)),
                'at_lower_bound': rule in ('hga','cosine') and self.lr == cfg.server_min,
                'at_upper_bound': rule in ('hga','cosine') and self.lr == cfg.server_max}
