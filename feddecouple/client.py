import copy
import time
import torch
import torch.nn.functional as F
from .backend import task_loss, optimizer_for
from .config import METHODS
from .proximal import make_pairs, detection_discrepancy, sol_discrepancy
from .state import params, buffers, restore_buffers, rng_state, restore_rng, grads, dot, conflict

def one_step(model, teacher, criterion, optimizer, images, targets, cfg,
             controls=None, diagnose=False):
    rule = METHODS[cfg.method][0]
    ps = params(model)
    old = {n: p.detach().clone() for n, p in ps.items()}
    saved_buffers, rng = buffers(model), rng_state()
    logs = {'pairs': 0, 'dcls': 0., 'dbox': 0., 'prox_norm': 0., 'perturb_norm': 0.}
    probe = rule in ('joint', 'joint_query', 'cls', 'box', 'l2', 'sol', 'sol_adaptive', 'sam')
    gp, pairs, teacher_out = None, None, None
    try:
        if probe or rule == 'direct':
            if rule == 'l2':
                ref = dict(teacher.named_parameters())
                loss = sum((p-ref[n]).square().sum()*.5 for n, p in ps.items())
            elif rule == 'sam':
                loss, _ = task_loss(model, criterion, images, targets)
            else:
                with torch.no_grad():
                    teacher_out = teacher(images, targets)
                out = model(images, targets)
                if rule in ('sol', 'sol_adaptive'):
                    loss = sol_discrepancy(out, teacher_out)
                else:
                    pairs = make_pairs(out, teacher_out, targets, criterion.matcher, rule == 'joint_query')
                    loss, extra = detection_discrepancy(out, teacher_out, pairs, cfg, rule)
                    logs.update(extra)
            gp = grads(loss, model)
            norm = dot(gp, gp)**.5
            logs['prox_norm'] = norm
            logs['zero_prox'] = norm < cfg.zero_threshold
            restore_buffers(model, saved_buffers)
            restore_rng(rng)
            delta = {n: torch.zeros_like(p) for n, p in ps.items()}
            if probe and norm >= cfg.zero_threshold:
                for n in ps:
                    delta[n] = cfg.rho*gp[n]/(norm+cfg.eps)
                    if rule == 'sol_adaptive':
                        ref = dict(teacher.named_parameters())[n]
                        delta[n] *= F.normalize((old[n]-ref).abs(), p=2, dim=0)
            logs['perturb_norm'] = dot(delta, delta)**.5
        else:
            delta = {n: torch.zeros_like(p) for n, p in ps.items()}

        # Diagnostics execute on independent model/optimizer copies and restore RNG.
        if diagnose and gp is not None:
            start = time.perf_counter()
            baseline = copy.deepcopy(model)
            baseline_opt = optimizer_for(baseline, cfg)
            baseline_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            baseline_opt.zero_grad(set_to_none=True)
            loss0, _ = task_loss(baseline, criterion, images, targets)
            loss0.backward()
            g0 = {n: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n, p in params(baseline).items()}
            if cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(baseline.parameters(), cfg.clip_grad)
            baseline_opt.step()
            u0 = {n: old[n]-p.detach() for n, p in params(baseline).items()}
            logs.update(conflict_before=conflict(g0, gp), update_conflict_before=conflict(u0, gp), task_loss_before=loss0.item())
            del baseline, baseline_opt
            restore_rng(rng)
            if images.is_cuda:
                torch.cuda.synchronize()
            logs['diagnostic_seconds'] = time.perf_counter()-start

        with torch.no_grad():
            for n, p in ps.items():
                p.copy_(old[n]+delta[n])
        optimizer.zero_grad(set_to_none=True)
        loss, out = task_loss(model, criterion, images, targets)
        objective = loss
        if rule == 'prox':
            ref = dict(teacher.named_parameters())
            objective = objective + cfg.mu*.5*sum((p-ref[n]).square().sum() for n,p in ps.items())
        elif rule == 'direct':
            prox, _ = detection_discrepancy(out, teacher_out, pairs, cfg)
            objective = objective + cfg.direct_weight*prox
        objective.backward()
        if rule == 'scaffold':
            global_c, local_c = controls
            for n, p in ps.items():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(global_c[n].to(p)-local_c[n].to(p))
        if diagnose and gp is not None:
            g1 = {n: torch.zeros_like(p) if p.grad is None else p.grad.detach() for n,p in ps.items()}
            logs['conflict_after'] = conflict(g1, gp)
            logs['delta_conflict'] = logs['conflict_after']-logs['conflict_before']
        # Restore exactly, before the single optimizer update. Task BN buffers stay.
        with torch.no_grad():
            for n, p in ps.items():
                p.copy_(old[n])
        if cfg.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite local objective')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in ps.values()):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        if diagnose and gp is not None:
            u1 = {n: old[n]-p.detach() for n,p in ps.items()}
            logs['update_conflict_after'] = conflict(u1, gp)
        logs['task_loss'] = loss.item()
        logs['objective'] = objective.item()
        return logs
    except BaseException:
        with torch.no_grad():
            for n,p in ps.items():
                p.copy_(old[n])
        restore_buffers(model, saved_buffers)
        restore_rng(rng)
        raise

class HyperLocal:
    """FedHyper-CL update from official update.py; uses actual parameter movements."""
    def __init__(self, optimizer, previous_global, participants):
        self.base = [g['lr'] for g in optimizer.param_groups]
        self.multiplier = 1.
        self.previous = previous_global
        self.global_update = previous_global
        self.participants = participants

    def step(self, movement, optimizer, batch_index):
        # Source skips the first batch in EACH epoch, and keeps multiplier/history.
        if batch_index == 0:
            return
        gamma = min(1.1, 1 + .1*dot(movement, self.global_update)/self.participants + dot(movement, self.previous))
        self.multiplier = min(10., max(.1, self.multiplier*gamma))
        for group, base in zip(optimizer.param_groups, self.base):
            group['lr'] = base*self.multiplier
        self.previous = {n: x.clone() for n,x in movement.items()}
