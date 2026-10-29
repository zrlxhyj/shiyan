"""Paired, counterfactual diagnostics on isolated model/optimizer/criterion copies."""
import copy
import hashlib
import math
import pickle
import time

import torch

from .backend import task_loss
from .proximal import make_pairs, detection_discrepancy
from .state import params, rng_state, restore_rng, grads, dot


def cosine(a, b, cfg):
    aa, bb = dot(a, a), dot(b, b)
    if aa <= cfg.zero_threshold**2 or bb <= cfg.zero_threshold**2:
        return None
    return max(-1., min(1., dot(a, b)/((aa*bb)**.5+cfg.eps)))


def paired_diagnostics(model, teacher, criterion, optimizer, images, targets, cfg):
    """Measure original/probed task branches; neither branch performs formal training.

    The common reference gradient is the joint prediction discrepancy against the
    frozen round-start teacher, irrespective of which of the two methods is trained.
    Probe gradients are evaluated at theta+delta, then applied to theta, matching
    the single formal optimizer update used by FedDecouple.
    """
    if cfg.method not in ('fedavg', 'feddecouple'):
        raise ValueError('Paired mechanism diagnostics currently support fedavg and feddecouple only')
    if teacher is None:
        raise ValueError('Paired diagnostics require the frozen round-start teacher')
    state = rng_state()
    if images.is_cuda:
        torch.cuda.synchronize(images.device)
    started = time.perf_counter()
    try:
        # Inputs and all mutable compute state are isolated too.
        x, y = images.detach().clone(), copy.deepcopy(targets)
        reference = copy.deepcopy(teacher).eval().requires_grad_(False)
        reference_criterion = copy.deepcopy(criterion)
        work = copy.deepcopy(model)
        restore_rng(state)
        with torch.no_grad():
            reference_out = reference(x.clone(), copy.deepcopy(y))
        restore_rng(state)
        out = work(x.clone(), copy.deepcopy(y))
        pairs = make_pairs(out, reference_out, copy.deepcopy(y), reference_criterion.matcher)
        discrepancy, extra = detection_discrepancy(out, reference_out, pairs, cfg, 'joint')
        reference_gradient = grads(discrepancy, work)
        norm = dot(reference_gradient, reference_gradient)**.5
        if not math.isfinite(norm):raise FloatingPointError('Nonfinite diagnostic reference gradient')
        delta = {n: cfg.rho*g/(norm+cfg.eps) if norm >= cfg.zero_threshold
                 else torch.zeros_like(g) for n, g in reference_gradient.items()}
        del out, discrepancy, work, reference

        def branch(probed):
            # Deep-copy the tuple so optimizer groups keep references to copied params.
            student, opt = copy.deepcopy((model, optimizer))
            crit = copy.deepcopy(criterion)
            ps = params(student)
            initial = {n:p.detach().clone() for n,p in ps.items()}
            with torch.no_grad():
                if probed:
                    for n,p in ps.items(): p.add_(delta[n])
            opt.zero_grad(set_to_none=True)
            restore_rng(state)
            loss, _ = task_loss(student, crit, x.clone(), copy.deepcopy(y))
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite diagnostic task loss')
            loss.backward()
            gradient = {n:torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
                        for n,p in ps.items()}
            if any(not torch.isfinite(g).all() for g in gradient.values()):
                raise FloatingPointError('Nonfinite diagnostic gradient')
            # Keep task-forward buffer updates, restore the parameter origin exactly.
            with torch.no_grad():
                for n,p in ps.items(): p.copy_(initial[n])
            if cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.clip_grad)
            opt.step()
            update = {n:initial[n]-p.detach() for n,p in ps.items()}
            gradient_cosine = cosine(gradient, reference_gradient, cfg)
            update_cosine = cosine(update, reference_gradient, cfg)
            # Use identical post-update RNG and loss/matcher starting state in both arms.
            post_criterion = copy.deepcopy(criterion)
            restore_rng(state)
            with torch.no_grad():
                post_loss, post_out = task_loss(student, post_criterion, x.clone(), copy.deepcopy(y))
                post_pairs = make_pairs(post_out, reference_out, copy.deepcopy(y), post_criterion.matcher)
                post_discrepancy, post_extra = detection_discrepancy(post_out, reference_out, post_pairs, cfg, 'joint')
            if not torch.isfinite(post_loss) or not torch.isfinite(post_discrepancy):
                raise FloatingPointError('Nonfinite post-update diagnostic')
            values = dict(gradient_cosine=gradient_cosine, update_cosine=update_cosine,
                          conflict=None if gradient_cosine is None else max(0., -gradient_cosine),
                          update_conflict=None if update_cosine is None else max(0., -update_cosine),
                          task_loss=float(loss.detach()), post_task_loss=float(post_loss),
                          post_prediction_discrepancy=float(post_discrepancy),
                          post_dcls=post_extra['dcls'],post_dbox=post_extra['dbox'],
                          post_pairs=post_extra['pairs'], update_norm=dot(update, update)**.5)
            return values, gradient

        original, g0 = branch(False)
        probed, g1 = branch(True)
        actual = original if cfg.method == 'fedavg' else probed
        digest = hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str((str(x.dtype),tuple(x.shape))).encode())
        for target in y:
            for key, value in sorted(target.items()):
                digest.update(key.encode())
                if torch.is_tensor(value):
                    digest.update(str((str(value.dtype),tuple(value.shape))).encode())
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
        logs = dict(diagnostic_schema_version=2,
                    diagnostic_reference='joint_discrepancy_to_frozen_round_start_teacher',
                    diagnostic_status='ok' if norm >= cfg.zero_threshold else 'zero_reference_gradient',
                    diagnostic_reference_norm=norm, diagnostic_pairs=extra['pairs'],
                    diagnostic_probe_norm=dot(delta, delta)**.5,
                    diagnostic_actual_branch='original' if cfg.method=='fedavg' else 'probed',
                    diagnostic_input_sha256=digest.hexdigest(),
                    diagnostic_rng_sha256=hashlib.sha256(pickle.dumps(state)).hexdigest(),
                    task_gradient_cosine=cosine(g0,g1,cfg))
        for suffix, values in [('before', original), ('after', probed)]:
            for key, value in values.items():
                logs[f'{key}_{suffix}'] = value
        # Positive improvement consistently means a reduction of the diagnostic.
        for key in ('conflict','update_conflict','post_task_loss','post_prediction_discrepancy'):
            before, after = original[key], probed[key]
            logs[f'{key}_improvement'] = None if before is None or after is None else before-after
        logs['delta_conflict'] = None if logs['conflict_improvement'] is None else -logs['conflict_improvement']
        logs['actual_gradient_conflict'] = actual['conflict']
        logs['actual_update_conflict'] = actual['update_conflict']
        logs['actual_post_task_loss'] = actual['post_task_loss']
        logs['actual_post_prediction_discrepancy'] = actual['post_prediction_discrepancy']
        if images.is_cuda: torch.cuda.synchronize(images.device)
        logs['diagnostic_seconds'] = time.perf_counter()-started
        return logs
    finally:
        # Includes failures; diagnostic forward/backward/step never consumes training RNG.
        restore_rng(state)
