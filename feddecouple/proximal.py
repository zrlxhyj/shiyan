"""Manuscript equations 9-16. Pair indices are detached and fixed per step."""
import torch
import torch.nn.functional as F

@torch.no_grad()
def make_pairs(student, teacher, targets, matcher, same_query=False):
    a = matcher(student, targets)
    b = matcher(teacher, targets)
    pairs = []
    for image, ((sq, sj), (tq, tj)) in enumerate(zip(a, b)):
        smap = dict(zip(sj.tolist(), sq.tolist()))
        tmap = dict(zip(tj.tolist(), tq.tolist()))
        for target in sorted(smap.keys() & tmap.keys()):
            pairs.append((image, smap[target], smap[target] if same_query else tmap[target], target))
    return pairs

def detection_discrepancy(student, teacher, pairs, cfg, component='joint'):
    zero = student['pred_logits'].sum()*0 + student['pred_boxes'].sum()*0
    if not pairs:
        return zero, {'pairs': 0, 'dcls': 0., 'dbox': 0.}
    device = student['pred_logits'].device
    ix = torch.tensor(pairs, device=device)
    bi, si, ti = ix[:, 0], ix[:, 1], ix[:, 2]
    q = (student['pred_logits'][bi, si].float()/cfg.temperature).sigmoid().clamp(cfg.probability_eps, 1-cfg.probability_eps)
    p = (teacher['pred_logits'][bi, ti].detach().float()/cfg.temperature).sigmoid().clamp(cfg.probability_eps, 1-cfg.probability_eps)
    cls = (p*(p.log()-q.log()) + (1-p)*((1-p).log()-(1-q).log())).mean()*cfg.temperature**2
    box = F.smooth_l1_loss(student['pred_boxes'][bi, si].float(), teacher['pred_boxes'][bi, ti].detach().float(), beta=cfg.smooth_l1_beta)
    loss = (0 if component == 'box' else cfg.lambda_cls)*cls + (0 if component == 'cls' else cfg.lambda_box)*box
    return loss, {'pairs': len(pairs), 'dcls': cls.detach().item(), 'dbox': box.detach().item()}

def sol_discrepancy(student, teacher):
    # Explicit detection adaptation of original classifier FedSOL fixed-radius:
    # generic categorical KL on all same-index regular queries, T=3, no T^2.
    s = student['pred_logits'].flatten(0, 1)
    t = teacher['pred_logits'].detach().flatten(0, 1)
    return F.kl_div(F.log_softmax(s/3, dim=-1), F.softmax(t/3, dim=-1), reduction='batchmean')
