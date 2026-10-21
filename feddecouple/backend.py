import sys
import types
import importlib
from pathlib import Path
import torch

UPSTREAM_COMMIT = 'df2a96c8c3a098567a1149bf0248da6efe6d495d'

def build(cfg):
    root = Path(cfg.upstream).resolve()
    if not (root/'src').is_dir():
        raise FileNotFoundError('Run python tools/setup_backend.py first, or set upstream to official rtdetr_pytorch directory')
    marker = root.parent/'COMMIT'
    if marker.exists() and marker.read_text(encoding='utf8').strip() != UPSTREAM_COMMIT:
        raise ValueError('Backend COMMIT differs from tested version')
    if 'src' in sys.modules and str(root/'src') not in list(getattr(sys.modules['src'], '__path__', [])):
        raise RuntimeError('Another package named src is already imported; run in a fresh Python process')
    sys.path.insert(0, str(root))
    # Load only model registries. Upstream src/__init__ eagerly imports the old
    # torchvision.datapoints data pipeline, which is unused by this project.
    if 'src' not in sys.modules:
        package = types.ModuleType('src')
        package.__path__ = [str(root/'src')]
        sys.modules['src'] = package
        for name, relative in [('src.nn','nn'), ('src.nn.backbone','nn/backbone')]:
            subpackage = types.ModuleType(name)
            subpackage.__path__ = [str(root/'src'/relative)]
            sys.modules[name] = subpackage
        importlib.import_module('src.nn.backbone.presnet')
        importlib.import_module('src.zoo')
    from src.core import YAMLConfig
    native = YAMLConfig(str(root/cfg.model_config), num_classes=10,
                        remap_mscoco_category=False, use_focal_loss=True)
    native.yaml_cfg['PResNet']['pretrained'] = False  # no implicit download
    native.yaml_cfg['PResNet']['freeze_norm'] = False
    native.yaml_cfg['PResNet']['freeze_at'] = -1
    native.yaml_cfg['RTDETRTransformer']['eval_spatial_size'] = [cfg.image_size]*2
    native.yaml_cfg['HybridEncoder']['eval_spatial_size'] = [cfg.image_size]*2
    native.yaml_cfg['RTDETR']['multi_scale'] = None  # augmentation once, outside forwards
    model, criterion = native.model, native.criterion
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.momentum = .1
    if cfg.pretrained:
        checkpoint = torch.load(cfg.pretrained, map_location='cpu', weights_only=False)
        weights = checkpoint.get('ema', {}).get('module') or checkpoint.get('model', checkpoint)
        own = model.state_dict()
        compatible = {k: v for k, v in weights.items() if k in own and v.shape == own[k].shape}
        missing = model.load_state_dict(compatible, strict=False)
        if len(compatible) < .8*len(own):
            raise ValueError('Pretrained checkpoint has fewer than 80% compatible tensors; check architecture')
        print('Pretrained load missing (expected class heads):', missing.missing_keys)
    # Match the manuscript, independently of defaults in the upstream config.
    criterion.matcher.cost_class = 2
    criterion.matcher.cost_bbox = 5
    criterion.matcher.cost_giou = 2
    criterion.matcher.alpha = .25
    criterion.matcher.gamma = 2.
    criterion.matcher.use_focal_loss = True
    return model, criterion

def task_loss(model, criterion, images, targets):
    outputs = model(images, targets)
    losses = criterion(outputs, targets)
    return sum(losses.values()), outputs

def optimizer_for(model, cfg):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr = cfg.lr*(cfg.backbone_lr_ratio if name.startswith('backbone.') else 1.)
        decay = 0. if p.ndim == 1 or name.endswith('.bias') else cfg.weight_decay
        groups.setdefault((lr, decay), []).append(p)
    args = [{'params': v, 'lr': lr, 'weight_decay': decay} for (lr, decay), v in groups.items()]
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(args, momentum=cfg.momentum)
    return torch.optim.AdamW(args, betas=(.9, .999))
