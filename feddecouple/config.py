from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

# local rule, server rule. Detection adaptations are identified in README.
METHODS = {
    'fedavg': ('plain', 'fixed'),
    'fedprox': ('prox', 'fixed'),
    'fedsol': ('sol', 'fixed'),
    'fedsol_adaptive': ('sol_adaptive', 'fixed'),
    'fedhyper_g': ('plain', 'hyper_g'),
    'fedhyper_server': ('plain', 'hyper_gm'),
    'fedhyper_full': ('hyper_local', 'hyper_gm'),
    'fedsol_fedhyper': ('sol', 'hyper_gm'),
    'feddecouple': ('joint', 'hga'),
    'pgd_only': ('joint', 'fixed'),
    'hga_only': ('plain', 'hga'),
    'probe_l2': ('l2', 'fixed'),
    'probe_cls': ('cls', 'fixed'),
    'probe_box': ('box', 'fixed'),
    'direct_joint': ('direct', 'fixed'),
    'cosine_hga': ('joint', 'cosine'),
    'same_query': ('joint_query', 'fixed'),
    'fedsam': ('sam', 'fixed'),
    'fedavgm': ('plain', 'momentum'),
    'fedadam': ('plain', 'adam'),
    'fedadagrad': ('plain', 'adagrad'),
    'scaffold': ('scaffold', 'fixed'),
    'fedbn': ('plain', 'fixed'),
}

@dataclass
class Config:
    method: str = 'feddecouple'
    clients: int = 5
    participation: float = .5
    rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 8
    image_size: int = 640
    seed: int = 42
    device: str = 'cuda:0'
    workers: int = 0
    lr: float = .0001
    optimizer: str = 'adamw'
    momentum: float = .9
    weight_decay: float = .0001
    backbone_lr_ratio: float = .1
    clip_grad: float = .1
    persistent_optimizer: bool = False
    bn_mode: str = 'weighted'
    rho: float = .05
    mu: float = .01
    direct_weight: float = 1.
    lambda_cls: float = 1.
    lambda_box: float = 1.
    temperature: float = 2.
    smooth_l1_beta: float = .1
    probability_eps: float = 1e-6
    eps: float = 1e-12
    zero_threshold: float = 1e-12
    server_lr: float = 1.
    hga_beta: float = .01
    server_min: float = .1
    server_max: float = 2.
    hyper_rate: float = 1.
    server_beta1: float = .9
    server_beta2: float = .99
    server_tau: float = .001
    manifest: str = 'data/partition.json'
    output: str = 'runs'
    upstream: str = 'third_party/RT-DETR/rtdetr_pytorch'
    model_config: str = 'configs/rtdetr/rtdetr_r18vd_6x_coco.yml'
    pretrained: str = ''
    eval_every: int = 1
    diagnostic_every: int = 0
    max_steps: int = 0

    def validate(self):
        if self.method not in METHODS:
            raise ValueError(f'Unknown method {self.method}; available: {list(METHODS)}')
        if self.clients < 1 or not 0 < self.participation <= 1:
            raise ValueError('Invalid client count/participation')
        if min(self.rounds, self.local_epochs, self.batch_size, self.eval_every) < 1:
            raise ValueError('Budgets must be positive')
        if self.diagnostic_every < 0:
            raise ValueError('diagnostic_every must be nonnegative')
        if self.diagnostic_every and self.method not in ('fedavg','feddecouple'):
            raise ValueError('Paired mechanism diagnostics currently support fedavg and feddecouple only')
        if self.rho < 0 or self.lr <= 0 or self.temperature <= 0 or self.smooth_l1_beta <= 0:
            raise ValueError('Invalid optimizer/proximal settings')
        if not 0 < self.server_min <= self.server_lr <= self.server_max:
            raise ValueError('Invalid server bounds')
        if self.bn_mode not in ('weighted', 'local'):
            raise ValueError('bn_mode must be weighted or local')
        if self.optimizer not in ('adamw', 'sgd'):
            raise ValueError('Unsupported optimizer')
        if self.method == 'scaffold' and (self.optimizer != 'sgd' or self.momentum != 0 or self.weight_decay != 0 or self.clip_grad != 0):
            raise ValueError('SCAFFOLD option-II requires SGD, momentum=0, weight_decay=0, clip_grad=0; apply same optimizer to comparison group')
        return self

    @property
    def selected_count(self):
        return max(1, math.ceil(self.clients * self.participation))

    def dump(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding='utf8')

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8-sig'))).validate()
