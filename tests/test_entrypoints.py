import importlib.util
import pytest

from feddecouple import train
from feddecouple.config import Config, METHODS


@pytest.mark.parametrize('method', list(METHODS))
def test_method_entry_uses_shared_config_and_separate_results(monkeypatch, method):
    monkeypatch.setattr(train.Config, 'load', lambda path: Config(rounds=1, batch_size=1))
    captured = []
    monkeypatch.setattr(train, 'run', lambda cfg, resume: captured.append((cfg, resume)))
    assert importlib.util.find_spec(f'{method}.train') is not None
    train.main(default_method=None if method == 'feddecouple' else method, argv=[])
    cfg, resume = captured[0]
    assert cfg.method == method and cfg.output == 'results' and not resume
    cfg.validate()


def test_explicit_config_preserves_legacy_output_and_cli_overrides(monkeypatch):
    monkeypatch.setattr(train.Config, 'load', lambda path: Config(output='runs/pilot', method='feddecouple'))
    captured = []
    monkeypatch.setattr(train, 'run', lambda cfg, resume: captured.append((cfg, resume)))
    train.main('fedavg', ['--config', 'configs/pilot.json', '--rounds', '1',
                         '--batch-size', '1', '--max-steps', '1', '--seed', '43', '--resume'])
    cfg, resume = captured[0]
    assert cfg.method == 'fedavg' and cfg.output == 'runs/pilot'
    assert (cfg.rounds, cfg.batch_size, cfg.max_steps, cfg.seed, resume) == (1, 1, 1, 43, True)


def test_method_entry_rejects_conflicting_method(monkeypatch):
    with pytest.raises(SystemExit):
        train.main('fedavg', ['--method', 'fedprox'])
