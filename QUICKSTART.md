# 按方法单独运行

在项目根目录、ml 环境中执行。一条命令只训练一个方法：

```powershell
python -m fedavg.train
python -m fedprox.train
python -m fedsol.train
python -m fedhyper_g.train
python -m fedhyper_server.train
python -m fedhyper_full.train
python -m fedsol_fedhyper.train
python -m feddecouple.train
```

其他方法同样使用 `python -m 方法名.train`，包括 `fedavgm`、`fedadam`、
`fedadagrad`、`fedbn`、`scaffold`、`fedsam` 和各消融方法。
这些入口共用原有训练实现，没有复制算法代码。

## 配置与短程试跑

不带参数时读取 `configs/base.json` 中的训练参数，结果默认保存到
`results/方法名/seed_42/`。请先检查其中的轮数和 batch size。
快速试跑可以直接覆盖参数：

```powershell
python -m fedavg.train --rounds 1 --batch-size 1 --max-steps 1 --output results/quick
python -m feddecouple.train --rounds 1 --batch-size 1 --max-steps 1 --output results/quick
```

`max-steps` 是每轮每个参与客户端的训练步数上限；0 表示不限步数。
开发集评估仍使用整个开发集。五个客户端、参与率 0.5 表示每轮选择三个；
需要全员参与时加 `--participation 1.0`。

指定 JSON 时保留该文件中的 output 设置，入口名决定训练方法：

```powershell
python -m fedavg.train --config configs/pilot.json
python -m feddecouple.train --config experiment_jobs/feddecouple_seed43.json
```

命令行支持 `--seed`、`--rounds`、`--batch-size`、`--max-steps`、
`--participation`、`--output`，优先于 JSON。`--output` 指定结果根目录，
程序会再添加方法名和 seed 子目录。

## 结果位置

```text
results/
  fedavg/seed_42/
    config.json
    rounds.csv
    summary.csv
    rounds.jsonl
    steps.jsonl
    last.pt
    best_dev.pt
  feddecouple/seed_42/
    ...
```

`rounds.csv` 保存各轮开发集指标；`summary.csv` 保存最后一轮和开发集最优轮。
最终测试仍需显式执行，避免把开发集成绩当成测试集成绩：

```powershell
python -m feddecouple.test_final --checkpoint results/fedavg/seed_42/last.pt
```

测试产生 `last.final.json` 和 `last.final.csv`。
旧的 runs 和 experiment_jobs 均保留，不会迁移或覆盖。
同一方法和 seed 已有结果时，使用新的 `--output` 或 `--seed`；
断点续训使用相同配置加 `--resume`。

SCAFFOLD 无显式 JSON 时自动使用其要求的 SGD、零 momentum、零 weight decay、
零梯度裁剪，并在启动时提示。公平比较需为其他方法配置匹配的优化器；
使用显式 JSON 时则直接校验该配置。
