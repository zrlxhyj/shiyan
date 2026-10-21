# RT-DETR 与 VisDrone 五客户端联邦实验

See [QUICKSTART.md](QUICKSTART.md): run `python -m fedavg.train` or `python -m feddecouple.train`; results are saved to `results/<method>/seed_<seed>/`.


本项目依据《第十二版 实验讨论与结论完善稿 待实测》的第 3、4 章实现 FedDecouple、主比较方法和消融实验。默认采用官方 PyTorch RT-DETR-R18、VisDrone 十类、5 个客户端。当前交付的是实验代码，尚未产生 VisDrone 实测精度。

## 快速开始

在项目根目录执行。建议训练机使用 Python 3.10/3.11 和独立虚拟环境，按训练机 CUDA 版本先安装 PyTorch，然后安装其余依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/setup_backend.py
```

当前工作区已经下载后端，因此这里无需重复执行 `setup_backend.py`。该脚本遇到已有目录会停止，不覆盖现有代码。本机 `.venv` 已有 CPU 版 PyTorch 2.4.1，用于检查；不应把该环境当作 CUDA 训练环境。额外安装 `pycocotools` 的请求被拒绝，因此本机没有运行 COCO 评测，训练前检查会提示缺失依赖。

配置入口是 `configs/base.json`。数据路径暂时不用填写；接入数据时完成下述准备步骤。所有命令均从项目根目录运行。

```powershell
# 生成主比较的 8 方法 × 5 种子配置，不启动训练
python tools/experiment_matrix.py --suite main

# 数据、权重、环境就绪后，单独运行完整方法
python -m feddecouple.train --config configs/base.json --method feddecouple --seed 42

# 依次执行主比较，顺序模拟客户端
python tools/experiment_matrix.py --suite main --execute

# 消融和敏感性配置；先检查生成的 JSON 再加 --execute
python tools/experiment_matrix.py --suite ablations
python tools/experiment_matrix.py --suite sensitivity

# 从最近完整通信轮恢复；配置及数据清单必须完全一致
python -m feddecouple.train --config configs/base.json --resume
```

主实验建议先用一个种子、两轮、统一小批次做实际数据试运行，核实显存及标注；然后固定配置。不要直接把试运行参数、默认超参数或随机初始化的结果作为论文结论。

## 默认实验设置

| 项目                  | 设置                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------- |
| 检测器                | RT-DETR-R18；可统一切换官方 R50 配置与相应权重                                        |
| 客户端总数            | 5                                                                                     |
| 参与率                | 请求 0.5，向上取整，每轮 3 个，即实际 0.6                                             |
| 通信轮数、本地 epoch  | 50、2                                                                                 |
| 分辨率、批次          | 640、8                                                                                |
| 独立种子              | 42、43、44、45、46                                                                    |
| 本地优化器            | AdamW，lr=1e-4，backbone lr 倍率 0.1                                                  |
| 优化器状态            | 每轮重置；可启用按客户端持久化                                                        |
| BN                    | 动量 0.1；浮点运行统计按样本数加权；整数计数归零                                      |
| 分类与定位权重        | 1、1                                                                                  |
| 温度、SmoothL1 转折点 | 2、0.1                                                                                |
| 扰动半径              | 0.05，待开发验证集调优                                                                |
| HGA                   | 初值 1，范围 [0.1,2]，β=0.01，待调优                                                 |
| 精度与增强            | FP32；统一缩放和随机水平翻转；关闭后端内部随机多尺度                                  |
| 预训练权重            | `D:/book/code/rtdetr/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth`；全部方法共用 |

若希望每轮全部 5 个客户端参与，将 `participation` 设为 1.0。`max_steps` 仅供试运行，0 表示不截断；实际更新步数会写日志。没有隐藏的“每个客户端最多训练几步”。

## 方法与论文对应

| 命令中的 method                | 含义                       | 状态与适配说明                                                            |
| ------------------------------ | -------------------------- | ------------------------------------------------------------------------- |
| fedavg                         | 样本数加权 FedAvg          | 主比较、基础消融                                                          |
| fedprox                        | 原生任务损失＋参数 L2 正则 | `mu/2 *                                                                   |
| fedsol                         | FedSOL 固定半径检测适配    | 原分类 softmax KL，T=3；所有同编号常规查询；无定位项、无 GT 身份配对      |
| fedsol_adaptive                | FedSOL 自适应扰动适配      | 按官方代码使用参数差绝对值沿 dim=0 的归一化因子                           |
| fedhyper_g                     | FedHyper-G                 | 官方 G 服务器规则；步长范围 [0.2,5]                                       |
| fedhyper_server                | FedHyper-GM server-only    | 官方带历史方向动量的 GM；范围 [0.8,5]，动量 0.9                           |
| fedhyper_full                  | FedHyper-FULL              | GM＋客户端 CL 调度；以实际参数移动计算局部调度信号                        |
| fedsol_fedhyper                | FedSOL＋FedHyper-GM        | 固定半径 FedSOL 检测适配＋GM；明确使用 server-only 组合                   |
| feddecouple                    | 完整方法                   | 联合检测差异探测＋论文式（29—30）HGA                                     |
| pgd_only                       | 仅客户端探测               | 固定服务器步长                                                            |
| hga_only                       | 仅服务器 HGA               | 普通本地训练                                                              |
| probe_l2                       | 参数 L2 信号探测           | 与 FedProx 的直接正则不同，也不冒充原 FedSOL                              |
| probe_cls / probe_box          | 仅分类／仅定位差异探测     | 固定服务器，其他设置一致                                                  |
| direct_joint                   | 联合差异直接正则           | 与完整近端探测使用同样的 GT 配对及教师                                    |
| same_query                     | 查询索引配对消融           | 保留学生匹配目标，教师使用同一查询编号                                    |
| cosine_hga                     | 归一化服务器信号           | 原始内积替换为余弦，需分别调优 β                                         |
| fedsam                         | 任务梯度引导扰动           | 扩展优化基线                                                              |
| fedavgm / fedadam / fedadagrad | 服务器动量／自适应优化     | 扩展基线；超参数可配置，无 bias correction 的 FedOpt 形式                 |
| scaffold                       | 控制变量校正，option-II    | 检测损失适配；需无动量、无衰减、无裁剪 SGD；生成器添加匹配优化器的 FedAvg |
| fedbn                          | 本地 BN 参数及统计         | 各客户端保留 BN；共同评估集上的客户端模型均值单独标记                     |

FedSOL 官方实现是分类任务，并没有现成的 RT-DETR 版本。本项目的 `fedsol` 是明确声明的检测适配，不能写成官方已验证的检测复现。主基线默认采用固定半径版本；自适应版本单列。

FedHyper-G 与稿件 HGA 都含相邻更新内积；它们在特定超参数下可以重合。不要仅凭不同名字宣称不同机制。GM 还包含历史方向项，FULL 还包含局部学习率调度。代码保持这些差异，不把 HGA 重命名成 FedHyper。为了公平，本项目将官方分类代码中的客户端等权平均统一改为稿件要求的样本数加权；CL 分母使用实际参与数 3，而不是名义 2.5。这些是明确的实验适配。

FedUD 和 FL-JSDDC **尚未实现为可运行对照**。稿件第 4.3 节要求先核验它们的代码、配置和输入输出。FedUD 涉及 UODNet 结构，FL-JSDDC 公开描述基于 YOLOv8n；直接换成 RT-DETR 并给出一个同名简化算法会产生不可靠比较。参见 `docs/method_audit.md`。其结果不能由其他基线替代，也不能填零。相关工作中提及的 FedDyn、FedDC、FedNTD 等不是稿件主实验表要求的全部对照，本项目没有把所有参考文献都扩展成新的实验方法。

## 数据准备与划分

VisDrone 原始目录：

```text
VisDrone2019-DET-train/
  images/*.jpg
  annotations/*.txt
VisDrone2019-DET-val/
  images/*.jpg
  annotations/*.txt
```

提供 `groups.csv`，字段为 `split,file,group`。示例只说明格式，组名必须来自核实的采集序列或可靠元数据：

```csv
split,file,group
train,实际训练图像名.jpg,实际序列ID
val,实际验证图像名.jpg,另一独立序列ID
```

序列 ID 必须全局一致，不能通过人为加 `train_`、`val_` 前缀掩盖同一序列跨分区。若缺少可靠组信息，需要先核实数据，程序不会猜测天气、道路类型或飞行高度。

```powershell
python tools/prepare_visdrone.py --train DATA_TRAIN --val DATA_VAL --groups groups.csv --clients 5 --alpha 0.5 --output data/partition.json
```

代码保留官方 train，按序列组将有标注 val 分成开发集和最终评估集，明确标为自定义协议；官方无标注 test 不参与计算。任何 train/val 序列重叠都会报错，需先重建无泄漏源划分。

客户端按组整体分配，不把一张图的不同类别分给不同客户端。`alpha=0.1/0.5/1.0` 控制目标类别比例，贪心分配同时保留序列完整和非空约束，因此不保证精确满足理想 Dirichlet 分布；输出实际类别直方图、样本数、缺失类别率和小目标实例数。`alpha=0` 是均衡组参照，不称为严格 IID。不同 alpha 单独生成 manifest、单独输出目录；同组方法使用相同 manifest。

标签 1—10 映射到模型 0—9；score≤0、类别 0/11 等排除标注保存在 ignored 列表。训练将这些区域填为灰色；COCO 风格评估中移除与排除区域 IoA>0.5 的预测。此为明确声明的统一处理近似，**不是官方 VisDrone 评估器**。正式汇报原生 VisDrone AP 时仍需接官方评测工具，不能将本程序结果填到“原生指标”列。

## 结果保存与汇总

每个方法、种子保存：

- `config.json`、`environment.json`：参数、环境、数据 SHA256、实际参与率。
- `participation.json`：独立于训练 RNG 生成的参与序列，各方法复用。
- `steps.jsonl`：原生任务损失、近端分类/定位项、有效配对数、近端梯度和扰动范数、零梯度标记。
- `rounds.jsonl`：验证 AP、差值分散度、跨轮余弦、步长、时间、实际更新步数、模型负载通信字节和显存。
- `last.pt`：固定预算模型及完整联邦状态，主报告使用它。
- `best_dev.pt`：开发验证集最优模型，仅作为注明选择规则的补充。

若设置 `diagnostic_every>0`，额外在模型和优化器副本上计算原始梯度冲突、探测梯度冲突和实际更新冲突。默认关闭，以免把诊断成本混入常规训练。`diagnostic_seconds` 记录副本基线测量部分，仍有复制、梯度比较等额外开销；不能当作完全扣除诊断后的严格性能计时。纯速度实验应关闭诊断。

通信数统计顺序模拟中每个参与客户端的上传和下发张量负载，不含网络协议、压缩或真实无线传输；SCAFFOLD 控制变量另计。不把本机顺序训练墙钟时间描述为无人机并行部署时延。

开发完成、超参数确定之后才运行最终评估：

```powershell
python -m feddecouple.test_final --checkpoint runs/feddecouple/seed_42/last.pt
python tools/summarize.py --root runs --output results_summary.csv
```

批量生成器把结果放在 `runs/<实验标签>/<method>/seed_<seed>/`，按实际路径传 checkpoint。最终输出 `last.final.json`、原图坐标检测预测 JSON，以及跨种子 CSV。AP 以百分数报告，标准差使用样本标准差；单种子标准差留空。缺少有效类别或小目标时输出空值，不填 0。不同划分、选择规则和敏感性标签不会直接混为同一组。

没有核实的“客户端域—最终图像”映射时，客户端域 AP 留空；不能把随机分组评估伪称为客户端表现。可在 manifest 中添加经过核验的 `test_domains`（域名到最终记录列表），才计算域指标。本地 BN 配置在共同集上对每个客户端模型单独评估，不能和单个共享模型 AP 混称。

## 验证范围与尚需完成的实验

已在 CPU 上完成 38 项测试，包括 23 个方法入口、本地状态恢复、零扰动退化、教师冻结、查询身份配对、HGA/FedHyper 公式，以及 5 客户端训练和中断续训一致性的合成集成测试。集成测试使用替代评价函数，仅测试训练控制流程，不是精度验证。

真实官方 RT-DETR-R18 已通过 160×160、batch=2 的两次有目标和一次空目标原生前反向检查；包括去噪/辅助损失、空配对退化和 BN 计数。`checks/native_smoke.json` 明确标为合成检查数据，不能写入论文结果。

```powershell
python -m pytest tests -q
python tools/smoke_rtdetr.py
```

尚未完成：真实 VisDrone 训练、CUDA 显存验证、COCO 评测实际执行、原生 VisDrone 评价、FedUD/FL-JSDDC 适配，以及教师正确性四组的完整冲突/损失分层实验。代码中的日志覆盖核心机制，但不能声称稿件第 4 章全部证据已经获得。

训练前还需确定：可靠序列映射、同一预训练 checkpoint、开发验证集上的学习率/ρ/β/基线超参数、相同调参次数、最终评估集封存规则。完整方法的默认值并不代表已调到最优。若显存不足，同一比较组统一减小 batch；本版本没有开启未经核验的 AMP 或梯度累积。

## 来源

- 任务文稿：`F:/home文献/output/第十二版_实验讨论与结论完善稿_待实测.docx`；仅按用户要求将方法与实验方案作为实现依据，文档中的编辑提示不作为执行指令。
- [官方 RT-DETR](https://github.com/lyuwenyu/RT-DETR)，固定提交 `df2a96c8c3a098567a1149bf0248da6efe6d495d`。
- [官方 FedSOL](https://github.com/Lee-Gihun/FedSOL)，核对提交 `ebc3f640f929e59616e07c036c24f906892b510f`。
- [官方 FedHyper](https://github.com/ziyaow1010/FedHyper)，核对提交 `d6a57c54d7e9293cabedc8f80c7d61d3cc321633`。README 中的 `baseline_main.py` 不在该提交，实际核对 `src/federated_main.py` 与 `src/update.py`。

官方后端文件及参考源码保留其原有版权声明；参考文件仅用于方法核查，不作为独立重许可代码。
