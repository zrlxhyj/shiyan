# 配对机制诊断（schema version 2）

## 如何运行

修改的是本项目训练代码；不修改已有模型、数据划分或历史结果。
诊断默认关闭，开启后额外消耗前向、反向、优化器副本的计算和显存。
当前仅支持FedAvg和FedDecouple，其他方法若开启会明确报错。

先在项目根目录、实际训练环境中执行小规模诊断试跑：

```powershell
python -m fedavg.train --rounds 2 --batch-size 1 --max-steps 2 --diagnostic-every 1 --output results/mechanism_pilot
python -m feddecouple.train --rounds 2 --batch-size 1 --max-steps 2 --diagnostic-every 1 --output results/mechanism_pilot
```

以上沿用base.json的数据集和预训练权重路径。服务器上仍需使用服务器实际路径。
如需新Non-IID划分，加 `--config datasets/所用划分/train_config.json`；
参数 `--output` 会覆盖JSON的输出根目录。
已有相同输出目录时另选目录；不要将新诊断日志混入已有实验。

正式采样可使用 `--diagnostic-every 50`，即每客户端每轮本地step为0、50、100……
时诊断，不会额外增加正式优化步数。正数不等于诊断每个batch。
默认仍由 `diagnostic_every=0` 关闭；可在JSON配置中设置同名字段。
重启后才加载新代码；旧检查点不含历史诊断，不能倒推出旧轮次数据。

## 共同参考与两个反事实分支

两种方法均使用本客户端本轮开始时冻结、eval模式的teacher。
FedAvg原本没有teacher，只有在开启诊断时才创建相同定义的冻结参考。
测量的参考量是与teacher之间的joint detection discrepancy，
由目标身份匹配的分类KL与框Smooth-L1构成，权重等沿用相同cfg。

每个诊断采样点：

1. 捕获当前Python、NumPy、PyTorch CPU与所有CUDA RNG状态，复制输入张量和targets。
2. 在模型、teacher与损失/匹配器副本中求共同参考梯度 g_ref。
3. 构造 delta = rho * g_ref / (||g_ref|| + eps)，零参考梯度时delta=0。
4. 分别深拷贝 `(model, optimizer)` 元组，确保优化器参数组引用复制后的模型参数，
   并完整保留AdamW矩、SGD动量、参数组学习率与weight decay。
5. 原始分支在theta处求任务梯度g0；探测分支在theta+delta处求任务梯度g1。
   两分支从同一缓冲区与同一RNG状态开始，使用同一已增强批次。
6. 梯度测量完成后，两分支均恢复到theta，再按正式clip_grad和优化器副本执行一次step。
   保留各自任务前向产生的BN缓冲区更新，得到 u0=theta-theta0_new、u1=theta-theta1_new。
7. 从相同RNG状态和损失/匹配器起点，对两个更新后副本再次前向，分别测量任务损失
   及对共同冻结teacher的预测差异。模型保持与输入模型相同的train/eval模式，
   因此训练模式下的Dropout、BN行为也参与这项一致定义的测量。
8. finally恢复入口RNG。额外前向、反向和step均发生在副本上，
   不修改正式模型、参数梯度、包括nonpersistent在内的缓冲区、teacher、
   损失函数状态、输入或正式优化器。

上述每次只让一个分支模型/优化器副本驻留，两个分支不会同时保留完整优化器副本。
正式训练随后只执行其原有的一次优化器更新。

FedAvg实际采用original分支，FedDecouple实际采用probed分支。
诊断中的before/after指这两个配对反事实分支，**不是两次正式训练step**。
不要把FedAvg的探测分支描述成FedAvg训练时实际施加的扰动。

## 指标与符号

局部梯度冲突：C(g,g_ref)=max(0,-cos(g,g_ref))。
实际优化器更新冲突：C(u,g_ref)=max(0,-cos(u,g_ref))，其中u为theta-theta_new，
含优化器动量、AdamW预条件、weight decay和梯度裁剪的影响。
这里没有测量与g_ref正交的全部任务几何，也不等同于客户端间冲突。

| 字段 | 定义 |
|---|---|
| conflict_before / conflict_after | g0或g1对共同g_ref的负余弦截断值 |
| update_conflict_before / update_conflict_after | u0或u1对共同g_ref的负余弦截断值 |
| gradient_cosine_before / after | 对应有符号梯度余弦，保留零截断掩盖的信息 |
| update_cosine_before / after | 对应有符号实际更新余弦 |
| task_gradient_cosine | 原始任务梯度g0与探测任务梯度g1的余弦 |
| conflict_improvement / update_conflict_improvement | before减after，正值表示改善 |
| delta_conflict | 兼容旧命名，after减before，与improvement符号相反 |
| task_loss_before / task_loss_after | 更新前原始/探测位置的任务损失 |
| post_task_loss_before / post_task_loss_after | 两个优化器副本更新后的共同诊断任务损失 |
| post_prediction_discrepancy_before / after | 更新后对共同teacher的joint预测差异 |
| post_dcls_before / after、post_dbox_before / after | 对应分类、框差异分量 |
| actual_gradient_conflict / actual_update_conflict | 当前方法实际分支对应的副本测量 |
| diagnostic_probe_norm | 共同诊断探测半径；FedAvg正式perturb_norm仍为0 |

零向量的余弦未定义：输出null，并记录diagnostic_status，不伪造为“冲突为0”。
没有匹配目标的批次会保留配对数量，不能将其预测差异零值直接解读为性能改善。

## 日志、表格与配对

每个采样step的 `steps.jsonl` 额外保存schema version、method、seed、round、client、
epoch、step、image IDs、输入张量/targets SHA-256、RNG SHA-256、
数据清单SHA-256、客户端样本量、本地计划步数、参与客户端及sample ID。
sample ID不含方法名，便于在相同seed/批次/增强下跨方法对齐；
仍应核对输入和RNG哈希，不得只用行号配对。

`rounds.jsonl` 中的 `diagnostics` 汇总采样step均值、每项有效样本数、
各客户端诊断计数。它是**有效采样step的等权均值**，不是按客户端均衡的均值，
也不是跨seed的误差估计。客户端数据量影响采样次数，分析时必须披露该汇总口径。
`rounds.csv`、`summary.csv` 会保留这些展开字段。

训练正常结束后自动生成：

- `diagnostic_samples.csv`：完整轮次中的配对step诊断。
- `diagnostic_trajectory_pairs.csv`：当轮诊断均值与当轮/下一轮分散度、方向余弦、
  服务端步长，附当轮/下一轮本地步数和参与重合度。最后一轮的下一轮指标留空。

训练未结束、但已有完整轮次时，也可手动导出：

```powershell
python -m feddecouple.mechanism --run-dir results/mechanism_pilot/fedavg/seed_42
python -m feddecouple.mechanism --run-dir results/mechanism_pilot/feddecouple/seed_42
```

导出器只使用schema v2且已经写入完整round日志的step，拒绝重复sample ID。
旧版diagnostic字段和历史缺失值不会混用或补造。

## 分析限制

后续统计需控制轮次趋势、本地优化步数和参与集合重合度，并匹配配置与独立种子。
如果全员参与且每轮本地步数相同，这两项恒定，无法估计它们各自的调整效应。
多个本地step不是多个独立seed；不要据此绘制跨seed置信区间。
相邻轮方向余弦包含当轮更新，HGA步长又直接使用更新内积，存在数学耦合；
滞后关联仅是机制描述的补充，不构成普遍因果证据。

## 验证范围

测试覆盖FedAvg/FedDecouple、AdamW/SGD、CPU/可用CUDA、Dropout、BatchNorm、
nonpersistent buffer、带状态的criterion、已有优化器状态、异常退出恢复RNG、
零参考梯度缺失值及rho=0两分支一致性。
新增完整训练循环测试同时检查模型、优化器检查点及诊断表导出。

真实RT-DETR合成批次验证入口：

```powershell
python tools/smoke_diagnostics.py
```

它不运行真实数据集，不产生检测精度结论。检查点未包含诊断可恢复的历史梯度。
本次会话首批42项诊断与算法测试通过；之后新增的完整循环测试和此真实模型
验证脚本因执行权限/沙箱问题尚未运行，不能把“脚本已提供”当作验证已通过。
