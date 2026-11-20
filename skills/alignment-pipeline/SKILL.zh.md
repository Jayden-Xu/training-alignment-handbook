---
name: alignment-pipeline
description: 把一个逐位对齐问题分到正确的流程，并从头到尾执行。预训练（两次本应一致的运行、改动后的 loss 尖峰、恢复训练后出现漂移、怀疑 SDC）依次走预检、参考运行、OpGuard 捕获或图探针、首个分歧点、修复、确认、回归条目。强化学习训推（rollout 与 trainer 的 log-prob 不同）依次走模型审计、逐算子 batch-invariant、整模型 L1–L3、L5、采样、对引擎的 L4、在线记录、staleness/数值拆分、离线定位、回归。先用这个 skill，再用它调用的更具体的 skill。
---

# 对齐流程

这个 skill 只做分流。每一步写明进入条件、完成工作的工具或 skill、必须留下的证据，以及何时停止。
具体工作和规则在链接的 skill 和手册各部分里；这里不再重复。

所有地方的相等都是逐位相等：没有容差，也不用舍入来强行相等。
每个结论都写明 GPU 和版本，或者写“未测试”。

## 选哪个流程

| 用户手里有 | 流程 |
|---|---|
| 两次本应一致的训练运行；升级、迁移、配置改动或恢复训练之后出现尖峰或漂移 | A. 预训练 |
| 一台怀疑有静默数据损坏的机器 | A，只做步骤 A0 |
| rollout 的 log-prob 与 trainer 的不同；一个悄悄偏离 on-policy 的强化学习运行 | B. 强化学习训推 |
| 单个算子，没有 batch-invariant 实现 | 直接用 [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) |
| 框架迁移、检查点转换、预先算好的参考 log-prob，或其他需要一致的一对 | [第 5 部分](../../handbook/5-scenarios/README.md)；转换用 [tools/ckpt_compare.py](../../tools/ckpt_compare.py) |
| 新的软硬件栈（torch、Triton、NCCL、驱动、CUDA、GPU） | [regression](../../handbook/4-operations/regression.md)：先跑测试套件 |

在任一流程之前，如果是新机器或新栈，先查看那里测过什么：
[硬件覆盖](../../handbook/4-operations/hardware.md)。尚未测试或未解决的部分（没有真实的
引擎记录 hook、没有盲测、FSDP 是模拟的，……）列在
[待办事项](../../handbook/open-items.md)里。

## A. 预训练

| 步骤 | 进入条件 | 工具或 skill | 要留下的证据 | 何时停止 |
|---|---|---|---|---|
| A0. SDC | 怀疑某个设备，或在长时间运行之前 | [tools/sdc_scan.py](../../tools/sdc_scan.py)（先 `--quick`，再 fleet 和 golden）；作业内的 `sdc_guard`、`cross_rank_check`（[案例](../../handbook/1-pretraining/sdc.md)） | 扫描结果 JSON，写明设备、算子和重复次数 | 某个设备与同组设备不一致（先排除软件差异），或全部通过 |
| A1. 预检 | 总是，两次运行都做 | [tools/determinism_preflight.py](../../tools/determinism_preflight.py) 在恢复后每个 rank 跑 `report`；`diff(a, b, declared=[...])` | 两份报告和 diff，只剩已声明的差异 | 出现未声明的差异，使逐位比较失去意义：修正设置，重新运行 |
| A1'. 数据 | 预检干净 | [tools/data_fingerprint.py](../../tools/data_fingerprint.py)：两次运行都在 loader 出口用 `Recorder(dir, rank)` 和 `record(step, batch)`；`compare`（DP 规模不同时加 `--global`）（[页面](../../handbook/1-pretraining/data-alignment.md)） | 两份记录目录和 compare 结果 | 有记录不同：按类别（order、samples、packing、masks）修复后再比较 |
| A2. 参考运行 | 预检干净 | [第 1 部分第 1 步](../../handbook/1-pretraining/README.md#step-1-pick-a-reference-run) | 写下待测的那一个差异 | — |
| A3. 捕获 | eager 运行 | 通过 [tools/opguard_bridge.py](../../tools/opguard_bridge.py) `run_pair` 使用 OpGuard；检查 `coverage(run)` | 两份 trace 和覆盖报告 | 覆盖缺少所关心的区域：先修正包装 |
| A3'. 捕获 | `torch.compile` 或 CUDA graphs | [tools/graph_probe.py](../../tools/graph_probe.py) 加 `isolate=True`（[案例](../../handbook/1-pretraining/capture-compiled-code.md)） | 探针 trace；带探针和不带探针的编译给出相同的比特 | 探针改变了编译后的比特：把它们隔离 |
| A4. 首个分歧点 | 两份 trace | `opguard_bridge.first_divergence` | 分歧点信息：算子、代码行、调用、不同的字段、它之前未对齐的事件 | 没有事件不同且最终参数相等：两次运行一致 |
| A5. 原因 | 一个分歧点 | [pretraining-alignment](../pretraining-alignment/SKILL.md)（步骤 5–6）配合 OpGuard 的 harness skill；用 `suggest()` 找线索 | 一个用最小干预检验过的假设 | 该干预消除了分歧点处的差异 |
| A6. 修复 | 一个已确认的原因 | 设置或状态：[训练步](../../handbook/1-pretraining/training-step.md)、[DP 和 FSDP 风格（模拟：扁平梯度、reduce-scatter、分片 AdamW）](../../handbook/1-pretraining/data-parallel.md)；kernel：[batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | 改动及其代价（预热、同步、中位数） | — |
| A7. 确认 | 一个修复 | 重新捕获这一对运行 | 没有不同的事件，没有未对齐的事件，最终参数逐位相同 | 已确认；否则回到 A4 |
| A8. 回归 | 已确认 | 一个在 `bi-results-v1` 中带 `--quick`/`--json` 的示例，在 [suite/manifest.json](../../suite/manifest.json) 中加一条（[regression](../../handbook/4-operations/regression.md#how-to-add-a-check-or-an-example)） | 示例、它的结果文件、一个基线 | 测试套件在该栈上通过 |

## B. 强化学习训推

顺序很重要：每个级别都假定前面的级别成立。只有在某一步的进入条件不成立时才跳过它，并且要说明。

| 步骤 | 进入条件 | 工具或 skill | 要留下的证据 | 何时停止 |
|---|---|---|---|---|
| B1. 审计 | trainer 的模型 | 在模型上运行 [tools/bi_audit.py](../../tools/bi_audit.py) | 算子清单和逐模块的 L2 判定 | 每个模块都是 L2，或列出了不满足的模块 |
| B2. 逐算子 L2 | 一个不满足的算子 | [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | 它的验证矩阵和代价 | 每个算子都通过，或有记录在案的退路 |
| B3. 整模型 L1–L3 | 所有算子都是 L2 | [整模型案例](../../handbook/2-rl/whole-model.md)的方法：每种服务模式、激活梯度 | 每种模式下的 logprob 相等 | 引擎用到的每种模式都满足 L1–L3 |
| B4. L5 | trainer 和引擎使用不同的 TP、DP 或 EP 度数 | [whole-model-tp](../../handbook/2-rl/parallelism.md)、[training-dp-fsdp](../../handbook/1-pretraining/data-parallel.md)、recipe 10 | 跨度数相等，前向和反向 | 在用到的每个度数下都相等 |
| B5. 采样 | token 经过采样、重放、抢占或推测 | [tools/bi_sampling.py](../../tools/bi_sampling.py)（[案例](../../handbook/2-rl/sampling.md)） | 重新计算后的重放、回滚、n、推测分块、新进程 | token 逐位重放 |
| B6. L4 | 两侧都满足 B3 | [engine-parity](../engine-parity/SKILL.md)（[vLLM 案例](../../handbook/2-rl/engine-parity-vllm.md)）；LoRA：[案例](../../handbook/2-rl/lora.md) | 两侧的舍入序列；逐算子、逐层相等 | 0 个 logprob 不同，或剩下的每个差异都是记录在案的决定，并附代价 |
| B7. 在线记录 | 强化学习作业 | OpGuard 的 `opguard_harness.rl_records.RecordWriter`：rollout 批次、trainer log-prob、前向、权重同步、快照（[第 2 部分](../../handbook/2-rl/README.md#synchronous-and-asynchronous-runs)） | 每个 rank 的记录文件 | — |
| B8. 拆分 | 记录 | 先 `rl.weight_sync_check`，再 `rl.decompose` | 未匹配的权重行（无）、每个 token 的 `numeric` 和 `staleness`、混合版本的 token 单独放开 | `numeric` 逐位为零：基础设施没有问题，差距来自 staleness |
| B9. 定位 | `numeric` ≠ 0 | OpGuard 的 `rl-alignment` skill（`harness/skills/rl-alignment`）：`rl.replay_request`、用 `OPGUARD_XOR_ROWS=1` 重新捕获、`vectors.*`；[示例](../../handbook/2-rl/rl-loop.md) | 出现分歧的算子及另一侧的证据；它的原因类别 | 一个用最小改动检验过的原因；然后回到 B2、B4、B5 或 B6 去修复 |
| B10. 回归 | 一个修复 | 同 A8 | 示例和基线 | 测试套件在该栈上通过 |

## 何时停下来询问

- 修复的代价超过用户同意的范围：报告测得的代价，以及已经成立的更便宜的级别。
- 差异是算法上的（同一数学的两种精确形式）：这是选择，不是 bug。报告两种形式及其代价。
- 引擎或 kernel 是闭源的，没有可调的开关：报告退路选项（batch-invariant-kernel，第 2 步）。
- 新栈上一项必需的回归检查失败：这是要报告的发现，不能靠调参消掉。
