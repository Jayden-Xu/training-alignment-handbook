---
name: pretraining-alignment
description: 找出两次预训练运行从哪里开始不再逐位一致，以及原因。选定参考运行，检查并比对两次运行的确定性设置，用 OpGuard 捕获两次运行，取首个分歧点，交给 OpGuard 的 harness skill，把算子映射到手册的算子类别，通过确定性设置或 batch-invariant-kernel skill 修复，确认相同前缀变长，并记录一条回归条目。适用于训练运行出了问题（loss 尖峰、升级或改配置后的漂移、恢复训练后没有沿同一轨迹继续），或两次本应一致的运行不一致时。
---

# 预训练对齐

只负责一个问题：两次本应逐位一致的训练运行不一致。
返回它们首次不再一致的位置、算子类别和原因（或一个有边界的无结论结果）、
修复及其代价，以及一条回归条目。

方法见[第 1 部分](../../handbook/1-pretraining/README.md)。有实测数据的完整演练见
[预训练案例](../../handbook/1-pretraining/localize-with-opguard.md)。
工具：
- [tools/determinism_preflight.py](../../tools/determinism_preflight.py)：
  `report`、`findings`、`diff`；
- [tools/opguard_bridge.py](../../tools/opguard_bridge.py)：`run_pair`、
  `first_divergence`、`coverage`、`suggest`、fingerprint（指纹）、`TraceWriter`；
- OpGuard 本身（`python -m opguard.tools`、`opguard-harness`），本 skill
  驱动它，但从不修改它。

## 谁负责什么

| 步骤 | 负责方 | 返回 |
|---|---|---|
| 参考、预检、比对 | 本 skill | 问题、两份报告、比对结果 |
| 捕获与比较 | OpGuard 后端，经由 harness（`run_pair`） | 每个 rank 的 Perfetto 导出，`verdict: not_evaluated` |
| 首个分歧点的事实 | OpGuard harness 只读工具（由 `first_divergence` 调用） | pivot 事件、上一个 ok 事件、未对齐计数、fingerprint |
| 对应关系、向上游追溯 | OpGuard skill `investigate-first-diff`，然后 `trace-upstream` | 一个收窄的边界，或所缺的最小观测 |
| 检验一个原因 | OpGuard skill `validate-hypothesis` | 支持、否定或无结论，附对照 |
| 算子类别、配方 | 本 skill（`suggest` 给出线索） | 类别、配方、修复选项 |
| kernel 修复 | skill [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) | 改过的 kernel、验证、代价 |
| 确认、回归条目 | 本 skill | 无分歧的重跑、条目 |

OpGuard 产出事实。判断（哪些事件相互对应、哪个差异最先、原因是什么）由你来做，
并且要附证据。`suggest` 给出线索，从不给出判定。

## 1. 选定参考

按问题选（第 1 部分，步骤 1 的表）：残余非确定性或 SDC（静默数据损坏）用自我重放，
回归用最后一个正常的提交，某个特性用关掉一个开关的运行，升级用旧版本。
两次运行使用相同的数据、种子和检查点。写下被测的那一处差异。

证据：两条启动命令、各自的提交、被测的差异。

## 2. 预检与比对

在每次运行中，模型、优化器、调度器和加载器都已构建、任何检查点都已恢复之后，
就在第一步之前：

```python
rep = determinism_preflight.report(optimizer=opt, dataloader=loader, scheduler=sched,
                                   autocast=amp, model=model)
```

然后 `determinism_preflight.diff(rep_a, rep_b, declared=[...])`，声明被测的差异
（例如 `launch.argv`、`stack.torch`）。

- `bitwise_comparison: not_meaningful`：在声明的差异之外还有阻断性差异。
  捕获之前修复或声明它；否则首个分歧点只会显示那项配置。
- `at_risk`（来自 `findings`）：某一侧未受控的面。常见的一个：
  `warn_only=True` 让 SDPA 的反向保持非确定性（实测）。先修复这些；
  之后自我重放可能就已经一致。
- 恢复训练时，恢复之后 `optimizer.state`、`scheduler`、`seeds` 或加载器位置
  的差异本身就是 bug（实测：步数计数器在任何捕获之前就出现在这里）。

证据：两份报告文件和比对结果。

### loader 出口的数据

两次运行的每个 rank 上，都在 loader 返回 batch 时记录它：`rec = data_fingerprint.Recorder(dir, rank=dp_rank, meta={...})`，然后 `rec.record(step, batch)`（[数据对齐](../../handbook/1-pretraining/data-alignment.md)）。抓取之前，`python tools/data_fingerprint.py compare A B`（DP 规模不同时加 `--global`）必须报告没有不同的记录；否则第一处分歧就在数据上。按类别修复。证据：两份记录目录和 compare 的输出。

## 3. 用 OpGuard 捕获两次运行

每一侧使用专用的检出目录（`stage_checkouts`），然后 `run_pair`：四次
工作负载启动（每侧一次发现、一次捕获）加一次比较。
运行要短：几步，一到两块 GPU。两侧使用相同的
`CUDA_VISIBLE_DEVICES` 索引：OpGuard 比较的是设备字符串。

然后 `coverage(run)`：在检出目录中发现、但未计划插桩的调用点没有边界记录。
实测：用 `torch.nn.Linear` 时，GEMM 从 torch 的 `linear.py` 启动，在检出目录之外，
所以不会被捕获；用在模型代码中调用 `F.linear` 的 Linear 时则会被捕获。
OpGuard 只捕获 eager 执行。`torch.compile` 或 CUDA graph 内部的边界需要在那里
取 fingerprint（bridge 的纯 torch 函数在两者内部都能用），并用 `TraceWriter` 写出。

证据：实验 ID、阶段列表、启动次数和墙钟时间、覆盖情况。

## 4. 取首个分歧点

`facts = first_divergence(report, root, og)`。按以下顺序读：
1. `status_counts` 和 `unaligned_before`：未对齐事件是空缺，不是差异。
   pivot 之前有未对齐事件时，更早的分歧可能被掩盖。
2. `pivot`：算子、文件、行、调用序号、`diff_fields`、两侧的输入和输出、
   不同的区域。如果输入已经不同，pivot 是一个消费者；源头在上游。
3. `site_statuses` / `sites`：某个调用点的哪些调用一致，哪些不同。
   梯度消费者处先是一串 `o` 再是 `d`，说明有些梯度一致：
   源头是一个只给不同的那些梯度供数的反向算子。
4. `previous_ok`：最后一个一致的事件，不一定是生产者。

没有差异不能证明等价：检查覆盖情况，并用 `bi_check.bit_equal` 比较最终参数。

## 5. 交给 OpGuard 的 harness skill

在 OpGuard 的 `harness/skills/` 中：
- `investigate-first-diff`：负责边界和证据记录。返回根因解释或一个有边界的
  无结论结果。
- `trace-upstream`：在输入已经不同或生产者不清楚时使用。
  返回一个收窄的边界、一个候选机制，或所缺的观测。
- `validate-hypothesis`：一个候选原因，一个能区分的干预。
  返回 supported / refuted / inconclusive，附对照。

`suggest(facts, preflight=diff)["handoff"]` 说明该进入哪一个。

## 6. 映射到算子类别

`suggest` 把 pivot 映射到手册的类别
（[第 3 部分](../../handbook/3-kernels/README.md)、
[算子配方](../batch-invariant-kernel/references/operator-recipes.md)）
以及第 1 部分的根因模式，并给出理由。各类线索的含义，以及在实测案例中确认它
需要什么，见 [references/leads.md](references/leads.md)。

## 7. 修复

先用最便宜的：
1. 确定性设置或恢复修复（预检发现、缺失的状态）；
2. 与参考对齐的舍入点（两侧在同一处舍入）；
3. 通过 [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) 实现的
   batch-invariant（批不变）或确定性 kernel。
把代价记在修复旁边。

## 8. 确认

带着修复通过 OpGuard 重跑同一对运行。当 OpGuard 没有把任何事件标为 `diff`、
没有未对齐事件、且最终参数逐位相同时，就完成了。
“相同前缀变长”是指在相同步数内首个分歧点后移或消失；如果只是后移，回到第 4 步。

## 9. 回归条目

通过 `tools/bi_report.py` 给示例脚本加一项检查：原来那一对作为
`expect="fail"`（有记录的差异），修复后的一对作为 `expect="pass"`。
写一条案例条目：场景、症状、pivot、类别、修复、代价、启动次数。

## 何时停止

- 原因得到支持且重跑已确认：完成。
- 预检显示的阻断性差异就是原因（恢复状态、`warn_only`）：修复并确认；
  可能不需要捕获。
- pivot 之前有无法配对的未对齐事件，或覆盖漏掉了该区域：报告收窄的边界和
  所缺的最小观测（要加的边界、一个 rank、一步）。不要猜。
- 启动预算用完：报告哪些已经确定，哪些没有。

## 约束

- 只做逐位比较。没有容差，不对输出做舍入。
- 永远不要为了让捕获结果一致而修改 OpGuard 或工作负载的数值计算。
- 未配对的事件是空缺，不是差异。
- XOR fingerprint 相等不能证明张量相等；零 fingerprint
  永远不能当作证据。
- 统计启动次数：一次 `run_pair` 是四次工作负载启动。
