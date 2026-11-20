---
name: pretraining-alignment
description: 两次预训练本应逐 bit 一致却对不上时，找出它们从哪里开始分开，并查明原因。依次做这些事：选定参考运行；检查两次运行的确定性设置并做 diff；用 OpGuard 采集两次运行；取首个分歧点，交给 OpGuard 的 harness skill；把出问题的算子归到手册的算子类别；用确定性设置或 batch-invariant-kernel skill 修复；确认两次运行逐 bit 一致的前缀变长了；最后记一条回归条目。训练出了问题时使用这个 skill（loss 尖峰、升级或改配置之后开始漂移、从 checkpoint 恢复后没有沿原来的轨迹继续），两次本应一致的运行对不上时也用它。
---

# 预训练对齐

这个 skill 只处理一种情况：两次训练本应逐 bit 一致，实际却对不上。它要交付：

- 两次运行第一次出现不一致的地方；
- 算子类别和原因；如果查不出原因，就给出无结论的结果，并写清哪些已经确定、
  哪些还不知道；
- 修复方法和它的代价；
- 一条回归条目。

方法在[第 1 部分](../../handbook/1-pretraining/README.md)里讲，
[预训练案例](../../handbook/1-pretraining/localize-with-opguard.md)用实测数据把整个过程
走了一遍。用到的工具有：

- [tools/determinism_preflight.py](../../tools/determinism_preflight.py)，提供
  `report`、`findings` 和 `diff`；
- [tools/opguard_bridge.py](../../tools/opguard_bridge.py)，提供 `run_pair`、
  `first_divergence`、`coverage`、`suggest`、计算 fingerprint（指纹）的函数，以及
  `TraceWriter`；
- OpGuard 本身（`python -m opguard.tools`、`opguard-harness`）。本 skill 只驱动它，
  从不修改它。

## 每部分工作由谁来做

| 步骤 | 由谁完成 | 返回什么 |
|---|---|---|
| 选定参考运行、预检和 diff | 本 skill | 要回答的问题、两份报告和 diff 结果 |
| 采集与比较 | OpGuard 后端，通过 harness 驱动（`run_pair`） | 每个 rank 一份 Perfetto 导出，带 `verdict: not_evaluated` |
| 首个分歧点的事实 | OpGuard harness 的只读工具，由 `first_divergence` 调用 | pivot 事件、它之前最后一个一致的事件、未对齐事件的计数，以及 fingerprint |
| 判断哪些事件相互对应，并往上游追 | OpGuard 的 `investigate-first-diff` skill，然后是 `trace-upstream` | 收窄后的边界，或者还缺的那一项最小观测 |
| 检验一个原因 | OpGuard 的 `validate-hypothesis` skill | 支持、否定或无结论，并附上对照实验 |
| 确定算子类别和对应的 recipe | 本 skill，线索来自 `suggest` | 类别、recipe 和修复选项 |
| 修复 kernel | [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) skill | 改过的 kernel、它的验证结果和代价 |
| 确认修复并添加回归条目 | 本 skill | 一次没有分歧的重跑，以及这条条目 |

OpGuard 只产出事实。判断由你来做：哪些事件相互对应，哪个差异最先出现，原因是什么。
每个判断都要附上证据。`suggest` 只给线索，从不给判定。

## 1. 选定参考运行

按你要回答的问题来选参考运行（见第 1 部分第 1 步的表）。查残余的非确定性或 SDC
（静默数据损坏）时，让同一次运行和它自己的重放比较；查回归时，用最后一个正常的提交；
查新特性时，用只关掉这个开关、其余都相同的运行；查升级时，用旧版本。两次运行要用
相同的数据、种子和 checkpoint。把要检验的那一处差异写下来。

要保留的证据：两条启动命令、两次运行各自的提交，以及要检验的差异。

## 2. 对两次运行做预检，再 diff 两份报告

在每次运行里，等模型、优化器、scheduler 和 loader 都构建好、checkpoint 也恢复之后，
在第一步之前调用预检：

```python
rep = determinism_preflight.report(optimizer=opt, dataloader=loader, scheduler=sched,
                                   autocast=amp, model=model)
```

然后用 `determinism_preflight.diff(rep_a, rep_b, declared=[...])` 比较两份报告，
并声明要检验的那处差异（例如 `launch.argv` 或 `stack.torch`）。

- 如果出现 `bitwise_comparison: not_meaningful`，说明除了声明的差异之外，还有一处
  阻断性的差异。在采集之前先修掉它，或者把它也声明出来；否则首个分歧点只会把这处
  配置差异指给你看。
- `at_risk`（来自 `findings`）列出某次运行没有控制住的非确定性来源。最常见的是
  `warn_only=True`，它让 SDPA 的反向仍然是非确定性的（实测）。先把这些修好，
  之后自我重放可能直接就一致了。
- 如果要查的是恢复训练，那么恢复之后 `optimizer.state`、`scheduler`、`seeds`
  或 loader 位置上的任何差异，本身就是 bug。在实测案例中，step 计数器的问题在采集
  任何东西之前就在这里暴露了。

要保留的证据：两份报告文件和 diff 结果。

### 检查 loader 出口的数据

在两次运行的每个 rank 上，把 loader 返回的每个 batch 都记下来。先用
`rec = data_fingerprint.Recorder(dir, rank=dp_rank, meta={...})` 创建 recorder，
再对每个 batch 调用 `rec.record(step, batch)`（见[数据对齐](../../handbook/1-pretraining/data-alignment.md)）。
采集之前，`python tools/data_fingerprint.py compare A B` 必须报告没有任何记录不同；
如果两次运行的 DP 规模不同，要加上 `--global`。只要有一条记录不同，首个分歧点就出在
数据上，按差异的类别去修。要保留的证据：两份记录目录和 compare 的输出。

## 3. 用 OpGuard 采集两次运行

给每次运行准备一个单独的 checkout（`stage_checkouts`），然后调用 `run_pair`。它会把
工作负载启动四次，每次运行各做一次发现（discovery）和一次采集，最后比较两次采集的
结果。运行要短，跑几步、用一两块 GPU 就够。两次运行要使用相同的
`CUDA_VISIBLE_DEVICES` 编号，因为 OpGuard 比较的是设备字符串。

然后调用 `coverage(run)`。它列出发现阶段在 checkout 里找到、但没有安排插桩的调用点；
这些调用点没有边界记录。在实测案例中，用 `torch.nn.Linear` 时，GEMM 是从 torch 的
`linear.py` 里发起的，而这个文件不在 checkout 里，所以这些 GEMM 不会被采集；换成在
模型代码里调用 `F.linear` 的 Linear，就能采集到。OpGuard 只采集 eager 执行。对于
`torch.compile` 或 CUDA graph 内部的边界，要在那里算 fingerprint（bridge 里基于纯 torch
的函数在两者内部都能用），再用 `TraceWriter` 写出来。

要保留的证据：实验 ID、阶段列表、启动次数和墙钟时间，以及覆盖报告。

## 4. 读取首个分歧点

调用 `facts = first_divergence(report, root, og)`，按下面的顺序读结果：

1. `status_counts` 和 `unaligned_before`。未对齐的事件是配对里的空缺，不是差异。
   如果 pivot 之前有未对齐的事件，更早的分歧可能就藏在里面。
2. `pivot`，也就是第一个不同的事件。它给出算子、文件、行号、调用序号、`diff_fields`、
   两次运行各自的输入和输出，以及不同的区域。如果输入已经不同，说明 pivot 只是接收了
   更早产生的差异，源头还在上游。
3. `site_statuses` / `sites`，显示同一个调用点的哪些调用一致、哪些不同。如果某个梯度的
   消费者先出现一串 `o`、后出现 `d`，说明有些梯度一致、有些不一致，源头就是一个只给
   不一致的那些梯度提供输入的反向算子。
4. `previous_ok`，即最后一个一致的事件。它不一定是 pivot 输入的生产者。

没有发现差异，并不能证明两次运行等价。还要检查覆盖情况，并用 `bi_check.bit_equal`
比较最终参数。

## 5. 把分歧点交给 OpGuard 的 harness skill

OpGuard 的 `harness/skills/` 里有三个 skill 负责这一步：

- `investigate-first-diff` 负责检查这个边界并记下证据。它返回根因解释，或者一个写清哪些已经
  确定、哪些还不知道的无结论结果。
- `trace-upstream` 用在输入已经不同、或者不清楚输入是谁产生的时候。它返回收窄后的
  边界、一个候选机制，或者还缺的那项观测。
- `validate-hypothesis` 针对一个候选原因，用一个能把几种可能区分开的干预去检验。
  它返回 supported、refuted 或 inconclusive，并附上对照实验。

`suggest(facts, preflight=diff)["handoff"]` 会告诉你先从这三个里的哪一个开始。

## 6. 把 pivot 归到算子类别

`suggest` 会把 pivot 对应到手册中的算子类别
（[第 3 部分](../../handbook/3-kernels/README.md)、
[算子 recipe](../batch-invariant-kernel/references/operator-recipes.md)）
和第 1 部分列出的根因模式，并给出理由。每一类线索是什么意思、在实测案例中确认它
需要做什么，见 [references/leads.md](references/leads.md)。

## 7. 修复原因

先试代价最小的修复：

1. 改一项确定性设置，或者补上保存和恢复时漏掉的内容（对应预检发现的问题或缺失的状态）；
2. 挪动一个舍入点，让它和参考运行一致，这样两次运行就在同一个地方做舍入；
3. 用 [batch-invariant-kernel](../batch-invariant-kernel/SKILL.md) skill 做一个
   batch-invariant 或确定性的 kernel。

把代价记在修复旁边。

## 8. 确认修复

带上修复，通过 OpGuard 重跑同一对运行。当 OpGuard 没有把任何事件标成 `diff`、
没有未对齐的事件、最终参数也逐 bit 一致时，就完成了。在同样的步数内，有效的修复会让
首个分歧点往后移，或者让它消失；这两种情况下，两次运行逐 bit 一致的前缀都变长了。
如果分歧点只是往后移了，回到第 4 步。

## 9. 添加回归条目

用 `tools/bi_report.py` 给一个示例脚本加一项检查。把原来那一对运行登记为
`expect="fail"`，因为它的差异现在有了记录；把修复后的那一对登记为 `expect="pass"`。
再写一条案例条目，记下场景、症状、pivot、算子类别、修复方法、代价和启动次数。

## 何时停止

- 原因得到支持，重跑也确认了修复，就完成了。
- 如果预检发现的阻断性差异本身就是原因（恢复的状态、`warn_only`），修好并确认即可，
  可能根本不需要采集。
- 如果 pivot 之前有无法配对的未对齐事件，或者覆盖漏掉了这块区域，就报告收窄后的边界
  和还缺的最小一项观测（要加的一个边界、一个 rank、一步）。不要猜。
- 启动次数的预算用完时，报告哪些已经确定、哪些还没有。

## 约束

- 只做逐 bit 比较：不设容差，也不对输出做舍入。
- 永远不要为了让两次采集一致，去修改 OpGuard 或工作负载的数值计算。
- 没有配对的事件是空缺，不是差异。
- XOR fingerprint 相等不能证明张量相等；为零的 fingerprint 永远不能当作证据。
- 记下启动次数：一次 `run_pair` 会启动四次工作负载。
