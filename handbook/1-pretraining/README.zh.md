# 1. 预训练：找到运行出错的地方

## 我们在哪里

[第 0 部分](../0-foundations/README.md)说明了和的比特取决于求和顺序，也说明了两次运行可以在算子边界上用 fingerprint 逐位比较。本部分把这些用到两次本应一致却不一致的训练运行上。常见的起因：

- 升级、迁移或改配置之后 loss 突刺或缓慢漂移；
- 恢复训练后没有回到同一条轨迹；
- 怀疑某台机器有静默数据损坏（SDC）。

这个工作流把你从症状带到经过确认的修复，再到一个回归检查。它就是 OpGuard 的方法，[pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md) 以 agent 的方式执行它。

<figure class="fx" data-fig="pre-workflow" aria-label="预训练工作流" markdown>
<div class="fx-stage" markdown>
<div class="fx-steps" markdown="span"><span class="fx-step" data-hot="0">[1. 选择参考运行](#step-1-pick-a-reference-run)</span><span class="fx-step" data-hot="1">[2. 预检与控制](#step-2-preflight-and-control)</span><span class="fx-step" data-hot="2">[3. 采集](#step-3-capture)</span><span class="fx-step" data-hot="3">[4. 首个分歧点](#step-4-take-the-first-divergence)</span><span class="fx-step is-key" data-hot="4">[5. 分类](#step-5-classify)</span><span class="fx-step" data-hot="6">[6. 修复](#step-6-fix)</span><span class="fx-step" data-hot="7">[7. 确认](#step-7-confirm)</span><span class="fx-step" data-hot="8">[8. 回归](#step-8-keep-it-as-a-regression-check)</span></div>
<div class="fx-seq is-col" markdown="span"><span class="fx-node is-key fx-root" data-hot="4">5. 分类</span><span class="fx-fork" style="--n:4"><span class="fx-leg" data-hot="5"><span class="fx-when">backward、优化器、loss、恢复训练</span><span class="fx-node" data-hot="5 6">[训练 step](training-step.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">梯度集合通信</span><span class="fx-node" data-hot="5 6">[数据并行](data-parallel.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">融合代码、graph 分桶</span><span class="fx-node" data-hot="2 5 6">[编译代码](capture-compiled-code.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">设备本身</span><span class="fx-node" data-hot="5 6">[SDC](sdc.md)</span></span></span></div>
</div>
<ol class="fx-say">
<li>1. 选择参考运行：它决定这次比较回答哪个问题。</li>
<li>2. 预检与控制：两次运行只在被测的那一件事上不同。</li>
<li data-hold="1.5">3. 在两次运行的每个边界上采集 fingerprint。如果作业运行编译代码，先读它那一页。</li>
<li>4. 取第一个比特不同的边界。</li>
<li>5. 分类：首个分歧点指向一个原因。</li>
<li data-hold="1.5">常见的原因各有一页。</li>
<li>6. 在那里修复，用能消除原因的最小改动。</li>
<li>7. 确认：重新采集这一对，处处一致。</li>
<li data-hold="2">8. 把这一对留在回归套件里，让它一直保持一致。</li>
</ol>
<figcaption>图：第 1 到第 8 步是主线。到第 5 步，首个分歧点指向一个原因，常见的原因各有一页；在那里修复，然后确认（第 7 步）。编译代码也是一种采集情形：如果生产作业运行编译代码，在第 3 步之前先读它那一页。</figcaption>
</figure>

| 步骤 | 解决什么 | 页面 |
|---|---|---|
| 1. 选择参考运行 | 这次比较回答哪个问题 | 本页 |
| 2. 预检与控制 | 两次运行只在待测的那一点上不同 | [确定性与预检](determinism-and-preflight.md)；训练 step 见[训练 step](training-step.md)；梯度集合通信见[数据并行](data-parallel.md) |
| 2. 预检与控制：数据 | 每个 rank 在每一步看到的 token 比特都相同 | [数据对齐](data-alignment.md) |
| 3. 采集 | 两次运行中每个模型级边界的 fingerprint | [用 OpGuard 定位](localize-with-opguard.md)；如果作业运行 `torch.compile` 或 CUDA graph，再看[作业运行编译代码时](capture-compiled-code.md) |
| 4. 首个分歧点 | 比特第一次不同的边界 | [用 OpGuard 定位](localize-with-opguard.md) |
| 5. 分类 | 原因属于哪一类：设置、训练 step 的某部分、集合通信、编译代码、kernel、硬件 | 本页，然后去该原因的页面 |
| 6. 修复 | 消除原因的最小改动 | 该原因的页面；kernel 见[第 3 部分](../3-kernels/README.md) |
| 7. 确认 | 重新采集这一对，处处一致 | [用 OpGuard 定位](localize-with-opguard.md) |
| 8. 回归 | 这一对留在测试集里并持续一致 | [回归](../4-operations/regression.md) |

本部分的页面按工作流排列。先是第 2 步：[确定性与预检](determinism-and-preflight.md)，然后是[数据对齐](data-alignment.md)。接着是核心循环[用 OpGuard 定位](localize-with-opguard.md)：采集、首个分歧点、向外追溯、修复、确认。它之后的页面对应首个分歧点指向的地方：[训练 step](training-step.md)（backward、优化器、loss、恢复训练）、[数据并行](data-parallel.md)（梯度集合通信）、[作业运行编译代码时](capture-compiled-code.md)（`torch.compile` 或 CUDA graph）和 [SDC](sdc.md)（硬件）。

## 思路：两次运行，一处差异

除一个声明过的差异外，其余全部保持相同，那么第一个逐位差异要么就是这个差异，要么是一个故障。一个小例子：两次运行完全相同，只是 suspect 的 norm 在舍入到 bf16 之前先乘权重。它们按调用顺序的边界 fingerprint：

<figure class="fx" data-fig="first-divergence" aria-label="两次运行逐个边界比较">
<div class="fx-stage">
<div class="fx-grid" style="--cols:3">
<span class="fx-th">边界</span><span class="fx-th">参考</span><span class="fx-th">suspect</span><span class="fx-th">相等？</span>
<span class="fx-label" data-hot="1">token embedding</span><span class="fx-node is-mono" data-hot="1">e</span><span class="fx-node is-mono" data-hot="1">e</span><span class="fx-pair is-ok" data-at="1">是</span>
<span class="fx-label" data-hot="2 5">attention norm</span><span class="fx-node is-mono" data-hot="2 5">n</span><span class="fx-node is-mono to-bad" data-hot="2 5">n′</span><span class="fx-pair is-bad" data-at="2">否：首个分歧点</span>
<span class="fx-label" data-hot="3">q 投影</span><span class="fx-node is-mono" data-hot="3">q</span><span class="fx-node is-mono to-bad" data-hot="3">q′</span><span class="fx-pair" data-at="3">否</span>
<span class="fx-label" data-hot="4">… 之后的每个边界</span><span class="fx-node is-mono" data-hot="4">…</span><span class="fx-node is-mono to-bad" data-hot="4">…</span><span class="fx-pair" data-at="4">否</span>
</div>
</div>
<ol class="fx-say">
<li>两次运行完全相同，只是 suspect 的 norm 乘权重的位置不同。按调用顺序的 fingerprint。</li>
<li>token embedding 在两次运行上相等。</li>
<li data-hold="1.5">attention norm 不同：首个分歧点。</li>
<li>q 投影也不同，因为它的输入不同了。</li>
<li>之后的每个边界也都不同。</li>
<li data-hold="2">只读第一个差异，然后从它向外走到原因。</li>
</ol>
<figcaption>图：除一个声明过的差异外其余全部相同，第一个逐位差异要么就是这个差异，要么是一个故障。</figcaption>
</figure>

norm 之后的一切也都不同，因为它的输入不同了。所以只读第一个差异，然后从它向外走到原因。

**规则。** 让两次运行只在一件写下来的事情上不同，按调用顺序逐位比较，从第一个差异开始。这是一个对照实验：参考运行就是对照组。

## 步骤

### 第 1 步：选择参考运行 {#step-1-pick-a-reference-run}

**目标：** 一个能隔离你的问题的参考。一次采集的 suspect 运行和一次参考运行就够了，suspect 不必是确定的。

| 参考 | 回答 |
|---|---|
| 自重放（一切相同） | 是否有残余非确定性、竞争或 SDC？ |
| 最近一个稳定的提交 | 哪个改动引入了漂移？ |
| 改一个配置（关掉一个开关：overlap、offload、recompute、融合） | 这个特性是否改变了计算？ |
| 库或驱动版本 | 升级是否改变了数值？ |
| 跨栈（另一个框架，同一个模型） | 两个实现在哪里不一致？ |
| 单 GPU，或逻辑 TP 大于 1 的 TP 模拟器 | 并行实现是否有错？ |

**做法：** 两次运行使用相同的数据、种子和检查点。写下待测的那一个差异；预检的 diff 把它作为 `declared` 接收。

**成功：** 一句话说清这个差异，别的都不应不同。

### 第 2 步：预检与控制

**目标：** 消除可避免的变化，并证明它已消除。五个控制面：随机性和数据顺序、库 kernel、集合通信、检查点恢复、TP 度。不要固定调度或用户 kernel：如果它们改变了比特，那就是信号。

**做法：** 设置这些控制面，然后用 [determinism_preflight.py](../../tools/determinism_preflight.py) 在每个作业内部读回，并对两份报告做 diff（[确定性与预检](determinism-and-preflight.md)）。训练 step 的各部分见[训练 step](training-step.md)；梯度集合通信见[数据并行](data-parallel.md)。

**成功：** diff 只列出声明过的差异。

### 第 3 步：采集

**目标：** 两次运行中每个模型级边界输入和输出的 fingerprint，在算子所在的 stream 上计算，不增加同步。

**做法：** [opguard_bridge.py](../../tools/opguard_bridge.py) 让两边都走 OpGuard 的 harness：发现、计划、采集、比较。对 `torch.compile` 或 CUDA graph 下的代码，eager 采集看不到生产实际算出的比特；用 [graph_probe.py](../../tools/graph_probe.py)（[作业运行编译代码时](capture-compiled-code.md)）。

| 选项 | 给出什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| Eager OpGuard 采集 | 每个被发现的边界，带调用点 | 每个边界很小（几个设备 kernel，不同步）；整次采集的开销这里没有测量 | 运行是 eager 的，或 eager 重放能复现症状 |
| 编译代码中的 graph probe | 编译代码和 graph 重放产生的比特 | 在 CUDA graph 内每个 probe 很小，在 graph 外很大；小 step 上 probe 多了累加起来就大 | 生产使用 `torch.compile` 或 CUDA graph |
| 完整张量转储 | 值，而不是 fingerprint | 大（2 倍或更多） | 只用于 fingerprint 已经指出的区域 |

**成功：** `opguard_bridge.coverage(run)` 显示所关心的区域已被采集，包括 GEMM。

### 第 4 步：取首个分歧点

**目标：** 按调用顺序第一个 fingerprint 不同的边界。

**做法：** 两份 trace 从不按位置逐一对齐：overlap、融合和重写会移动事件，一边可能多出事件。OpGuard 用保序匹配把它们配对，以明确的边界为锚点，把不确定的事件留作空缺。`opguard_bridge.first_divergence(...)` 收集关于这个 pivot 的事实（[用 OpGuard 定位](localize-with-opguard.md)）。

**成功：** 一个 pivot，它之前没有未对齐的事件。未配对的事件不是差异。

### 第 5 步：分类 {#step-5-classify}

**目标：** 说出原因的类别，因为每一类有自己的修复和页面。

**做法：** 读 pivot 的源码行、它的生产者和调用点状态；`opguard_bridge.suggest(...)` 列出带理由的线索。OpGuard 在生产中定位到的根因可归为这些类别：

| 类别 | 例子 | 去哪里 |
|---|---|---|
| 设置缺失或错误 | 某个 backward kernel 没有开启确定性模式 | [确定性与预检](determinism-and-preflight.md) |
| 有竞争或非确定的 kernel | `torch.embedding` 的 backward、融合的 embedding backward、缺少屏障的 Triton kernel | [训练 step](training-step.md)；[第 3 部分](../3-kernels/README.md) |
| 状态记录 | 恢复训练时优化器的 step 变量被覆盖 | [训练 step](training-step.md) |
| 依赖拓扑的归约 | 不同 NCCL 拓扑下的 reduce-scatter | [数据并行](data-parallel.md) |
| 编译代码 | 融合或 graph 分桶改变了一个归约 | [作业运行编译代码时](capture-compiled-code.md) |
| stream 误用 | fp32 RMSNorm 的 stream 竞争、激活 offload | [用 OpGuard 定位](localize-with-opguard.md) |
| 布局假设 | QKV 投影依赖隐式的连续性 | [用 OpGuard 定位](localize-with-opguard.md) |
| MoE 路由 | 直方图分箱不匹配导致路由漂移 | [第 3 部分：MoE](../3-kernels/moe.md) |
| 硬件 SDC | 布尔 scatter 把一个 token 送错位置，在许多算子之后才表现为形状不匹配 | [SDC](sdc.md) |

**成功：** 一个假设，以及指向它的证据。

### 第 6 步：修复

**目标：** 消除原因的最小改动。

**做法：** 设置问题回到预检；训练 step 的部分按它的清单处理；kernel 走 [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md) 及其[配方](../../skills/batch-invariant-kernel/references/operator-recipes.md)。OpGuard 的 harness skill `validate-hypothesis` 一次检验一个原因。

**成功：** 这个干预按假设所预测的方式改变了 pivot。

### 第 7 步：确认

**目标：** 证明是这个修复、而不是别的，让两次运行一致。

**做法：** 带上修复，重新采集这一对。

**成功：** 没有事件不同，没有未对齐的事件，最终参数逐位相同。

### 第 8 步：作为回归检查保留

**目标：** 下次升级之后一致性仍然成立。

**做法：** 把这一对，或能复现它的最小案例，加入回归测试集（[回归](../4-operations/regression.md)），并记录它成立的栈。

**成功：** 测试集运行这个案例并报告相等。

## 可能出错的地方

!!! warning "两次运行之间不止一处差异"
    两处未声明的差异会产生一个任何一处都无法单独解释的首个分歧点。你在追一个混合体。采集之前，对预检报告做 diff，去掉声明之外的所有差异。

!!! warning "采集漏掉了该区域"
    没有插桩的边界不可能成为首个分歧点，于是 pivot 落在更后面、落在错误的算子上。读首个差异之前先检查 `coverage`。

!!! warning "把 pivot 当成原因"
    首个差异是比特第一次显现的地方，不是它们产生的地方：没有边界的非确定 backward 会先出现在下一个消费者处。从 pivot 向上游走，并检验假设。

!!! warning "过早把设备间的不一致称为 SDC"
    不同的驱动或库在没有任何故障时也会给出不同的比特。你会下线一个健康的节点。先排除软件差异（[SDC](sdc.md)）。

## 小结

- 选一个能隔离单个问题的参考运行，并把那一个差异写下来。
- 控制可避免的非确定性，并用预检 diff 检查。
- 采集模型级边界，检查覆盖，取第一个逐位差异，且它之前没有未对齐的事件。
- 给原因分类，用最小改动修复，重新采集来确认，并把这一对作为回归检查保留。

## 深入阅读

- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)：agent 执行的工作流。
- [opguard_bridge.py](../../tools/opguard_bridge.py)、[determinism_preflight.py](../../tools/determinism_preflight.py)、[graph_probe.py](../../tools/graph_probe.py)、[sdc_scan.py](../../tools/sdc_scan.py)。
- [报告：OpGuard 演练，以及带生产数字的本概览旧版](../../examples/pretraining_e2e/REPORT.md)。
- 原因是没有不变实现的 kernel 时，见[第 3 部分](../3-kernels/README.md)。
