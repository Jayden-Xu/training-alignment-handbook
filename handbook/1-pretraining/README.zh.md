# 1. 预训练：找出两次运行从哪里开始出错

## 我们在哪里

[第 0 部分](../0-foundations/README.md)讲过，求和结果的每个 bit 都取决于加法的顺序；也讲过，可以在算子边界上用 fingerprint 逐 bit 比较两次运行。这一部分把这些方法用在两次本该一致、实际却不一致的训练上。遇到下面这些情况，通常就该走这条流程：

- 升级、迁移或改配置之后，loss 突然尖峰，或者慢慢漂移；
- 从 checkpoint 恢复训练后，没有沿着原来的轨迹继续走；
- 怀疑某台机器有静默数据损坏（SDC）。

这条流程带你从症状出发，一直走到一个经过确认的修复，再把它留成回归检查。OpGuard 就是围绕这套方法设计的；[pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md) 让 agent 也能按这套方法执行。

<figure class="fx" data-fig="pre-workflow" aria-label="预训练对齐的工作流程" markdown>
<div class="fx-stage" markdown>
<div class="fx-steps" markdown="span"><span class="fx-step" data-hot="0">[1. 选参考运行](#step-1-pick-a-reference-run)</span><span class="fx-step" data-hot="1">[2. 预检与控制](#step-2-preflight-and-control)</span><span class="fx-step" data-hot="2">[3. 采集](#step-3-capture)</span><span class="fx-step" data-hot="3">[4. 首个分歧点](#step-4-take-the-first-divergence)</span><span class="fx-step is-key" data-hot="4">[5. 判断原因类别](#step-5-classify)</span><span class="fx-step" data-hot="6">[6. 修复](#step-6-fix)</span><span class="fx-step" data-hot="7">[7. 确认](#step-7-confirm)</span><span class="fx-step" data-hot="8">[8. 回归检查](#step-8-keep-it-as-a-regression-check)</span></div>
<div class="fx-seq is-col" markdown="span"><span class="fx-node is-key fx-root" data-hot="4">5. 判断原因类别</span><span class="fx-fork" style="--n:4"><span class="fx-leg" data-hot="5"><span class="fx-when">反向、优化器、loss 或恢复训练</span><span class="fx-node" data-hot="5 6">[训练 step](training-step.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">对梯度求和的集合通信</span><span class="fx-node" data-hot="5 6">[数据并行](data-parallel.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">融合后的代码或 graph 分桶</span><span class="fx-node" data-hot="2 5 6">[torch.compile 与 CUDA graph](capture-compiled-code.md)</span></span><span class="fx-leg" data-hot="5"><span class="fx-when">设备本身</span><span class="fx-node" data-hot="5 6">[SDC](sdc.md)</span></span></span></div>
</div>
<ol class="fx-say">
<li>1. 选定参考运行。它决定了这次比较要回答哪个问题。</li>
<li>2. 预检与控制：确保两次运行只在你要测试的那一点上不同。</li>
<li data-hold="1.5">3. 在两次运行的每个边界上采集 fingerprint。如果训练任务用了 torch.compile 或 CUDA graph，先读那一页。</li>
<li>4. 找出第一个结果不同的边界。</li>
<li>5. 判断原因类别：首个分歧点会指向某一类原因。</li>
<li data-hold="1.5">常见的每一类原因都有专门的一页。</li>
<li>6. 到那一页去修复，用能消除原因的最小改动。</li>
<li>7. 确认：重新采集这一对运行，检查它们处处一致。</li>
<li data-hold="2">8. 把这一对运行放进回归套件，让它以后也一直保持一致。</li>
</ol>
<figcaption>图：第 1 到第 8 步是主线。到第 5 步时，首个分歧点会指向某一类原因，常见的每一类都有专门的一页；在那一页修复，再回来确认（第 7 步）。torch.compile 和 CUDA graph 还会改变采集的方式：如果生产环境的训练任务用了它们，在第 3 步之前先读那一页。</figcaption>
</figure>

| 步骤 | 这一步要弄清什么 | 在哪一页 |
|---|---|---|
| 1. 选参考运行 | 这次比较要回答哪个问题 | 本页 |
| 2. 预检与控制 | 两次运行只在你要测试的那一点上不同 | [确定性与预检](determinism-and-preflight.md)；训练 step 的各个部分见[训练 step](training-step.md)；对梯度求和的集合通信见[数据并行](data-parallel.md) |
| 2. 预检与控制（数据部分） | 每个 rank 在每一步拿到的 token 逐 bit 一致 | [数据对齐](data-alignment.md) |
| 3. 采集 | 两次运行在每个模型级边界上都有 fingerprint | [用 OpGuard 定位](localize-with-opguard.md)；如果训练任务用了 `torch.compile` 或 CUDA graph，还要看[用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md) |
| 4. 首个分歧点 | 第一个结果不同的边界是哪一个 | [用 OpGuard 定位](localize-with-opguard.md) |
| 5. 判断原因类别 | 原因属于哪一类：设置、训练 step 的某个部分、集合通信、torch.compile 或 CUDA graph、kernel，还是硬件 | 先看本页，再去这类原因对应的那一页 |
| 6. 修复 | 能消除原因的最小改动 | 这类原因对应的那一页；如果是 kernel，见[第 3 部分](../3-kernels/README.md) |
| 7. 确认 | 重新采集这一对运行后，处处一致 | [用 OpGuard 定位](localize-with-opguard.md) |
| 8. 回归检查 | 这一对运行留在回归套件里，并且一直保持一致 | [回归](../4-operations/regression.md) |

这一部分的各页按流程的顺序排列。最先做的是第 2 步，分成[确定性与预检](determinism-and-preflight.md)和[数据对齐](data-alignment.md)两页。接下来是核心循环[用 OpGuard 定位](localize-with-opguard.md)：采集两次运行，找到首个分歧点，从它追到原因，修复，再确认修复有效。后面几页对应首个分歧点最常指向的地方：[训练 step](training-step.md) 讲反向、优化器、loss 和恢复训练；[数据并行](data-parallel.md)讲对梯度求和的集合通信；[用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md)讲怎样在 `torch.compile` 生成的代码和 CUDA graph 重放内部取 fingerprint；[SDC](sdc.md) 讲硬件。

## 思路：让两次运行只差一件事

如果除了一件事先声明好的差异之外，其他一切都保持相同，那么两次运行之间第一个逐 bit 的差异，要么就是这件事造成的，要么就是一个故障。举个小例子：两次运行完全相同，只是 suspect 运行里的 norm 先乘权重、再舍入到 bf16。下面按调用顺序列出它们在各个边界上的 fingerprint：

<figure class="fx" data-fig="first-divergence" aria-label="两次运行逐个边界比较">
<div class="fx-stage">
<div class="fx-grid" style="--cols:3">
<span class="fx-th">边界</span><span class="fx-th">参考运行</span><span class="fx-th">suspect 运行</span><span class="fx-th">相等吗？</span>
<span class="fx-label" data-hot="1">token embedding</span><span class="fx-node is-mono" data-hot="1">e</span><span class="fx-node is-mono" data-hot="1">e</span><span class="fx-pair is-ok" data-at="1">相等</span>
<span class="fx-label" data-hot="2 5">attention norm</span><span class="fx-node is-mono" data-hot="2 5">n</span><span class="fx-node is-mono to-bad" data-hot="2 5">n′</span><span class="fx-pair is-bad" data-at="2">不等，这就是首个分歧点</span>
<span class="fx-label" data-hot="3">q 投影</span><span class="fx-node is-mono" data-hot="3">q</span><span class="fx-node is-mono to-bad" data-hot="3">q′</span><span class="fx-pair" data-at="3">不等</span>
<span class="fx-label" data-hot="4">… 之后的每个边界</span><span class="fx-node is-mono" data-hot="4">…</span><span class="fx-node is-mono to-bad" data-hot="4">…</span><span class="fx-pair" data-at="4">不等</span>
</div>
</div>
<ol class="fx-say">
<li>两次运行完全相同，只是 suspect 运行里 norm 乘权重的时机不同。下面按调用顺序列出 fingerprint。</li>
<li>token embedding 在两次运行里相等。</li>
<li data-hold="1.5">attention norm 不相等，这就是首个分歧点。</li>
<li>q 投影也不相等，因为它的输入已经不同了。</li>
<li>之后的每个边界也都不相等。</li>
<li data-hold="2">只看第一个差异，再从它追到原因。</li>
</ol>
<figcaption>图：除了一件事先声明的差异，其他一切都保持相同时，第一个逐 bit 的差异要么来自这件事，要么来自一个故障。</figcaption>
</figure>

norm 之后的所有边界也都不同，因为它们的输入已经变了。所以只看第一个差异，然后从它出发去找原因。

**规则。** 让两次运行只差一件事，并把这件事写下来。按调用顺序逐 bit 比较，从第一个差异开始查。这其实就是一次对照实验，参考运行就是对照组。

## 步骤

### 第 1 步：选一个能单独回答你的问题的参考运行

**目标：** 选一个参考运行，让它把你关心的问题单独隔离出来。只要一次采集过的 suspect 运行和一次参考运行就够了，suspect 运行本身不必是确定的。

| 参考运行 | 它回答的问题 |
|---|---|
| 同一个任务再跑一遍，一切都不变 | 有没有残余的非确定性、竞争或 SDC？ |
| 最近一个稳定的提交 | 是哪个改动引入了漂移？ |
| 同样的配置，只关掉一个特性（overlap、offload、recompute 或融合） | 这个特性会不会改变计算？ |
| 升级前的库或驱动版本 | 升级有没有改变数值？ |
| 同一个模型在另一个框架里的实现 | 两种实现在哪里不一致？ |
| 单 GPU，或一个让逻辑 TP 保持大于 1 的 TP 模拟器 | 并行实现有没有写错？ |

**做法：** 让两次运行用同样的数据、种子和 checkpoint。把你要测试的那一个差异写下来；之后把它作为 `declared` 传给预检的 diff。

**成功标准：** 你能用一句话说清这个差异，而且除此之外没有别的地方应该不同。

### 第 2 步：控制可避免的变化，并确认它们已经消除

**目标：** 消除可避免的变化，并证明它们确实没有了。这些变化来自五个方面：随机性和数据顺序、库里的 kernel、集合通信、从 checkpoint 恢复训练，以及 TP 规模。不要去固定调度或用户自己写的 kernel：如果它们改变了结果，这个变化正是你要找的信号。

**做法：** 设置好这五个方面，然后在每个训练任务内部用 [determinism_preflight.py](../../tools/determinism_preflight.py) 把它们读回来，再对两份报告做 diff（[确定性与预检](determinism-and-preflight.md)）。训练 step 各部分的设置见[训练 step](training-step.md)；对梯度求和的集合通信见[数据并行](data-parallel.md)。

**成功标准：** diff 里只剩下声明过的那个差异。

### 第 3 步：在两次运行上采集 fingerprint

**目标：** 在两次运行的每个模型级边界上，记录输入和输出的 fingerprint。fingerprint 在算子自己的 stream 上计算，不会额外引入同步。

**做法：** [opguard_bridge.py](../../tools/opguard_bridge.py) 让两次运行都经过 OpGuard 的 harness：先发现边界，再规划在哪里插桩，然后采集并比较。如果代码跑在 `torch.compile` 或 CUDA graph 下，eager 采集看不到生产环境实际算出的结果；这时改用 [graph_probe.py](../../tools/graph_probe.py)（见[用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md)）。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 用 OpGuard 做 eager 采集 | 发现的每个边界，以及它的调用点 | 每个边界都很小（几个设备上的 kernel，没有同步）；整次采集的开销这里没有测量 | 运行本身是 eager 的，或者用 eager 重放能复现症状 |
| 在编译后的代码里放 graph probe | 编译后的代码和 graph 重放实际算出的结果 | 在 CUDA graph 内部，每个 probe 都很小；在 graph 外面就大了；step 很小而 probe 很多时，加起来也会变大 | 生产环境用了 `torch.compile` 或 CUDA graph |
| 完整转储张量 | 数值本身，而不是 fingerprint | 大（2 倍或更多） | 只用于 fingerprint 已经指出的区域 |

**成功标准：** `opguard_bridge.coverage(run)` 显示你关心的区域（包括其中的 GEMM）都已采集。

### 第 4 步：找到首个分歧点

**目标：** 按调用顺序，找出第一个 fingerprint 不同的边界。

**做法：** 两份 trace 从来不会按位置一一对齐：overlap、融合和改写都会挪动事件，而且其中一份 trace 可能多出另一份没有的事件。所以 OpGuard 用保序匹配来配对事件：先以能确定对应的边界为锚点，拿不准的事件就不配对，留作空缺。第一对不相等的事件叫 **pivot**。`opguard_bridge.first_divergence(...)` 会收集关于它的所有事实（[用 OpGuard 定位](localize-with-opguard.md)）。

**成功标准：** 找到一个 pivot，而且它之前没有未对齐的事件。注意，没配上对的事件不算差异。

### 第 5 步：判断原因属于哪一类

**目标：** 说出原因属于哪一类，因为每一类都有自己的修法和专门的一页。

**做法：** 看 pivot 所在的源码行、产生它输入的那些算子，以及它所在调用点上每次调用的状态。`opguard_bridge.suggest(...)` 会列出可能的线索，每条都附有理由。OpGuard 在生产环境中定位到的根因，可以归为下面几类：

| 原因类别 | 例子 | 去哪一页 |
|---|---|---|
| 设置缺失或设错 | 某个反向 kernel 没有打开确定性模式 | [确定性与预检](determinism-and-preflight.md) |
| kernel 有竞争或本身不确定 | `torch.embedding` 的反向、融合过的 embedding 反向、缺少 barrier 的 Triton kernel | [训练 step](training-step.md)；[第 3 部分](../3-kernels/README.md) |
| 状态没有正确恢复 | 恢复训练时，优化器的 step 变量被覆盖了 | [训练 step](training-step.md) |
| 归约结果依赖拓扑 | NCCL 拓扑不同时，reduce-scatter 的结果不同 | [数据并行](data-parallel.md) |
| torch.compile 或 CUDA graph | 融合或 graph 分桶改变了某个归约 | [用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md) |
| stream 用错了 | fp32 RMSNorm 里的 stream 竞争；activation offload | [用 OpGuard 定位](localize-with-opguard.md) |
| 对内存布局的错误假设 | QKV 投影想当然地认为张量是连续的 | [用 OpGuard 定位](localize-with-opguard.md) |
| MoE 路由 | 直方图分箱对不上，导致路由慢慢漂移 | [第 3 部分：MoE](../3-kernels/moe.md) |
| 硬件的静默数据损坏 | 一次布尔 scatter 把一个 token 送错了位置，直到许多算子之后才表现为形状不匹配 | [SDC](sdc.md) |

**成功标准：** 得到一个假设，以及指向它的证据。

### 第 6 步：修复原因

**目标：** 用最小的改动消除原因。

**做法：** 修法取决于原因的类别。设置问题，回到预检里改正；训练 step 某个部分的问题，按那一页的清单处理；kernel 的问题，用 [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md) 和它的[recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md)来修。OpGuard harness 里的 `validate-hypothesis` skill 每次只检验一个原因。

**成功标准：** 改动对 pivot 的影响，和假设预测的一样。

### 第 7 步：确认是修复让两次运行一致的

**目标：** 证明是这个修复、而不是别的东西，让两次运行变得一致。

**做法：** 带上修复，重新采集这一对运行。

**成功标准：** 没有任何事件不同，没有未对齐的事件，最终参数逐 bit 一致。

### 第 8 步：把这一对运行留作回归检查

**目标：** 下次升级之后，两次运行仍然一致。

**做法：** 把这一对运行，或者能复现它的最小案例，加进回归套件（[回归](../4-operations/regression.md)），并记下这个一致性是在哪个软件栈上成立的。

**成功标准：** 回归套件跑这个案例时，报告结果一致。

## 可能出错的地方

!!! warning "两次运行之间不止一处差异"
    如果有两处没声明的差异，它们会落在同一个首个分歧点上，但哪一处都不能单独解释它，你追的其实是几种原因叠在一起的结果。采集之前，先对预检报告做 diff，把声明之外的差异全部去掉。

!!! warning "采集漏掉了要查的区域"
    没有插桩的边界不可能成为首个分歧点，于是 pivot 会落到更后面、落在错误的算子上。读首个差异之前，先检查 `coverage`。

!!! warning "把 pivot 当成了原因"
    第一个差异只说明错误从哪里开始能被看到，并不说明错误是在哪里产生的。比如不确定的反向计算没有自己的边界，它的影响要到下一个用它输出的算子才显现出来。从 pivot 往上游追，并检验你的假设。

!!! warning "过早把设备之间的不一致判定为 SDC"
    驱动或库不同时，即使没有任何故障，结果也会不同。如果把这当成 SDC，你会把一个健康的节点下线。先排除软件上的差异（[SDC](sdc.md)）。

## 小结

- 选一个能单独回答一个问题的参考运行，并把两次运行之间唯一的差异写下来。
- 控制可避免的非确定性，并用预检的 diff 确认。
- 在模型级边界上采集，检查覆盖范围，然后取第一个逐 bit 的差异，它之前不能有未对齐的事件。
- 判断原因属于哪一类，用最小的改动修复，重新采集来确认，再把这一对运行留作回归检查。

## 深入阅读

- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md) 让 agent 按这条流程执行。
- 工具：[opguard_bridge.py](../../tools/opguard_bridge.py)、[determinism_preflight.py](../../tools/determinism_preflight.py)、[graph_probe.py](../../tools/graph_probe.py) 和 [sdc_scan.py](../../tools/sdc_scan.py)。
- [报告：OpGuard 演练，以及这篇概览的早期版本（含生产环境的数字）](../../examples/pretraining_e2e/REPORT.md)。
- 如果原因是一个还没有不变实现的 kernel，见[第 3 部分](../3-kernels/README.md)。
