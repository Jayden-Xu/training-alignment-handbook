# 用 OpGuard 定位

## 我们在哪里

这一页是[预训练流程](README.md)的核心循环，涵盖第 3、4、6、7 步：采集两次运行，找到首个分歧点，修复原因，再确认修复有效。你手上有一次参考运行和一次 suspect 运行，它们只在声明过的那一点上不同（[预检](determinism-and-preflight.md)），读到的 batch 也完全相同（[数据对齐](data-alignment.md)）。现在还不知道的是：两次运行从哪里开始分开，以及为什么。判断原因属于哪一类（第 5 步）放在[流程总览页](README.md#step-5-classify)里讲；这一页后面的几页，分别讲首个分歧点最常指向的几类原因：[训练 step](training-step.md)、[梯度的集合通信](data-parallel.md)、[torch.compile 与 CUDA graph](capture-compiled-code.md) 和[硬件](sdc.md)。

## 思路：第一个差异只告诉你从哪里查起，不等于原因

OpGuard 在两次运行的每个模型级边界上记录 fingerprint，把两份 trace 配对，然后报告第一对不相等的事件。这一对事件叫 **pivot**。

<figure class="fx" data-fig="pairing" aria-label="两份 trace 配对并找到 pivot">
<div class="fx-stage">
<div class="fx-grid" style="--cols:5">
<span class="fx-label">参考运行</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-empty" data-hot="3"></span><span class="fx-node" data-hot="4">attn</span><span class="fx-node">mlp</span>
<span></span><span class="fx-pair is-ok" data-at="1">相等</span><span class="fx-pair is-ok" data-at="2">相等</span><span class="fx-pair" data-at="3">空缺，不配对</span><span class="fx-pair is-bad" data-at="4">第一个差异，即 pivot</span><span></span>
<span class="fx-label">suspect 运行</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-ghost" data-hot="3">多出的 cast</span><span class="fx-node to-bad" data-hot="4">attn</span><span class="fx-node">mlp</span>
</div>
</div>
<ol class="fx-say">
<li>两份 trace 按调用顺序排开，每个边界都有一个 fingerprint。</li>
<li>embed 和 embed 配成一对，结果相等。</li>
<li>norm 和 norm 配成一对，结果相等。</li>
<li data-hold="1.5">suspect 运行多了一个 cast。保序匹配不给它配对，把它留作空缺；空缺不算差异。</li>
<li data-hold="2">attn 不相等。第一对不相等的事件就是 pivot，从它出发去找原因。</li>
</ol>
<figcaption>图：两份 trace 按调用顺序排开。匹配把相等的边界配成对，多出的 cast 不配对，然后报告第一对不相等的事件。</figcaption>
</figure>

这张图里有两点很重要。

- **空缺不是差异。** suspect 运行有一个参考运行里没有的事件。保序匹配不会强行给它配对，而是让它保持未配对；只有 fingerprint 不相等的配对才会被报告。
- **pivot 不一定就是原因。** 举个小例子：注意力的反向用原子操作累加 dQ，而它自己没有边界。第一个看到它结果的边界是梯度范数，那已经在整个反向计算之后了。所以 pivot 是这个范数，原因却在上游。真正指向原因的，是调用点上的状态：它告诉你每个参数的梯度是相等还是不同。在最后一个 block 里，只有 q 投影的梯度不同，而这个梯度正是 dQ。

**规则。** 取第一个逐 bit 的差异，并且它之前不能有未对齐的事件。从这里出发，沿着产生它输入的算子往上游追，也往下游追到症状，再用尽量小的干预去检验你怀疑的原因。这是第 0 部分"在算子边界上比较"的具体应用：边界能定位第一个可见的影响，而对上游算子的推理才能定位原因。

## 步骤

### 1. 确认要查的区域已经被采集

**目标：** 你关心的区域已经被采集。

**做法：** `opguard_bridge.coverage(run)` 读取 OpGuard 的发现文件和计划文件，列出哪些边界被发现了、哪些被计划采集。

**成功标准：** 你怀疑的算子，包括其中的 GEMM，都在采集计划里。

### 2. 采集两次运行

**目标：** 两次运行在每个计划好的边界上都有 fingerprint。

**做法：** 调用 [opguard_bridge.py](../../tools/opguard_bridge.py) 里的 `opguard_bridge.run_pair(reference, suspect, ...)`，让两次运行都经过 OpGuard 的 harness：发现边界，规划插桩，采集，然后比较。发现和采集两个阶段，各自会把两次运行都启动一遍。如果生产环境的训练任务用了 `torch.compile` 或 CUDA graph，这次采集会改用 eager 模式运行它，于是可能看不到生产环境实际算出的结果，见[用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md)。

**成功标准：** 两份 trace 都已生成，摘要里统计了相等和不相等的事件各有多少。

### 3. 收集首个分歧点的全部事实

**目标：** 把关于 pivot 能知道的事实全部拿到。

**做法：** `opguard_bridge.first_divergence(export, ...)` 依次调用 harness 的 `trace_summary`、`first_diff`、`inspect_event`、`compare_events` 和 `trace_neighbors`，返回：算子、源码行、调用序号、哪些字段不同、两次运行的输入和输出、不同的区间、pivot 之前未对齐的事件，以及 pivot 所在调用点上每次调用的状态。它本身不做任何判断。

**成功标准：** 拿到一个 `unaligned_before` 为零的 pivot。

### 4. 从 pivot 追到原因，并检验它

**目标：** 找到原因，并拿到支持它的证据。

**做法：** 用 OpGuard harness 里的几个 skill：`investigate-first-diff` 负责检查这个边界并记下证据；输入已经不同时，`trace-upstream` 沿着上游算子往回追；`validate-hypothesis` 每次检验一个原因。`opguard_bridge.suggest(facts)` 会把 pivot 对应到本手册的算子类别和[recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md)，每条线索都附有理由。

| 证据 | 能告诉你什么 | 代价等级 | 什么时候用 |
|---|---|---|---|
| pivot 的源码行和 dtype | 直接看出舍入点是否被挪动，或者多了一次类型转换 | 免费 | pivot 处输出的 dtype 或形状不同 |
| 调用点状态（每次调用一个字母） | 同一个调用点上哪些调用不同，进而知道是哪个参数、哪一步 | 免费 | pivot 要用到许多上游算子的输出（比如对所有梯度求范数） |
| 输入相等，输出却不同 | 这个算子自身的状态或权重更早就变了 | 免费 | 查表类算子，以及其他看起来没有状态的算子 |
| `suggest` 给出的线索 | 算子类别和对应的 recipe，附带理由 | 免费 | 开始读 kernel 代码之前 |
| 重新采集某个区域的完整张量 | 数值本身，可以看出它们差在哪里 | 对这个区域来说大（2 倍或更多） | fingerprint 相等但怀疑是碰撞，或者需要知道差异有多大 |
| 在固定输入上反复运行同一个算子 | 这个算子是不是不确定的 | 小 | 你怀疑原因是某个不确定的 kernel |

**成功标准：** 得到一个假设，而且一次干预就能确认它。

### 5. 修复原因，并确认修复有效

**目标：** 证明是这个修复、而不是别的东西，让两次运行变得一致。

**做法：** 应用修复（改一个设置、改训练 step 的某个部分，或者按 [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md) 改写某个 kernel），然后再调用一次 `run_pair`。

**成功标准：** 每个事件都相等，没有未对齐的事件，最终参数逐 bit 一致。

在一次已知答案的演练里，我们预先植入了三个原因：norm 的舍入点被挪动、注意力的反向跑在 `warn_only` 下、恢复训练时没有恢复优化器的 step 计数器。在一块 GPU、一个 rank 上，OpGuard 找到的每个首个差异之前都没有未对齐的事件；三个 pivot 里有两个并不是原因本身；每次修复后重跑，结果处处相等，最终参数逐 bit 一致（[报告](../../examples/pretraining_e2e/REPORT.md#what-this-shows)）。这是一次已知答案的演示，不是盲测评估。

## 可能出错的地方

!!! warning "`torch.nn.Linear` 把 GEMM 藏了起来"
    GEMM 是在 torch 自己的源文件里启动的，所以调用投影的那一行虽然被发现了，却不会被计划采集。结果没有任何投影被采集，pivot 落到了更后面。在模型代码里改用 `F.linear`，并检查 `coverage`。

!!! warning "模型代码里用了单字母变量名"
    OpGuard 的插桩器根据调用点周围的名字来匹配调用点，并且会忽略单字母的名字。于是注意力和投影虽然列进了采集计划，却从来没有被包装上。改用有意义的名字，并检查 `coverage`。

!!! warning "边界落在了行末的类型转换上"
    像 `q = q_norm(x).float()` 这样的一行，类型转换并不是真正的计算，但 pivot 会指向这次转换。应该把算子本身（`q_norm`）设为边界。

!!! warning "把没配上对的事件当成了差异"
    其中一次运行可能因为融合或改写而多出一些事件。强行给它们配对，会凭空造出并不存在的差异。让没配上对的事件保持为空缺。

!!! warning "把 pivot 当成了原因"
    没有边界的上游计算（反向计算、优化器的 step、恢复训练）会先在下一个用到它输出的算子处显现出来。如果直接修 pivot，就修错了算子。动手修复之前，先往上游追，并检验假设。

## 小结

- 先检查覆盖范围：没被采集的算子不可能成为 pivot。
- pivot 是第一对不相等的事件，而且它之前没有未对齐的事件；空缺不是差异。
- pivot 只是起点。用源码行、调用点状态和输入，从它一路追到原因。
- 修复要靠重新采集来确认：所有事件都相等，没有未对齐的事件，参数逐 bit 一致。

## 深入阅读

- [报告：OpGuard 演练](../../examples/pretraining_e2e/REPORT.md)：实验配置、OpGuard 对每一对运行报告了什么、修复和重跑、每次排查的代价、覆盖范围，以及对 fingerprint 的检查。
- 代码：[pretraining_e2e.py](../../examples/pretraining_e2e/pretraining_e2e.py) 和 [opguard_bridge.py](../../tools/opguard_bridge.py)。
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)。
- [用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md)：训练任务用了 `torch.compile` 或 CUDA graph 时该怎么办。
