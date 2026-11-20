# 用 OpGuard 定位

## 我们在哪里

本页是[预训练工作流](README.md)的核心循环：第 3、4、6、7 步，即采集、首个分歧点、修复和确认。你有一次参考运行和一次 suspect 运行，它们只在声明过的那一点上不同（[预检](determinism-and-preflight.md)），读到的 batch 也相同（[数据对齐](data-alignment.md)）。还没解决的是：两次运行在哪里第一次分开，以及为什么。分类（第 5 步）在[工作流页面](README.md#step-5-classify)；本页之后的几页讲首个分歧点最常指向的原因：[训练 step](training-step.md)、[梯度集合通信](data-parallel.md)、[编译代码](capture-compiled-code.md)和[硬件](sdc.md)。

## 思路：首个差异是起点，不是原因

OpGuard 在两次运行的每个模型级边界上记录 fingerprint，把两份 trace 配对，并报告第一对不同的事件：**pivot**。

<figure class="fx" data-fig="pairing" aria-label="两条 trace 配对并找到 pivot">
<div class="fx-stage">
<div class="fx-grid" style="--cols:5">
<span class="fx-label">参照</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-empty" data-hot="3"></span><span class="fx-node" data-hot="4">attn</span><span class="fx-node">mlp</span>
<span></span><span class="fx-pair is-ok" data-at="1">相等</span><span class="fx-pair is-ok" data-at="2">相等</span><span class="fx-pair" data-at="3">空缺：不配对</span><span class="fx-pair is-bad" data-at="4">第一个差别：pivot</span><span></span>
<span class="fx-label">嫌疑</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-ghost" data-hot="3">多出的 cast</span><span class="fx-node to-bad" data-hot="4">attn</span><span class="fx-node">mlp</span>
</div>
</div>
<ol class="fx-say">
<li>两条 trace 按调用顺序排列，每个边界一个 fingerprint。</li>
<li>embed 和 embed 配对：相等。</li>
<li>norm 和 norm 配对：相等。</li>
<li data-hold="1.5">嫌疑运行多了一个 cast。保序匹配让它不配对：这是空缺，不是差别。</li>
<li data-hold="2">attn 不同：第一个不同的对就是 pivot。从它走到原因。</li>
</ol>
<figcaption>图：两条 trace 按顺序排列。匹配把相同的边界配成对，多出的 cast 不配对，并报告第一个不同的对。</figcaption>
</figure>

这张图里有两点要紧。

- **空缺不是差异。** suspect 有一个参考没有的事件。保序匹配让它保持未配对，而不是强行配对；遍历只报告 fingerprint 不同的配对。
- **pivot 不总是原因。** 一个小例子：注意力的 backward 用原子操作累加 dQ，而它自己没有边界。第一个看到其结果的边界是梯度范数，在整个 backward 之后。pivot 是范数；原因在上游。指出原因的是调用点状态：每个参数的梯度，相等还是不同。在最后一个 block 中只有 q 投影的梯度不同，那就是 dQ。

**规则。** 取第一个逐位差异，且它之前没有未对齐的事件；然后沿生产者向上游走，向下游走到症状，并用最小的干预检验原因。这是第 0 部分"在算子边界上比较"的一个实例：边界定位第一个可见的影响，对生产者的推理定位原因。

## 步骤

### 1. 检查覆盖

**目标：** 所关心的区域已被采集。

**做法：** `opguard_bridge.coverage(run)` 读取 OpGuard 的发现和计划文件，列出发现了什么、计划了什么。

**成功：** 你怀疑的算子，包括 GEMM，都在计划中。

### 2. 采集这一对

**目标：** 两次运行在每个计划边界上的 fingerprint。

**做法：** [opguard_bridge.py](../../tools/opguard_bridge.py) 中的 `opguard_bridge.run_pair(reference, suspect, ...)` 让两边都走 OpGuard 的 harness：发现、插桩计划、采集、比较。发现和采集各让每一边启动一次。如果生产作业运行 `torch.compile` 或 CUDA graph，这次采集会在 eager 下运行它，可能看不到生产实际算出的比特：见[作业运行编译代码时](capture-compiled-code.md)。

**成功：** 两份 trace 都存在，摘要统计了相等和不同的事件。

### 3. 取首个分歧点

**目标：** 关于 pivot 的事实。

**做法：** `opguard_bridge.first_divergence(export, ...)` 调用 harness 的 `trace_summary`、`first_diff`、`inspect_event`、`compare_events` 和 `trace_neighbors`，返回：算子、行、调用序号、哪些字段不同、两边的输入和输出、不同的区间、它之前未对齐的事件，以及其调用点上每次调用的状态。它不做任何判断。

**成功：** 一个 `unaligned_before` 为零的 pivot。

### 4. 向外走并检验

**目标：** 原因，以及证据。

**做法：** 用 OpGuard 的 harness skill：`investigate-first-diff` 负责边界和证据记录，`trace-upstream` 在输入已经不同时追踪生产者，`validate-hypothesis` 检验一个原因。`opguard_bridge.suggest(facts)` 把 pivot 对应到本手册的算子类别和[配方](../../skills/batch-invariant-kernel/references/operator-recipes.md)，作为带理由的线索。

| 证据 | 给出什么 | 代价等级 | 何时使用 |
|---|---|---|---|
| pivot 的源码行和数据类型 | 直接看到被移动的舍入点或类型转换 | 免费 | pivot 处输出的 dtype 或形状不同 |
| 调用点状态（每次调用一个字母） | 同一调用点的哪些调用不同：哪个参数、哪一步 | 免费 | pivot 是许多生产者的消费者（对梯度求范数） |
| 输入相等，输出不同 | 算子自身的状态或权重在更早时就变了 | 免费 | 查表以及其他看似无状态的算子 |
| `suggest` 线索 | 算子类别和配方，带理由 | 免费 | 读 kernel 之前 |
| 重新采集某一区域的完整张量 | 值本身，看它们怎样不同 | 对该区域大（2 倍或更多） | fingerprint 相等但怀疑碰撞，或需要数值大小 |
| 在固定输入上重复检查一个算子 | 这个算子是否非确定 | 小 | 假设是某个非确定 kernel |

**成功：** 一个由单次干预确认的假设。

### 5. 修复并确认

**目标：** 证明是修复、而不是别的，让两次运行一致。

**做法：** 应用修复（一个设置、训练 step 的某部分、经 [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md) 处理的 kernel），然后再次 `run_pair`。

**成功：** 每个事件相等，没有未对齐的，最终参数逐位相同。

在一个植入三种已知原因的演练中（norm 舍入点被移动、`warn_only` 下的注意力 backward、恢复训练时缺少优化器的 step 计数器），在一块 GPU、一个 rank 上，OpGuard 定位的每个首个差异之前都没有未对齐的事件；三个 pivot 中有两个不是原因；每次修复后的重跑处处相等，最终参数相同（[报告](../../examples/pretraining_e2e/REPORT.md#what-this-shows)）。这是已知答案的演示，不是盲测评估。

## 可能出错的地方

!!! warning "`torch.nn.Linear` 藏起了 GEMM"
    GEMM 是从 torch 自己的文件里启动的，所以调用投影的那一行被发现了却没有被计划。没有投影被采集，pivot 落在更后面。在模型代码中调用 `F.linear`，并检查 `coverage`。

!!! warning "模型代码中的单字母变量名"
    OpGuard 的插桩器按周围的名字匹配调用点，并忽略单字母名字。注意力和投影被计划了却没有被包装。使用有描述性的名字，并检查 `coverage`。

!!! warning "边界落在末尾的类型转换上"
    像 `q = q_norm(x).float()` 这样的行，类型转换不是计算本身。pivot 会指向这个转换。选算子（`q_norm`）作为边界。

!!! warning "把未配对的事件读成差异"
    一边可能多出事件（融合、重写）。强行配对会凭空制造差异。让未配对的事件保持为空缺。

!!! warning "把 pivot 读成原因"
    没有边界的生产者（backward、优化器 step、恢复训练）会先出现在它的下一个消费者处。你会修错算子。修复之前先向上游走并检验。

## 小结

- 先检查覆盖：没被采集的算子不可能成为 pivot。
- pivot 是第一对不同的事件，且它之前没有未对齐的事件；空缺不是差异。
- pivot 是起点：用源码行、调用点状态和输入走到原因。
- 修复要通过重新采集来确认：全部相等，没有未对齐的，参数逐位相同。

## 深入阅读

- [报告：OpGuard 演练](../../examples/pretraining_e2e/REPORT.md)：配置、OpGuard 对每一对报告了什么、修复和重跑、每次排查的代价、覆盖、fingerprint 检查。
- 代码：[pretraining_e2e.py](../../examples/pretraining_e2e/pretraining_e2e.py) 和 [opguard_bridge.py](../../tools/opguard_bridge.py)。
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)。
- [作业运行编译代码时](capture-compiled-code.md)：`torch.compile` 或 CUDA graph 下的运行。
