# 3. Batch-invariant kernel：让一个算子的结果不随 batch 变化

## 我们在哪里

第 1 部分最后找到两次运行的首个分歧点，第 2 部分最后对整个模型做一次审计。两条路都可能把问题
落到同一个算子上：它还没有 batch-invariant 的实现，也就是说，同一行的结果会随它所在的 batch 变化。
这个算子可能是一种新的注意力变体、MoE 的 combine、量化 GEMM，也可能是一次集合通信。这一部分讲
怎样处理这样一个算子。起点是一个结果会随 batch 变化、或者还没人检查过的算子；终点是它在一个明确的
[级别](../0-foundations/README.md)上验证为不变，或者有一个写清楚的回退方案。

<figure class="fx" data-fig="kernel-workflow" aria-label="让一个 kernel 成为 batch-invariant 的五个步骤" markdown>
<div class="fx-stage" markdown>
<div class="fx-seq" style="--n:5" markdown="span"><span class="fx-step" data-hot="0 5">[1. 找出什么在变](#1-locate-the-variance)</span><span class="fx-step" data-hot="1 5">[2. 选最便宜的修法](#2-choose-the-cheapest-strategy-that-holds)</span><span class="fx-step" data-hot="2 5">[3. 在同样的地方舍入](#3-keep-the-rounding-points)</span><span class="fx-step" data-hot="3 6">[4. 逐 bit 验证](#4-verify-the-matrix)<span class="fx-swap"><span class="fx-note fx-fail" data-at="3" data-to="5">某个用例失败</span><span class="fx-note fx-pass" data-at="6">没有任何不一致</span></span></span><span class="fx-step" data-hot="7">[5. 投入使用](#5-integrate)</span><span class="fx-loop" style="grid-column: 1 / 5; --k: 4" data-hot="4">第 4 步失败时，回到第 1 步。</span></div>
</div>
<ol class="fx-say">
<li>1. 找出算子内部哪个选择依赖了元素自身求和长度以外的东西。</li>
<li>2. 选一个能消除这种依赖、代价又最小的策略。</li>
<li>3. 让 kernel 和它要对齐的那一端在同样的地方做舍入。</li>
<li>4. 在完整的矩阵上逐 bit 验证，覆盖不同的 batch size、位置、模式、布局和并行度。</li>
<li data-hold="1.5">某个用例失败时，带着这个失败的模式回到第 1 步。</li>
<li data-hold="1.5">重新找原因、重新选策略，舍入点保持不变。</li>
<li>4. 再次验证，直到没有任何不一致。</li>
<li data-hold="2">5. 让验证过的 kernel 成为真正运行的那一个，并记下它的代价。</li>
</ol>
<figcaption>图：处理一个算子的五个步骤。第 4 步要的是证据，而不是读代码得出的推断；任何一处失败都要回到第 1 步。</figcaption>
</figure>

执行这些步骤的 agent 流程是
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)。
本页之后的几页把它用到六类算子上，从最简单的讲到最难的。

## 思路：加法的顺序也是结果的一部分

浮点加法每做一次就舍入一次，所以先把哪几项加在一起，会影响最后的答案。在 fp32 里，1e8 + 1
会舍入回 1e8，因为比 1e8 大的下一个 fp32 数要大 8：

<figure class="fx" data-fig="two-groupings" aria-label="同样三个数按两种方式分组相加">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">(1e8 + 1) − 1e8</span><div class="fx-chain is-tight is-grid"><span class="fx-swap" data-at="1"><span class="fx-node is-mono" data-at="1" data-to="2">1e8 + 1 = 100000001</span><span class="fx-node is-mono is-warn" data-at="2">1e8 + 1 → 1e8<span class="fx-note">被舍入了，因为下一个数相距 8</span></span></span><span class="fx-arrow" data-at="3" aria-hidden="true"></span><span class="fx-node is-mono" data-at="3">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="4" aria-hidden="true"></span><span class="fx-node is-mono is-key" data-at="4">0</span></div></div>
<div class="fx-row"><span class="fx-label fx-mono">(1e8 − 1e8) + 1</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="3" aria-hidden="true"></span><span class="fx-node is-mono" data-at="3">0 + 1 = 1</span><span class="fx-arrow" data-at="4" aria-hidden="true"></span><span class="fx-node is-mono is-key" data-at="4">1</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>取同样三个 fp32 数 1e8、1 和 −1e8，按两种方式分组。</li>
<li>两种分组都先把前两个数相加。</li>
<li data-hold="1.5">1e8 + 1 舍入回 1e8，因为 1e8 之后的下一个数相距 8。</li>
<li>然后各自加上剩下的那个数。</li>
<li data-hold="2">同样三个数，得出两个不同的答案：0 和 1。</li>
</ol>
<figcaption>图：浮点加法每加一次就舍入一次，所以分组方式会改变答案。</figcaption>
</figure>

kernel 要为每个输出元素选定这样一种分组，而且往往是看整个调用的形状来选。以矩阵乘法里的一行为例。
单独计算这一行时，batch 里只有它一行，库可能把这一行很长的点积拆成四段，交给四个 program 分别计算，
最后把四个部分和加起来（这就是 split-K）。如果 batch 里有 64 行，不拆分也有足够的工作让 GPU 忙起来，
库就一遍算完这个点积。两种情况下这一行的数学完全相同，分组却不同，所以结果的最后几个 bit 不一样。

**规则。** 归约的顺序只能取决于这个元素自身求和的长度。对于产生某个输出元素的那次求和，它的顺序、
分组、所用的指令和精度可以取决于要加多少项，但不能取决于别的东西：不能取决于 batch 里有多少行、
这一行排在第几、其他行的值、grid 大小或 GPU 核心数、调度器划定的 page 边界或分块边界，也不能取决于时序。
这其实就是纯函数的原则用在归约树上：归约树只由这个元素自己的输入决定。

**确定性不等于 batch-invariant。** 库里的 GEMM 每次运行同一个 batch，结果都逐 bit 一致，所以它是
确定的（L1）。但 batch 变大时，同一行的结果仍可能改变，因为它是按 batch size 挑选 kernel 的，所以它
不是 batch-invariant 的（L2）。每一级都有典型的破坏原因：浮点原子操作破坏 L1，按形状做出的选择破坏
L2，每种推理模式各用一个 kernel 则破坏 L3（测得：见[报告](../../examples/kernel_cases/REPORT.md#stock-ops-determinism-is-not-batch-invariance)）。

## 步骤

### 1. 找出结果随什么而变

- **目标：** 找到算子内部那个依赖了元素自身求和长度以外因素的选择。
- **做法：** 运行 [tools/bi_check.py](../../tools/bi_check.py) 里的 `bi_check.check_batch`；
  如果算子的行为类似注意力，再运行 `check_modes`。然后阅读启动 kernel、挑选配置的那部分代码。
  如果是从整个模型出发，[tools/bi_audit.py](../../tools/bi_audit.py) 会指出第一个输出随 batch 变化的模块。
- **成功标准：** 拿到一个失败的模式，并且能用一个机制解释它。

| 变化藏在哪里 | 例子 |
|---|---|
| 库按行数挑选配置 | tile 大小、split-K、针对小 M 的快速路径 |
| 工作量少时，kernel 把一次归约拆开 | split-KV decode、把一行拆到多个 block 上 |
| 按数值到达的先后累加 | 浮点原子操作、scatter-add |
| 在整个 batch 上计算统计量 | 动态量化 scale、MoE 的 capacity 上限 |
| 内存布局决定了分组 | paged KV、打包的序列、消息分块 |
| 同一套数学有两种写法 | 吸收式与显式 MLA，分块与递归两种线性注意力 |
| kernel 之上的某一层改变了实际运行的东西 | 编译器融合、每个 padding 尺寸各录一张 CUDA graph |

[验证参考](../../skills/batch-invariant-kernel/references/verification.md#reading-failures)
列出了常见的失败模式，以及每种模式背后的机制。

### 2. 选一个代价最小、又能消除依赖的修法

- **目标：** 用最小的改动、最低的代价去掉这个依赖。
- **做法：** 按下表的顺序逐个尝试，每个方案的细节见
  [skill](../../skills/batch-invariant-kernel/SKILL.md#2-choose-the-cheapest-strategy-that-holds)。
- **成功标准：** 得到一个候选 kernel，交给第 4 步验证。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 配置库 | 用一个设置固定顺序，比如只用一个 split、关掉 split-K 或固定算法 | 免费到大，视算子而定 | 库提供了这样的设置 |
| 用现成的 kernel | 别人已经写好的 batch-invariant 算子 | 小到中等 | 你的算子已有现成实现；要在你自己的形状上重新验证 |
| 固定一套配置 | 所有 batch size 都用同一套 tile 设置 | 小到中等；batch 小时 GPU 利用率偏低 | 依赖来自按形状做出的选择 |
| 修改 kernel | 自己写下并固定一棵归约树 | 中等 | 没有可用的设置；按这类算子的 recipe 来做 |
| 处处用同一种形式 | 每种模式、训推两端都用同一个算法 | 中等到大 | 这段数学有两种代数上相等的写法 |
| 回退 | 换掉这个 kernel，或保留一个逐行计算的慢速 oracle（作为调试用的参考实现） | 大 | kernel 闭源；不要把它算进任何一致性声明 |

### 3. 和要对齐的那一端在同样的地方舍入

- **目标：** 让不变的 kernel 和它要对齐的那一端在相同的位置舍入，也就是 cast 相同、融合方式相同、
  累加器的 dtype 相同，epilogue 里各步的顺序也相同。
- **做法：** 动手修改之前，先按顺序写下参考实现在哪些地方舍入；见
  [舍入点](../../skills/engine-parity/references/rounding-points.md)。
- **成功标准：** 在旧 kernel 还没有出现变化的 batch 上，新 kernel 与旧 kernel 逐 bit 一致；否则就让
  两端一起改。

一个 kernel 可以是 batch-invariant 的，却仍有一处舍入和另一个引擎不同。这样它能通过 L2，却通不过 L4
（见[整个模型](../2-rl/whole-model.md)）。

### 4. 在完整的测试矩阵上逐 bit 验证

- **目标：** 拿到 kernel 确实不变的证据，而不是靠读代码推断。
- **做法：** 运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)。
  它会改变每个阈值附近的 batch size、行的位置、同一 batch 里其他行的内容（随机值、零、大值）、模式、
  布局，还包括重复运行、反向、并行度、torch.compile 和 CUDA graph 下的部署，以及不同的引擎。
- **成功标准：** 没有任何不一致。比较时通过整数视图逐 bit 比较，`-0.0` 和 `+0.0` 算作不同，不设容差。

### 5. 让验证过的 kernel 真正跑在代码路径上

- **目标：** 确认真正运行的就是验证过的 kernel，并且知道它的代价。
- **做法：** 把它注册为不透明的自定义算子，让编译器无法再把它和相邻的算子融合，并用一个开关控制
  是否启用。在 prefill 和 decode 常见的 batch size 上测吞吐，与基线对比。最后按
  [skill](../../skills/batch-invariant-kernel/SKILL.md#6-report) 的要求写报告条目。
- **成功标准：** 实际部署的代码路径通过同一个矩阵，代价也记录在案。

## 各类算子，从最简单到最难

后面每一页都建立在前一页的基础上。每类算子的详细 recipe（包括 kernel 参数）见
[算子 recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md)。

| 页面 | 直观地说，什么在变 | 怎么修 | 级别 |
|---|---|---|---|
| [GEMM 与量化 GEMM](gemm-and-quantized.md) | 每个元素是一个点积。库根据 batch size 决定怎样拆分它，而在整个 batch 上算出的 scale 又把各行绑在一起。 | 所有 batch size 都用同一套配置；scale 用 per-token 的或静态的。 | L2 |
| [Softmax 注意力](attention.md) | 在 key 上还有第二次归约，prefill、decode、paged KV 和打包各自的走法不同。 | 从绝对位置 0 起按固定的 block 遍历 key，每种模式用同一个 kernel；反向不用原子操作。 | L2, L3 |
| [MLA 与稀疏注意力](mla-and-sparse.md) | 两种代数上相等的形式在不同的地方舍入；分数的一点微小变化，就可能改变选中哪些 key。 | 处处用同一种形式；让分数不变，并使用稳定的 top-k。 | L2, L3 |
| [线性注意力](linear-attention.md) | 状态从一个 token 带到下一个 token，结果取决于分块边界落在哪里。 | 处处用递归形式，或者把分块边界放在绝对位置上。 | L3, L4 |
| [MoE](moe.md) | 每个 expert 看到哪些行、combine 按什么顺序把各 expert 的输出加起来、有没有 token 被丢弃，都会影响结果。 | 让 expert 的 GEMM 不变，每个 token 从固定的槽位按固定顺序求和，路由时不丢弃 token。 | L2, L5 |
| [集合通信](collectives.md) | 库决定跨 GPU 求和的顺序，而并行度又会改变部分和。 | 固定归约树，或回退到固定顺序；按固定的逻辑 TP 规模计算。 | L2, L5 |

其余几类算子在手册的其他部分讲：逐行归约
（[recipe 1](../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy)）、
[LoRA](../2-rl/lora.md)、[采样](../2-rl/sampling.md)、
[训练 step](../1-pretraining/training-step.md)和
[数据并行梯度](../1-pretraining/data-parallel.md)。这些 kernel 怎样组合成一个模型，见
[整个模型](../2-rl/whole-model.md)和[并行](../2-rl/parallelism.md)。

## 可能出错的地方

!!! warning "确定性模式并不能让 kernel 成为 batch-invariant"
    确定性设置去掉了原子操作和竞争，但只能保证 L1。同一行的结果仍会随 batch 变化，所以要单独验证 L2。

!!! warning "只有两项的求和会掩盖顺序错误"
    a + b 与 b + a 完全相等，所以只有两项的测试不管什么顺序都能通过。要用三项或更多来测，
    比如 top-k 至少取 3，或者用三个及以上的 rank。

!!! warning "同 batch 其他行的内容太单一，会掩盖行与行之间的耦合"
    如果同一 batch 的其他行全是零，或者是复制出来的，在整个 batch 上算的统计量就不会变。其他行
    还要用随机值和大值来测。

!!! warning "在一种布局下自洽，换一种布局仍可能出错"
    如果 kernel 的 block 跟着页大小或打包序列的偏移走，那么在任何一种设置下它都自洽，
    设置一变，结果就跟着变。测试时要改变布局。

!!! warning "不变性只对你测过的情况成立"
    一次通过只覆盖你跑过的大小、GPU 和软件版本。把它们列出来，并在代码里找到的每个阈值上下
    都取一些大小来测。

## 小结

- 归约的顺序只能取决于元素自身求和的长度。
- 先找出依赖，选代价最小的办法去掉它，和要对齐的那一端在同样的地方舍入，在完整矩阵上逐 bit 验证，
  最后才投入使用。
- L1 不意味着 L2；L2 和 L3 加在一起也不意味着 L4。
- 代价也是结果的一部分，要和基线放在一起记录。

## 深入阅读

- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)，
  以及[算子 recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md)、
  它们的[证据表](../../skills/batch-invariant-kernel/references/operator-recipes.md#evidence-behind-each-recipe)
  和[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)。
- 工具：[tools/bi_check.py](../../tools/bi_check.py)（检查器）、
  [tools/bi_kernels.py](../../tools/bi_kernels.py)（参考 kernel）和
  [tools/bi_audit.py](../../tools/bi_audit.py)（给每个模块一个判定）。
- 报告：[kernel 案例](../../examples/kernel_cases/REPORT.md)，其中测量了原生算子、注意力前向、
  FP8 scale，以及普通线性注意力的分块和递归两种形式。
