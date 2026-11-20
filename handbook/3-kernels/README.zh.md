# 3. Batch-invariant kernel：让一个算子不变

## 我们在哪里

第 1 部分止于首个分歧点，第 2 部分止于一次审计。两者都可能指向同一种情况：某个算子没有
batch-invariant（批不变）实现。它可能是新的注意力变体、MoE combine、量化 GEMM 或集合通信。
这一部分讲的就是处理这一个算子的流程：把它从"有变化或未知"带到"在给定级别上验证为不变"
（见[级别](../0-foundations/README.md)），或者带到一个有文档的回退方案。

<figure class="fig">
<div class="fig-chain"><span class="fig-node">1. 找到变化的来源</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">2. 选最便宜的策略</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">3. 保留舍入点</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">4. 验证矩阵</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">5. 集成</span></div>
<div class="fig-back">第 4 步失败时，回到第 1 步。</div>
</figure>

执行这些步骤的 agent 流程是
[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)。
本页之后的各类页面把它用到六类算子上，从最简单到最难。

## 思路：求和的顺序是结果的一部分

浮点加法每加一次就舍入一次，所以分组方式会改变答案。在 fp32 里，1e8 + 1 会舍入回 1e8，
因为 1e8 之后的下一个数相距 8：

| 分组 | 步骤 | 结果 |
|---|---|---|
| (1e8 + 1) − 1e8 | 1e8 + 1 舍入为 1e8；再算 1e8 − 1e8 | 0 |
| (1e8 − 1e8) + 1 | 0；再算 0 + 1 | 1 |

同样三个数，两个答案。kernel 要为每个输出元素做出这个选择，而且常常依据整个调用的形状来做。
看矩阵乘中的一行。单独计算时批里只有一行，库可能把这一行的长点积拆成四段，放在四个 program 上，
再把四个部分和相加（split-K）。在 64 行的批里工作量足够，不必拆分，于是一遍走完点积。
这一行的数学相同，分组不同，最后几位也就不同。

**规则。** 一次归约的顺序只取决于该元素自身的长度。产生一个输出元素的求和，其顺序、分组、
指令和精度可以依赖这个和有多长，除此之外什么都不能依赖：不能依赖行数、该行的位置、其他行的值、
grid 大小或 GPU 核心数、调度器决定的分页或分块边界，也不能依赖时序。
这是纯函数原则在归约树上的体现：归约树只是该元素自身输入的函数。

**确定性不等于批不变。** 库里的 GEMM 对同一批的每次运行都给出相同的位（L1），
但批变大时某一行的位仍会改变（L2），因为它按批大小挑选 kernel。原子操作破坏 L1；
依赖形状的选择破坏 L2；每种服务模式用不同 kernel 则破坏 L3
（测得：见[报告](../../examples/kernel_cases/REPORT.md#stock-ops-determinism-is-not-batch-invariance)）。

## 步骤

### 1. 定位变化来源

- **目标：** 知道算子内部哪个选择依赖了元素自身长度以外的东西。
- **方法：** 运行 [tools/bi_check.py](../../tools/bi_check.py) 中的 `bi_check.check_batch`
  （类注意力算子再加 `check_modes`）；阅读 kernel 的启动和分派代码。对整个模型，
  [tools/bi_audit.py](../../tools/bi_audit.py) 会指出第一个有变化的模块。
- **成功标准：** 一个能用单一机制解释的失败模式。

| 变化藏在哪里 | 例子 |
|---|---|
| 按行数选择的配置 | tile 大小、split-K、小 M 快速路径 |
| 按可用工作量拆分的归约 | split-KV decode、少行时一行被拆到多个 block |
| 按到达顺序累加 | 浮点原子操作、scatter-add |
| 对整个批的统计量 | 动态量化 scale、MoE 容量上限 |
| 由布局决定的分组 | KV 分页、打包的序列、消息分块 |
| 同一数学的两种形式 | 吸收式和显式 MLA、分块和递归线性注意力 |
| kernel 之上的层 | 编译器融合、每个填充大小一张 CUDA graph |

[验证参考](../../skills/batch-invariant-kernel/references/verification.md#reading-failures)
把失败模式对应到机制。

### 2. 选择成立的最便宜策略

- **目标：** 用最小的改动和代价去掉这个依赖。
- **方法：** 按下表顺序尝试；
  [skill](../../skills/batch-invariant-kernel/SKILL.md#2-choose-the-cheapest-strategy-that-holds)
  逐项说明。
- **成功标准：** 得到一个候选方案，再在第 4 步验证。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 配置 | 用库的开关固定顺序：单一 split、关闭 split-K、固定算法 | 视算子从免费到大 | 库提供了这个开关 |
| 选用 | 现成的批不变算子 | 小到中等 | 你的算子有现成实现；在你的形状上重新验证 |
| 固定一个配置 | 所有批大小用同一套 tile 设置 | 小到中等；小批时利用率较低 | 依赖来自按形状的选择 |
| 修改 kernel | 写明的固定归约树 | 中等 | 没有开关；按该类的 recipe 来 |
| 处处同一形式 | 每种模式、两侧都用同一算法 | 中等到大 | 存在两种代数相等的形式 |
| 回退 | 替换 kernel，或保留一个逐行的慢速 oracle 用于调试 | 大 | kernel 闭源；把它排除在一致性声明之外 |

### 3. 保持舍入点

- **目标：** 不变的 kernel 在与它要对齐的一侧相同的位置舍入：相同的 cast、融合、累加器 dtype
  和 epilogue 顺序。
- **方法：** 动手前写下参考的舍入序列；见
  [舍入点](../../skills/engine-parity/references/rounding-points.md)。
- **成功标准：** 在旧 kernel 尚未出现变化的批上，新 kernel 与旧 kernel 逐位相等；
  或者两侧一起改变。

一个批不变但舍入位置与另一引擎差一处的 kernel，能通过 L2，仍然通不过 L4
（[整个模型](../2-rl/whole-model.md)）。

### 4. 验证矩阵

- **目标：** 拿到证据，而不是从代码推理。
- **方法：** [验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)：
  每个阈值附近的批大小、行位置、同批内容（随机、零、大值）、模式、布局、重复运行、反向、
  并行度、编译和 graph 部署、跨引擎。
- **成功标准：** 零不一致，通过整数视图逐位比较（`-0.0` 不等于 `+0.0`），没有容差。

### 5. 集成

- **目标：** 真正运行的是验证过的 kernel，并且代价已知。
- **方法：** 把它注册为不透明的 custom op，让编译器无法重新融合；用开关控制；
  在 prefill 和 decode 批大小上测吞吐，与基线并列；写出
  [skill](../../skills/batch-invariant-kernel/SKILL.md#6-report) 要求的报告条目。
- **成功标准：** 部署路径通过同一矩阵，代价有记录。

## 各类算子，从简单到难

每页都从前一页接着讲。每类的详细 recipe（含 kernel 参数）在
[算子 recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md) 中。

| 页面 | 直观上什么在变 | 怎么修 | 级别 |
|---|---|---|---|
| [GEMM 与量化 GEMM](gemm-and-quantized.md) | 每个元素一个点积；库按批选择拆分；批级 scale 把各行耦合起来 | 所有批大小用同一配置；per-token 或静态 scale | L2 |
| [Softmax 注意力](attention.md) | 对 key 的第二次归约，在 prefill、decode、分页和打包下走法不同 | 从绝对位置 0 起的固定 key block，每种模式同一 kernel；无原子操作的反向 | L2, L3 |
| [MLA 与稀疏注意力](mla-and-sparse.md) | 两种相等形式在不同位置舍入；分数的微小变化就让选择翻转 | 处处同一形式；不变的分数和稳定的 top-k | L2, L3 |
| [线性注意力](linear-attention.md) | 携带的状态；分块边界落在哪里 | 处处递归，或在绝对边界上分块 | L3, L4 |
| [MoE](moe.md) | 专家看到哪些行；combine 的顺序；丢弃 | 不变的专家 GEMM、从固定槽位按固定顺序的逐 token 求和、无丢弃路由 | L2, L5 |
| [集合通信](collectives.md) | 库决定跨 GPU 求和的顺序；并行度改变部分和 | 固定的 tree 或固定顺序的回退；逻辑 TP 大小 | L2, L5 |

其他部分讲到的类：逐行归约
（[recipe 1](../../skills/batch-invariant-kernel/references/operator-recipes.md#1-row-wise-reductions-rmsnorm-layernorm-softmax-log_softmax-row-sums-cross-entropy)）、
[LoRA](../2-rl/lora.md)、[采样](../2-rl/sampling.md)、
[训练步](../1-pretraining/training-step.md)和
[数据并行梯度](../1-pretraining/data-parallel.md)。组装成一个模型的 kernel 见
[整个模型](../2-rl/whole-model.md)和[并行](../2-rl/parallelism.md)。

## 可能出错的地方

!!! warning "确定性模式不等于批不变"
    确定性设置去掉了原子操作和竞争，只给出 L1。行仍会随批改变，所以要单独验证 L2。

!!! warning "两项求和会掩盖顺序错误"
    a + b 与 b + a 完全相等，所以两项的测试对任何顺序都能通过。要用三项或更多来测：
    top-k 至少 3，三个或更多 rank。

!!! warning "均匀的填充行会掩盖耦合"
    零行或复制的同批行不会改变批级统计量。还要用随机和大值的填充行。

!!! warning "自洽的布局仍可能是错的"
    block 跟随页大小或打包偏移的 kernel，在任一设置下自洽，换设置就变。测试时要改变布局。

!!! warning "不变性是对你测过的东西的声明"
    通过只覆盖你运行过的大小、GPU 和版本。把它们列出来，并包含代码中每个阈值附近的大小。

## 小结

- 归约的顺序只能取决于元素自身的长度。
- 定位依赖，选最便宜的去除方法，保持舍入点，在完整矩阵上逐位验证，然后集成。
- L1 不蕴含 L2，L2 和 L3 不蕴含 L4。
- 代价是结果的一部分：与基线并列记录。

## 深入阅读

- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)，
  以及[算子 recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md)、
  其[证据表](../../skills/batch-invariant-kernel/references/operator-recipes.md#evidence-behind-each-recipe)
  和[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)。
- 工具：[tools/bi_check.py](../../tools/bi_check.py)（检查器）、
  [tools/bi_kernels.py](../../tools/bi_kernels.py)（参考 kernel）、
  [tools/bi_audit.py](../../tools/bi_audit.py)（逐模块判定）。
- 报告：[kernel 案例](../../examples/kernel_cases/REPORT.md)，含测得的原生算子、注意力前向、
  FP8 scale 以及普通线性注意力的对比。
