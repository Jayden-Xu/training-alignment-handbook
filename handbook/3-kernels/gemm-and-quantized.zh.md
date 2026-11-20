# GEMM 与量化 GEMM

## 我们在哪里

[流程](README.md)分五步：找出什么在变，选最便宜的修法，和要对齐的那一端在同样的地方舍入，验证，
然后投入使用。GEMM 是练习这套流程最简单的算子：每个输出元素都是一个长度固定为 K 的点积，算子里
没有别的归约。后面每一页都要用到不变的 GEMM，因为注意力的投影、MLA 每个头上的乘积、MoE 的 expert
和张量并行层里都有 GEMM。

## 思路：库看 batch 的大小来决定怎样拆分求和

输出元素 C[m, n] 是 K 个乘积之和。GEMM 库把输出切成若干 tile，并在每次调用时决定沿 K 怎么走。
它根据问题的形状做这个决定，而形状里就包括 M，也就是 batch 里的行数。

<figure class="fx" data-fig="gemm-split" aria-label="矩阵乘法的第 0 行，分别放在 1 行和 64 行的 batch 里">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">batch 只有 1 行，工作量小<span class="fx-batch fx-mini"><span class="fx-rowbar is-mine"></span></span></span><div class="fx-tree" style="--n:4">
<div class="fx-level"><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span></div>
<div class="fx-level" data-at="1"><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 0</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 1</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 2</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 3</span></span></div>
<div class="fx-level" data-at="2" data-hot="4"><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-key">第 0 行：值 A</span></span></div>
</div></div>
<div class="fx-row"><span class="fx-label">batch 有 64 行，工作量足够<span class="fx-batch fx-mini"><span class="fx-rowbar is-mine"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span></span></span><div class="fx-tree" style="--n:4">
<div class="fx-level fx-sweep" data-hot="3"><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span></div>
<div class="fx-level" data-at="3"><span class="fx-walk" style="--w:4">一个 program 按顺序走完整个 K</span></div>
<div class="fx-level" data-at="4"><span class="fx-j" style="--w:4;--k:1"><span class="fx-node is-bad">第 0 行：值 B ≠ A</span></span></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>第 0 行（高亮）在两个 batch 里都是沿 K 的同一个点积。</li>
<li data-hold="1.5">batch 只有 1 行时工作量小，所以库把 K 切成 4 段，交给 4 个 program 计算。</li>
<li data-hold="1.5">再把 4 个部分和加起来，第 0 行得到值 A。</li>
<li data-hold="2">batch 有 64 行时，不切分工作量也足够，于是一个 program 按顺序走完整个 K。</li>
<li data-hold="2">这时第 0 行得到值 B ≠ A：数学相同，分组不同。</li>
</ol>
<figcaption>图：库根据行数决定怎样切分 K，所以同一行放进另一个 batch，结果就不再逐 bit 一致。</figcaption>
</figure>

batch 小时，库可能为了让 GPU 忙起来而拆分 K（split-K 或 stream-K），也可能调用专为小 M 写的 kernel；
batch 大时，它一遍走完 K。这些选择每一个都是确定的，所以同一个 batch 每次运行，结果都逐 bit 一致（L1）。
但同一行放进另一个 batch，求和的分组就变了，于是 L2 不成立。

**GEMM 的规则。** 所有 M 都用同一个 tile 配置，不用 split-K 或 stream-K，沿 K 按升序累加进
fp32 累加器，epilogue 也保持不变。代价是 M 小时利用率偏低。

**量化 GEMM 还多一种行与行之间的耦合：scale。** FP8 激活存成一个小数乘以一个 scale，而 scale
是根据最大值选出来的。如果这个最大值取自整个 batch，一行的 scale 就取决于和它同 batch 的其他行：

| 行 x = [0.5, 3] | scale 从哪里来 | scale | x 会怎样 |
|---|---|---|---|
| 单独一行 | 它自己的最大值 3 | 细 | 在它自己的数值范围内以全精度存储 |
| 旁边有一行含 100 | 整个 batch 的最大值 100 | 粗 | 在更粗的网格上舍入，结果不再逐 bit 一致 |

per-token 或 per-block 的 scale 只用这一行自己的值，耦合也就消失了。真正的 FP8 GEMM 还要在
kernel 内部固定两件事：部分和每隔多久提升一次精度，以及按什么顺序乘上各个 scale。这两件事
都是归约的一部分。

## 步骤

### 1. 找出结果随 batch 变化的 GEMM 形状

- **目标：** 知道模型里哪些 GEMM 形状在 M 变化时结果会变。
- **做法：** 对模型实际运行的每个形状 (K, N)，调用 [tools/bi_check.py](../../tools/bi_check.py)
  里的 `bi_check.check_batch`。batch size 取在 2 的幂和 tile 大小的倍数附近，同一 batch 的其他行
  分别填随机值、零和大值。
- **成功标准：** 得到一份会失败的形状和大小的清单。测得：库里的 GEMM 在测过的每个形状上都失败，
  往往 batch 里有 2 行时就开始失败（[报告](../../examples/moe_combine/REPORT.md#1-locate-the-variance)）。

### 2. 所有 batch size 都用同一套配置

- **目标：** 无论 M 是多少，每个元素都由同一棵归约树求和。
- **做法：** 使用 [tools/bi_kernels.py](../../tools/bi_kernels.py) 里的单配置 GEMM
  `mm_one_config`，或者按
  [recipe 2](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16)
  自己写 kernel。如果多个头或多个 expert 都需要 GEMM，就在循环里调用同一个 kernel 体，或者给它
  传一个头索引（[recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)）。
- **成功标准：** `check_batch` 在每个大小上都通过。这在一种 GPU、三个形状上得到了证明
  （[报告](../../examples/moe_combine/REPORT.md#1-locate-the-variance)）。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 原样使用库里的 GEMM | 只有 L1 | 免费 | 需要一致性时永远不选 |
| 单配置 kernel | 所有 M 上都满足 L2 | 在测过的形状上最多中等，小形状上有时反而更快 | 默认的修法 |
| 固定算法的库 GEMM | 如果固定的算法也固定了拆分，就满足 L2 | 免费到小 | 这里尚未证明 |
| 把行切成固定的块，循环调用库 GEMM | L2 | 大 | 只作调试用的正确性 oracle，不用于生产 |

如果还想另加一个专给小 M 用的 kernel，它对每个元素沿 K 的分组必须完全相同，所用指令也必须相同。

### 3. 只根据这一行自己来算量化 scale

- **目标：** 一行的量化值只取决于这一行。
- **做法：** 激活用 per-token 或 per-block 的 scale，或者用静态 scale
  （[recipe 9](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4)）。
  GEMM 本身按第 2 步来做，并固定精度提升的间隔和乘 scale 的顺序。
- **成功标准：** 同一 batch 的其他行填大值时，`check_batch` 仍然通过。这是在模拟的 FP8 数值下证明的
  （[报告](../../examples/kernel_cases/REPORT.md#fp8-activation-scales-l2)）。

| 激活的 scale | L2 | 代价等级 | 什么时候选 |
|---|---|---|---|
| per tensor，运行时取整个 batch 的最大值 | 不成立 | 免费 | 需要一致性时不选 |
| per token | 成立 | 小 | 默认 |
| 沿 K 的 per block | 成立 | 小 | block-scaled 方案 |
| 静态 | 成立 | 免费 | 已经校准过的部署 |

**尚未证明：** 在原生支持 FP8 或 FP4 的硬件上，真正的 FP8 或 FP4 GEMM 每隔多久提升一次部分和的精度。

### 4. 验证 kernel 并投入使用

对模型用到的每个 GEMM 形状运行
[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)。然后把 kernel
注册为不透明的算子，并在 prefill 和 decode 常见的 batch size 上测量，具体做法见[流程](README.md#5-integrate)。

## 可能出错的地方

!!! warning "同 batch 其他行全是零，会掩盖在整个 batch 上计算的 scale"
    全零的行不会抬高 batch 的最大值，所以测试能通过，真实流量下结果却会变。其他行要放大值来测。

!!! warning "对固定的块做 batched matmul，仍然不是不变的"
    对固定行块的一次 batched 调用，仍然会按块的数量挑选 kernel。应该对各块循环调用，或者用单配置 kernel。

!!! warning "小形状上快，不代表大形状上也快"
    未经调优的单配置 kernel 在小的 expert 形状上比库更快，但在大形状上可能更慢。要在你自己的形状上测量。

## 小结

- GEMM 的每个元素是一个点积，库根据 M 决定怎样分组。
- 修法是：所有 M 用同一套配置，不用 split-K，沿 K 按升序累加进 fp32，epilogue 相同。
- 在整个 batch 上计算的量化 scale 会把各行耦合起来；改用 per-token、per-block 或静态 scale。
- 后面每一类算子都含有 GEMM，所以先把 GEMM 做成不变的。

## 深入阅读

- 报告：[单 GPU 上的 MoE](../../examples/moe_combine/REPORT.md)（GEMM 的形状和计时）和
  [kernel 案例](../../examples/kernel_cases/REPORT.md)（原生 GEMM 与 FP8 scale）。
- Recipe：[2，稠密 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16)、
  [3，分组 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)、
  [9，量化 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4)。
- 代码：[tools/bi_kernels.py](../../tools/bi_kernels.py)、
  [examples/kernel_cases/kernel_cases.py](../../examples/kernel_cases/kernel_cases.py)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
