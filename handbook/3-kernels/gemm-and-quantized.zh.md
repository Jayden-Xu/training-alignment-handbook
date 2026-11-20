# GEMM 与量化 GEMM

## 我们在哪里

[流程](README.md)是：定位变化来源，选最便宜的策略，保持舍入点，验证，集成。GEMM 是用它
最简单的地方。每个输出元素是一个固定长度 K 的点积，算子里没有别的归约。后面每一页都建立在
不变的 GEMM 上：注意力投影、MLA 的逐头乘积、MoE 专家和张量并行层都含有 GEMM。

## 思路：库按批来选择拆分

输出元素 C[m, n] 是 K 个乘积之和。GEMM 库把输出切成 tile，并在每次调用时决定怎么走 K。
这个决定以问题形状为键，而形状包括 M，即批中的行数。

<figure class="fx" data-fig="gemm-split" aria-label="矩阵乘的第 0 行在 1 行的 batch 和 64 行的 batch 中">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">1 行的 batch：工作量小<span class="fx-batch fx-mini"><span class="fx-rowbar is-mine"></span></span></span><div class="fx-tree" style="--n:4">
<div class="fx-level"><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span><span class="fx-cell" data-hot="1">K/4</span></div>
<div class="fx-level" data-at="1"><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 0</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 1</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 2</span></span><span class="fx-j" style="--k:1"><span class="fx-node is-mono">program 3</span></span></div>
<div class="fx-level" data-at="2" data-hot="4"><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-key">第 0 行：值 A</span></span></div>
</div></div>
<div class="fx-row"><span class="fx-label">64 行的 batch：工作量足够<span class="fx-batch fx-mini"><span class="fx-rowbar is-mine"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span><span class="fx-rowbar"></span></span></span><div class="fx-tree" style="--n:4">
<div class="fx-level fx-sweep" data-hot="3"><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span><span class="fx-cell">K/4</span></div>
<div class="fx-level" data-at="3"><span class="fx-walk" style="--w:4">一个 program 按顺序走完整个 K</span></div>
<div class="fx-level" data-at="4"><span class="fx-j" style="--w:4;--k:1"><span class="fx-node is-bad">第 0 行：值 B ≠ A</span></span></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>第 0 行（高亮）在两个 batch 里是同一个沿 K 的点积。</li>
<li data-hold="1.5">1 行的 batch：工作量小，所以库把 K 切成 4 段，由 4 个 program 计算。</li>
<li data-hold="1.5">把 4 个部分和相加：第 0 行得到值 A。</li>
<li data-hold="2">64 行的 batch：不切分工作量也足够。一个 program 按顺序走完整个 K。</li>
<li data-hold="2">第 0 行得到值 B ≠ A：数学相同，分组不同。</li>
</ol>
<figcaption>图：库根据行数选择 K 的切分方式，所以同一行在另一个 batch 里得到不同的位。</figcaption>
</figure>

批小时，库可能拆分 K 以让 GPU 忙起来（split-K 或 stream-K），或调用专门的小 M kernel。
批大时它一遍走完 K。每种选择都是确定的，所以同一批每次运行位都相同（L1）。
但同一行放进另一个批，就以另一种分组求和（L2 失败）。

**GEMM 的规则。** 所有 M 用同一个 tile 配置，不用 split-K 或 stream-K，K 按升序累加进
fp32 累加器，epilogue 相同。代价是小 M 时利用率较低。

**量化 GEMM 多了一种耦合：scale。** FP8 激活存成一个小数乘以一个 scale，而 scale 由最大值
决定。如果最大值取自整个批，一行的 scale 就取决于它的邻居：

| 行 x = [0.5, 3] | scale 来自 | scale | x 变成什么 |
|---|---|---|---|
| 单独 | 它自己的最大值 3 | 细 | 在它的范围内以全精度存储 |
| 旁边有一行含 100 | 批最大值 100 | 粗 | 在更粗的网格上舍入：位不同 |

per-token 或 per-block 的 scale 只用该行自己的值，耦合就没有了。真正的 FP8 GEMM 还要在
kernel 内部固定部分和多久提升一次精度、以什么顺序乘 scale；两者都是归约的一部分。

## 步骤

### 1. 定位变化来源

- **目标：** 知道模型中哪些 GEMM 形状会随 M 改变位。
- **方法：** 对模型运行的每个形状 (K, N)，用 [tools/bi_check.py](../../tools/bi_check.py)
  中的 `bi_check.check_batch`，在 2 的幂和 tile 倍数附近的批大小上，配随机、零和大值填充行。
- **成功标准：** 一份失败形状和大小的清单。测得：库 GEMM 在测过的每个形状上都失败，常常从
  2 行就开始（[报告](../../examples/moe_combine/REPORT.md#1-locate-the-variance)）。

### 2. 固定一个配置

- **目标：** 无论 M 是多少，每个元素只有一棵归约树。
- **方法：** [tools/bi_kernels.py](../../tools/bi_kernels.py) 中的单配置 GEMM
  `mm_one_config`，或按
  [recipe 2](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16)
  自己写的 kernel。多个头或专家用循环或头索引复用同一个 kernel 体
  （[recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)）。
- **成功标准：** `check_batch` 在每个大小上通过。在一种 GPU 上、三个形状上得到证明
  （[报告](../../examples/moe_combine/REPORT.md#1-locate-the-variance)）。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 原样使用库 GEMM | 只有 L1 | 免费 | 一致性场景下永远不选 |
| 单配置 kernel | 所有 M 上 L2 | 在测过的形状上最多中等；小形状上可能更快 | 默认修法 |
| 固定算法的库 GEMM | 若固定了拆分则 L2 | 免费到小 | 这里尚未证明 |
| 对固定行块循环调用库 GEMM | L2 | 大 | 用于调试的正确性 oracle，不是产品 |

第二个小 M kernel 只有在复现相同的逐元素 K 分组和指令时才允许。

### 3. 固定量化 scale

- **目标：** 一行的量化值只取决于这一行。
- **方法：** per-token 或 per-block 的激活 scale，或静态 scale；
  [recipe 9](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4)。
  GEMM 本身按第 2 步来，并固定提升间隔和 scale 顺序。
- **成功标准：** 配大值填充行时 `check_batch` 通过。在模拟的 FP8 数值下得到证明
  （[报告](../../examples/kernel_cases/REPORT.md#fp8-activation-scales-l2)）。

| 激活 scale | L2 | 代价等级 | 何时选择 |
|---|---|---|---|
| per tensor，动态（批最大值） | 失败 | 免费 | 一致性场景下不选 |
| per token | 成立 | 小 | 默认 |
| 沿 K 的 per block | 成立 | 小 | block-scaled 方案 |
| 静态 | 成立 | 免费 | 已校准的部署 |

**尚未证明：** 在原生支持的硬件上，真正的 FP8 或 FP4 GEMM 的提升间隔。

### 4. 验证与集成

对模型用到的每个 GEMM 形状运行
[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)，然后把 kernel
注册为不透明算子，并在 prefill 和 decode 批大小上测量，做法同[流程](README.md#5-integrate)。

## 可能出错的地方

!!! warning "零填充会掩盖批级 scale"
    零行不会抬高批最大值。测试通过，真实流量却改变位；要用大值填充行测试。

!!! warning "对固定块的批量 matmul 并不不变"
    对固定行块的一次批量调用，仍按块数挑选 kernel。要对块循环，或用单配置 kernel。

!!! warning "小形状上的快不能推广"
    未调优的单配置 kernel 在小的专家形状上比库快；在大形状上可能更慢。要在你自己的形状上测量。

## 小结

- GEMM 元素是一个点积；库按 M 选择它的分组。
- 修法：所有 M 同一配置，不用 split-K，K 升序累加进 fp32，epilogue 相同。
- 批级量化 scale 把各行耦合起来；改用 per-token、per-block 或静态 scale。
- 后面每一类都含有 GEMM：先把它做成不变的。

## 深入阅读

- 报告：[单 GPU 上的 MoE](../../examples/moe_combine/REPORT.md)（GEMM 形状和计时）、
  [kernel 案例](../../examples/kernel_cases/REPORT.md)（原生 GEMM 和 FP8 scale）。
- Recipe：[2，稠密 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#2-dense-gemm-bf16fp16)、
  [3，分组 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)、
  [9，量化 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#9-quantized-gemm-fp8-block-scaled-mxfp8-nvfp4)。
- 代码：[tools/bi_kernels.py](../../tools/bi_kernels.py)、
  [examples/kernel_cases/kernel_cases.py](../../examples/kernel_cases/kernel_cases.py)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
