# 线性注意力（gated delta rule）

## 我们在哪里

[MLA](mla-and-sparse.md) 说明了同一数学的两种相等形式是一个选择，每种模式都要做出相同的选择。
线性注意力以更尖锐的方式面对同样的选择。训练和 prefill 通常运行分块算法，decode 运行递归步。
这里结果还通过状态从一个 token 传到下一个，所以关键在于分块边界落在哪里。

## 思路：状态按分块决定的分组求和

线性注意力层（这里是 Gated DeltaNet）为每个序列和头保留一个固定大小的状态 S，并随每个 token
更新。递归形式每个 token 做一次更新。分块形式把 C 个 token 分成一组，用矩阵乘积合并它们，再把
状态从一块传到下一块。

取十个 token、每块 4 个：

<figure class="fx" data-fig="chunks" aria-label="三种调用方式和循环形式下块边界的位置">
<div class="fx-stage">
<div class="fx-rows" style="--n:10">
<div class="fx-row"><span class="fx-label">十个 token，C = 4</span><div class="fx-track"><div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">一次 prefill token 0–9</span><div class="fx-track"><div class="fx-level fx-sweep" data-at="1" data-hot="1"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:2">{8–9}</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">prefill 0–7，然后每个 token 一次 decode 调用</span><div class="fx-track"><div class="fx-level" data-at="2" data-hot="3"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-bad" style="--w:1">{8}</span><span class="fx-span to-bad" style="--w:1">{9}</span></div><span class="fx-verdict"><span class="fx-node is-bad" data-at="3">位随调用大小变化</span></span></div></div>
<div class="fx-row"><span class="fx-label">对齐：在 token 8 处保留状态，重算未满的块</span><div class="fx-track"><div class="fx-swap is-block"><div class="fx-level" data-at="4" data-to="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:1">{8}</span></div><div class="fx-level" data-at="5" data-hot="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-ok" style="--w:2">{8–9}</span></div></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="5">与 prefill 相同</span></span></div></div>
<div class="fx-row"><span class="fx-label">循环形式：每步一个 token</span><div class="fx-track"><div class="fx-level fx-sweep is-unit" data-at="6" data-hot="6"><span class="fx-span" style="--w:1">0</span><span class="fx-span" style="--w:1">1</span><span class="fx-span" style="--w:1">2</span><span class="fx-span" style="--w:1">3</span><span class="fx-span" style="--w:1">4</span><span class="fx-span" style="--w:1">5</span><span class="fx-span" style="--w:1">6</span><span class="fx-span" style="--w:1">7</span><span class="fx-span" style="--w:1">8</span><span class="fx-span" style="--w:1">9</span></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="6">每种模式下位都相同</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>十个 token，每块 C = 4 个。块决定状态的和如何分组。</li>
<li>一次 prefill token 0–9：块 {0–3}、{4–7}、{8–9}。</li>
<li data-hold="1.5">prefill 0–7，然后每个 token 一次 decode 调用：每次调用开始自己的块，先是 {8}，然后是 {9}。</li>
<li data-hold="1.5">token 9 以另一种分组求和：它的输出和状态在最后几位上不同。</li>
<li data-hold="1.5">对齐：在 token 8（一个块边界）处保留状态，token 9 到来时重算未满的块：</li>
<li data-hold="1.5">又是 {8–9}，与 prefill 相同，所以位相同。</li>
<li data-hold="2">循环形式：每种模式下每个 token 更新一次，位天然相同。</li>
</ol>
<figcaption>图：分块形式只有在每个块都从 C 的绝对整数倍开始时才满足模式不变性；循环形式总是满足。</figcaption>
</figure>

第二种方式以不同的分组对相同的项求和，所以 token 9 的输出和状态在最后几位上不同。

由此得出三点：

- **问题不在批。** 每个 program 负责一个序列和一个头；没有跨序列的归约，所以两种形式天然批不变
  （L2）。
- **模式不变性取决于边界。** 递归形式天然模式不变：每种模式运行同一个循环体。分块形式只有在
  每块都从 C 的绝对倍数开始时才模式不变。
- **分块与递归不同，差多少由精度决定。** 分块的矩阵乘积用 fp32 时两者精度相同；用 bf16 操作数
  （常见 kernel 的做法）时，分块形式误差大得多，也占了实际差距的大部分。

## 步骤

### 1. 确认批不变

- **目标：** 先排除廉价的失败。
- **方法：** 对一个末块不满、初始状态非零的序列，运行 [tools/bi_check.py](../../tools/bi_check.py)
  中的 `bi_check.check_batch`。
- **成功标准：** 两种形式都通过。在一种 GPU 上得到证明
  （[报告](../../examples/linear_attention/REPORT.md#results)）。

### 2. 选择 trainer 与引擎如何一致

- **目标：** 两侧的前向用同一种算术。
- **方法：** [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py) 中的递归和分块
  kernel，以及 [gdn_bi.py](../../examples/linear_attention/gdn_bi.py) 中的对齐分块包装
  `AlignedChunked`；
  [recipe 7](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style)。
- **成功标准：** decode 和每种分块大小在输出和状态上都与完整 prefill 逐位相等。在一种 GPU 上
  得到证明（[报告](../../examples/linear_attention/REPORT.md#recommendation-with-the-measured-cost-of-each-option)）。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| (a) 处处用递归前向 | L1–L3；两侧同一 kernel | trainer 前向在短和中等序列上中等，长序列上大；decode 最便宜 | 一致性的默认选择 |
| (b) 处处分块，decode 作为对齐的分块步 | L1–L3；trainer 不变 | trainer 免费；decode 大 | 长序列，(a) 代价最高时 |
| (c) 接受差异 | 不逐位 | 免费 | 不需要一致性；此时优先用 fp32 的分块乘积 |

(a) 和 (b) 哪个总体更便宜，取决于工作负载在 trainer 前向和 rollout decode 之间怎么分配时间。

### 3. 两侧固定同一配置

- **目标：** trainer 和引擎的 kernel 内部用同一棵归约树。
- **方法：** 递归 kernel 用相同的 tile 宽度和 warp 数；分块 kernel 用相同的块大小和乘积精度。
- **成功标准：** 两侧 kernel 在相同输入上给出相同的位。不同的递归配置各自通过 L1–L3，彼此却
  不同（[报告](../../examples/linear_attention/REPORT.md#what-this-shows)）。

### 4. 从保存的状态验证各模式

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)，让每种模式从保存
的 fp32 状态继续，比较 cast 之前的 fp32 输出和最终状态，而不只比较 bf16 输出。

## 可能出错的地方

!!! warning "单 token 的块不是递归步"
    用分块 kernel 对每个新 token 做 decode，每次调用都会开始一个新块。它与 prefill 不逐位相等；
    要在最后一个绝对边界处保留状态，并重算未满的块。

!!! warning "两种配置就是两棵归约树"
    递归 kernel 的 tile 宽度和 warp 数会改变状态更新里的求和。为了 L4，两侧都要固定它们。

!!! warning "bf16 输出会掩盖状态的差异"
    大多数差异在最后的 cast 中消失，又在后面的 token 中回来。要比较 fp32 输出和状态。

**尚未证明：** 递归前向的反向；对齐包装下各序列处于不同位置的 decode 批；与生产 kernel 库的一致性。

## 小结

- 两种形式天然批不变；模式不变性取决于分块边界落在哪里。
- 递归天然模式不变；分块在边界位于 C 的绝对倍数时模式不变。
- 分块与递归的实际差距主要来自分块矩阵乘积的精度。
- 按代价落在哪一侧，在 (a) 递归和 (b) 对齐分块之间选择，并在两侧固定 kernel 配置。

## 深入阅读

- 报告：[线性注意力（GDN）](../../examples/linear_attention/REPORT.md)、
  [kernel 案例](../../examples/kernel_cases/REPORT.md#plain-linear-attention-chunked-vs-recurrent-an-algorithmic-difference)
  （普通线性注意力）。
- 代码：[examples/linear_attention/gdn_bi.py](../../examples/linear_attention/gdn_bi.py)、
  [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py)。
- Recipe：[7，线性注意力](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style)；
  只用递归做训练的已报告代价见[参考文献](../references.md#rl-traininference-alignment)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
