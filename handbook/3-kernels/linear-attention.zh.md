# 线性注意力（gated delta rule）

## 我们在哪里

[MLA 一页](mla-and-sparse.md)说明，同一套数学有两种相等的写法时，每种模式都必须做出同样的选择。
线性注意力遇到的是同一个问题，而且更尖锐：训练和 prefill 通常用分块算法，decode 则一步一步递归。
这里的结果还会通过一个状态从一个 token 带到下一个 token，所以关键在于分块的边界落在哪里。

## 思路：分块边界决定了状态的求和怎样分组

线性注意力层（这里以 Gated DeltaNet 为例）为每个序列的每个头维护一个固定大小的状态 S，每来一个
token 就更新一次。递归形式每个 token 做一次更新；分块形式每次取 C 个 token，用矩阵乘法把它们
合在一起，再把状态从一块传到下一块。

取十个 token，每块 4 个：

<figure class="fx" data-fig="chunks" aria-label="三种调用方式和递归形式下，分块边界分别落在哪里">
<div class="fx-stage">
<div class="fx-rows" style="--n:10">
<div class="fx-row"><span class="fx-label">十个 token，C = 4</span><div class="fx-track"><div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">一次 prefill 处理 token 0–9</span><div class="fx-track"><div class="fx-level fx-sweep" data-at="1" data-hot="1"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:2">{8–9}</span></div><span class="fx-verdict"></span></div></div>
<div class="fx-row"><span class="fx-label">先 prefill 0–7，再每个 token 调一次 decode</span><div class="fx-track"><div class="fx-level" data-at="2" data-hot="3"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-bad" style="--w:1">{8}</span><span class="fx-span to-bad" style="--w:1">{9}</span></div><span class="fx-verdict"><span class="fx-node is-bad" data-at="3">结果随每次调用的长度而变</span></span></div></div>
<div class="fx-row"><span class="fx-label">对齐：在 token 8 处保留状态，重算没填满的块</span><div class="fx-track"><div class="fx-swap is-block"><div class="fx-level" data-at="4" data-to="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span" style="--w:1">{8}</span></div><div class="fx-level" data-at="5" data-hot="5"><span class="fx-span" style="--w:4">{0–3}</span><span class="fx-span" style="--w:4">{4–7}</span><span class="fx-span to-ok" style="--w:2">{8–9}</span></div></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="5">与 prefill 相同</span></span></div></div>
<div class="fx-row"><span class="fx-label">递归形式：每步一个 token</span><div class="fx-track"><div class="fx-level fx-sweep is-unit" data-at="6" data-hot="6"><span class="fx-span" style="--w:1">0</span><span class="fx-span" style="--w:1">1</span><span class="fx-span" style="--w:1">2</span><span class="fx-span" style="--w:1">3</span><span class="fx-span" style="--w:1">4</span><span class="fx-span" style="--w:1">5</span><span class="fx-span" style="--w:1">6</span><span class="fx-span" style="--w:1">7</span><span class="fx-span" style="--w:1">8</span><span class="fx-span" style="--w:1">9</span></div><span class="fx-verdict"><span class="fx-node is-ok" data-at="6">每种模式下都逐 bit 一致</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>十个 token，每块 C = 4 个。分块决定了状态的求和怎样分组。</li>
<li>一次 prefill 处理 token 0–9，分块是 {0–3}、{4–7} 和 {8–9}。</li>
<li data-hold="1.5">如果先 prefill 0–7，再每个 token 调一次 decode，每次调用都会自己开一个新块：先是 {8}，然后是 {9}。</li>
<li data-hold="1.5">token 9 按另一种分组求和，所以它的输出和状态在最后几个 bit 上不同。</li>
<li data-hold="1.5">对齐的做法是在 token 8 处保留状态（这里正好是块边界），等 token 9 到来时，重算这个没填满的块。</li>
<li data-hold="1.5">分块又是 {8–9}，和 prefill 一样，所以结果逐 bit 一致。</li>
<li data-hold="2">递归形式在每种模式下都是每个 token 更新一次，不需要额外代价就逐 bit 一致。</li>
</ol>
<figcaption>图：只有每个块都从 C 的绝对整数倍开始，分块形式才在各模式下给出相同的结果；递归形式则总能做到。</figcaption>
</figure>

第二种调用方式把同样的项按另一种分组相加，所以 token 9 的输出和状态在最后几个 bit 上不同。

由此可以得出三点：

- **问题不在 batch。** 每个 program 只负责一个序列的一个头，没有跨序列的归约，所以两种形式
  天然就是 batch-invariant 的（L2）。
- **能否在各模式下不变，取决于分块边界。** 递归形式在各模式下天然不变，不需要额外代价，因为每种模式
  运行的都是同一个循环体。分块形式只有在每块都从 C 的绝对整数倍开始时，才在各模式下不变。
- **分块和递归的结果不同，差多少由精度决定。** 分块中的矩阵乘法用 fp32 时，两者精度相当；如果像
  常见 kernel 那样用 bf16 操作数，分块形式的误差会大得多，实际看到的差距大部分就来自这里。

## 步骤

### 1. 先确认这一层是 batch-invariant 的

- **目标：** 先排除代价最小的那类失败。
- **做法：** 找一个最后一块没填满、初始状态不为零的序列，运行
  [tools/bi_check.py](../../tools/bi_check.py) 里的 `bi_check.check_batch`。
- **成功标准：** 两种形式都通过。这在一种 GPU 上得到了证明
  （[报告](../../examples/linear_attention/REPORT.md#results)）。

### 2. 选一种 trainer 和引擎都用的形式

- **目标：** 两端的前向用同一种算术。
- **做法：** 使用 [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py) 里的递归
  kernel 和分块 kernel，以及 [gdn_bi.py](../../examples/linear_attention/gdn_bi.py) 里的对齐分块
  包装 `AlignedChunked`；做法见
  [recipe 7](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style)。
- **成功标准：** decode 和每种分块大小得到的输出和状态，都与一次完整的 prefill 逐 bit 一致。
  这在一种 GPU 上得到了证明
  （[报告](../../examples/linear_attention/REPORT.md#recommendation-with-the-measured-cost-of-each-option)）。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| (a) 处处用递归前向 | L1–L3，两端用同一个 kernel | trainer 前向：短序列和中等序列上中等，长序列上大；decode 最便宜 | 需要一致性时的默认选择 |
| (b) 处处用分块，decode 按对齐的分块步进行 | L1–L3，trainer 不用改 | trainer：免费；decode：大 | 序列很长、(a) 代价最高时 |
| (c) 接受差异 | 不逐 bit 一致 | 免费 | 不要求一致性时；这时分块乘法尽量用 fp32 |

(a) 和 (b) 哪个总体更便宜，取决于你的工作负载把多少时间花在 trainer 前向上、多少花在
rollout decode 上。

### 3. 两端使用同一套 kernel 配置

- **目标：** trainer 和引擎里的 kernel 内部用同一棵归约树。
- **做法：** 递归 kernel 在两端用相同的 tile 宽度和 warp 数；分块 kernel 用相同的块大小和
  相同的乘法精度。
- **成功标准：** 两端的 kernel 对同一输入给出逐 bit 一致的结果。不同的递归配置各自都能通过
  L1–L3，但彼此之间结果不同（[报告](../../examples/linear_attention/REPORT.md#what-this-shows)）。

### 4. 从保存的状态出发，逐个模式验证

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)，让每种模式都从
保存下来的 fp32 状态继续计算。比较时要看 cast 之前的 fp32 输出和最终状态，而不只看 bf16 输出。

## 可能出错的地方

!!! warning "只有一个 token 的块并不是递归步"
    如果用分块 kernel 每次只 decode 一个新 token，每次调用都会开一个新块，结果和 prefill 并不
    逐 bit 一致。正确的做法是在最后一个绝对边界处保留状态，再重算没填满的块。

!!! warning "两种配置就是两棵归约树"
    递归 kernel 的 tile 宽度和 warp 数会改变状态更新里求和的分组。要达到 L4，两端必须固定成同样的值。

!!! warning "bf16 输出会掩盖状态上的差异"
    大多数差异在最后 cast 成 bf16 时被抹掉，又会在后面的 token 里重新出现。要比较 fp32 输出和状态。

**尚未证明：** 递归前向对应的反向；对齐包装下、各序列处在不同位置的 decode batch；以及与生产环境
kernel 库的一致性。

## 小结

- 两种形式天然是 batch-invariant 的；能否在各模式下不变，取决于分块边界落在哪里。
- 递归形式在各模式下天然不变，不需要额外代价；分块形式要让每块都从 C 的绝对整数倍开始，才在各模式下不变。
- 分块和递归在实际中的差距，主要来自分块矩阵乘法的精度。
- 看代价落在哪里，在 (a) 递归和 (b) 对齐分块之间选一个，并在两端固定 kernel 配置。

## 深入阅读

- 报告：[线性注意力（GDN）](../../examples/linear_attention/REPORT.md)和
  [kernel 案例](../../examples/kernel_cases/REPORT.md#plain-linear-attention-chunked-vs-recurrent-an-algorithmic-difference)
  （普通线性注意力）。
- 代码：[examples/linear_attention/gdn_bi.py](../../examples/linear_attention/gdn_bi.py)、
  [gdn_kernels.py](../../examples/linear_attention/gdn_kernels.py)。
- Recipe：[7，线性注意力](../../skills/batch-invariant-kernel/references/operator-recipes.md#7-linear-attention-and-ssms-gdn-mamba-style)；
  其他人报告的、只用递归形式训练的代价，见[参考文献](../references.md#rl-traininference-alignment)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
