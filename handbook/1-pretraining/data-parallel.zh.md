# 数据并行与 FSDP 式切分下的梯度

## 我们在哪里

当首个分歧点指向一个由多个 rank 共同求和的梯度时，本页服务于[预训练工作流](README.md)的第 5 步（分类）；对任何使用数据并行的运行，它也服务于第 2 步（控制）。[训练 step](training-step.md) 已经在单个 rank 上固定；还没解决的是跨 rank 的求和。DDP 把梯度分桶做 all-reduce，FSDP 和 ZeRO 对梯度的扁平副本做 reduce-scatter，两者还都要跨 rank 归约梯度范数和 loss 均值。每一个求和都有顺序，而这个顺序可能取决于与该参数毫无关系的东西。

## 思路：顺序跟着位置走，并行度改变的是和本身

**小例子：位置。** 四个 rank 持有某个参数的梯度 g0、g1、g2、g3。ring all-reduce 把消息切成若干块，每块的求和从不同的 rank 开始：

| 元素所在的块 | 求和顺序（一个例子） |
|---|---|
| 0 | ((g1 + g2) + g3) + g0 |
| 2 | ((g3 + g0) + g1) + g2 |

把同一个参数放到消息中的另一个偏移处（另一个桶大小、另一种参数顺序、前面多一个 padding 元素），它的元素就落进别的块：另一个顺序，另一些比特。DDP 甚至会改变自己的打包：构造后的第一次迭代把所有梯度放进一个桶，之后的迭代重建分桶，每次重启又从头开始。reduce-scatter 通过归属 rank 有同样的性质：一个 rank 保留的切片从下一个 rank 开始求和，所以重新切分会移动顺序。

**小例子：并行度。** 16 条序列在 d = 2 时，每个 rank 对 8 条序列做一次 backward，再把两个部分梯度相加。d = 4 时，是四个各含 4 条序列的部分和。这是不同的和，而不是同一个和换了顺序。没有哪种集合通信设置能修复它。

<figure class="fx" data-fig="dp-tree" aria-label="并行度为 2 和 4 时，按 rank 求和与按逻辑分片求和的梯度">
<div class="fx-stage">
<div class="fx-head"><span class="fx-label">16 条序列，数据并行度</span><span class="fx-swap"><span class="fx-chip" data-at="1" data-to="3">d = 2</span><span class="fx-chip is-key" data-at="3">d = 4</span></span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">每个 rank 各做反向</span><div class="fx-tree" style="--n:16">
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-band" data-c="0" style="--w:8">rank 0</span><span class="fx-band" data-c="1" style="--w:8">rank 1</span></div><div class="fx-level" data-at="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span><span class="fx-band" data-c="1" style="--w:4">rank 1</span><span class="fx-band" data-c="2" style="--w:4">rank 2</span><span class="fx-band" data-c="3" style="--w:4">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span><span class="fx-cell">10</span><span class="fx-cell">11</span><span class="fx-cell">12</span><span class="fx-cell">13</span><span class="fx-cell">14</span><span class="fx-cell">15</span></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 0–7</span></span><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 8–15</span></span></div><div class="fx-level" data-at="3"><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 0–3</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 4–7</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 8–11</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 12–15</span></span></div></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="2" data-to="3"><span class="fx-j" style="--w:16"><span class="fx-node is-mono is-key">x₂</span></span></div><div class="fx-level" data-at="4"><span class="fx-j" style="--w:16;--k:4"><span class="fx-node is-mono is-bad">x₄ ≠ x₂</span></span></div></div>
</div></div>
<div class="fx-row"><span class="fx-label">逻辑分片，任意 d</span><div class="fx-tree" style="--n:16">
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-band" data-c="0" style="--w:8">rank 0</span><span class="fx-band" data-c="1" style="--w:8">rank 1</span></div><div class="fx-level" data-at="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span><span class="fx-band" data-c="1" style="--w:4">rank 1</span><span class="fx-band" data-c="2" style="--w:4">rank 2</span><span class="fx-band" data-c="3" style="--w:4">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span><span class="fx-cell">10</span><span class="fx-cell">11</span><span class="fx-cell">12</span><span class="fx-cell">13</span><span class="fx-cell">14</span><span class="fx-cell">15</span></div>
<div class="fx-level" data-at="1" data-hot="3"><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-op">+</span></span></div>
<div class="fx-level" data-at="1" data-hot="3"><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:4"><span class="fx-node is-op">+</span></span></div>
<div class="fx-level" data-at="1" data-hot="4"><span class="fx-j" style="--w:8"><span class="fx-node is-op">+</span></span><span class="fx-j" style="--w:8"><span class="fx-node is-op">+</span></span></div>
<div class="fx-level" data-at="2" data-hot="4"><span class="fx-j" style="--w:16"><span class="fx-node is-mono is-ok">x</span></span></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>一个全局 batch 里有 16 条序列。</li>
<li data-hold="2">d = 2：每个 rank 拥有 8 条序列。按 rank：一次反向，一个部分和。逻辑分片：每个 rank 求一棵固定的树中自己的子树。</li>
<li data-hold="1.5">各 rank 的部分和相加：x₂。逻辑分片完成同一棵树：x。</li>
<li data-hold="2">d = 4：每个 rank 拥有 4 条序列。按 rank：四个部分和，是另外的和。逻辑分片：同一棵树，每个 rank 的子树更小。</li>
<li data-hold="2">按 rank：x₄ ≠ x₂，结果随并行度变化。逻辑分片：同一棵树，相同的位 x。</li>
</ol>
<figcaption>图：d = 2 和 d = 4 时，按 rank 的部分和取决于并行度。用逻辑分片时，每个并行度求的都是同一棵树。</figcaption>
</figure>

**规则。** 跨 rank 求和的顺序必须是元素和批次的固定逻辑切分的函数，绝不能是消息位置、归属 rank 或并行度的函数。这是第 0 部分的原则在 L5 上的体现：并行度可以改变由谁计算某个部分和，但不能改变有哪些部分和。

## 步骤

### 固定跨 rank 的求和

**目标：** 参数的梯度、范数、loss 和更新后的权重不取决于分桶、切分布局或并行度。

**做法：** 选择你的运行需要的选项。代码：[training_dp.py](../../examples/training_dp/training_dp.py)（有序集合通信、DDP comm hook `Worker.ordered_hook`、logical shards、逻辑范数）；每次比较都用 [bi_check.py](../../tools/bi_check.py) 逐位进行。

| 选项 | 给出什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| all-reduce 用 `NCCL_ALGO=Tree` | 固定并行度下，无论如何分桶，每个参数只有一个梯度 | 大消息小到中等，小消息大 | 固定并行度的 DDP；最快的修复 |
| 有序 all-reduce（一次 all-to-all，按 rank 顺序在 fp32 中求和，一次 all-gather；作为 DDP comm hook） | 同上，且不依赖 NCCL 的算法 | 单节点小到中等 | 需要在任何 NCCL 设置下成立 |
| 有序 reduce-scatter（rank r 接收每条消息的第 r 片，并按 rank 顺序把 d 个切片相加） | FSDP 式梯度不受 padding、包装和重新切分影响 | 单节点中等（2 倍以内）；跨节点用两级路由 | 切分训练；任何布局变化 |
| logical shards：S 个固定的 micro-batch，每个一次 backward，按 shard 索引做一棵成对的树，树中包括跨 rank 的求和 | 每个并行度下梯度、权重和 Adam 状态都相同；弹性恢复 | 每个 shard 含多条序列时中等；shard 很小时大 | 并行度会变，或两个并行度下的运行必须一致 |
| 逻辑范数和 loss：固定的块，fp64 平方和，一棵成对的树；loss 用 shard loss 上的树 | 只取决于数值本身的裁剪范数和 loss | 范数本身中等；在整个 step 中小 | 使用 logical shards 时总是需要 |
| single-tensor AdamW（`foreach=False, fused=False`） | 逐元素更新，所以复制和切分的权重给出相同比特 | 小 | 总是；固定一条路径 |

logical shards 要求[训练 step](training-step.md) 已经固定：每个 shard 的 backward 必须在每个 rank 上可重复，用原生算子时这需要严格确定性模式。跨两个节点时，先在节点内路由交换（两级路由）；在较慢的链路上做扁平 all-to-all 代价高得多。

**成功：** 梯度、loss、范数、权重和 Adam 状态在每个 rank 上都与 d = 1 的运行逐位相等，并且在另一个并行度下的弹性恢复与不间断运行一致。

这在 fp32 梯度下得到证明，最多到两个节点上的 16 块 GPU，包括复制模式和模拟的 FSDP（扁平梯度、reduce-scatter、切分的 AdamW；没有运行框架自带的 FSDP）：Tree 和有序集合通信消除了分桶的影响，有序 reduce-scatter 消除了归属 rank 的影响，logical shards 在测试的每个并行度下以及弹性恢复后都给出相同结果（[报告](../../examples/training_dp/REPORT.md#what-this-shows)）。

### 给梯度求和中的首个分歧点分类

| 观察 | 原因 |
|---|---|
| 同一并行度下，改了桶大小、参数顺序或模型之后梯度改变 | ring all-reduce 的顺序跟着消息偏移走 |
| 第一次迭代，或重启后的第一次，与其余不同 | DDP 第一次迭代只用一个桶 |
| 每个不同的元素都换了归属 rank | reduce-scatter 的顺序跟着归属 rank 走 |
| 换一个并行度，在任何集合通信下几乎每个元素都不同 | 每个 rank 的 backward 是算术的一部分：使用 logical shards |
| 裁剪范数只在某些放置下不同 | 切分后的平方和由 NCCL 相加 |

## 可能出错的地方

!!! warning "DDP 与自己都不一致"
    DDP 构造后的第一次迭代只用一个桶，而每次重启都会重新构造 DDP。在 Ring 下，单是一次重启就会改变梯度。使用 Tree 或有序 hook，而不是调桶大小。

!!! warning "Tree 管不到 reduce-scatter"
    `NCCL_ALGO=Tree` 只改变 all-reduce。切分训练仍然跟着归属 rank 走。FSDP 式切分要用有序 reduce-scatter。

!!! warning "跨 rank 的部分又交还给 NCCL"
    如果树的最后一步是 NCCL 的 all-reduce 或 reduce-scatter，无论 Ring 还是 Tree，logical shards 从三个 rank 起就会失败。那时的树已不是同一棵树。每个跨 rank 的求和都必须是这棵固定树的一部分。

!!! warning "侥幸通过的标量"
    几十项的和常常舍入到同一个值，所以原生的切分范数在很多放置下能一致。它仍然是切分方式的函数，在其他放置下会失败。使用按块对齐的逻辑范数。

!!! warning "logical shards 不等于整批梯度"
    d = 1 时的 logical shards 与对同样序列做一次整批 backward 的结果不同。两者都是正确的舍入，只是不同的算术。选一个作为参考，并处处使用它。

## 小结

- 在 Ring 下，梯度的比特取决于它在消息中的位置；在 reduce-scatter 下，取决于归属 rank。Tree 或有序集合通信修复前者，有序 reduce-scatter 修复后者。
- 另一个并行度就是另一组部分和：只有使用一棵固定树的 logical shards 才能给出 L5 和弹性恢复。
- 范数和 loss 均值也是跨 rank 的和：给它们同一棵树。
- 固定优化器路径。

## 深入阅读

- [报告：数据并行与 FSDP 式切分](../../examples/training_dp/REPORT.md)：每种打包、布局和放置，耗时，被否决的设计，局限。
- 代码：[training_dp.py](../../examples/training_dp/training_dp.py)。
- [集合通信](../3-kernels/collectives.md)：张量并行层中的同样的集合通信。
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)，配方 10。
