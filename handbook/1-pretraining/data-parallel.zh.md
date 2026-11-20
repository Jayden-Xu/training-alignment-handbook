# 数据并行与 FSDP

## 我们在哪里

如果在[预训练流程](README.md)的第 5 步（判断原因类别）发现，首个分歧点指向一个由多个 rank 共同求和的梯度，就读这一页；凡是用了数据并行的运行，第 2 步（控制）也要用到这一页。[训练 step](training-step.md) 已经在单个 rank 上固定好了，剩下的是跨 rank 的求和。DDP 把梯度分桶后做 all-reduce，FSDP 和 ZeRO 对梯度展平后的副本做 reduce-scatter，两者还都要跨 rank 归约梯度范数和 loss 均值。这些求和各有各的顺序，而这个顺序可能取决于一些和参数本身毫无关系的因素。

## 思路：梯度在消息里的位置决定求和顺序，并行度决定有哪些部分和

**小例子：位置。** 四个 rank 各持有同一个参数的梯度 g0、g1、g2、g3。ring all-reduce 会把消息切成若干块，每一块的求和从不同的 rank 开始：

| 元素所在的块 | 求和顺序（举例） |
|---|---|
| 0 | ((g1 + g2) + g3) + g0 |
| 2 | ((g3 + g0) + g1) + g2 |

如果同一个参数在消息里的偏移变了（因为桶的大小变了、参数顺序变了，或者前面多了一个 padding 元素），它的元素就会落进别的块，按别的顺序求和，结果也就变了。DDP 甚至会自己改变打包方式：构造后的第一次迭代把所有梯度放进一个桶，之后的迭代再重新分桶，而每次重启又会从头来一遍。reduce-scatter 也有同样的问题，只不过取决于切片归哪个 rank：一个 rank 保留的那一片，是从下一个 rank 开始求和的，所以一旦重新切分，顺序也跟着变。

**小例子：并行度。** 取 16 条序列。数据并行度 d = 2 时，每个 rank 对 8 条序列做一次反向计算，再把两个部分梯度相加；d = 4 时，相加的是四个各含 4 条序列的部分梯度。这是两个不同的和，而不是同一个和换了顺序，任何集合通信设置都无法让它们相等。

<figure class="fx" data-fig="dp-tree" aria-label="并行度为 2 和 4 时，按 rank 求和与按 logical shard 求和的梯度">
<div class="fx-stage">
<div class="fx-head"><span class="fx-label">16 条序列，数据并行度</span><span class="fx-swap"><span class="fx-chip" data-at="1" data-to="3">d = 2</span><span class="fx-chip is-key" data-at="3">d = 4</span></span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">每个 rank 各做一次反向</span><div class="fx-tree" style="--n:16">
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-band" data-c="0" style="--w:8">rank 0</span><span class="fx-band" data-c="1" style="--w:8">rank 1</span></div><div class="fx-level" data-at="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span><span class="fx-band" data-c="1" style="--w:4">rank 1</span><span class="fx-band" data-c="2" style="--w:4">rank 2</span><span class="fx-band" data-c="3" style="--w:4">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">0</span><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span><span class="fx-cell">5</span><span class="fx-cell">6</span><span class="fx-cell">7</span><span class="fx-cell">8</span><span class="fx-cell">9</span><span class="fx-cell">10</span><span class="fx-cell">11</span><span class="fx-cell">12</span><span class="fx-cell">13</span><span class="fx-cell">14</span><span class="fx-cell">15</span></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="1" data-to="3"><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 0–7</span></span><span class="fx-j" style="--w:8;--k:8"><span class="fx-node is-mono">Σ 8–15</span></span></div><div class="fx-level" data-at="3"><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 0–3</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 4–7</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 8–11</span></span><span class="fx-j" style="--w:4;--k:4"><span class="fx-node is-mono">Σ 12–15</span></span></div></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="2" data-to="3"><span class="fx-j" style="--w:16"><span class="fx-node is-mono is-key">x₂</span></span></div><div class="fx-level" data-at="4"><span class="fx-j" style="--w:16;--k:4"><span class="fx-node is-mono is-bad">x₄ ≠ x₂</span></span></div></div>
</div></div>
<div class="fx-row"><span class="fx-label">logical shard，任意 d</span><div class="fx-tree" style="--n:16">
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
<li data-hold="2">d = 2 时，每个 rank 负责 8 条序列。每个 rank 各做一次反向，就得到一个部分和；用 logical shard 时，每个 rank 只求一棵固定的树里属于自己的那棵子树。</li>
<li data-hold="1.5">接着各 rank 把部分和加起来，得到 x₂；logical shard 则把同一棵固定的树求完，得到 x。</li>
<li data-hold="2">d = 4 时，每个 rank 负责 4 条序列。每个 rank 各做一次反向，现在有四个部分和，求的已经是另外的和了；用 logical shard 时还是同一棵树，只是每个 rank 分到的子树更小。</li>
<li data-hold="2">每个 rank 各做一次反向时，x₄ ≠ x₂，结果随并行度而变；用 logical shard 时树不变，x 的每个 bit 也不变。</li>
</ol>
<figcaption>图：每个 rank 各做一次反向时，d = 2 和 d = 4 下的部分和取决于并行度；用 logical shard 时，无论并行度多少，求的都是同一棵树。</figcaption>
</figure>

**规则。** 跨 rank 求和的顺序，只能取决于元素本身和 batch 的一个固定逻辑切分，绝不能取决于元素在消息里的位置、它归哪个 rank，或者并行度。这是第 0 部分的原则在 L5 上的体现：并行度可以改变由谁来算某个部分和，但不能改变有哪些部分和。

## 步骤

### 固定每一个跨 rank 求和的顺序

参数的梯度、梯度范数、loss 和更新后的权重，都不应受分桶方式、切分布局或并行度的影响。按你的运行需要选用下面的选项。代码在 [training_dp.py](../../examples/training_dp/training_dp.py) 里（有序的集合通信、DDP comm hook `Worker.ordered_hook`、logical shard 和逻辑范数），每一次比较都用 [bi_check.py](../../tools/bi_check.py) 逐 bit 进行。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| all-reduce 用 `NCCL_ALGO=Tree` | 并行度固定时，无论怎样分桶，每个参数得到的梯度都相同 | 大消息时小到中等，小消息时大 | 并行度固定的 DDP；最快的修法 |
| 有序 all-reduce（先 all-to-all，再在 fp32 里按 rank 顺序求和，最后 all-gather；以 DDP comm hook 的形式装上） | 同上，而且与 NCCL 选用的算法无关 | 单节点上小到中等 | 需要在任何 NCCL 设置下都成立 |
| 有序 reduce-scatter（rank r 接收每条消息的第 r 片，并按 rank 顺序把 d 个切片相加） | FSDP 式的梯度不受 padding、包装方式或重新切分的影响 | 单节点上中等（不到 2 倍）；跨节点时用两级路由 | 切分训练，或者布局有任何变化 |
| logical shard：S 个固定的 micro-batch，每个各做一次反向，再按 shard 编号用一棵两两相加的树求和，跨 rank 的求和也包含在这棵树里 | 任何并行度下，梯度、权重和 Adam 状态都完全相同；支持弹性恢复 | 每个 shard 含多条序列时中等；shard 很小时大 | 并行度会变，或者两个并行度下的运行必须一致 |
| 逻辑范数和 loss：固定分块，在 fp64 里求平方和，再用一棵两两相加的树；loss 也用一棵树对各 shard 的 loss 求和 | 裁剪用的范数和 loss 只取决于数值本身 | 范数本身中等；放在整个 step 里看很小 | 用了 logical shard 就一定要用 |
| single-tensor AdamW（`foreach=False, fused=False`） | 逐元素更新，所以复制的权重和切分的权重逐 bit 一致 | 小 | 总是要选；固定一条代码路径 |

logical shard 的前提是[训练 step](training-step.md) 已经固定好：每个 shard 的反向计算在每个 rank 上都必须能逐 bit 重复，而用原生算子时，这需要打开严格确定性模式。跨两个节点时，先在节点内部完成交换（两级路由）；在节点之间较慢的链路上直接做扁平的 all-to-all，代价要高得多。

做到了的标志是：在每个 rank 上，梯度、loss、范数、权重和 Adam 状态都和 d = 1 的运行逐 bit 一致；在另一个并行度下弹性恢复的运行，和一直没停的运行一致。

这一点在 fp32 梯度下得到了证明，最多用到两个节点上的 16 块 GPU，既测了复制权重的情形，也测了模拟的 FSDP（展平的梯度、reduce-scatter、切分的 AdamW；没有运行框架自带的 FSDP）。Tree 和有序集合通信消除了分桶的影响，有序 reduce-scatter 消除了切片归属的影响，logical shard 在测试过的每个并行度下、以及弹性恢复之后，都给出了完全相同的结果（[报告](../../examples/training_dp/REPORT.md#what-this-shows)）。

### 找出梯度求和中首个分歧点的原因

| 你观察到的现象 | 原因 |
|---|---|
| 并行度不变，但改了桶大小、参数顺序或模型之后，梯度变了 | ring all-reduce 的求和顺序随参数在消息里的偏移而变 |
| 第一次迭代，或者重启后的第一次迭代，和其余迭代不同 | DDP 的第一次迭代把所有梯度放在一个桶里 |
| 每个不同的元素，都换了归属的 rank | reduce-scatter 的求和顺序随归属 rank 而变 |
| 换一个并行度，无论用哪种集合通信，几乎每个元素都不同 | 每个 rank 上的反向计算本身就是算术的一部分，要用 logical shard |
| 裁剪范数只在某些放置方式下不同 | 各 shard 的平方和是由 NCCL 加起来的 |

## 可能出错的地方

!!! warning "DDP 连自己都不一致"
    DDP 构造之后的第一次迭代只用一个桶，而每次重启都会重新构造 DDP。所以在 Ring 下，光是重启一次就会改变梯度。要用 Tree 或有序 hook，调桶大小解决不了问题。

!!! warning "Tree 管不到 reduce-scatter"
    `NCCL_ALGO=Tree` 只改变 all-reduce，切分训练的求和顺序仍然随归属 rank 而变。FSDP 式切分要用有序 reduce-scatter。

!!! warning "又把跨 rank 的求和交还给 NCCL"
    如果树的最后一步是 NCCL 的 all-reduce 或 reduce-scatter，无论 Ring 还是 Tree，logical shard 从三个 rank 起就会失效，因为那时的树已经不是同一棵树了。每一个跨 rank 的求和，都必须是这棵固定的树的一部分。

!!! warning "侥幸一致的标量"
    几十项的和常常会舍入到同一个值，所以原生的切分范数在很多放置方式下都能碰巧一致。但它的结果仍然取决于切分方式，换一些放置方式就会失败。要用按块对齐的逻辑范数。

!!! warning "logical shard 得到的不是整批计算的梯度"
    d = 1 时用 logical shard 算出的梯度，和对同样的序列整批做一次反向得到的梯度并不相同。两者都是正确的舍入，只是算术不同。选其中一个作为参考，并且处处都用它。

## 小结

- 在 Ring 下，梯度的结果取决于它在消息里的位置；在 reduce-scatter 下，取决于它归哪个 rank。Tree 或有序集合通信解决前一个问题，有序 reduce-scatter 解决后一个。
- 换一个并行度，就是换了一组部分和，所以只有用一棵固定树的 logical shard 才能做到 L5 和弹性恢复。
- 范数和 loss 均值也是跨 rank 的求和，要让它们也用同一棵固定的树。
- 固定优化器的代码路径。

## 深入阅读

- [报告：数据并行与 FSDP 式切分](../../examples/training_dp/REPORT.md)：每一种打包、布局和放置方式，耗时，被否决的设计，以及局限。
- 代码在 [training_dp.py](../../examples/training_dp/training_dp.py)。
- [集合通信](../3-kernels/collectives.md)讲张量并行层里的同类集合通信。
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)，recipe 10。
