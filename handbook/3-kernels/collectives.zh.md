# 集合通信与张量并行层

## 我们在哪里

在 [MoE 一页](moe.md)里，GPU 之间的通信只是搬运行，求和是我们自己写的。张量并行（TP）层不一样：
每个 GPU 先算出一个部分乘积，再由集合通信在通信库内部把这些部分乘积加起来。这次求和的顺序由库
来定，而不是由我们的代码来定；并行度本身还决定了有哪些部分和。所以集合通信是最难的一类：
batch（L2）和并行度（L5）都会影响这次归约。

## 思路：通信库根据消息来决定求和的顺序

做求和的 all-reduce 对每个元素在各个 GPU 之间一项一项地加，每加一次就舍入一次。用 ring 算法时，
库把消息切成若干 chunk，每个 chunk 的求和从不同的 GPU 开始，沿着环走一圈：

<figure class="fx" data-fig="ring" aria-label="4 个 GPU 上的 ring all-reduce，消息分成两个 chunk">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">位于 chunk 0 的元素</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node" data-hot="1">GPU 1</span><span class="fx-note fx-mono" data-at="1">g1</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="2">+ GPU 2</span><span class="fx-note fx-mono" data-at="2">g1 + g2</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="3">+ GPU 3</span><span class="fx-note fx-mono" data-at="3">(g1 + g2) + g3</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node to-key" data-hot="4 5">+ GPU 0</span><span class="fx-note fx-mono" data-at="4">(…) + g0</span></span></div></div>
<div class="fx-row"><span class="fx-label">位于 chunk 1 的元素</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node" data-hot="1">GPU 2</span><span class="fx-note fx-mono" data-at="1">g2</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="2">+ GPU 3</span><span class="fx-note fx-mono" data-at="2">g2 + g3</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="3">+ GPU 0</span><span class="fx-note fx-mono" data-at="3">(g2 + g3) + g0</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node to-key" data-hot="4 5">+ GPU 1</span><span class="fx-note fx-mono" data-at="4">(…) + g1</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>4 个 GPU 各自持有每个元素的一个值 g，ring 把消息切成若干 chunk。</li>
<li>每个 chunk 的求和从不同的 GPU 开始：chunk 0 从 GPU 1 开始，chunk 1 从 GPU 2 开始。</li>
<li>每一步由下一个 GPU 加上自己的值并舍入，</li>
<li>再把部分和沿着环传下去。</li>
<li data-hold="1.5">加了 3 次之后，每个 chunk 都得到了自己的和，但各自的加法顺序不同。</li>
<li data-hold="2">元素落在哪个 chunk，取决于它的偏移和消息大小。batch 一变大，它就可能被挪到别的 chunk，结果也跟着变。</li>
</ol>
<figcaption>图：4 个 GPU 上的 ring all-reduce。每个 chunk 的求和从不同的 GPU 开始。</figcaption>
</figure>

一个元素落在哪个 chunk，取决于它的偏移和消息大小。在 TP 层里，消息就是整个 batch 的输出，所以
batch 变大或者这一行换了位置时，这一行的求和就会从另一个 GPU 开始，L2 也就不成立了。其他算法
的表现各不相同：

| 算法 | 一个元素按什么顺序求和 | 依赖 batch 吗？ |
|---|---|---|
| Ring all-reduce | 从元素所在 chunk 决定的那个 GPU 开始 | 是 |
| 单节点上的 Tree all-reduce | 一条固定的链，对每个元素都一样 | 否 |
| Ring reduce-scatter | 每个输出 chunk 的顺序固定，由持有这个 chunk 的 GPU 决定 | 是，当持有某一行的 GPU 随 batch 变化时 |
| 任何算法，只有两个 GPU | x0 + x1 = x1 + x0 精确成立 | 否 |

**并行度是第二个依赖。** 行并行层先在每个 GPU 上把部分乘积舍入，再把各个部分和加起来。
2 个 GPU 时，部分和各覆盖 K 的一半；4 个 GPU 时，各覆盖四分之一。部分和不同，总和就不同，所以
库里没有任何设置能给出 L5。

**集合通信的规则。** 固定跨 GPU 求和的顺序，让它不依赖消息：要么固定用 tree 算法，要么把部分和
交换过来，自己按固定顺序相加。要达到 L5，还要把部分和本身固定下来：用一个逻辑 TP 规模 C 把 K
切成 C 个固定的块，不管这些块由多少个 GPU 持有，都按同一棵固定的树相加。

## 步骤

### 1. 找出哪些集合通信和设置会改变一行的结果

- **目标：** 知道哪些集合通信、哪些设置会改变一行的结果。
- **做法：** 把一个固定的切片单独归约一次，再把它放进多种大小的消息、放在多个位置上归约，并从库的
  tuning 日志里记下每次调用选了什么。
  [examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py)
  做的就是这件事，它还带一个诊断工具，能解释每个元素的和是按什么顺序加出来的。
- **成功标准：** 每个失败的元素都能用一个已知的顺序解释
  （[报告](../../examples/collectives_tp/REPORT.md#what-this-shows)）。

### 2. 固定跨 GPU 求和的顺序（L2）

- **目标：** 一行的跨 GPU 求和，在任何 batch 里都相同。
- **做法：** 按 [recipe 10](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp) 来做。
- **成功标准：** TP 层通过 `check_batch`。这在一个 8 GPU 节点和两个 8 GPU 节点上得到了证明
  （[报告](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)）。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 库的默认设置 | L2 不成立 | 免费 | 需要一致性时永远不选 |
| 固定用 ring 算法，协议任意 | L2 不成立 | 免费 | 永远不选：固定算法并没有固定分块方式 |
| 固定用 tree 算法（`NCCL_ALGO=Tree`） | 单节点和两个节点上的 all-reduce 满足 L2 | 小到中等，某些消息大小下大 | 只能改设置、最多两个节点时；它管不到 reduce-scatter |
| 先 all-gather，再按 GPU 顺序在本地用 fp32 求和 | L2 | 大 | 消息很小时，或者作为参考 |
| 先 all-to-all，按 GPU 顺序在本地用 fp32 求和，再 all-gather | L2；前一半可以替代 reduce-scatter | 单节点上小到中等 | 单节点上默认的回退方案 |
| 两级固定顺序：先在节点内求和，再跨节点交换一次 | L2 | 中等 | 跨节点时 |

这些回退方案改变了算术（用 fp32 累加，只舍入一次），所以要达到 L4，两端必须用同一种回退方案。

### 3. 把部分和本身固定下来（L5）

- **目标：** 同样的行在 TP 为 1、2、4 或更大时，结果都逐 bit 一致。
- **做法：** 选一个逻辑 TP 规模 C。K 被切成 C 个固定的块，每个 GPU 计算若干个完整的块，输出用
  fp32，部分和按一棵固定的两两求和树相加。并行度能整除 C 时，每个 GPU 分到的都是完整的块；
  并行度大于 C 时，把每块的输出列拆给多个 GPU，但绝不拆它在 K 上的范围。
- **成功标准：** 每个并行度下都得到同一个结果。这在两个节点、最多 16 个 GPU 上得到了证明
  （[报告](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)）。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 行并行配任意固定顺序的求和 | L2，但结果随并行度变化 | 小到中等 | 各次运行之间并行度从不改变 |
| 列并行（只用 all-gather） | L5，因为没有跨 GPU 的求和 | 免费 | 可以按列拆分的层 |
| 逻辑 TP 规模配固定的树 | 在能整除 C 的每个并行度上满足 L5，大于 C 时靠拆列满足 | 中等到大 | trainer 和引擎用不同的并行度 |

### 4. 在序列并行下换掉 reduce-scatter

reduce-scatter 让每一行按持有它的那个 GPU 的顺序求和，而在序列并行下，由哪个 GPU 持有取决于这一行
的位置。把它换成上面回退方案里 all-to-all 的那一半；固定 tree 算法对 reduce-scatter 不起作用。

### 5. 用三个或更多 GPU 验证

沿[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)里的并行各维度运行：
用三个或更多 GPU；把一个固定切片放进多种大小、多个位置的消息里；测你要部署的每个并行度；再测
生产环境里可能遇到的各种库设置。

## 可能出错的地方

!!! warning "固定 ring 算法并不能带来不变性"
    固定 ring 之后，每种协议仍然让每个元素的求和从它所在 chunk 决定的那个 GPU 开始。要把一个固定
    切片放进多种大小的消息里检查。

!!! warning "固定 tree 算法管不到 reduce-scatter"
    库在 reduce-scatter 上仍然用 ring，所以带序列并行的层还是会失败。改用 all-to-all 的回退方案。

!!! warning "两个 GPU 会掩盖所有顺序错误"
    两项相加和顺序无关，所以 TP 2 的测试不管什么顺序都能通过。要用三个或更多 GPU 来测。

!!! warning "固定 tree 算法只在最多两个节点上得到了证明"
    两个节点时，顶层只把两个操作数相加，和顺序无关。三个或更多节点时，各个 channel 的树可能不同；
    这一点尚未证明。

!!! warning "两个逻辑规模就是两种算术"
    逻辑 TP 8 和逻辑 TP 16 的结果彼此不同，也都和普通 GEMM 不同。整个部署要固定用一个 C。

## 小结

- 通信库根据消息决定跨 GPU 求和的顺序；用 ring 时，顺序跟着元素所在的 chunk 走。
- 要达到 L2：固定用 tree 算法（最多两个节点，只限 all-reduce），或者把部分和交换过来，自己按
  固定顺序相加。
- 要达到 L5：用逻辑 TP 规模和一棵固定的树，把部分和本身固定下来。
- 用三个或更多 GPU、多种消息大小来测试。

## 深入阅读

- 报告：[集合通信与 TP 线性层](../../examples/collectives_tp/REPORT.md)。
- 代码：[examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py)。
- Recipe：[10，集合通信](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp)。
- 相关：[数据并行梯度](../1-pretraining/data-parallel.md)，以及在多个 TP 规模下运行的整个模型
  （[并行](../2-rl/parallelism.md)）。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
