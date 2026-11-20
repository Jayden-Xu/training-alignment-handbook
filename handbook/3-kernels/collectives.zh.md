# 集合通信与张量并行层

## 我们在哪里

在 [MoE](moe.md) 中，跨 GPU 的通信只搬运行，求和由我们自己写。张量并行（TP）层不同：每个
GPU 算出一个部分乘积，再由集合通信在通信库内部把部分和相加。这个求和的顺序现在由库来选，
而不是由我们的代码；而且并行度本身决定了有哪些部分和。这使集合通信成为最难的一类：批（L2）
和并行度（L5）都会影响归约。

## 思路：库按消息来选择顺序

求和的 all-reduce 对每个元素在各 GPU 上一次加一项，每加一次舍入一次。用 ring 算法时，消息被
切成块，每块的求和从不同的 GPU 开始，沿环走一圈：

<figure class="fx" data-fig="ring" aria-label="4 个 GPU 上的 ring all-reduce，两个 chunk">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">位于 chunk 0 的元素</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node" data-hot="1">GPU 1</span><span class="fx-note fx-mono" data-at="1">g1</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="2">+ GPU 2</span><span class="fx-note fx-mono" data-at="2">g1 + g2</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="3">+ GPU 3</span><span class="fx-note fx-mono" data-at="3">(g1 + g2) + g3</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node to-key" data-hot="4 5">+ GPU 0</span><span class="fx-note fx-mono" data-at="4">(…) + g0</span></span></div></div>
<div class="fx-row"><span class="fx-label">位于 chunk 1 的元素</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node" data-hot="1">GPU 2</span><span class="fx-note fx-mono" data-at="1">g2</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="2">+ GPU 3</span><span class="fx-note fx-mono" data-at="2">g2 + g3</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node" data-hot="3">+ GPU 0</span><span class="fx-note fx-mono" data-at="3">(g2 + g3) + g0</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node to-key" data-hot="4 5">+ GPU 1</span><span class="fx-note fx-mono" data-at="4">(…) + g1</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>4 个 GPU 各持有每个元素的一个值 g。ring 把消息切成 chunk。</li>
<li>每个 chunk 的和从不同的 GPU 开始：chunk 0 从 GPU 1 开始，chunk 1 从 GPU 2 开始。</li>
<li>每一步加上下一个 GPU 的值并舍入，</li>
<li>再把部分和沿 ring 传下去。</li>
<li data-hold="1.5">加 3 次之后，每个 chunk 都有了自己的和，各按自己的顺序。</li>
<li data-hold="2">元素落在哪个 chunk 取决于它的偏移和消息大小：batch 变大会把它挪走，它的位也跟着变。</li>
</ol>
<figcaption>图：4 个 GPU 上的 ring all-reduce：每个 chunk 的和从不同的 GPU 开始。</figcaption>
</figure>

一个元素落在哪一块，取决于它的偏移和消息大小。在 TP 层中，消息是整个批的输出，所以当批变大
或行移动时，这一行的求和会从另一个 GPU 开始（L2 失败）。其他算法的表现不同：

| 算法 | 一个元素求和的顺序 | 依赖批吗？ |
|---|---|---|
| Ring all-reduce | 从由元素所在块决定的 GPU 开始 | 是 |
| 单节点上的 Tree all-reduce | 一条链，对每个元素都相同 | 否 |
| Ring reduce-scatter | 每个输出块固定，由拥有它的 GPU 决定 | 是，当行的拥有者随批变化时 |
| 任意算法，两个 GPU | x0 + x1 = x1 + x0 精确成立 | 否 |

**并行度是第二个依赖。** row-parallel 层先把每个 GPU 的部分乘积舍入，再把部分和相加。在 2 个
GPU 上，部分和是 K 的两半；在 4 个 GPU 上是四分之一。部分和不同，总和就不同，所以没有任何库
设置能给出 L5。

**集合通信的规则。** 固定跨 GPU 求和的顺序，使它不依赖消息（固定一个 tree，或交换部分和后自己
按固定顺序求和）；为了 L5，还要固定部分和本身：逻辑 TP 大小 C 把 K 切成 C 个固定块，不论由
多少 GPU 持有，这些块都按同一棵固定的树求和。

## 步骤

### 1. 定位变化来源

- **目标：** 知道哪些集合通信和设置会改变一行的位。
- **方法：** 把一个固定切片单独归约，再放进多种大小的消息、多个位置中归约，并从库的 tuning
  日志记录每次调用的选择；
  [examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py)
  做的就是这件事，还带一个解释每个元素求和顺序的诊断。
- **成功标准：** 每个失败元素都能用一个已知顺序解释
  （[报告](../../examples/collectives_tp/REPORT.md#what-this-shows)）。

### 2. 固定求和顺序（L2）

- **目标：** 一行的跨 GPU 求和在任何批中都相同。
- **方法：** [recipe 10](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp)。
- **成功标准：** TP 层通过 `check_batch`。在一个 8 GPU 节点和两个 8 GPU 节点上得到证明
  （[报告](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)）。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 库默认 | L2 失败 | 免费 | 一致性场景下永远不选 |
| 固定 Ring，任意协议 | L2 失败 | 免费 | 永远不选：固定算法并没有固定分块 |
| 固定 Tree（`NCCL_ALGO=Tree`） | 一个和两个节点上 all-reduce 的 L2 | 小到中等；某些消息大小下大 | 只靠配置，最多两个节点；覆盖不到 reduce-scatter |
| All-gather，然后按 GPU 顺序做本地 fp32 求和 | L2 | 大 | 小消息，或作为参考 |
| All-to-all，按 GPU 顺序做本地 fp32 求和，再 all-gather | L2；前半部分可替代 reduce-scatter | 单节点上小到中等 | 单节点上的默认回退 |
| 两级固定顺序：先节点内，再跨节点交换一次 | L2 | 中等 | 跨节点 |

回退方案改变了算术（fp32 累加、一次舍入），所以为了 L4，两侧必须用同一种。

### 3. 固定部分和（L5）

- **目标：** 同样的行在 TP 1、2、4 及更大时给出相同的位。
- **方法：** 逻辑 TP 大小 C：K 被切成 C 个固定块，每个 GPU 以 fp32 输出计算整块，部分和按一棵
  固定的两两求和树相加。整除 C 的并行度让每个 GPU 拿到整块；大于 C 时，拆分每块的输出列，
  绝不拆它的 K 范围。
- **成功标准：** 每个并行度下只有一个结果。在两个节点上最多 16 个 GPU 得到证明
  （[报告](../../examples/collectives_tp/REPORT.md#what-the-second-node-changes)）。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| row-parallel 配任意固定顺序求和 | L2；随并行度变化 | 小到中等 | 各次运行之间并行度从不改变 |
| column-parallel（只有 all-gather） | L5：没有跨 GPU 的求和 | 免费 | 可以按列拆分的层 |
| 逻辑 TP 大小，固定的树 | 在整除 C 的每个并行度上 L5，大于 C 时靠列拆分 | 中等到大 | trainer 和引擎运行不同的并行度 |

### 4. 序列并行

reduce-scatter 让每一行采用拥有它的 GPU 的顺序，而在序列并行下，拥有者跟随行的位置。把它换成
回退方案的 all-to-all 那一半；固定的 tree 覆盖不到它。

### 5. 用三个或更多 GPU 验证

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)的并行各轴：三个或
更多 GPU，固定切片放进多种大小和位置的消息，你部署的每个并行度，以及你可能遇到的库设置。

## 可能出错的地方

!!! warning "固定 ring 不等于不变"
    在固定 ring 下，每种协议仍然让每个元素的求和从由它所在块决定的 GPU 开始。要在多种消息大小
    中检查一个固定切片。

!!! warning "固定的 tree 覆盖不到 reduce-scatter"
    库对 reduce-scatter 继续使用 ring。带序列并行的层仍然失败；改用 all-to-all 回退。

!!! warning "两个 GPU 会掩盖所有顺序错误"
    两项求和可交换。TP-2 的测试对任何顺序都通过；要用三个或更多 GPU 测试。

!!! warning "固定的 tree 只在最多两个节点上得到证明"
    两个节点时，顶层只加两个操作数，可交换。三个或更多节点时，各 channel 的树可能不同；尚未证明。

!!! warning "两个逻辑大小就是两种算术"
    逻辑 TP 8 和 16 彼此不同，也都与普通 GEMM 不同。整个部署要固定一个 C。

## 小结

- 通信库按消息选择跨 GPU 求和的顺序；ring 的顺序跟随元素所在的块。
- 为了 L2：固定一个 tree（最多两个节点，只限 all-reduce），或交换部分和后自己按固定顺序求和。
- 为了 L5：用逻辑 TP 大小和一棵固定的树固定部分和。
- 用三个或更多 GPU、多种消息大小来测试。

## 深入阅读

- 报告：[集合通信与 TP 线性层](../../examples/collectives_tp/REPORT.md)。
- 代码：[examples/collectives_tp/collectives_tp.py](../../examples/collectives_tp/collectives_tp.py)。
- Recipe：[10，集合通信](../../skills/batch-invariant-kernel/references/operator-recipes.md#10-collectives-tp--ep--sp)。
- 相关：[数据并行梯度](../1-pretraining/data-parallel.md)，以及多个 TP 并行度下的整个模型
  （[并行](../2-rl/parallelism.md)）。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
