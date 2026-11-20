# MoE：路由、专家 GEMM 和 combine

## 我们在哪里

到[线性注意力](linear-attention.md)为止，每次归约的长度都由元素自身决定：GEMM 是 K，注意力
是位置。混合专家层加入了依赖数据的路由。一个专家 GEMM 看到哪些 token 随批变化，而每个 token
的输出是 k 个专家输出之和，这些输出以某种顺序到达。本页的专家部分建立在
[GEMM 页](gemm-and-quantized.md)上，路由部分借用[稀疏注意力](mla-and-sparse.md)的稳定 top-k，
然后加入专家并行，此时 token 要在 GPU 之间传递。

## 思路：token 的输出是 k 项之和，顺序由我们来定

一个 token 的输出是 y = w1·e1 + w2·e2 + … + wk·ek，每个被选中的专家一项。有三样东西会让这个和
依赖 token 以外的东西：

- **专家的批。** 专家 i 对路由给它的 M_i 个 token 做 GEMM，而 M_i 取决于邻居。用库 GEMM 时，
  该 token 的 e_i 会随邻居改变。
- **加法的顺序。** 用原子加法累加每个返回行的 combine，按到达顺序求和，而到达顺序跟随时序。
- **丢弃。** 容量上限保留最先到达专家的 token，所以这个 token 的项是否存在取决于它的邻居。

顺序从三项起才有影响。按三位有效数字舍入：

| 项 | 分组 | 结果 |
|---|---|---|
| a = 1.00, b = 0.004 | a + b 或 b + a | 都是 1.00 |
| a = 1.00, b = 0.004, c = 0.004 | (a + b) + c | 1.00 |
| a = 1.00, b = 0.004, c = 0.004 | a + (b + c) | 1.01 |

两项精确可交换，所以 top-2 模型会掩盖所有 combine 顺序的错误。

**在专家并行下**，专家分布在不同 GPU 上，两次 all-to-all 把行送出去再送回来。它们只搬运数据，
不会改变位。但先对每个 GPU 上的项求和、再把各 GPU 的部分和相加的 combine，会按哪些专家同在
一个 GPU 来分组：一个 GPU 上的 ((a + b) + c) + d，在两个 GPU 上可能变成 (a + c) + (b + d)。
于是分组取决于并行度（L5）。

**MoE 的规则。** 不变的路由器配稳定的 top-k，无丢弃路由，专家 GEMM 对所有 M 用同一配置，对每个
token 的 k 项按固定顺序（top-k 排名）用 fp32 求和，从固定槽位读取，不用原子操作。

## 步骤

### 1. 每个批都以相同方式路由

- **目标：** 一个 token 的专家和权重只取决于该 token。
- **方法：** 路由器 GEMM 走单配置 kernel；top-k 用按（分数降序，专家 id 升序）的稳定排序；
  权重在 fp32 中按 top-k 排名顺序重新归一化；没有容量上限。
- **成功标准：** 路由器的 id 和权重在各批和各并行度下逐位相等
  （[报告](../../examples/moe_ep/REPORT.md#under-expert-parallelism)）。

### 2. 让专家 GEMM 不变

- **目标：** e_i 不随 M_i 变化。
- **方法：** 每个专家一次单配置 GEMM 调用
  （[recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)）。
- **成功标准：** 每个专家形状上 `check_batch` 都通过。在两种 GPU 上得到证明
  （[报告](../../examples/moe_combine/REPORT.md#1-locate-the-variance)）。

### 3. 按固定顺序 combine

- **目标：** 一个 token 的 k 项之和只有一种顺序和一种精度。
- **方法：** 给每个返回行一个固定槽位（token，top-k 排名）；每个 token 一个 program，按排名
  顺序读取它的槽位，在 fp32 中累加，舍入一次。这就是
  [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py) 中的 `fixed_slot`；
  [recipe 8](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine)。
- **成功标准：** L1、L2 和 L5 通过。在一个 GPU 上，以及两个节点上最多 16 个 GPU 上得到证明
  （[报告](../../examples/moe_ep/REPORT.md#what-the-expert-parallel-results-show)）。

| Combine | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 原子 scatter-add | 从 top-3 起 L1 失败 | 免费 | 一致性场景下永远不选 |
| 对按专家排序的行做确定性 scatter-add | L1、L2；顺序来自分发器的排序，而不是代码 | 免费 | 仅当那个排序有保证时 |
| 每 GPU 先归约，再加各 GPU 部分和 | L1、L2；从 top-3 起 L5 失败 | 中等 | 并行度可能变化时不选 |
| 固定槽位，按排名的逐 token fp32 求和 | L1、L2、L5 | 免费到中等（专家并行下不比原子操作慢） | 默认修法 |
| 对固定行块循环调用库 GEMM | L1、L2 | 大 | 只作正确性 oracle |

### 4. 用三项或更多验证

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)：top-k 至少 3，
同批 token 放在同一 GPU、其他 GPU 和两者上，多种 token 到 GPU 的分片方式，以及你部署的每个
专家并行度，都与单 GPU 比较。

## 可能出错的地方

!!! warning "Top-2 会掩盖 combine 顺序的错误"
    两项精确可交换，所以原子或每 GPU 的 combine 在 top-2 下通过、从 top-3 起失败。top-2 的
    测试无法为更大的 k 证明一个 combine。

!!! warning "容量让 token 依赖它的邻居"
    有容量上限时，这个 token 旁边的 token 决定它的哪些分配能保留。测到的每个不一致都伴随着
    丢弃集合的改变；一致性场景要用无丢弃路由。

!!! warning "确定性不等于并行不变"
    每 GPU 先归约的做法每次运行位都相同，却随专家并行度变化。要单独测 L5。

!!! warning "确定性 scatter-add 继承了一个没有文档的顺序"
    它按行在缓冲区中的顺序求和，而换一个分发器就可能改变这个顺序。显式的逐 token 求和在代码里
    写明了顺序。

**尚未证明：** 专家并行下的反向；同一求和中的共享专家；融合的 dispatch-and-combine kernel；
与 trainer 的 MoE 的一致性（L4）。

## 小结

- 一个 token 的输出是 k 个专家项之和；自己固定顺序（top-k 排名）和精度（fp32）。
- 专家 GEMM 看到的 M 随批变化：用单配置 GEMM。
- 不用原子操作，不做容量丢弃，用稳定的 top-k。
- 专家并行下把行送回固定槽位；all-to-all 只搬运数据。
- 用至少为 3 的 top-k 测试。

## 深入阅读

- 报告：[单 GPU 上的 MoE](../../examples/moe_combine/REPORT.md)、
  [专家并行下的 MoE](../../examples/moe_ep/REPORT.md)。
- 代码：[examples/moe_combine/moe_bi.py](../../examples/moe_combine/moe_bi.py)、
  [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py)。
- Recipe：[3，分组 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)、
  [8，MoE 路由与 combine](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
