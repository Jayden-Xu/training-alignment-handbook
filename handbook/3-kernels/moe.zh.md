# MoE

## 我们在哪里

到[线性注意力](linear-attention.md)为止，每次归约的长度都由元素自身决定：GEMM 是 K，注意力是
query 的位置。MoE（mixture of experts）层多了一步依赖数据的路由。一个 expert 的 GEMM 会处理哪些
token，随 batch 而变；每个 token 的输出又是 k 个 expert 输出之和，而这些输出到达的先后并不固定。
本页的 expert 部分建立在 [GEMM 一页](gemm-and-quantized.md)上，路由部分借用
[稀疏注意力](mla-and-sparse.md)里的稳定 top-k，最后再加上 expert 并行，这时 token 要在 GPU 之间
来回传递。

## 思路：token 的输出是 k 项之和，加的顺序由你来定

一个 token 的输出是 y = w1·e1 + w2·e2 + … + wk·ek，每个被选中的 expert 贡献一项。有三样东西
会让这个和不只取决于 token 本身：

- **expert 看到的 batch。** expert i 对路由给它的 M_i 个 token 做 GEMM，而 M_i 取决于同一 batch
  里的其他 token。如果用库里的 GEMM，这个 token 的 e_i 就会跟着其他 token 变。
- **加法的顺序。** 如果 combine 对每个返回的行做原子加，求和顺序就是行到达的顺序，而到达顺序取决于时序。
- **丢弃。** 有 capacity 上限时，expert 只保留最先到达的那些 token，所以这个 token 的这一项还在不在，
  取决于其他 token。

顺序要到三项或更多时才有影响。把每一步的结果都舍入到三位有效数字来看：

| 项 | 分组 | 结果 |
|---|---|---|
| a = 1.00, b = 0.004 | a + b 或 b + a | 都是 1.00 |
| a = 1.00, b = 0.004, c = 0.004 | (a + b) + c | 1.00 |
| a = 1.00, b = 0.004, c = 0.004 | a + (b + c) | 1.01 |

两项交换顺序，结果完全不变，所以 top-2 的模型会把 combine 顺序上的所有错误都掩盖掉。

**在 expert 并行下**，各个 expert 分布在不同的 GPU 上，两次 all-to-all 把行送到 expert 那里，
再送回来。all-to-all 只搬运数据，不会改变任何 bit。但有一种 combine 先在每个 GPU 上把本地的项
加起来，再把各 GPU 的部分和相加，它就是按"哪些 expert 在同一个 GPU 上"来分组的：一个 GPU 上的
((a + b) + c) + d，换成两个 GPU 就可能变成 (a + c) + (b + d)。于是分组取决于并行度（L5）。

**MoE 的规则。** 路由器要不变，并使用稳定的 top-k；路由时不丢弃 token；expert 的 GEMM 对所有 M
用同一套配置。然后对每个 token 的 k 项按固定顺序（top-k 排名）用 fp32 求和，从固定的槽位读取，
不用原子操作。

## 步骤

### 1. 每个 token 在任何 batch 里都走同样的路由

路由器的 GEMM 用单配置 kernel；top-k 用稳定排序，先按分数降序、再按 expert id 升序；权重按 top-k 排名的
顺序在 fp32 下重新归一化；不设 capacity 上限。这样一个 token 选中哪些 expert、权重是多少，只取决于这个
token，在不同 batch 和不同并行度下都逐 bit 一致（[报告](../../examples/moe_ep/REPORT.md#under-expert-parallelism)）。

### 2. 让 expert 的 GEMM 不变

每个 expert 调用一次单配置 GEMM
（[recipe 3](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)），
让 e_i 不随 M_i 变化。每个 expert 形状上 `check_batch` 都要通过；这在两种 GPU 上得到了证明
（[报告](../../examples/moe_combine/REPORT.md#1-locate-the-variance)）。

### 3. 按固定顺序把每个 token 的 expert 输出加起来

给每个返回的行一个固定的槽位，用（token，top-k 排名）来索引。每个 token 由一个 program 负责，它按排名
顺序读取自己的槽位，在 fp32 下累加，最后只舍入一次，这样一个 token 的 k 项之和，顺序和精度都只有一种。
这就是 [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py) 里的 `fixed_slot`；做法见
[recipe 8](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine)。
L1、L2 和 L5 都要通过；这在单个 GPU 上，以及两个节点、最多 16 个 GPU 上得到了证明
（[报告](../../examples/moe_ep/REPORT.md#what-the-expert-parallel-results-show)）。

| combine 怎么加 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 原子 scatter-add | 从 top-3 起 L1 就不成立 | 免费 | 需要一致性时永远不选 |
| 对按 expert 排好序的行做确定性 scatter-add | L1、L2；顺序来自分发器的排序，而不是写在代码里 | 免费 | 只有当这个排序有保证时 |
| 每个 GPU 先归约，再把各 GPU 的部分和相加 | L1、L2；从 top-3 起 L5 不成立 | 中等 | 并行度可能变化时不选 |
| 固定槽位，每个 token 按排名顺序做 fp32 求和 | L1、L2、L5 | 免费到中等（在 expert 并行下不比原子操作慢） | 默认的修法 |
| 把行切成固定的块，循环调用库 GEMM | L1、L2 | 大 | 只作正确性 oracle |

### 4. 每个 token 至少用三项来验证

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)，top-k 至少取 3。
同一 batch 的其他 token 分别放在同一个 GPU、其他 GPU 以及两边都有的情况下测；试几种把 token 分片
到各 GPU 的方式；你要部署的每个 expert 并行度都要测，并且都和单 GPU 的结果比较。

## 可能出错的地方

!!! warning "top-2 会掩盖 combine 顺序的错误"
    两项交换顺序结果不变，所以原子加或按 GPU 先归约的 combine 在 top-2 下能通过，从 top-3 起
    就失败。top-2 的测试无法证明一个 combine 在更大的 k 下也正确。

!!! warning "capacity 上限让一个 token 依赖其他 token"
    有 capacity 上限时，同一 batch 里的其他 token 决定了这个 token 的哪些 expert 分配能留下来。我们测到的
    每一处不一致，都伴随着被丢弃的 token 集合发生了变化；需要一致性时，就用不丢弃 token 的路由。

!!! warning "确定性不等于对并行度不变"
    每个 GPU 先归约的做法每次运行结果都逐 bit 一致，但 expert 并行度一变，结果就变。要单独测 L5。

!!! warning "确定性 scatter-add 依赖的是一个没人写下来的顺序"
    它按行在缓冲区里的先后求和，换一个分发器，这个顺序就可能变。显式地对每个 token 求和，
    顺序就写在代码里。

**尚未证明：** expert 并行下的反向；求和里还包含 shared expert 的情况；融合的
dispatch-and-combine kernel；以及与 trainer 端 MoE 的一致性（L4）。

## 小结

- 一个 token 的输出是 k 个 expert 项之和；顺序（top-k 排名）和精度（fp32）要自己固定下来。
- 每个 expert 的 GEMM 看到的 M 随 batch 变化，所以要用单配置 GEMM。
- 不用原子操作，不因 capacity 丢弃 token，使用稳定的 top-k。
- 在 expert 并行下，把行送回固定的槽位；all-to-all 本身只搬运数据。
- 测试时 top-k 至少取 3。

## 深入阅读

- 报告：[单 GPU 上的 MoE](../../examples/moe_combine/REPORT.md)和
  [expert 并行下的 MoE](../../examples/moe_ep/REPORT.md)。
- 代码：[examples/moe_combine/moe_bi.py](../../examples/moe_combine/moe_bi.py)、
  [examples/moe_ep/moe_ep.py](../../examples/moe_ep/moe_ep.py)。
- Recipe：[3，分组 GEMM](../../skills/batch-invariant-kernel/references/operator-recipes.md#3-batched-and-grouped-gemm-moe-experts-bmm-vs-mm)、
  [8，MoE 路由与 combine](../../skills/batch-invariant-kernel/references/operator-recipes.md#8-moe-routing-dispatch-and-combine)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
