# Softmax 注意力：前向、反向和服务层

## 我们在哪里

[GEMM 页](gemm-and-quantized.md)通过固定 K 的走法，让每个元素的点积不变。注意力保留这些
GEMM，再加一次归约：每个 query 带着一个运行中的 softmax，对所有更早的 key 求和。这次归约在
每种服务模式下（prefill、decode、分块 prefill）、在分页缓存上、在打包的批中、在编译器下走法都
不同。本页先固定对 key 的走法，再处理反向，最后处理 kernel 外围的服务层。

## 思路：每种模式都用同样的 block 走 key

一个 query 的输出是对 key 的加权和。kernel 按 block 访问 key，并维护运行中的最大值和总和
（online softmax），所以 block 边界就是这次归约的分组。边界一动，位就动。

<figure class="fig">
<div class="fig-rows">
<div class="fig-group"><span class="fig-label">从位置 0 起的固定分块</span><div class="fig-chain"><span class="fig-node">key 0–63</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">key 64–127</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">key 128–t</span></div></div>
<div class="fig-group"><span class="fig-label">split-KV decode：切分由 batch 决定</span><div class="fig-chain"><div class="fig-stack"><span class="fig-node">key 0–t/2</span><span class="fig-node">key t/2–t</span></div><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-bad">合并两个部分结果</span></div></div>
</div>
</figure>

有三样东西会移动边界：

- **模式。** prefill、decode 和分块 prefill 常常运行不同的 kernel；decode kernel 还可能把
  key 拆成几段，段数由批大小和 GPU 核心数决定（split-KV）。
- **布局。** 让每个 block 等于一页的分页 kernel，把分组绑到页大小上；在整个打包上铺 block 的
  kernel，把分组绑到邻居的长度上。
- **后端。** 库调用可能按参数挑选后端，所以传入 mask 而不是 causal 标志就会换 kernel。

**注意力的规则。** 每个 query 都按固定大小的 block 走 key，从它所在序列的绝对位置 0 开始计，
不用 split-KV（或只用固定大小的拆分），而且每种模式都运行同一个 kernel，只传入 query 的绝对
偏移。超过 query 位置的 block 被完全屏蔽，是精确的空操作：重缩放因子恰好是 1，加上的权重恰好
是 0。因此短的 decode 调用和长的 prefill 算出相同的和。

**反向沿另一个方向归约。** 前向中每个输出行由一个 program 负责。反向中，一个 key 的梯度要对
所有更晚的 query 求和，一个 query 的梯度要对所有更早的 key 求和。常见 kernel 用原子操作把这些
份额加起来，于是顺序跟随时序，连 L1 都会失败。修法是两遍计算：每个 program 负责一个输出，
按固定顺序走完它的和。

## 步骤

### 1. 前向：每种模式都用固定的 key block

- **目标：** prefill、decode 和分块 prefill 在每个位置给出相同的位，一个序列在任何批中给出
  相同的位（L2、L3）。
- **方法：** [tools/bi_kernels.py](../../tools/bi_kernels.py) 中的固定 block kernel
  `attention_fixed_blocks`，遵循
  [recipe 4](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv)。
  在 kernel 运行前把新的 key 和 value 写入缓存，这样 decode 读取它们的方式与 prefill 相同。
- **成功标准：** [tools/bi_check.py](../../tools/bi_check.py) 中的 `check_modes` 和
  `check_batch` 通过。在一种 GPU 上得到证明，原生后端在那里两项都失败
  （[报告](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)）。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 原生后端 | L1；L2 和 L3 失败 | 免费 | 一致性场景下永远不选 |
| 固定 block kernel，每种模式同一个 | L2、L3 | prefill 中等（未调优）；decode 不更慢 | 默认修法 |
| 长 decode 用固定大小的 split-KV | 若拆分大小以 token 计固定，则 L2、L3 | 小 | 长上下文；这里尚未证明 |

### 2. 反向：两遍，无原子操作

- **目标：** 同一序列单独、在更大的批中、右填充时都得到相同的梯度（L1、L2）。
- **方法：** [tools/bi_kernels.py](../../tools/bi_kernels.py) 中无原子操作的反向
  `attention_fixed_blocks_train`。第一遍让每个 program 负责一个 key block；它按固定顺序对组内
  的 query 头和更晚的 query 求和，只写一次 dK 和 dV。第二遍让每个 program 负责一个 query
  block；它像前向一样从位置 0 走 key，只写一次 dQ。
- **成功标准：** 重复运行下 L1 成立；单独、成批和填充时 L2 成立，填充部分的输出梯度为零。
  在一种 GPU 上得到证明（[报告](../../examples/attention_backward/REPORT.md#results)）。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 库的默认反向 | L1 失败（原子操作） | 免费 | 一致性场景下永远不选 |
| 库的反向，严格确定性模式 | 只有 L1：它的前向已经随批变化 | 中等到大 | 只需要 L1 时 |
| 库的 memory-efficient 后端，严格确定性 | 在测过的形状上 L1 和 L2 | 大 | 只靠配置；推理引擎不运行它，所以没有 L4 |
| 无原子操作的两遍 kernel | L1、L2，含右填充 | 中等 | 默认修法 |

这里右填充是精确的：因果性让填充不进入真实 token 的前向，填充行的输出梯度为零，所以它们的
贡献是精确的零。左填充会移动绝对位置，因此不在覆盖范围内。

### 3. 服务层：分页、打包、编译器、graph

- **目标：** 引擎在 kernel 外围加的各层都保持位不变。
- **方法：** [examples/attention_serving](../../examples/attention_serving/serving_kernels.py)
  中的分页和 varlen kernel 及其检查；编译器和 graph 见
  [recipe 12](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs)。
- **成功标准：** 每种页大小、打包偏移、编译和 graph 变体都给出连续 kernel 的位。在一种 GPU
  上得到证明（[报告](../../examples/attention_serving/REPORT.md#what-the-serving-layer-results-show)）。

| 层 | 做法 | 代价等级 |
|---|---|---|
| 分页 KV | 计算 block 留在绝对位置上；无论页大小，都把页收集进 block | 小到中等，取决于页大小和流水 |
| 打包的变长批 | 每个序列的 block 从它自己的第一个 token 开始，而不是落在打包的网格上 | 小 |
| `torch.compile` | 把每个会舍入的算子（norm、激活，不只是 kernel）注册为不透明 custom op，使编译器无法移动舍入点 | 在 CUDA graph 下小；小批 eager 下主机开销大 |
| CUDA graph | 使用不变的 kernel；填充后的批以 bucket 大小运行，位相同 | 免费 |

### 4. 验证

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)，加上注意力的
各轴：不整除 block 的分块大小下的各模式，随机放页的多种页大小，不是 block 倍数的打包偏移，
编译对比 eager，以及带填充的反向。

## 可能出错的地方

!!! warning "只警告的确定性模式并不确定"
    在 `warn_only=True` 下，库先警告，然后照样运行非确定性的反向。梯度仍逐次变化；要用
    `warn_only=False`，它会对任何没有确定性版本的算子报错。

!!! warning "反向不可能比它的前向更不变"
    在随批变化的前向上做确定性反向，结果仍随批变化。先修前向。

!!! warning "单一页大小总是自洽"
    block 等于页的 kernel，在任一页大小下跨模式自洽，而每种页大小给出不同的答案。要测多种。

!!! warning "打包偏移恰为 block 倍数时会侥幸通过"
    落在打包网格上的 kernel，恰在序列从 block 边界开始时才正确。要测不是 block 倍数的偏移。

!!! warning "编译器会移动舍入点"
    编译后的 norm 和激活保持批不变，但与 eager 不同。如果引擎要对齐 eager 的 trainer，就把
    每个会舍入的算子都做成不透明的。

## 小结

- 注意力对 key 归约；block 边界就是分组。
- 从绝对位置 0 起的固定 block，不用 split-KV，每种模式同一 kernel。
- 反向需要两遍无原子操作的计算；只开严格确定性模式只给出 L1，不给 L2。
- 分页、打包、编译器和 graph 只有在计算 block 不理会它们时才保持位不变。

## 深入阅读

- 报告：[kernel 案例](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)
  （前向）、[注意力反向](../../examples/attention_backward/REPORT.md)、
  [服务层](../../examples/attention_serving/REPORT.md)。
- 代码：[tools/bi_kernels.py](../../tools/bi_kernels.py)、
  [examples/attention_backward/attn_bwd.py](../../examples/attention_backward/attn_bwd.py)、
  [examples/attention_serving/serving_cases.py](../../examples/attention_serving/serving_cases.py)。
- Recipe：[4，softmax 注意力](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv)、
  [12，部署包装层](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
