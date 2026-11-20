# Softmax 注意力：前向、反向，以及引擎在 kernel 外面加的各层

## 我们在哪里

[GEMM 一页](gemm-and-quantized.md)通过固定沿 K 的走法，让每个元素的点积不再随 batch 变化。注意力
保留了这些 GEMM，又多了一次归约：每个 query 要对它之前的所有 key 求和，同时维护一个滚动的
softmax。引擎在不同的推理模式（prefill、decode、chunked prefill）下、在 paged KV cache 上、在打包的
batch 里、在 torch.compile 之下，走这次归约的方式都可能不同。本页先固定对 key 的遍历，再处理反向，最后处理
引擎包在 kernel 外面的各个推理服务层。

## 思路：每种模式都按同样的 block 遍历 key

一个 query 的输出是对所有 key 的加权和。kernel 按 block 访问 key，同时维护滚动的最大值和滚动的和
（online softmax），所以 block 的边界决定了这次归约怎样分组。边界一动，结果就不再逐 bit 一致。

<figure class="fx" data-fig="kv-split" aria-label="一个 query 的 key：按固定的 block 遍历，和按 split-KV decode 遍历">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">从位置 0 起数的固定 block</span><div class="fx-tree" style="--n:10">
<div class="fx-level"><span class="fx-cell" style="--w:4" data-hot="1">key 0–63</span><span class="fx-cell" style="--w:4" data-hot="2">key 64–127</span><span class="fx-cell" style="--w:2" data-hot="3">128–t</span></div>
<div class="fx-level" data-at="1"><span class="fx-walk" style="--w:10">一遍走完，同时维护滚动的最大值和总和</span></div>
<div class="fx-level" data-at="3"><span class="fx-j" style="--w:10;--k:1"><span class="fx-node is-ok">每种模式算出的和都一样</span></span></div>
</div></div>
<div class="fx-row"><span class="fx-label">split-KV decode：切成几份由 batch 决定</span><div class="fx-tree" style="--n:10">
<div class="fx-swap is-block"><div class="fx-level" data-to="6"><span class="fx-cell" style="--w:5" data-hot="4">key 0–t/2</span><span class="fx-cell" style="--w:5" data-hot="4">key t/2–t</span></div><div class="fx-level" data-at="6"><span class="fx-cell" style="--w:3">0–t/3</span><span class="fx-cell" style="--w:3">t/3–2t/3</span><span class="fx-cell" style="--w:4">2t/3–t</span></div></div>
<div class="fx-swap is-block"><div class="fx-level" data-at="5" data-to="6"><span class="fx-j" style="--w:10"><span class="fx-node is-bad">合并两个部分结果</span></span></div><div class="fx-level" data-at="6"><span class="fx-j" style="--w:10;--k:3"><span class="fx-node is-bad">合并三个部分结果</span></span></div></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>位置 t 上的一个 query 对 key 0 到 t 求和，同时维护滚动的最大值和总和。</li>
<li>按从位置 0 起数的固定 block，它先访问 key 0–63，</li>
<li>然后是 key 64–127，</li>
<li data-hold="1.5">然后是 key 128–t。prefill、decode 和 chunked prefill 都按这几个 block 走。</li>
<li data-hold="1.5">在 split-KV decode 里，batch size 和核心数决定切成几份。这里切成两份，同时计算。</li>
<li data-hold="1.5">两个部分结果再合并起来，这是同一个和的另一种分组。</li>
<li data-hold="2">换一个 batch，切成了三份：边界移动了，结果也跟着变。</li>
</ol>
<figcaption>图：block 的边界决定了归约怎样分组。边界一动，结果就变。</figcaption>
</figure>

有三样东西会移动这些边界：

- **模式。** prefill、decode 和 chunked prefill 常常运行不同的 kernel。decode kernel 还可能把
  key 拆成几段，段数由 batch size 和 GPU 核心数决定（split-KV）。
- **布局。** 如果 paged kernel 让每个 block 恰好等于一页，分组就和页大小绑在了一起。如果处理打包
  batch 的 kernel 在整个打包上铺 block，分组就和相邻序列的长度绑在了一起。
- **后端。** 库函数可能根据参数挑选后端，所以传入一个 mask 而不是 causal 标志，实际运行的 kernel
  就可能换掉。

**注意力的规则。** 每个 query 都按固定大小的 block 遍历 key，从它所在序列的绝对位置 0 开始数。
不用 split-KV，或者只用固定大小的切分；每种模式都运行同一个 kernel，只是传入 query 的绝对偏移。
位于 query 之后的 block 被完全屏蔽，不改变任何东西：它的重缩放因子恰好是 1，加上的权重恰好是 0。
所以一次短的 decode 调用和一次长的 prefill 算出的和完全相同。

**反向沿另一个方向归约。** 前向里，每个输出行由一个 program 负责。反向里，一个 key 的梯度是对它
之后所有 query 的求和，一个 query 的梯度是对它之前所有 key 的求和。常见的 kernel 用原子操作把
这些贡献加起来，于是加法顺序取决于时序，连 L1 都保不住。修法是分成两遍计算：每一遍里，每个
program 只负责一个输出，并按固定顺序完成它的求和。

## 步骤

### 1. 前向在每种模式下都按同样的 block 遍历 key

- **目标：** prefill、decode 和 chunked prefill 在每个位置上的结果逐 bit 一致，一个序列放在任何 batch 里，
  结果也逐 bit 一致（L2、L3）。
- **做法：** 使用 [tools/bi_kernels.py](../../tools/bi_kernels.py) 里的固定 block kernel
  `attention_fixed_blocks`，它按
  [recipe 4](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv)
  实现。在 kernel 运行之前先把新的 key 和 value 写进 cache，这样 decode 读取它们的方式就和
  prefill 一样。
- **成功标准：** [tools/bi_check.py](../../tools/bi_check.py) 里的 `check_modes` 和
  `check_batch` 都通过。这在一种 GPU 上得到了证明，而原生后端在那里两项检查都没通过
  （[报告](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)）。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 原生后端 | L1，但 L2 和 L3 不成立 | 免费 | 需要一致性时永远不选 |
| 固定 block 的 kernel，每种模式用同一个 | L2、L3 | prefill 中等（未调优）；decode 不变慢 | 默认的修法 |
| 长 decode 用固定大小的 split-KV | 如果切分大小按 token 数固定，就满足 L2 和 L3 | 小 | 长上下文；这里尚未证明 |

### 2. 反向分两遍计算，不用原子操作

- **目标：** 同一个序列单独运行、放在更大的 batch 里，还是在右边做了 padding，得到的梯度都逐 bit 一致（L1、L2）。
- **做法：** 使用 [tools/bi_kernels.py](../../tools/bi_kernels.py) 里不用原子操作的反向
  `attention_fixed_blocks_train`。第一遍给每个 program 分一个 key block，它按固定顺序对本组的
  各个 query 头和之后的所有 query 求和，dK 和 dV 各只写一次。第二遍给每个 program 分一个
  query block，它像前向那样从位置 0 开始遍历 key，dQ 只写一次。
- **成功标准：** 重复运行时 L1 成立；单独运行、放进 batch 运行和带 padding 运行时 L2 都成立，padding 部分的
  输出梯度为零。这在一种 GPU 上得到了证明（[报告](../../examples/attention_backward/REPORT.md#results)）。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 库的反向，默认设置 | 因为原子操作，L1 不成立 | 免费 | 需要一致性时永远不选 |
| 库的反向，严格确定性模式 | 只有 L1，因为它的前向已经随 batch 变化 | 中等到大 | 只需要 L1 时 |
| 库里节省显存的后端，严格确定性模式 | 在测过的形状上满足 L1 和 L2 | 大 | 只能改设置时；推理引擎不运行这个后端，所以得不到 L4 |
| 不用原子操作的两遍 kernel | L1、L2，包括右 padding | 中等 | 默认的修法 |

这里对右 padding 的处理是精确的。因果性保证 padding 不会进入真实 token 的前向，而 padding 行的输出梯度为零，
所以它们的贡献恰好是零。左 padding 会改变绝对位置，所以不在覆盖范围内。

### 3. paged KV、打包、torch.compile 和 CUDA graph 都不能改变结果

- **目标：** 引擎在 kernel 外面加的这些层，不能改变结果的任何一个 bit。
- **做法：** 使用 [examples/attention_serving](../../examples/attention_serving/serving_kernels.py)
  里的 paged kernel、varlen kernel 和它们的检查；torch.compile 和 CUDA graph 的做法见
  [recipe 12](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs)。
- **成功标准：** 每种页大小、每种打包偏移、每个编译版本和 CUDA graph 版本，结果都与连续 cache 上的
  kernel 逐 bit 一致。这在一种 GPU 上得到了证明
  （[报告](../../examples/attention_serving/REPORT.md#what-the-serving-layer-results-show)）。

| 层 | 怎么做 | 代价等级 |
|---|---|---|
| paged KV | 计算用的 block 始终按绝对位置划分，无论页大小如何，都把各页的数据收集到 block 里 | 小到中等，取决于页大小和流水线设置 |
| 变长序列打包成的 batch | 每个序列的 block 从它自己的第一个 token 开始，而不是落在整个打包的网格上 | 小 |
| `torch.compile` | 把每个会舍入的算子（norm、激活函数，而不只是 kernel）注册为不透明的自定义算子，让编译器无法移动舍入点 | 在 CUDA graph 下小；batch 小时在 eager 下 host 端开销大 |
| CUDA graph | 使用不变的 kernel；padding 后的 batch 按桶的大小运行，结果仍然逐 bit 一致 | 免费 |

### 4. 按注意力特有的维度验证

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)，并加上注意力
特有的维度：在不能整除 block 大小的 chunk 大小下测每种模式，用多种页大小并随机摆放页，用不是
block 大小倍数的打包偏移，比较 torch.compile 和 eager 的结果，以及测带 padding 的反向。

## 可能出错的地方

!!! warning "只发警告的确定性模式并不确定"
    在 `warn_only=True` 下，库先打出一条警告，然后照样运行非确定性的反向，所以梯度每次运行仍会变。
    要用 `warn_only=False`，这样遇到没有确定性版本的算子时会直接报错。

!!! warning "反向不可能比它的前向更不变"
    如果前向会随 batch 变化，在它上面做确定性的反向，结果仍然随 batch 变化。先修前向。

!!! warning "只测一种页大小，发现不了依赖页大小的 kernel"
    如果 kernel 的 block 恰好等于一页，那么在任何一种页大小下它在各模式之间都自洽，但每换一种
    页大小，答案就不同。要测多种页大小。

!!! warning "打包偏移恰好是 block 的倍数时，测试会侥幸通过"
    跟着打包网格走的 kernel，只有在序列恰好从 block 边界开始时才正确。要测不是 block 大小倍数的偏移。

!!! warning "编译器会移动舍入点"
    编译后的 norm 和激活函数仍然是 batch-invariant 的，但和 eager 下的结果不再一致。如果引擎要和 eager 的
    trainer 对齐，就把每个会舍入的算子都对编译器设为不透明。

## 小结

- 注意力在 key 上做归约，block 的边界决定了怎样分组。
- 使用从绝对位置 0 起数的固定 block，不用 split-KV，每种模式都用同一个 kernel。
- 反向需要分成两遍、不用原子操作；只打开严格确定性模式只能得到 L1，得不到 L2。
- 只有当计算用的 block 完全不理会 paged KV、打包、torch.compile 和 CUDA graph 时，这些层才不会改变结果。

## 深入阅读

- 报告：[kernel 案例](../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)
  （前向）、[注意力反向](../../examples/attention_backward/REPORT.md)和
  [推理服务层](../../examples/attention_serving/REPORT.md)。
- 代码：[tools/bi_kernels.py](../../tools/bi_kernels.py)、
  [examples/attention_backward/attn_bwd.py](../../examples/attention_backward/attn_bwd.py)、
  [examples/attention_serving/serving_cases.py](../../examples/attention_serving/serving_cases.py)。
- Recipe：[4，softmax 注意力](../../skills/batch-invariant-kernel/references/operator-recipes.md#4-softmax-attention-prefill-decode-chunked-prefill-paged-kv)、
  [12，部署包装层](../../skills/batch-invariant-kernel/references/operator-recipes.md#12-deployment-wrappers-torchcompile-and-cuda-graphs)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
