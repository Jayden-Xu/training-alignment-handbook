# 换 TP 规模，结果仍逐 bit 一致（L5）

## 我们在哪里

完成[整个模型](whole-model.md)这一步后，只要在单个 GPU 上运行，一端对同一条序列在
任何 batch、任何推理模式下都能算出逐 bit 一致的结果。可是在 RL 中，trainer 和引擎很少
使用相同的 TP 规模；PD 分离部署（prefill 和 decode 放在不同机器上）甚至让 prefill 用一种
TP 规模、decode 用另一种。这一步让结果不再依赖 TP 规模（L5）。

## 思路

行并行层把归约维度 K 切开，分给各个 rank：每个 rank 先算出一个部分和，再由集合通信
把这些部分和加起来。以四个乘积 a、b、c、d 为例：

| TP 规模 | 实际算的是什么 |
|---|---|
| TP 1 | ((a + b) + c) + d，最后舍入一次 |
| TP 2 | round(a + b) + round(c + d)：每个 rank 的部分和先舍入到 bf16，再把两者相加 |
| TP 4 | a + b + c + d，相加顺序由 all-reduce 决定 |

每种 TP 规模的求和树不同，舍入次数也不同，所以得到的数也不同。只要有两个 rank 就会出现
这个问题：x0 + x1 与 x1 + x0 固然相等，但部分和本身就已经不同于单 GPU 上算出的和。

解决办法是使用**逻辑 TP 规模**：不管实际的 TP 规模是多少，都把 K 切成同样的 C 块
（chunk），并用一棵固定的树把这些块加起来。每个 rank 负责一棵对齐的子树，部分和保持
fp32；随后各 rank 交换子树，每个 rank 都完成同一棵树，最后只舍入一次。这样，TP 1
（C 块都在一个 GPU 上）的树就和 TP C（每个 rank 一块）的树完全相同。规则是：归约树
属于模型，而不属于集群。这是 batch-invariant 原则在 L5 上的形式；单个 GEMM 的情形见
[集合通信](../3-kernels/collectives.md)。

<figure class="fx" data-fig="tp-tree" aria-label="TP 1、2、4 使用同一棵固定的归约树">
<div class="fx-stage">
<div class="fx-tree" style="--n:4">
<div class="fx-lrow"><span class="fx-label">TP 规模</span><div class="fx-chain is-tight"><span class="fx-chip" data-hot="0-2 6">TP 1<span class="fx-pass" data-at="2">&nbsp;→ y</span></span><span class="fx-chip" data-hot="3 4 6">TP 2<span class="fx-pass" data-at="4">&nbsp;→ y</span></span><span class="fx-chip" data-hot="5 6">TP 4<span class="fx-pass" data-at="6">&nbsp;→ y</span></span></div></div>
<div class="fx-lrow"><span class="fx-label">rank</span><div class="fx-swap is-block"><div class="fx-level" data-to="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span></div><div class="fx-level" data-at="3" data-to="5"><span class="fx-band" data-c="0" style="--w:2">rank 0</span><span class="fx-band" data-c="1" style="--w:2">rank 1</span></div><div class="fx-level" data-at="5"><span class="fx-band" data-c="0">rank 0</span><span class="fx-band" data-c="1">rank 1</span><span class="fx-band" data-c="2">rank 2</span><span class="fx-band" data-c="3">rank 3</span></div></div></div>
<div class="fx-lrow"><span class="fx-label">K 切成 4 块</span><div class="fx-level fx-sweep" data-hot="0 5"><span class="fx-cell">c0</span><span class="fx-cell">c1</span><span class="fx-cell">c2</span><span class="fx-cell">c3</span></div></div>
<div class="fx-lrow"><span class="fx-label">固定的树，用 fp32 相加</span><div class="fx-level" data-at="1"><span class="fx-j" style="--w:2" data-hot="1 3 5"><span class="fx-node is-mono">c0 + c1</span></span><span class="fx-j" style="--w:2" data-hot="1 3 5"><span class="fx-node is-mono">c2 + c3</span></span></div></div>
<div class="fx-lrow"><span></span><div class="fx-level" data-at="2"><span class="fx-j" style="--w:4" data-hot="2 4 5"><span class="fx-node is-mono">(c0 + c1) + (c2 + c3)</span></span></div></div>
<div class="fx-lrow"><span class="fx-label">只舍入一次</span><div class="fx-level" data-at="2"><span class="fx-j" style="--w:4;--k:1" data-hot="2 4 6"><span class="fx-node is-key">bf16: y</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>无论 TP 规模是多少，K 都切成 C = 4 块。TP 1 时，四块都在一个 GPU 上。</li>
<li>按照固定的树，先用 fp32 把各块两两相加，</li>
<li>再把两半加起来，只舍入一次到 bf16，得到 y。</li>
<li data-hold="1.5">TP 2 时，每个 rank 用 fp32 求出自己那棵对齐的子树。</li>
<li data-hold="1.5">各 rank 交换子树，完成同一棵树，只舍入一次，得到的还是 y。</li>
<li data-hold="1.5">TP 4 时，每个 rank 只有一块。各 rank 交换各自的块，再按同一棵树相加。</li>
<li data-hold="2">分块相同、树相同、都只舍入一次，所以 TP 1、TP 2 和 TP 4 的结果逐 bit 一致。</li>
</ol>
<figcaption>图：不管 TP 规模是多少，分块和树都相同，变的只是这些块怎样分给各个 rank。每个 rank 求一棵对齐的子树，结果只舍入一次，所以 TP 1、TP 2 和 TP 4 的结果逐 bit 一致。</figcaption>
</figure>

还有三处跨 rank 的求和，需要同样处理：

- 词表并行的 log_softmax：它的指数和是按词表 shard 分开计算的；
- 列并行层的反向：它的输入梯度要对分布在不同 rank 上的输出列求和；
- KV cache：写入它的投影层如果有任何波动，它都会原样继承。

前向的列并行层不需要处理，因为每个输出列都在一个 rank 上由同一个 kernel 完整算出。
一旦每个部分都满足不变性，把 KV cache 从 prefill 机器交给 decode 机器，就只是一次
逐字节的拷贝。

## 步骤

1. **让每个行并行求和都使用逻辑 TP 规模。**
   - 目标：o_proj、down_proj 以及其他所有行并行层，在每种 TP 规模下都给出逐 bit 一致的结果。
   - 做法：使用 [tp_model.py](../../examples/whole_model_tp/tp_model.py) 中按 TP 切分的模型
     （变体 `inv`）；单个 GEMM 的方案见[集合通信](../3-kernels/collectives.md)。
   - 成功标准：TP t 下的 logits 与 TP 1 下的逐 bit 一致。
2. **让词表并行的 log_softmax 按固定顺序计算。**
   - 目标：每种 TP 规模下，采样出的 token 的 logprob 都相等。
   - 做法：最大值用 all-reduce 求，它在任何顺序下都是精确的；指数和按固定的词表块计算，
     再用一棵固定的树把各块加起来。
   - 成功标准：结果等于把 logits 收集到一个 GPU 上再做 log_softmax 的结果。
3. **用同样的办法修好反向。**
   - 目标：每种 TP 规模下激活梯度都相等。
   - 做法：列并行层的输入梯度也使用逻辑 TP 归约。
   - 成功标准：在每个层边界上，输入梯度都等于 TP 1 下的输入梯度。
4. **检查 TP 下引擎的各种推理模式，以及 KV cache 的交接。**
   - 目标：L2、L3 和 L5 同时成立。
   - 做法：[whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py)
     在 TP 2 和 4 下，在不断变化的 batch 里跑 decode 和 chunked prefill，还会在一种 TP 规模下
     做 prefill、在另一种 TP 规模下接着 decode。
   - 成功标准：每种模式都等于 TP 1 下 trainer 的前向；收到的 KV cache 与发出的完全相同。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 原生 TP（bf16 部分和，使用库的 all-reduce） | 不同 TP 规模之间没有任何保证；每个采样 token 的 logprob 都会变 | 免费 | trainer 和引擎的 TP 规模相同，而且不需要审计 |
| 只在前向使用逻辑 TP | logprob 达到 L5，但反向仍然会变 | prefill 为中等（不到 2 倍）；decode 因为多出的集合通信，可能接近 2 倍 | 只用来给 rollout 打分 |
| 前向和反向都使用逻辑 TP | logprob 和激活梯度都达到 L5 | 同上，再加上反向的归约 | trainer 用和引擎不同的 TP 规模重算并训练 |

已证明：最多用到两个节点时，不同 TP 规模下的 logits 和采样 token 的 logprob 逐 bit
一致，同时满足 L2 和 L3，经过 PD 分离的交接后也成立；激活梯度在最多四个 GPU 上逐 bit 一致。
这些结果都是在一种 GPU 型号、PCIe 互连的条件下得到的（[报告](../../examples/whole_model_tp/REPORT.md#results)）。
尚未证明：MoE、SP（序列并行）、不能整除 C 的 TP 规模、参数梯度，以及生产引擎自带的 all-reduce。

## 可能出错的地方

!!! warning "逻辑 TP 算的是另一种算术"
    TP 1 下使用逻辑 TP 的模型，并不等于普通的单 GPU 模型。如果只有引擎采用它，
    trainer 反而离得更远。两端必须使用相同的 C 和同一棵树。

!!! warning "修好前向，不等于修好了反向"
    列并行层的输入梯度是另一个独立的跨 rank 求和。如果用库的 all-reduce，即使前向
    相等，它也会破坏 L5。要明确地修好反向的归约，并且测试它。

!!! warning "一个 shard 就是一个形状不同的 GEMM"
    列并行 shard 的 N 更小，GEMM 库可能在任何集合通信发生之前，就为它换了一个 kernel。
    每个 shard 上都要使用不变的 GEMM，而不能只让跨 rank 的求和不变。

!!! warning "集合通信翻倍会增加延迟"
    原来的一次 all-reduce，被一次 fp32 子树交换加一次 gather 取代。decode 受延迟限制，
    所以比 prefill 慢得更多。在评估代价之前，先把交换融合起来，或者跨节点分两级路由。

## 小结

- TP 规模会改变每个行并行求和的归约树。
- 要固定的是树，而不是 TP 规模：采用逻辑 TP 规模，部分和保持 fp32、只转换一次，
  就能在每种 TP 规模下得到逐 bit 一致的结果。
- 词表并行的 log_softmax 和反向也需要同样的修复。
- 所有部分都满足不变性之后，PD 分离时 prefill 和 decode 之间只需要拷贝 KV cache。

## 深入阅读

- [实验报告](../../examples/whole_model_tp/REPORT.md)：切分方式表、两种变体、全部结果、
  各 TP 规模下的代价，以及局限。
- 代码：[tp_model.py](../../examples/whole_model_tp/tp_model.py)、
  [whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py)。
- 单个 GEMM 的情形：[集合通信](../3-kernels/collectives.md)。
- 下一步：[采样](sampling.md)。
