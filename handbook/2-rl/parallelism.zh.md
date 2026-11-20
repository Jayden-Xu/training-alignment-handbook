# 并行度：每个 TP 大小下位都相同（L5）

## 我们在哪里

完成[整个模型](whole-model.md)这一步后，单侧在单个 GPU 上、每种批次和服务模式下
都对一个序列给出相同的位。在 RL 中，trainer 和引擎很少使用相同的张量并行（TP）度，
分离式服务也会让 prefill 用一个度、decode 用另一个度。这一步让结果与并行度无关（L5）。

## 思路

行并行层把归约维度 K 切到各个 rank 上：每个 rank 算一个部分和，再由集合通信把部分和相加。
取四个乘积 a、b、c、d：

| 并行度 | 计算的是什么 |
|---|---|
| TP 1 | ((a + b) + c) + d，最后舍入一次 |
| TP 2 | round(a + b) + round(c + d)：两个部分和，各自舍入到 bf16，再相加 |
| TP 4 | a + b + c + d，顺序由 all-reduce 决定 |

每个并行度给出不同的求和树和不同的舍入次数，于是得到不同的数。两个 rank 就够了：
x0 + x1 满足交换律，但部分和本身已经不是不切分时的和。

修法是**逻辑 TP 大小**：不管并行度是多少，都把 K 切成固定的 C 块，用一棵固定的树相加。
每个 rank 拥有一棵对齐的子树，部分和保持 fp32，rank 之间交换子树，
每个 rank 完成同一棵树，只舍入一次。TP 1（C 块都在一个 GPU 上）的树就是
TP C（每个 rank 一块）的树。规则：归约树是模型的属性，不是集群的属性。
这是批不变性原则的 L5 形式，单个 GEMM 的情形见[集合通信](../3-kernels/collectives.md)。

<figure class="fx" data-fig="tp-tree" aria-label="TP 1、2、4 下的同一棵固定归约树">
<div class="fx-stage">
<div class="fx-head"><span class="fx-label">K 切成 C = 4 个 chunk；张量并行度</span><span class="fx-swap"><span class="fx-chip" data-to="3">TP 1</span><span class="fx-chip is-key" data-at="3" data-to="5">TP 2</span><span class="fx-chip is-key" data-at="5">TP 4</span></span></div>
<div class="fx-tree" style="--n:4">
<div class="fx-swap is-block"><div class="fx-level" data-to="3"><span class="fx-band" data-c="0" style="--w:4">rank 0</span></div><div class="fx-level" data-at="3" data-to="5"><span class="fx-band" data-c="0" style="--w:2">rank 0</span><span class="fx-band" data-c="1" style="--w:2">rank 1</span></div><div class="fx-level" data-at="5"><span class="fx-band" data-c="0" style="--w:1">rank 0</span><span class="fx-band" data-c="1" style="--w:1">rank 1</span><span class="fx-band" data-c="2" style="--w:1">rank 2</span><span class="fx-band" data-c="3" style="--w:1">rank 3</span></div></div>
<div class="fx-level"><span class="fx-cell">c0</span><span class="fx-cell">c1</span><span class="fx-cell">c2</span><span class="fx-cell">c3</span></div>
<div class="fx-level" data-at="1" data-hot="3 5"><span class="fx-j" style="--w:2"><span class="fx-node is-mono">c0 + c1</span></span><span class="fx-j" style="--w:2"><span class="fx-node is-mono">c2 + c3</span></span></div>
<div class="fx-level" data-at="2" data-hot="4 6"><span class="fx-j" style="--w:4"><span class="fx-node is-mono">同一棵树，fp32</span></span></div>
<div class="fx-level" data-at="2" data-hot="4 6"><span class="fx-j" style="--w:4;--k:1"><span class="fx-node is-key">一次 cast 到 bf16：y</span></span></div>
</div>
<div class="fx-chain is-tight"><span class="fx-label">结果</span><span class="fx-chip is-ok" data-at="2">TP 1: y</span><span class="fx-chip is-ok" data-at="4">TP 2: y</span><span class="fx-chip is-ok" data-at="6">TP 4: y</span></div>
</div>
<ol class="fx-say">
<li>无论并行度多少，K 都切成 C = 4 个 chunk。TP 1：一个 GPU 拿着全部四个。</li>
<li>固定的树先把 chunk 两两相加，用 fp32，</li>
<li>再把两半相加，只舍入一次到 bf16：y。</li>
<li data-hold="1.5">TP 2：每个 rank 求一棵对齐的子树，用 fp32。</li>
<li data-hold="1.5">各 rank 交换子树，完成同一棵树，只舍入一次：还是 y。</li>
<li data-hold="1.5">TP 4：每个 rank 一个 chunk。各 rank 交换 chunk，按同一棵树相加。</li>
<li data-hold="2">同一棵树，一次舍入：TP 1、TP 2 和 TP 4 给出相同的位。</li>
</ol>
<figcaption>图：每个 rank 求一棵对齐的子树；各 rank 完成同一棵树，只舍入一次，所以 TP 1 和 TP 2 给出相同的位。</figcaption>
</figure>

还有三处跨 rank 求和，需要同样处理：
- 词表并行的 log_softmax，它的指数和按词表分片切开；
- 列并行层的反向，它的输入梯度对分布在不同 rank 上的输出列求和；
- KV cache，它继承了写入它的投影层的任何波动。

列并行层不需要处理：每个输出列都在一个 rank 上由同一个 kernel 完整算出。
一旦每个部分都不变，分离式的 KV 交接就只是字节拷贝。

## 步骤

1. **让每个行并行求和都使用逻辑 TP。**
   - 目标：o_proj、down_proj 以及其他行并行层在每个并行度下位都相同。
   - 做法：[tp_model.py](../../examples/whole_model_tp/tp_model.py) 中的分片模型
     （变体 `inv`）；单个 GEMM 的方案见[集合通信](../3-kernels/collectives.md)。
   - 成功标准：TP t 下的 logits 与 TP 1 逐位相等。
2. **让词表并行的 log_softmax 顺序固定。**
   - 目标：每个并行度下实际 token 的 logprob 相等。
   - 做法：最大值用 all-reduce 求（任何顺序都精确）；指数和按固定的词表块计算，
     用固定的树相加。
   - 成功标准：等于对收集起来的 logits 做单 GPU log_softmax 的结果。
3. **用同样的办法修反向。**
   - 目标：每个并行度下激活梯度相等。
   - 做法：列并行输入梯度使用逻辑 TP 归约。
   - 成功标准：每个层边界上的输入梯度都等于 TP 1。
4. **检查 TP 下引擎的各种模式，以及交接。**
   - 目标：L2、L3 和 L5 同时成立。
   - 做法：[whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py)
     在 TP 2 和 4 下运行变化批次中的 decode 和分块 prefill，
     并在一个并行度下 prefill、另一个并行度下 decode。
   - 成功标准：每种模式都等于 TP 1 的 trainer 前向；收到的 KV cache 等于发出的。

| 选项 | 得到什么 | 代价类别 | 何时选择 |
|---|---|---|---|
| 原生 TP（bf16 部分和，库的 all-reduce） | 跨并行度没有任何保证；每个实际 token 的 logprob 都会变 | 免费 | trainer 和引擎并行度相同，且不需要审计 |
| 只在前向使用逻辑 TP | logprob 达到 L5；反向仍会变 | prefill 中等（不到 2x）；decode 因额外的集合通信可能接近 2x | 只给 rollout 打分 |
| 前向和反向都用逻辑 TP | logprob 和激活梯度都达到 L5 | 同上，再加上反向的归约 | trainer 以不同于引擎的并行度重算并训练 |

已证明：跨 TP 度（最多两个节点）logits 和实际 token 的 logprob 逐位相等，
与 L2、L3 同时成立，经过分离式交接也成立；激活梯度在最多四个 GPU 上相等；
在一种 GPU 型号上、经 PCIe 互连
（[报告](../../examples/whole_model_tp/REPORT.md#results)）。尚未证明：
MoE、序列并行、不能整除 C 的并行度、参数梯度、生产引擎自己的 all-reduce。

## 可能出错的地方

!!! warning "逻辑 TP 是另一种算术"
    TP 1 下的逻辑 TP 模型并不等于普通的单 GPU 模型。如果只有引擎采用它，
    trainer 反而离得更远。两侧必须使用相同的 C 和相同的树。

!!! warning "前向的修复不会自动带到反向"
    列并行输入梯度是另一个跨 rank 求和。用库的 all-reduce 时，即使前向相等，
    它也会破坏 L5。明确地修反向的归约，并且测试它。

!!! warning "一个分片就是另一种形状的 GEMM"
    列并行分片的 N 更小，GEMM 库可能在任何集合通信之前就为它选了另一个 kernel。
    每个分片都要用不变的 GEMM，而不只是让求和不变。

!!! warning "集合通信翻倍会增加延迟"
    交换 fp32 子树再加一次 gather，取代了一次 all-reduce，所以受延迟限制的 decode
    比 prefill 慢得更多。在评估代价之前，先融合交换，或者跨节点分两级路由。

## 小结

- TP 度改变每个行并行求和的归约树。
- 固定树，而不是固定并行度：用 fp32 部分和、只转换一次的逻辑 TP 大小，
  在每个并行度下给出相同的位。
- 词表并行的 log_softmax 和反向需要同样的修复。
- 各部分不变之后，分离式 prefill/decode 只是一次 KV cache 拷贝。

## 深入阅读

- [实验报告](../../examples/whole_model_tp/REPORT.md)：分片表、两种变体、所有结果、
  各并行度的代价、局限。
- 代码：[tp_model.py](../../examples/whole_model_tp/tp_model.py)、
  [whole_model_tp.py](../../examples/whole_model_tp/whole_model_tp.py)。
- 单个 GEMM 的情形：[集合通信](../3-kernels/collectives.md)。
- 下一步：[采样](sampling.md)。
