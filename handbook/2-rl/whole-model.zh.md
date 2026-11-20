# 整个模型的不变性（L1–L3）

## 我们在哪里

这是 [RL 工作流程](README.md)的第一步。你手上有一个 trainer 和引擎都在跑的模型，
即使权重完全相同，两端的 logprob 也对不上。这一步先审查模型的算子，让其中一端对同一条
序列，不管怎样组 batch、用什么推理模式，都算出逐 bit 一致的结果：多次运行之间相同（L1），
在任何 batch 里相同（L2），在任何推理模式下相同（L3）。让两端彼此一致，是后面
[引擎一致性](engine-parity-vllm.md)那一步的事。

## 思路

trainer 和引擎算的是同样的 token，但计算时的形状不同。trainer 用一次 prefill 把整条
序列算完，而且是和其他训练序列放在同一个 batch 里一起算。引擎则借助 KV cache 每一步
decode 一个 token，它所在的 batch 每一步都有请求加入、有请求离开。引擎还可能对 prompt
做 chunked prefill、复用已经缓存的前缀，或者验证 speculative decoding 给出的草稿 token。

<figure class="fx" data-fig="prefill-decode" aria-label="同一条序列在 trainer 里用 prefill 计算，在引擎里用 decode 计算">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">trainer<span class="fx-note">prefill：一次算完所有 token</span></span><div class="fx-lane" style="--n:6"><div class="fx-cells"><span class="fx-cell" data-hot="1">p</span><span class="fx-cell" data-hot="1">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="1">t2</span><span class="fx-cell is-mine" data-hot="1">t3</span><span class="fx-cell is-mine" data-hot="1">t4</span></div><div class="fx-cells"><span></span><span></span><span class="fx-span" style="--w:4" data-at="1">在 B 条序列的 batch 里一次算完</span></div><div class="fx-cells fx-lp"><span></span><span></span><span class="fx-chip" data-at="2">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="2">ℓ3</span><span class="fx-chip" data-at="2">ℓ4</span></div></div></div>
<div class="fx-row"><span class="fx-label">引擎<span class="fx-note">decode：每步一个 token</span></span><div class="fx-lane" style="--n:6"><div class="fx-cells"><span class="fx-cell">p</span><span class="fx-cell">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="2">t2</span><span class="fx-cell is-mine" data-hot="3">t3</span><span class="fx-cell is-mine" data-hot="4">t4</span></div><div class="fx-cells fx-who"><span></span><span></span><span class="fx-note" data-at="1">+ A、B</span><span class="fx-note" data-at="2">+ B、C</span><span class="fx-note" data-at="3">只有它自己</span><span class="fx-note" data-at="4">+ D、E</span></div><div class="fx-cells fx-lp"><span></span><span></span><span class="fx-chip" data-at="1">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="3">ℓ3</span><span class="fx-chip" data-at="4">ℓ4</span></div></div></div>
</div>
<div class="fx-chain is-tight"><span class="fx-node is-decision" data-at="5">ℓ1–ℓ4 这几个 logprob 是否逐 bit 一致？</span></div>
</div>
<ol class="fx-say">
<li>以一条序列为例：一段 prompt（p），后面跟着采样出的四个 token，t1 到 t4。</li>
<li data-hold="1.5">trainer 在一个有 B 条序列的 batch 里，用一次 prefill 算出所有 token。引擎用 decode 算出 t1，同一个 batch 里还有请求 A 和 B。</li>
<li data-hold="1.5">trainer 已经得到每个 token 的 logprob。引擎接着算 t2，这时 A 已经离开，C 加入了 batch。</li>
<li>引擎算 t3，此时 batch 里只剩它自己。</li>
<li>引擎算 t4，batch 里又来了两个新请求 D 和 E。</li>
<li data-hold="2">token 是同样的，计算时的形状和 batch 却都不同。这一步要让两端的 logprob 依然逐 bit 一致。</li>
</ol>
<figcaption>图：trainer 和引擎以不同的形状计算同样的 token。trainer 在固定的 batch 里做一次 prefill；引擎在不断变化的 batch 里一步一步地 decode。</figcaption>
</figure>

浮点求和的结果取决于加法的顺序。在 fp32 中把 1e8、-1e8 和 1 相加：从左往右加得到 1；
但如果先把 1 加到 -1e8 上，1 就会被舍掉，结果变成 0。因此，一个按行数选择归约顺序的
kernel，会让同一行在 batch size 为 1 和为 64 时得到不同的结果。两端之间有三样东西会改变
求和顺序：

| 来源 | 发生了什么 | 破坏的级别 |
|---|---|---|
| 随 batch 变化的 kernel | 行数一变，GEMM 库或归约就换用另一个 kernel，或者换一种切分方式 | L2，batch |
| 依赖模式的 kernel | prefill、chunked prefill 和 decode 走的是不同的注意力代码 | L3，模式 |
| 舍入点 | 两端在不同的地方转换到 bf16，例如一端在乘 norm 权重之前转换，另一端在之后 | L4，跨引擎 |

所以规则是：从 token 到 logprob 的路径上，每个算子的归约顺序都只能由这一行自己的数据
决定，不能由 batch 或模式决定。这就是[第 3 部分](../3-kernels/README.md)讲的
batch-invariant 原则，只不过用到了整个模型上。表格最后一行说明了为什么只有不变的 kernel 还不够：一个做到了
L2 和 L3、却有一处舍入不同的引擎，离 trainer 的距离和未经修改的引擎一样远。
这一行留到后面的步骤解决。

## 步骤

1. **审查算子。**
   - 目标：列出输出会随 batch 变化的模块。
   - 做法：[tools/bi_audit.py](../../tools/bi_audit.py) 把每个模块分别在单独一行和
     整个 batch 上重放，找出第一个输出不同的模块。被 bf16 输出掩盖住的 fp32 归约，它也能查出来。
   - 成功标准：每个模块都报告为不变，或者你拿到了一份需要替换的模块清单。
2. **替换会随 batch 变化的算子。**
   - 目标：得到一套归约顺序都固定的算子。
   - 做法：按照 [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)
     及其[算子 recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md)来做。
   - 成功标准：审查通过。
3. **对照 trainer 的前向，检查引擎用到的每种推理模式。**
   - 目标：整个模型达到 L1–L3，而不只是每个算子单独达到。
   - 做法：测试在不断变化的 batch 里 decode、chunked prefill、命中 prefix cache、验证草稿 token、
     抢占、padding 和变长 batch；[whole_model.py](../../examples/whole_model/whole_model.py)
     就是对参考模型做的这项检查。
   - 成功标准：每种模式下 logits 都逐 bit 一致，激活梯度也逐 bit 一致。
4. **写下舍入序列。**
   - 目标：留下一份记录，后面对齐引擎时以它为准。
   - 做法：记下每一次类型转换、哪些运算用 fp32、RoPE 的公式、norm 权重在哪里乘入。
     要记哪些内容，见[舍入点](../../skills/engine-parity/references/rounding-points.md)
     检查清单；[model.py](../../examples/whole_model/model.py) 是一个完整的例子。
   - 成功标准：每个舍入点一行，按发生的先后排列。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 保留原生算子 | 没有任何逐 bit 保证；不同 batch、不同模式之间，大多数 token 的结果都不同 | 免费 | 只用来测量差距 |
| 两端都使用 batch-invariant 算子集 | 每一端的前向和激活梯度都达到 L1–L3 | 中等（不到 2 倍）；prefill 很大时会到大 | 两端的模型代码都由你掌控 |
| batch-invariant 算子集，再加上两端统一的舍入序列 | 达到 L4 的前提 | 同上，另外还要对齐引擎的类型转换 | trainer 和引擎是两套不同的代码（[引擎一致性](engine-parity-vllm.md)） |

已证明：由 batch-invariant kernel 构成的模型，在测过的每种推理模式下，logits 和激活梯度
都逐 bit 一致；测了四种架构，都是在一种 GPU 型号上、eager 模式下得到的。换用原生算子时，几乎每个 logprob
都不同（[报告：不变性](../../examples/whole_model/REPORT.md#invariance)、
[RL 会看到什么](../../examples/whole_model/REPORT.md#what-rl-would-see-with-stock-ops)、
[代价](../../examples/whole_model/REPORT.md#cost)）。尚未证明：
`torch.compile`、CUDA graph、TP（下一步）以及真实引擎。

## 可能出错的地方

!!! warning "bf16 输出可能掩盖随 batch 变化的 fp32 归约"
    模块内部的 fp32 归约（例如 RMSNorm 里的方差）可能随行数变化，而模块的 bf16 输出
    却保持不变。只检查输出的话，会在一种形状下通过，换一种形状就失败。这样的归约照样要
    替换；审查工具会把它报告为 "variant inside"（[报告](../../examples/whole_model/REPORT.md#where-the-variance-comes-from)）。

!!! warning "在 torch.compile 下构建的 RoPE 表和 eager 下的不一样"
    在编译区域里构建的表，用的是编译器生成的数学实现，有少数条目和 eager 下构建的表不同，
    而且之后的 eager 运行也会继续用这张表。要先在 eager 模式下把表建好，再编译。

!!! warning "参数梯度最多只能做到 L1"
    参数梯度是对整个 batch 求和得到的，没法逐行比较。对它只检查多次运行之间是否相同，
    L2 检查则留给激活梯度。

!!! warning "两个 Hugging Face 参考模型并不是同一个模型"
    加载之后再转换成 bf16 的模型，会把 RoPE 频率也舍入到 bf16；直接以 bf16 加载的模型
    则不会。两者的 logits 有明显差别。要说清楚引擎是和哪一个比较。

!!! warning "要测试你实际会用到的序列长度"
    一个注意力 kernel 可能在短序列上满足不变性，在长序列上却随 batch 变化。
    短序列上测试通过，并不能说明长序列也没问题。

## 小结

- trainer 和引擎在不同的 batch、不同的模式下计算同样的 token，而原生 kernel 的归约顺序
  会随这两者改变。
- 先审查模型，替换每个会变的算子，再对照 trainer 的前向检查每种推理模式。
- 每一端各自 batch-invariant 是必要条件，但不是充分条件：两端还必须在相同的地方舍入。
- 现在就把舍入序列写下来，后面对齐引擎时要以它为准。

## 深入阅读

- [实验报告](../../examples/whole_model/REPORT.md)：模型变体、算子集、执行模式、
  全部结果表、审查工具，以及代价和局限。
- 代码：[model.py](../../examples/whole_model/model.py)、
  [whole_model.py](../../examples/whole_model/whole_model.py)、
  [tools/bi_audit.py](../../tools/bi_audit.py)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)、
  [engine-parity](../../skills/engine-parity/SKILL.md)。
- 下一步：[并行](parallelism.md)。
