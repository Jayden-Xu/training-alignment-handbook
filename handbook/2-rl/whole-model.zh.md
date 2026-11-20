# 让单侧不变：整个模型（L1–L3）

## 我们在哪里

这是 [RL 工作流程](README.md)的第一步。你有一个 trainer 和引擎都在运行的模型，
即使权重相同，logprob 也对不上。这一步审查它的算子，让单侧对一个序列给出相同的位，
不管它怎样被批处理、怎样被服务：多次运行之间（L1）、任意批次中（L2）、任意模式下（L3）。
让两侧一致放在后面的[引擎一致性](engine-parity-vllm.md)。

## 思路

trainer 和引擎以不同的形状计算同样的 token。trainer 把整个序列一次算完（prefill），
放在一批训练序列里。引擎从 KV cache 每步生成一个 token（decode），所在的批次
每一步都有成员加入和离开；它还可能把 prompt 分块、复用已缓存的前缀，或者验证投机 token。

<figure class="fx" data-fig="prefill-decode" aria-label="同一条序列：trainer 用 prefill 计算，引擎用 decode 计算">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label"><b>trainer</b> prefill：一次算全部 token，batch 有 B 条序列</span><div class="fx-col is-wide"><div class="fx-batch"><span class="fx-rowbar"></span><span class="fx-rowbar"></span><div class="fx-cells" style="--n:6"><span class="fx-cell" data-hot="1">p</span><span class="fx-cell" data-hot="1">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="1">t2</span><span class="fx-cell is-mine" data-hot="1">t3</span><span class="fx-cell is-mine" data-hot="1">t4</span></div><span class="fx-rowbar"></span></div><div class="fx-cells fx-lp" style="--n:6"><span></span><span></span><span class="fx-chip" data-at="2">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="2">ℓ3</span><span class="fx-chip" data-at="2">ℓ4</span></div></div></div>
<div class="fx-row"><span class="fx-label"><b>engine</b> decode：每步一个 token，batch 每步都在变</span><div class="fx-col is-wide"><div class="fx-batch"><span class="fx-rowbar" data-at="1" data-to="3"></span><span class="fx-rowbar" data-at="1" data-to="2"></span><div class="fx-cells" style="--n:6"><span class="fx-cell">p</span><span class="fx-cell">p</span><span class="fx-cell is-mine" data-hot="1">t1</span><span class="fx-cell is-mine" data-hot="2">t2</span><span class="fx-cell is-mine" data-hot="3">t3</span><span class="fx-cell is-mine" data-hot="4">t4</span></div><span class="fx-rowbar" data-at="2" data-to="3"></span><span class="fx-rowbar" data-at="4"></span><span class="fx-rowbar" data-at="4"></span></div><div class="fx-cells fx-lp" style="--n:6"><span></span><span></span><span class="fx-chip" data-at="1">ℓ1</span><span class="fx-chip" data-at="2">ℓ2</span><span class="fx-chip" data-at="3">ℓ3</span><span class="fx-chip" data-at="4">ℓ4</span></div></div></div>
</div>
<div class="fx-chain is-tight"><span class="fx-node is-decision" data-at="5">logprob ℓ1–ℓ4：逐位相等？</span></div>
</div>
<ol class="fx-say">
<li>一条序列：prompt（p）和四个采样出的 token，t1 到 t4。灰条是 batch 里的其他序列。</li>
<li data-hold="1.5">trainer：一次 prefill 同时算出所有 token，batch 有 B 条序列。引擎：decode 算出 t1，同 batch 还有两个请求。</li>
<li data-hold="1.5">trainer：每个 token 一个 logprob。引擎：t2；一个请求离开，另一个加入。</li>
<li>引擎：t3，此时 batch 里只有它。</li>
<li>引擎：t4，又来了两个新请求。</li>
<li data-hold="2">同样的 token，不同的形状，不同的 batch。这一步要让 logprob 逐位相等。</li>
</ol>
<figcaption>图：trainer 和引擎用不同的形状计算同样的 token：一边是固定 batch 里的一次 prefill，一边是不断变化的 batch 里一次一步的 decode。</figcaption>
</figure>

浮点求和的结果取决于顺序。在 fp32 中把 1e8、-1e8 和 1 相加：从左到右得到 1；
但先把 1 加到 -1e8 上，1 会被舍掉，结果是 0。一个按行数选择归约顺序的 kernel，
因此会让同一行在批大小为 1 和 64 时得到不同的位。两侧之间有三样东西会改变顺序：

| 来源 | 发生了什么 | 破坏的级别 |
|---|---|---|
| 随批次变化的 kernel | 行数变化时，GEMM 库或归约选择另一个 kernel 或另一种切分 | L2，批次 |
| 依赖模式的 kernel | prefill、分块 prefill 和 decode 走不同的注意力代码路径 | L3，模式 |
| 舍入点 | 两侧在不同位置转换到 bf16，例如在乘上 norm 权重之前或之后 | L4，跨引擎 |

规则：从 token 到 logprob 路径上的每个算子，归约顺序都必须只由该行自己的数据决定，
不能由批次或模式决定。这是[第三部分](../3-kernels/README.md)的批不变性原则在整个模型上的应用。
表的最后一行说明为什么只有不变的 kernel 还不够：一个达到 L2 和 L3、但有一处舍入不同的引擎，
离 trainer 和原生引擎一样远。这一行在后面的步骤里解决。

## 步骤

1. **审查算子。**
   - 目标：输出依赖批次的模块清单。
   - 做法：[tools/bi_audit.py](../../tools/bi_audit.py) 把每个模块分别在单行和批次中重放，
     指出第一个输出不同的模块，包括被 bf16 输出掩盖的 fp32 归约。
   - 成功标准：每个模块都报告为不变，或者你拿到了需要替换的清单。
2. **替换会变的算子。**
   - 目标：一套归约顺序固定的算子。
   - 做法：[batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)
     及其[算子配方](../../skills/batch-invariant-kernel/references/operator-recipes.md)。
   - 成功标准：审查通过。
3. **对照 trainer 前向，检查引擎用到的每种服务模式。**
   - 目标：整个模型达到 L1–L3，而不只是单个算子。
   - 做法：在变化的批次中 decode、分块 prefill、前缀命中、投机验证、抢占、
     padding、varlen；[whole_model.py](../../examples/whole_model/whole_model.py)
     就是参考模型的这项检查。
   - 成功标准：每种模式下 logits 逐位相等，激活梯度也相等。
4. **写下舍入序列。**
   - 目标：以后对齐引擎时要依据的记录。
   - 做法：每一次转换、哪些算子用 fp32、RoPE 公式、norm 权重在哪里相乘；
     检查清单见[舍入点](../../skills/engine-parity/references/rounding-points.md)，
     实例见 [model.py](../../examples/whole_model/model.py)。
   - 成功标准：按顺序，每个舍入点一行。

| 选项 | 得到什么 | 代价类别 | 何时选择 |
|---|---|---|---|
| 保留原生算子 | 没有任何逐位保证；不同批次和模式之间大多数 token 不同 | 免费 | 只用来测量差距 |
| 两侧都用批不变算子集 | 每侧达到 L1–L3，前向和激活梯度 | 中等（不到 2x），大 prefill 时可达大 | 两侧的模型代码都归你管 |
| 批不变算子集加上统一的舍入序列 | L4 的前提 | 同上，再加上对齐引擎的类型转换 | trainer 和引擎是不同的代码（[引擎一致性](engine-parity-vllm.md)） |

已证明：由批不变 kernel 构成的模型，在测过的每种服务模式下 logits 和激活梯度都逐位相等，
覆盖四种架构，在一种 GPU 型号上、eager 模式下；用原生算子时几乎每个 logprob 都不同
（[报告：不变性](../../examples/whole_model/REPORT.md#invariance)、
[RL 会看到什么](../../examples/whole_model/REPORT.md#what-rl-would-see-with-stock-ops)、
[代价](../../examples/whole_model/REPORT.md#cost)）。尚未证明：
`torch.compile`、CUDA graphs、张量并行（下一步）、真实引擎。

## 可能出错的地方

!!! warning "bf16 输出可能掩盖随批次变化的 fp32 归约"
    模块内部的 fp32 归约（RMSNorm 的方差）可能随行数变化，而模块的 bf16 输出不变。
    只检查输出，会在一种形状下通过、在另一种形状下失败。照样替换它；审查工具会把它
    报告为 "variant inside"（[报告](../../examples/whole_model/REPORT.md#where-the-variance-comes-from)）。

!!! warning "在编译器下构建的 RoPE 表不是 eager 的表"
    在编译区域内构建的表使用编译器的数学，与 eager 的表有少数条目不同，
    并且会留给之后的 eager 运行使用。先在 eager 下构建，再编译。

!!! warning "参数梯度只能做到 L1"
    参数梯度对整个批次求和，无法逐行比较。对它做多次运行之间的检查，
    对激活梯度做 L2 检查。

!!! warning "两个 Hugging Face 参考不是同一个模型"
    加载后再转换为 bf16 的模型会把 RoPE 频率舍入到 bf16，直接以 bf16 加载的模型不会，
    两者的 logits 有明显差别。说清楚引擎是和哪一个比较。

!!! warning "测试你实际运行的序列长度"
    一个注意力 kernel 可能在短行上不变，在长行上随批次变化。
    短长度上的通过对长长度说明不了什么。

## 小结

- trainer 和引擎在不同的批次和模式下运行同样的 token；原生 kernel 会随两者改变归约顺序。
- 审查模型，替换每个会变的算子，然后对照 trainer 前向检查每种服务模式。
- 每侧的批不变性是必要条件，不是充分条件：两侧还必须在相同的位置舍入。
- 现在就写下舍入序列；以后按它对齐引擎。

## 深入阅读

- [实验报告](../../examples/whole_model/REPORT.md)：模型变体、算子集、执行模式、
  所有结果表、审查工具、代价和局限。
- 代码：[model.py](../../examples/whole_model/model.py)、
  [whole_model.py](../../examples/whole_model/whole_model.py)、
  [tools/bi_audit.py](../../tools/bi_audit.py)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)、
  [engine-parity](../../skills/engine-parity/SKILL.md)。
- 下一步：[并行](parallelism.md)。
