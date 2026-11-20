# 0. 基础

## 我们在哪里

本手册讲三个工作流：找到两次预训练运行从哪里分开（[第 1 部分](../1-pretraining/README.md)），让 rollout 引擎和 trainer 一致（[第 2 部分](../2-rl/README.md)），让一个算子具备不变性（[第 3 部分](../3-kernels/README.md)）。三者都建立在同样几个想法上。本部分讲这些想法：

1. 浮点加法不满足结合律；
2. 归约顺序取决于它运行时的形状；
3. 确定性不等于不变性；
4. 五个不变性级别，L1 到 L5；
5. 如何比较两次运行：逐位比较，在算子边界上，用 fingerprint。

两次运行**对齐**，是指它们本应用同样的数据、种子和权重计算同一个模型，并且在你观察的位置上**逐位**一致。对齐问的不是"它们接近吗？"，而是"它们第一次不再相同是在哪里？"。这个位置就是调试的起点。

## 浮点加法不满足结合律

取三个数，用两种顺序相加。在普通的 Python（fp64）里：

```python
>>> (0.1 + 0.2) + 0.3
0.6000000000000001
>>> 0.1 + (0.2 + 0.3)
0.6
```

fp32 里也一样，而且用的是模型每天都会遇到的数。令 a = 1e8，b = 1，c = −1e8：

<figure class="fx" data-fig="two-orders" aria-label="三个数按两种顺序相加">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">fp32</span><span class="fx-chip">a = 1e8</span><span class="fx-chip">b = 1</span><span class="fx-chip">c = −1e8</span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">(a + b) + c</span><div class="fx-chain is-tight is-grid"><span class="fx-swap" data-at="1"><span class="fx-node is-mono" data-at="1" data-to="2">a + b = 100000001</span><span class="fx-node is-mono is-warn" data-at="2">a + b → 1e8<span class="fx-note">舍入</span></span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">1e8 + c = 0</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">0</span></div></div>
<div class="fx-row"><span class="fx-label fx-mono">(a + c) + b</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">a + c = 0<span class="fx-note">精确</span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">0 + b = 1</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">1</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>三个 fp32 数：a = 1e8，b = 1，c = −1e8。用两种顺序把它们相加。</li>
<li>每种顺序先形成自己的第一个部分和。</li>
<li data-hold="1.5">a + b = 100000001 在 fp32 里无法表示，舍入为 1e8。</li>
<li>每种顺序再加上最后一个数。</li>
<li data-hold="2">同样三个数，两种顺序，两个结果：0 和 1。</li>
</ol>
<figcaption>图：fp32 下的 (a + b) + c 与 (a + c) + b。形成哪些部分和，决定了发生哪些舍入。</figcaption>
</figure>

每次加法都把结果舍入到最近的可表示数。发生哪些舍入，取决于形成了哪些部分和。所以：

**规则。** 浮点和的比特取决于加法的顺序和舍入的位置。同样的数学换一个顺序，就是另一个数。

这是本手册每一页背后的总原则：*归约顺序是结果的一部分*。矩阵乘、norm、softmax、注意力、梯度求和和集合通信都是求和，所以每一个都有顺序。两次运行只有在顺序一致时才会一致。

## 归约顺序取决于形状

kernel 不是按数学来选顺序的，而是按拿到的形状来选。所以算术会随批次变化。取一个四项点积，t = [1e8, 1, −1e8, 1]，fp32：

<figure class="fx" data-fig="three-orders" aria-label="同一个点积按三种顺序归约">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">t</span><span class="fx-chip">1e8</span><span class="fx-chip">1</span><span class="fx-chip">−1e8</span><span class="fx-chip">1</span></div>
<div class="fx-rows" style="--lw: 6.5rem; --cols: minmax(0, 1.2fr) auto minmax(0, 1fr) auto minmax(0, 1.2fr); --narrow: flex">
<div class="fx-row"><span class="fx-label">一个 program，从左到右</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">0 + 1 = 1</span></div></div>
<div class="fx-row"><span class="fx-label">split-K，分成两半</span><div class="fx-chain is-tight is-grid"><div class="fx-stack" data-at="1"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-node is-mono" data-at="1">−1e8 + 1 = −1e8</span></div><span class="fx-arrow is-long" data-at="2" style="grid-column: 2 / 5"></span><span class="fx-node is-mono is-key" data-at="2" data-hot="4">1e8 + (−1e8) = 0</span></div></div>
<div class="fx-row"><span class="fx-label">atomics，某一种完成顺序</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">0 + 1 = 1</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">1 + 1 = 2</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>同样四项 t = [1e8, 1, −1e8, 1]，在 fp32 下用三种方式归约。</li>
<li>第一次加法。split-K 在两个 program 上同时求两半的和。</li>
<li>split-K 把两个部分和相加，结束：0。</li>
<li>另外两种也结束了：从左到右得 1，这种完成顺序得 2。</li>
<li data-hold="2">三种顺序，三个结果：1、0 和 2。每个都是某种顺序的正确舍入。</li>
</ol>
<figcaption>图：fp32 下的 t = [1e8, 1, −1e8, 1]：同样四个数，三种顺序，三个结果。</figcaption>
</figure>

三种顺序，三个答案：1、0 和 2。每一个都是某种顺序下正确的舍入。决定顺序的机制有三种：

| 机制 | 是什么 | 由什么决定顺序 |
|---|---|---|
| split-K | 把归约维切成几段，由不同的 program 分别求和，再把部分和相加 | 库按矩阵形状选段数，常常按行数选 |
| tile | 一个 program 按 tile 宽度分块归约它的行，再合并各块 | tile 宽度和块数按形状选；到某个尺寸阈值，kernel 可能换成另一个 kernel |
| 原子操作 | 各 program 完成时把部分和加到同一个内存位置 | 完成的先后，每次运行都会变 |

**规则。** 如果 kernel 的顺序是形状的函数，一行的结果就是它所在批次的函数。如果顺序是时序的函数，结果就是这次运行的函数。

## 确定性不等于不变性

这是两个不同的承诺：

- **确定性**（运行间）：同样的输入、同样的形状，每次运行给出同样的比特。原子操作会破坏它。
- **不变性**（这里指 batch-invariant，批不变）：一行无论和什么一起计算都给出同样的比特：批大小是 1 还是 64，在批首还是批尾。

一个小例子。没有原子操作的矩阵乘是确定的。一行时库选了分两半的 split-K；64 行时它给每个输出一个 program。各跑两次：每种情况都精确重复自己的比特。但单独计算的第 0 行和在 64 行批次里计算的第 0 行，用的是上面两种不同的顺序，所以可能不同。

`torch.use_deterministic_algorithms(True)` 买到的是第一个承诺，不是第二个。一个 kernel 也可以是确定的，而它的顺序仍是实现细节（比如取决于它 scatter 的索引个数）。所以两个各自确定、都正确的框架不一定一致。

**规则。** 确定性消除随时间的变化。不变性消除随上下文的变化。两个不同配置之间的对齐需要不变性。

## 五个不变性级别

每个级别问的是：某一样东西改变时，比特能否不变。

| 级别 | 成立条件 | 一行破坏的例子 |
|---|---|---|
| L1 运行间 | 同一输入每次运行给出同样的比特 | 用原子操作累加梯度行的 backward，第二次运行给出不同的权重梯度 |
| L2 批 | 一行的结果不取决于批大小、它的位置或其他行 | GEMM 在一行时选 split-K，在 64 行时不选，于是一个序列的 logit 随批次改变 |
| L3 模式 | 预填充、解码、分块预填充和分页布局对每个位置给出同样的比特 | 解码把 KV cache 分给多个 program，预填充不分，于是同一个 token 得到两个注意力输出 |
| L4 跨引擎 | trainer 和推理引擎给出同样的比特 | 引擎在 fp32 下乘 norm 权重再转 bf16；trainer 先转换 |
| L5 并行 | 不同的 TP、EP、SP 或 DP 度给出同样的比特 | TP 2 时两个部分和跨 rank 相加；TP 4 时是四个，按另一棵树 |

每个场景需要不同的子集：

| 场景 | 需要 |
|---|---|
| 预训练调试（重放、改配置、升级） | 两次运行都满足 L1，且受控面上配置相同 |
| 推理确定性 | L1 + L2（使用分块预填充或前缀缓存时 + L3） |
| 强化学习训推一致 | L1–L4（两边切分不同时 + L5） |
| 迁移或升级的等价性 | 新旧栈之间 L1 + L4（并行方式改变时 + L5） |

每个级别都在一个**栈**上成立：一种 GPU、一个驱动和一组库版本。跨 GPU 代际通常不期望逐位一致（[硬件覆盖](../4-operations/hardware.md)）。在每个栈上分别验证各级别；跨栈的逐位比较是另一个需要单独测量的结论。[alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md) 把每个场景引到对应的工作流。

## 受控与残余非确定性

只消除**可避免的**变化，也就是让两次正确运行做不同算术的那些：

- 种子和数据顺序；
- 库的算法选择和数值模式；
- 集合通信的归约树；
- 不完整的检查点恢复。

其余的留着可见。用户或第三方 kernel 的结果随调度变化（有竞争的原子操作、缺少屏障、stream 竞争），那是 bug，不是噪声。把它控制住，就会藏起你要找的故障。

## 如何比较

比较两次运行是每个工作流都会重复的步骤。它有三个选择：判据、位置和摘要。

**判据：逐位。** 容差回答不了"第一个差异在哪里？"。宽到能藏住无害归约顺序噪声的阈值，也会藏住少量被损坏的元素。在生产 bug 上，用容差会失去很大一部分精确定位（测量结果：见[报告](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison)）。应当控制可避免的变化，然后比较比特。

**位置：模型级算子边界。** 也就是模型代码中启动设备计算的调用：一个投影、一个 norm、一个注意力块、一个专家合并。

- 整个 step 或整层太粗，无法定位。
- 单个 kernel 随每个库版本和融合选择而变。

每个忠实的实现里都有模型级边界，所以两个栈可以在那里比较。

**摘要：fingerprint（指纹）。** 转储每个张量代价太高。fingerprint 必须能抓住任何单比特变化，不能让小的损坏相互抵消，也不能同步设备。用一个四个值的张量举例：

<figure class="fx" data-fig="sum-xor" aria-label="和与 XOR fingerprint 的对比">
<div class="fx-stage">
<div class="fx-rows" style="--cols: minmax(0, 1.25fr) auto minmax(0, 1fr) minmax(0, 1fr); --cols-narrow: auto minmax(0, 1fr) minmax(0, 1fr)">
<div class="fx-row"><span class="fx-label">参考</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="1"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span></div><span class="fx-arrow" data-at="1"></span><span class="fx-node" data-at="1">和 <span class="fx-mono">10</span></span><span class="fx-node" data-at="1">原始 32 位字的 XOR <span class="fx-mono">w</span></span></div></div>
<div class="fx-row" data-at="2"><span class="fx-label">一个值翻转了一位</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="3"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 + ε</span></div><span class="fx-arrow" data-at="3"></span><span class="fx-node is-ok" data-at="3">和 <span class="fx-mono">改变</span><span class="fx-note">抓到</span></span><span class="fx-node is-ok" data-at="3">XOR <span class="fx-mono">改变</span><span class="fx-note">抓到</span></span></div></div>
<div class="fx-row" data-at="4"><span class="fx-label">两个相互抵消的错误</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="5"><span class="fx-cell is-bad">1 + ε</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 − ε</span></div><span class="fx-arrow" data-at="5"></span><span class="fx-node is-bad" data-at="5">和 <span class="fx-mono">10，不变</span><span class="fx-note">漏掉</span></span><span class="fx-node is-ok" data-at="5">XOR <span class="fx-mono">改变</span><span class="fx-note">抓到</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>一个有四个值的张量：[1, 2, 3, 4]。</li>
<li>它的和是 10；原始 32 位字的 XOR 是 w。</li>
<li>把一个值翻转一位：4 变成 4 + ε。</li>
<li>两种摘要都变了：这次翻转被抓到。</li>
<li>再看两个相互抵消的错误：1 + ε 和 4 − ε。</li>
<li data-hold="2">和又是 10，两个错误都漏掉了；XOR 仍然改变。</li>
</ol>
<figcaption>图：和会让错误相互抵消；对原始比特做 XOR 不会丢掉任何一次翻转。</figcaption>
</figure>

和或均值会让错误抵消；对原始比特做 XOR 不会丢掉单个翻转。OpGuard 在设备上计算 XOR fingerprint，有两种形式：

- **分区**（默认）：对展平张量的等长字节区间分别做 XOR。代价低，并能看出哪个区间不同。
- **按向量**（可选）：每个最后一维向量一个 XOR。XOR 可以合并，所以一个 token 的各向量无论布局如何都合成该 token 的行 fingerprint：`[T, heads, dim]` 和 `[T, hidden]` 得到同一行。这样在以不同方式组批、打包和解码的栈之间也能比较行，并指出不同的是哪个 head。

<figure class="fx" data-fig="xor-fold" aria-label="按向量的 fingerprint 合并成行 fingerprint">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">[T, heads, dim]</span><div class="fx-tree" style="--n:4">
<div class="fx-level"><span class="fx-cell">head 0</span><span class="fx-cell">head 1</span><span class="fx-cell">head 2</span><span class="fx-cell">head 3</span></div>
<div class="fx-level"><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f0</span></span><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f1</span></span><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f2</span></span><span class="fx-j" style="--k:1" data-at="1"><span class="fx-node is-mono">f3</span></span></div>
<div class="fx-level"><span class="fx-j" style="--w:2" data-at="2"><span class="fx-node is-mono">f0 ⊕ f1</span></span><span class="fx-j" style="--w:2" data-at="2"><span class="fx-node is-mono">f2 ⊕ f3</span></span></div>
<div class="fx-level"><span class="fx-j" style="--w:4" data-at="3" data-hot="5"><span class="fx-node is-mono is-key">行 r</span></span></div>
</div></div>
<div class="fx-row" data-at="4"><span class="fx-label fx-mono">[T, hidden]</span><div class="fx-tree" style="--n:4">
<div class="fx-level"><span class="fx-cell" style="--w:4">一个 hidden 向量</span></div>
<div class="fx-level"><span class="fx-j" style="--w:4;--k:1" data-at="4" data-hot="5"><span class="fx-node is-mono is-ok">行 r</span></span></div>
</div></div>
</div>
</div>
<ol class="fx-say">
<li>一个 token 的行，按 [T, heads, dim] 存放：四个 head 向量。</li>
<li>按向量：每个最后一维向量一个 XOR，f0 到 f3。</li>
<li>XOR 可以合并，所以各向量两两合并，</li>
<li>合成这一行的 fingerprint r。</li>
<li>按 [T, hidden] 存放时，同一行是一个向量，它的 XOR 也是同一个 r。</li>
<li data-hold="2">行可以跨布局比较；不同时，不同的那个 f 指出是哪个 head。</li>
</ol>
<figcaption>图：无论布局如何，按向量的 fingerprint 都合并成同一个行 fingerprint。</figcaption>
</figure>

各选项，从最便宜到最完整：

| 选项 | 给出什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 每个边界做分区 fingerprint | 哪个边界最先不同，哪个区间 | 每个边界很小：几个设备 kernel，不同步 | 总是，最先做 |
| 按向量 fingerprint | 哪一行或哪个 head 不同；跨布局可比 | 小到中等 | 比较组批或打包方式不同的栈 |
| 转储某一区域的完整张量 | 值本身，用于诊断 | 大（2 倍或更多），外加存储 | fingerprint 指出位置之后 |
| 对值使用容差 | "接近与否" | 免费 | 只用于检查精度，绝不用于定位 |

工具是 [opguard_bridge](../../tools/opguard_bridge.py)（用纯 torch 实现的 OpGuard fingerprint 格式）和 OpGuard 本身；第 1 部分两者都用。成功的标志：本应一致的两次运行在每个边界上 fingerprint 相等，而人为植入的单比特变化在植入它的边界被找到。

**事实与判断。** 让工具保持精确和通用，把案例知识放进 skill。工具计算 fingerprint，记录在已知位置观察到的内容，回答是否相等。由 agent 或工程师来决定哪些记录相互对应、缺失的一行是否被融合掉了、哪个分歧是首个、原因是什么，并把每个决定连同证据写下来。按我们的经验，这些决定的内置启发式在每个新案例上都会失效。

## 可能出错的地方

!!! warning "容差藏住了故障"
    能吸收归约顺序噪声的阈值，也会吸收小的损坏。你会定位到错误的地方，或什么都找不到。应控制可避免的非确定性，并逐位比较。

!!! warning "把确定性模式当成不变性"
    确定性算法让一次运行能重复自己；它不让一行独立于它的批次或另一个框架。两个确定的配置仍可能不一致。直接检查你需要的级别（L2、L4、L5）。

!!! warning "把 fingerprint 相等当成值相等"
    XOR 是 fingerprint，不是哈希：置换和成对抵消不会改变它，32 位的值在规模大时会碰撞。只有在宽度一致且值非零时才把相等当作证据；对怀疑的区域重新采集完整张量。

!!! warning "把跨栈比较当成同一个栈"
    各级别在一种 GPU 和一组库版本上成立。跨 GPU 代际的逐位差异是预期的，不是 bug。分别验证每个栈。

## 小结

- 归约顺序是结果的一部分：三个数按两种顺序相加可以得到两个答案。
- kernel 按形状（split-K、tile）或按时序（原子操作）选顺序；所以一行的比特可能取决于它的批次或这次运行。
- 确定性（L1）不等于不变性（L2–L5）；说清一个结论在哪个级别成立。
- 逐位比较，在模型级算子边界上，用 XOR fingerprint。
- 控制可避免的非确定性；让残余非确定性保持可见，因为它就是 bug 的信号。

## 深入阅读

- [术语表](../glossary.md)：这里用到的每个术语。
- [第 1 部分](../1-pretraining/README.md)：把这种比较用于两次训练运行。
- [第 3 部分](../3-kernels/README.md)：如何写一个顺序固定的 kernel。
- [opguard_bridge.py](../../tools/opguard_bridge.py)：fingerprint 格式。
- [报告：容差与逐位比较](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison)。
- [alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md)。
