# 0. 基础

## 我们在哪里

本手册讲三条工作流程：找出两次预训练从哪里开始不一致（[第 1 部分](../1-pretraining/README.md)），让 rollout 引擎和 trainer 的结果一致（[第 2 部分](../2-rl/README.md)），以及把一个算子改成 batch-invariant 的实现（[第 3 部分](../3-kernels/README.md)）。这三条流程依赖同样几个基本概念，本部分就讲这些概念：

1. 浮点加法不满足结合律；
2. kernel 按输入的形状决定求和的顺序；
3. 确定的 kernel 不一定是不变的；
4. 不变性分五个级别，从 L1 到 L5；
5. 比较两次运行时要逐 bit 比较，比较的地方选在算子边界上，并用一种叫 fingerprint（指纹）的小摘要代替完整的张量。

两次运行本来就该用同样的数据、种子和权重算同一个模型；如果在你观察的每个位置上，它们的结果都**逐 bit** 一致，就说这两次运行是**对齐**的。对齐关心的不是两次运行"差不多"，而是它们从哪里开始不再完全相同，因为调试就从那里开始。

## 浮点加法不满足结合律

取三个数，按两种顺序相加。普通的 Python 用的是 fp64：

```python
>>> (0.1 + 0.2) + 0.3
0.6000000000000001
>>> 0.1 + (0.2 + 0.3)
0.6
```

换成 fp32、用模型里随处可见的数值，也是一样。令 a = 1e8，b = 1，c = −1e8：

<figure class="fx" data-fig="two-orders" aria-label="三个数按两种顺序相加">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">fp32</span><span class="fx-chip">a = 1e8</span><span class="fx-chip">b = 1</span><span class="fx-chip">c = −1e8</span></div>
<div class="fx-rows">
<div class="fx-row"><span class="fx-label fx-mono">(a + b) + c</span><div class="fx-chain is-tight is-grid"><span class="fx-swap" data-at="1"><span class="fx-node is-mono" data-at="1" data-to="2">a + b = 100000001</span><span class="fx-node is-mono is-warn" data-at="2">a + b → 1e8<span class="fx-note">舍入</span></span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">1e8 + c = 0</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">0</span></div></div>
<div class="fx-row"><span class="fx-label fx-mono">(a + c) + b</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">a + c = 0<span class="fx-note">精确</span></span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono" data-at="3">0 + b = 1</span><span class="fx-arrow" data-at="4"></span><span class="fx-node is-mono is-key" data-at="4">1</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>三个 fp32 数：a = 1e8，b = 1，c = −1e8。按两种顺序把它们加起来。</li>
<li>两种顺序各自先算出一个部分和。</li>
<li data-hold="1.5">fp32 表示不了 a + b = 100000001，只能舍入成 1e8。</li>
<li>两种顺序再各自加上剩下的那个数。</li>
<li data-hold="2">同样三个数，一种顺序得 0，另一种顺序得 1。</li>
</ol>
<figcaption>图：fp32 下的 (a + b) + c 与 (a + c) + b。先算出哪些部分和，决定了会在哪里发生舍入。</figcaption>
</figure>

每做一次加法，结果都要舍入到最近的可表示数。会发生哪些舍入，取决于先算出了哪些部分和。所以：

**规则。** 浮点求和的结果取决于加法的顺序和舍入发生的位置。数学上相同的式子，换一个顺序计算，得到的可能是另一个数。

这是本手册每一页背后的总原则：*归约顺序本身就是结果的一部分*。矩阵乘、norm、softmax、注意力、梯度求和、集合通信，本质上都是求和，所以都有一个求和顺序。两次运行只有在顺序一致时，结果才会一致。

## kernel 按输入的形状决定归约顺序

kernel 决定加法顺序时，看的不是数学，而是它拿到的输入是什么形状。所以 batch 一变，实际做的算术也跟着变。以一个四项点积 t = [1e8, 1, −1e8, 1] 为例，在 fp32 下计算：

<figure class="fx" data-fig="three-orders" aria-label="同一个点积按三种顺序归约">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-label">t</span><span class="fx-chip">1e8</span><span class="fx-chip">1</span><span class="fx-chip">−1e8</span><span class="fx-chip">1</span></div>
<div class="fx-rows" style="--lw: 6.5rem; --cols: minmax(0, 1.2fr) auto minmax(0, 1fr) auto minmax(0, 1.2fr); --narrow: flex">
<div class="fx-row"><span class="fx-label">一个 program 从左加到右</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">0 + 1 = 1</span></div></div>
<div class="fx-row"><span class="fx-label">split-K 分成两半</span><div class="fx-chain is-tight is-grid"><div class="fx-stack" data-at="1"><span class="fx-node is-mono" data-at="1">1e8 + 1 = 1e8</span><span class="fx-node is-mono" data-at="1">−1e8 + 1 = −1e8</span></div><span class="fx-arrow is-long" data-at="2" style="grid-column: 2 / 5"></span><span class="fx-node is-mono is-key" data-at="2" data-hot="4">1e8 + (−1e8) = 0</span></div></div>
<div class="fx-row"><span class="fx-label">原子操作的某一种完成顺序</span><div class="fx-chain is-tight is-grid"><span class="fx-node is-mono" data-at="1">1e8 − 1e8 = 0</span><span class="fx-arrow" data-at="2"></span><span class="fx-node is-mono" data-at="2">0 + 1 = 1</span><span class="fx-arrow" data-at="3"></span><span class="fx-node is-mono is-key" data-at="3" data-hot="4">1 + 1 = 2</span></div></div>
</div>
</div>
<ol class="fx-say">
<li>同样四项 t = [1e8, 1, −1e8, 1]，在 fp32 下用三种方式归约。</li>
<li>三种方式先各做第一步加法。split-K 让两个 program 同时分别求两半的和。</li>
<li>split-K 接着把两个部分和相加，得到 0，就结束了。</li>
<li>另外两种方式也算完了：从左加到右得 1，原子操作按这种先后完成则得 2。</li>
<li data-hold="2">三种顺序得到三个结果：1、0 和 2。每个结果都是按它自己的顺序正确舍入得到的。</li>
</ol>
<figcaption>图：同样四个 fp32 数 t = [1e8, 1, −1e8, 1]，按三种顺序求和，得到三个不同的结果。</figcaption>
</figure>

三种顺序给出三个答案：1、0 和 2，而且每个答案都是按各自顺序正确舍入的结果。kernel 用哪种顺序，由下面三种机制决定：

| 机制 | 它做什么 | 顺序由什么决定 |
|---|---|---|
| split-K | 把归约维切成几段，由不同的 program 分别求和，最后把这些部分和加起来 | 库根据矩阵形状决定切几段，常常就看行数 |
| tile | 一个 program 按 tile 的宽度把它那一行分块归约，再把各块合起来 | tile 宽度和块数随形状而定；尺寸超过某个阈值时，库可能干脆换一个 kernel |
| 原子操作 | 各个 program 算完后，各自把部分和加到同一个内存位置上 | 取决于各 program 完成的先后，而这个先后每次运行都可能不同 |

**规则。** 如果 kernel 的顺序取决于形状，那么一行的结果就取决于它所在的 batch。如果顺序取决于时序，那么结果每次运行都可能不同。

## 确定性和不变性不是一回事

这两个词是两种不同的保证：

- **确定性**（运行间）：输入相同、形状相同时，每次运行的结果都逐 bit 一致。原子操作会破坏这一点。
- **不变性**（这里指 batch-invariant，即结果不随 batch 变化）：一行的结果与它和什么一起计算无关。不管它所在的 batch 是 1 行还是 64 行，排在开头还是末尾，结果都逐 bit 一致。

举个小例子。不用原子操作的矩阵乘是确定的。只有一行时，库用 split-K 把归约分成两半；有 64 行时，库给每个输出分配一个 program。两种情况各跑两次，每种情况都能逐 bit 重复自己的结果。但是单独计算的第 0 行和放在 64 行的 batch 里计算的第 0 行，用的是上面两种不同的顺序，所以结果可能不同。

`torch.use_deterministic_algorithms(True)` 只提供第一种保证，不提供第二种。还有一种情况：kernel 是确定的，但它用哪种顺序只是实现细节（比如取决于要 scatter 多少个索引）。所以两个框架即使各自确定、也都正确，结果也不一定一致。

**规则。** 确定性消除的是随时间的变化：同一个任务每次运行都能重复自己的结果。不变性消除的是随上下文的变化：一行的结果与它和什么一起计算无关。要让两套不同的配置对齐，需要的是不变性。

## 五个不变性级别

每个级别问的都是：某一样东西变了，结果还能不能逐 bit 一致。

| 级别 | 什么时候成立 | 破坏它的一个例子 |
|---|---|---|
| L1 运行间 | 同一输入每次运行的结果都逐 bit 一致 | 反向传播用原子操作累加梯度行，第二次运行得到的权重梯度就不一样 |
| L2 batch | 一行的结果与 batch size、它在 batch 中的位置以及其他行都无关 | GEMM 在一行时用 split-K，在 64 行时不用，于是同一条序列的 logit 随 batch 变化 |
| L3 模式 | 同一个位置不管用 prefill、decode、chunked prefill 还是 paged KV cache 来算，结果都逐 bit 一致 | decode 把 KV cache 分给多个 program 处理，prefill 不分，于是同一个 token 得到两个不同的注意力输出 |
| L4 跨引擎 | trainer 和推理引擎的结果逐 bit 一致 | 引擎在 fp32 下乘上 norm 权重再转成 bf16，而 trainer 先转换再乘 |
| L5 并行 | 不同的 TP、EP、SP 或 DP 并行度下，结果逐 bit 一致 | TP 2 时把两个部分和相加；TP 4 时要加四个，而且加法树也不同 |

不同的场景需要满足的级别也不同：

| 场景 | 需要满足的级别 |
|---|---|
| 调试预训练（重放、改配置、升级） | 两次运行都满足 L1，并且所有能控制的设置都相同 |
| 确定性推理 | L1 和 L2；用了 chunked prefill 或 prefix caching 时，还要 L3 |
| 强化学习中 trainer 和 rollout 引擎要一致 | L1–L4；两端切分模型的方式不同时，还要 L5 |
| 证明迁移或升级没有改变任何结果 | 新旧软件栈之间满足 L1 和 L4；并行方式改变时，还要 L5 |

每个级别都只在一个**软件栈**内成立。一个栈指一种 GPU、一个驱动版本和一组库版本。跨 GPU 代际时，一般不指望逐 bit 一致（[硬件覆盖](../4-operations/hardware.md)）。要在每个栈上分别验证这些级别；跨栈逐 bit 一致是另一个结论，需要单独测量。[alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md) 会根据场景把你引到对应的工作流程。

## 哪些非确定性要消除，哪些要留着

只消除**可避免的**变化，也就是那些让两次正确运行做了不同算术的因素：

- 种子和数据顺序；
- 库所选的算法和数值模式；
- 集合通信的归约树；
- 只恢复了一部分的 checkpoint。

消除了这些，非确定性就是**受控**的。剩下的叫**残余**非确定性，要让它保持可见。如果用户或第三方 kernel 的结果会随调度变化（原子操作有竞争、缺少屏障、stream 之间有竞争），那是 bug，不是噪声。把它也控制住，反而会把你要找的故障藏起来。

## 怎样比较两次运行

每条工作流程都会反复做一件事：比较两次运行。要比较得好，需要做三个选择：什么算相等，在哪里比较，以及怎样概括每个张量。

**只有逐 bit 一致才算相等。** 用容差比较，回答不了"第一个差异在哪里"。容差宽到足以掩盖归约顺序带来的无害差异，也就足以掩盖少数被损坏的元素。在真实的生产 bug 上，有很大一部分本来可以精确定位的案例，换成容差比较就定位不到了（测量结果见[报告](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison)）。正确的做法是先控制可避免的变化，再逐 bit 比较。

**在模型级的算子边界上比较。** 算子边界指模型代码中启动设备计算的一次调用，比如一个投影、一个 norm、一个注意力块或一次 expert combine。选这个粒度有两个原因：

- 以整个 step 或整层为单位比较太粗，看不出差异从哪里开始。
- 以单个 kernel 为单位又太细，因为每换一个库版本、每改一次融合方式，kernel 就会变。

模型级的边界则在每个忠实的实现里都存在，所以两个软件栈可以在这里互相比较。

**用 fingerprint 概括每个张量。** 把每个张量都转储下来代价太高，所以要把每个张量压缩成一个小摘要，也就是它的 fingerprint。fingerprint 必须能发现任何一个 bit 的变化，不能让几处小的损坏相互抵消，也不能强制设备同步。下面用一个只有四个值的张量举例：

<figure class="fx" data-fig="sum-xor" aria-label="求和与 XOR fingerprint 的对比">
<div class="fx-stage">
<div class="fx-rows" style="--cols: minmax(0, 1.25fr) auto minmax(0, 1fr) minmax(0, 1fr); --cols-narrow: auto minmax(0, 1fr) minmax(0, 1fr)">
<div class="fx-row"><span class="fx-label">参考</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="1"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell">4</span></div><span class="fx-arrow" data-at="1"></span><span class="fx-node" data-at="1">和 <span class="fx-mono">10</span></span><span class="fx-node" data-at="1">原始 32-bit 字的 XOR <span class="fx-mono">w</span></span></div></div>
<div class="fx-row" data-at="2"><span class="fx-label">一个值翻转了一个 bit</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="3"><span class="fx-cell">1</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 + ε</span></div><span class="fx-arrow" data-at="3"></span><span class="fx-node is-ok" data-at="3">和 <span class="fx-mono">变了</span><span class="fx-note">发现</span></span><span class="fx-node is-ok" data-at="3">XOR <span class="fx-mono">变了</span><span class="fx-note">发现</span></span></div></div>
<div class="fx-row" data-at="4"><span class="fx-label">两个相互抵消的错误</span><div class="fx-chain is-tight is-grid"><div class="fx-cells fx-sweep" style="--n:4" data-hot="5"><span class="fx-cell is-bad">1 + ε</span><span class="fx-cell">2</span><span class="fx-cell">3</span><span class="fx-cell is-bad">4 − ε</span></div><span class="fx-arrow" data-at="5"></span><span class="fx-node is-bad" data-at="5">和 <span class="fx-mono">仍是 10</span><span class="fx-note">漏掉</span></span><span class="fx-node is-ok" data-at="5">XOR <span class="fx-mono">变了</span><span class="fx-note">发现</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>一个有四个值的张量：[1, 2, 3, 4]。</li>
<li>它的和是 10；把它的原始 32-bit 字做 XOR，得到某个值 w。</li>
<li>让其中一个值翻转一个 bit：4 变成 4 + ε。</li>
<li>两种摘要都变了，所以哪一种都能发现这次翻转。</li>
<li>再制造两个相互抵消的错误：1 变成 1 + ε，4 变成 4 − ε。</li>
<li data-hold="2">和又回到 10，两个错误都漏掉了；XOR 却仍然变了。</li>
</ol>
<figcaption>图：求和时错误会相互抵消；对原始 bit 做 XOR，任何一个 bit 的翻转都不会漏掉。</figcaption>
</figure>

求和或求均值时，错误会相互抵消；对原始 bit 做 XOR，任何一个 bit 的翻转都不会漏掉。OpGuard 在设备上计算 XOR fingerprint，有两种形式：

- **分区** fingerprint 是默认形式：把展平的张量切成等长的字节区间，每个区间做一次 XOR。它代价低，还能看出张量的哪个区间不同。
- **逐向量** fingerprint 需要时再打开：沿最后一维，每个向量做一次 XOR。由于 XOR 的结果可以再合并，一个 token 的各个向量可以合成这个 token 那一行的 fingerprint，而且与布局无关：`[T, heads, dim]` 和 `[T, hidden]` 得到的行 fingerprint 相同。这样，即使两个软件栈组 batch、打包和 decode 的方式不同，也能逐行比较；某一行不同时，各向量的值还能指出是哪个 head 不同。

<figure class="fx" data-fig="xor-fold" aria-label="逐向量的 fingerprint 合并成行 fingerprint">
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
<li>一个 token 的一行按 [T, heads, dim] 存放，分成四个 head 向量。</li>
<li>逐向量的形式给每个向量算一个 XOR，得到 f0 到 f3。</li>
<li>XOR 的结果可以再合并，于是这些向量 fingerprint 两两合并，</li>
<li>最后合成这一行的 fingerprint r。</li>
<li>如果同一行按 [T, hidden] 存放，它就是一个向量，它的 XOR 也是同一个 r。</li>
<li data-hold="2">所以不同布局之间也能逐行比较；两行不同时，不同的那个 f 指出是哪个 head。</li>
</ol>
<figcaption>图：无论布局如何，逐向量的 fingerprint 都会合成同一个行 fingerprint。</figcaption>
</figure>

可选的做法如下，从最便宜到最完整：

| 选项 | 能告诉你什么 | 代价等级 | 什么时候用 |
|---|---|---|---|
| 在每个边界上取分区 fingerprint | 哪个边界最先不同，以及它的哪个区间不同 | 每个边界都很小（几个设备 kernel，不需要同步） | 总是要做，而且最先做 |
| 逐向量的 fingerprint | 哪一行或哪个 head 不同，而且能跨布局比较 | 小到中等 | 两个软件栈组 batch 或打包的方式不同时 |
| 转储某个区域的完整张量 | 数值本身，用来诊断差异 | 大（2 倍或更多），还要占存储 | fingerprint 已经指出位置之后 |
| 对数值用容差比较 | 数值是否接近 | 免费 | 只用来检查精度，绝不用来定位差异 |

计算这些 fingerprint 的工具有两个：[opguard_bridge](../../tools/opguard_bridge.py)，它用纯 PyTorch 实现了 OpGuard 的 fingerprint 格式；以及 OpGuard 本身。第 1 部分两个都会用到。比较做对了的标志是：本该一致的两次运行在每个边界上的 fingerprint 都相等，而故意植入的单个 bit 的变化，恰好在植入它的那个边界上被找到。

**工具记录事实，判断交给 agent 和工程师。** 工具要保持精确、通用，针对具体案例的知识放进 skill。工具负责计算 fingerprint、记录它在已知位置看到的内容，并回答两样东西是否相等。哪些记录相互对应、缺失的一行是不是被融合掉了、哪个分歧最先出现、原因是什么，这些由 agent 或工程师判断，并且每个判断都要连同证据一起写下来。按我们的经验，把这类判断写成启发式规则内置在工具里，每遇到一个新案例都会失效。

## 可能出错的地方

!!! warning "容差把故障藏了起来"
    能吸收归约顺序噪声的阈值，同样会吸收小的损坏。结果你会定位到错误的地方，或者什么也找不到。应当控制可避免的非确定性，然后逐 bit 比较。

!!! warning "把确定性模式当成不变性"
    确定性算法只保证一次运行能重复自己，并不能让一行的结果与它所在的 batch 无关，也不能让它与另一个框架一致。所以两套都是确定的配置，仍然可能不一致。需要哪个级别（L2、L4 或 L5），就直接检查哪个级别。

!!! warning "把 fingerprint 相等当成数值相等"
    XOR 是 fingerprint，不是哈希。打乱各个字的顺序不会改变它，两处相互抵消的改动也不会；张量多了以后，32-bit 的值偶尔还会碰撞。只有在宽度一致、且值不为零时，才把 fingerprint 相等当作证据；对怀疑的区域，要重新采集完整张量。

!!! warning "把两个软件栈当成一个来比较"
    各个级别只在一种 GPU 和一组库版本上成立。不同代际的 GPU 之间出现逐 bit 差异是预期之内的，不是 bug。每个软件栈要分别验证。

## 小结

- 归约顺序是结果的一部分：同样三个数，按两种顺序相加，可能得到两个不同的答案。
- kernel 要么按形状选顺序（split-K、tile），要么把顺序交给时序（原子操作），所以一行的结果可能取决于它所在的 batch，也可能每次运行都不同。
- 确定性（L1）不等于不变性（L2–L5）；说一个结论成立时，要说清是在哪个级别成立。
- 在模型级的算子边界上，用 XOR fingerprint 逐 bit 比较。
- 控制可避免的非确定性，让残余的非确定性保持可见，因为它正是 bug 的信号。

## 深入阅读

- [术语表](../glossary.md)解释了这里用到的每个术语。
- [第 1 部分](../1-pretraining/README.md)把这种比较方法用在两次训练上。
- [第 3 部分](../3-kernels/README.md)讲怎样写一个顺序固定的 kernel。
- [opguard_bridge.py](../../tools/opguard_bridge.py) 定义了 fingerprint 的格式。
- [报告：容差比较与逐 bit 比较](../../examples/pretraining_e2e/REPORT.md#overview-tolerances-against-bitwise-comparison)。
- [alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md)。
