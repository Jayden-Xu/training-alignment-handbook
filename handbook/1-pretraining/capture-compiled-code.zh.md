# 作业运行编译代码时

## 我们在哪里

你已经能用 OpGuard 采集一对运行，并从首个分歧点走到原因（[用 OpGuard 定位](localize-with-opguard.md)）；前两页讲了出在[训练 step](training-step.md) 和[梯度集合通信](data-parallel.md)里的原因。本页讲的是采集本身的一个缺口。很多作业并不逐个算子地运行模型，而是经过 `torch.compile` 运行，或者作为 CUDA graph 重放。为了采集，OpGuard 逐个算子地运行作业（eager 模式），而 eager 模式运行的 kernel 和编译后的作业不同。所以你采到的比特，可能不是生产中出现分歧的那些比特。

你会从[工作流](README.md)的两个步骤来到这里：在采集（第 3 步），生产作业是编译过的；在分类（第 5 步），eager 重放看不到症状，或者 pivot 落在编译器生成的代码里。本页先讲编译器和 CUDA graph 做了什么、各自为什么会改变比特，再讲怎样从它们内部记录 OpGuard 格式的 fingerprint，使工作流的其余部分照常运行。

## 编译代码做了什么

### torch.compile：许多小算子变成少数几个生成的 kernel

在 eager 模式下，PyTorch 在 Python 调用一个算子时就运行它：一个算子一个 kernel。每个 kernel 从 GPU 内存读入输入，计算，再把输出写回内存，写回时舍入到输出的 dtype。一个由几个小算子写成的 norm，就是同样多个 kernel，每个都写出一个舍入过的中间结果。

`torch.compile` 把 Python 读一遍（由它的前端 Dynamo 追踪），把算子记成一张图，再交给一个编译器（默认是 Inductor）为这张图写新的 kernel。编译器把一串逐元素算子，连同其间的归约，**融合**进一个 kernel：中间值留在寄存器里（张量是 bf16 时用 fp32），只写出结果。每个归约怎样在线程间切分，由它自己决定。它还会改写模式：矩阵乘后面跟一个加法，可以变成一个顺带做加法的矩阵乘（`addmm`）。

<figure class="fx" data-fig="fusion" aria-label="同一个 norm 在 eager 下运行，和编译成一个 kernel 运行">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">eager：一个算子一个 kernel</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node is-mono" data-hot="1">x²</span><span class="fx-chip" data-at="1">↓ bf16</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="2">均值</span><span class="fx-chip" data-at="2">↓ bf16</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="3">rsqrt</span><span class="fx-chip" data-at="3">↓ bf16</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="4">× x</span><span class="fx-chip" data-at="4">↓ bf16</span></span><span class="fx-arrow" data-hot="5" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="5">× w</span><span class="fx-chip" data-at="5">↓ bf16</span></span></div></div>
<div class="fx-row"><span class="fx-label">编译后：一个生成的 kernel</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-box to-key" data-hot="6"><span class="fx-note">一个 kernel</span><span class="fx-chain is-tight fx-sweep" data-hot="6"><span class="fx-chip">x²</span><span class="fx-chip">均值</span><span class="fx-chip">rsqrt</span><span class="fx-chip">× x</span><span class="fx-chip">× w</span></span><span class="fx-note">中间结果留在寄存器里，fp32</span></span><span class="fx-chip" data-at="7">↓ bf16 只写结果</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>同一个 norm，写成五个算子；w 是它的权重。</li>
<li>eager：一个算子一个 kernel。x² 写进内存，舍入到 bf16。</li>
<li>均值把它读回来，归约，再写出一个舍入过的结果。</li>
<li>rsqrt：又一个 kernel，又一次舍入。</li>
<li>× x：又一个 kernel，又一次舍入。</li>
<li>× w：五个 kernel，内存里五个舍入过的中间结果。</li>
<li data-hold="2">编译后：一个生成的 kernel 在 fp32 寄存器里跑完全部五个算子，并按自己的顺序求均值。</li>
<li data-hold="2">它只写出结果：一次舍入，而不是五次。</li>
</ol>
<figcaption>图：同一个 norm（w 是它的权重）。eager 把每个中间结果按它的 dtype 写进内存；融合的 kernel 把它们留在寄存器里，按自己的顺序求均值，只写出结果。</figcaption>
</figure>

### CUDA graph：把 launch 录一次，然后重放

从 Python 逐个 launch kernel，每个 kernel 都要花 CPU 时间。step 很小时，这些时间可能超过 GPU 上的计算。**CUDA graph** 把一串 launch（哪些 kernel、什么参数、在哪些缓冲区上）录制一次，之后用一次调用重放整串。重放不运行 Python，也不重新做任何选择：kernel、形状和内存地址都是录制时的那些。

所以一张 graph 只适合一种形状。输入大小多变的作业会录几张 graph，每个大小桶一张，并把每个输入 padding 到上面最近的那个录制大小。

<figure class="fx" data-fig="graph-pad" aria-label="5 行的 batch 在为 8 行录制的 graph 里重放">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-box to-key" data-hot="2"><span class="fx-swap"><span class="fx-note" data-to="2">5 行的 batch</span><span class="fx-note" data-at="2">为 8 行录制的 graph</span></span><span class="fx-cells fx-sweep" style="--n:8;--cw:1.7rem" data-hot="2"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span><span class="fx-cell is-pad" data-at="1">5</span><span class="fx-cell is-pad" data-at="1">6</span><span class="fx-cell is-pad" data-at="1">7</span></span></span><span class="fx-arrow" data-at="3">保留</span><span class="fx-box" data-at="3"><span class="fx-note">前 5 行</span><span class="fx-cells" style="--n:5;--cw:1.7rem"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span></span></span></div>
</div>
<ol class="fx-say">
<li>来了一个 5 行的 batch。</li>
<li>把它 padding 到下一个录制过的大小：3 行 padding（5 到 7）凑成 8 行。</li>
<li data-hold="2">为 8 行录制的 graph 开始重放：其中每个 kernel 看到的都是 8 行。</li>
<li data-hold="2">只有前 5 行作为答案保留。</li>
</ol>
<figcaption>图：5 行的 batch 在为 8 行录制的 graph 里运行。答案只用真实的行，但重放中的每个 kernel 看到的都是 8 行。</figcaption>
</figure>

## 比特为什么会变

**融合移动了舍入点。** 一个 bf16 的小例子：在 1 附近，相邻两个 bf16 数的间距是 ε = 2⁻⁷。令 a = b = 1 + ε，c = −(1 + 2ε)，计算 a · b + c。

| | eager：两个 kernel | 编译后：一个融合 kernel |
|---|---|---|
| a · b | 1 + 2ε + ε²，写出时舍入到 bf16：ε² 不到半个间距，所以是 1 + 2ε | 1 + 2ε + ε²，以 fp32 留在寄存器里 |
| + c | (1 + 2ε) − (1 + 2ε) = 0 | ε² |
| bf16 结果 | 0 | ε² |

数学相同，舍入的次数不同。融合 kernel 自己的归约顺序改变一个和，方式和第 0 部分那三个数一样（[第 0 部分](../0-foundations/README.md#floating-point-addition-is-not-associative)）。

**按 padding 后的大小重放，改变了 GEMM 看到的形状。** padding 的行不在答案里，却在矩阵里：重放中的 GEMM 乘的是 8 行，不是 5 行。GEMM 库按形状选 kernel，也就选定了归约顺序（[第 0 部分：归约的顺序取决于形状](../0-foundations/README.md#a-reductions-order-depends-on-shape)）。所以这 5 行真实数据的舍入，可能和 eager 下按 5 行运行时不同。batch-invariant（批不变）的 GEMM（[第 3 部分](../3-kernels/gemm-and-quantized.md)）能消除这一项。

**规则。** 编译代码和它所来自的 eager 代码是不同的算术：舍入点不同，归约顺序不同，形状也不同。在 eager 下做的比较，说明的只是 eager。

## 常规采集为什么看不到内部

OpGuard 的采集方式是包装模型 Python 代码中的调用：被包装的调用运行时，包装层对它的输入和输出取 fingerprint（[用 OpGuard 定位](localize-with-opguard.md)）。编译代码并不逐个发出这些调用。编译器把 Python 读过一遍之后，运行的是它生成的 kernel；graph 重放则完全不运行 Python。在编译后的图或一次重放内部，已经没有可以包装的 Python 调用了。

所以常规采集是在 eager 下运行作业，也就是上一节说的另一种算术。症状在 eager 下能复现时，这就够了。不能复现时，观察者必须放进编译代码内部。

## 思路：一个编译器会保留、又无法穿透的观察者

**编译代码内部的观察者，必须是编译器会保留的算子。** 自定义算子就是这样：编译器在图中它所在的位置把它当作黑盒调用，CUDA graph 也像录制其他 kernel 一样录下它，所以每次重放都会再运行它，它取到的是编译代码实际算出的张量的 fingerprint。

**probe 可能改变它测量的东西。** 一个小例子：编译代码计算 `s = x + o_proj(h)` 并把 `s` 传给下一个 norm。加法是逐元素的，所以 Inductor 把它融合进 norm kernel，而 GEMM 先对输出舍入。加一个读取 `s` 的 probe，`s` 就有了一个非逐元素的使用者；Inductor 于是把加法折叠进 GEMM（`addmm`），在唯一的一次舍入之前就做了加法。对 `s` 的探测改变了 `s`。

**规则。** 从编译代码内部观察它，并让观察者对编译器不可见：probe 必须读取一个编译器无法融合穿透的精确副本。然后用有 probe 与无 probe 两次运行的比较来证明这一点。这是一个一般测量原则的实例：测量工具不能改变被测的量。

## 步骤

### 探测编译代码并用 OpGuard 比较

**目标：** 编译代码和 graph 重放内部选定张量的 fingerprint，且 OpGuard 的 harness 能读取。

**做法：** [graph_probe.py](../../tools/graph_probe.py)。

1. 在编译之前，用 eager 构建惰性缓存（RoPE 表之类）。
2. 创建一个 `ProbeSet`，在需要的边界调用 `ps.probe(name, *tensors)`，每步调用一次 `ps.step()`。每个 probe 是一个自定义算子（`graph_probe::record`），Dynamo 保留它，Inductor 把它当作不透明 kernel 调用；它把 `xor_signature`（可选 `xor_vectors`）写进设备上的环形缓冲区，所以每次 graph 重放写自己的一行。
3. 保持 `isolate=True`（默认）：在编译下，probe 读取的是一个精确的逐元素副本（`neg(neg(x))`，整数用 `x ^ 0`），而不是张量本身。
4. 带 probe 和不带 probe 各运行一次，逐位比较输出。
5. `ps.read()` 和 `ps.emit(writer)` 为每一步、每个被探测的张量写一条 `TraceWriter` 记录；`export_pair` 和 OpGuard 的 harness 照常比较它们（[定位](localize-with-opguard.md)）。

| 选项 | 给出什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 对 eager 重放做 eager OpGuard 采集 | 每个边界、调用点，不用写新代码 | 每个边界很小 | 症状在 eager 下能复现 |
| CUDA graph 内的 graph probe | graph 重放产生的比特 | 每个 probe 很小（只有它的 kernel）；小 step 上 probe 多了累加起来就大 | 生产重放 CUDA graph |
| 不用 graph、只用 `torch.compile` 时的 graph probe | 编译代码产生的比特 | 大：每个 probe 都要运行自定义算子的 Python 函数体 | 只编译不用 graph；少探测几个张量 |
| 每个舍入算子都不透明（`bi_opaque_all`-style 自定义算子） | 编译后的比特等于 eager 的比特 | 这里没有测量 | 需要编译等于 eager，而不仅仅是采集 |

**成功：** probe 的记录与同一张量上的 `opguard_bridge.xor_signature` 相等；带与不带 probe 的输出逐位相等；OpGuard 在导出的一对上给出的 pivot 与直接比较记录的结果相同。

这在一种 GPU 上、用一个小模型的解码 step 得到证明：probe 的 fingerprint 与 bridge 和 OpGuard 自己的函数一致，OpGuard 在每一对上都找到了与直接比较相同的首个分歧点，并且有隔离时，probe 在每种编译模式下都没有改变编译后的比特（[报告](../../examples/graph_probe/REPORT.md#what-this-shows)）。

### 解读 probe 显示的内容

| 观察 | 含义 |
|---|---|
| 编译结果在第一个 norm 处就与 eager 不同 | Inductor 融合的 norm 有自己的顺序：对原生算子来说是预期的 |
| CUDA graph 在相同形状下等于 eager，padding 进更大的桶后不同 | GEMM 按桶大小选择了 kernel：一个批变的 kernel（L2） |
| 只有每个舍入算子都不透明时，编译结果才等于 eager | 差异来自融合代码；在编译下，只有 batch-invariant kernel 还不够 |

首个差异之后，每个后续 probe 也都不同；probe 无法说明后面的算子在输入相等时是否会一致。要回答这个问题，需要单独编译每个算子（[注意力：服务层](../3-kernels/attention.md)）。

## 可能出错的地方

!!! warning "没有隔离的 probe"
    对一个张量加 probe 会给它一个非逐元素的使用者，Inductor 可能因此改变计划（把加法折叠进 `addmm`）。你测到的是一个生产从不运行的版本。保持 `isolate=True`，并做有无 probe 的对比检查。

!!! warning "在编译代码内部填充的惰性缓存"
    首次使用时才构建的表，如果首次使用发生在编译代码中，就会由 Inductor 计算，然后供之后的每次运行使用，包括 eager。eager 运行也随之改变。在编译之前用 eager 构建这类缓存。

!!! warning "用 `clone()` 来隔离"
    编译器会把 clone 当作空操作删除，而融合 kernel 内部的 dtype view 可能编译失败。probe 于是没有被隔离。使用工具所用的双重取负。

!!! warning "会变化的形状"
    probe 的槽位保留第一次调用时的形状，动态形状没有测试过。同名下换一个形状是错误的。给它另起一个名字。

## 小结

- `torch.compile` 融合算子并自己选择归约顺序；CUDA graph 按 padding 后的形状重放固定的 launch。两者都是与 eager 不同的算术。
- OpGuard 包装的是 Python 调用，而编译代码和 graph 重放一个调用也不发出，所以 eager 采集显示的是 eager 的比特：编译代码要从内部采集。
- probe 是一个带设备环形缓冲区的自定义算子；graph 重放写各自的行。
- 每个 probe 都要用精确的逐元素副本隔离，并用有无 probe 的对比来证明。
- 在编译之前用 eager 构建惰性缓存。

## 深入阅读

- [报告：torch.compile 与 CUDA graph 内部的 fingerprint](../../examples/graph_probe/REPORT.md)：比较过的每种模式、桶和算子集，probe 的影响，代价，复现命令。
- 代码：[graph_probe.py](../../tools/graph_probe.py) 和 [graph_probe_cases.py](../../examples/graph_probe/graph_probe_cases.py)。
- [注意力](../3-kernels/attention.md)：分页、varlen 和编译后的注意力。
- [用 OpGuard 定位](localize-with-opguard.md)：拿到记录之后怎么做。
