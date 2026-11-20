# 用了 torch.compile 或 CUDA graph 时怎样采集

## 我们在哪里

你已经能用 OpGuard 采集一对运行，并从首个分歧点追到原因（[用 OpGuard 定位](localize-with-opguard.md)）；前两页讲的是出在[训练 step](training-step.md) 和[梯度的集合通信](data-parallel.md)里的原因。这一页讲的是采集本身的一个盲区。很多训练任务并不是一个算子接一个算子地运行模型，而是经过 `torch.compile` 运行，或者以 CUDA graph 的形式重放。为了采集，OpGuard 会让任务逐个算子地运行（eager 模式），而 eager 模式跑的 kernel 和编译后的任务并不一样。所以你采集到的数值，未必就是生产环境里实际算出、并且出现了分歧的那些数值。

你会在[流程](README.md)的两个步骤里来到这一页。第 3 步（采集）时，如果生产环境的训练任务是编译过的，就来这里；第 5 步（判断原因类别）时，如果用 eager 重放看不到症状，或者 pivot 落在编译器生成的代码里，也来这里。这一页先讲编译器和 CUDA graph 各自做了什么、为什么都会让结果变掉，再讲怎样从它们内部按 OpGuard 的格式记录 fingerprint，让流程的其余部分照常进行。

## 编译之后，代码做了什么

### torch.compile 把许多小算子合成少数几个生成的 kernel

在 eager 模式下，Python 每调用一个算子，PyTorch 就立刻运行它，一个算子对应一个 kernel。每个 kernel 从 GPU 显存读入输入，算完后把输出写回显存，写回时按输出的 dtype 舍入。一个由几个小算子拼成的 norm，就要跑同样多个 kernel，每个都写出一个舍入过的中间结果。

`torch.compile` 先把 Python 代码读一遍（由它的前端 Dynamo 做 trace），把其中的算子记成一张图，再把图交给编译器（默认是 Inductor），由编译器为这张图写出新的 kernel。编译器会把一串逐元素算子，连同夹在其中的归约，**融合**进同一个 kernel：中间值都留在寄存器里（张量是 bf16 时用 fp32），只把最终结果写出去。每个归约怎样分给各个线程，也由编译器自己决定。它还会改写一些固定的模式，比如矩阵乘后面跟一个加法，可以变成一个顺带做加法的矩阵乘（`addmm`）。

<figure class="fx" data-fig="fusion" aria-label="同一个 norm 按 eager 运行，和编译成一个 kernel 运行">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">eager：一个算子一个 kernel</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-node is-mono" data-hot="1">x²</span><span class="fx-chip" data-at="1">↓ bf16</span></span><span class="fx-arrow" data-hot="2" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="2">均值</span><span class="fx-chip" data-at="2">↓ bf16</span></span><span class="fx-arrow" data-hot="3" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="3">rsqrt</span><span class="fx-chip" data-at="3">↓ bf16</span></span><span class="fx-arrow" data-hot="4" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="4">× x</span><span class="fx-chip" data-at="4">↓ bf16</span></span><span class="fx-arrow" data-hot="5" aria-hidden="true"></span><span class="fx-col"><span class="fx-node is-mono" data-hot="5">× w</span><span class="fx-chip" data-at="5">↓ bf16</span></span></div></div>
<div class="fx-row"><span class="fx-label">编译后：一个生成的 kernel</span><div class="fx-chain is-tight"><span class="fx-col"><span class="fx-box to-key" data-hot="6"><span class="fx-note">一个 kernel</span><span class="fx-chain is-tight fx-sweep" data-hot="6"><span class="fx-chip">x²</span><span class="fx-chip">均值</span><span class="fx-chip">rsqrt</span><span class="fx-chip">× x</span><span class="fx-chip">× w</span></span><span class="fx-note">中间值都留在 fp32 寄存器里</span></span><span class="fx-chip" data-at="7">↓ bf16，只写出结果</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li>同一个 norm，写成五个算子；w 是它的权重。</li>
<li>eager 模式下一个算子一个 kernel。x² 先写进显存，舍入成 bf16。</li>
<li>求均值时再把它读回来，做归约，写出一个舍入过的结果。</li>
<li>rsqrt 又是一个 kernel，又舍入一次。</li>
<li>× x 又是一个 kernel，又舍入一次。</li>
<li>做完 × w，一共跑了五个 kernel，显存里留下五个舍入过的中间结果。</li>
<li data-hold="2">编译之后，一个生成的 kernel 在 fp32 寄存器里跑完全部五个算子，并按自己的顺序求均值。</li>
<li data-hold="2">它只写出最终结果：舍入一次，而不是五次。</li>
</ol>
<figcaption>图：同一个 norm（w 是它的权重）。eager 模式把每个中间结果按它的 dtype 写进显存；融合后的 kernel 把中间结果留在寄存器里，按自己的顺序求均值，只写出最终结果。</figcaption>
</figure>

### CUDA graph 把 kernel 的 launch 录制一次，之后反复重放

从 Python 一个个地 launch kernel，每个 kernel 都要花 CPU 时间。step 很小的时候，这部分时间甚至可能比 GPU 上的计算还长。**CUDA graph** 把一串 launch（哪些 kernel、什么参数、用哪些缓冲区）录制一次，之后用一次调用重放整串 launch。重放时不运行 Python，也不会重新做任何选择：kernel、形状和显存地址都和录制时一模一样。

所以一张 graph 只适合一种形状。输入大小多变的训练任务会录制几张 graph，每个大小区间（桶）一张，再把每个输入 padding 到比它大的最近一个录制尺寸。

<figure class="fx" data-fig="graph-pad" aria-label="5 行的 batch 在为 8 行录制的 graph 里重放">
<div class="fx-stage">
<div class="fx-chain is-tight"><span class="fx-box to-key" data-hot="2"><span class="fx-swap"><span class="fx-note" data-to="2">5 行的 batch</span><span class="fx-note" data-at="2">为 8 行录制的 graph</span></span><span class="fx-cells fx-sweep" style="--n:8;--cw:1.7rem" data-hot="2"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span><span class="fx-cell is-pad" data-at="1">5</span><span class="fx-cell is-pad" data-at="1">6</span><span class="fx-cell is-pad" data-at="1">7</span></span></span><span class="fx-arrow" data-at="3">保留</span><span class="fx-box" data-at="3"><span class="fx-note">前 5 行</span><span class="fx-cells" style="--n:5;--cw:1.7rem"><span class="fx-cell is-mine">0</span><span class="fx-cell is-mine">1</span><span class="fx-cell is-mine">2</span><span class="fx-cell is-mine">3</span><span class="fx-cell is-mine">4</span></span></span></div>
</div>
<ol class="fx-say">
<li>来了一个 5 行的 batch。</li>
<li>它被 padding 到下一个录制过的尺寸：补 3 行 padding（第 5 到 7 行），凑成 8 行。</li>
<li data-hold="2">为 8 行录制的 graph 开始重放，其中每个 kernel 看到的都是 8 行。</li>
<li data-hold="2">最后只保留前 5 行作为结果。</li>
</ol>
<figcaption>图：5 行的 batch 在为 8 行录制的 graph 里运行。结果只取真实的那几行，但重放中的每个 kernel 处理的都是 8 行。</figcaption>
</figure>

## 编译后的代码为什么会算出不同的结果

**融合挪动了舍入点。** 举一个 bf16 的小例子：在 1 附近，相邻两个 bf16 数的间距是 ε = 2⁻⁷。令 a = b = 1 + ε，c = −(1 + 2ε)，计算 a · b + c。

| | eager：两个 kernel | 编译后：一个融合的 kernel |
|---|---|---|
| a · b | 1 + 2ε + ε²，写出时舍入成 bf16：ε² 不到半个间距，所以结果是 1 + 2ε | 1 + 2ε + ε²，以 fp32 留在寄存器里 |
| + c | (1 + 2ε) − (1 + 2ε) = 0 | ε² |
| bf16 下的结果 | 0 | ε² |

数学上完全一样，舍入的次数却不同。融合后的 kernel 自己选的归约顺序，也会像第 0 部分那三个数一样改变求和结果（[第 0 部分](../0-foundations/README.md#floating-point-addition-is-not-associative)）。

**按 padding 后的尺寸重放，改变了 GEMM 看到的形状。** padding 的行不属于结果，却属于矩阵：重放中的 GEMM 乘的是 8 行，而不是 5 行。GEMM 库会根据形状选 kernel，也就同时选定了归约顺序（[第 0 部分：kernel 按输入的形状决定归约顺序](../0-foundations/README.md#a-reductions-order-depends-on-shape)）。所以这 5 行真实数据的舍入，可能和 eager 下直接算 5 行时不同。batch-invariant 的 GEMM（[第 3 部分](../3-kernels/gemm-and-quantized.md)）可以消除这一项影响。

**规则。** 编译后的代码和它所来自的 eager 代码，做的是不同的算术：舍入点不同，归约顺序不同，形状也可能不同。在 eager 模式下做的比较，说明的只是 eager 模式。

## 常规采集为什么看不到编译后代码的内部

OpGuard 的采集方式，是包装模型 Python 代码里的调用：被包装的调用一运行，包装层就对它的输入和输出取 fingerprint（[用 OpGuard 定位](localize-with-opguard.md)）。编译后的代码不会再逐个发出这些调用。编译器把 Python 读过一遍之后，运行的就是它自己生成的 kernel；而 graph 重放根本不运行 Python。在编译后的图或一次重放内部，已经没有可以包装的 Python 调用了。

所以常规采集只能让训练任务以 eager 模式运行，也就是上一节说的另一种算术。如果症状在 eager 模式下也能复现，这就够了；如果不能，就必须把观察点放进编译后的代码内部。

## 思路：放一个编译器会保留、又没法跨过它做优化的观察点

**编译后代码里的观察点，必须是编译器会保留的算子。**自定义算子就满足这一点：编译器会在图中它原来的位置把它当作黑盒来调用，CUDA graph 也会像录制其他 kernel 一样把它录下来。所以每次重放都会再运行它，它取到的正是编译后的代码实际算出的张量的 fingerprint。

**probe 可能改变它要测的东西。** 举个小例子：编译后的代码计算 `s = x + o_proj(h)`，再把 `s` 传给下一个 norm。这个加法是逐元素的，所以 Inductor 会把它融合进 norm 的 kernel，而 GEMM 先对自己的输出舍入。现在加一个读取 `s` 的 probe。`s` 于是多了一个不是逐元素的使用者，Inductor 就改为把加法折叠进 GEMM（`addmm`），在 GEMM 唯一的那次舍入之前就做了加法。也就是说，对 `s` 的探测改变了 `s` 本身。

**规则。** 要从编译后代码的内部去观察它，并且让观察点对编译器不可见：probe 读取的必须是张量的一个精确副本，而且编译器无法穿过这个副本做融合。然后要证明这一点：把带 probe 的运行和不带 probe 的运行逐 bit 比较。这是一条通用的测量原则：测量工具不能改变被测的量。

## 步骤

### 在编译后的代码里放 probe，再用 OpGuard 比较

**目标：** 在编译后的代码和 graph 重放内部，拿到选定张量的 fingerprint，并且格式能被 OpGuard 的 harness 读取。

**做法：** 用 [graph_probe.py](../../tools/graph_probe.py)。

1. 在编译之前，先用 eager 模式把惰性缓存（RoPE 表之类）建好。
2. 创建一个 `ProbeSet`，在你想观察的边界处调用 `ps.probe(name, *tensors)`，每一步调用一次 `ps.step()`。每个 probe 都是一个自定义算子（`graph_probe::record`），Dynamo 会保留它，Inductor 会把它当作不透明的 kernel 来调用。它把 `xor_signature`（也可以加上 `xor_vectors`）写进设备上的一个环形缓冲区，所以每次 graph 重放都会写下自己的一行。
3. 保持 `isolate=True`（默认就是）。这样在编译模式下，probe 读到的是张量的一个精确的逐元素副本（`neg(neg(x))`，整数则用 `x ^ 0`），而不是张量本身。
4. 带 probe 和不带 probe 各运行一次，逐 bit 比较两次的输出。
5. `ps.read()` 和 `ps.emit(writer)` 为每一步、每个被探测的张量写一条 `TraceWriter` 记录；之后 `export_pair` 和 OpGuard 的 harness 照常比较这些记录（[定位](localize-with-opguard.md)）。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 用 OpGuard 对 eager 重放做 eager 采集 | 每个边界和它的调用点，不用写新代码 | 每个边界都很小 | 症状在 eager 模式下能复现 |
| 在 CUDA graph 里放 graph probe | graph 重放实际算出的结果 | 每个 probe 很小（只多它自己的几个 kernel）；step 很小而 probe 很多时，加起来就大了 | 生产环境用 CUDA graph 重放 |
| 只用 `torch.compile`、不用 CUDA graph 时放 graph probe | 编译后的代码实际算出的结果 | 大：每个 probe 都要运行一遍自定义算子的 Python 函数体 | 只编译、不用 graph；只探测少数几个张量 |
| 把每个会舍入的算子都做成不透明的（类似 `bi_opaque_all` 的自定义算子） | 编译后的结果和 eager 逐 bit 一致 | 这里没有测量 | 你需要编译结果等于 eager 结果，而不只是采集 |

**成功标准：** probe 的记录和同一张量上的 `opguard_bridge.xor_signature` 相等；带与不带 probe 的输出逐 bit 一致；OpGuard 在导出的这一对运行上找到的 pivot，和直接比较记录得到的结果相同。

这一点在一种 GPU 上、用一个小模型的 decode step 得到了证明：probe 的 fingerprint 和 bridge 以及 OpGuard 自己的函数都一致；在每一对运行上，OpGuard 找到的首个分歧点都和直接比较的结果相同；做了隔离之后，probe 在每种编译模式下都没有改变编译后的结果（[报告](../../examples/graph_probe/REPORT.md#what-this-shows)）。

### 解读 probe 看到的结果

| 你观察到的现象 | 说明什么 |
|---|---|
| 编译后的结果在第一个 norm 处就和 eager 不同 | Inductor 融合出的 norm 按自己的顺序求和；用原生算子时这是预期之内的 |
| 形状相同时，CUDA graph 和 eager 一致；padding 进更大的桶之后就不同了 | GEMM 按桶的大小选了 kernel，说明这个 kernel 的结果随 batch 而变（不满足 L2） |
| 只有把每个会舍入的算子都做成不透明的，编译后的结果才和 eager 一致 | 差异来自融合后的代码；在编译模式下，光有 batch-invariant 的 kernel 还不够 |

一旦出现第一个差异，之后的每个 probe 也都会不同，所以 probe 回答不了"如果输入相等，后面的算子会不会一致"。要回答这个问题，需要把每个算子单独编译来测（[注意力：推理服务中的各层](../3-kernels/attention.md)）。

## 可能出错的地方

!!! warning "没有隔离的 probe"
    给一个张量加上 probe，就给它添了一个不是逐元素的使用者，Inductor 可能因此改变计划（比如把加法折叠进 `addmm`）。这样测到的，是一个生产环境从来不会跑的版本。保持 `isolate=True`，并做带与不带 probe 的对比。

!!! warning "惰性缓存在编译后的代码里才被填充"
    首次使用时才构建的表，如果首次使用发生在编译后的代码里，就会由 Inductor 来算，然后提供给之后的每一次运行，eager 运行也不例外。于是 eager 运行的结果也跟着变了。要在编译之前，用 eager 模式把这类缓存建好。

!!! warning "用 `clone()` 来做隔离"
    编译器会把 clone 当作无用操作删掉，而在融合后的 kernel 里做 dtype view 可能根本编译不过。两种情况下 probe 都没有被隔离。要用工具里用的双重取负。

!!! warning "会变化的形状"
    probe 的槽位会记住第一次调用时的形状，而动态形状还没有测试过。在同一个名字下换一个形状，记录就是错的。换一个形状就换一个名字。

## 小结

- `torch.compile` 会融合算子，并自己选择归约顺序；CUDA graph 则按 padding 后的形状重放一串固定的 launch。两者做的都是和 eager 不同的算术。
- OpGuard 包装的是 Python 调用，而编译后的代码和 graph 重放一个这样的调用也不发出，所以 eager 采集看到的只是 eager 算出的结果。编译后的代码要从内部采集。
- probe 是一个自定义算子，它把结果写进设备上的环形缓冲区；每次 graph 重放都写下自己的一行。
- 每个 probe 都要通过一个精确的逐元素副本来隔离，并用带与不带 probe 的对比来证明隔离有效。
- 在编译之前，用 eager 模式把惰性缓存建好。

## 深入阅读

- [报告：torch.compile 和 CUDA graph 内部的 fingerprint](../../examples/graph_probe/REPORT.md)：比较了每种编译模式、每个桶和每套 kernel，还有 probe 本身的影响、各项代价，以及复现结果的命令。
- 代码在 [graph_probe.py](../../tools/graph_probe.py) 和 [graph_probe_cases.py](../../examples/graph_probe/graph_probe_cases.py)。
- [注意力](../3-kernels/attention.md)讲 paged、varlen 和编译后的注意力。
- [用 OpGuard 定位](localize-with-opguard.md)讲拿到这些记录之后该怎么做。
