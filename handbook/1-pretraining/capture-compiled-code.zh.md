# torch.compile 与 CUDA graph 内部的 fingerprint

## 我们在哪里

对于使用 `torch.compile` 或 CUDA graph 的运行，这是[预训练工作流](README.md)的第 3 步（采集）。两次运行已经受控（[第 2 步](determinism-and-preflight.md)）。还没解决的是：OpGuard 采集的是 eager 执行，而编译代码运行的是别的 kernel，所以 eager 采集可能看不到生产实际算出的比特。本页从编译代码和 graph 重放内部记录 OpGuard 格式的 fingerprint，使工作流的其余部分（[定位](localize-with-opguard.md)）无需改动。

## 思路：编译代码是不同的算术，而 probe 可能改变它

**编译改变顺序。** 在 eager 模式下，RMSNorm 是几个 kernel，每个都对输出舍入。Inductor 把 GEMM 之间的算子（norm、RoPE、SiLU·mul、残差加）融合进它自己的 Triton kernel，带有自己的归约顺序和数学函数，还会把 `mm` + add 改写成一个 `addmm`。同一个 norm 于是成了另一个和。

**CUDA graph 改变形状。** graph 重放的是采集时选定的 kernel。一个请求被 padding 进某个桶，就按桶的批大小运行，而 GEMM 库按这个大小选 kernel：这就是第 0 部分 split-K 例子的重演。

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

- 编译代码和 CUDA graph 是与 eager 不同的算术：要从内部采集。
- probe 是一个带设备环形缓冲区的自定义算子；graph 重放写各自的行。
- 每个 probe 都要用精确的逐元素副本隔离，并用有无 probe 的对比来证明。
- 在编译之前用 eager 构建惰性缓存。

## 深入阅读

- [报告：torch.compile 与 CUDA graph 内部的 fingerprint](../../examples/graph_probe/REPORT.md)：比较过的每种模式、桶和算子集，probe 的影响，代价，复现命令。
- 代码：[graph_probe.py](../../tools/graph_probe.py) 和 [graph_probe_cases.py](../../examples/graph_probe/graph_probe_cases.py)。
- [注意力](../3-kernels/attention.md)：分页、varlen 和编译后的注意力。
- [用 OpGuard 定位](localize-with-opguard.md)：拿到记录之后怎么做。
