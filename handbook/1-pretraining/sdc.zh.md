# 静默数据损坏（SDC）

## 我们在哪里

这一页用于[预训练流程](README.md)的第 5 步（判断原因类别），也用于长时间训练开始之前。前面几页讲的都是软件上的原因：[训练 step](training-step.md)、[集合通信](data-parallel.md)和 [torch.compile 与 CUDA graph](capture-compiled-code.md)。排除了软件原因之后，你会来到这一页：比如同一个任务重跑一遍，设置和代码都没变，结果却不同；某个 rank 和其他 rank 不一致；或者一个节点刚修好回来。这时要回答的问题是：设备本身是不是在不报错的情况下返回了错误的数值？这就是静默数据损坏（SDC）。

## 思路：输入、kernel 和软件栈都相同，结果却不一致，就是设备有故障

**定义。** 如果在同一个软件栈上，对完全相同的输入运行完全相同的确定性 kernel，一个设备却给出了不一样的结果，这个设备就有故障。这句话里每个条件都不能少：一个本身不确定的算子，或者两块驱动不同的设备，在毫无故障时结果也会不同。

所以扫描要做两件事：只运行结果逐 bit 固定的计算；对输出做摘要时，要保证任何损坏都能显现出来。第 0 部分的 XOR fingerprint 能发现任何单个 bit 的翻转，但扫描需要更多，因为一个故障可能把同一个 bit 翻转两次。下面以一个张量的 32-bit 字为例：

| 故障 | `xor32` | `sum64`（各字之和 mod 2^64） | `wsum64`（(2i+1)·w_i 之和） |
|---|---|---|---|
| 翻转了一个 bit | 变了 | 变了 | 变了 |
| 两个字里的同一个 bit 都被翻转，都是 0 变 1 | 没变 | 变了 | 变了 |
| 两个字里的同一个 bit 都被翻转，一个 0 变 1，一个 1 变 0 | 没变 | 没变 | 变了 |
| 两个字互换了位置 | 没变 | 没变 | 变了 |

**规则。** 只对确定性的计算下判定；只在软件栈完全相同的一组设备内比较 fingerprint；同时使用多种 fingerprint，让任何简单的故障模式都无法相互抵消。设备之间不一致时，先怀疑软件：不一致只是一个有待解释的事实，还不是判定。

## 步骤

### 1. 扫描设备

**目标：** 给每个设备一个通过或失败的结论，失败时指出是哪个算子、哪一轮重复。

**做法：** [sdc_scan.py](../../tools/sdc_scan.py) 在每块 GPU 上起一个 worker，运行一套固定的、设了种子的 golden 工作负载，并在设备上比较 fingerprint。这套工作负载包括 GEMM、注意力、norm、softmax、逐元素数学运算、gather 和 scatter、排序、整数运算、内存读写模式、长归约、一次优化器 step、一个小的训练 step，以及每一对 GPU 之间的 NCCL 集合通信。

```
python tools/sdc_scan.py --quick --json out.json        # few repeats + fault-injection self-test
python tools/sdc_scan.py --json out.json                # more repeats, NCCL pairs, self-test
python tools/sdc_scan.py --phases stress --stress SECONDS [--offsets] --part s.json
python tools/sdc_scan.py --merge a.json b.json --json fleet.json --record-golden g.json
python tools/sdc_scan.py --golden g.json ...            # compare with a recorded run
python tools/sdc_scan.py --list-ops
```

| 模式 | 比较什么 | 能发现什么 | 发现不了什么 | 代价等级 |
|---|---|---|---|---|
| 自重放 | 在同一个设备上，把每一轮重复和第一轮比较 | 偶发故障和竞争 | 每一轮都存在的故障 | 小 |
| 已知答案 | 把精确算子（不舍入）的结果和 CPU 比较 | 不舍入的算子中的持续性故障；还能说明哪一边是对的 | 会舍入的算子 | 已包含 |
| fleet | 把每个设备和同组的多数设备比较（需要三个或更多设备） | 任何算子中的持续性故障 | 整组设备共有的故障 | 已包含 |
| golden 文件 | 把每个设备和之前为它所在组记录的 fingerprint 比较 | 持续性故障，只有一个设备也能查 | 记录这个文件时就已经存在的故障 | 已包含 |
| 压力测试 | 同样的比较，但在负载下持续一段设定的时间，并不断变换显存地址，外加一遍显存扫描 | 只在高温、高功耗或特定地址下才出现的故障 | 只在用户自己的 kernel 和负载下才出现的故障 | 中等 |
| NCCL 配对 | 对每一对 GPU，两个 rank 的结果都和 CPU 算出的答案比较 | 测试过的路径上，链路和拷贝引擎的故障 | 其他拓扑和协议 | 小 |

**组**指的是一批 GPU 型号、SM 数、显存大小、驱动、torch、CUDA 构建、cuDNN、cuBLAS、Triton 和 NCCL 都相同的设备。fleet 和 golden 文件的判定从不跨组比较。

**成功标准：** 每个设备都通过；故障注入自检报出了每一个注入的故障，且没有报出别的；退出码为零。

### 2. 决定什么时候扫描

| 什么时候 | 跑什么 | 代价等级 |
|---|---|---|
| 长时间训练开始之前，每个节点都要 | 完整扫描，并和本组的 golden 文件比较 | 小 |
| 硬件有变动（换了新卡、重新插拔）之后，或者驱动、库升级之后 | 完整扫描，再加一段压力测试；如果是软件有变动，还要重新记录 golden 文件，因为组已经变了 | 中等 |
| 定期，或者节点修好回来时 | `--quick` 加 `--golden` | 小 |
| 训练过程中 | 每 N 步对其中一步用 `sdc_guard`，并以同样的频率做 `cross_rank_check` | N 大时小；每个被守护的 step 多做一次前向 |

### 3. 在训练任务内部检查

**目标：** 抓住只在训练任务自己的 kernel、形状和负载下才出现的故障，这些是扫描测不到的。

**做法：** `sdc_guard(fn, *inputs)` 先返回 `fn(*inputs)`，然后在关闭 autograd、恢复随机状态的情况下再算一遍 `fn`，比较两次的 fingerprint；不一致时抛出 `SDCMismatch`，指出是哪个输出、哪些字不同。`fn` 必须是确定的，而且不能改变任何状态。`cross_rank_check(fn, *inputs)` 把 rank 0 的 micro-batch 广播出去，让每个 rank 都在它上面算一遍 `fn`，再把输入和输出的 fingerprint 做 all-gather。只有两个 rank 时，只能知道它们不一样；有三个或更多 rank，才能指出哪一个是出错的。`replica_check(params)` 比较各个副本的参数。示例见 [guard_demo.py](../../examples/sdc_scan/guard_demo.py)。

**成功标准：** 在健康的 step 上没有误报；故意植入的一个 bit 翻转，能在正确的元素和 rank 上被报出来。

### 4. 同组设备不一致时怎么办

1. 先排除软件：比较组的标识和各主机上的软件栈。
2. 在这个设备上重复运行。如果 fingerprint 每次都在变，指向的是硬件；如果每次都一样，仍可能是软件差异，或者是一个持久性的硬件故障。
3. 下结论。对精确算子来说，CPU 的答案能说明哪一边是对的。否则就把卡换一个插槽，重跑一遍，并在两块设备上采集这个算子的输出，看看数值到底差在哪里。

### 往 golden 工作负载里添加算子

```python
@op("my_op_bf16", "my_class", exact=False, full=False,
    x=N(ROWS, COLS, dtype=BF16), idx=I(ROWS, hi=COLS))
def _my_op(x, idx):
    return {"out": my_kernel(x, idx), "aux": ...}   # tensor or dict of tensors
```

- 输入用规格来描述：`N` 是钟形分布的浮点数，`NZ` 是非零浮点数，`U` 是互不相同的浮点数，`I` 是某个范围内的整数，`R` 是随机的 64-bit 字，`P` 是一个排列，`K` 是一个 mask，`L` 是指数分布很宽的浮点数，`W` 是显存测试用的模式。一个输入的字节只取决于种子和这个输入自己的名字。
- 只有当任何输出都不经过舍入时（整数、拷贝、单次 IEEE 运算），才设 `exact=True`；这时由 CPU 给出已知答案。
- 在一个健康的设备上，对这个算子做多轮重复，再用 `--offsets` 跑一段压力测试。如果有任何一轮不同，就固定一个确定性的版本，或者用 `probe=True, det=False, why=...` 注册它：只记录，从不判定。
- 改变算子的输入或输出，会改变它的 fingerprint，所以改完之后要重新记录 golden 文件。

这一点是在两种 GPU 上、用软件注入故障的方式证明的：自检报出了每一个注入的故障，且没有报出别的；每个健康设备都通过了，没有误报。手上没有真正出故障的硬件可供测试（[报告](../../examples/sdc_scan/REPORT.md#fault-injection-self-test)）。

## 可能出错的地方

!!! warning "被判定的算子里混进了不确定的算子"
    浮点原子操作、默认模式下的 `cumsum`、注意力的反向，即使在健康的设备上结果也会变。扫描于是频繁误报，大家也就不再信任它。只对确定性的算子下判定，其余的作为 probe 记录。

!!! warning "数据在显存中的放置位置改变了结果"
    有些归约在输入按某个边界对齐时给出一个结果，按更小的边界对齐时又给出另一个结果。随放置位置变化的 fingerprint，看起来就像故障。要保持对齐方式固定：偏移量取 256 字节的倍数。

!!! warning "拿 CPU 当会舍入的算子的已知答案"
    CPU 和 GPU 的数学库对 `sqrt` 这类函数的舍入可能不同，而两者都没有坏。这时已知答案检查会把一个健康的设备判为失败。只对不舍入的算子使用 CPU 的答案。

!!! warning "跨组比较"
    驱动或库的版本不同，同样的计算可能算出不同的结果。跨组下判定会把这种差异当成故障。只在组内比较；跨组一致只能作为参考信息。

!!! warning "把通过扫描当成合格证"
    扫描测试的是它自己的算子、形状和负载，而不是训练任务的。训练任务自己的 kernel 里有故障，也可能通过扫描。要用训练任务内部的检查，或者直接在训练任务上用 OpGuard 的 fingerprint。

## 小结

- SDC 指的是：输入相同、确定性 kernel 相同、软件栈也相同，结果却不一致。
- 只对确定性的计算下判定；同时使用多种 fingerprint（`xor32`、`sum64`、`wsum64`），让故障无法相互抵消。
- 长时间训练开始之前和软硬件变动之后要扫描；训练过程中，在任务内部抽样守护部分 step。
- 不一致只是一个有待解释的事实，先从软件查起，查清之后才能称为 SDC。

## 深入阅读

- [报告：扫描 GPU 查找 SDC](../../examples/sdc_scan/REPORT.md)：完整的算子列表和被排除的算子、自检、每个设备的结果、运行时间，以及训练任务内部检查的代价。
- 工具是 [sdc_scan.py](../../tools/sdc_scan.py)；示例和驱动脚本在 [examples/sdc_scan/](../../examples/sdc_scan/)。
- [用 OpGuard 定位](localize-with-opguard.md)：pivot 最终查到硬件故障时该怎么办。
