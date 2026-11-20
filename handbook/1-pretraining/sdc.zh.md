# 扫描 GPU 的静默数据损坏

## 我们在哪里

本页服务于[预训练工作流](README.md)的第 5 步（分类），以及长时间运行开始之前。当自重放在没有设置或代码差异的情况下出现不同、某个 rank 与其他 rank 不一致、或者一个节点维修归来时，你会来到这里。还没解决的是：设备本身是否在不报错的情况下返回错误的值？这就是静默数据损坏（SDC）。

## 思路：同样的输入、同样的 kernel、同样的栈，不同的比特

**定义。** 对相同的输入和相同的确定性 kernel，在同一个软件栈上，给出不同比特的设备是有故障的。这句话的每个部分都要紧：一个非确定的算子，或两块驱动不同的设备，在毫无故障时也会给出不同的比特。

所以扫描有两项任务：只运行比特固定的计算，并对输出做摘要，使任何损坏都能显现。第 0 部分的 XOR fingerprint 能抓住任何单个翻转；扫描需要更多，因为故障可能把同一个比特翻转两次。对一个张量的 32 位字举个小例子：

| 故障 | `xor32` | `sum64`（各字之和 mod 2^64） | `wsum64`（(2i+1)·w_i 之和） |
|---|---|---|---|
| 翻转一个比特 | 改变 | 改变 | 改变 |
| 两个字中同一比特翻转，都是 0→1 | 不变 | 改变 | 改变 |
| 两个字中同一比特翻转，一个 0→1，一个 1→0 | 不变 | 不变 | 改变 |
| 两个字交换 | 不变 | 不变 | 改变 |

**规则。** 只判定确定性的计算，只在栈相同的组内比较 fingerprint，并使用多种 fingerprint，使简单的故障模式无法抵消。设备不一致时先怀疑软件：那是一个需要解释的事实，还不是判定。

## 步骤

### 1. 扫描设备

**目标：** 每个设备给出通过或失败，并指出算子和重复轮次。

**做法：** [sdc_scan.py](../../tools/sdc_scan.py) 运行一个固定的、设了种子的黄金工作负载（GEMM、注意力、norm、softmax、逐元素数学、gather 和 scatter、排序、整数运算、内存模式、长归约、一次优化器 step、一个小的训练 step，以及每对 GPU 之间的 NCCL 集合通信），每块 GPU 一个 worker，在设备上比较 fingerprint。

```
python tools/sdc_scan.py --quick --json out.json        # few repeats + fault-injection self-test
python tools/sdc_scan.py --json out.json                # more repeats, NCCL pairs, self-test
python tools/sdc_scan.py --phases stress --stress SECONDS [--offsets] --part s.json
python tools/sdc_scan.py --merge a.json b.json --json fleet.json --record-golden g.json
python tools/sdc_scan.py --golden g.json ...            # compare with a recorded run
python tools/sdc_scan.py --list-ops
```

| 模式 | 比较什么 | 能抓到 | 抓不到 | 代价等级 |
|---|---|---|---|---|
| 自重放 | 在一个设备上，每轮重复与第一轮比较 | 瞬时故障、竞争 | 每轮都存在的故障 | 小 |
| 已知答案 | 精确算子（不舍入）与 CPU 比较 | 不舍入算子中的持久故障；能说明哪一边是对的 | 会舍入的算子 | 已包含 |
| 机群 | 每个设备与其组内多数比较（需要三个或更多） | 任何算子中的持久故障 | 整组共有的故障 | 已包含 |
| 黄金文件 | 每个设备与之前为其组记录的 fingerprint 比较 | 持久故障，单个设备也行 | 记录文件时就已存在的故障 | 已包含 |
| 压力 | 同上，在负载下持续一段设定时间，地址不断变化，外加内存扫描 | 需要高温、功耗或特定地址才出现的故障 | 用户自己的 kernel 和负载下的故障 | 中等 |
| NCCL 配对 | 每对 GPU、两个 rank，与 CPU 的答案比较 | 测试路径上的链路和拷贝引擎故障 | 其他拓扑和协议 | 小 |

**组**是指相同的 GPU 型号、SM 数、显存大小、驱动、torch、CUDA 构建、cuDNN、cuBLAS、Triton 和 NCCL。机群和黄金文件的判定从不跨组。

**成功：** 每个设备都通过，故障注入自检报出每个注入的故障且不报别的，退出码为零。

### 2. 安排时间表

| 何时 | 做什么 | 代价等级 |
|---|---|---|
| 长时间运行之前，每个节点 | 完整扫描，与组的黄金文件比较 | 小 |
| 硬件变动（新卡、重新插拔）或驱动、库升级之后 | 完整扫描加一段压力测试；软件变动之后记录新的黄金文件，因为组已经变了 | 中等 |
| 定期，或节点维修归来时 | `--quick` 加 `--golden` | 小 |
| 训练期间 | 每 N 步对一步用 `sdc_guard`，并以同样频率做 `cross_rank_check` | N 大时小；每个被守护的 step 多一次前向 |

### 3. 在作业内部检查

**目标：** 抓住作业自己的 kernel、形状和负载下的故障，这些是扫描测不到的。

**做法：** `sdc_guard(fn, *inputs)` 返回 `fn(*inputs)`，然后在不带 autograd、恢复随机状态的情况下重新计算 `fn`，比较 fingerprint，不一致时抛出 `SDCMismatch`，指出输出和不同的字；`fn` 必须是确定的，且不能改变状态。`cross_rank_check(fn, *inputs)` 广播 rank 0 的 micro-batch，让每个 rank 在它上面计算 `fn`，并 all-gather 输入和输出的 fingerprint；两个 rank 只能说它们不同，三个或更多才能指出离群者。`replica_check(params)` 比较各副本的参数。示例：[guard_demo.py](../../examples/sdc_scan/guard_demo.py)。

**成功：** 健康的 step 上没有误报，植入的比特翻转在正确的元素和 rank 上被指出。

### 4. 组内设备不一致时

1. 排除软件：比较组键和主机的软件栈。
2. 在该设备上重复。fingerprint 在变化，指向硬件；稳定不变，则仍可能是软件或硬故障。
3. 定论：精确算子的 CPU 答案能说明哪一边是对的。否则换插槽、重跑，并在两块设备上采集该算子的输出，看值是怎样不同的。

### 向黄金工作负载添加算子

```python
@op("my_op_bf16", "my_class", exact=False, full=False,
    x=N(ROWS, COLS, dtype=BF16), idx=I(ROWS, hi=COLS))
def _my_op(x, idx):
    return {"out": my_kernel(x, idx), "aux": ...}   # tensor or dict of tensors
```

- 输入是规格：`N` 钟形分布的浮点数，`NZ` 非零浮点数，`U` 互不相同的浮点数，`I` 某范围内的整数，`R` 随机 64 位字，`P` 一个排列，`K` 一个掩码，`L` 宽指数，`W` 内存测试模式。一个输入的字节只取决于种子和它自己的名字。
- 只有当没有任何输出需要舍入时（整数、拷贝、单次 IEEE 运算）才设 `exact=True`；这时由 CPU 给出已知答案。
- 在健康设备上对该算子做多轮重复，并用 `--offsets` 跑一段压力测试。如果任何一轮不同，就固定一个确定性版本，或用 `probe=True, det=False, why=...` 注册它：记录，但从不判定。
- 改变算子的输入或输出会改变它的 fingerprint：记录新的黄金文件。

这在两种 GPU 上用软件故障注入得到证明：自检报出了每个注入的故障且不报别的，每个健康设备都通过且没有误报。没有可用的真实故障硬件（[报告](../../examples/sdc_scan/REPORT.md#fault-injection-self-test)）。

## 可能出错的地方

!!! warning "被判定的集合里有非确定的算子"
    浮点原子操作、默认模式下的 `cumsum` 或注意力 backward 在健康设备上也会变化。扫描会误报，失去信任。只判定确定性算子；其余作为 probe 记录。

!!! warning "放置位置改变了结果"
    有些归约在输入按某个边界对齐时给出一个结果，按更小的边界对齐时给出另一个。随放置变化的 fingerprint 看起来像故障。保持对齐固定：偏移取 256 字节的倍数。

!!! warning "把 CPU 当作舍入算子的已知答案"
    CPU 和 GPU 的数学库对 `sqrt` 这样的函数可能舍入不同，而两者都没坏。已知答案检查于是会判一个健康设备失败。只对不舍入的算子使用 CPU。

!!! warning "跨组比较"
    两个驱动或库版本在同样的计算上可能算出不同的比特。跨组判定会把这当成故障。只在组内比较；跨组相等只作参考信息。

!!! warning "把通过当成证书"
    扫描测试的是它自己的算子、形状和负载，而不是作业的。作业 kernel 中的故障可能通过。使用作业内检查，或在作业本身上使用 OpGuard 的 fingerprint。

## 小结

- SDC 是相同输入、相同确定性 kernel、相同栈却得到不同比特。
- 只判定确定性的计算；使用多种 fingerprint（`xor32`、`sum64`、`wsum64`），让故障无法抵消。
- 长时间运行之前和变动之后扫描；在作业内部守护抽样的 step。
- 不一致是需要解释的事实，先从软件查起，然后才能称为 SDC。

## 深入阅读

- [报告：扫描 GPU 的 SDC](../../examples/sdc_scan/REPORT.md)：完整的算子列表、被排除的算子、自检、每个设备的结果、运行时间和作业内检查的代价。
- 工具：[sdc_scan.py](../../tools/sdc_scan.py)；示例和驱动脚本：[examples/sdc_scan/](../../examples/sdc_scan/)。
- [用 OpGuard 定位](localize-with-opguard.md)：最终落到硬件故障的 pivot。
