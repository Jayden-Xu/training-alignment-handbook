# 确定性与预检

## 我们在哪里

这是[预训练工作流](README.md)的第 2 步。你已经选好了参考运行，并写下了待测的那一个差异。还没解决的是：两次运行是否还有别的不同。一个没人写下来的设置会让比较失去意义。所以这一步先控制可避免的非确定性，再从每个作业内部证明只剩下声明过的那个差异。

## 思路：你设置的不等于实际运行的

一个小例子。两次运行从同一个检查点恢复。suspect 恢复了模型和 Adam 的矩，却没有恢复优化器的 step 计数器。Adam 的偏差校正要除以一个由这个计数器算出的因子，所以恢复后的第一次更新在两边大小不同。恢复时模型权重相等；下一次前向就不相等了。

启动命令里看不出这一点。差异存在于作业内部的状态中。库在 import 时设置的开关、某个节点上的环境变量、种子不同的 dataloader 生成器，也是一样。

**规则。** 控制可避免的变化，然后在运行中的作业内部按 rank 读回：在恢复之后、第一步之前，并对两次运行做 diff。这是第 0 部分受控与残余非确定性原则的一个实例：消除让两次正确运行做不同算术的东西，让故障保持可见。

## 步骤

### 2a. 设置五个控制面

**目标：** 消除让两次正确运行产生差异的变化。

| 控制面 | 设置什么 | 代价等级 |
|---|---|---|
| 随机性和数据顺序 | CPU 和 CUDA 的 RNG 种子、dropout 流、dataloader worker、sampler 状态、分布式初始化；也要给 numpy 设种子 | 免费 |
| 库 kernel | `torch.use_deterministic_algorithms(True)` 且 `warn_only=False`，确定性 cuDNN，不做 autotune，关闭 TF32，`CUBLAS_WORKSPACE_CONFIG=:16:8`；融合注意力和 MoE kernel 的确定性版本；相同的工具链、GPU 架构和 SM 数 | 免费到大：严格模式可能占 step 的很大一部分；有针对性的固定顺序 kernel 代价更小（[训练 step](training-step.md)） |
| 集合通信 | 算法、协议、拓扑和分桶，使部分和沿同一棵树相加 | 小到大，取决于消息大小（[数据并行](data-parallel.md)） |
| 检查点恢复 | RNG 流、含 step 计数器的优化器状态、scheduler、GradScaler、dataloader 位置，每个 rank 都要 | 免费 |
| TP 度 | TP 模拟器模式：逻辑 TP 保持大于 1，而每个 rank 运行未切分的算术，于是 TP=N 与 TP=1 逐位一致 | 这里没有测量 |

**不要**固定调度，也不要固定用户和第三方 kernel。如果它们改变了比特，那就是 bug 的信号。

**成功：** 每个控制面都在代码或启动配置中设置好，每个 rank 都是。

### 2b. 报告与 diff

**目标：** 证明两次运行的控制面就是你以为的样子。

**做法：** 在每个作业里调用 [determinism_preflight.py](../../tools/determinism_preflight.py) 的 `determinism_preflight.report(...)`，时机在恢复之后、第一步之前。它按 rank 以 JSON 读回：确定性开关及其 `warn_only`、cuDNN、TF32 和 matmul 精度、`CUBLAS_WORKSPACE_CONFIG`、SDPA 后端、`NCCL_*` 和 NCCL 版本、RNG 状态、dataloader 的 worker 和生成器、优化器路径（fused、foreach）和状态、scheduler 计数器、AMP、compile 和 CUDA graph 的使用、GPU、驱动和库版本。装有 OpGuard 时会嵌入它的 `doctor` 输出。然后 `diff(a, b)` 列出两次运行的不同之处，以及哪些不同会让逐位比较失去意义；`findings` 列出一次运行没有控制住的东西。

**成功：** `findings` 为空，diff 只列出声明过的差异。

在一次已知答案的演练中（一块 GPU、一个 rank、几步），预检在任何采集之前就发现了 `warn_only` 下的注意力 backward 和缺失的 step 计数器，每个被省略的恢复项都出现在各自的字段里；被移动的舍入点没有出现，因为它是代码（[报告](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight)）。

## 可能出错的地方

!!! warning "`warn_only=True`"
    `warn_only=True` 的确定性模式在某个 kernel 没有确定性版本时只发出警告，于是注意力的 backward 仍是非确定的。运行看似受控，其实不是。使用 `warn_only=False`，或替换这个 kernel（[注意力](../3-kernels/attention.md)）。

!!! warning "代码差异不是设置"
    预检读的是设置和状态。被移动的类型转换或不同的 kernel 会不被察觉地通过。代码差异只能在采集中显现。

!!! warning "生成器状态不是数据位置"
    预检看到的是 loader 的生成器状态，而不是之后消耗了多少个 batch。跳过错误数量 batch 的恢复会通过检查。显式恢复并比较 sampler 的位置。

!!! warning "报告取得时机不对"
    在恢复之前，报告显示的是新建的状态，而不是恢复后的状态。于是 diff 干净而运行却不同。在恢复之后、第一步之前取报告。

!!! warning "某个库的 RNG 没有设种子"
    工作负载从未设种子的随机源（numpy 很常见）在每一对运行中都不同。它会挡住每一次比较。即使你认为没有东西用它，也要给它设种子。

## 小结

- 设置五个控制面：随机性、库 kernel、集合通信、恢复状态、TP 度。调度和用户 kernel 不要动。
- 在每个作业内部按 rank 读回，时机在恢复之后。
- 对两次运行做 diff；只能剩下声明过的差异。
- 预检看到的是设置，不是代码：舍入的改变只会在采集中出现。

## 深入阅读

- [determinism_preflight.py](../../tools/determinism_preflight.py)：工具本身。
- [报告：预检发现了什么，以及逐个省略每个恢复项](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight)。
- [训练 step](training-step.md)：完整 step 中哪些设置和状态重要，以及每项修复的代价。
- [数据并行](data-parallel.md)：梯度的集合通信设置。
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)。
