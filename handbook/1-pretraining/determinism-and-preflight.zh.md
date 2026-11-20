# 确定性与预检

## 我们在哪里

这是[预训练流程](README.md)的第 2 步。你已经选好了参考运行，也写下了要测试的那一个差异。现在还不知道的是：两次运行在别的地方是不是也有不同。只要有一个没人记下来的设置不一样，整个比较就失去了意义。所以这一步先控制可避免的非确定性，再从每个训练任务内部证明，两次运行之间只剩下声明过的那个差异。

## 思路：你配置的，不一定就是实际生效的

举个小例子。两次运行从同一个 checkpoint 恢复训练。suspect 运行恢复了模型和 Adam 的两个矩，却没有恢复优化器的 step 计数器。Adam 的偏差校正要除以一个由这个计数器算出来的系数，所以恢复后的第一次更新，两次运行的步长就不一样。刚恢复时两次运行的模型权重还相等，到下一次前向就不相等了。

启动命令里看不出任何问题，因为差异藏在训练任务内部的状态里。库在 import 时悄悄设置的开关、只在某一个节点上设置的环境变量、种子不同的 dataloader 生成器，都是同样的情况。

**规则。** 先控制可避免的变化，再在运行中的训练任务内部把设置读回来：每个 rank 都要读，时机是恢复之后、第一步之前。然后对两次运行做 diff。这是第 0 部分"受控与残余非确定性"原则的具体应用：消除那些让两次正确运行做出不同算术的因素，让真正的故障保持可见。

## 步骤

### 2a. 控制五类可避免的变化

把这五类都写进代码或启动配置，每个 rank 上都要如此，这样两次正确的运行才不会产生差异。

| 会变化的地方 | 要设置什么 | 代价等级 |
|---|---|---|
| 随机性和数据顺序 | CPU 和 CUDA 的 RNG 种子、dropout 的随机流、dataloader 的 worker、sampler 的状态、分布式初始化；numpy 的种子也要设 | 免费 |
| 库里的 kernel | `torch.use_deterministic_algorithms(True)` 并设 `warn_only=False`；cuDNN 用确定性模式，关闭 autotune，关闭 TF32，设置 `CUBLAS_WORKSPACE_CONFIG=:16:8`；融合注意力和 MoE kernel 用确定性版本；工具链、GPU 架构和 SM 数都保持相同 | 从免费到大：严格模式可能占掉一个 step 的很大一部分时间，只替换少数几个固定顺序的 kernel 则便宜得多（[训练 step](training-step.md)） |
| 集合通信 | 算法、协议、拓扑和分桶方式，保证部分和按同一棵树相加 | 从小到大，取决于消息大小（[数据并行](data-parallel.md)） |
| 从 checkpoint 恢复训练 | RNG 状态、优化器状态（包括 step 计数器）、scheduler、GradScaler、dataloader 的位置，每个 rank 都要恢复 | 免费 |
| TP 规模 | TP 模拟模式：逻辑上的 TP 保持大于 1，但每个 rank 都按不切分的方式计算，于是 TP=N 和 TP=1 的结果逐 bit 一致 | 这里没有测量 |

**不要**去控制调度，也不要去压住你自己或第三方 kernel 的不确定性。如果它们改变了结果，这个变化正是 bug 的信号。

### 2b. 从每次运行里读回设置，再做 diff

为了证明两次运行的实际设置就是你以为的那样，在每个训练任务里，恢复之后、第一步之前，调用 [determinism_preflight.py](../../tools/determinism_preflight.py) 提供的 `determinism_preflight.report(...)`。它在每个 rank 上读回下列内容，写成 JSON：确定性开关和它的 `warn_only`；cuDNN、TF32 和 matmul 精度；`CUBLAS_WORKSPACE_CONFIG`；SDPA 后端；`NCCL_*` 和 NCCL 版本；RNG 状态；dataloader 的 worker 和生成器；优化器走的代码路径（fused 或 foreach）及其状态；scheduler 的计数器；AMP；是否用了 compile 和 CUDA graph；GPU、驱动和库的版本。装了 OpGuard 时，还会附上它的 `doctor` 输出。之后，`diff(a, b)` 列出两次运行的不同之处，并指出其中哪些会让逐 bit 比较失去意义；`findings` 列出单次运行里没有控制住的东西。`findings` 为空、diff 里只剩下声明过的那个差异，检查就通过了。

在一次已知答案的演练里（一块 GPU、一个 rank、跑几步），预检在采集之前就发现了两个问题：注意力的反向跑在 `warn_only` 下，以及 step 计数器没有恢复；每一项被故意漏掉的恢复状态，也都出现在各自的字段里。被挪动过的舍入点则没有被发现，因为那是代码的改动，不是设置（[报告](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight)）。

## 可能出错的地方

!!! warning "`warn_only=True`"
    设了 `warn_only=True` 的确定性模式，在某个 kernel 没有确定性版本时只发一个警告，于是注意力的反向仍然是不确定的。运行看起来受控，其实没有。改用 `warn_only=False`，或者换掉这个 kernel（[注意力](../3-kernels/attention.md)）。

!!! warning "代码上的差异不是设置"
    预检读的是设置和状态。挪动过的类型转换、换掉的 kernel，都能悄无声息地通过预检。代码上的差异只会在采集时显现出来。

!!! warning "生成器的状态说明不了数据读到了哪里"
    预检看到的是 loader 生成器的状态，而不是设种子之后已经消耗了多少个 batch。所以恢复时跳过的 batch 数不对，预检也会通过。要显式地恢复并比较 sampler 的位置。

!!! warning "读取报告的时机不对"
    在恢复之前读的报告，反映的是全新的状态，而不是恢复后的状态。于是 diff 很干净，两次运行却并不相同。报告要在恢复之后、第一步之前读取。

!!! warning "某个库的 RNG 没人设种子"
    如果工作负载里有一个随机源从来没设过种子（numpy 很常见），它在每一对运行里都不一样，所有比较都做不下去。即使你认为没有代码用它，也要给它设种子。

## 小结

- 控制五类可避免的变化：随机性、库里的 kernel、集合通信、恢复训练的状态、TP 规模。调度和你自己写的 kernel 不要动。
- 在每个训练任务内部、每个 rank 上，恢复之后把设置读回来。
- 对两次运行做 diff，结果里只能剩下声明过的那个差异。
- 预检看的是设置，不是代码，所以舍入方式的改动只会在采集时显现。

## 深入阅读

- [determinism_preflight.py](../../tools/determinism_preflight.py)：工具本身。
- [报告：预检发现了什么，以及每漏掉一项恢复状态时的表现](../../examples/pretraining_e2e/REPORT.md#the-determinism-preflight)。
- [训练 step](training-step.md) 列出完整的训练 step 中哪些设置和状态有影响，以及每种修法的代价。
- [数据并行](data-parallel.md)讲梯度的集合通信该怎样设置。
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)。
