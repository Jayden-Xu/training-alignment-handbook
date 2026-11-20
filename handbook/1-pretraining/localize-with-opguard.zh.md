# 用 OpGuard 定位

## 我们在哪里

这一页是[预训练流程](README.md)的核心循环，涵盖第 3、4、6、7 步：采集两次运行，找到首个分歧点，修复原因，再确认修复有效。你手上有一次参考运行和一次 suspect 运行，它们只在声明过的那一点上不同（[预检](determinism-and-preflight.md)），读到的 batch 也完全相同（[数据对齐](data-alignment.md)）。现在还不知道的是：两次运行从哪里开始分开，以及为什么。判断原因属于哪一类（第 5 步）放在[流程总览页](README.md#step-5-classify)里讲；这一页后面的几页，分别讲首个分歧点最常指向的几类原因：[训练 step](training-step.md)、[梯度的集合通信](data-parallel.md)、[torch.compile 与 CUDA graph](capture-compiled-code.md) 和[硬件](sdc.md)。

## 为什么这条流程用 OpGuard {#why-opguard}

比较两次运行，有看起来更省事的办法：画出两条 loss 曲线，或者 dump 一些张量，在容差范围内比一比。loss 曲线把上百万次运算压成一个数，只能告诉你两次运行不一样，说不出是在哪里不一样。容差的问题正好相反：它宽到足以盖住真实 bug 留下的那几个错误元素。OpGuard 就是为了避开这两种情况而设计的，这也是本部分围绕它来组织的原因。

- **它比较的是精确的 bit，所以一旦不一致，就说明真有问题。** 它先去掉能够避免的不确定性，再要求每个记录下来的张量逐 bit 一致。作者在真实的生产 bug 上做过评估：把逐 bit 比较换成容差以后，能精确定位的 bug 少了很多。
- **它足够轻，可以直接跑在真实的训练任务上。** 把每个张量都 dump 下来，会让训练任务慢上几个数量级，所以 OpGuard 只给每个张量记一个很小的 fingerprint。训练任务的各项优化照常开着，也不用有人手工改模型代码来插桩。
- **代码不同的两次运行也能比。** 它在模型级的算子上记录（norm、attention、投影），只要忠实地实现了这个模型，任何代码都会算这些算子。所以参考运行可以是旧的 commit、另一个框架、另一套 kernel 或另一种硬件，suspect 运行本身也不需要是确定性的。
- **它交给你的是一条事实，而不是一份日志。** 它先确认前面一段完全相同，再报告这之后的第一个差异。你或你的 agent 只需要读一个事件，而不是几千行日志；它的 agent harness 通过工具提供这些事实，判断则留给 skill。
- **它是为多节点的大规模训练任务做的，也确实在上面用过。** 它采集每一个 rank，给记录排序时不往训练任务里加时钟同步或 barrier，而且先写到节点本地的磁盘，再在后台同步到共享存储（见下一节）。据作者报告，它已经用在大规模 GPU 集群上的生产 LLM 训练中，找到了二十多个问题的根因，其中有 kernel 里的竞态，也有静默数据损坏（SDC），把排查时间从几天缩短到了几分钟（[OSDI '26 论文](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)）。

### 大规模训练任务：先写本地磁盘，再定期同步 {#periodic-sync}

一个大规模训练任务有几千个 rank。如果每个 rank 都把自己的 trace 文件直接写到共享存储上，共享存储会在同一时刻收到大量的小文件写入，训练任务也只能等着它。OpGuard 的本地暂存（local staging）就是为了避免这种情况：每个 rank 以本地磁盘的速度写到自己所在的节点上，每个节点再由一个上传进程按固定间隔，把新增的数据打成一个包（bundle），拷到共享存储上。

<figure class="fx" data-fig="periodic-sync" aria-label="trace 先写到节点本地磁盘，再定期同步到共享存储">
<div class="fx-stage">
<div class="fx-rows">
<div class="fx-row"><span class="fx-label">不做暂存</span><div class="fx-chain is-tight"><span class="fx-node is-bad" data-hot="0">每个 rank<span class="fx-note">大量小文件</span></span><span class="fx-arrow" data-hot="0">同时写</span><span class="fx-node is-bad" data-hot="0">共享存储<span class="fx-note">训练任务要等它</span></span></div></div>
<div class="fx-row" data-at="1"><span class="fx-label">训练过程中</span><div class="fx-chain is-tight"><div class="fx-stack"><span class="fx-node to-key" data-hot="1 2">节点 0 的本地磁盘<span class="fx-note">本节点的 rank 写在这里</span></span><span class="fx-node to-key" data-hot="1 3">节点 1 的本地磁盘<span class="fx-note">本节点的 rank 写在这里</span></span></div><span class="fx-arrow" data-at="2" data-hot="3"><span class="fx-swap"><span data-to="3">节点 0 的包</span><span data-at="3">节点 1 的包，稍晚一点</span></span></span><span class="fx-node to-ok" data-hot="2 3">共享存储<span class="fx-note">每次一个包，限速</span></span></div></div>
<div class="fx-row" data-at="4"><span class="fx-label">训练结束后</span><div class="fx-chain is-tight"><span class="fx-node" data-hot="4">最后一次同步<span class="fx-note">传完剩下的数据</span></span><span class="fx-arrow" data-at="5"></span><span class="fx-node to-ok" data-hot="5">比较<span class="fx-note">先把包恢复到本地</span></span></div></div>
</div>
</div>
<ol class="fx-say">
<li data-hold="1.5">不做暂存时，每个 rank 都把自己的小 trace 文件直接写到共享存储，而且是同时写。</li>
<li>做了暂存，每个 rank 以本地速度写到自己节点的磁盘上。</li>
<li data-hold="1.5">每到一个间隔，节点 0 的上传进程把新增的数据打成一个包，在限速下发布到共享存储。</li>
<li data-hold="1.5">节点 1 稍晚一点做同样的事：间隔带一点随机，各个节点不会同时上传。</li>
<li>训练任务退出时，最后一次同步把剩下的数据传上去。</li>
<li data-hold="2">比较两次运行之前，OpGuard 先把这些包恢复到本地目录，并做校验。</li>
</ol>
<figcaption>图：rank 写到所在节点的本地磁盘；每个节点每个间隔上传一个包，各节点错开时间、各自限速；训练任务退出时，最后一次同步把剩下的传完。</figcaption>
</figure>

有两点让它可以在整个训练过程中一直开着。一个包在共享存储上要么完整出现，要么根本不出现，上传失败的包会留下来重试，所以谁也不会读到半个包。每个节点的上传也各有带宽上限，同步占用的网络不会超过你允许的量。

打开它只需要在训练任务的启动脚本里设几项：每个节点本地磁盘上的目录、要同步到的共享目录、同步间隔，以及每个节点的带宽上限。真正需要权衡的只有间隔：间隔越短，节点挂掉时丢的数据越少，但上传也越频繁。具体的设置项和默认值见 OpGuard 的[采集说明](https://github.com/ziming-zh/OpGuard/blob/main/docs/Capture.md)。

本 handbook 尚未证明：它自己的例子最多只跑了两个节点，trace 是直接写的，所以这里没有实际用过本地暂存。

## 思路：第一个差异只告诉你从哪里查起，不等于原因

OpGuard 在两次运行的每个模型级边界上记录 fingerprint，把两份 trace 配对，然后报告第一对不相等的事件。这一对事件叫 **pivot**。

<figure class="fx" data-fig="pairing" aria-label="两份 trace 配对并找到 pivot">
<div class="fx-stage">
<div class="fx-grid" style="--cols:5">
<span class="fx-label">参考运行</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-empty" data-hot="3"></span><span class="fx-node" data-hot="4">attn</span><span class="fx-node">mlp</span>
<span></span><span class="fx-pair is-ok" data-at="1">相等</span><span class="fx-pair is-ok" data-at="2">相等</span><span class="fx-pair" data-at="3">空缺，不配对</span><span class="fx-pair is-bad" data-at="4">第一个差异，即 pivot</span><span></span>
<span class="fx-label">suspect 运行</span><span class="fx-node" data-hot="1">embed</span><span class="fx-node" data-hot="2">norm</span><span class="fx-node is-ghost" data-hot="3">多出的 cast</span><span class="fx-node to-bad" data-hot="4">attn</span><span class="fx-node">mlp</span>
</div>
</div>
<ol class="fx-say">
<li>两份 trace 按调用顺序排开，每个边界都有一个 fingerprint。</li>
<li>embed 和 embed 配成一对，结果相等。</li>
<li>norm 和 norm 配成一对，结果相等。</li>
<li data-hold="1.5">suspect 运行多了一个 cast。保序匹配不给它配对，把它留作空缺；空缺不算差异。</li>
<li data-hold="2">attn 不相等。第一对不相等的事件就是 pivot，从它出发去找原因。</li>
</ol>
<figcaption>图：两份 trace 按调用顺序排开。匹配把相等的边界配成对，多出的 cast 不配对，然后报告第一对不相等的事件。</figcaption>
</figure>

这张图里有两点很重要。

- **空缺不是差异。** suspect 运行有一个参考运行里没有的事件。保序匹配不会强行给它配对，而是让它保持未配对；只有 fingerprint 不相等的配对才会被报告。
- **pivot 不一定就是原因。** 举个小例子：注意力的反向用原子操作累加 dQ，而它自己没有边界。第一个看到它结果的边界是梯度范数，那已经在整个反向计算之后了。所以 pivot 是这个范数，原因却在上游。真正指向原因的，是调用点上的状态：它告诉你每个参数的梯度是相等还是不同。在最后一个 block 里，只有 q 投影的梯度不同，而这个梯度正是 dQ。

**规则。** 取第一个逐 bit 的差异，并且它之前不能有未对齐的事件。从这里出发，沿着产生它输入的算子往上游追，也往下游追到症状，再用尽量小的干预去检验你怀疑的原因。这是第 0 部分"在算子边界上比较"的具体应用：边界能定位第一个可见的影响，而对上游算子的推理才能定位原因。

## 步骤

### 1. 确认要查的区域已经被采集

**目标：** 你关心的区域已经被采集。

**做法：** `opguard_bridge.coverage(run)` 读取 OpGuard 的发现文件和计划文件，列出哪些边界被发现了、哪些被计划采集。

**成功标准：** 你怀疑的算子，包括其中的 GEMM，都在采集计划里。

### 2. 采集两次运行

**目标：** 两次运行在每个计划好的边界上都有 fingerprint。

**做法：** 调用 [opguard_bridge.py](../../tools/opguard_bridge.py) 里的 `opguard_bridge.run_pair(reference, suspect, ...)`，让两次运行都经过 OpGuard 的 harness：发现边界，规划插桩，采集，然后比较。发现和采集两个阶段，各自会把两次运行都启动一遍。在多节点的训练任务上，采集之前先打开[本地暂存](#periodic-sync)。如果生产环境的训练任务用了 `torch.compile` 或 CUDA graph，这次采集会改用 eager 模式运行它，于是可能看不到生产环境实际算出的结果，见[用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md)。

**成功标准：** 两份 trace 都已生成，摘要里统计了相等和不相等的事件各有多少。

### 3. 收集首个分歧点的全部事实

**目标：** 把关于 pivot 能知道的事实全部拿到。

**做法：** `opguard_bridge.first_divergence(export, ...)` 依次调用 harness 的 `trace_summary`、`first_diff`、`inspect_event`、`compare_events` 和 `trace_neighbors`，返回：算子、源码行、调用序号、哪些字段不同、两次运行的输入和输出、不同的区间、pivot 之前未对齐的事件，以及 pivot 所在调用点上每次调用的状态。它本身不做任何判断。

**成功标准：** 拿到一个 `unaligned_before` 为零的 pivot。

### 4. 从 pivot 追到原因，并检验它

**目标：** 找到原因，并拿到支持它的证据。

**做法：** 用 OpGuard harness 里的几个 skill：`investigate-first-diff` 负责检查这个边界并记下证据；输入已经不同时，`trace-upstream` 沿着上游算子往回追；`validate-hypothesis` 每次检验一个原因。`opguard_bridge.suggest(facts)` 会把 pivot 对应到本手册的算子类别和[recipe](../../skills/batch-invariant-kernel/references/operator-recipes.md)，每条线索都附有理由。

| 证据 | 能告诉你什么 | 代价等级 | 什么时候用 |
|---|---|---|---|
| pivot 的源码行和 dtype | 直接看出舍入点是否被挪动，或者多了一次类型转换 | 免费 | pivot 处输出的 dtype 或形状不同 |
| 调用点状态（每次调用一个字母） | 同一个调用点上哪些调用不同，进而知道是哪个参数、哪一步 | 免费 | pivot 要用到许多上游算子的输出（比如对所有梯度求范数） |
| 输入相等，输出却不同 | 这个算子自身的状态或权重更早就变了 | 免费 | 查表类算子，以及其他看起来没有状态的算子 |
| `suggest` 给出的线索 | 算子类别和对应的 recipe，附带理由 | 免费 | 开始读 kernel 代码之前 |
| 重新采集某个区域的完整张量 | 数值本身，可以看出它们差在哪里 | 对这个区域来说大（2 倍或更多） | fingerprint 相等但怀疑是碰撞，或者需要知道差异有多大 |
| 在固定输入上反复运行同一个算子 | 这个算子是不是不确定的 | 小 | 你怀疑原因是某个不确定的 kernel |

**成功标准：** 得到一个假设，而且一次干预就能确认它。

### 5. 修复原因，并确认修复有效

**目标：** 证明是这个修复、而不是别的东西，让两次运行变得一致。

**做法：** 应用修复（改一个设置、改训练 step 的某个部分，或者按 [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md) 改写某个 kernel），然后再调用一次 `run_pair`。

**成功标准：** 每个事件都相等，没有未对齐的事件，最终参数逐 bit 一致。

在一次已知答案的演练里，我们预先植入了三个原因：norm 的舍入点被挪动、注意力的反向跑在 `warn_only` 下、恢复训练时没有恢复优化器的 step 计数器。在一块 GPU、一个 rank 上，OpGuard 找到的每个首个差异之前都没有未对齐的事件；三个 pivot 里有两个并不是原因本身；每次修复后重跑，结果处处相等，最终参数逐 bit 一致（[报告](../../examples/pretraining_e2e/REPORT.md#what-this-shows)）。这是一次已知答案的演示，不是盲测评估。

## 可能出错的地方

!!! warning "`torch.nn.Linear` 把 GEMM 藏了起来"
    GEMM 是在 torch 自己的源文件里启动的，所以调用投影的那一行虽然被发现了，却不会被计划采集。结果没有任何投影被采集，pivot 落到了更后面。在模型代码里改用 `F.linear`，并检查 `coverage`。

!!! warning "模型代码里用了单字母变量名"
    OpGuard 的插桩器根据调用点周围的名字来匹配调用点，并且会忽略单字母的名字。于是注意力和投影虽然列进了采集计划，却从来没有被包装上。改用有意义的名字，并检查 `coverage`。

!!! warning "边界落在了行末的类型转换上"
    像 `q = q_norm(x).float()` 这样的一行，类型转换并不是真正的计算，但 pivot 会指向这次转换。应该把算子本身（`q_norm`）设为边界。

!!! warning "把没配上对的事件当成了差异"
    其中一次运行可能因为融合或改写而多出一些事件。强行给它们配对，会凭空造出并不存在的差异。让没配上对的事件保持为空缺。

!!! warning "把 pivot 当成了原因"
    没有边界的上游计算（反向计算、优化器的 step、恢复训练）会先在下一个用到它输出的算子处显现出来。如果直接修 pivot，就修错了算子。动手修复之前，先往上游追，并检验假设。

## 小结

- 先检查覆盖范围：没被采集的算子不可能成为 pivot。
- pivot 是第一对不相等的事件，而且它之前没有未对齐的事件；空缺不是差异。
- pivot 只是起点。用源码行、调用点状态和输入，从它一路追到原因。
- 修复要靠重新采集来确认：所有事件都相等，没有未对齐的事件，参数逐 bit 一致。

## 深入阅读

- [报告：OpGuard 演练](../../examples/pretraining_e2e/REPORT.md)：实验配置、OpGuard 对每一对运行报告了什么、修复和重跑、每次排查的代价、覆盖范围，以及对 fingerprint 的检查。
- 代码：[pretraining_e2e.py](../../examples/pretraining_e2e/pretraining_e2e.py) 和 [opguard_bridge.py](../../tools/opguard_bridge.py)。
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)。
- [用了 torch.compile 或 CUDA graph 时怎样采集](capture-compiled-code.md)：训练任务用了 `torch.compile` 或 CUDA graph 时该怎么办。
