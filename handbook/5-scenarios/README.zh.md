# 5. 其他对齐场景

## 我们在哪里

第 1 到第 4 部分用两条工作流程讲了同一套方法：找到两次预训练运行在哪里分开
（[第 1 部分](../1-pretraining/README.md)），让 rollout 引擎和 trainer 一致
（[第 2 部分](../2-rl/README.md)），让单个算子满足不变性
（[第 3 部分](../3-kernels/README.md)），以及在技术栈变化时保持结果成立
（[第 4 部分](../4-operations/README.md)）。还有很多需求其实是同一个问题，只是换了一对程序：
框架迁移、检查点转换、在推理引擎里跑参考模型。这一部分讲怎样把已有的方法用到七个这样的场景上。
其中三个有单独的页面，另外四个在本页各有一小节。

## 思路：写下这一对、输入和要求

每个场景都由四样东西组成：参考方、被检方、两边收到的输入，以及一个要求——哪些输出应该相等、
相等到什么程度。把这四样写下来，就能套用已有的工作流程。几个例子：

| 这一对 | 相同的输入 | 应该相等的是什么 | 预期 |
|---|---|---|---|
| 一个检查点和它转换后的副本 | 无：什么都不计算 | 每个权重的每一行 | 永远逐位相等 |
| 同一个模型在两个训练框架里 | 数据、权重、种子 | 两边共有的每个边界 | 只有对齐所有舍入点之后才逐位相等 |
| trainer 和预先算好的参考 log-prob | 相同的 token id 和权重 | 每个 token 的 log-prob | 做到引擎一致性后逐位相等 |
| 单独发送的请求和繁忙服务器里的同一个请求 | 这个请求 | 它的 token 和 log-prob | 服务端批不变后逐位相等 |
| 同一个作业在两代 GPU 上 | 全部相同 | 位级别上没有要求 | 位会不同；改为定位 |

最后一行和其他行一样重要。在没有承诺逐位相等的地方，目标是找到第一个位不同的算子并对它做出判断，
而不是不惜代价追求相等。规则是：**比较任何东西之前，先写下这一对、输入和要求。**
这就是[第 1 部分](../1-pretraining/README.md#the-idea-two-runs-one-difference)的对照实验，
只是把要求明确写了出来。

## 各个场景

| 场景 | 复用的工作流程 | 工具与 skill | 特有的问题 |
|---|---|---|---|
| [框架迁移与库升级](framework-migration.md) | [第 1 部分](../1-pretraining/README.md)，以另一个技术栈作参考；[引擎一致性](../2-rl/engine-parity-vllm.md)的舍入点方法 | [determinism_preflight.py](../../tools/determinism_preflight.py)、[opguard_bridge.py](../../tools/opguard_bridge.py)、[engine-parity](../../skills/engine-parity/SKILL.md) | 两个程序的 trace 对不齐；每处差异要么是 bug，要么是选择 |
| [检查点转换与重新分片](checkpoint-conversion.md) | 没有计算：先做副本检查，再做一次前向 | [ckpt_compare.py](../../tools/ckpt_compare.py) | 行在重新打包后保持不变；位置要靠前向检查 |
| [离线后训练](offline-post-training.md)：SFT 和 DPO 的参考 log-prob、蒸馏的 teacher logits | [第 2 部分](../2-rl/README.md)的 L4，没有采样 | [engine-parity](../../skills/engine-parity/SKILL.md)、OpGuard 的 RL 记录 | 对哪些 token 打分、对话模板、padding |
| [奖励模型与评测](#reward-models-and-evaluation) | [整个模型](../2-rl/whole-model.md)、[采样](../2-rl/sampling.md)、[回归](../4-operations/regression.md) | [bi_audit.py](../../tools/bi_audit.py)、[bi_sampling.py](../../tools/bi_sampling.py) | 阈值附近的分数会把一位之差变成不同的判定 |
| [确定性服务](#deterministic-serving) | 引擎一侧的[整个模型](../2-rl/whole-model.md)；[第 3 部分](../3-kernels/README.md) | [bi_audit.py](../../tools/bi_audit.py)、[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md) | 请求可能遇到的每一种服务模式 |
| [投机解码与 MTP](#speculative-decoding-and-mtp) | [采样](../2-rl/sampling.md)、L3 下的[整个模型](../2-rl/whole-model.md) | [bi_sampling.py](../../tools/bi_sampling.py) | 验证是一个 chunk，解码是一个 token |
| [硬件迁移与混合机群](#hardware-migration-and-mixed-fleets) | [硬件](../4-operations/hardware.md)、[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md) | [sdc_scan.py](../../tools/sdc_scan.py)、[bi_suite.py](../../tools/bi_suite.py) | 不同代 GPU 之间不期望逐位相等 |

## 奖励模型与评测

**目标：**奖励模型对一个（prompt，response）对在任何批次里给出相同的分数；评测重跑一次得到相同的数。

奖励模型就是带一个标量输出头的完整模型，所以它和策略模型一样依赖批次和模式。不同的是用法：
分数要和阈值比较，要和其他回答排序，或者被平均成通过率。离阈值只差一位的分数会改变判定，
判定变了，训练信号或报告的数字也就变了。

怎样套用已有的页面：

1. 像对待任何模型一样让奖励模型满足不变性：用 [bi_audit.py](../../tools/bi_audit.py) 审查算子，
   比较一个样本单独计算和放在批次里、padding 和 packing 的结果（[整个模型](../2-rl/whole-model.md)）。
2. 如果分数在引擎里算、又由 trainer 重算，那就是 L4：按[引擎一致性](../2-rl/engine-parity-vllm.md)来做。
3. 如果评测要采样，就让 token 能从 logits 和种子重放
   （[采样](../2-rl/sampling.md)、[bi_sampling.py](../../tools/bi_sampling.py)）。
4. 把一个固定的评测集和它的分数作为[回归](../4-operations/regression.md)条目保存。

**成功标准：**每个样本的分数在各种批大小下逐位相同；换一种批次组成重跑，评测结果也相同。

尚未证明：这里没有测量过任何奖励模型或评测框架。

## 确定性服务

**目标：**一个请求的 token 和 log-prob 不依赖于和它同批的其他请求（L2），也不依赖于服务器怎样
分块、缓存或解码它（L3）。

这就是[整个模型](../2-rl/whole-model.md)这一步的引擎一侧，只是从用户的角度看：
两个相同的请求在 temperature 0 下得到不同的回答，因为服务器当时的负载不同。原因和修法都一样。

怎样套用已有的页面：

1. 审查服务的模型，并测试请求可能遇到的每种模式：变化的 decode 批次、分块 prefill、
   命中前缀缓存、被抢占（[整个模型](../2-rl/whole-model.md)）。
2. 把每个随批次变化的算子换成不变的 kernel（[第 3 部分](../3-kernels/README.md)）；
   让采样器保持可重放（[采样](../2-rl/sampling.md)）。
3. 在你的 decode 批大小下测量代价。代价太高时，只给需要的请求走不变路径，而不是全部请求。

**成功标准：**每个请求单独服务和在负载下服务时，log-prob 逐位相等，服务器用到的每种模式都如此。

尚未证明：真实流量下的生产服务器，多节点服务。

## 投机解码与 MTP

**目标：**投机解码或多 token 预测（MTP）解码器输出的 token 和 log-prob，与目标模型单独解码时完全相同。

验证阶段让目标模型一次处理几个草稿 token，像一次短的 prefill；普通解码一次处理一个 token。
如果这两条路径的舍入方式不同，被接受的 token 的 log-prob 就和普通解码的不同，接受判断本身也可能翻转。
只要接受时比较的是目标模型自己的采样结果，草稿模型自身的数值只影响速度，不影响结果。

怎样套用已有的页面：

1. 让目标模型的验证前向等于它的解码前向：chunk 对单 token 正是[整个模型](../2-rl/whole-model.md)里的 L3 情形，
   注意力的 prefill 和 decode 要对 key 使用同一种划分（[注意力](../3-kernels/attention.md)）。
2. 让接受过程可重放：使用逐位相等验证时，投机解码给出的 token 和不用草稿的解码相同
   （[采样](../2-rl/sampling.md)、[bi_sampling.py](../../tools/bi_sampling.py)）。
3. 用作草稿的 MTP 头按同样方式处理；预训练里的 MTP 损失属于训练步骤（[第 1 部分](../1-pretraining/README.md)）。

**成功标准：**有无投机解码，在每种草稿长度下，token 和 log-prob 的位都相同。

尚未证明：真实引擎的投机解码，MTP 头。

## 硬件迁移与混合机群

**目标：**知道哪些结果能带到新 GPU 上；两台机器不一致时，找到在哪里。

不同代 GPU 之间不期望逐位相等：库会按架构选择不同的 kernel 和累加顺序
（[硬件](../4-operations/hardware.md)）。所以目标从相等变成定位。第一个位不同的算子应该是库的 GEMM，
或由库决定顺序的归约；其他任何情况都是一项发现。

怎样套用已有的页面：

1. 鉴定新硬件：用已提交的基线跑回归套件，并做跨组的 SDC 扫描
   （[硬件](../4-operations/hardware.md)、[bi_suite.py](../../tools/bi_suite.py)、[sdc_scan.py](../../tools/sdc_scan.py)）。
2. 同一代、同一技术栈内，逐位比较设备：这里的不一致就是故障（[SDC](../1-pretraining/sdc.md)）。
3. 跨代时，用 OpGuard 在相同输入上采集两次运行，取首个分歧点
   （[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md)）。给它分类：
   随架构而变的 kernel 选择（预期之内），或其他（报告出来）。
4. 在强化学习里，如果 trainer 和引擎在不同代 GPU 上，只有归约顺序写在源码里的 kernel 才可能带过去，
   而这在不同架构之间尚未测试。否则就测量数值差距，把它记为一项决定，
   就像[引擎一致性](../2-rl/engine-parity-vllm.md)对待两边都无法改动的差异那样。

**成功标准：**不同代之间的每一处差异都已分类，预期之内的差异连同所属技术栈一起记录下来。

尚未证明：手册的 kernel 在不同 GPU 架构上的表现，跨代混合的强化学习运行。

## 可能出错的地方

!!! warning "在没有承诺的地方期待逐位相等"
    两代 GPU、两个库的 kernel，本来就不是为最后一位一致而设计的。在那里追求相等，
    时间都花在预期之内的差异上。先写下要求；要求不是相等时，就定位并做决定。

!!! warning "用容差代替逐位检查"
    "足够接近"的输出会掩盖恰好很小的故障：一次只舍入小权重的类型转换，
    在少量输入上可能一个 logit 都不改变（[检查点转换](checkpoint-conversion.md)）。
    副本和重算都要逐位检查；容差只用于不可能一致的配对。

!!! warning "比较的是两个不同的计算"
    参考 log-prob 打分的是别的 token，只有一侧套了模板，分数经过了不同的后处理：
    数字不同，但数值上并没有错。比较数值之前，先确认两边算的是同一件事。

## 小结

- 每个场景都是一对程序、两边收到的输入和一个要求；先把它们写下来。
- 副本必须逐位相等；两个程序可以通过对齐舍入点做到逐位相等；两代 GPU 只做定位，不追求相等。
- 已有的工作流程原样适用：预检、采集、首个分歧点、舍入点、决定、回归。
- 每一节都写明了尚未证明的部分，把它们当作待办。

## 深入阅读

- 页面：[框架迁移](framework-migration.md)、
  [检查点转换](checkpoint-conversion.md)、
  [离线后训练](offline-post-training.md)。
- [报告：检查点转换](../../examples/ckpt_conversion/REPORT.md)，
  包括植入的故障和每项检查看到了什么。
- Skill：[alignment-pipeline](../../skills/alignment-pipeline/SKILL.md)、
  [engine-parity](../../skills/engine-parity/SKILL.md)、
  [pretraining-alignment](../../skills/pretraining-alignment/SKILL.md)。
- 所有尚未测试的内容见[待办事项](../open-items.md)。
