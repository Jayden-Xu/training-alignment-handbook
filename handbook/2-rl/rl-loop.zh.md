# RL 循环：记录、staleness 拆分、离线定位

## 我们在哪里

完成[引擎一致性](engine-parity-vllm.md)之后，trainer 和引擎在一组固定的 prompt 上一致。
运行中的 RL 循环带来了那次测试没有的东西：每个版本都要推送权重、异步 rollout
用的是旧版本、批次每一步都在变。这一步负责看守循环：在线做低成本记录，
把每个差距拆成 staleness 和数值误差，只对真正的基础设施故障做离线定位。
这是 [RL 工作流程](README.md)的第 7–9 步。

## 思路

[工作流程页面](README.md)把差距精确拆开：

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

拆分是否可信取决于三件事，它们也决定了步骤的顺序：

1. **权重必须是声称的那份。** 如果权重推送后引擎持有错误的行，那么这个版本上的每次比较
   都没有意义。在看任何 logprob 之前先检查权重。
2. **一个 token 必须只属于一个版本。** 新权重到达时正在生成、且保留了 KV cache 的序列，
   含有来自两个策略的 token。它在哪个版本下都无法重放，所以先放到一边。
3. **分歧需要两侧的证据。** 沿模型追查非零的数值误差时，在另一侧找不到对应的行只是一个
   空缺（另一侧融合掉的一步），不是分歧。分歧是两侧相同的输入行给出不同的输出行。

定位沿数据流进行，而不是沿 trace 顺序。从两侧一致的位置开始，例如 embedding 的输出，
跟着每个张量走到它的使用者，直到第一对输入相等、输出不同的位置。

<figure class="fig">
<div class="fig-spine"><span class="fig-node">记录（在线）</span><span class="fig-arrow" aria-hidden="true"></span><div class="fig-chain"><span class="fig-node is-decision">权重一致？</span><span class="fig-arrow">否</span><span class="fig-node is-bad">修复同步；停止</span></div><span class="fig-arrow">是</span><div class="fig-chain"><span class="fig-node">把每个 token 拆成 numeric 和 staleness</span><span class="fig-arrow">numeric = 0</span><span class="fig-node is-ok">基础设施无责</span></div><span class="fig-arrow">numeric ≠ 0</span><div class="fig-chain"><span class="fig-node">两侧重新捕获，沿数据流走</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-key">对原因分类，测试最小的修复</span></div></div>
</figure>

## 步骤

OpGuard 的 RL harness 实现了下面的每个动作。记录用
`opguard_harness.rl_records.RecordWriter` 写入；各个动作（对记录文件的 `rl.*`、
对捕获的 `vectors.*`）只报告事实，返回 `verdict: not_evaluated`，不使用任何容差。
协议和每个字段见 OpGuard 的 `docs/RLAlignment.md`；OpGuard 的 skill `rl-alignment`
（`harness/skills/rl-alignment`）把这些步骤作为 agent 工作流程来执行。

1. **在线记录。**
   - 目标：足以在之后拆分和重放任何采样批次，而且成本低。
   - 做法：`RecordWriter` 记录 `rollout_batch`（token 和 prompt 的 id、位置、每个 token 的
     策略版本、rollout logprob 的位、批次组成、引擎所持权重的指纹）、`forward`
     （每次前向及其行布局）、`weight_sync`（每次同步时两侧逐行的权重指纹）、`snapshot`
     （各版本权重存放的位置），以及 staleness 为零时的 `trainer_logprob`。
   - 成功标准：记录摘要列出了每种类型，快照版本按抽样保存，成本有上限。
2. **检查权重。**
   - 目标：引擎运行的就是 trainer 推送的权重。
   - 做法：`rl.weight_sync_check` 把逐行指纹作为多重集合比较，因此不同的打包方式
     （例如融合的 QKV）不算差异。
   - 成功标准：每个比较的版本都干净；否则在两侧分别指出出错的参数和行。
3. **拆分差距。**
   - 目标：每个 token 的数值误差和 staleness。
   - 做法：`rl.decompose`，用精确算术；它会标出混合版本的 token，并说明每个批次与哪个
     记录的权重指纹相符。对较旧的版本，`rl.replay_request` 构造重放请求（版本 v 的快照、
     相同的 token、原始的或批不变的组成），由你的工作负载离线运行。
   - 成功标准：每个非混合 token 的数值误差为零，或者得到不为零的 token 清单。
4. **离线定位。**
   - 目标：数值误差最先出现的算子。
   - 做法：用逐向量指纹重新捕获两侧（OpGuard `OPGUARD_XOR_ROWS=1`），用 `rl.forwards`
     列出前向，用 `vectors.describe`、`vectors.lookup`、`vectors.coverage` 和
     `vectors.compare` 查询，从一致的锚点开始沿数据流走。运行编译代码或 CUDA graphs
     的引擎用 [tools/graph_probe.py](../../tools/graph_probe.py) 取指纹
     （[捕获编译后的代码](../1-pretraining/capture-compiled-code.md)）。
   - 成功标准：一对候选：两侧输入行相同，输出行不同。
5. **归类并检验原因。**
   - 目标：一个原因类别，并经过干预确认。
   - 做法：舍入点或融合顺序；批次依赖；算法选择；并行布局；不同的 logprob 公式；
     版本记账或权重同步。做出能让该算子处数值误差变为零的最小改动，然后重放。
   - 成功标准：带这个改动的重放等于另一侧。
6. **保留下来。**
   - 目标：修好的东西保持修好。
   - 做法：把这项检查连同基线加入[回归](../4-operations/regression.md)测试集。
   - 成功标准：之后的版本必须复现这个结果。

| 选项 | 得到什么 | 代价类别 | 何时选择 |
|---|---|---|---|
| 开启一致性、同步、短窗口 | 每个 token 的差距恰好为 0，或者一个待定位的失败 | 一致性配置的代价，只在窗口内 | 长时间运行之前，或者要为基础设施排除嫌疑时 |
| 每步记录和指纹，快照抽样保存 | 离线拆分任何记录过的批次 | 不含快照时小（几个百分点）；快照随模型规模增长 | 每一次你可能需要审计的生产运行 |
| 离线重放和重新捕获 | 定位一个 token 的数值误差 | 每次定位几秒，不在训练路径上 | 只用于数值误差非零的 token |

已证明：在一个小循环中，两侧都使用批不变算子集时，同步和异步下每个非混合 token 的数值误差
都逐位为零，所以 staleness 承担了全部差距；权重检查穿过融合的重新打包，指出了人为植入的错误行；
人为植入的代码故障在一次已知答案的演练中被定位到对应算子，演练的数据流顺序是根据模型代码写出的，
而不是自动发现的（[报告](../../examples/rl_loop/REPORT.md)）。
尚未证明：大模型、任一侧的 TP/EP、CUDA graphs、循环中的生产引擎。

## 可能出错的地方

!!! warning "引擎的标志只是声明"
    rollout 记录的标志可能写着批不变，实际跑的却是原生算子集。相信它们会把搜索引向错误的一侧。
    把标志当作声明，让数值拆分来判断。

!!! warning "版本标签故障看起来像数值误差"
    标为某个版本的批次，可能用的是另一个版本的权重。它在线的数值误差不为零，
    但在记录的版本上重放却等于 trainer，也找不到不同的行。把这个批次的权重指纹
    和每个同步版本比较，然后在相符的版本上重放。

!!! warning "按顺序配对在形状不同时会沉默"
    一次 trainer prefill 和许多次引擎 decode 前向调用函数的次数不同，
    所以按顺序配对事件的 diff 可能什么都不报告。那里的沉默不是证据。
    在声明的 token 维度下比较逐向量指纹。

!!! warning "零指纹不代表一致"
    有些行的 XOR 本来就是零（重复的两半，例如分组查询注意力在位置 0 处）。
    把它们算作相等会让追查过早停止。只靠零指纹的一致视为没有证据。

!!! warning "融合的生产者没有对应项"
    融合投影的输出行比 trainer 的任何单行都宽。要求宽度相等的追查找不到配对。
    在拆分之后的第一个使用者处比较它。

!!! warning "重放只在声明的组成下复现"
    重放的 rollout 只有在使用记录的批次组成、或使用批不变 kernel 时，才会得到原来的数。
    两者都不满足时，重放会因为与故障无关的原因而不同。说明你用的是哪一种。

## 小结

- 在线做低成本记录；快照只按你负担得起的频率保存。
- 先检查权重，再拆分每个差距；混合版本的 token 放到一边。
- 数值误差为零，就为这些批次排除了基础设施的嫌疑；staleness 交给重要性采样校正。
- 非零的数值误差沿数据流离线定位，要有两侧的证据，并通过干预确认原因。

## 深入阅读

- [实验报告](../../examples/rl_loop/REPORT.md)：循环、每个场景、植入的故障、
  harness 的发现及其修复、代价。
- 代码：[rl_core.py](../../examples/rl_loop/rl_core.py)、
  [rl_loop.py](../../examples/rl_loop/rl_loop.py)、
  [tools/opguard_bridge.py](../../tools/opguard_bridge.py)、
  [tools/graph_probe.py](../../tools/graph_probe.py)。
- 对真实引擎做的单元素定位：
  [报告](../../examples/cross_engine/REPORT.md#worked-example-one-element-at-q_norm)。
- OpGuard：`docs/RLAlignment.md` 和 `rl-alignment` skill。
- 下一步：[回归](../4-operations/regression.md)。
