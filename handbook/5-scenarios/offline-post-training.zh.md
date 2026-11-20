# 离线后训练：参考 log-prob 与 teacher logits

## 我们在哪里

好几种后训练方法会提前算好损失里的一项，而且常常放在推理引擎里算，因为更快。
DPO 及其变体要用参考模型对被选回答和被拒回答的 log-prob；带 KL 项的 SFT 要用参考 log-prob；
蒸馏要用 teacher 的 logits 或它的 top-k log-prob。trainer 随后把存下来的数当成自己算的来用。
如果引擎和 trainer 对相同的 token 算出不同的数，损失里就带着一个没人选择的误差。
这就是[第 2 部分](../2-rl/README.md)的 L4 问题，只是去掉了采样器：token 由数据集固定，
参考权重也不变，所以没有 staleness。剩下的只有记账问题和数值误差。

## 思路：第 0 步的免费检查

DPO 在每一对回答上比较策略和参考：一个偏好的回答 y_w，一个被拒的回答 y_l：

```
margin = (logp_policy(y_w) - logp_ref(y_w)) - (logp_policy(y_l) - logp_ref(y_l))
loss   = -log sigmoid(beta * margin)
```

第 0 步时策略就是参考，所以每个 margin 都恰好是 0，损失恰好是 log 2。
如果存下来的参考 log-prob 来自引擎，第 0 步就能告诉你它们对得有多准：

| logp_ref 的来源 | 第 0 步的 margin |
|---|---|
| trainer 自己的前向 | 每一对都恰好为 0 |
| 在别处舍入的引擎 | 很小但不为零，每一对都不同 |
| 对别的 token 打分的引擎（另一个模板、别的位置） | 并不小：打分的是另一个序列 |

所以在第 0 步，让 trainer 对一部分样本对重算参考 log-prob，与存下来的值逐位比较。
相等：预计算是精确的。不相等：先问打分的是不是同样的 token，再问用的是不是同样的权重，最后才调试数值。

规则是：**预先算好的数，trainer 必须能从相同的 token、相同的权重逐位复现；依次检查 token、权重、数值。**
这就是第 2 部分拆分差距的做法：先把记账问题和数值误差分开，再去调试其中任何一个。

## 步骤

1. **确认打分的是同样的 token。**
   - 目标：两边在相同位置上对相同的 token id 打分。
   - 做法：每个预计算值都和它的 token id、位置以及每个 log-prob 的精确位一起存下来，
     使用 OpGuard 的 `rollout_batch` 记录的字段（`opguard_harness.rl_records.RecordWriter`）；
     trainer 把自己算的值写成 `trainer_logprob` 记录。先比较 id，再比较任何值。
     重点检查通常会不同的地方：对话模板（重复的 BOS、系统提示、生成提示）、哪些 token 被打分
     （只有回答部分，是否包含结束 token）、padding 方向和注意力掩码、packing 序列里的位置 id、截断。
   - 成功标准：每个存下来的序列，id 和位置都相等。
2. **确认用的是同样的权重。**
   - 目标：引擎持有的参考或 teacher 权重，就是 trainer 加载的那份。
   - 做法：用一条带两边逐行指纹的 `weight_sync` 记录，用 OpGuard 的 `rl.weight_sync_check` 检查；
     如果引擎加载的是转换过的检查点，就在两个文件上运行 [ckpt_compare.py](../../tools/ckpt_compare.py)
     （[检查点转换](checkpoint-conversion.md)）。
   - 成功标准：没有不匹配的行。
3. **比较数值。**
   - 目标：每个被打分的 token，trainer 的 log-prob 与存下来的逐位相等。
   - 做法：把存下来的 token id 以 teacher forcing 方式送进 trainer 的前向，不带容差地比较 fp32 的位；
     在参考版本上运行 `rl.decompose`，它按 token 报告数值误差（staleness 按构造为零）
     （[强化学习循环](../2-rl/rl-loop.md)）。
   - 成功标准：每个 token 的数值误差都为零，或者给出不为零的 token 列表。
4. **数值不同时，让引擎对齐，或者换个地方算。**
   - 目标：数值误差为零，或者有一个决定。
   - 做法：引擎会把不同长度的序列放进同一批，所以先让它自身满足不变性（[整个模型](../2-rl/whole-model.md)），
     再把它的舍入点和 trainer 对齐（[引擎一致性](../2-rl/engine-parity-vllm.md)、
     [engine-parity skill](../../skills/engine-parity/SKILL.md)）；剩下的差异像[强化学习循环](../2-rl/rl-loop.md)
     那样用逐向量指纹定位。或者在 trainer 里算参考。
   - 成功标准：第 3 步的检查通过，或者测得的差距成为一项记录下来的决定。

| 选项 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 在 trainer 里算参考 | 按构造精确 | 中等（不到 2 倍）：每个样本多一次参考模型的前向 | 吞吐允许时的默认选择 |
| 做到一致性的引擎 | 精确，并且是引擎的速度 | 引擎不变模式的代价（中等到大），加一次性的对齐工作 | 大数据集，打分一次、反复使用 |
| 引擎保持原样，测量差距 | 一个已知、已记录的误差 | 免费 | 这个用途接受测得的数值误差 |

这些代价等级来自每个选项增加的工作量，这里没有针对后训练测量过。
引擎不变模式的代价在[引擎一致性报告](../../examples/cross_engine/REPORT.md)里有测量。

对**蒸馏**，还有两件事决定数值能不能对上：

- **top-k 的选择。**只存 teacher 的 top-k log-prob 时，选择本身就可能不同：
  第 k 名附近有两个值相等或几乎相等。先比较选出的 id，再比较值。
- **归一化。**在整个词表上归一化的 log-prob，和在 top k 上重新归一化的不是同一个数；
  temperature 也可以在归一化之前或之后施加。写下每一侧算的是哪一种。

尚未证明：这里没有测量过任何 DPO 或蒸馏运行。各个部分是分别证明的：
一个模型上的引擎一致性（[报告](../../examples/cross_engine/REPORT.md)），
以及小循环上的记录、权重检查和拆分（[报告](../../examples/rl_loop/REPORT.md)）。

## 可能出错的地方

!!! warning "某一侧用了另一个模板"
    引擎套用了自己的对话模板，或者加了一个 trainer 会再加一次的 BOS。两边打分的于是是不同的 token 序列，
    差距很大，看起来却像数值问题。先比较 token id，再比较值。

!!! warning "padding 改变了数值"
    随批次变化的引擎或 trainer，在 padding 或 packing 方式不同时会给同一序列不同的位。
    差距于是取决于哪些序列同在一批。先检查每一侧单独计算和批量计算是否一致
    （[整个模型](../2-rl/whole-model.md)）。

!!! warning "打分的位置不同"
    一侧对 prompt 和回答一起求和，另一侧只对回答求和；或者一侧计入了结束 token。
    每个 token 的值都一致，总和却不同。存储并比较带位置的逐 token 值，而不是总和。

!!! warning "引擎给的是处理过的 log-prob"
    引擎可能报告经过 temperature、惩罚项或其他处理之后的 log-prob。它们不是模型的 log-prob，
    任何 trainer 都对不上。向引擎要原始 log-prob，并用它自己 logits 的 log_softmax 核对一个。

## 小结

- 预先算好的参考 log-prob 或 teacher logit 就是去掉了采样器的 rollout：trainer 必须逐位复现它。
- 第 0 步时 DPO 的 margin 必须恰好为零；在那里重算一部分样本是一项免费的检查。
- 依次检查 token、权重、数值；大的差距大多是记账问题。
- 数值不同时，用引擎一致性让引擎对齐，或者在 trainer 里算参考。

## 深入阅读

- [引擎一致性](../2-rl/engine-parity-vllm.md)和 [engine-parity skill](../../skills/engine-parity/SKILL.md)。
- [整个模型](../2-rl/whole-model.md)：每一侧的批不变性和模式不变性。
- [强化学习循环](../2-rl/rl-loop.md)：本页复用的记录、权重检查和拆分；OpGuard 的协议见它的 `docs/RLAlignment.md`。
- 引擎加载转换过的参考模型时，见[检查点转换](checkpoint-conversion.md)。
