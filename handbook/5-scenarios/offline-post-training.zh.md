# 离线后训练

## 我们在哪里

好几种后训练方法会提前算好 loss 里的某一项，而且为了速度，常常放在推理引擎里算。DPO 及其变体
要用参考模型对被选回答和被拒回答给出的 logprob；带 KL 项的 SFT 要用参考 logprob；蒸馏要用
teacher 的 logits，或者它的 top-k logprob。trainer 随后就把这些存下来的数当成自己算的来用。
如果引擎和 trainer 对同样的 token 算出了不同的数，loss 里就混进了一个谁也没打算引入的误差。
这其实就是[第 2 部分](../2-rl/README.md)里的 L4 问题，只是没有采样器：token 由数据集固定，
参考权重也不会变，所以没有 staleness，剩下的只有记账错误和数值误差。

## 思路：第 0 步自带一项免费的检查

DPO 在每一对回答上比较策略和参考，一个是偏好的回答 y_w，一个是被拒的回答 y_l：

```
margin = (logp_policy(y_w) - logp_ref(y_w)) - (logp_policy(y_l) - logp_ref(y_l))
loss   = -log sigmoid(beta * margin)
```

第 0 步时，策略就是参考本身，所以每个 margin 都恰好是 0，loss 恰好是 log 2。因此，如果存下来的
参考 logprob 来自引擎，第 0 步就能告诉你它们对得有多准：

| logp_ref 从哪里来 | 第 0 步的 margin |
|---|---|
| trainer 自己的前向 | 每一对都恰好为 0 |
| 在别的地方舍入的引擎 | 很小但不为零，而且每一对都不一样 |
| 给别的 token 打了分的引擎（模板不同、位置不同） | 并不小，因为打分的是另一个序列 |

所以在第 0 步，让 trainer 对一部分样本对重新计算参考 logprob，并和存下来的值逐 bit 比较。
如果相等，说明预计算是精确的。如果不相等，先问打分的是不是同样的 token，再问用的是不是
同样的权重，最后才去调试数值问题。

所以规则是：**预先算好的数，trainer 必须能用相同的 token 和相同的权重逐 bit 复现出来；
检查时先看 token，再看权重，最后看数值。**这就是第 2 部分拆分差距的做法：先把记账错误和
数值误差分开，再去调试其中任何一个。

## 步骤

1. **确认打分的是同样的 token**，而且位置相同。每个预先算好的值，都要连同它的 token id、位置，
   以及每个 logprob 的精确 bit 一起存下来，使用 OpGuard 的 `rollout_batch` 记录里的字段
   （`opguard_harness.rl_records.RecordWriter`）；trainer 则把自己算的值写成一条
   `trainer_logprob` 记录。比较任何值之前，先比较 id，并重点检查通常会不一样的地方：对话模板
   （重复的 BOS、系统提示、生成提示）、哪些 token 参与打分（是否只算回答部分，是否包含结束
   token）、padding 在左边还是右边以及 attention mask、packing 序列里的 position id，以及截断。
   每一条存下来的序列，id 和位置都必须相等。
2. **确认用的是同样的权重**：引擎持有的参考模型或 teacher 权重，必须正是 trainer 加载的那一份。
   写一条带有两端逐行 fingerprint 的 `weight_sync` 记录，用 OpGuard 的 `rl.weight_sync_check`
   检查，每一行都必须配上对。如果引擎加载的是转换过的 checkpoint，就改为在两个文件上运行
   [ckpt_compare.py](../../tools/ckpt_compare.py)（[checkpoint 转换](checkpoint-conversion.md)）。
3. **比较数值。**用 teacher forcing 把存下来的 token id 送进 trainer 的前向，逐 bit 比较 fp32
   的值，不设任何容差。在参考模型的版本上运行 `rl.decompose`，它会按 token 报告数值误差（这里的
   staleness 天然为零）（[RL 循环](../2-rl/rl-loop.md)）。最后你得到的是：每个参与打分的 token
   数值误差都为零，或者一份数值误差不为零的 token 清单。
4. **如果数值不同，就让引擎对齐，或者换个地方算。**引擎会把不同长度的序列放进同一个 batch，
   所以先让它自己满足不变性（[整个模型](../2-rl/whole-model.md)），再把它的舍入点和 trainer 对齐
   （[引擎一致性](../2-rl/engine-parity-vllm.md)、[engine-parity skill](../../skills/engine-parity/SKILL.md)），
   剩下的差异则像 [RL 循环](../2-rl/rl-loop.md)里那样，用逐向量 fingerprint 定位。另一种办法是
   直接在 trainer 里算参考。第 3 步的检查通过，或者测得的差距已经作为一项决定记录下来，就算完成了。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 在 trainer 里算参考 | 天然精确 | 中等（不到 2 倍）：每个样本多做一次参考模型的前向 | 吞吐允许时的默认做法 |
| 使用与 trainer 逐 bit 一致的引擎 | 精确，而且是引擎的速度 | 引擎开启不变模式的代价（中等到大），再加一次性的对齐工作 | 数据集很大，打一次分、反复使用 |
| 引擎保持原样，只测量差距 | 一个已知、已记录的误差 | 免费 | 对这个用途来说，测得的数值误差可以接受 |

这些代价等级是根据每个选项额外增加的工作量推断的，这里并没有针对后训练实测过。引擎不变模式
的代价，在[引擎一致性报告](../../examples/cross_engine/REPORT.md)里有实测。

对于**蒸馏**，还有两件事决定数值能不能对上：

- **top-k 选的是哪些。** 如果只存了 teacher 的 top-k logprob，选出来的集合本身就可能不同：
  第 k 名附近有两个值相等或几乎相等。先比较选出的 id，再比较值。
- **归一化方式。** 在整个词表上归一化的 logprob，和只在 top k 里重新归一化的，不是同一个数；
  temperature 也可能在归一化之前或之后施加。要写清楚每一端算的是哪一种。

尚未证明：这里还没有实测过任何 DPO 或蒸馏训练。各个组成部分是分别证明的：一个模型上的引擎
一致性（[报告](../../examples/cross_engine/REPORT.md)），以及一个小训练循环上的记录、权重检查
和差距拆分（[报告](../../examples/rl_loop/REPORT.md)）。

## 可能出错的地方

!!! warning "有一端用了不同的模板"
    引擎套用了自己的对话模板，或者加了一个 BOS，而 trainer 又加了一次。两端打分的于是是不同的
    token 序列，差距很大，看起来却像数值问题。要先比较 token id，再比较值。

!!! warning "padding 改变了数值"
    不是 batch-invariant 的引擎或 trainer，在 padding 或 packing 方式不同时，会给同一条序列
    算出不同的结果。差距于是取决于哪些序列碰巧在同一个 batch 里。先检查每一端单独计算和放进
    batch 计算的结果是否一致（[整个模型](../2-rl/whole-model.md)）。

!!! warning "打分的位置不同"
    一端对 prompt 和回答一起求和，另一端只对回答求和；或者一端把结束 token 也算了进去。这时
    每个 token 的值都一致，总和却不同。要存储并比较带位置的逐 token 值，而不是总和。

!!! warning "引擎给出的是处理过的 logprob"
    引擎报告的 logprob 可能已经经过 temperature、惩罚项或其他处理。它们不是模型本身的 logprob，
    任何 trainer 都对不上。要向引擎要原始的 logprob，并拿其中一个和引擎自己 logits 的
    log_softmax 核对。

## 小结

- 预先算好的参考 logprob 或 teacher logit，就是一个去掉了采样器的 rollout，trainer 必须
  逐 bit 复现它。
- 第 0 步时 DPO 的 margin 必须恰好为零，所以在这一步重算一部分样本对，就是一项免费的检查。
- 先查 token，再查权重，最后查数值；大的差距多半是记账错误。
- 如果数值不同，要么通过引擎一致性让引擎对齐，要么在 trainer 里算参考。

## 深入阅读

- [引擎一致性](../2-rl/engine-parity-vllm.md)和 [engine-parity skill](../../skills/engine-parity/SKILL.md)。
- [整个模型](../2-rl/whole-model.md)：让每一端不受 batch 和推理模式的影响。
- [RL 循环](../2-rl/rl-loop.md)：本页沿用的记录、权重检查和差距拆分；OpGuard 的协议见它的
  `docs/RLAlignment.md`。
- 引擎加载的是转换过的参考模型时，见[checkpoint 转换](checkpoint-conversion.md)。
