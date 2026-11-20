# `suggest` 给出的线索

`opguard_bridge.suggest(facts, preflight)` 返回一组按优先级排好的线索。每条线索都带
一个 `kind`、它所依据的证据、一个要运行的 `check`，以及一处手册引用。在被某个干预
确认之前，任何一条线索都还不是原因。

| 类别 | 依据什么 | 怎样确认 |
|---|---|---|
| `preflight` | 预检报告里有阻断性的差异，或者有没控制住的非确定性来源 | 让两次运行在这一项上保持一致，然后重新采集 |
| `alignment` | pivot 之前有未对齐的事件 | 检查第一个未对齐的事件；只有在有证据表明两个事件相互对应时，才把它们配对（`review_pair`） |
| `rounding-point` | 同一次调用在两次运行里记录的 dtype 不同，或者一次运行里有另一次从不运行的调用点 | 写下两次运行的舍入序列，把 suspect 运行里的类型转换挪到参考运行做转换的地方，再重跑 |
| `layout` | 同一次调用在两次运行里看到的形状不同 | 逐行比较逐向量的 fingerprint，并检查这条路径是不是 batch-invariant 的（L2） |
| `nondeterministic-backward` | 前面的前向事件都一致，某个梯度消费者收到的输入却不同 | 对怀疑的反向算子运行 `bi_check.check_repeat`，再试试严格确定性模式 |
| `consumer` | 输入已经不同 | 用 `trace-upstream` 顺着差异往上游追 |
| `operator-class` | 算子名、它所在的源码行或它的调用者里出现了某个类别名 | 对这个算子运行 `check_repeat`（L1）和 `check_batch`（L2） |
| `rng` | 随机算子收到的输入相同，输出却不同 | 比较预检报告里的 RNG 摘要 |
| `state` | 查找操作的索引相同，结果却不同 | 比较恢复之后优化器和 scheduler 的状态 |
| `data` | 第一处差异出现在数据边界上 | 比较种子、sampler 的状态和 loader 的位置 |

在实测案例中（[pretraining-opguard-e2e](../../../handbook/1-pretraining/localize-with-opguard.md)），
每个事先植入的原因，都是排在预检线索之后的第一条线索。那一页还列出了每个原因除了
`suggest` 之外，还要做什么才能得到确认。
