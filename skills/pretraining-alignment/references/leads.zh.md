# `suggest` 给出的线索及其确认方法

`opguard_bridge.suggest(facts, preflight)` 返回排好序的线索。每条线索有一个
`kind`、它所依据的证据、一个 `check` 和一处手册引用。在干预确认之前，
任何一条都不是原因。

| 类别 | 依据 | 确认方法 |
|---|---|---|
| `preflight` | 报告中的阻断性差异或未受控的面 | 让两次运行在这一点上一致，然后重新捕获 |
| `alignment` | pivot 之前的未对齐事件 | 检查第一个未对齐事件；只凭证据配对（`review_pair`） |
| `rounding-point` | 同一次调用记录了不同的 dtype，或一侧有另一侧从未运行的调用点 | 写出两侧的舍入序列；把可疑一侧的类型转换移到参考一侧的位置；重新运行 |
| `layout` | 同一次调用的形状不同 | 逐行比较逐向量 fingerprint；路径的批不变性（L2） |
| `nondeterministic-backward` | 前向事件一致之后，某个梯度消费者的输入不同 | 对可疑的反向使用 `bi_check.check_repeat`；严格确定性模式 |
| `consumer` | 输入已经不同 | `trace-upstream` |
| `operator-class` | 算子名、其所在行或其调用者中出现某个类名 | 对该算子运行 `check_repeat`（L1）、`check_batch`（L2） |
| `rng` | 随机算子的输入相等而输出不同 | 预检报告中的 RNG 摘要 |
| `state` | 查找操作的索引相等而结果不同 | 恢复之后的优化器和调度器状态 |
| `data` | 第一处差异出现在数据边界 | 种子、采样器状态、加载器位置 |

在实测案例中（[pretraining-opguard-e2e](../../../handbook/1-pretraining/localize-with-opguard.md)），
每个人为植入的原因都是第一条非预检线索；每个原因在 `suggest` 之外还需要什么，
都列在那一页。
