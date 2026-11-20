# MLA 与稀疏注意力

## 我们在哪里

[注意力一页](attention.md)固定了遍历 key 的方式，让每种模式的结果逐 bit 一致。两种较新的注意力
设计又带来了固定遍历方式解决不了的问题。MLA（multi-head latent attention，多头潜在注意力）
可以用两种代数上相等的形式计算，而引擎常常在 prefill 用一种、在 decode 用另一种。稀疏注意力
先挑出要读哪些 key，而挑选是一个离散的选择。前两页的工具在这里依然适用：不变的 GEMM，以及
固定的 key 遍历方式。

## 思路

### 同一套数学的两种写法，在不同的地方舍入

MLA 为每个 token 只缓存一个小的 latent 向量，而不是为每个头分别缓存 key 和 value。每个头的
key 由这个向量乘以矩阵 W 得到。于是 query q 对缓存的 latent c 的分数，可以按两种顺序来算：

| 形式 | 计算顺序 | 哪些量被舍入到 bf16 |
|---|---|---|
| 显式 | 先解压，k = W·c，再算 q·k | 每个头的 key 和 value |
| 吸收式 | 先把 W 并入 query，q' = q·W，再算 q'·c | 吸收后的 query 和 latent 输出 |

(q·W)·c 和 q·(W·c) 在精确算术里是同一个数，舍入之后就成了两个不同的数。两种形式的精度不分
高下，只是两种不同的选择。规则是把注意力的规则往上推一层：**每种模式、训推两端都用同一种形式。**
如果引擎在 prefill 用显式、在 decode 用吸收式，它就和只用一种形式的 trainer 对不上了。

### 分数的一点微小变化，就能改变选中哪些 key

稀疏注意力用一个小的 indexer 给之前的 key 打分，只保留分数最高的 k 个，并且只在它们上面做
注意力。假设第 k 名和第 k+1 名的分数是 0.7001 和 0.7000。如果某个随 batch 变化的 GEMM 让前一个分数
在最后几个 bit 上小了一点，这两个 key 就会互换名次。于是另一个 key 进入了求和，输出的变化
远远超过一次舍入。

所以挑选这一步需要三样东西：分数本身不变，也就是喂给它的每个 GEMM 都不变；明确写下的平局顺序
（先按分数降序，再按索引升序）；以及按索引升序对选中的 key 求和，而不是按挑选结果返回的顺序。

## 步骤

### 1. 让所有 GEMM 都不变，尤其是行数随步变化的那些

- **目标：** 投影、解压和每个头上的吸收，在 prefill、chunk 和 decode 里看到的行数各不相同，
  所以它们每一个都必须对 M 不变。
- **做法：** 用 [GEMM 一页](gemm-and-quantized.md)里的单配置 GEMM；每个头上的乘积用分组的
  单配置 GEMM。
- **成功标准：** 每一部分的 `check_batch` 都通过。这在一种 GPU 上得到了证明：注意力核心本身已经
  不变，但外面包着的是库里的 GEMM，结果仍然失败
  （[报告](../../examples/mla_sparse/REPORT.md#what-this-shows)）。

### 2. 每种模式、训推两端都用同一种 MLA 形式

- **目标：** prompt、生成出来的 token 和 trainer 都用同一种形式。
- **做法：** 用一个按所选形式计算、并以固定方式遍历 latent cache 的注意力核心，见
  [recipe 5](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention)
  和 [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py)。
- **成功标准：** 在所选形式下，所有模式和所有 batch 的结果逐 bit 一致，引擎的运行结果也和 trainer 的
  相等。这在一种 GPU 上、单层和两层堆叠的情况下得到了证明
  （[报告](../../examples/mla_sparse/REPORT.md#part-a-mla-absorbed-vs-explicit)）。

| 方案 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| prefill 用显式、decode 用吸收式（引擎里常见） | 和 trainer 对不上 | 免费 | 需要一致性时不选 |
| 处处用吸收式 | L2、L3，两端用同一种形式 | prefill 中等 | 追求一致性时的常用选择 |
| 处处用显式，每一步都解压 cache | L2、L3 | decode 大 | 只适合短上下文 |
| 处处用显式，并为每个头缓存 key 和 value | L2、L3 | decode 中等到大，另外 cache 很大 | 内存允许时 |

### 3. 让 key 的挑选保持稳定

- **目标：** 选中的 key 集合和输出都不随 batch 或模式变化。
- **做法：** 按
  [recipe 6](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k)
  来做：投影和 indexer 的 GEMM 都要不变；indexer 对各个头求和时顺序固定；top-k 用稳定排序，
  先按分数降序、再按索引升序；选中的索引按升序排列，注意力也按这个顺序访问它们。
- **成功标准：** 先确认没有任何位置的选中集合发生变化，再确认没有任何输出元素发生变化。这在
  一种 GPU 上得到了证明
  （[报告](../../examples/mla_sparse/REPORT.md#part-b-sparse-attention-with-selection-dsa-like)）。

| 部件 | 做法 | 代价等级 |
|---|---|---|
| 投影 GEMM | 单配置 | 免费到小；测到的每一次翻转都是靠它消除的 |
| Indexer | 按固定顺序、在 fp32 下对各个头求和 | 免费（比未融合的基线还快） |
| Top-k | 按（分数，索引）稳定排序 | 对整层来说小；不过 top-k 这一步本身比库里的慢 |
| 稀疏注意力 | 按索引升序，固定 block | 免费 |

### 4. 先比较选中的集合，再比较数值

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)。对稀疏注意力，
要先比较选中的索引集合，因为只比数值会把离散的翻转和普通的舍入误差混在一起。要在 batch 里测完整的
decode 过程，而不只是一步，并且至少测两层堆叠。

## 可能出错的地方

!!! warning "只测一层，会掩盖混用两种形式的错误"
    只有一层时，latent cache 与形式无关，所以用吸收式的 trainer 可能碰巧对得上。从第 2 层开始，
    cache 来自另一种形式的输出，所以要测两层或更多。

!!! warning "只修注意力核心是不够的"
    解压用的 GEMM（它的 M 是缓存的 token 数）和每个头上对输出的吸收，很容易被漏掉。只要其中一个
    还用库里的实现，它的 M 每一步都在变，就会破坏 L2 和 L3。

!!! warning "库里 top-k 处理平局的顺序没有保证"
    每次测试中，它都在分数相同的 key 里选了索引最小的那个，但这个顺序并没有写进文档。用稳定排序
    把平局顺序明确下来。

!!! warning "大多数翻转来自分数本身，而不是平局"
    投影还随 batch 变化的时候，稳定的 top-k 一次翻转也没能消除。先修 GEMM，再修 top-k。

## 小结

- 吸收式和显式 MLA 精度相同，但结果并不逐 bit 一致，所以每种模式、训推两端都要用同一种形式。
- 每个行数随步变化的 GEMM 都必须不变，包括解压和每个头上的吸收。
- 分数的微小变化就能让挑选结果翻转，所以要用不变的分数、按（分数，索引）排序的稳定 top-k，
  并按排好的顺序累加。
- 先比较选中的集合，再比较数值，并且至少测两层。

## 深入阅读

- 报告：[MLA 与稀疏注意力](../../examples/mla_sparse/REPORT.md)。
- 代码：[examples/mla_sparse/mla_sparse.py](../../examples/mla_sparse/mla_sparse.py)、
  [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py)。
- Recipe：[5，MLA](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention)、
  [6，稀疏注意力](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
