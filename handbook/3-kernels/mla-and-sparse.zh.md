# MLA 与稀疏注意力

## 我们在哪里

[注意力页](attention.md)固定了对 key 的走法，让每种模式给出相同的位。两种较新的注意力设计带来
了走法覆盖不到的问题。MLA（latent attention）可以用两种代数相等的形式计算，引擎常在 prefill
用一种、在 decode 用另一种。稀疏注意力先选出要读哪些 key，而选择是离散的。前两页的工具仍然
适用：不变的 GEMM 和固定的 key 走法。

## 思路

### 两种相等的形式在不同位置舍入

MLA 为每个 token 缓存一个小的 latent 向量，而不是逐头的 key 和 value。逐头的 key 通过矩阵 W
从它得到。query q 对缓存的 latent c 的分数可以按两种顺序计算：

| 形式 | 顺序 | 什么被舍入到 bf16 |
|---|---|---|
| 显式 | 先解压：k = W·c，再算 q·k | 逐头的 key 和 value |
| 吸收式 | 先把 W 并入 query：q' = q·W，再算 q'·c | 吸收后的 query 和 latent 输出 |

(q·W)·c 和 q·(W·c) 在精确算术中是同一个数，舍入后是不同的数。两种形式精度一样，只是不同的
选择。规则是注意力规则的再上一层：**每种模式、两侧都用同一种形式。** 在 prefill 用显式、在
decode 用吸收式的引擎，与只运行一种形式的 trainer 不再一致。

### 选择会因微小变化而翻转

稀疏注意力用一个小的 indexer 给过去的 key 打分，保留前 k 个，只在它们上面做注意力。假设第 k 个
和下一个分数是 0.7001 和 0.7000。如果一个随批变化的 GEMM 让前者在最后一位降低几个单位，两个
key 就互换，一个不同的 key 进入求和，输出的变化远大于一次舍入。

所以选择需要三样东西：不变的分数（喂给它的每个 GEMM）、写明的平局顺序（分数降序，再按索引
升序），以及按索引升序而不是按选择返回的顺序对选中的 key 求和。

## 步骤

### 1. 让每个 M 随步变化的 GEMM 都不变

- **目标：** 投影、解压和逐头吸收在 prefill、分块和 decode 中看到的行数不同；每个都必须对 M
  不变。
- **方法：** [GEMM 页](gemm-and-quantized.md)的单配置 GEMM，逐头乘积用分组的单配置 GEMM。
- **成功标准：** 每一块的 `check_batch` 都通过。在一种 GPU 上得到证明：不变的注意力核心外面
  包着库 GEMM，仍然失败（[报告](../../examples/mla_sparse/REPORT.md#what-this-shows)）。

### 2. 每种模式选同一种 MLA 形式

- **目标：** prompt、生成的 token 和 trainer 都用同一种形式。
- **方法：** 一个服务所选形式、对 latent 缓存采用固定走法的注意力核心，见
  [recipe 5](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention)
  和 [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py)。
- **成功标准：** 在同一形式内，所有模式和批都逐位相等，引擎运行与 trainer 运行相等。在一种 GPU
  上、一层和两层堆叠下得到证明
  （[报告](../../examples/mla_sparse/REPORT.md#part-a-mla-absorbed-vs-explicit)）。

| 方案 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 显式 prefill、吸收式 decode（引擎常见做法） | 与 trainer 不一致 | 免费 | 一致性场景下不选 |
| 处处吸收式 | L2、L3，两侧同一形式 | prefill 中等 | 一致性的常用选择 |
| 处处显式，每步解压缓存 | L2、L3 | decode 大 | 只适合短上下文 |
| 处处显式，配逐头的 key/value 缓存 | L2、L3 | decode 中等到大，外加大的缓存 | 内存允许时 |

### 3. 让选择稳定

- **目标：** 选中的集合和输出不随批或模式变化。
- **方法：** [recipe 6](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k)：
  不变的投影和 indexer GEMM，indexer 对头求和的顺序固定，按（分数降序，索引升序）的稳定 top-k，
  选中的索引升序排列，注意力按这个顺序访问。
- **成功标准：** 没有位置改变选中集合，然后没有输出元素改变。在一种 GPU 上得到证明
  （[报告](../../examples/mla_sparse/REPORT.md#part-b-sparse-attention-with-selection-dsa-like)）。

| 部件 | 方案 | 代价等级 |
|---|---|---|
| 投影 GEMM | 单配置 | 免费到小；它消除了测到的每一次翻转 |
| Indexer | 对头按固定顺序的 fp32 求和 | 免费（比未融合的基线更快） |
| Top-k | 按（分数，索引）的稳定排序 | 对整层来说小；top-k 这一步本身比库的慢 |
| 稀疏注意力 | 索引升序，固定 block | 免费 |

### 4. 先验证集合，再验证数值

运行[验证矩阵](../../skills/batch-invariant-kernel/references/verification.md)；对稀疏注意力，
先比较选中的索引集合：只比数值会把离散的翻转和舍入漂移混在一起。要在批中测试完整的 decode，
而不只是一步；要至少测两层堆叠。

## 可能出错的地方

!!! warning "一层会掩盖混合形式的误差"
    只有一层时，latent 缓存与形式无关，所以吸收式的 trainer 可能碰巧一致。从第 2 层起，缓存
    来自另一种形式的输出；要测两层或更多。

!!! warning "只有注意力核心是不够的"
    解压 GEMM（它的 M 是缓存的 token 数）和逐头输出吸收很容易漏掉。任何一个留在库上，随步变化
    的 M 就会破坏 L2 和 L3。

!!! warning "库 top-k 的平局顺序不是约定"
    每次测试中它都在平局里选了最低的索引，但这个顺序没有文档保证。用稳定排序把平局顺序写明。

!!! warning "大多数翻转来自分数，而不是平局"
    投影还随批变化时，稳定的 top-k 一次翻转也没有消除。先修 GEMM，再修 top-k。

## 小结

- 吸收式和显式 MLA 精度一样、逐位不同：每种模式、两侧都选同一种形式。
- 每个行数随步变化的 GEMM 都必须不变，包括解压和逐头吸收。
- 分数的微小变化就会让选择翻转：不变的分数、稳定的（分数，索引）top-k、排序后累加。
- 先比较选中集合再比较数值，并测两层。

## 深入阅读

- 报告：[MLA 与稀疏注意力](../../examples/mla_sparse/REPORT.md)。
- 代码：[examples/mla_sparse/mla_sparse.py](../../examples/mla_sparse/mla_sparse.py)、
  [mla_sparse_kernels.py](../../examples/mla_sparse/mla_sparse_kernels.py)。
- Recipe：[5，MLA](../../skills/batch-invariant-kernel/references/operator-recipes.md#5-mla-latent-attention)、
  [6，稀疏注意力](../../skills/batch-invariant-kernel/references/operator-recipes.md#6-sparse-attention-with-selection-indexer-plus-top-k)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
