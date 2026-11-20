# trainer 与 vLLM 舍入不同的地方

本页列出 Llama/Qwen2 风格的 decoder 里，trainer 和引擎可能舍入不同的每一个地方，
以及两端在那里各自怎么做。这里的“Trainer”指
[examples/whole_model/model.py](../../../examples/whole_model/model.py)，
它沿用 transformers 的运算顺序；“vLLM”指在一种 GPU 上用 bf16 部署
Qwen2.5-0.5B-Instruct 的 vLLM 0.19.1，torch 版本是 2.10.0+cu128。硬件和结果见
[examples/cross_engine/REPORT.md](../../../examples/cross_engine/REPORT.md)。

“证据”一列用到三种证据：

- **src**：读自 vLLM 0.19.1 的 Python 源码（写明了文件）。
- **replay**：用算子重放测得。把从引擎采集到的输入送进 trainer 的候选实现，再逐 bit
  比较输出（[cross_engine.py](../../../examples/cross_engine/cross_engine.py) 的
  `op_replay`，用一个 50 token 的 prompt，跑一次 prefill 和两步 decode，覆盖全部 24 层）。
- **tokens**：在实际出现的 token 的 logprob 上测得（64 个 prompt，batch size 1–64）。

wheel 里没有 vLLM 自定义算子（`torch.ops._C.*`）和 FlashAttention 的 CUDA 源码，
所以下面关于它们的行为都是测出来的，不是从源码里读出来的。

## 逐个算子看舍入点

| # | 舍入点 | Trainer（model.py，未改动） | vLLM eager（自定义算子） | vLLM batch-invariant eager（`VLLM_BATCH_INVARIANT=1`，在所测 GPU 上） | 证据 |
|---|---|---|---|---|---|
| 1 | Embedding | gather，结果精确 | gather，结果精确 | gather，结果精确 | src |
| 2 | norm 之前的残差相加 | `h = dt(h + o)`，norm 随后读的是舍入后的 `h` | 融合进 `fused_add_rms_norm`，原地计算 | 先在 bf16 里算 `x + residual`（`layernorm.py fused_add_rms_norm`），再做 norm | src；replay 显示三种做法都会先把和舍入一次 |
| 3 | RMSNorm 的方差 | 用 torch 的归约算 `mean(fp32(x)^2)`，再乘 `* rsqrt(var + eps)` | 一个 CUDA kernel；141 次调用里有 137 次和 trainer 公式的结果相同，可见归约顺序不一样 | Triton `_rms_norm_kernel`：在 1024 宽的块上做 `tl.sum`，算 `sum / n`，再算 `1 / sqrt(mean + eps)` | src（BI）；replay |
| 4 | Norm 权重 | 先转成 bf16，再在 bf16 里 `* w` | 和 trainer 相同（replay 结果相等） | 在 fp32 里算 `(x * inv) * w`，最后只转换一次 | src（BI）；replay：74% 的元素和 trainer 的结果相同 |
| 5 | Q/K/V 打包 | 三个 GEMM（N = 896, 128, 128） | 打包成一个 GEMM，N = 1152 | 打包成一个 GEMM，N = 1152 | src（`qwen2.py` QKVParallelLinear）；replay：分开调用 cuBLAS GEMM 时，72 次调用里有 30 次结果不同 |
| 6 | Q/K/V 的 bias | 在 GEMM epilogue 里、舍入之前加上，即 `dt(sum + b)`（cuBLAS addmm） | 相同（`F.linear(x, W, b)`） | 在类型转换之后才加，即 `dt(dt(sum) + b)`（`linear_batch_invariant`） | src；replay 里，选对做法时 72/72 次调用相等，选另一种时 0/72 |
| 7 | GEMM kernel（q/k/v、o、gate/up、down） | cuBLAS，按形状挑 kernel | cuBLAS，按形状挑 kernel（达不到 L2） | `matmul_persistent`（Triton，128x128x64 的 tile，在 fp32 里按 K 的顺序累加）。M = 50 时，cuBLAS 在 q/k/v 和 o 上给出相同的结果，在 gate/up（25/72）和 down（24/72）上则不同 | src；replay |
| 8 | Gate/up 打包 | 两个 GEMM | 打包成一个 GEMM，N = 9728 | 打包成一个 GEMM，N = 9728 | src；replay：分开调用 cuBLAS 时 48/72 次相等，打包后 72/72 次相等（eager） |
| 9 | RoPE 表 | transformers 在 CPU 上算 `inv_freq`，在 GPU 上用 fp32 算角度和 cos/sin，再转成 bf16 | `cos_sin_cache` 在 GPU 上用 fp32 构建（`base ** (arange/D)` 在设备上计算），再转成 bf16 | 和 eager 相同 | src（`rotary_embedding/base.py`）；实测：表里 262,144 个条目中有 26 个不同，第一个出现在位置 459 |
| 10 | RoPE 公式 | `dt(dt(q cos) + dt(rot(q) sin))` | `_C.rotary_embedding`，原地计算 | 和 eager 相同 | replay：和 trainer 的公式逐 bit 一致（50 个 token 时，用哪张表都是 72/72） |
| 11 | 注意力 | SDPA，先把 K/V 扩展到 query head 的个数 | paged KV（block 16）上的 FlashAttention 2（vllm_flash_attn），`num_splits=0`，也就是由启发式规则决定怎么拆分 | 相同，但用 `num_splits=1` | src（`flash_attn.py`）；replay：SDPA 在 24 层里有 11–12 层相等，从 trainer 里调用 FA2 则 24/24 层相等 |
| 12 | decode 和 prefill 里的注意力 | trainer 只跑 prefill | 每个 decode 步都对 cache 做注意力 | 相同 | replay：FA2 的 decode 输出等于 trainer 对整个序列做 FA2 prefill 的结果，48/48 次调用（num_splits 为 0 和 1） |
| 13 | SiLU 与乘法 | 在 bf16 里算 `dt(silu(g)) * u` | `_C.silu_and_mul` | 相同 | replay：逐 bit 一致（72/72） |
| 14 | 最后的 norm | 同第 2–4 行 | 融合的相加 + norm | 在 bf16 的和上做 Triton norm | src；replay |
| 15 | lm_head | 在全部 T 个位置上用 cuBLAS | cuBLAS（`F.linear`），只算要采样的那些行（decode 时 M 等于 batch size） | **cuBLAS**。这个模式只在列出的几种 compute capability 上覆盖 `aten::mm/addmm/linear`（`batch_invariant.py enable_batch_invariant_mode`），所测 GPU 不在其中。lm_head 走的是 `F.linear`，所以仍然用 cuBLAS；这个模式只固定了 cuBLAS 的 workspace | src；tokens：这破坏了引擎自己的 L2（单独运行和放进 batch 运行相比，15,231 个 logprob 里有 1,116 个不同） |
| 16 | 送进采样器的 logits 的 dtype | 先把 bf16 logits 转成 fp32，再在 fp32 里做 `log_softmax` | 在 GPU 上调用 `logits.log_softmax(-1, dtype=float32)` | `aten::_log_softmax` 被一个 Triton kernel 替换：在每个 1024 宽的块上取 max，逐块累加 `sum(tl.exp(x - max))`，最后返回 `x - max - log(sum)` | src（`sampler.py compute_logprobs`、`batch_invariant.py`）；实测：eager 模式下，引擎的 logprob 和 GPU 上 `log_softmax(fp32(logits))` 的结果逐 bit 一致 |
| 17 | log_softmax kernel | torch | torch | Triton（见上一行） | tokens：把它换回 torch，会改变 56% 的 logprob，最多改变 1.9e-6 |
| 18 | Prompt logprobs | 在同一次前向里算出 | 对 prompt 的那些行单独调用一次 `compute_logits`（M 等于 chunk 大小） | 相同 | src |
| 19 | 编译模式（默认） | eager | 关掉自定义算子；RMSNorm、RoPE、SiLU 通过 Inductor 运行 `forward_native` 并被融合；使用 CUDA graph | 相同，另外用 Triton GEMM | src（`config/vllm.py` 在 Inductor 下把 custom_ops 设为 "none"）；tokens：我们试过的所有 trainer 做法都对不上 |
| 20 | Cascade attention | 不使用 | 有共享前缀时可能会用 | 被这个模式禁用 | src（`config/vllm.py`） |
| 21 | CPU 和 GPU 上的 `log_softmax` | GPU | GPU | GPU | 实测：在检查过的值上，CPU 的结果相差约 2.5e-5，所以两端都要在 GPU 上算 |

## 在所测 GPU 上达到 L4 的舍入序列

引擎端使用 `VLLM_BATCH_INVARIANT=1`、`enforce_eager=True` 和
`attention_backend="FLASH_ATTN"`，关闭 prefix caching，并让 lm_head 走 vLLM 自己的
`linear_batch_invariant`。这一步是对 `UnquantizedEmbeddingMethod.apply` 打的运行时补丁，
由 `engine_run.py --patch-lmhead` 打上。

trainer 端使用相同的权重，并做下面这些选择（`trainer.forward_matched` 配合
`TARGETS["bi_eager_lmhead"]`）：

- 把 Q/K/V 和 gate/up 分别打包成一个 GEMM；
- 每个 linear 层都用 `matmul_persistent`；
- 在 bf16 转换之后再加 bias；
- 在 bf16 的残差和上做 vLLM 的 Triton RMSNorm；
- 使用 vLLM 的 `cos_sin_cache`；
- RoPE 和 SiLU 用哪种写法都可以，两种写法逐 bit 一致；
- FA2 跑在 paged KV 上，`num_splits=1`；
- lm_head 用 `matmul_persistent`；
- 在 fp32 logits 上运行 vLLM 的 Triton `log_softmax`。

从这个状态出发做一轮 leave-one-out：每次只把一项选择改回去，统计 76,155 个 token 里
有多少个不同：

- GEMM 改回分开调用的 cuBLAS：76,155；
- norm 改回去：76,155；
- 注意力改回去：75,585；
- RoPE 表改回去：9,470（都在长于 459 个 token 的 prompt 上）；
- log_softmax 改回去：42,515；
- lm_head 改回 cuBLAS：1,510；
- RoPE 公式或 SiLU 改回去：0。

## 新模型或新引擎的检查清单

换了新模型或新引擎时，逐项检查：

- 投影是怎样打包的（QKV、gate/up、融合的 expert），每个 bias 在哪里加。
- 残差相加时，和是在进入 norm 之前就舍入，还是以 fp32 传给 norm？融合 kernel 和编译后的
  `forward_native` 路径在这一点上做法不同。
- Norm 的归约顺序；用 `sum/n` 还是 `sum * (1/n)`，用 `rsqrt` 还是 `1/sqrt`；
  权重在类型转换之前还是之后乘上。
- RoPE 表在什么设备上、用什么 dtype 构建，公式是什么，旋转是交错的还是对半拆分的。
- 注意力用哪个 kernel，split-KV 怎么设置，paged 布局是什么样，decode 和 prefill
  是否走不同的路径，以及有没有 cascade attention。
- 激活函数里，`silu(g)` 在乘法之前有没有先舍入。
- lm_head 在你的 GPU 上实际跑的是哪个 kernel，M 是多少。
- 算 logprob 时 logits 的 dtype、log_softmax kernel、计算设备、logprobs 模式
  （raw 还是 processed），以及 temperature 是在之前还是之后应用。
- 走编译路径时，编译器融合了哪些算子，有没有保留中间的类型转换。
