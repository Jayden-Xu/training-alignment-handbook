# 舍入点：trainer（model.py）与 vLLM

在 Llama/Qwen2 风格的 decoder 中，trainer（训练端）和引擎可能以不同方式舍入的每个位置，
以及每一侧的做法。“Trainer”指
[examples/whole_model/model.py](../../../examples/whole_model/model.py)
（transformers 的计算序列）。“vLLM”指在一种 GPU 型号上以 bf16 服务
Qwen2.5-0.5B-Instruct 的 vLLM 0.19.1，torch 2.10.0+cu128（硬件和结果见
[examples/cross_engine/REPORT.md](../../../examples/cross_engine/REPORT.md)）。

证据列：
- **src** —— 读自 vLLM 0.19.1 的 Python 源码（注明文件）。
- **replay** —— 通过算子重放测得：把引擎捕获的输入送入 trainer 的候选实现，
  逐位比较输出
  （[cross_engine.py](../../../examples/cross_engine/cross_engine.py) `op_replay`；
  一个 50 token 的提示，prefill 和两步 decode，24 层）。
- **tokens** —— 在实际 token 的 logprob（对数概率）上测得（64 个提示，批大小 1–64）。

vLLM 自定义算子（`torch.ops._C.*`）和 FlashAttention 的 CUDA 源码不在 wheel 中；
下面关于它们的行为是测得的，不是读出来的。

## 逐算子

| # | 舍入点 | Trainer（model.py，原生） | vLLM eager（自定义算子） | vLLM batch-invariant eager（`VLLM_BATCH_INVARIANT=1`，在所测 GPU 上） | 证据 |
|---|---|---|---|---|---|
| 1 | Embedding | gather，精确 | gather，精确 | gather，精确 | src |
| 2 | norm 之前的残差相加 | `h = dt(h + o)`，然后 norm 读取 `h` | 融合进 `fused_add_rms_norm`（原地） | 在 bf16 中做 `x + residual`（`layernorm.py fused_add_rms_norm`），然后 norm | src；replay：三者都是舍入后的和 |
| 3 | RMSNorm 方差 | `mean(fp32(x)^2)`（torch 归约），`* rsqrt(var + eps)` | CUDA kernel；141 次调用中有 137 次等于 trainer 的公式，所以归约顺序不同 | Triton `_rms_norm_kernel`：在 1024 宽的块上 `tl.sum`，`sum / n`，`1 / sqrt(mean + eps)` | src（BI）；replay |
| 4 | Norm 权重 | 转为 bf16，然后在 bf16 中 `* w` | 与 trainer 相同（replay 相等） | 在 fp32 中 `(x * inv) * w`，一次转换 | src（BI）；replay：74% 的元素等于 trainer 的结果 |
| 5 | Q/K/V 打包 | 三个 GEMM（N = 896, 128, 128） | 一个 GEMM，N = 1152 | 一个 GEMM，N = 1152 | src（`qwen2.py` QKVParallelLinear）；replay：分开的 cuBLAS GEMM 在 72 次调用中有 30 次给出不同的比特 |
| 6 | Q/K/V bias | 在 GEMM epilogue 中：`dt(sum + b)`（cuBLAS addmm） | 相同（`F.linear(x, W, b)`） | 在转换之后：`dt(dt(sum) + b)`（`linear_batch_invariant`） | src；对应的选择 replay 72/72，另一种 0/72 |
| 7 | GEMM kernel（q/k/v、o、gate/up、down） | cuBLAS，按形状选择 kernel | cuBLAS，按形状选择 kernel（不是 L2） | `matmul_persistent`（Triton，128x128x64 tile，按 K 顺序 fp32 累加）；M = 50 时 cuBLAS 对 q/k/v 和 o 给出相同的比特，对 gate/up（25/72）和 down（24/72）则不同 | src；replay |
| 8 | Gate/up 打包 | 两个 GEMM | 一个 GEMM，N = 9728 | 一个 GEMM，N = 9728 | src；replay：分开的 cuBLAS 48/72，打包的 72/72（eager） |
| 9 | RoPE 表 | transformers：`inv_freq` 在 CPU 上，角度和 cos/sin 在 GPU 上以 fp32 计算，转为 bf16 | `cos_sin_cache` 在 GPU 上以 fp32 构建（`base ** (arange/D)` 在设备上），转为 bf16 | 与 eager 相同 | src（`rotary_embedding/base.py`）；测得：262,144 个条目中有 26 个不同，第一个在位置 459 |
| 10 | RoPE 公式 | `dt(dt(q cos) + dt(rot(q) sin))` | `_C.rotary_embedding`，原地 | 与 eager 相同 | replay：与 trainer 的公式逐位相等（50 个 token 时，用任一张表都是 72/72） |
| 11 | 注意力 | 在扩展到 query head 数的 K/V 上做 SDPA | FlashAttention 2（vllm_flash_attn），分页 KV（block 16），`num_splits=0`（启发式） | 相同，`num_splits=1` | src（`flash_attn.py`）；replay：SDPA 在 24 层中有 11–12 层相等；从 trainer 调用 FA2 为 24/24 |
| 12 | 注意力，decode 与 prefill | trainer 只有 prefill | 依据 cache 的 decode 步 | 相同 | replay：FA2 decode 输出等于 trainer 中对整个序列做的 FA2 prefill，48/48 次调用（num_splits 0 和 1） |
| 13 | SiLU 和乘法 | 在 bf16 中 `dt(silu(g)) * u` | `_C.silu_and_mul` | 相同 | replay：逐位相等（72/72） |
| 14 | 最终 norm | 同 2–4 | 融合的相加 + norm | 在 bf16 和上做 Triton norm | src；replay |
| 15 | lm_head | 在全部 T 个位置上用 cuBLAS | cuBLAS（`F.linear`），只在它采样的行上（decode 时 M = 批大小） | **cuBLAS**：该模式只在列出的一组计算能力上覆盖 `aten::mm/addmm/linear`（`batch_invariant.py enable_batch_invariant_mode`），所测 GPU 不在其中，而 lm_head 走 `F.linear`；只固定了 cuBLAS workspace | src；tokens：破坏引擎 L2（15,231 个 logprob 中有 1,116 个，单独与批量相比） |
| 16 | 送入采样器的 logits dtype | 把 bf16 logits 转为 fp32 后做 fp32 `log_softmax` | 在 GPU 上 `logits.log_softmax(-1, dtype=float32)` | `aten::_log_softmax` 被一个 Triton kernel 覆盖：每个 1024 块取 max，逐块 `sum(tl.exp(x - max))`，`x - max - log(sum)` | src（`sampler.py compute_logprobs`、`batch_invariant.py`）；测得：在 eager 模式下，引擎 logprob 与 GPU `log_softmax(fp32(logits))` 逐位相等 |
| 17 | log_softmax kernel | torch | torch | Triton（见上） | tokens：撤回它会改变 56% 的 logprob，最多 1.9e-6 |
| 18 | Prompt logprobs | 同一个前向 | 在提示行上单独调用 `compute_logits`（M = chunk） | 相同 | src |
| 19 | 编译模式（默认） | eager | 关闭自定义算子；RMSNorm、RoPE、SiLU 通过 Inductor 运行 `forward_native`，被融合；CUDA graph | 相同，配 Triton GEMM | src（`config/vllm.py`：Inductor 下 custom_ops 为 "none"）；tokens：尝试过的 trainer 选择都无法匹配 |
| 20 | Cascade attention | 无 | 共享前缀时可能使用 | 被该模式禁用 | src（`config/vllm.py`） |
| 21 | CPU 与 GPU 的 `log_softmax` | GPU | GPU | GPU | 测得：在检查的值上 CPU 结果相差约 2.5e-5；两侧都在 GPU 上计算 |

## 在所测 GPU 上得到 L4 的序列

引擎：`VLLM_BATCH_INVARIANT=1`、`enforce_eager=True`、
`attention_backend="FLASH_ATTN"`、关闭前缀缓存，并让 lm_head 走 vLLM 自己的
`linear_batch_invariant`（对 `UnquantizedEmbeddingMethod.apply` 的运行时补丁；
`engine_run.py --patch-lmhead`）。

Trainer：相同的权重，做以下选择（`trainer.forward_matched`，
`TARGETS["bi_eager_lmhead"]`）：
打包的 Q/K/V 和 gate/up；每个 linear 都用 `matmul_persistent`；bias 在 bf16 转换之后；
在 bf16 残差和上用 vLLM 的 Triton RMSNorm；vLLM 的 `cos_sin_cache`；
RoPE 和 SiLU 用任一形式（逐位相等）；FA2 分页，`num_splits=1`；
lm_head 用 `matmul_persistent`；在 fp32 logits 上用 vLLM 的 Triton `log_softmax`。

从这个状态做 leave-one-out（76,155 个 token 中不同的数量）：
GEMM 改回分开的 cuBLAS 76,155；norm 改回 76,155；注意力改回
75,585；RoPE 表改回 9,470（长于 459 个 token 的提示）；log_softmax
改回 42,515；lm_head 改回 cuBLAS 1,510；RoPE 公式或 SiLU 改回 0。

## 新模型或新引擎的检查清单

- 投影的打包（QKV、gate/up、融合的专家）以及每个 bias 在哪里加入。
- 残差相加：在 norm 之前舍入，还是以 fp32 进入 norm（融合 kernel
  和编译后的 `forward_native` 路径不同）。
- Norm：归约顺序，`sum/n` 还是 `sum * (1/n)`，`rsqrt` 还是 `1/sqrt`，权重在转换之前还是之后。
- RoPE：表的设备和 dtype、公式、交错还是对半拆分。
- 注意力：kernel、split-KV 设置、分页布局、decode 与 prefill 路径、
  cascade attention。
- 激活：`silu(g)` 在乘法之前是否舍入。
- lm_head：在你的 GPU 上实际运行的是哪个 kernel，以及 M 是多少。
- Logprob：logits 的 dtype、log_softmax kernel、设备、logprobs 模式（raw
  或 processed），temperature 在之前还是之后应用。
- 编译路径：编译器融合了哪些算子，以及它是否保留中间的类型转换。
