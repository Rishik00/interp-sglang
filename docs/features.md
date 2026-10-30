# Features of Mini-SGLang

## Prefill activation capture

The offline Python interface can capture selected transformer signals and the
normalized input to the LM head:

```python
from minisgl.capture import ActivationCaptureConfig
from minisgl.llm import LLM

llm = LLM(
    model_path="Qwen/Qwen3-0.6B",
    activation_capture=ActivationCaptureConfig(
        layer_ids=(1, 6, 13, 20, 27),
        capture_lm_head_input=True,
    ),
)
# After llm.generate(...):
captures = llm.take_activation_captures()
head_inputs = [item for item in captures if item.signal == "lm_head_input"]
```

`lm_head_input` is opt-in (default `False`) and records every newly processed
prefill token after final normalization, before the LM head selects positions
for projection. Its tensor has shape `[tokens, hidden_size]`; it contains no
vocabulary logits. These model-level records have `layer_id=None`, independent
of `layer_ids`. Set `layer_ids=()` to capture only the head input. The hook is in
the shared `ParallelLMHead`, so it applies to every model using that head.

Like the block signals, each record includes token IDs and request/token spans.
Chunked prefill returns separate records for each chunk. Captures are staged on
GPU until drained to CPU, so drain regularly to release staging memory. Decode
is skipped, and cached prefix tokens are not recomputed or captured. Existing
block capture counts and benchmark defaults are unchanged unless the new signal
is explicitly enabled.

## Online Serving

Mini-SGLang supports online serving with an OpenAI-compatible API server. It provides the standard `/v1/chat/completions` endpoint, allowing seamless integration with existing tools and clients. For detailed command-line arguments and configuration options, run `python -m minisgl --help`.

## Interactive Shell Mode

For demonstration and testing purposes, an interactive shell mode is available. In this mode, users can input prompts directly, and the LLM will generate responses in real-time. The shell automatically caches chat history to maintain context. To clear the conversation history and start a new session, use the `/reset` command.

Example:

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## Distributed Serving

To scale performance across multiple GPUs, Mini-SGLang supports Tensor Parallelism (TP). You can enable distributed serving by specifying the number of GPUs with the `--tp n` argument, where `n` is the degree of parallelism.

## Supported Models

Our framework currently supports the following dense model architectures:

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) series
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) series (including MoE)
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) series

## Chunked Prefill

Chunked Prefill, a technique introduced by [Sarathi-Serve](https://arxiv.org/abs/2403.02310), is enabled by default. This feature splits long prompts into smaller chunks during the prefill phase, significantly reducing peak memory usage and preventing Out-Of-Memory (OOM) errors in long-context serving. The chunk size can be configured using `--max-prefill-length n`. Note that setting `n` to a very small value (e.g., 128) is not recommended as it may significantly degrade performance.

## Page Size

You can specify the page size of the system using the `--page-size` argument.

## Attention Backends

Mini-SGLang integrates high-performance attention kernels, including [`FlashAttention`](https://github.com/Dao-AILab/flash-attention) (`fa`), [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer) (`fi`) and [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM) (`trtllm`). It supports using different backends for the prefill and decode phases to maximize efficiency. For example, on NVIDIA Hopper GPUs, `FlashAttention 3` is used for prefill and `FlashInfer` for decode by default.

You can specify the backend using the `--attn` argument. If two values are provided (e.g., `--attn fa,fi`), the first specifies the prefill backend and the second the decode backend. Note that some attention backend might override the user-provided page size (e.g. `trtllm` only supports page size 16,32,64).

## CUDA Graph

To minimize CPU launch overhead during decoding, Mini-SGLang supports capturing and replaying CUDA graphs. This feature is enabled by default. The maximum batch size for CUDA graph capture can be set with `--cuda-graph-max-bs n`. Setting `n` to `0` disables this feature.

## Radix Cache

Adopting the original design from [SGLang](https://github.com/sgl-project/sglang.git), Mini-SGLang implements a Radix Cache to manage the Key-Value (KV) cache. This allows the reuse of KV cache for shared prefixes across requests, reducing redundant computation. This feature is enabled by default but can be switched to a naive cache management strategy using `--cache naive`.

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Illustration of Radix Attention from [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/).*

## Overlap Scheduling

To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in [NanoFlow](https://arxiv.org/abs/2408.12757). This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Illustration of Overlap Scheduling from [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).*
