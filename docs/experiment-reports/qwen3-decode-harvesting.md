# Qwen3-1.7B decode harvesting prototype

The prototype captures activations directly during CUDA graph decode replay. It
uses fixed graph output buffers, copies each real request row into a bounded GPU
window, then offloads and losslessly compresses each sealed window. It does not
rerun the generated sequence through prefill to obtain the harvested values.

The experiment uses a Tesla T4, FP16, and 20 pinned WildChat prompts, with five
prompts from each of four input-length ranges. Inputs contain 26–2,956 tokens.
Generation is greedy, with at most four concurrent requests and 100 generated
tokens per request. All 28 layers' attention outputs and post-block residuals,
plus the final LM-head input, are captured: 57 vectors per generated token.
CUDA graphs cover batch sizes 1, 2 and 4. Overlap scheduling is disabled, the KV
cache is naive, and prompt prefill is chunked into at most 256 tokens.

Each generated token is identified by its request, token ID, absolute position,
and generation index. The final sampled token gets one extra decode forward
without sampling another token, before the request's KV state is freed. This
includes EOS when it occurs. All 20 requests in this experiment reached the
100-token limit, so this run does not establish early-EOS behavior.

The GPU window has room for 64 token rows and flushes after 16 decode batches or
when full. Graph outputs and the window together reserve 15,876,096 bytes
(15.14 MiB), below the configured 128 MiB capture ceiling. Allocation happens
before KV cache sizing. CPU offload, gzip compression and file writes are
synchronous; a slow sink stalls generation rather than accumulating tensors.

An initial run exposed a CUDA stream-ordering bug in the final partial-window
flush. Generation equality and token metadata passed, but all 57 comparisons for
each of the last four requests failed: 912 of 1,140 comparisons passed. The CPU
copy ran after leaving the engine stream and could race pending GPU writes.
The final flush and shutdown flush now run on the engine stream. Numerical
thresholds were kept unchanged.

The corrected run passed all **1,140 of 1,140 activation comparisons**: one
comparison per signal per request, covering all 100 generated positions. Every
generated token ID matched the uninstrumented baseline. All 2,000 rows were
captured, including 20 terminal rows, across 32 windows and 500 CUDA graph
forwards, with no eager decode forwards. Token positions and checksums passed,
and the KV cache integrity check passed after generation.

| Measurement | Result |
|---|---:|
| Baseline end-to-end generation time | 13.571 seconds |
| Capture, offload, compression and write time | 33.997 seconds |
| Baseline sampled tokens per second | 147.38 |
| Capture sampled tokens per second | 58.83 |
| Baseline throughput retained | 39.92% |
| Serialization, compression and file writing alone | 20.212 seconds |
| Raw tensor payload | 445.3125 MiB |
| Gzip tensor payload | 413.6660 MiB |
| Payload size saved by gzip | 7.11% |
| Worst relative activation L2 error | 0.0248884 |
| Lowest whole-signal cosine | 0.9996906 |
| Lowest individual-token cosine | 0.9863350 |

These timings include prompt prefill and final flushing, and exclude model
loading, graph construction, warmup and reference validation. They come from
one paired run, not a repeated performance study. The lossless compression
preserves the captured FP16 bytes exactly, but is expensive and gives only modest
size savings. This implementation does **not** meet the 80–85% throughput target.

The reference teacher-forces the exact prompt and generated IDs using
Transformers 4.57.3 eager attention, while the engine uses FlashInfer and
PyTorch 2.8.0 with CUDA 12.8. Thresholds are relative L2 at most 0.05,
whole-signal cosine at least 0.999, and individual-token cosine at least 0.98.
Reference next-token argmax agreed at 1,997 of 2,000 positions; this is a separate
metric from the exact equality between capture-on and capture-off generation.

The validated runtime commit is `6131942afcae31e5aaa73593ddabea743e107c6c`.
The [corrected Modal run](https://modal.com/apps/a-nerdy-penguin/main/ap-NaydSOLT0HLjtoG6jQSHce)
contains the execution logs. Compressed windows and their index persist in the
shared Modal cache volume. The benchmark saves the prompt manifest, token IDs,
per-signal comparison results, capture metadata and storage location locally.

Local validation passed all 20 capture and utility tests, including memory-budget
checks, buffer reuse, padded-row exclusion, sink failures and bit-exact gzip
round-trips. Ruff and whitespace checks passed. This is a single-GPU prototype;
asynchronous transfers, background compression, CDN uploads, gradients and paper
reproductions are outside this implementation.
