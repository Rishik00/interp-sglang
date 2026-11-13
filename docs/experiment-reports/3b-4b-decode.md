# Qwen3-4B and Llama-3.2-3B decode capture

Both larger checkpoints completed baseline and CPU-offload capture runs on T4.
Every generated token ID matched between capture on and off, including EOS.
The retained-window activation checks passed for both models. Compression,
activation checksums, byte serialization and activation payload writes remained
disabled.

Each model used 50 length-stratified WildChat prompts, a fixed sampling seed,
greedy generation and a 250-token cap. Tokenization was model-specific, so the
selected examples differ across model families. Input lengths were 16–3,025
tokens for Qwen and 2–3,065 for Llama. These are within-model capture comparisons,
not a controlled comparison of the two model families' generation speeds.

The configuration was FP16, up to four concurrent requests, naive KV caching,
256-token prefill chunks and a 4,096-token sequence limit. CUDA graphs used
batch sizes 1, 2 and 4, with overlap scheduling disabled. All layers' attention
outputs and post-block residuals, plus the final LM-head input, were captured
during decode. Prompt prefill activations were not captured.

The serving memory ratio was raised to 0.75 for both models. After timed
generation, the runner retains the final CPU window, shuts down and releases the
serving engine, then loads the Transformers reference. Engine shutdown now
releases its global context, allowing KV cache and model allocations to be
collected. Before reference loading, free GPU memory was 13.68 GiB for Qwen and
13.67 GiB for Llama. This avoids holding two FP16 model copies alongside the KV
cache. Teardown and reference computation are outside generation timing.

| Model | Qwen3-4B | Llama-3.2-3B |
|---|---:|---:|
| Captured layers | 36 | 28 |
| Vectors per generated token | 73 | 57 |
| Hidden width | 2,560 | 3,072 |
| Generated tokens in each mode | 11,895 | 6,234 |
| Requests reaching token cap | 46 | 23 |
| Requests stopping at EOS | 4 | 27 |
| Requests with a measurable decode interval | 50 | 34 |

Sixteen Llama requests ended on their first token and therefore have no decode
interval. Median decode rates exclude these requests. Time to first token
includes queueing and prefill; decode rates use the interval from first to last
token receipt and exclude the first token from the token count.

| Measurement | Qwen baseline | Qwen capture | Llama baseline | Llama capture |
|---|---:|---:|---:|---:|
| Median per-request decode tokens/sec | 19.491 | 21.557 | 25.998 | 25.001 |
| Median inter-token interval, milliseconds | 51.305 | 46.388 | 38.474 | 40.003 |
| Median time to first token, seconds | 81.423 | 73.186 | 34.229 | 34.644 |
| Entire generation job, seconds | 170.189 | 153.433 | 69.731 | 70.939 |
| Aggregate sampled tokens/sec | 69.893 | 77.526 | 89.401 | 87.879 |

Llama retained 96.17% of median per-request decode speed and 98.30% of aggregate
throughput. Qwen's capture run measured 110.60% of baseline median decode speed
and 110.92% of baseline aggregate throughput. The faster Qwen capture run does
not establish a performance benefit from capture: these are single pairs in
fresh containers, and the experiment does not separate run-to-run variation
from capture overhead. No precise overhead estimate is claimed.

| Capture measurement | Qwen3-4B | Llama-3.2-3B |
|---|---:|---:|
| GPU buffer allocation, MiB | 24.24 | 22.71 |
| Maximum staging rows | 64 | 64 |
| Offloaded tensor bytes | 4,445,875,200 | 2,183,196,672 |
| Offloaded windows | 197 | 108 |
| Captured CUDA graph forwards | 3,138 | 1,728 |
| Terminal token rows | 50 | 50 |
| Eager decode forwards | 0 | 0 |
| Final-window signal comparisons passed | 73/73 | 57/57 |
| Final-window token rows checked | 2 | 16 |

KV cache integrity and final-window token IDs, positions and terminal metadata
passed for both models. Activation thresholds remained relative L2 at most
0.05, whole-signal cosine at least 0.999, and individual-token cosine at least
0.98. Earlier CPU windows were discarded; activation parity is only established
for the final retained windows, not every generated position in all 50 prompts.

All 25 targeted local capture, context-lifecycle and utility tests passed, as did
lint and whitespace checks. The broader local core suite could not collect
because the installed Hugging Face Hub 2.0 conflicts with Transformers' required
version below 1.0. The actual T4 runs used the compatible Modal dependency image.

Runtime commit: `32650737d0b2025c5b3c7a0e8b1f5cd71cd302a2`.
[Qwen3-4B run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-rPWYTJhZVGE0kFIEfk7vxJ),
[Llama-3.2-3B run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-iHndGr1zDQPRK6MPBIOsjj).
