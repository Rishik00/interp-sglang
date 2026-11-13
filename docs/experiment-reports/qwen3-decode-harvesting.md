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

## Per-request timing rerun

A subsequent run measured CPU-observed token receipt times for every request. All requests were submitted together. Time to first token includes queueing and prefill. Decode speed counts the 99 tokens after the first token and divides by the elapsed time between the first and last token callbacks. These are per-request speeds while sharing the GPU with up to three other requests, not isolated single-request speeds. They include scheduling pauses and synchronous capture writes during generation. The final job flush after the last token callback is excluded from token latency.

| Measurement across 20 requests | Baseline | Capture with gzip |
|---|---:|---:|
| Minimum decode tokens per second | 36.171 | 14.414 |
| Median decode tokens per second | 48.644 | 16.501 |
| Maximum decode tokens per second | 53.286 | 17.219 |
| Median inter-token interval, milliseconds | 20.562 | 60.602 |
| Median time to first token, seconds | 6.296 | 14.056 |

| Request | Input tokens | Baseline first token, seconds | Baseline decode tokens/sec | Capture first token, seconds | Capture decode tokens/sec |
|---|---:|---:|---:|---:|---:|
| 0 | 1835 | 0.331 | 46.621 | 0.331 | 16.501 |
| 1 | 51 | 0.331 | 46.621 | 0.331 | 16.501 |
| 2 | 26 | 0.331 | 46.621 | 0.331 | 16.501 |
| 3 | 1641 | 0.585 | 52.955 | 0.582 | 17.219 |
| 4 | 1396 | 2.724 | 37.574 | 6.591 | 15.155 |
| 5 | 467 | 2.803 | 38.736 | 6.670 | 15.341 |
| 6 | 2746 | 3.293 | 47.922 | 7.153 | 16.580 |
| 7 | 879 | 3.402 | 50.583 | 7.268 | 16.908 |
| 8 | 2281 | 5.761 | 36.171 | 13.522 | 14.925 |
| 9 | 1845 | 6.100 | 41.284 | 13.858 | 15.722 |
| 10 | 2435 | 6.492 | 49.366 | 14.255 | 16.780 |
| 11 | 27 | 6.492 | 49.366 | 14.255 | 16.780 |
| 12 | 775 | 8.674 | 44.014 | 20.330 | 14.414 |
| 13 | 2252 | 9.027 | 52.207 | 20.681 | 15.191 |
| 14 | 33 | 9.027 | 52.207 | 20.681 | 15.191 |
| 15 | 267 | 9.065 | 53.286 | 20.720 | 15.283 |
| 16 | 1110 | 11.137 | 40.304 | 27.395 | 15.528 |
| 17 | 2956 | 11.629 | 50.412 | 27.885 | 16.820 |
| 18 | 28 | 11.629 | 50.412 | 27.885 | 16.820 |
| 19 | 291 | 11.690 | 52.029 | 27.946 | 16.996 |

All 20 requests again produced 100 tokens. Capture-on and capture-off output IDs matched exactly, all 1,140 activation comparisons passed, and all 21 local tests passed. This remains one paired measurement, with no repeat-run confidence intervals.

Runtime commit: `3632402daf63416f848e20ff7b019309ff4823f0`. [Per-request timing run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-th68IlbF0hfaGNN8mD9sEh).

## GPU-buffer-only isolation

To test whether GPU capture itself caused the slowdown, a fresh paired run kept
the CUDA graph capture copies, GPU staging copies, token metadata, window
rotation and final-token forward, while removing CPU activation transfers,
compression and activation file writes from the timed region. The same 20
WildChat prompts, four-request concurrency limit, 100-token generation limit,
T4, FP16 and 57 captured vectors per token were used.

| Measurement | Fresh baseline | GPU buffers only |
|---|---:|---:|
| Median per-request decode tokens per second | 49.156 | 48.713 |
| Minimum per-request decode tokens per second | 36.665 | 36.302 |
| Maximum per-request decode tokens per second | 53.769 | 53.307 |
| Median inter-token interval, milliseconds | 20.348 | 20.533 |
| Entire generation job, seconds | 13.479 | 13.639 |
| Aggregate sampled tokens per second | 148.377 | 146.644 |

GPU-only capture retained **99.10% of median per-request decode speed** and
**98.83% of aggregate job throughput**. Removing the downstream offload and
storage work brought performance close to baseline. This points to that
downstream pipeline as the source of most of the earlier regression, rather
than the GPU buffer copies. The experiment does not separate CPU transfer,
compression and file-writing costs from each other. It is one paired run, so
the roughly one-percent difference is not a precise or statistically established
estimate of GPU capture overhead.

All 2,000 generated token IDs matched the uninstrumented baseline. The engine
captured 2,000 rows, including 20 terminal rows, through 500 CUDA graph forwards,
with no eager decode forwards. It recycled 32 windows and kept the same
15.14 MiB capture allocation. The KV cache integrity check passed.

No activation payload was transferred to CPU or saved during timed generation.
Earlier windows were discarded as they were recycled. Only the final borrowed
GPU window was retained, synchronized, and copied to CPU after timing ended.
All **228 of 228 comparisons** on its **16 token rows across four requests**
passed the unchanged Transformers thresholds, including token ID, position and
terminal-token checks. This is a final-window spot-check, not full-harvest
activation validation. The benchmark also passed all 22 local capture and
utility tests, including a new storage-reuse test for the borrowed window.

Runtime commit: `44bfe0e593df35fa10387d6a0848036a83f44814`.
[GPU-only isolation run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-3QlgljB0KcFAQdEYiMGqAy).

## CPU offload without serialization or storage

The next paired T4 run retained the blocking GPU-to-CPU copies but disabled
activation byte conversion, gzip, SHA-256 checksums and payload file writes.
Each CPU tensor was discarded after consumption, except for the final window
retained for untimed validation. The same prompts, concurrency, capture signals,
window size and generation limit were used.

| Measurement | Fresh baseline | Capture and CPU offload only |
|---|---:|---:|
| Median per-request decode tokens per second | 48.623 | 48.410 |
| Minimum per-request decode tokens per second | 36.362 | 36.093 |
| Maximum per-request decode tokens per second | 53.098 | 52.684 |
| Median inter-token interval, milliseconds | 20.571 | 20.663 |
| Entire generation job, seconds | 13.539 | 13.729 |
| Aggregate sampled tokens per second | 147.725 | 145.681 |

Capture with blocking CPU offload retained **99.56% of median per-request decode
speed** and **98.62% of aggregate job throughput**. This transferred exactly
466,944,000 bytes (445.3125 MiB) across 32 windows during timed generation. It
used 500 CUDA graph forwards, captured all 2,000 rows including 20 terminal rows,
and kept the same 15.14 MiB GPU capture allocation. No activation payload was
saved. Metrics were written locally after timing.

The large earlier regression is therefore associated with the subsequent
serialization, compression and payload-writing work as a group. Blocking CPU
offload alone stayed close to baseline in this configuration. These separate
paired runs do not isolate the individual contributions of byte conversion,
compression, hashing or file writing, and the small differences near baseline
are not precise overhead estimates from repeated trials.

All 2,000 generated token IDs matched baseline, and KV cache integrity passed.
All 228 comparisons for the final 16 token rows across four requests passed
the unchanged reference thresholds. Earlier CPU windows were discarded, so
activation parity covers the final window only. All 23 local tests passed,
including a check that offloaded CPU tensors remain valid when GPU staging
storage is reused.

Runtime commit: `f33e70293f3b6283bf22afd86f664d68745d0f13`.
[CPU-offload-only run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-fb9kaQ1bLikurMPEvvhV2W).

## Scale-up to 50 prompts and 250 tokens

Both benchmark scripts now default to 50 WildChat prompts and a 250-token
generation cap. The sampler supports counts that are not divisible by four:
50 prompts are distributed as 13, 13, 12 and 12 across the existing input-length
buckets. Sampling remains deterministic and does not truncate prompts. The
selected inputs in this run contained 16–3,025 tokens.

The paired T4 run used CPU offload without byte conversion, compression,
checksums or activation payload writes. It produced 11,948 sampled tokens:
46 requests reached the 250-token cap and four stopped at EOS. Capture-on and
capture-off generated exactly the same token IDs, including EOS.

| Measurement | Fresh baseline | Capture and CPU offload only |
|---|---:|---:|
| Median per-request decode tokens per second | 47.587 | 43.749 |
| Minimum per-request decode tokens per second | 38.625 | 34.727 |
| Maximum per-request decode tokens per second | 57.848 | 53.161 |
| Median inter-token interval, milliseconds | 21.015 | 22.860 |
| Entire generation job, seconds | 68.769 | 76.057 |
| Aggregate sampled tokens per second | 173.741 | 157.092 |

CPU-offload capture retained **91.93% of median per-request decode speed** and
**90.42% of aggregate throughput** in this pair. This difference is larger than
in the 20-prompt, 100-token test. These are single pairs in separate fresh T4
containers; they do not establish a precise overhead estimate or isolate
container performance variation from capture and offload costs.

The engine captured and offloaded all 11,948 token rows, including 50 terminal
rows, transferring 2,789,523,456 bytes (2.60 GiB) across 196 windows. There were
3,121 CUDA graph forwards and no eager decode forwards. GPU capture storage
remained 15.14 MiB with a 64-row capacity. KV cache integrity passed, including
requests that ended early at EOS.

Earlier CPU windows were discarded. The final retained window contained only
one token row, so reference validation in this run is limited to that token:
all 57 signal comparisons passed, as did its token ID, position and terminal
metadata checks. This is not full activation validation of the 50-prompt
workload. All 24 local capture and utility tests passed, including deterministic
balanced sampling of exactly 50 prompts.

Raw byte conversion is only needed by the current gzip and payload-writing
implementation. It preserves FP16 values exactly but creates an additional
CPU byte copy; the offload-only path does not need or perform that conversion.

Runtime commit: `b0db3f995416a09900fe02b0197f13b642ef56c6`.
[50-prompt, 250-token run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-276umZ4KE99GEdvZAU9WGp).

## Compression deferred

The active decode prototype now defaults to GPU-to-CPU offload only. The gzip
writer, compressed activation readers, raw byte conversion, activation checksums
and compression CLI mode have been removed. The GPU-only diagnostic remains
available. The benchmark sink keeps the final window for validation and discards
earlier windows; it does not persist activation tensors. Metrics and prompt
manifests are still saved after timing.

The defaults remain 50 prompts and a 250-token cap. All 23 remaining local tests
pass; the removed test covered the deleted gzip round-trip path. CLI help and
lint checks pass. The CPU offload mechanism is unchanged from the preceding
50-prompt T4 run; no additional GPU performance run was performed for this
cleanup. Historical results above remain as recorded.
