# Llama-3.2-1B decode validation

The base Llama-3.2-1B checkpoint successfully ran through the shared decode
capture and GPU-to-CPU offload pipeline. The supplied authorized credential
resolved the earlier Hugging Face access restriction. Local credential files
are excluded from version control and Modal source uploads.

The experiment used a Tesla T4, FP16, 50 pinned WildChat prompts, greedy
generation and a 250-token cap. Prompts contained 2–3,065 input tokens, without
truncation. Length buckets contained 13, 13, 12 and 12 prompts. Tokenization uses
Llama's tokenizer, so the selected examples are not necessarily identical to
the Qwen sample despite using the same dataset revision and random seed.

All 16 layers' attention outputs and post-block residuals, plus the final
LM-head input, were captured: 33 vectors per generated token. Up to four
requests ran concurrently. CUDA graph batch sizes were 1, 2 and 4; overlap
scheduling was disabled. The KV cache was naive, prompt prefill used 256-token
chunks, and the maximum sequence length was 4,096.

Both modes generated exactly the same 6,298 token IDs, including EOS.
Twenty-seven requests stopped at EOS and 23 reached the 250-token cap. Fifteen
requests ended on their first sampled token and therefore had no measurable
decode interval. Per-request decode statistics below cover the remaining
35 requests; time-to-first-token statistics cover all 50.

| Measurement | Baseline | Capture and CPU offload |
|---|---:|---:|
| Median per-request decode tokens per second | 64.445 | 62.514 |
| Median inter-token interval, milliseconds | 15.517 | 15.996 |
| Median time to first token, seconds | 13.394 | 13.853 |
| Entire generation job, seconds | 27.912 | 28.676 |
| Aggregate sampled tokens per second | 225.641 | 219.623 |

Capture retained **97.00% of median per-request decode speed** and **97.33% of
aggregate throughput**. The timings include prompt prefill and final offload,
but exclude model loading, graph construction, warmup and reference validation.
Time to first token includes queueing from submission of the entire job. This
is one paired run in fresh containers, not a repeated performance study.

The engine captured and offloaded all 6,298 generated-token rows, including
50 terminal rows. It transferred 851,288,064 bytes (811.85 MiB) across 111
windows. GPU capture buffers occupied 8.77 MiB with a 64-row staging capacity.
There were 1,773 captured CUDA graph forwards and no eager decode forwards.
KV cache integrity passed after generation.

No byte serialization, compression, activation checksums or activation payload
writes were performed. Earlier CPU windows were discarded. The final retained
window contained 13 token rows from one request; all **33 of 33 signal
comparisons** passed against Transformers eager attention, including the
terminal token. Token IDs, positions and terminal metadata matched. The same
reference forward also matched all 250 next-token argmax predictions for that
request's teacher-forced sequence.

The unchanged thresholds were relative L2 at most 0.05, whole-signal cosine at
least 0.999, and individual-token cosine at least 0.98. Activation parity in this
run covers the final retained window only, not every token across all 50
requests. This validates this Llama checkpoint and configuration, rather than
every model in the Llama family.

Runtime commit: `600352e15f292afc029dc8654d758c23717062df`.
[Llama T4 run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-0eyg6qR5BAFRImJ4qtr2lK).
