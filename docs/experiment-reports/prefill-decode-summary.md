# All-layer prefill and decode capture on T4

We tested whether all-layer activation harvesting preserves generation and agrees with a Transformers reference, while measuring its timing and memory costs.

All four checkpoints completed the new prefill performance runs. Capture on/off produced identical generated token IDs for every prompt. Full-prompt activation validation is a stricter result: the numerical pass counts and failures below retain the original tolerances. The decode results are the completed windowed CPU-offload runs, consolidated here without rerunning or mixing them into prefill timing.

Only Llama-3.2-1B passed every full-prompt prefill activation comparison. Qwen3-1.7B had 125 failures, Qwen3-4B had 442, and Llama-3.2-3B had 13. Qwen3-4B also failed aggregate similarity thresholds; its discrepancies are not limited to isolated-token cosine. Earlier final-window decode checks passed for all four models, with the limited coverage documented below.

## Workload and configuration

| Setting | Value |
|---|---|
| GPU and arithmetic | One NVIDIA T4; FP16; no model or activation quantization |
| Dataset | WildChat-4.8M, revision c827c6df8fcf008219ffaffa4d1dd77491099367 |
| Sampling | First shard, first 4,096 candidate rows; seed 42; English/Chinese first-user messages; toxic/redacted rows excluded; deduplicated |
| Prompts per model | 50, balanced 13/13/12/12 across 1–256, 257–1,024, 1,025–2,048 and 2,049–3,072 input tokens; no truncation |
| Performance generation | Greedy; up to 250 sampled tokens; EOS respected |
| Captured signals | Attention output after output projection and post-block residual at every layer; final LM-head input once |
| KV cache and sequence limit | Naive cache; 256-token prefill chunks; 4,096-token sequence limit |
| CUDA graphs | Performance batches 1, 2 and 4; prefill validation batch 1; overlap scheduling disabled |
| Prefill performance | One request at a time; capture on/off interleaved; order alternates by prompt |
| Decode performance | Up to four concurrent requests; baseline and capture use fresh containers |
| Prefill reference checks | All 50 prompts plus a mixed two-request batch: 51 cases, 52 request instances; three generated steps for alignment only |
| Reference | Transformers eager attention, FP16; comparison arithmetic in FP64 on CPU |
| Host RAM | 16 GiB for performance; 64 GiB for full-prompt prefill reference validation |
| Activation processing | CPU offload only; no compression, payload checksums, byte serialization or activation-file writes |

Tokenization determines the length strata and selected examples, so the sampled sets can differ across model families. These are within-model capture comparisons. They do not establish which model family is faster on a controlled shared token workload. All-layer capture means these three signals, rather than every internal tensor in the architecture.

| Model | Layers | Hidden width | Vectors per token | Input token range | Serving memory ratio |
|---|---|---|---|---|---|
| Qwen3-1.7B | 28 | 2048 | 57 | 16–3,025 | 0.45 |
| Qwen3-4B | 36 | 2560 | 73 | 16–3,025 | 0.75 |
| Llama-3.2-1B | 16 | 2048 | 33 | 2–3,065 | 0.45 |
| Llama-3.2-3B | 28 | 3072 | 57 | 2–3,065 | 0.75 |

## Prefill performance

Prompt activations stay on GPU during generation. The runner offloads them after generation in a separately timed CPU drain. Time to first token therefore measures the prefill capture path and excludes that drain. Decode harvesting is disabled in this phase; the decode rates in the next table describe generation while only prompt activations are retained.

| Model | Median first token off, ms | On, ms | Median paired first-token change | Median input tokens/s off | On | Median CPU drain on, ms |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 139.676 | 174.290 | 12.20% | 5,831.542 | 5,062.694 | 166.438 |
| Qwen3-4B | 458.115 | 460.003 | 0.44% | 1,794.126 | 1,791.421 | 106.688 |
| Llama-3.2-1B | 92.416 | 107.803 | 3.02% | 8,829.366 | 8,158.021 | 45.798 |
| Llama-3.2-3B | 272.476 | 274.845 | 1.02% | 3,182.079 | 3,152.121 | 107.429 |

The paired percentage is the median of each prompt's percentage change. It is not the percentage change between the two displayed medians. Input tokens/s is input length divided by first-token time; it is not generated-token throughput.

| Model | Capture | Sampled tokens | EOS requests / 50 | Requests with decode interval | Median decode tokens/s | Median generation, s | Median input + output tokens/s including drain |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | Off | 11947 | 4 | 50 | 59.359 | 4.347 | 275.967 |
| Qwen3-1.7B | On | 11947 | 4 | 50 | 59.370 | 4.375 | 258.989 |
| Qwen3-4B | Off | 11802 | 4 | 50 | 17.935 | 14.201 | 82.273 |
| Qwen3-4B | On | 11802 | 4 | 50 | 17.922 | 14.264 | 82.659 |
| Llama-3.2-1B | Off | 6298 | 27 | 35 | 87.926 | 1.366 | 937.070 |
| Llama-3.2-1B | On | 6298 | 27 | 35 | 88.164 | 1.369 | 866.367 |
| Llama-3.2-3B | Off | 6234 | 27 | 34 | 35.751 | 3.416 | 351.647 |
| Llama-3.2-3B | On | 6234 | 27 | 34 | 35.827 | 3.425 | 319.158 |

Sampled token counts include terminal EOS even when the returned text omits it. A one-token generation has no decode interval and is excluded from decode-rate medians. The last column counts input plus output tokens, whereas the windowed decode job throughput below counts generated tokens only. Each configuration has one measured pair per prompt; these results do not quantify repeated-run uncertainty.

Token equality is checked within each phase. Serial prefill and concurrent decode runs are not asserted to generate identical continuations across phases: Qwen3-1.7B generated 11,947 versus 11,948 tokens, and Qwen3-4B generated 11,802 versus 11,895. The cause of these cross-phase differences was not isolated.

| Model | Median prompt payload, MiB | Largest payload, MiB | Median peak allocated off, GiB | On, GiB | Largest peak allocated on, GiB | Largest peak reserved on, GiB | Retained reference captures on CPU, GiB |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 184.14 | 673.54 | 6.591 | 6.766 | 7.246 | 7.303 | 12.173 |
| Qwen3-4B | 294.78 | 1,078.25 | 10.412 | 10.707 | 11.503 | 11.553 | 19.488 |
| Llama-3.2-1B | 116.98 | 395.10 | 6.606 | 6.712 | 6.989 | 7.004 | 6.837 |
| Llama-3.2-3B | 303.09 | 1,023.66 | 10.899 | 11.194 | 11.921 | 11.945 | 17.714 |

Memory columns use PyTorch allocator peaks measured during generation. Driver usage in the detailed CSV is sampled at checkpoints, rather than continuously measured peak usage. Full-prompt validation retains all prompt captures on CPU before reference evaluation; this is an experiment memory requirement, not the bounded decode staging policy. The serving engine and KV cache are released before loading the reference model.

## Full-prompt prefill activation validation

A signal comparison passes only when every value is finite, the tensor is nonzero, relative L2 is at most 0.05, whole-tensor cosine is at least 0.999, and the minimum individual-token cosine is at least 0.98. First-token agreement against the reference is tracked separately. These are activation similarity checks, not task accuracy measurements.

| Model | Activation comparisons passed | Failed | Reference first-token agreement | Worst relative L2 | Lowest whole cosine | Lowest token cosine |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 2,839/2,964 | 125 | 52/52 | 0.024922 | 0.999689 | 0.669684 |
| Qwen3-4B | 3,354/3,796 | 442 | 52/52 | 0.206898 | 0.978570 | -0.202784 |
| Llama-3.2-1B | 1,716/1,716 | 0 | 52/52 | 0.003788 | 0.999993 | 0.999791 |
| Llama-3.2-3B | 2,951/2,964 | 13 | 52/52 | 0.008233 | 0.999966 | 0.926885 |

| Model | Cases with failures / 51 | Attention failures | Residual failures | LM-head failures | Nonfinite | Zero tensors | L2 failures | Whole-cosine failures | Token-cosine failures |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 12 | 76 | 46 | 3 | 0 | 0 | 0 | 0 | 125 |
| Qwen3-4B | 18 | 236 | 194 | 12 | 0 | 0 | 30 | 37 | 442 |
| Llama-3.2-1B | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Llama-3.2-3B | 4 | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 13 |

Threshold failure columns overlap: one comparison can fail multiple criteria. Signal failure counts count each failed comparison once.

A good whole-tensor cosine can hide a poor individual token; the minimum-token check catches that. The failures above are retained without changing thresholds. Their root cause has not been isolated, so the runs with failures do not establish full activation parity. Identical capture-on/off generation and reference first-token agreement do not override failed activation comparisons. The complete per-layer, per-request metrics and failure flags are included in the numeric export.

## Windowed decode capture

These completed runs capture generated-token forwards with CUDA graphs enabled. A GPU window holds up to 64 token rows and flushes after 16 decode batches or when full. One decode batch contributes one row per active request. CPU transfer blocks the serving path in this prototype; older CPU windows are discarded after consumption, and only the final window is retained for reference validation. The terminal generated token gets an extra forward for activation coverage without sampling another output token.

| Model | Capture | Sampled tokens | Requests with decode interval | Median request tokens/s | Median mean inter-token gap, ms | Aggregate generated tokens/s | Whole job, s |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | Off | 11948 | 50 | 47.587 | 21.015 | 173.741 | 68.769 |
| Qwen3-1.7B | On | 11948 | 50 | 43.749 | 22.860 | 157.092 | 76.057 |
| Qwen3-4B | Off | 11895 | 50 | 19.491 | 51.305 | 69.893 | 170.189 |
| Qwen3-4B | On | 11895 | 50 | 21.557 | 46.388 | 77.526 | 153.433 |
| Llama-3.2-1B | Off | 6298 | 35 | 64.445 | 15.517 | 225.641 | 27.912 |
| Llama-3.2-1B | On | 6298 | 35 | 62.514 | 15.996 | 219.623 | 28.676 |
| Llama-3.2-3B | Off | 6234 | 34 | 25.998 | 38.474 | 89.401 | 69.731 |
| Llama-3.2-3B | On | 6234 | 34 | 25.001 | 40.003 | 87.879 | 70.939 |

| Model | Capture | Median first token, s | Median completion, s | Peak allocated, GiB | Peak reserved, GiB |
|---|---|---|---|---|---|
| Qwen3-1.7B | Off | 33.932 | 37.467 | 6.591 | 6.639 |
| Qwen3-1.7B | On | 37.540 | 41.419 | 6.608 | 6.658 |
| Qwen3-4B | Off | 81.423 | 89.675 | 10.412 | 10.973 |
| Qwen3-4B | On | 73.186 | 81.151 | 10.437 | 10.971 |
| Llama-3.2-1B | Off | 13.394 | 16.478 | 6.606 | 6.621 |
| Llama-3.2-1B | On | 13.853 | 17.015 | 6.614 | 6.643 |
| Llama-3.2-3B | Off | 34.229 | 39.480 | 10.899 | 10.953 |
| Llama-3.2-3B | On | 34.644 | 40.024 | 10.878 | 10.953 |

| Model | Median request speed retained | Aggregate throughput retained | EOS requests | Token-cap requests |
|---|---|---|---|---|
| Qwen3-1.7B | 91.93% | 90.42% | 4 | 46 |
| Qwen3-4B | 110.60% | 110.92% | 4 | 46 |
| Llama-3.2-1B | 97.00% | 97.33% | 27 | 23 |
| Llama-3.2-3B | 96.17% | 98.30% | 27 | 23 |

Per-request decode tokens/s is generated tokens after the first token divided by first-to-last token receipt time. It includes scheduling and synchronous offload stalls between tokens. First-token and completion latencies start at whole-job submission and therefore include queueing. The final flush after the last token callback is included in whole-job duration. The faster Qwen3-4B capture run is a single fresh-container comparison; it does not establish a speedup or negative capture overhead.

| Model | GPU staging, MiB | Captured token rows | Offloaded bytes | Windows | Graph forwards | Eager forwards | Terminal rows | Final-window comparisons passed | Final-window token rows checked |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 15.14 | 11948 | 2,789,523,456 | 196 | 3121 | 0 | 50 | 57/57 | 1 |
| Qwen3-4B | 24.24 | 11895 | 4,445,875,200 | 197 | 3138 | 0 | 50 | 73/73 | 2 |
| Llama-3.2-1B | 8.77 | 6298 | 851,288,064 | 111 | 1773 | 0 | 50 | 33/33 | 13 |
| Llama-3.2-3B | 22.71 | 6234 | 2,183,196,672 | 108 | 1728 | 0 | 50 | 57/57 | 16 |

Every generated token ID matched between baseline and capture for all four models. KV integrity, terminal coverage and retained-window token/position alignment passed. Decode numerical checks use the same thresholds as prefill, but cover only the final retained windows: they do not validate every activation in the generation job. GPU staging allocation includes more than the 64-row payload, so it should not be inferred from payload bytes alone.

For each model, the final window contains positions from one request. Teacher-forcing that request's full generated continuation gave 250/250 matching greedy token predictions in the reference. Activation comparisons still cover only the retained positions listed in the table.

## Earlier pipeline ablations

These historical Qwen3-1.7B runs used 20 prompts and a 100-token cap, producing 2,000 sampled tokens each. They are a separate cohort from the 50-prompt results above. Each stage has its own baseline, and all generated tokens matched.

| Pipeline | Median request tokens/s off | On | Request speed retained | Job off, s | On, s | Aggregate throughput retained |
|---|---|---|---|---|---|---|
| Gzip and writes | 48.644 | 16.501 | 33.92% | 13.595 | 33.949 | 40.04% |
| GPU buffers only | 49.156 | 48.713 | 99.10% | 13.479 | 13.639 | 98.83% |
| CPU offload only | 48.623 | 48.410 | 99.56% | 13.539 | 13.729 | 98.62% |

The historical gzip level-1 writer stored 433,760,280 bytes from 466,944,000 raw bytes: 1.0765:1 compression, saving 7.1066%. Its blocking write path took 20.238 seconds. In the discard modes, zero stored bytes means no activation file was written; it is not a compression ratio. Compression, checksums and activation-file writes have since been removed.

## Local numeric exports and run logs

CSV exports are kept locally and ignored by Git. Only the Markdown reports are committed.

| Export | Contents |
|---|---|
| Prefill summary | Eight model/mode rows, timings, payload sizes and memory peaks |
| Prefill per-request measurements | 400 rows, including memory snapshots and dataset row identifiers |
| Paired prefill changes | 200 capture-on/off comparisons |
| Full-prompt activation comparisons | Every layer/signal comparison, unrounded metrics and pass flags |
| Activation validation summary | Eight model/phase rows, extrema, token coverage, thresholds and reference memory |
| Decode summary | Eight model/mode rows, job timings, memory and capture statistics |
| Decode per-request measurements | 400 rows, first-token and completion times, decode rates and mean inter-token gaps |
| Retained-window decode comparisons | 220 signal comparisons with token-row coverage |
| Historical pipeline ablations | Three separate 20-prompt pipeline measurements |
| Run index | Source revisions, raw result identifiers, configuration and Modal logs |

| Model | Prefill performance and validation | Decode |
|---|---|---|
| Qwen3-1.7B | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-Fr6H5yrbTjyWz5kDWAHod3) | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-276umZ4KE99GEdvZAU9WGp) |
| Qwen3-4B | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-vcJlAFU6T1R0O4OPHtJHmI) | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-rPWYTJhZVGE0kFIEfk7vxJ) |
| Llama-3.2-1B | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-tteRPL49lffYMhW6vQULtQ) | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-0eyg6qR5BAFRImJ4qtr2lK) |
| Llama-3.2-3B | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-qcHEGLAYkGMzNZXHwxyjNc) | [Run and logs](https://modal.com/apps/a-nerdy-penguin/main/ap-iHndGr1zDQPRK6MPBIOsjj) |

All new prefill runs used source revision 5097bfea07503f2486348423809223c28f7543d6. Decode revisions are recorded per run in the index because those measurements predate the prefill changes. Detailed CSVs preserve numeric precision; displayed tables round for readability.

The 26 targeted local capture, context-lifecycle and utility checks passed, together with lint and whitespace checks. The broader local core suite could not collect because the installed Hugging Face Hub 2.0 conflicts with Transformers' required version below 1.0. The actual GPU experiments ran in the compatible Modal dependency image. Numerical validation failures remain failures and produce a nonzero process exit after saving the measurements.
