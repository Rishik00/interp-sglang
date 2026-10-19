# Mini-SGLang activation capture: implementation and T4 benchmark report

## Executive summary

We added an experimental activation-capture path to Mini-SGLang and tested it with `Qwen/Qwen3-0.6B` on a Modal Tesla T4. The implementation captures two signals from five selected transformer layers during prefill:

- Layers: 1, 6, 13, 20, and 27, using zero-based layer numbering.
- Signals: the attention output after the output projection, and the residual stream after the complete transformer block.

The final benchmark used ten different prompts containing real Mini-SGLang source. Nine prompts were exactly 2,048 chat-template tokens and one was 2,047 tokens. The model greedily decoded every response until the EOS token. Responses ranged from 514 to 1,697 generated steps, and none reached the 3,072-token safety limit.

Capture-on and capture-off produced identical token IDs for all ten prompts. Weighted decode throughput was 113.26 tokens per second without capture and 113.51 tokens per second with capture. This tiny apparent improvement is benchmark noise, not a speedup. The experiment did not detect a meaningful steady-state decode penalty because this implementation captures prefill only.

Each 2,048-token capture retained exactly 40 MiB of activation tensors. Peak active PyTorch allocation increased by 36 MiB. Reserved and driver-used memory were unchanged, indicating that capture fit within the allocator's existing reservation.

The result is encouraging for this narrow experiment, but it is not yet a continuous production capture pipeline. It lacks decode capture, bounded staging pools, asynchronous device-to-host transfer, background sharded writing, backpressure, crash recovery, and multi-rank capture semantics.

## What was implemented

### Capture configuration and manager

`python/minisgl/capture.py` introduces:

- `ActivationCaptureConfig`, which selects layer IDs and signal types.
- `ActivationSpan`, which maps slices of a flattened tensor back to request and token positions.
- `CapturedActivation`, the CPU-facing capture record.
- `ActivationCaptureManager`, which records selected tensors on the GPU and drains them to the CPU on demand.

The manager is attached to the global model execution context. The engine validates layer IDs at startup, constructs the manager when capture is configured, and makes it available to model layers through the existing global context.

Capture is currently restricted to prefill. This is an explicit guard in `ActivationCaptureManager.capture()`. Decode normally runs through CUDA graph replay, and this first implementation leaves the decode graph untouched.

### Captured signals

The attention signal is collected in `RopeAttn.forward()` after `o_proj`. It is therefore the final hidden-size attention contribution that would be added into the residual path. It is not an attention probability matrix and it is not the unprojected per-head context tensor.

Mini-SGLang carries the accumulated residual and the current sublayer output separately so FlashInfer can fuse their addition with normalization. Consequently, the completed output of block N is captured when those values are combined at the start of block N+1. The fused operation normalizes `x` in place but leaves the completed, unnormalized block output in `residual`, which is the tensor recorded here. The last block has no next layer, so its completed `residual` is captured immediately after the final fused add-and-normalize call. This gives layer 27 the same unnormalized post-block residual semantics as the other selected layers.

Equivalent capture points were added to both the Qwen3 and Llama implementations, although the reported GPU experiment used Qwen3 only.

### Why tensors are cloned

The source tensors can be mutated or reused by later fused residual and normalization operations. The manager therefore performs `tensor.detach().clone()` while the model is executing. The clone is a device-to-device copy issued on the current inference stream. Captured GPU tensors remain alive until the caller drains the manager.

This design is intentionally simple. It allocates one new GPU tensor for every selected layer and signal in a prefill batch. It is suitable for correctness experiments and small runs, but it is not the final design for sustained dataset production.

### Provenance

Each capture includes:

- Signal name.
- Layer ID.
- Inference phase.
- Captured tensor.
- Corresponding token IDs.
- Request ID and tensor/token ranges for each request represented in the flattened batch.

This is enough to trace every captured row back to its request and token position in the tested single-request workload.

### User-facing retrieval

`LLM.take_activation_captures()` drains staged tensors to the CPU and clears the manager's staging list. The smoke harness can serialize the result with `torch.save`. The benchmark drains the records after each generation and measures the drain separately from inference.

## Modal test environment

The Modal setup keeps slow-changing dependencies in a cached image and mounts the local Mini-SGLang checkout separately. Source changes therefore do not require publishing a package or rebuilding the dependency image.

The environment used:

- GPU: NVIDIA Tesla T4.
- Model: `Qwen/Qwen3-0.6B`.
- Model dtype: FP16.
- PyTorch base image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`.
- Attention backend: automatically selected FlashInfer backend.
- CUDA graphs: enabled for decode, with graph batch sizes 4, 2, and 1 captured by Mini-SGLang.
- Tensor parallel size: one.
- KV-cache manager: naive, which disables prefix reuse and prevents the second member of a paired test from receiving an unfair prefix-cache hit.
- Engine memory ratio: 0.60.
- Allocated KV-cache capacity at initialization: approximately 7.07 GiB.
- Maximum sequence length for the final benchmark: 6,144 tokens.
- Maximum concurrent requests: one.

The dependency cache is stored in a Modal Volume. Mini-SGLang source is mounted from the local checkout into `/workspace/mini-sglang` and exposed through `PYTHONPATH`.

## Smoke testing and execution tracing

Before benchmarking, `modal_smoke.py` was created to validate that the local checkout could be executed on a rented GPU. It can:

- Run deterministic offline inference.
- Enable or disable CUDA graphs.
- Enable or disable activation capture.
- Install a filtered Python function-call profiler for Mini-SGLang packages.
- Write the remote execution trace into a local `.log` file.
- Validate expected capture keys, tensor finiteness, shape, dtype, layer, signal, and phase.
- Serialize captured activations into a local `.activations.pt` artifact.

The call profiler uses `sys.setprofile()` rather than line-by-line `sys.settrace()`. It records calls and returns from the scheduler, engine, model, layer, attention, and KV-cache modules. It does not attempt to describe individual CUDA kernels.

## Final benchmark protocol

Each of the ten prompts was executed once with capture disabled and once with capture enabled. The order alternated by prompt to reduce systematic order bias.

Generation settings were:

- Qwen chat template.
- Thinking mode disabled.
- Greedy sampling with `top_k=1` and temperature 0.
- EOS respected.
- Maximum of 3,072 generated steps as an emergency limit.

The harness wrapped Mini-SGLang's result callback to record first-token time, last-token time, generated-step count, final token ID, and finish reason. It also detected Mini-SGLang's behavior of not returning the terminal EOS token in the public output list. Generated-step counts in this report include that terminal step; returned-token counts do not.

Before every measured generation, the harness:

1. Drained and verified the capture queue was empty.
2. Ran Python garbage collection.
3. Called `torch.cuda.empty_cache()`.
4. Synchronized the GPU.
5. Reset PyTorch peak-memory statistics.
6. Recorded a baseline memory snapshot.

After generation, it recorded active allocation, reserved memory, CUDA driver free/used memory, exact capture payload bytes, CPU-drain time, and memory after release. Capture-on and capture-off token IDs were compared, and the run would fail immediately if they differed.

## Prompt and output coverage

The ten tasks covered:

1. End-to-end inference-server request flow.
2. Simple cloning versus a staged asynchronous writer pipeline.
3. Prefill and decode tensor shapes.
4. Activation dataset provenance and shard integrity.
5. Why activation capture may have little effect on decode throughput.
6. Eager execution versus CUDA graph replay.
7. Debugging deterministic inference benchmarks.
8. Tensor parallelism and activation sharding.
9. Simplifying the server to a single-GPU research runtime.
10. Semantic differences among attention, MLP, and residual signals.

Generated lengths, including the terminal step, were:

| Prompt | Input tokens | Generated steps | Finish reason |
| ---: | ---: | ---: | --- |
| 0 | 2,048 | 514 | EOS |
| 1 | 2,048 | 792 | EOS |
| 2 | 2,048 | 941 | EOS |
| 3 | 2,048 | 747 | EOS |
| 4 | 2,048 | 576 | EOS |
| 5 | 2,048 | 1,003 | EOS |
| 6 | 2,048 | 723 | EOS |
| 7 | 2,048 | 772 | EOS |
| 8 | 2,048 | 1,697 | EOS |
| 9 | 2,047 | 823 | EOS |

Across the ten unique prompts, the model executed 8,588 generated steps including terminal steps. Capture-on and capture-off generated identical returned token sequences for every prompt.

## Final performance results

### Decode throughput

Weighted decode throughput was calculated by summing decode tokens and decode time across all ten prompts, rather than averaging ten per-prompt rates. The decode interval begins after the first generated step is delivered and ends when the terminal step is delivered. Its numerator is therefore `generated steps - 1`; the terminal EOS step is included and the already-completed first step is not.

| Mode | Weighted decode throughput |
| --- | ---: |
| Capture off | 113.26 tokens/s |
| Capture on | 113.51 tokens/s |
| Observed difference | +0.22% |

The paired per-prompt changes ranged from -6.46% to +2.15%, and the paired median was +0.57%. This implementation adds no capture operation to decode, so there is no expected direct decode speedup. The +0.22% aggregate difference is small relative to the observed variation between pairs and should not be interpreted as an improvement.

The correct conclusion is not that capture improves performance. It is that this experiment detected no material steady-state decode penalty.

That result is plausible because the implementation performs no capture work during decode. Prefill clones remain resident while decode proceeds, but no clone, allocation, or host transfer is added to each decode step.

### Time to first token and prefill

| Mode | Median time to first token | Median derived prefill throughput |
| --- | ---: | ---: |
| Capture off | 122.12 ms | 16,766.93 tokens/s |
| Capture on | 118.81 ms | 17,237.58 tokens/s |

Time to first token runs from immediately before `llm.generate()` until Mini-SGLang delivers its first generated step. It includes scheduling, the complete prefill, sampling, and callback overhead. The reported "derived prefill throughput" is simply input-token count divided by that interval; it is not an isolated measurement of GPU prefill kernels.

The paired median time-to-first-token change was +0.25%, but individual paired changes ranged from -34.31% to +10.82%. The median values happen to favor capture because they summarize different individual observations. The difference between the two mode medians is not the median of the ten paired differences. These values should not be interpreted as a capture speedup.

More repetitions are required for a precise prefill-overhead estimate. The safe conclusion is that the present test did not resolve a consistent time-to-first-token penalty beyond run-to-run noise.

### End-to-end timing and drain cost

| Mode | Total generation time | Total generation plus drain | Aggregate all-token throughput |
| --- | ---: | ---: | ---: |
| Capture off | 77.004 s | 77.004 s | 377.47 tokens/s |
| Capture on | 76.794 s | 76.913 s | 377.92 tokens/s including drain |

All-token throughput uses input tokens plus generated steps as its numerator. It is intentionally different from decode throughput, which measures only steps after the first generated step.

Capture-on spent 119.38 ms draining activations across ten requests, or approximately 11.94 ms per request. Capture-off spent about 0.50 ms in the same empty-drain path.

Generation timing variability was larger than the drain penalty in this single paired run, so total capture-on time still happened to be slightly lower. This does not erase the measured drain cost. The ten capture drains added about 119 ms in total, compared with approximately 77 seconds of generation. One measurement pair per prompt is insufficient to separate that cost from timing variability.

## Final memory results

### Exact payload size

Qwen3-0.6B has a hidden size of 1,024. The benchmark captures five layers and two hidden-size signals in FP16 for each input token.

Plain-text calculation:

`5 layers * 2 signals * 2,048 tokens * 1,024 values * 2 bytes = 41,943,040 bytes = 40 MiB`

Prompt 9 contained 2,047 tokens and therefore produced 39.98 MiB. The other nine prompts each produced exactly 40 MiB.

### Observed GPU memory

| Measurement | Capture off | Capture on | Difference |
| --- | ---: | ---: | ---: |
| Median peak active PyTorch allocation | 8,587.55 MiB | 8,623.55 MiB | +36 MiB |
| Median PyTorch reserved memory | 9,038.00 MiB | 9,038.00 MiB | 0 MiB |
| Median CUDA driver-used memory | 9,204.19 MiB | 9,204.19 MiB | 0 MiB |
| Exact retained capture payload | 0 MiB | 40 MiB | +40 MiB |

The exact retained payload and the difference between two allocation peaks measure related but different things. Capture retains 40 MiB, while the peak active-allocation reading rises by 36 MiB. Peak differences also depend on the lifetimes and overlap of temporary prefill tensors, so the 4 MiB difference should not be read as missing capture data.

Reserved and driver-used memory were unchanged, indicating that the observed capture fit within the allocator's existing reservation. Calling `torch.cuda.empty_cache()` before a run releases eligible unused cached blocks, but it does not guarantee that every reserved segment or reusable region disappears. A zero change in reserved memory does not mean capture used no memory; the active-allocation and exact-payload measurements show that it did.

The 40 MiB payload is approximately 0.44% of the engine's 9,038 MiB PyTorch reservation. That is why five-layer prefill capture is modest on a 16 GB T4 even though the tensors are real and remain live through generation.

Memory scales linearly with token count, hidden size, number of signals, number of layers, and bytes per value. Capturing both signals from all 28 Qwen3 layers for 2,048 tokens would require about 224 MiB, before allocator overhead and provenance. Capturing decode steps as separate retained tensors could grow with generated length and would need a bounded pipeline.

## Supported scope

The experiment supports the following claims:

- The chosen attention-output and post-residual capture points produced finite tensors of the expected shapes for Qwen3-0.6B prefill.
- Five selected layers produce ten capture tensors per request.
- Captures preserve token and request provenance for the tested workload.
- Capture-on and capture-off greedy outputs were identical across ten long prompts.
- CUDA graphs remained enabled for decode.
- A 2,048-token, five-layer, two-signal FP16 capture retains exactly 40 MiB.
- The synchronous CPU drain costs about 12 ms per request in this setup.
- No material decode-throughput penalty was detected because capture is prefill-only.

## Outside the tested scope

The experiment does not establish:

- That capture is free. It has a measurable memory and drain cost.
- A precise sub-percent prefill overhead. More repetitions are needed.
- Performance with concurrent requests or continuous batching.
- Performance with radix prefix caching.
- Performance when capturing every layer.
- Performance when capturing MLP intermediate activations.
- Performance when capturing decode activations.
- Safe sustained writing of millions or billions of activation vectors.
- Correct multi-GPU or tensor-parallel gathering behavior.
- Crash-safe artifact production.
- Attention probability extraction. The current attention signal is the projected attention output, not the probability matrix.
- Equivalence to Goodfire's complete production architecture.

## Known implementation limitations

### Unbounded retained GPU tensors

Every capture currently creates a fresh clone and stores it in a Python list. If captures are not drained, GPU memory grows without a bound. The final benchmark drains once per completed request, so at most one prefill's selected tensors remain staged.

### Synchronous CPU drain

`tensor.to(device="cpu")` is performed during drain. There is no pinned CPU pool, separate copy stream, CUDA-event handoff, or background writer. The caller pays the transfer latency at the drain boundary.

### No backpressure or checkpoint writer

The current code does not shard activations to disk after every N tokens, write checksums, create atomic completion markers, or block on a bounded writer queue. Those mechanisms are required before running a long corpus extraction job.

### Prefill only

The manager skips decode batches. This preserves the current CUDA graph path but means the dataset contains prompt-token activations only. Decode-token interpretability requires an explicit graph-compatible capture design or controlled eager fallbacks.

### Single-rank semantics only

The benchmark used tensor parallel size one. On multiple ranks, row-parallel outputs may already be reduced while other tensors may remain sharded. A production capture API must state whether each signal is rank-local, gathered, reduced, or written as coordinated shards.

## Recommended next steps

1. Repeat each final prompt several times and report confidence intervals for time to first token, prefill throughput, and decode throughput.
2. Benchmark capture sizes of 1, 5, 14, and 28 layers to verify linear memory scaling.
3. Add the gated MLP activation immediately after `silu_and_mul` as an optional third signal.
4. Introduce a bounded GPU staging pool so capture does not allocate a fresh tensor each time.
5. Copy into reusable pinned CPU buffers on a dedicated CUDA stream, with CUDA events establishing safe ownership transfer.
6. Add a bounded writer queue, sharded files, metadata manifests, checksums, atomic completion markers, and shutdown flushing.
7. Define an explicit backpressure policy for cases where disk is slower than inference.
8. Add long-run tests for queue saturation, repeated drain, shutdown, simulated writer crashes, and artifact recovery.
9. Design decode capture around static graph-owned buffers or periodic eager capture windows without disabling CUDA graphs globally.
10. Define tensor-parallel capture semantics before testing more than one GPU.

## Reproduction

The final run used branch `main` at base commit `7f896d198fca46b392bc8faed7a65a43b3aacc37`, plus the uncommitted activation-capture and Modal harness changes described in this report. Reproducing it exactly therefore requires this working tree, not the base commit by itself.

Install and authenticate the Modal CLI, ensure the account can rent a T4, then run from the repository root:

```powershell
modal run modal_benchmark.py
```

The benchmark harness is [`modal_benchmark.py`](modal_benchmark.py). The smoke and call-tracing harness is [`modal_smoke.py`](modal_smoke.py).

Primary artifacts:

- [`artifacts/modal-traces/benchmark-10-prompts-20261017T091744Z.prompts.txt`](artifacts/modal-traces/benchmark-10-prompts-20261017T091744Z.prompts.txt): exact prompts.
- [`artifacts/modal-traces/benchmark-10-prompts-20261017T091744Z.json`](artifacts/modal-traces/benchmark-10-prompts-20261017T091744Z.json): prompts, outputs, token IDs, finish reasons, timings, memory snapshots, and summaries.
- [`artifacts/modal-traces/benchmark-10-prompts-20261017T091159Z.json`](artifacts/modal-traces/benchmark-10-prompts-20261017T091159Z.json): completed short-prompt intermediate run, superseded for long-prefill conclusions.

Validation performed on the final harness:

- Python compilation with `python -m py_compile modal_benchmark.py`.
- Ruff static checks with `python -m ruff check modal_benchmark.py`.
- Full Modal execution on a T4.
- EOS completion for all ten prompts.
- Token-for-token capture-on/off equality.
- Expected capture count validation.
- Exact payload-byte measurement.
- PyTorch active and reserved memory measurement.
- CUDA driver free/used memory measurement through `torch.cuda.mem_get_info()`.

Tensor finiteness and detailed signal metadata were validated in the smoke harness. The final benchmark independently validated expected capture count, exact payload bytes, EOS completion, token equality, and memory/timing measurements.

## Appendix: benchmark development history

The benchmark was corrected several times as weaknesses became apparent. The rejected or superseded runs explain why the final methodology is stricter.

### Initial fixed-length benchmark

The first performance test used one synthetic 2,048-token technical passage and forced 64 output steps with EOS ignored. It warmed both modes, alternated capture order, and excluded model initialization and graph construction. It reported roughly 106 decode tokens per second and a 10.56 ms capture drain, but it was only a smoke test: 64 forced steps do not exercise normal EOS handling, and one repeated passage does not show workload variability.

### First ten-prompt attempt

The next harness introduced ten distinct questions, paired capture-on/off runs, EOS accounting, output-identity checks, and detailed memory snapshots. Its first attempt retained the 1,024-token safety limit. Several generations reached that limit rather than EOS. That run was stopped and rejected instead of being reported as complete.

### Completed short-prompt run

The answer instruction was tightened and the safety limit raised to 3,072 tokens. A run with 140 to 173 input tokens per prompt completed successfully: every response reached EOS and all paired outputs matched. However, these prompts were not comparable to the earlier 2,048-token prefill. They produced only about 2.7 to 3.4 MiB of captured tensors. That run was retained as an intermediate artifact but superseded for memory conclusions.

### Final 2,048-token run

For the final test, each question was combined with a real Mini-SGLang source excerpt. The source prefix was selected with a tokenizer-aware binary search so that the complete chat-templated prompt was no longer than 2,048 tokens. This produced nine 2,048-token prompts and one 2,047-token prompt. Exact prompts and complete raw results are linked in the reproduction section.

## Bottom line

Five-layer prefill capture is genuinely small enough to be practical on this model and GPU. It retains 40 MiB for a 2,048-token request, compared with roughly 9 GiB already reserved by the initialized inference engine. The T4 sustains about 113 decoded tokens per second for these long-context, single-request generations, and this prefill-only capture path did not measurably reduce decode throughput.

That is a solid experimental baseline, not the finished interpretability infrastructure. The next engineering threshold is not another hook. It is a bounded, asynchronous, crash-safe pipeline that can continuously move activations off the GPU without unbounded memory growth or silently corrupting token provenance.
