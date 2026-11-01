# Interp Mini-SGLang

Mini-SGLang with experimental activation harvesting for interpretability work.
GPU execution uses the existing engine, scheduler, attention backend and CUDA graphs.

## Decode harvesting prototype

Run the paired uninstrumented baseline and GPU-to-CPU capture experiment on a Modal T4:

```powershell
python -X utf8 modal_test_decode.py --model-id Qwen/Qwen3-1.7B --gpu T4 --num-prompts 50 --max-tokens 250
```

CPU offload is the default (`--capture-sink cpu`). There is no activation byte
serialization, compression, checksum calculation or payload file writing.
Each bounded GPU window is copied synchronously into an owned CPU tensor and
consumed by the sink. The benchmark sink discards earlier windows and retains
only the final CPU window for untimed reference validation. It does not retain
or persist the complete activation harvest.

For gated Llama weights, the runner privately forwards the locally cached HF
credential or `HF_TOKEN` environment variable to both preparation and GPU jobs.
The account must already have model access. To use an existing Modal secret:

```powershell
python -X utf8 modal_test_decode.py --model-id meta-llama/Llama-3.2-1B --gpu T4 --hf-secret <secret-name>
```

For a GPU-only diagnostic with CPU offload disabled during generation:

```powershell
python -X utf8 modal_test_decode.py --model-id Qwen/Qwen3-1.7B --gpu T4 --num-prompts 50 --max-tokens 250 --capture-sink gpu
```

GPU mode recycles GPU windows rather than saving the harvest. It keeps only the
last borrowed window for an untimed CPU transfer and Transformers activation
spot-check after generation. Its reference results explicitly cover only that
final window; earlier windows are discarded. Token equality and captured row
counts are checked across the full workload in both modes. GPU sinks receive a
borrowed view and must finish or enqueue consumption on the inference stream
before the window is reused.

Both benchmark scripts now default to 50 prompts and a 250-token generation cap.
The script uses a pinned, length-stratified WildChat sample: 13 prompts from
each of the first two length buckets and 12 from each of the last two. All 28 layers'
attention outputs and post-block residuals, plus the final LM-head input, are
captured in FP16 during decode. Prefill capture is disabled. CUDA graphs remain
enabled for batch sizes 1, 2 and 4; at most four requests run concurrently.
Greedy generation stops at EOS or 250 sampled tokens. The terminal token gets
one additional forward **without sampling** before its KV cache is released.
EOS is retained in capture metadata even though normal returned text omits it.

Capture memory is reserved before sizing the KV cache. The default staging window
flushes after 16 decode batches (or when full), with a 128 MiB GPU capture budget.
The prototype offloads synchronously, applying backpressure instead of retaining
an unbounded queue. It supports one GPU and requires overlap scheduling to be
disabled; the shared Modal image
sets `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`.

The baseline and instrumented run use fresh containers. The instrumented run also
checks signals in the final retained window against a Hugging Face teacher-forced
reference. Earlier discarded windows are not numerically validated. Reports
include throughput, generation equality, capture memory, offloaded bytes and
reference comparison failures. Numerical failures produce exit status 1.

Each run also records per-request host token receipt times. Time to first token
and completion latency start when the whole job is submitted, so they include
queueing and prefill. Per-request decode speed is `(generated tokens - 1) /
(last token time - first token time)`; it excludes the first token produced by
prefill, and includes scheduling stalls, capture and synchronous offloads between
tokens. A one-token request has no decode-speed measurement. The final job flush
after the last token callback remains part of total job time, not token latency.

Benchmark metrics and prompt manifests are saved locally under
`artifacts/modal-traces/` after timing. Activation tensors are not written to
storage. Compression and activation payload storage are deferred; the old gzip
writer, readers and CLI mode have been removed. Historical experiment reports
and previously stored captures are unchanged.

Run the local capture and utility tests (a unique temporary directory avoids a
shared Windows pytest temporary-directory permission problem):

```powershell
$env:PYTHONPATH = "$PWD/python"
python -m pytest tests/core/test_capture.py tests/misc/test_benchmark_utils.py -q --basetemp artifacts/pytest-decode-local
```

## Prefill benchmark and reference checks

```powershell
python -X utf8 modal_benchmark.py --gpu T4 --model-id Qwen/Qwen3-1.7B --memory-ratio 0.55 --num-prompts 50
python -X utf8 modal_benchmark.py --mode sample --model-id Qwen/Qwen3-1.7B --num-prompts 50
```

Both benchmark and validation run by default. Use `--mode benchmark` or
`--mode validate` for one phase. `--mode sample` prepares a prompt manifest on CPU.
Shared experiment helpers are in `utils.py`; decode-specific helpers have their
own `Decode` section. This prototype does not implement paper reproductions,
gradient support, compression, or storage uploads to a CDN.

The completed Qwen3-1.7B T4 run, measured overhead and validation limits are
documented in the [decode harvesting report](docs/experiment-reports/qwen3-decode-harvesting.md).
