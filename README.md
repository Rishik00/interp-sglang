# Interp Mini-SGLang

Mini-SGLang with experimental activation harvesting for interpretability work.
GPU execution uses the existing engine, scheduler, attention backend and CUDA graphs.

## Decode harvesting prototype

Run the paired uninstrumented baseline and capture experiment on a Modal T4:

```powershell
python -X utf8 modal_test_decode.py --model-id Qwen/Qwen3-1.7B --gpu T4 --num-prompts 20 --max-tokens 100
```

The script uses a pinned, length-stratified WildChat sample. All 28 layers'
attention outputs and post-block residuals, plus the final LM-head input, are
captured in FP16 during decode. Prefill capture is disabled. CUDA graphs remain
enabled for batch sizes 1, 2 and 4; at most four requests run concurrently.
Greedy generation stops at EOS or 100 sampled tokens. The terminal token gets
one additional forward **without sampling** before its KV cache is released.
EOS is retained in capture metadata even though normal returned text omits it.

Capture memory is reserved before sizing the KV cache. The default staging window
flushes after 16 decode batches (or when full), with a 128 MiB GPU capture budget.
The prototype offloads synchronously and writes independent lossless gzip chunks.
This applies backpressure instead of retaining an unbounded queue. It supports
one GPU and requires overlap scheduling to be disabled; the shared Modal image
sets `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`.

The baseline and instrumented run use fresh containers. The instrumented run also
checks every captured signal against a Hugging Face teacher-forced reference,
including the terminal token, and verifies decompressed chunk checksums. Reports
include actual throughput, generation equality, capture memory, compression sizes,
and reference comparison failures. Numerical failures produce exit status 1.

Results are saved locally under `artifacts/modal-traces/`. Compressed captures and
their index persist on the `mini-sglang-cache` Modal volume under the directory
printed in the result's storage metadata. Download a run with:

```powershell
python -m modal volume get mini-sglang-cache decode-harvest/<run-id> artifacts/decode-harvest/<run-id>
```

Run the local capture and utility tests (a unique temporary directory avoids a
shared Windows pytest temporary-directory permission problem):

```powershell
$env:PYTHONPATH = "$PWD/python"
python -m pytest tests/core/test_capture.py tests/misc/test_benchmark_utils.py -q --basetemp artifacts/pytest-decode-local
```

## Prefill benchmark and reference checks

```powershell
python -X utf8 modal_benchmark.py --gpu T4 --model-id Qwen/Qwen3-1.7B --memory-ratio 0.55 --num-prompts 8
python -X utf8 modal_benchmark.py --mode sample --model-id Qwen/Qwen3-1.7B --num-prompts 8
```

Both benchmark and validation run by default. Use `--mode benchmark` or
`--mode validate` for one phase. `--mode sample` prepares a prompt manifest on CPU.
Shared experiment helpers are in `utils.py`; decode-specific helpers have their
own `Decode` section. This prototype does not implement paper reproductions,
gradient support, asynchronous compression, or storage uploads to a CDN.
