# Qwen3-1.7B on WildChat

The current runner replaces the earlier QA and repetitive synthetic prompts with
real first-user turns from `allenai/WildChat-4.8M`. Both phases use the same
sample. This is a small, length-stratified smoke test from one shard, not an
accuracy evaluation or a representative benchmark of the full dataset.

## Reproduction

```powershell
python -X utf8 modal_benchmark.py --gpu T4 --model-id Qwen/Qwen3-1.7B --memory-ratio 0.55 --num-prompts 8 --max-tokens 512
```

Inspect the exact sample without allocating a GPU:

```powershell
python -X utf8 modal_benchmark.py --mode sample --model-id Qwen/Qwen3-1.7B --num-prompts 8
```

Selection runs on CPU before requesting a GPU in the current runner. The first
GPU run preceded this optimization and selected the same sample inside its GPU
container. The independent CPU sample matched every source row and token length.

## Sample provenance

- Dataset revision: `c827c6df8fcf008219ffaffa4d1dd77491099367`.
- Split: `train`; shard: `data/train-00000-of-00086.parquet`.
- Scan: first 4,096 rows; seed: 42.
- Selection: two prompts per input-token range 1–256, 257–1,024,
  1,025–2,048, and 2,049–3,072, followed by a seeded shuffle.
- Uses the first user turn; English and Chinese; excludes records/turns marked
  redacted or toxic; deduplicates prompt text by SHA-256.
- No truncation or repeated text is used to manufacture long inputs.
- Rows in execution order: 1122, 1442, 1946, 327, 1359, 2546, 3665, 2125.
- Input lengths in that order: 1110, 1835, 33, 2956, 2746, 775, 26, 291.

Full prompt text, source identifiers, hashes, and token IDs are saved in
`artifacts/modal-traces/wildchat-prompts-Qwen--Qwen3-1.7B-20261031T110201Z.json`.
Benchmark and validation artifacts also contain the full workload manifest.

## Configuration

One T4, Qwen3-1.7B, FP16, naive prefix cache, overlap disabled, sequence limit
4,096. Six layers (0, 1, 6, 13, 20, 27); attention outputs, post-block residuals,
and final LM-head input; capture is prefill-only. Qwen chat templating disables
thinking and adds the generation prompt.

Benchmark: one request at a time, max prefill extension 3,072, memory ratio 0.55,
greedy generation capped at 512 tokens, stopping at EOS. Each prompt runs with
capture off and on, alternating the order. An eight-token warmup in each mode
is excluded from the measurements.

Validation: max prefill extension 256, max two concurrent requests, memory ratio
0.45, three greedy generation steps with EOS ignored. All eight prompts are
checked individually, followed by the two shortest prompts in a shared batch.
Reference: HF eager attention. Error metrics use FP64 reductions on CPU with
unchanged thresholds: relative L2 <= 5%, whole-tensor cosine >= 0.999,
and every individual token cosine >= 0.98.

## Capture limitations

`cache_type` selects prefix reuse: `naive` does not reuse prefixes across requests;
`radix` stores shared prefixes in a radix tree. Both use a paged KV pool during
inference. Prefix-cache hits skip forward computation and therefore have no new
activation records. Full-prompt parity validation deliberately uses naive caching.

Decode generation and throughput are exercised. Decode activation capture is
not implemented: `ActivationCaptureManager.capture` explicitly returns for
decode batches, and CUDA graph replay bypasses Python capture hooks. Supporting
it needs graph-compatible output buffers and correct per-step token provenance;
changing the prefill guard alone is insufficient.

## Local checks

15 tests passed, including reproducible balanced sampling, nontruncation,
rejection when natural long prompts are missing, and missing-prefix detection.
Ruff, Python compilation, and changed-file whitespace checks passed.

## Run artifacts

Modal run: https://modal.com/apps/a-nerdy-penguin/main/ap-lcxdgsjwZeHGcsiV1IiAnj

Log: `artifacts/qwen3-1.7b-wildchat.log`.


## Results

All eight benchmark pairs produced identical generated token IDs with capture
off/on. One response reached EOS; the other seven reached the 512-token safety
cap. These capped outputs must not be described as completed answers.

Activation validation passed 129 of 130 comparisons. All first-generated-token
comparisons with HF matched, and capture on/off generation matched in every
case. Shapes, token IDs, and span reconstruction checks all passed. The full
workload manifest in both GPU artifacts exactly matched the independent CPU
manifest.

The one failed comparison was in dataset row 2546 (775 input tokens). Its
minimum individual-token cosine was 0.9765599593, below the unchanged threshold
of 0.98. All whole-tensor relative L2 and cosine checks passed; the worst relative
L2 across the run was 1.479%. The command returned exit status 1 to preserve
this failure. This is still a numerical parity failure and needs investigation;
it does not establish full Qwen3 support.

Artifacts:

- Benchmark: `artifacts/modal-traces/benchmark-Qwen--Qwen3-1.7B-20261031T110618Z.json`.
- Validation: `artifacts/modal-traces/validate-Qwen--Qwen3-1.7B-20261031T110701Z.json`.
