# Qwen3-1.7B and Llama-3.2-1B validation

Historical run: this report used the earlier QA/synthetic prompts. The runner
now defaults to WildChat; the commands below use that new workload. See
`qwen3-wildchat-validation.md` for the dataset run.

## Runner

Activation validation is merged into `modal_benchmark.py`; the old standalone
script has been removed. Both phases run by default, with `--mode benchmark`
and `--mode validate` available separately. Shared layer selection, tokenization,
summary, memory, metric, and artifact helpers live in `utils.py`.

Each phase runs in a single-use Modal container: the engine requires CUDA to be
uninitialized at construction. Reusing the benchmark process for validation
fails that requirement. The combined invocation was rerun after this correction
and completed both phases, exiting with status 1 for the recorded parity failures.

```powershell
python -X utf8 modal_benchmark.py --gpu T4 --model-id Qwen/Qwen3-1.7B --memory-ratio 0.55
python -X utf8 modal_benchmark.py --gpu T4 --model-id meta-llama/Llama-3.2-1B-Instruct --memory-ratio 0.55
```

Layers are selected from model depth; for Qwen3-1.7B they are 0, 1, 6, 13, 20, 27.
Both phases capture attention output, post-block residual, and final LM-head input.
Benchmark sizing flags apply to benchmarking; validation retains the reference
configuration: naive KV cache, 256-token chunks, two concurrent requests,
4,096-token limit, engine memory ratio 0.45, overlap disabled, FP16 on one T4.

## Qwen3-1.7B

The ten QA benchmark pairs produced identical greedy token IDs with capture on
and off. All ten reached EOS within the 512-token cap.

Validation passed 59 of 65 comparisons. All short-prompt and multi-request
comparisons passed. Six comparisons failed in the 2,048-token chunked case:

- LM-head input.
- Layer 13 attention output and post-block residual.
- Layer 20 attention output and post-block residual.
- Layer 27 post-block residual.

The worst relative L2 error was 5.165%; the minimum individual-token cosine was
0.46464. All captured tensor shapes, finite/nonzero checks, token IDs, and span
reconstruction checks passed. Capture on/off outputs matched for every case,
and HF's first generated token matched for every request. This does not establish
activation parity on long prompts; the differences need investigation before
claiming full support. Acceptance thresholds were not relaxed.

Metric reductions now use FP64 on CPU: the first FP32 run produced some invalid
whole-tensor cosine values above 1. The final FP64 run still failed the same six
comparisons, so the validation failures are not removed by that correction.

Modal runs:

- Combined run: https://modal.com/apps/a-nerdy-penguin/main/ap-Vlt3fqP5jTI8Fj14hgrQ1P
- Final FP64 validation: https://modal.com/apps/a-nerdy-penguin/main/ap-cNrbxX3WvdYD1y56FjqSWx

## Llama access blocker

The T4 attempt for `meta-llama/Llama-3.2-1B-Instruct` failed at config download:
Hugging Face returned HTTP 403 and stated that the cached account's request for
access had been rejected. An authenticated check of `meta-llama/Llama-3.2-1B`
returned the same rejection. No Llama weights were loaded and no Llama inference
or parity results are available. A token belonging to an account with repository
access is required. The runner accepts `--hf-secret NAME` for a Modal secret
containing `HF_TOKEN`, or uses the existing local Hugging Face login through an
ephemeral Modal secret. Tokens are not written into artifacts or printed.

## Local checks

Ruff passed. Twelve tests passed: the existing capture tests plus model-depth
selection and a regression test requiring a bad individual token to fail even
when whole-tensor metrics pass. Python compilation and CLI help also passed.
