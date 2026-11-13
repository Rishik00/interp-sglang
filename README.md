# Interp Mini-SGLang

Mini-SGLang fork with activation capture for interpretability work.

While the model generates text, it records the activations of every layer (attention outputs, the
residual stream after each block, and the input to the LM head) and copies them from the GPU to the
CPU. It also checks that capturing doesn't change the generated text, compares the captured
activations against a Hugging Face reference, and measures how much capturing slows generation down.

## Minimal example

Qwen3-1.7B on a T4, 50 prompts, up to 250 generated tokens each:

```powershell
python -X utf8 modal_test_decode.py
```

This runs generation twice in fresh containers, once without capture and once with it, then prints
the throughput, whether the generated text matched, capture memory, and the reference check.
Benchmark metrics are saved under `artifacts/modal-traces/`. The activations themselves are not
saved.

Common options:

- `--model-id meta-llama/Llama-3.2-1B` runs a different model. Llama weights are gated, so set
  `HF_TOKEN` first or pass `--hf-secret <modal-secret-name>`.
- `--memory-ratio 0.75` is needed for the 3B–4B models on a T4.

## Reports

If you're interested in how things are so far, check out: 
- [Prefill and decode summary, all four models](docs/experiment-reports/prefill-decode-summary.md)
