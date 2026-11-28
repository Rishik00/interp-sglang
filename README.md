# sgl_interp

Mini-SGLang fork with activation capture for interpretability work.

While the model generates text, it records the activations of every layer (attention outputs, the
residual stream after each block, and the input to the LM head) and copies them from the GPU to the
CPU. It also checks that capturing doesn't change the generated text, compares the captured
activations against a Hugging Face reference, and measures how much capturing slows generation down.

## Install

Needs Linux and an NVIDIA GPU (the kernels it depends on are CUDA-only).

```bash
pip install git+https://github.com/Rishik00/interp-sglang.git
# Or, from a checkout containing the rename:
pip install -e .
```

The import name is `sgl_interp`. Launch the server with:

```bash
sgl_interp --model Qwen/Qwen3-1.7B
```

The module entry point is `python -m sgl_interp`. Python imports use:

```python
from sgl_interp.capture import ActivationCaptureConfig
from sgl_interp.llm import LLM
```

The package was renamed from `minisgl`; update existing imports and launch commands.

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

### Activation parity with Hugging Face

Three scripts check that the engine's captured activations match the model's. All model work runs
on a Modal T4. Nothing is downloaded locally, and results land in `artifacts/matched-prefill/`
(gitignored). The report is
[Qwen3-1.7B activation parity](docs/experiment-reports/qwen3-activation-parity.md).

| Script | What it does |
|---|---|
| `modal_validate_activations.py` | Engine FP16 against HF FP16 eager, on the same saved token IDs, at 57 capture points per token (for Qwen3-1.7B), across many prompts. Saves per-token L2, HF vector norm and relative error. `--audit-weights` instead checks that the engine loads exactly HF's weights. |
| `modal_trace_outlier.py` | Deep dive on one outlier token: traces 18 points per block, replays each engine operation on HF's inputs, swaps operation families for HF's versions, and compares against one HF FP32 pass. |
| `modal_fp32_referee.py` | Per prompt: engine FP16, HF FP16 and HF FP32 (TF32 off). Answers which FP16 implementation is closer to FP32, token by token, and pools the tokens across prompts. |

Shared capture and comparison code is in `activation_validation.py`; its CPU tests are in
`tests/misc/test_activation_validation.py`.

```powershell
# 1. Engine vs HF FP16 on 100 WildChat prompts, whole-prompt prefill on both sides
python -X utf8 modal_validate_activations.py --model-id Qwen/Qwen3-1.7B --mode whole --num-prompts 100
#    Same comparison with 256-token chunked prefill on both sides
python -X utf8 modal_validate_activations.py --mode chunked --chunk-size 256
#    Check that the engine loads exactly HF's weights (no prompts)
python -X utf8 modal_validate_activations.py --audit-weights --model-id Qwen/Qwen3-1.7B Qwen/Qwen3-4B

# 2. Deep dive on the run's worst final-state token (or pass --row and --position)
python -X utf8 modal_trace_outlier.py --pick-target --run artifacts/matched-prefill/<model>-whole-<time>.jsonl --workload artifacts/matched-prefill/<model>-workload-<time>.json
python -X utf8 modal_trace_outlier.py --stage trace     # T4: trace and shared-input replay
python -X utf8 modal_trace_outlier.py --stage profile   # T4: swap one operation family at a time
python -X utf8 modal_trace_outlier.py --stage relative  # Modal CPU: relative errors from saved tensors
python -X utf8 modal_trace_outlier.py --stage fp32      # T4: one HF FP32 pass for this prompt

# 3. Engine FP16 and HF FP16 against HF FP32, on chosen rows of a saved workload
python -X utf8 modal_fp32_referee.py --workload artifacts/matched-prefill/<model>-workload-<time>.json --rows 2802 2555 2197 540 5632 1359
#    Re-summarize a finished batch locally, with no Modal calls
python -X utf8 modal_fp32_referee.py --summarize artifacts/matched-prefill/fp32-referee-prompts-<time>

# CPU unit tests for the comparison code (synthetic tensors, no model)
python -m pytest tests/misc/test_activation_validation.py
```

Script 1 reuses each model's earlier 50-prompt WildChat workload under `artifacts/modal-traces/` for
its sampling settings (dataset revision, seed, filters, length ranges). `--num-prompts` draws a new
sample of that size on Modal, then saves its token IDs before inference. Every result is written to
the Modal cache volume first, then downloaded and checked against its size and SHA256. A lost
connection can never be recorded as a completed run.

### Other reports

- [Prefill and decode summary, all four models](docs/experiment-reports/prefill-decode-summary.md)
