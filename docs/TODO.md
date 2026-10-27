# TODO

## 1. Stress-test the current setup

### Correctness against a reference
- [ ] Compare captured `post_residual` with HF `transformers` `output_hidden_states=True`
      for Qwen3-0.6B and a Llama model (layer N here = `hidden_states[N + 1]` there). Set fp32/fp16 tolerances.
- [ ] Compare `attention_output` with a forward hook on HF's `o_proj`.
- [ ] Greedy outputs are identical with capture on and off (re-run on this base).

### Scheduler interactions that could break capture provenance
- [ ] **Chunked prefill**: a long prompt split into `ChunkedReq`s yields one capture per chunk;
      check that the spans line up and that concatenating the chunks matches an unchunked run.
- [ ] **Radix cache hits**: cached prefix tokens are never recomputed, so they are never captured.
      Decide on a policy (document the gap, or disable the radix cache when capturing) and test it.
      The benchmark currently hides this by using `cache_type="naive"`.
- [ ] **Multi-request batches**: several requests of different lengths in one prefill batch;
      spans must map each row to the right request and token position.
- [ ] **Overlap scheduling**: capture still lines up with the batch it came from when CPU scheduling
      runs ahead of the GPU.
- [ ] **CUDA graph padding**: padded dummy requests must not show up in captures.
- [ ] **Tensor parallelism (TP > 1)**: what each rank sees (is `o_proj` output all-reduced before
      the hook?), and whether only rank 0 should keep captures.

### Load and memory
- [ ] Long contexts (8k, 32k) with many layers selected: GPU memory grows with `_staged`
      until `drain()`. Find where it OOMs and add a cap or a clear error.
- [ ] Many requests without draining (leak check); drain partway through a run.
- [ ] Throughput overhead vs number of layers captured (1, 5, all), not just the fixed 5.
- [ ] Unit tests for the remaining coverage gaps in `capture.py` (wrong-shape error path,
      disabled-signal path).

## 2. SGLang components worth borrowing

Mini-SGLang already has radix cache, chunked prefill, overlap scheduling, TP, and FA/FlashInfer.
Candidates to evaluate in full SGLang:

- [ ] **Hidden-state return API** (`return_hidden_states`): compare its design and hook points with ours.
- [ ] **Deterministic / batch-invariant inference**: makes captured activations reproducible
      regardless of batch composition. Probably the most valuable for interp.
- [ ] **Hybrid linear-attention state cache** (used for Qwen3-Next): a prerequisite for Qwen 3.5 (§3).
- [ ] **LoRA serving** (multi-adapter batching): lets us capture activations from LoRA-tuned variants.
- [ ] **Custom logit processors / steering hooks**: a starting point for interventions, not just reads.
- [ ] **Weight update without restart** (`update_weights`): swap checkpoints without rebuilding the engine.
- [ ] **Quantization loaders** (FP8, AWQ/GPTQ): may be needed for Bonsai (§3).

For each: note where it lives in SGLang, how much code it pulls in, and whether it conflicts
with the capture hooks.

## 3. Deferred: more architectures

This work is lower priority than validating and hardening capture on the current supported models.

General: each new model needs both hooks (`attention_output` and `post_residual`), a registry
entry in `models/register.py`, and a parity test against HF (§1).

### Pythia (GPT-NeoX), e.g. EleutherAI/pythia-160m
- [ ] Missing layers: `LayerNorm` with bias (only RMSNorm exists), plain GELU MLP
      (only gated `silu_and_mul` / `gelu_and_mul`), biased linear layers.
- [ ] Partial rotary embeddings use only a fraction of each attention head for rotation
      (`rotary_pct=0.25`); current `RotaryEmbedding` requires the rotary dimension to equal
      the full head size, so it must be generalized.
- [ ] Parallel residual (`use_parallel_residual=True`): attention and MLP read the same input,
      so there is no post-attention residual. Define what `post_residual` means here.
- [ ] Untied embeddings (`embed_in` / `embed_out`) and fused `query_key_value` weight layout in the loader.

### Qwen 3.5
- [ ] Hybrid layout: roughly 3 Gated DeltaNet (linear attention) layers per full-attention layer.
      Needs a recurrent-state cache alongside the KV cache, plus DeltaNet and causal-conv1d kernels.
- [ ] HF ships it as `Qwen3_5ForConditionalGeneration` (multimodal). Load the text backbone only.
- [ ] Define the capture signal for DeltaNet layers (there is no `o_proj` attention output in the same sense).
- [ ] Start with the smallest checkpoint (~0.8B).

## 4. TransformerLens integration

Goal: harvest activations here at serving speed, then analyze and intervene with TransformerLens tooling.

- [ ] **Name mapping**: map our signals to TL hook names
      (`post_residual` at layer N → `blocks.N.hook_resid_post`, `attention_output` → `blocks.N.hook_attn_out`),
      and document the differences (TL folds LayerNorm and centers weights by default).
- [ ] **Export format**: write drained captures to an `ActivationCache`-compatible structure
      (or to disk shards) keyed by TL hook names, keeping request and token provenance.
- [ ] **Parity check**: same prompt through `HookedTransformer.run_with_cache` and through our engine;
      compare activations per hook (load TL with `fold_ln=False`, `center_writing_weights=False`).
- [ ] **Inference on captured activations**: feed saved residuals back into TL
      (e.g. `run_with_hooks` from layer N onward) for patching and probing without recomputing earlier layers.
- [ ] Decide whether interventions (patching/steering) should also run inside our engine
      or stay TL-only.
- [ ] Check TL support for each target model (Qwen3 and Pythia are supported; Qwen 3.5 and Bonsai probably aren't).
