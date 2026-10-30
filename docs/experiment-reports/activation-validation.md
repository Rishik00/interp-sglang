# Activation reference validation

Run artifact: `artifacts/modal-traces/activation-validation-20261030T190224Z.json`.
Modal run: https://modal.com/apps/a-nerdy-penguin/main/ap-8i3lqgEGfc6j5SSxuKakVO

All 60 activation comparisons passed on Qwen/Qwen3-0.6B, Tesla T4,
FP16. The reference used Transformers 4.57.3 with eager attention, while
Mini-SGLang used FlashInfer. Reproduce with:

```powershell
python -X utf8 modal_validate_activations.py
```

The reusable harness is `modal_validate_activations.py`. It compares layers
0, 1, 6, 13, 20, and 27 at two independently hooked HF locations:

- `attention_output`: the output of `self_attn.o_proj`.
- `post_residual`: the decoder block output, before the next normalization.
  This includes the final block **before** the model's final RMSNorm. HF's
  final `output_hidden_states` entry is normalized and is not this signal.

Cases cover two independent short prompts (19 and 21 tokens), a 2,048-token
prompt split across eight 256-token prefill chunks, and two different-length
requests (16 and 23 tokens) sharing one prefill batch. Chunk captures were
reassembled using their request/token spans and compared with a full HF forward.

| Metric | Worst observed |
|---|---:|
| Whole-tensor cosine similarity | 0.99997735 |
| Relative L2 error, attention output | 1.206% |
| Relative L2 error, post-block residual | 0.598% |
| Minimum cosine across individual token vectors | 0.982113 |

Acceptance thresholds were set before execution: relative L2 <= 5%,
whole-tensor cosine >= 0.999, and every token cosine >= 0.98. These are
numerical smoke-test tolerances, not guarantees of equivalence for every
downstream analysis. The worst individual-token agreement is lower than
the whole-tensor agreement; inspect the per-layer/per-case JSON when tighter
accuracy is required. Different FP16 kernels and reduction orders can differ.

Every tensor had the expected shape, finite values, nonzero norm and variance.
Every captured token ID and span matched the requested input, with no gaps or
overlaps after chunk reconstruction. No decode activations were returned despite
three generation steps per request, and a second drain was empty. Capture on/off
produced identical generated token IDs for every case; HF's first generated token
also matched for every request.

No inference or capture implementation changes were needed. This validates the
tested prefill signals on this Qwen model with a naive KV cache, one GPU, and
overlap scheduling disabled. Radix-cache hits, overlap scheduling, tensor
parallelism, other architectures, and long-running capture remain unvalidated.

## LM head input capture

The extended harness enables `capture_lm_head_input=True` and compares the new
model-level signal (`layer_id=None`) with HF's final RMSNorm output. The same
four cases now produce 65 comparisons, including five full-token head-input
comparisons. All passed, as did capture on/off output equality and token/span
alignment. The long case returned 104 records (13 signals across eight chunks);
the two-request case returned 13 records with both requests' spans intact.

Artifact: `artifacts/modal-traces/activation-validation-20261030T192037Z.json`.
Modal run: https://modal.com/apps/a-nerdy-penguin/main/ap-Qby5khgpUIX5Qp9KjTkl9J
