"""Validate Mini-SGLang prefill captures against independent HF forward hooks.

Run with: python -X utf8 modal_validate_activations.py
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import modal

from config import LOCAL_REPO, MODEL_ID, REMOTE_CACHE, app, cache_volume, dependency_image


@app.function(
    image=dependency_image, gpu="T4", memory=8192, timeout=1200,
    volumes={str(REMOTE_CACHE): cache_volume},
)
def validate() -> dict:
    import torch
    import torch.nn.functional as F
    import transformers
    from minisgl.capture import ActivationCaptureConfig
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = (0, 1, 6, 13, 20, 27)
    # Aggregate and per-token checks tolerate FP16 kernel/reduction differences.
    thresholds = {"relative_l2_max": 0.05, "cosine_min": 0.999,
                  "minimum_token_cosine_min": 0.98}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def encode(text):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        )

    long_ids = encode("Explain how the following facts relate. " +
                      "Water freezes at zero degrees Celsius. Plants use sunlight. " * 250)[:2048]
    cases = [
        ("short", [encode("What is the capital of France?")]),
        ("different_prompt", [encode("Calculate twelve times thirteen and explain your answer.")]),
        ("chunked_2048", [long_ids]),
        ("two_requests", [encode("Name three planets."),
                          encode("Explain why the sky appears blue during the daytime.")]),
    ]
    llm = LLM(
        model_path=MODEL_ID, dtype=torch.float16, cache_type="naive",
        cuda_graph_max_bs=1, max_extend_tokens=256, max_running_req=2,
        max_seq_len_override=4096, memory_ratio=0.45, use_pynccl=False,
        activation_capture=ActivationCaptureConfig(layer_ids=layers, capture_lm_head_input=True),
    )
    saved = []
    sampling = SamplingParams(top_k=1, temperature=0, max_tokens=3, ignore_eos=True)
    try:
        for name, prompts in cases:
            manager = llm.engine.capture_manager
            manager.enabled = False
            off = llm.generate(prompts, sampling)
            assert llm.take_activation_captures() == []
            manager.enabled = True
            on = llm.generate(prompts, sampling)
            captures = llm.take_activation_captures()
            assert llm.take_activation_captures() == []
            assert [r["token_ids"] for r in off] == [r["token_ids"] for r in on]
            saved.append((name, prompts, on, captures))
            print(f"[validate] captured {name}: {len(captures)} records", flush=True)

        # Hook the actual block outputs: HF's final hidden_states entry is
        # normalized, whereas our final post_residual is BEFORE the final norm.
        reference = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager",
        ).eval().to("cuda")
        hooked = {}

        def hook(key):
            def record(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                hooked[key] = value.detach().squeeze(0).cpu().clone()
            return record

        handles = []
        # The final norm emits the full head input before either runtime slices
        # positions for projection (HF's logits_to_keep also slices positions).
        handles.append(reference.model.norm.register_forward_hook(hook((None, "lm_head_input"))))
        for layer in layers:
            block = reference.model.layers[layer]
            handles.append(block.register_forward_hook(hook((layer, "post_residual"))))
            handles.append(block.self_attn.o_proj.register_forward_hook(
                hook((layer, "attention_output"))))

        rows = []
        metadata = []
        for name, prompts, outputs, captures in saved:
            expected_keys = {(layer, signal) for layer in layers
                             for signal in ("attention_output", "post_residual")}
            expected_keys.add((None, "lm_head_input"))
            assert {(c.layer_id, c.signal) for c in captures} == expected_keys
            metadata.append({"case": name, "input_tokens": [len(p) for p in prompts],
                             "capture_records": len(captures),
                             "spans": [asdict(s) for c in captures
                                       if c.layer_id == 0 and c.signal == "post_residual"
                                       for s in c.spans]})
            for request_id, ids in enumerate(prompts):
                hooked.clear()
                with torch.inference_mode():
                    result = reference(
                        input_ids=torch.tensor([ids], device="cuda"), use_cache=False,
                        logits_to_keep=1,
                    )
                assert set(hooked) == expected_keys
                first_token_matches = (int(result.logits[0, -1].argmax()) ==
                                       outputs[request_id]["token_ids"][0])
                del result
                for layer, signal in sorted(
                    expected_keys, key=lambda key: (-1 if key[0] is None else key[0], key[1])
                ):
                    pieces = []
                    for capture in captures:
                        if (capture.layer_id, capture.signal) != (layer, signal):
                            continue
                        assert capture.phase == "prefill"
                        for span in capture.spans:
                            if span.request_id != request_id:
                                continue
                            actual_ids = capture.token_ids[span.tensor_start:span.tensor_end]
                            assert actual_ids.cpu().tolist() == ids[span.token_start:span.token_end]
                            pieces.append((span.token_start, span.token_end,
                                           capture.tensor[span.tensor_start:span.tensor_end]))
                    pieces.sort(key=lambda p: p[0])
                    end = 0
                    for start, stop, _ in pieces:
                        assert start == end, (name, request_id, layer, signal, start, end)
                        end = stop
                    assert end == len(ids)
                    actual = torch.cat([p[2] for p in pieces]).float()
                    expected = hooked[(layer, signal)].float()
                    assert actual.shape == expected.shape == (len(ids), reference.config.hidden_size)
                    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
                    nonzero = bool(actual.norm() > 0 and actual.std() > 0)
                    difference = actual - expected
                    relative_l2 = float(difference.norm() / expected.norm().clamp_min(1e-12))
                    cosine = float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
                    token_cosine = float(F.cosine_similarity(actual, expected, dim=-1).min())
                    passed = (finite and nonzero and relative_l2 <= thresholds["relative_l2_max"]
                              and cosine >= thresholds["cosine_min"]
                              and token_cosine >= thresholds["minimum_token_cosine_min"])
                    rows.append({"case": name, "request_id": request_id, "layer": layer,
                                 "signal": signal, "shape": list(actual.shape),
                                 "finite": finite, "nonzero": nonzero,
                                 "relative_l2": relative_l2, "cosine": cosine,
                                 "minimum_token_cosine": token_cosine,
                                 "max_absolute_error": float(difference.abs().max()),
                                 "hf_first_token_matches": first_token_matches, "passed": passed})
                print(f"[validate] compared {name} request={request_id}", flush=True)
        for handle in handles:
            handle.remove()
        return {"model": MODEL_ID, "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "dtype": "float16", "hf_attention": "eager", "layers": layers,
                "thresholds": thresholds, "cases": metadata, "rows": rows,
                "capture_on_off_outputs_identical": True,
                "all_passed": all(r["passed"] for r in rows)}
    finally:
        llm.shutdown()


def main():
    with modal.enable_output(), app.run(detach=True):
        result = validate.spawn().get()
    path = LOCAL_REPO / "artifacts" / "modal-traces" / (
        "activation-validation-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results: {path}")
    print(json.dumps({"all_passed": result["all_passed"], "comparisons": len(result["rows"]),
                      "worst_relative_l2": max(r["relative_l2"] for r in result["rows"]),
                      "worst_cosine": min(r["cosine"] for r in result["rows"]),
                      "failures": [r for r in result["rows"] if not r["passed"]]}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
