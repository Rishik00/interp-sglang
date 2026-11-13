"""Bounded, CUDA-graph decode harvesting on a pinned WildChat sample.

Run: python -X utf8 modal_test_decode.py --model-id Qwen/Qwen3-1.7B --gpu T4
The uninstrumented baseline and capture run use separate, fresh CUDA processes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace

import modal

from config import LOCAL_REPO, REMOTE_CACHE, app, cache_volume, dependency_image
from modal_benchmark import prepare_workload
from utils import (
    DecodeWindowObserver,
    compare_activations,
    hugging_face_modal_secrets,
    measure_decode_generation,
    mem_cleanup,
    write_run_artifact,
)


def validate_decode_reference(model_id, workload, sampled, final_window):
    """Teacher-force exact token IDs and compare only generated-token positions."""
    import torch
    from transformers import AutoModelForCausalLM

    reference = (
        AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        .eval()
        .to("cuda")
    )
    thresholds = {"relative_l2_max": 0.05, "cosine_min": 0.999, "minimum_token_cosine_min": 0.98}
    rows = []
    matches = total_predictions = 0
    checked_token_rows = 0
    assert final_window.tokens, "Reference validation requires a nonempty final window"
    # A GPU-only diagnostic needs this transfer; CPU-offload mode already owns it.
    # Reference computation is always outside the timed generation region.
    final_values = final_window.tensor.cpu()
    checked_requests = {token.request_id for token in final_window.tokens}
    for request_id, (prompt, generated) in enumerate(zip(workload["prompts"], sampled)):
        if request_id not in checked_requests:
            continue
        positions = [
            i for i, token in enumerate(final_window.tokens) if token.request_id == request_id
        ]
        actual = final_values[:, positions]
        tokens = [vars(final_window.tokens[i]) for i in positions]
        keys = final_window.keys
        indices = [token["generated_index"] for token in tokens]
        checked_token_rows += len(tokens)
        prompt_length = len(prompt["token_ids"])
        assert [token["token_id"] for token in tokens] == [generated[i] for i in indices]
        assert [token["token_position"] for token in tokens] == [prompt_length + i for i in indices]
        assert all(
            token["terminal"] == (i == len(generated) - 1) for token, i in zip(tokens, indices)
        )
        expected = {}

        def hook(key):
            def capture(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                expected[key] = tensor[0, prompt_length:][indices].detach().cpu().clone()

            return capture

        handles = []
        for layer, signal in keys:
            if signal == "lm_head_input":
                module = reference.model.norm
            elif signal == "post_residual":
                module = reference.model.layers[layer]
            else:
                module = reference.model.layers[layer].self_attn.o_proj
            handles.append(module.register_forward_hook(hook((layer, signal))))
        try:
            with torch.inference_mode():
                result = reference(
                    torch.tensor([prompt["token_ids"] + generated], device="cuda"),
                    use_cache=False,
                    logits_to_keep=len(generated) + 1,
                )
                predicted = result.logits[0, :-1].argmax(dim=-1).cpu().tolist()
                matches += sum(a == b for a, b in zip(predicted, generated))
                total_predictions += len(generated)
                del result
            assert set(expected) == set(keys)
            for signal_index, (layer, signal) in enumerate(keys):
                assert actual[signal_index].shape == expected[(layer, signal)].shape
                rows.append(
                    {
                        "request_id": request_id,
                        "dataset_row": prompt["dataset_row"],
                        "layer": layer,
                        "signal": signal,
                        **compare_activations(
                            actual[signal_index], expected[(layer, signal)], thresholds
                        ),
                    }
                )
        finally:
            for handle in handles:
                handle.remove()
        print(f"[decode reference] request={request_id} tokens={len(generated)}", flush=True)
    return {
        "thresholds": thresholds,
        "rows": rows,
        "all_passed": all(row["passed"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "hf_greedy_token_matches": matches,
        "hf_greedy_tokens": total_predictions,
        "token_and_position_alignment": True,
        "terminal_token_coverage": True,
        "scope": "final_window_only",
        "checked_token_rows": checked_token_rows,
    }


@app.function(
    image=dependency_image,
    gpu="T4",
    memory=16384,
    timeout=2400,
    volumes={str(REMOTE_CACHE): cache_volume},
    single_use_containers=True,
)
def run_decode(
    model_id,
    workload,
    capture,
    source_commit,
    max_tokens=250,
    window_steps=16,
    gpu_budget_mib=128,
    ignore_eos=False,
    capture_sink="cpu",
    memory_ratio=0.45,
):
    import torch
    import transformers
    from minisgl.capture import ActivationCaptureConfig
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_id)
    layers = tuple(range(model_config.num_hidden_layers))
    if capture_sink not in {"cpu", "gpu"}:
        raise ValueError("Capture sink must be cpu or gpu; compression is not supported")
    window_observer = DecodeWindowObserver() if capture else None
    capture_config = (
        ActivationCaptureConfig(
            layer_ids=layers,
            capture_lm_head_input=True,
            capture_prefill=False,
            capture_decode=True,
            decode_window_steps=window_steps,
            decode_gpu_budget_bytes=gpu_budget_mib * 2**20,
            decode_sink=window_observer,
            decode_offload_to_cpu=capture_sink != "gpu",
        )
        if capture
        else None
    )
    llm = LLM(
        model_path=model_id,
        dtype=torch.float16,
        cache_type="naive",
        max_running_req=4,
        max_extend_tokens=256,
        max_seq_len_override=4096,
        cuda_graph_bs=[1, 2, 4],
        memory_ratio=memory_ratio,
        use_pynccl=False,
        activation_capture=capture_config,
    )
    manager = llm.engine.capture_manager
    try:
        if manager is not None:
            manager.enabled = False
        llm.generate(
            [workload["prompts"][0]["token_ids"]],
            SamplingParams(top_k=1, temperature=0, max_tokens=8, ignore_eos=True),
        )
        if manager is not None:
            manager.enabled = True
        metrics = measure_decode_generation(
            llm,
            [prompt["token_ids"] for prompt in workload["prompts"]],
            SamplingParams(top_k=1, temperature=0, max_tokens=max_tokens, ignore_eos=ignore_eos),
        )
        llm.cache_manager.check_integrity()
        assert llm.take_activation_captures() == []  # Prefill capture is disabled.
        result = {
            "model": model_id,
            "source_commit": source_commit,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "capture": capture,
            "workload": workload,
            "metrics": metrics,
            "config": {
                "layers": layers,
                "max_tokens": max_tokens,
                "window_steps": window_steps,
                "gpu_budget_mib": gpu_budget_mib,
                "dtype": "float16",
                "max_running_req": 4,
                "cache_type": "naive",
                "max_extend_tokens": 256,
                "max_seq_len": 4096,
                "memory_ratio": memory_ratio,
                "ignore_eos": ignore_eos,
                "overlap": False,
                "capture_sink": capture_sink if capture else None,
                "cuda_graph_batches": sorted(llm.engine.graph_runner.graph_map),
            },
            "cache_integrity": True,
        }
        if manager is not None:
            assert manager.decode_stats["captured_rows"] == metrics["sampled_tokens"]
            assert manager.decode_stats["terminal_rows"] == len(workload["prompts"])
            assert manager.decode_stats["graph_batches"] > 0
            assert manager.decode_stats["eager_batches"] == 0
            expected_offload = (
                metrics["sampled_tokens"]
                * len(manager.decode_keys)
                * model_config.hidden_size
                * torch.float16.itemsize
                if capture_sink != "gpu"
                else 0
            )
            assert manager.decode_stats["offloaded_bytes"] == expected_offload
            result["capture_stats"] = dict(manager.decode_stats)
            result["storage"] = {"persisted": False}
            assert window_observer.rows == metrics["sampled_tokens"]
            assert window_observer.windows == manager.decode_stats["windows"]
            assert window_observer.last_window.tensor.is_cuda == (capture_sink == "gpu")
            # Preserve just the bounded CPU window, then free serving weights,
            # KV cache and graphs before loading another FP16 model on the T4.
            final_window = replace(
                window_observer.last_window,
                tensor=window_observer.last_window.tensor.cpu(),
            )
            window_observer.last_window = final_window
            llm.shutdown()
            llm = None
            manager = None
            from minisgl.layers.rotary import get_rope

            get_rope.cache_clear()
            mem_cleanup()
            result["reference_free_gpu_bytes"] = torch.cuda.mem_get_info()[0]
            # Validate after timing, so reference computation is not capture overhead.
            result["reference"] = validate_decode_reference(
                model_id,
                workload,
                metrics["sampled_token_ids"],
                final_window=final_window,
            )
        return result
    finally:
        # Finish any pending offload before destroying graphs.
        if llm is not None:
            llm.shutdown()
        cache_volume.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--window-steps", type=int, default=16)
    parser.add_argument("--gpu-budget-mib", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--memory-ratio", type=float, default=0.45)
    parser.add_argument("--hf-secret", help="Existing Modal secret containing HF_TOKEN")
    parser.add_argument(
        "--capture-sink",
        choices=("cpu", "gpu"),
        default="cpu",
        help="CPU offload (default), or GPU-only diagnostic",
    )
    args = parser.parse_args()
    if args.num_prompts < 4:
        parser.error("--num-prompts must be at least four")
    if not 1 <= args.max_tokens <= 1023 or args.window_steps < 1 or args.gpu_budget_mib < 1:
        parser.error("Invalid token limit, window size or capture budget")
    if not 0 < args.memory_ratio < 1:
        parser.error("--memory-ratio must be between 0 and 1")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=LOCAL_REPO, text=True
    ).strip()
    results = {}
    output_dir = LOCAL_REPO / "artifacts" / "modal-traces"
    secrets = hugging_face_modal_secrets(args.hf_secret)
    with modal.enable_output(), app.run(detach=True):
        workload = (
            prepare_workload.with_options(secrets=secrets)
            .spawn(
                args.model_id,
                {
                    "count": args.num_prompts,
                    "seed": 42,
                    "revision": "c827c6df8fcf008219ffaffa4d1dd77491099367",
                    "max_tokens": 3072,
                    "scan_rows": 4096,
                },
            )
            .get()
        )
        for capture in (False, True):
            result = (
                run_decode.with_options(gpu=args.gpu, secrets=secrets)
                .spawn(
                    model_id=args.model_id,
                    workload=workload,
                    capture=capture,
                    source_commit=source_commit,
                    max_tokens=args.max_tokens,
                    window_steps=args.window_steps,
                    gpu_budget_mib=args.gpu_budget_mib,
                    ignore_eos=args.ignore_eos,
                    capture_sink=args.capture_sink,
                    memory_ratio=args.memory_ratio,
                )
                .get()
            )
            name = f"decode-{args.capture_sink}" if capture else "decode-baseline"
            results[name] = result
            path = write_run_artifact(result, output_dir, name)
            print(f"Results: {path}")
            print(
                json.dumps(
                    {
                        "mode": name,
                        "seconds": result["metrics"]["seconds"],
                        "tokens_per_second": result["metrics"]["tokens_per_second"],
                        "sampled_tokens": result["metrics"]["sampled_tokens"],
                        "capture_stats": result.get("capture_stats"),
                        "reference_passed": result.get("reference", {}).get("passed"),
                        "request_timings": [
                            {
                                key: value
                                for key, value in row.items()
                                if key != "token_receipt_seconds"
                            }
                            for row in result["metrics"]["request_timings"]
                        ],
                    },
                    indent=2,
                )
            )
    baseline = results["decode-baseline"]
    captured = results[f"decode-{args.capture_sink}"]
    identical = baseline["metrics"]["sampled_token_ids"] == captured["metrics"]["sampled_token_ids"]
    summary = {
        "model": args.model_id,
        "source_commit": source_commit,
        "capture_sink": args.capture_sink,
        "all_outputs_identical": identical,
        "throughput_retained": captured["metrics"]["tokens_per_second"]
        / baseline["metrics"]["tokens_per_second"],
        "reference_passed": captured["reference"]["all_passed"],
        "reference_scope": captured["reference"]["scope"],
        "storage": captured["storage"],
    }
    print(json.dumps(summary, indent=2))
    write_run_artifact(summary, output_dir, "decode-summary")
    if not identical or not summary["reference_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
