"""Paired capture/memory benchmark and HF activation validation.

A pinned, length-stratified WildChat sample replaces the synthetic/QA prompts.
Each prompt is run once with prefill activation capture disabled and once with
capture enabled. Generation stops at EOS, subject to --max-tokens.

Usage (both phases by default):
    python modal_benchmark.py --gpu T4 --model-id Qwen/Qwen3-1.7B
    python modal_benchmark.py --gpu T4 --model-id meta-llama/Llama-3.2-1B-Instruct
    python modal_benchmark.py --mode validate --model-id Qwen/Qwen3-1.7B

Benchmark sizing flags apply to the benchmark phase. Validation deliberately uses
naive caching, 256-token chunks, two concurrent requests, and a 0.45 memory ratio
so both the serving engine and the Hugging Face reference fit on a T4.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from minisgl.llm import LLM  # type: ignore

from config import (
    LOCAL_REPO,
    REMOTE_CACHE,
    REMOTE_REPO,
    app,
    cache_volume,
    dependency_image,
)
from utils import (
    capture_layers,
    compare_activations,
    load_wildchat_prompts,
    mem_cleanup,
    memory_snapshot,
    reconstruct_capture,
    summarize_medians,
    tokenize_prompt,
    write_run_artifact,
)


def run_once(
    manager,
    llm: LLM,
    index: int,
    capture: bool,
    ids: list[int],
    max_tokens: int,
    layers: tuple[int, ...],
    warmup: bool = False,
) -> dict[str, Any]:
    import torch
    from minisgl.core import SamplingParams  # type:ignore

    manager.enabled = capture
    if manager.drain():
        raise RuntimeError("Capture queue was not empty before iteration")

    mem_cleanup()
    memory_before = memory_snapshot()

    first_token_at: float | None = None
    last_token_at: float | None = None
    generated_steps = 0
    final_token_id: int | None = None
    original_send_result = llm.send_result

    def timed_send_result(reply: Any) -> None:
        nonlocal first_token_at, last_token_at, generated_steps, final_token_id
        now = time.perf_counter()
        # Offline LLM callbacks receive a list[DetokenizeMsg].
        for message in reply:
            generated_steps += 1
            final_token_id = int(message.next_token)
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
        original_send_result(reply)

    # LLM.send_result is the scheduler's result-delivery callback. In offline mode
    # it receives each generated step before LLM.generate() returns; wrap it to
    # timestamp token delivery while preserving the original output collection.
    llm.send_result = timed_send_result
    sampling = SamplingParams(
        top_k=1,
        temperature=0.0,
        ignore_eos=False,
        max_tokens=8 if warmup else max_tokens,
    )
    started = time.perf_counter()
    try:
        result = llm.generate([ids], sampling)
        torch.cuda.synchronize()
    finally:
        llm.send_result = original_send_result
    generation_finished = time.perf_counter()

    if first_token_at is None or last_token_at is None or final_token_id is None:
        raise RuntimeError("No generated tokens were observed")

    memory_before_drain = memory_snapshot()
    peak_driver_used = max(
        memory_before["driver_used_bytes"],
        memory_before_drain["driver_used_bytes"],
    )

    drain_started = time.perf_counter()
    captures = llm.take_activation_captures()
    torch.cuda.synchronize()
    drain_finished = time.perf_counter()
    memory_after_drain = memory_snapshot()

    expected_keys = {
        (layer, signal) for layer in layers for signal in ("attention_output", "post_residual")
    }
    expected_keys.add((None, "lm_head_input"))
    if capture:
        if {(c.layer_id, c.signal) for c in captures} != expected_keys:
            raise RuntimeError("Missing or unexpected capture signals")
        for layer, signal in expected_keys:
            reconstruct_capture(captures, layer, signal, 0, ids)
    elif captures:
        raise RuntimeError("Received captures while disabled")
    capture_bytes = sum(item.tensor.numel() * item.tensor.element_size() for item in captures)
    del captures

    mem_cleanup()
    memory_after_release = memory_snapshot()

    generation_seconds = generation_finished - started
    ttft_seconds = first_token_at - started
    decode_seconds = last_token_at - first_token_at
    total_tokens = len(ids) + generated_steps
    reached_eos = final_token_id == llm.eos_token_id

    # The core LLM returns outputs but does not report these timings or CUDA
    # allocator/driver snapshots, so the benchmark measures them at this boundary.
    row = {
        "prompt_index": index,
        "capture": capture,
        "input_tokens": len(ids),
        "generated_steps_including_terminal": generated_steps,
        "returned_token_count": len(result[0]["token_ids"]),
        "finish_reason": "eos" if reached_eos else "length",
        "reached_eos": reached_eos,
        "ttft_seconds": ttft_seconds,
        "prefill_tokens_per_second": len(ids) / ttft_seconds,
        "decode_tokens_per_second": (
            (generated_steps - 1) / decode_seconds if decode_seconds > 0 else float("inf")
        ),
        "generation_seconds": generation_seconds,
        "drain_seconds": drain_finished - drain_started,
        "end_to_end_tokens_per_second": total_tokens
        / (generation_seconds + drain_finished - drain_started),
        "capture_bytes": capture_bytes,
        "peak_driver_used_bytes": peak_driver_used,
        # Retain lifecycle snapshots to distinguish staged capture memory from
        # memory remaining after drain and explicit release.
        "memory_before": memory_before,
        "memory_before_drain": memory_before_drain,
        "memory_after_drain": memory_after_drain,
        "memory_after_release": memory_after_release,
        "result_text": result[0]["text"],
        "result_token_ids": result[0]["token_ids"],
    }
    print(
        f"[benchmark] prompt={index} capture={capture} input={len(ids)} "
        f"generated={generated_steps} finish={row['finish_reason']} "
        f"ttft={ttft_seconds:.6f}s decode_tps={row['decode_tokens_per_second']:.2f} "
        f"artifact={capture_bytes / 2**20:.2f}MiB",
        flush=True,
    )

    return row


def benchmark_model(
    max_tokens: int,
    model_id: str,
    cache_type: str,
    max_extend_tokens: int,
    max_running_req: int,
    max_seq_len: int,
    memory_ratio: float,
    use_pynccl: bool,
    layers: tuple[int, ...],
    workload: dict[str, Any],
) -> dict[str, Any]:

    import torch
    from minisgl.capture import ActivationCaptureConfig
    from minisgl.llm import LLM

    os.chdir(REMOTE_REPO)

    if max_tokens < 64:
        raise ValueError("max_tokens must be at least 64")

    llm = LLM(
        model_path=model_id,
        dtype=torch.float16,
        cache_type=cache_type,
        cuda_graph_max_bs=1,
        max_extend_tokens=max_extend_tokens,
        max_running_req=max_running_req,
        max_seq_len_override=max_seq_len,
        memory_ratio=memory_ratio,
        use_pynccl=use_pynccl,
        activation_capture=ActivationCaptureConfig(layer_ids=layers, capture_lm_head_input=True),
    )
    manager = llm.engine.capture_manager
    tokenizer = llm.tokenizer

    assert manager is not None
    assert tokenizer is not None

    prompt_ids = [prompt["token_ids"] for prompt in workload["prompts"]]
    if max(len(ids) for ids in prompt_ids) + max_tokens >= max_seq_len:
        raise RuntimeError("Prompt plus completion safety cap exceeds max sequence length")
    try:
        print("[benchmark] warming CUDA graph and both capture modes", flush=True)
        warmup_ids = tokenize_prompt(tokenizer, "Explain in one sentence what a tensor is.")
        run_once(manager, llm, -1, False, warmup_ids, max_tokens, layers, warmup=True)
        run_once(manager, llm, -1, True, warmup_ids, max_tokens, layers, warmup=True)

        rows: list[dict[str, Any]] = []
        for index, ids in enumerate(prompt_ids):
            # Alternate mode order across prompts to reduce systematic warm/order bias.
            order = (False, True) if index % 2 == 0 else (True, False)
            pair = [
                run_once(manager, llm, index, capture, ids, max_tokens, layers) for capture in order
            ]
            off = next(row for row in pair if not row["capture"])
            on = next(row for row in pair if row["capture"])
            if off["result_token_ids"] != on["result_token_ids"]:
                raise RuntimeError(f"Capture changed greedy output for prompt {index}")
            rows.extend(pair)

        off_summary = summarize_medians(rows, False)
        on_summary = summarize_medians(rows, True)
        pairs = []

        for index in range(len(prompt_ids)):
            off = next(row for row in rows if row["prompt_index"] == index and not row["capture"])
            on = next(row for row in rows if row["prompt_index"] == index and row["capture"])
            pairs.append(
                {
                    "prompt_index": index,
                    "input_tokens": on["input_tokens"],
                    "generated_steps": on["generated_steps_including_terminal"],
                    "finish_reason": on["finish_reason"],
                    "ttft_overhead_percent": (on["ttft_seconds"] / off["ttft_seconds"] - 1) * 100,
                    "decode_tps_change_percent": (
                        on["decode_tokens_per_second"] / off["decode_tokens_per_second"] - 1
                    )
                    * 100,
                    "peak_driver_used_delta_bytes": (
                        on["peak_driver_used_bytes"] - off["peak_driver_used_bytes"]
                    ),
                    "capture_bytes": on["capture_bytes"],
                }
            )

        result = {
            "model": model_id,
            "gpu": torch.cuda.get_device_name(0),
            "capture_layers": layers,
            "capture_signals": ("attention_output", "post_residual", "lm_head_input"),
            "capture_scope": "prefill_only",
            "max_tokens_safety_cap": max_tokens,
            "workload": workload,
            "prompts": workload["prompts"],
            "rows": rows,
            "pairs": pairs,
            "median_capture_off": off_summary,
            "median_capture_on": on_summary,
            "median_paired": {
                key: statistics.median(float(pair[key]) for pair in pairs)
                for key in (
                    "ttft_overhead_percent",
                    "decode_tps_change_percent",
                    "peak_driver_used_delta_bytes",
                    "capture_bytes",
                )
            },
            "all_outputs_identical": True,
            "all_reached_eos": all(pair["finish_reason"] == "eos" for pair in pairs),
        }
        return result
    finally:
        # LLM does not currently implement __enter__/__exit__; this guarantees
        # engine teardown on success and on any failed measurement.
        llm.shutdown()


def validate_model(model_id: str, layers: tuple[int, ...], workload: dict[str, Any]) -> dict:
    import torch
    import transformers
    from minisgl.capture import ActivationCaptureConfig
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM
    from transformers import AutoModelForCausalLM

    # Aggregate and per-token checks tolerate FP16 kernel/reduction differences.
    thresholds = {"relative_l2_max": 0.05, "cosine_min": 0.999, "minimum_token_cosine_min": 0.98}
    samples = workload["prompts"]
    cases = [(f"dataset_row_{p['dataset_row']}", [p["token_ids"]]) for p in samples]
    # Reuse two real prompts to exercise request/span separation in a batch.
    shortest = sorted(samples, key=lambda p: p["input_tokens"])[:2]
    cases.append(("two_requests", [p["token_ids"] for p in shortest]))
    llm = LLM(
        model_path=model_id,
        dtype=torch.float16,
        cache_type="naive",
        cuda_graph_max_bs=1,
        max_extend_tokens=256,
        max_running_req=2,
        max_seq_len_override=4096,
        memory_ratio=0.45,
        use_pynccl=False,
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
        reference = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                attn_implementation="eager",
            )
            .eval()
            .to("cuda")
        )
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
            handles.append(
                block.self_attn.o_proj.register_forward_hook(hook((layer, "attention_output")))
            )

        rows = []
        metadata = []
        for name, prompts, outputs, captures in saved:
            expected_keys = {
                (layer, signal)
                for layer in layers
                for signal in ("attention_output", "post_residual")
            }
            expected_keys.add((None, "lm_head_input"))
            assert {(c.layer_id, c.signal) for c in captures} == expected_keys
            metadata.append(
                {
                    "case": name,
                    "input_tokens": [len(p) for p in prompts],
                    "capture_records": len(captures),
                    "spans": [
                        asdict(s)
                        for c in captures
                        if c.layer_id == 0 and c.signal == "post_residual"
                        for s in c.spans
                    ],
                }
            )
            for request_id, ids in enumerate(prompts):
                hooked.clear()
                with torch.inference_mode():
                    result = reference(
                        input_ids=torch.tensor([ids], device="cuda"),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                assert set(hooked) == expected_keys
                first_token_matches = (
                    int(result.logits[0, -1].argmax()) == outputs[request_id]["token_ids"][0]
                )
                del result
                for layer, signal in sorted(
                    expected_keys, key=lambda key: (-1 if key[0] is None else key[0], key[1])
                ):
                    actual = reconstruct_capture(captures, layer, signal, request_id, ids)
                    expected = hooked[(layer, signal)].float()
                    assert (
                        actual.shape == expected.shape == (len(ids), reference.config.hidden_size)
                    )
                    metrics = compare_activations(actual, expected, thresholds)
                    metrics["passed"] &= first_token_matches
                    rows.append(
                        {
                            "case": name,
                            "request_id": request_id,
                            "layer": layer,
                            "signal": signal,
                            "shape": list(actual.shape),
                            **metrics,
                            "hf_first_token_matches": first_token_matches,
                        }
                    )
                print(f"[validate] compared {name} request={request_id}", flush=True)
        for handle in handles:
            handle.remove()
        return {
            "model": model_id,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "float16",
            "hf_attention": "eager",
            "metric_dtype": "float64",
            "layers": layers,
            "thresholds": thresholds,
            "workload": workload,
            "cases": metadata,
            "rows": rows,
            "capture_on_off_outputs_identical": True,
            "all_passed": all(r["passed"] for r in rows),
        }
    finally:
        llm.shutdown()


@app.function(
    image=dependency_image,
    memory=8192,
    timeout=1200,
    volumes={str(REMOTE_CACHE): cache_volume},
)
def prepare_workload(model_id: str, dataset_options: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    workload = load_wildchat_prompts(AutoTokenizer.from_pretrained(model_id), **dataset_options)
    print(
        f"[dataset] {workload['dataset']} revision={workload['revision']} "
        f"rows={[p['dataset_row'] for p in workload['prompts']]} "
        f"tokens={[p['input_tokens'] for p in workload['prompts']]}",
        flush=True,
    )
    return workload


# Each phase uses a separate container so allocator and CUDA graph state cannot
# contaminate benchmark timings or take VRAM away from the HF reference.
@app.function(
    image=dependency_image,
    gpu="T4",
    memory=8192,
    timeout=1200,
    volumes={str(REMOTE_CACHE): cache_volume},
    single_use_containers=True,
)
def run_benchmark(mode: str, model_id: str, workload: dict[str, Any], **kwargs) -> dict[str, Any]:
    from transformers import AutoConfig

    layers = capture_layers(AutoConfig.from_pretrained(model_id).num_hidden_layers)
    if mode == "validate":
        return validate_model(model_id, layers, workload)
    if mode == "benchmark":
        return benchmark_model(model_id=model_id, layers=layers, workload=workload, **kwargs)
    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture on/off benchmark and HF activation validation on Modal"
    )
    parser.add_argument("--num-prompts", type=int, default=50, help="WildChat samples; at least 4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision", default="c827c6df8fcf008219ffaffa4d1dd77491099367")
    parser.add_argument("--dataset-scan-rows", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument(
        "--mode", choices=("both", "benchmark", "validate", "sample"), default="both"
    )
    parser.add_argument(
        "--hf-secret", help="Existing Modal secret containing HF_TOKEN for gated models"
    )
    parser.add_argument("--gpu", default="T4", help="Modal GPU type, e.g. T4, L4, A10G")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cpu", type=float, default=2.0)
    parser.add_argument("--memory", type=int, default=8192, help="Container memory in MiB")
    parser.add_argument("--timeout", type=int, default=20 * 60, help="Seconds")
    parser.add_argument("--max-tokens", type=int, default=250, help="Generation safety cap")

    parser.add_argument(
        "--cache-type",
        choices=("naive", "radix"),
        default="naive",
        help="Prefix-cache implementation",
    )
    parser.add_argument(
        "--max-extend-tokens", type=int, default=3072, help="Maximum prefill extension size"
    )
    parser.add_argument(
        "--max-running-req", type=int, default=1, help="Maximum concurrent requests"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=4096, help="Maximum prompt plus generation length"
    )
    parser.add_argument(
        "--memory-ratio",
        type=float,
        default=0.75,
        help="Fraction of GPU memory available to the engine",
    )
    parser.add_argument(
        "--use-nccl", action=argparse.BooleanOptionalAction, default=False, help="Enable PyNCCL"
    )

    args = parser.parse_args()
    if args.num_prompts < 4:
        parser.error("--num-prompts must be at least 4")
    if args.dataset_scan_rows < args.num_prompts:
        parser.error("--dataset-scan-rows must be at least --num-prompts")
    if not 2049 <= args.max_input_tokens <= 4092:
        parser.error("--max-input-tokens must be between 2049 and 4092")
    if (
        args.mode in ("both", "benchmark")
        and args.max_input_tokens + args.max_tokens >= args.max_seq_len
    ):
        parser.error("--max-input-tokens plus --max-tokens must be below --max-seq-len")
    return args


def main() -> None:
    args = parse_args()
    options = {"gpu": args.gpu, "cpu": args.cpu, "memory": args.memory, "timeout": args.timeout}
    if args.hf_secret:
        options["secrets"] = [modal.Secret.from_name(args.hf_secret, required_keys=["HF_TOKEN"])]
    else:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            options["secrets"] = [modal.Secret.from_dict({"HF_TOKEN": token})]
    runner = run_benchmark.with_options(**options)
    modes = ("benchmark", "validate") if args.mode == "both" else (args.mode,)
    failed = False
    with modal.enable_output(), app.run(detach=True):
        sampler = prepare_workload.with_options(**{k: v for k, v in options.items() if k != "gpu"})
        workload = sampler.spawn(
            args.model_id,
            {
                "count": args.num_prompts,
                "seed": args.seed,
                "revision": args.dataset_revision,
                "max_tokens": args.max_input_tokens,
                "scan_rows": args.dataset_scan_rows,
            },
        ).get()
        manifest_path = write_run_artifact(
            {"model": args.model_id, "workload": workload},
            LOCAL_REPO / "artifacts" / "modal-traces",
            "wildchat-prompts",
        )
        print(f"Prompt manifest: {manifest_path}")
        if args.mode == "sample":
            return
        for mode in modes:
            kwargs = (
                {}
                if mode == "validate"
                else {
                    "max_tokens": args.max_tokens,
                    "cache_type": args.cache_type,
                    "max_extend_tokens": args.max_extend_tokens,
                    "max_running_req": args.max_running_req,
                    "max_seq_len": args.max_seq_len,
                    "memory_ratio": args.memory_ratio,
                    "use_pynccl": args.use_nccl,
                }
            )
            kwargs["workload"] = workload
            result = runner.spawn(mode=mode, model_id=args.model_id, **kwargs).get()
            path = write_run_artifact(result, LOCAL_REPO / "artifacts" / "modal-traces", mode)
            print(f"Results: {path}")
            if mode == "validate":
                failed |= not result["all_passed"]
                summary = {
                    "all_passed": result["all_passed"],
                    "comparisons": len(result["rows"]),
                    "failures": [r for r in result["rows"] if not r["passed"]],
                }
            else:
                summary = {
                    key: value
                    for key, value in result.items()
                    if key not in ("rows", "prompts", "pairs", "workload")
                }
            print(json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
