"""Paired capture/memory benchmark on ten basic QA prompts.

Each prompt is run once with prefill activation capture disabled and once with
capture enabled. Generation stops at EOS, subject to --max-tokens.

Usage:
    python modal_benchmark.py --gpu T4 --max-tokens 512
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from minisgl.llm import LLM  # type: ignore

from config import (
    CAPTURE_LAYERS,
    LOCAL_REPO,
    PROMPTS,
    REMOTE_CACHE,
    REMOTE_REPO,
    app,
    cache_volume,
    dependency_image,
)
from utils import mem_cleanup, memory_snapshot, tokenize_prompt


def _median(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError(f"Cannot summarize {key}: no benchmark rows")
    return statistics.median(float(row[key]) for row in rows)


def _summarize(rows: list[dict[str, Any]], capture: bool) -> dict[str, float]:
    selected = [row for row in rows if row["capture"] is capture]
    keys = (
        "ttft_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "generation_seconds",
        "drain_seconds",
        "end_to_end_tokens_per_second",
        "peak_driver_used_bytes",
    )
    return {key: _median(selected, key) for key in keys}


def run_once(
    manager,
    llm: LLM,
    index: int,
    capture: bool,
    ids: list[int],
    max_tokens: int,
    warmup: bool = False,
) -> dict[str, Any]:
    import torch
    from minisgl.core import SamplingParams #type:ignore

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

    expected_captures = 2 * len(CAPTURE_LAYERS) if capture else 0
    if len(captures) != expected_captures:
        raise RuntimeError(f"Expected {expected_captures} captures, received {len(captures)}")
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

# Defaults cover direct Modal invocation; CLI options supply per-run configuration.
@app.function(image=dependency_image, gpu="T4", volumes={str(REMOTE_CACHE): cache_volume})
def run_benchmark(
    max_tokens: int,
    model_id: str,
    cache_type: str,
    max_extend_tokens: int,
    max_running_req: int,
    max_seq_len: int,
    memory_ratio: float,
    use_pynccl: bool,
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
        activation_capture=ActivationCaptureConfig(layer_ids=CAPTURE_LAYERS),
    )
    manager = llm.engine.capture_manager
    tokenizer = llm.tokenizer

    assert manager is not None
    assert tokenizer is not None

    prompt_ids = [tokenize_prompt(tokenizer, prompt) for prompt in PROMPTS]
    if max(len(ids) for ids in prompt_ids) + max_tokens >= max_seq_len:
        raise RuntimeError("Prompt plus completion safety cap exceeds max sequence length")
    try:
        print("[benchmark] warming CUDA graph and both capture modes", flush=True)
        warmup_ids = tokenize_prompt(tokenizer, "Explain in one sentence what a tensor is.")
        run_once(manager, llm, -1, False, warmup_ids, max_tokens, warmup=True)
        run_once(manager, llm, -1, True, warmup_ids, max_tokens, warmup=True)

        rows: list[dict[str, Any]] = []
        for index, ids in enumerate(prompt_ids):
            # Alternate mode order across prompts to reduce systematic warm/order bias.
            order = (False, True) if index % 2 == 0 else (True, False)
            pair = [run_once(manager, llm, index, capture, ids, max_tokens) for capture in order]
            off = next(row for row in pair if not row["capture"])
            on = next(row for row in pair if row["capture"])
            if off["result_token_ids"] != on["result_token_ids"]:
                raise RuntimeError(f"Capture changed greedy output for prompt {index}")
            rows.extend(pair)

        off_summary = _summarize(rows, False)
        on_summary = _summarize(rows, True)
        pairs = []

        for index in range(len(PROMPTS)):
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
                    ) * 100,
                    "peak_driver_used_delta_bytes": (
                        on["peak_driver_used_bytes"] - off["peak_driver_used_bytes"]
                    ),
                    "capture_bytes": on["capture_bytes"],
                }
            )

        result = {
            "model": model_id,
            "gpu": torch.cuda.get_device_name(0),
            "capture_layers": CAPTURE_LAYERS,
            "capture_signals": ("attention_output", "post_residual"),
            "capture_scope": "prefill_only",
            "max_tokens_safety_cap": max_tokens,
            "prompts": [
                {"index": i, "user_prompt": prompt, "input_tokens": len(prompt_ids[i])}
                for i, prompt in enumerate(PROMPTS)
            ],
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture on/off benchmark on Modal")
    parser.add_argument("--gpu", default="T4", help="Modal GPU type, e.g. T4, L4, A10G")
    parser.add_argument('--model-id', type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cpu", type=float, default=2.0)
    parser.add_argument("--memory", type=int, default=8192, help="Container memory in MiB")
    parser.add_argument("--timeout", type=int, default=20 * 60, help="Seconds")
    parser.add_argument("--max-tokens", type=int, default=512, help="Generation safety cap")

    parser.add_argument("--cache-type", type=str, default="naive", help="KV cache implementation")
    parser.add_argument("--max-extend-tokens", type=int, default=3072, help="Maximum prefill extension size")
    parser.add_argument("--max-running-req", type=int, default=1, help="Maximum concurrent requests")
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum prompt plus generation length")
    parser.add_argument("--memory-ratio", type=float, default=0.75, help="Fraction of GPU memory available to the engine")
    parser.add_argument("--use-nccl", action=argparse.BooleanOptionalAction, default=False, help="Enable PyNCCL")

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    benchmark = run_benchmark.with_options(
        gpu=args.gpu, cpu=args.cpu, memory=args.memory, timeout=args.timeout
    )

    # Detached so the run survives a dropped client; spawn().get() waits for the result.
    with modal.enable_output(), app.run(detach=True):
        result = benchmark.spawn(
            max_tokens=args.max_tokens,
            model_id=args.model_id,
            cache_type=args.cache_type, 
            max_extend_tokens=args.max_extend_tokens,
            max_running_req=args.max_running_req,
            max_seq_len=args.max_seq_len,
            memory_ratio=args.memory_ratio,
            use_pynccl=args.use_nccl,
        ).get()

    output_dir = LOCAL_REPO / "artifacts" / "modal-traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_path = output_dir / f"benchmark-qa-{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {key: value for key, value in result.items() if key not in ("rows", "prompts")}
    print(json.dumps(summary, indent=2))
    print(f"\nRaw benchmark: {output_path}")

if __name__ == "__main__":
    main()
