"""Paired capture/memory benchmark on ten realistic prompts.

Each prompt is run once with prefill activation capture disabled and once with
capture enabled. Generation stops at EOS, subject to a configurable safety cap.
The two modes share one initialized engine, while the naive KV-cache manager
prevents prefix-cache reuse between runs.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import modal

from modal_smoke import (
    CAPTURE_LAYERS,
    LOCAL_REPO,
    MODEL_ID,
    REMOTE_CACHE,
    REMOTE_REPO,
    cache_volume,
    dependency_image,
)

app = modal.App("mini-sglang-capture-benchmark")

# These are deliberately not one-line trivia questions. They cover different
# output shapes and give the small model enough room to terminate naturally.
PROMPTS = (
    """You are reviewing an inference server used for mechanistic-interpretability experiments. It batches requests, performs a prefill pass, stores keys and values in a paged cache, and then uses a CUDA graph for token-by-token decoding. Explain, for an engineer who knows PyTorch but has never written an inference engine, how a single request moves through this system. Distinguish the residual stream, attention output, KV cache, page table, CUDA stream, and CUDA graph. Then identify three places where copying activations could silently change performance measurements. Finish with a compact checklist for validating that an activation-capture implementation is numerically faithful.""",
    """A research group wants to compare two activation-extraction designs for a 600-million-parameter transformer. Design A clones five layers of residual-stream and attention-output tensors on the GPU during prefill, keeps them alive until generation ends, and copies them to ordinary CPU memory afterward. Design B uses a bounded pool of GPU staging buffers, asynchronous copies into pinned CPU buffers, and a background writer. Analyze the expected latency, peak VRAM, host-memory, synchronization, and failure-mode differences. Explain when the simpler design is scientifically adequate and when the pipeline is necessary. State any assumptions and give concrete measurements the group should collect instead of relying on intuition.""",
    """Consider a decoder-only transformer serving one long request at a time. The model uses grouped-query attention, rotary positional embeddings, RMSNorm, gated MLPs, a paged KV cache, FlashInfer during decode, and FlashAttention during prefill. Give a careful walkthrough of tensor shapes for a hypothetical model with hidden size 1024, 16 query heads, 8 key/value heads, and head dimension 64. Follow 768 prompt tokens through one layer during prefill and then follow one new token during decode. Explain which tensors grow with sequence length, which are temporary, and which must persist for later tokens.""",
    """Suppose an activation dataset will be used to train sparse autoencoders on a small language model. The dataset must preserve enough provenance to connect every saved vector to its original request, token position, model layer, signal type, and inference phase. Propose a shard format and metadata schema, including completion markers and checksums. Discuss how to handle partial writes, process crashes, duplicated samples, tokenizer revisions, model revisions, padding, prefix-cache hits, and tensor dtype. End by describing a validation program that can detect both corrupted files and semantically misaligned activations.""",
    """An engineer reports that enabling activation capture changes decode throughput by less than one percent, even though the system retains tens of megabytes of additional GPU tensors. Evaluate whether that result is plausible. Separate memory capacity, memory bandwidth, allocator behavior, kernel scheduling, synchronization, and host-transfer costs. Explain why prefill-only capture may have little effect on steady-state decode, and describe an experiment capable of revealing hidden costs that a short fixed-length generation would miss. Include the GPU memory statistics and timing boundaries you would record.""",
    """Compare ordinary eager PyTorch execution with CUDA graph replay for autoregressive decoding. Explain what gets recorded during graph capture, why stable tensor addresses and static shapes matter, what Python does and does not execute during replay, and how an inference engine handles variable batch sizes. Then analyze four ways an observability feature could coexist with graph replay: prefill-only capture, copying into static graph-owned buffers, launching work after a replay event, and periodically falling back to eager execution. Give the tradeoffs without assuming that CUDA graphs must always be disabled.""",
    """You are debugging a deterministic greedy-generation benchmark. Capture-on and capture-off runs sometimes emit identical text but different token counts, and reported time-to-first-token varies by twenty percent. Produce a systematic debugging plan. Consider EOS accounting, whether the final token is returned to the caller, tokenizer chat templates, warmup, graph capture, asynchronous device work, Python callbacks, prefix caching, allocator state, and alternating run order. Specify invariants and logging fields that would let another researcher reproduce and audit every timing number.""",
    """Explain tensor parallelism for transformer inference to an engineer accustomed to a single GPU. Use an MLP and a multi-head attention layer to show which matrices are commonly split by columns or rows, where collective communication occurs, and why tensor parallelism is not the same thing as assigning different layers to different GPUs. Discuss prefill and decode separately, including how KV-cache heads are placed, what rank zero typically coordinates, and why activation extraction needs an explicit policy for gathering or sharding captured tensors.""",
    """A team wants to simplify a research inference server by removing abstractions that were designed for multi-GPU production serving. Propose a principled path to a single-GPU research runtime without accidentally deleting correctness-critical behavior. Classify components into model execution, request scheduling, tokenization, KV-cache allocation, attention metadata, CUDA graph management, sampling, and distributed coordination. For each category say what can be collapsed, what should remain isolated behind a small interface, and what tests must pass before and after the refactor. Favor clarity, but account for performance regressions.""",
    """Write a technical note explaining the difference between capturing an attention module's projected output, the residual stream immediately after attention, the MLP intermediate activation after its nonlinearity and gate, and the residual stream after the full transformer block. Describe what scientific questions each signal can answer and which signals are easy or difficult to obtain from fused kernels. Include the distinction between attention probabilities and attention outputs, explain why reconstructing probabilities may require queries and keys, and recommend a minimal first dataset for an interpretability experiment on five selected layers.""",
)

ANSWER_CONSTRAINT = (
    "Write a complete answer of roughly 250 to 400 words. After the conclusion, "
    "stop rather than repeating or extending the answer."
)
FULL_PROMPTS = tuple(f"{prompt}\n\n{ANSWER_CONSTRAINT}" for prompt in PROMPTS)

SOURCE_GROUPS = (
    ("python/minisgl/scheduler/scheduler.py", "python/minisgl/scheduler/prefill.py"),
    ("python/minisgl/capture.py", "python/minisgl/models/qwen3.py", "python/minisgl/models/utils.py"),
    ("python/minisgl/attention/fi.py", "python/minisgl/attention/fa3.py"),
    ("python/minisgl/engine/engine.py", "python/minisgl/engine/graph.py"),
    ("python/minisgl/core.py", "python/minisgl/scheduler/cache.py", "python/minisgl/scheduler/table.py"),
    ("python/minisgl/engine/graph.py", "python/minisgl/engine/engine.py", "python/minisgl/core.py"),
    ("python/minisgl/llm/llm.py", "python/minisgl/scheduler/scheduler.py"),
    (
        "python/minisgl/distributed/impl.py",
        "python/minisgl/layers/linear.py",
        "python/minisgl/models/weight.py",
    ),
    ("python/minisgl/scheduler/prefill.py", "python/minisgl/scheduler/decode.py", "python/minisgl/engine/config.py"),
    ("python/minisgl/models/qwen3.py", "python/minisgl/models/utils.py", "python/minisgl/layers/attention.py"),
)


def _median(rows: list[dict[str, Any]], key: str) -> float:
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
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_driver_used_bytes",
    )
    return {key: _median(selected, key) for key in keys}


@app.function(
    image=dependency_image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=20 * 60,
    volumes={str(REMOTE_CACHE): cache_volume},
)
def run_benchmark(max_tokens: int) -> dict[str, Any]:
    os.chdir(REMOTE_REPO)

    import torch
    from minisgl.capture import ActivationCaptureConfig
    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    if max_tokens < 64:
        raise ValueError("max_tokens must be at least 64")

    llm = LLM(
        model_path=MODEL_ID,
        dtype=torch.float16,
        cache_type="naive",
        cuda_graph_max_bs=1,
        max_extend_tokens=3072,
        max_running_req=1,
        max_seq_len_override=6144,
        memory_ratio=0.60,
        use_pynccl=False,
        activation_capture=ActivationCaptureConfig(layer_ids=CAPTURE_LAYERS),
    )
    manager = llm.engine.capture_manager
    assert manager is not None

    def format_prompt(user_prompt: str) -> list[int]:
        ids = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return list(ids)

    def build_long_prompt(question: str, source_files: tuple[str, ...]) -> tuple[str, list[int]]:
        source_files = source_files + ("README.md",)
        source = "\n\n".join(
            f"===== {relative_path} =====\n"
            f"{(Path(REMOTE_REPO) / relative_path).read_text(encoding='utf-8')}"
            for relative_path in source_files
        )
        source_ids = llm.tokenizer.encode(source, add_special_tokens=False)
        prefix = (
            "Use the following Mini-SGLang source as concrete context. The excerpt may end "
            "mid-file because the benchmark fixes the total prompt length.\n\n"
            "<source>\n"
        )
        suffix = f"\n</source>\n\nTASK:\n{question}"

        # Find the largest source prefix whose complete chat-templated prompt is
        # no longer than 2,048 tokens. Tokenizer decode/re-encode is not assumed
        # to preserve length, hence the binary search over the actual template.
        low, high = 0, len(source_ids)
        best_prompt = prefix + suffix
        best_ids = format_prompt(best_prompt)
        while low <= high:
            middle = (low + high) // 2
            excerpt = llm.tokenizer.decode(source_ids[:middle], skip_special_tokens=False)
            candidate_prompt = prefix + excerpt + suffix
            candidate_ids = format_prompt(candidate_prompt)
            if len(candidate_ids) <= 2048:
                best_prompt, best_ids = candidate_prompt, candidate_ids
                low = middle + 1
            else:
                high = middle - 1
        if len(best_ids) < 2035:
            raise RuntimeError(f"Could only construct a {len(best_ids)}-token long prompt")
        return best_prompt, best_ids

    built_prompts = [
        build_long_prompt(question, source_files)
        for question, source_files in zip(FULL_PROMPTS, SOURCE_GROUPS)
    ]
    long_prompts = [item[0] for item in built_prompts]
    prompt_ids = [item[1] for item in built_prompts]
    if max(len(ids) for ids in prompt_ids) + max_tokens >= 6144:
        raise RuntimeError("Prompt plus completion safety cap exceeds max sequence length")

    def memory_snapshot() -> dict[str, int]:
        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "driver_free_bytes": free,
            "driver_used_bytes": total - free,
            "driver_total_bytes": total,
        }

    def run_once(index: int, capture: bool, ids: list[int], warmup: bool = False) -> dict[str, Any]:
        manager.enabled = capture
        if manager.drain():
            raise RuntimeError("Capture queue was not empty before iteration")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory_before = memory_snapshot()

        first_token_at: float | None = None
        last_token_at: float | None = None
        generated_steps = 0
        final_token_id: int | None = None
        original_send_result = llm.send_result

        def timed_send_result(reply: Any) -> None:
            nonlocal first_token_at, last_token_at, generated_steps, final_token_id
            now = time.perf_counter()
            for message in reply.data:
                generated_steps += 1
                final_token_id = int(message.next_token)
                if first_token_at is None:
                    first_token_at = now
                last_token_at = now
            original_send_result(reply)

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
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
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
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        memory_after_release = memory_snapshot()

        generation_seconds = generation_finished - started
        ttft_seconds = first_token_at - started
        decode_seconds = last_token_at - first_token_at
        total_tokens = len(ids) + generated_steps
        reached_eos = final_token_id == llm.eos_token_id
        row = {
            "prompt_index": index,
            "capture": capture,
            "input_tokens": len(ids),
            "generated_steps_including_terminal": generated_steps,
            "returned_token_count": len(result[0]["token_ids"]),
            "final_token_id": final_token_id,
            "eos_token_id": llm.eos_token_id,
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
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_driver_used_bytes": peak_driver_used,
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
            f"peak_allocated={peak_allocated / 2**20:.2f}MiB "
            f"artifact={capture_bytes / 2**20:.2f}MiB",
            flush=True,
        )
        return row

    try:
        print("[benchmark] warming CUDA graph and both capture modes", flush=True)
        warmup_ids = format_prompt("Explain in one sentence what a tensor is.")
        run_once(-1, False, warmup_ids, warmup=True)
        run_once(-1, True, warmup_ids, warmup=True)

        rows: list[dict[str, Any]] = []
        for index, ids in enumerate(prompt_ids):
            order = (False, True) if index % 2 == 0 else (True, False)
            pair = [run_once(index, capture, ids) for capture in order]
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
                    "peak_allocated_delta_bytes": (
                        on["peak_allocated_bytes"] - off["peak_allocated_bytes"]
                    ),
                    "peak_reserved_delta_bytes": (
                        on["peak_reserved_bytes"] - off["peak_reserved_bytes"]
                    ),
                    "peak_driver_used_delta_bytes": (
                        on["peak_driver_used_bytes"] - off["peak_driver_used_bytes"]
                    ),
                    "capture_bytes": on["capture_bytes"],
                }
            )

        return {
            "model": MODEL_ID,
            "gpu": torch.cuda.get_device_name(0),
            "capture_layers": CAPTURE_LAYERS,
            "capture_signals": ("attention_output", "post_residual"),
            "capture_scope": "prefill_only",
            "max_tokens_safety_cap": max_tokens,
            "prompts": [
                {"index": i, "user_prompt": prompt, "input_tokens": len(prompt_ids[i])}
                for i, prompt in enumerate(long_prompts)
            ],
            "rows": rows,
            "pairs": pairs,
            "median_capture_off": off_summary,
            "median_capture_on": on_summary,
            "median_paired": {
                key: _median(pairs, key)
                for key in (
                    "ttft_overhead_percent",
                    "decode_tps_change_percent",
                    "peak_allocated_delta_bytes",
                    "peak_reserved_delta_bytes",
                    "peak_driver_used_delta_bytes",
                    "capture_bytes",
                )
            },
            "all_outputs_identical": True,
            "all_reached_eos": all(pair["finish_reason"] == "eos" for pair in pairs),
        }
    finally:
        llm.shutdown()


@app.local_entrypoint()
def main(max_tokens: int = 3072) -> None:
    result = run_benchmark.remote(max_tokens=max_tokens)
    output_dir = LOCAL_REPO / "artifacts" / "modal-traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_path = output_dir / f"benchmark-10-prompts-{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    prompt_path = output_dir / f"benchmark-10-prompts-{timestamp}.prompts.txt"
    prompt_path.write_text(
        "\n\n".join(
            f"===== PROMPT {prompt['index']} ({prompt['input_tokens']} tokens) =====\n"
            f"{prompt['user_prompt']}"
            for prompt in result["prompts"]
        ),
        encoding="utf-8",
    )

    summary = {key: value for key, value in result.items() if key not in ("rows", "prompts")}
    print(json.dumps(summary, indent=2))
    print(f"\nExact prompts: {prompt_path}")
    print(f"Raw benchmark: {output_path}")
