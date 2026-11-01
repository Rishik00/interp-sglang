import gc
import statistics
from typing import Any


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def summarize(rows: list[dict[str, Any]], capture: bool) -> dict[str, float]:
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
    return {key: mean(selected, key) for key in keys}


def memory_snapshot() -> dict[str, int]:
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "driver_free_bytes": free,
        "driver_used_bytes": total - free,
        "driver_total_bytes": total,
    }


def tokenize_prompt(tokenizer, user_prompt: str) -> list[int]:
    if not tokenizer.chat_template:
        return list(tokenizer.encode(user_prompt))
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(ids)


def mem_cleanup():
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def median(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        raise ValueError(f"Cannot summarize {key}: no benchmark rows")
    values = [float(row[key]) for row in rows if row[key] is not None]
    return statistics.median(values) if values else None


def summarize_medians(rows: list[dict[str, Any]], capture: bool) -> dict[str, float | None]:
    selected = [row for row in rows if row["capture"] is capture]
    keys = (
        "ttft_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "generation_seconds",
        "drain_seconds",
        "end_to_end_tokens_per_second",
        "peak_driver_used_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    return {key: median(selected, key) for key in keys}


def capture_layers(num_layers: int) -> tuple[int, ...]:
    """Sample six depths, including the first and last decoder blocks."""
    if num_layers < 1:
        raise ValueError("Model must have at least one layer")
    return tuple(
        sorted({0, 1 if num_layers > 1 else 0, *((num_layers - 1) * i // 4 for i in range(1, 5))})
    )


def write_run_artifact(result: dict[str, Any], output_dir, prefix: str):
    import json
    import time

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    model = result["model"].replace("/", "--")
    path = output_dir / f"{prefix}-{model}-{timestamp}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def compare_activations(actual, expected, thresholds: dict[str, float]) -> dict[str, Any]:
    """Use FP64 reductions on drained CPU tensors for stable error metrics."""
    import torch
    import torch.nn.functional as F

    actual, expected = actual.double(), expected.double()
    if actual.shape != expected.shape:
        raise ValueError(f"Activation shapes differ: {actual.shape} != {expected.shape}")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    nonzero = bool(actual.norm() > 0 and actual.std() > 0)
    difference = actual - expected
    relative_l2 = float(difference.norm() / expected.norm().clamp_min(1e-12))
    cosine = float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
    token_cosine = float(F.cosine_similarity(actual, expected, dim=-1).min())
    return {
        "finite": finite,
        "nonzero": nonzero,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "minimum_token_cosine": token_cosine,
        "max_absolute_error": float(difference.abs().max()),
        "passed": (
            finite
            and nonzero
            and relative_l2 <= thresholds["relative_l2_max"]
            and cosine >= thresholds["cosine_min"]
            and token_cosine >= thresholds["minimum_token_cosine_min"]
        ),
    }


WILDCHAT_ID = "allenai/WildChat-4.8M"
WILDCHAT_SHARD = "data/train-00000-of-00086.parquet"


def sample_wildchat_rows(rows, tokenizer, count: int, seed: int, max_tokens: int):
    """Length-stratified sample of first-user turns; never truncate prompts."""
    import hashlib
    import random

    if count < 4:
        raise ValueError("num_prompts must be at least 4")
    if max_tokens < 2049:
        raise ValueError("Allow at least 2049 input tokens to test natural long prompts")
    bounds = (256, 1024, 2048, max_tokens)
    buckets = [[] for _ in bounds]
    seen = set()
    scanned = 0
    for row_index, row in enumerate(rows):
        scanned += 1
        turn = next((t for t in (row.get("conversation") or []) if t.get("role") == "user"), None)
        if turn is None:
            continue
        text = (turn.get("content") or "").strip()
        if (
            not text
            or turn.get("language", row.get("language")) not in {"English", "Chinese"}
            or any(bool(obj.get(flag)) for obj in (row, turn) for flag in ("redacted", "toxic"))
        ):
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        ids = tokenize_prompt(tokenizer, text)
        if not ids or len(ids) > max_tokens:
            continue
        bucket = next(i for i, bound in enumerate(bounds) if len(ids) <= bound)
        buckets[bucket].append(
            {
                "dataset_row": row_index,
                "conversation_hash": row.get("conversation_hash"),
                "prompt_sha256": digest,
                "user_prompt": text,
                "token_ids": ids,
                "input_tokens": len(ids),
                "length_bucket": bucket,
            }
        )
    rng = random.Random(seed)
    selected = []
    for bucket, candidates in enumerate(buckets):
        # Spread any remainder across the first buckets: 50 -> 13, 13, 12, 12.
        needed = count // len(bounds) + (bucket < count % len(bounds))
        if len(candidates) < needed:
            raise ValueError(
                f"Length bucket {bucket} has only {len(candidates)} candidates; "
                f"need {needed}. Increase --dataset-scan-rows."
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:needed])
    rng.shuffle(selected)
    return selected, scanned


def load_wildchat_prompts(
    tokenizer, count: int, seed: int, revision: str, max_tokens: int, scan_rows: int
):
    from itertools import islice

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(WILDCHAT_ID, WILDCHAT_SHARD, repo_type="dataset", revision=revision)
    parquet = pq.ParquetFile(path)
    columns = [
        c
        for c in ("conversation", "conversation_hash", "language", "redacted", "toxic")
        if c in parquet.schema_arrow.names
    ]
    rows = (
        row
        for batch in parquet.iter_batches(batch_size=256, columns=columns)
        for row in batch.to_pylist()
    )
    prompts, scanned = sample_wildchat_rows(
        islice(rows, scan_rows), tokenizer, count, seed, max_tokens
    )
    return {
        "dataset": WILDCHAT_ID,
        "revision": revision,
        "split": "train",
        "shard": WILDCHAT_SHARD,
        "seed": seed,
        "scanned_rows": scanned,
        "sample_count": count,
        "selection": "equal samples from token ranges 1-256, 257-1024, 1025-2048, 2049-max",
        "max_input_tokens": max_tokens,
        "truncated": False,
        "filters": {
            "languages": ["English", "Chinese"],
            "exclude_redacted": True,
            "exclude_toxic": True,
            "turn": "first_user",
            "deduplicate": "prompt_sha256",
        },
        "prompts": prompts,
    }


def reconstruct_capture(captures, layer, signal, request_id: int, ids: list[int]):
    """Validate capture provenance and concatenate prefill chunks for one request."""
    import torch

    pieces = []
    for capture in captures:
        if (capture.layer_id, capture.signal) != (layer, signal):
            continue
        if capture.phase != "prefill":
            raise ValueError("Expected a prefill capture")
        for span in capture.spans:
            if span.request_id != request_id:
                continue
            if (
                capture.token_ids[span.tensor_start : span.tensor_end].cpu().tolist()
                != ids[span.token_start : span.token_end]
            ):
                raise ValueError("Captured token IDs do not match input")
            pieces.append(
                (
                    span.token_start,
                    span.token_end,
                    capture.tensor[span.tensor_start : span.tensor_end],
                )
            )
    pieces.sort(key=lambda p: p[0])
    end = 0
    for start, stop, _ in pieces:
        if start != end:
            raise ValueError(f"Capture gap or overlap at token {end}; next span starts at {start}")
        end = stop
    if end != len(ids):
        raise ValueError(f"Captured {end}/{len(ids)} tokens; cached prefixes may have been skipped")
    return torch.cat([p[2] for p in pieces])


# ================================ Decode ================================


def hugging_face_modal_secrets(secret_name=None):
    """Forward HF credentials privately to both dataset preparation and GPU jobs."""
    import modal

    if secret_name:
        return [modal.Secret.from_name(secret_name, required_keys=["HF_TOKEN"])]
    from huggingface_hub import get_token

    token = get_token()
    return [modal.Secret.from_dict({"HF_TOKEN": token})] if token else []


def decode_request_timing(request_id, input_tokens, token_receipt_seconds):
    """Host-observed latency from job submission, and first-to-last token speed."""
    count = len(token_receipt_seconds)
    if not count:
        raise ValueError("Request has no sampled tokens to time")
    first, last = token_receipt_seconds[0], token_receipt_seconds[-1]
    decode_seconds = last - first
    if first < 0 or any(b < a for a, b in zip(token_receipt_seconds, token_receipt_seconds[1:])):
        raise ValueError("Token receipt times must be nonnegative and ordered")
    # The first token comes from prefill; only subsequent tokens count as decode.
    return {
        "request_id": request_id,
        "input_tokens": input_tokens,
        "sampled_tokens": count,
        "time_to_first_token_seconds": first,
        "completion_seconds": last,
        "end_to_end_tokens_per_second": count / last if last > 0 else None,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (count - 1) / decode_seconds if decode_seconds > 0 else None,
        "mean_inter_token_ms": 1000 * decode_seconds / (count - 1) if count > 1 else None,
        "token_receipt_seconds": token_receipt_seconds,
    }


def measure_decode_generation(llm, prompt_ids, sampling):
    """Measure the whole job, including terminal capture, CPU flushes and compression."""
    import time

    import torch

    sampled = [[] for _ in prompt_ids]
    receipts = [[] for _ in prompt_ids]
    finishes = {}
    original_callback = llm.send_result

    def record_result(messages):
        # The scheduler has synchronized sampled IDs before calling this callback.
        # One timestamp per batch avoids attributing Python iteration to requests.
        received = time.perf_counter() - started
        for message in messages:
            sampled[message.uid].append(int(message.next_token))
            receipts[message.uid].append(received)
            if message.finished:
                finishes[message.uid] = (
                    "eos"
                    if message.next_token == llm.eos_token_id and not sampling.ignore_eos
                    else "length"
                )
        original_callback(messages)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    llm.send_result = record_result
    started = time.perf_counter()
    try:
        outputs = llm.generate(prompt_ids, sampling)
        torch.cuda.synchronize()
    finally:
        llm.send_result = original_callback
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "sampled_token_ids": sampled,
        "outputs": outputs,
        "finish_reasons": finishes,
        "sampled_tokens": sum(map(len, sampled)),
        "tokens_per_second": sum(map(len, sampled)) / elapsed,
        "request_timings": [
            decode_request_timing(uid, len(ids), receipts[uid])
            for uid, ids in enumerate(prompt_ids)
        ],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


class DecodeWindowObserver:
    """Benchmark sink: recycle windows, retain only the last CPU or GPU tensor.

    No additional copies or serialization occur in this callback. GPU views are
    borrowed; offloaded CPU tensors are owned. Check the last window after timing.
    """

    def __init__(self):
        self.last_window = None
        self.rows = 0
        self.windows = 0

    def __call__(self, window):
        self.last_window = window
        self.rows += len(window.tokens)
        self.windows += 1
