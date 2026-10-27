
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
