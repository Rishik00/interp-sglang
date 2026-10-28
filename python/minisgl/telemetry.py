from __future__ import annotations

import time
from typing import Any, Iterable

import torch


# Review: it should be mentioned at which step are we logging at. 
# I think we're logging the engine's 
class Telemetry:
    """Collect opt-in memory snapshots and generation timing for one generate() call.

    TTFT is measured from entry to ``LLM.generate`` until the first generated
    token is delivered. Prefill throughput uses CUDA event time around prefill
    model forwards. Decode throughput uses generated tokens after each request's first,
    divided by the host-observed first-to-last token interval.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._prefill_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        self._active = False
        self._started_at: float | None = None
        self._first_token_at: float | None = None
        self._last_token_at: float | None = None
        self._generated_tokens = 0
        self._seen_uids: set[int] = set()
        self._memory_snapshots: dict[str, dict[str, int]] = {}
        self.last_result: dict[str, Any] | None = None

    def __enter__(self) -> Telemetry:
        self.start_generation()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.finish_generation()
        return False

    def start_generation(self) -> None:
        self._prefill_events.clear()
        self.last_result = None
        self._started_at = time.perf_counter()
        self._first_token_at = None
        self._last_token_at = None
        self._generated_tokens = 0
        self._seen_uids.clear()
        self._memory_snapshots = {"generation_start": self._memory_snapshot()}
        self._active = True

    def record_batch(
        self,
        token_count: int,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
    ) -> None:
        if self._active:
            self._prefill_events.append((token_count, start, end))

    def record_tokens(self, messages: Iterable[Any]) -> None:
        if not self._active:
            return
        now = time.perf_counter()
        for message in messages:
            self._generated_tokens += 1
            self._seen_uids.add(message.uid)
            if self._first_token_at is None:
                self._first_token_at = now
                self._memory_snapshots["first_token"] = self._memory_snapshot()
            self._last_token_at = now

    def finish_generation(self) -> dict[str, Any]:
        self._memory_snapshots["generation_end"] = self._memory_snapshot()
        self._active = False

        prefill_tokens, prefill_seconds = self._prefill_totals()
        ttft = (
            self._first_token_at - self._started_at
            if self._first_token_at is not None and self._started_at is not None
            else None
        )
        first_to_last = (
            self._last_token_at - self._first_token_at
            if self._first_token_at is not None and self._last_token_at is not None
            else None
        )
        self.last_result = {
            "memory_snapshots": self._memory_snapshots,
            "prefill_tokens": prefill_tokens,
            "prefill_forward_seconds": prefill_seconds,
            "prefill_tokens_per_second": (
                prefill_tokens / prefill_seconds if prefill_seconds > 0 else None
            ),
            "ttft_seconds": ttft,
            "generated_tokens": self._generated_tokens,
            "first_to_last_token_seconds": first_to_last,
            # Each request's first token comes from prefill, so only later tokens count as decode.
            "decode_tokens_per_second": (
                (self._generated_tokens - len(self._seen_uids)) / first_to_last
                if first_to_last is not None and first_to_last > 0
                else None
            ),
        }
        return self.last_result

    def _prefill_totals(self) -> tuple[int, float]:
        token_count = 0
        elapsed_seconds = 0.0
        for count, start, end in self._prefill_events:
            end.synchronize()
            token_count += count
            elapsed_seconds += start.elapsed_time(end) / 1000
        return token_count, elapsed_seconds

    def _memory_snapshot(self) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "driver_free_bytes": free_bytes,
            "driver_used_bytes": total_bytes - free_bytes,
            "driver_total_bytes": total_bytes,
        }
