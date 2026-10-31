from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Literal, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


ActivationSignal = Literal["attention_output", "post_residual", "lm_head_input"]


@dataclass(frozen=True)
class ActivationCaptureConfig:
    """Select signals. Decode uses fixed graph outputs and a bounded staging window."""

    layer_ids: Tuple[int, ...]
    capture_attention_output: bool = True
    capture_post_residual: bool = True
    capture_lm_head_input: bool = False
    capture_prefill: bool = True
    capture_decode: bool = False
    decode_window_steps: int = 16
    decode_gpu_budget_bytes: int = 128 * 1024 * 1024
    capture_terminal_token: bool = True
    decode_sink: Callable[[DecodeCaptureWindow], None] | None = None


@dataclass(frozen=True)
class DecodeToken:
    request_id: int
    token_id: int
    token_position: int
    generated_index: int
    terminal: bool = False


@dataclass(frozen=True)
class DecodeCaptureWindow:
    """An owned CPU tensor: [signals, real token rows, hidden width]."""

    tensor: torch.Tensor
    keys: Tuple[Tuple[int | None, ActivationSignal], ...]
    tokens: Tuple[DecodeToken, ...]


@dataclass(frozen=True)
class ActivationSpan:
    """Map a slice of a flattened activation tensor back to one request."""

    request_id: int
    tensor_start: int
    tensor_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class CapturedActivation:
    signal: ActivationSignal
    layer_id: int | None  # None identifies the model-level LM head input.
    phase: str
    tensor: torch.Tensor
    token_ids: torch.Tensor
    spans: Tuple[ActivationSpan, ...]


@dataclass(frozen=True)
class _StagedActivation:
    signal: ActivationSignal
    layer_id: int | None
    phase: str
    tensor: torch.Tensor
    token_ids: torch.Tensor
    spans: Tuple[ActivationSpan, ...]


class ActivationCaptureManager:
    """Stage selected prefill activations on GPU and drain them to CPU on demand."""

    def __init__(self, config: ActivationCaptureConfig) -> None:
        self.config = config
        self.enabled = True
        self._layer_ids = frozenset(config.layer_ids)
        self._staged: List[_StagedActivation] = []
        self.decode_buffer: torch.Tensor | None = None
        self.decode_window: torch.Tensor | None = None
        self.recording_decode_graph = False
        self.decode_forward_active = False
        self.decode_written: set[Tuple[int | None, ActivationSignal]] = set()
        self.decode_keys: Tuple[Tuple[int | None, ActivationSignal], ...] = ()
        self.decode_key_indices: dict[Tuple[int | None, ActivationSignal], int] = {}
        self.decode_tokens: List[DecodeToken] = []
        self.decode_steps = 0
        self.decode_stats = {
            "reserved_bytes": 0,
            "capacity_rows": 0,
            "peak_rows": 0,
            "captured_rows": 0,
            "terminal_rows": 0,
            "windows": 0,
            "graph_batches": 0,
            "eager_batches": 0,
        }

    def prepare_decode(
        self, max_batch_size: int, hidden_size: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        """Reserve capture memory before the engine sizes its KV cache."""
        if not self.config.capture_decode:
            return
        if self.config.decode_sink is None:
            raise ValueError(
                "Decode capture requires a sink; CPU records are not retained indefinitely"
            )
        if self.config.decode_window_steps < 1 or max_batch_size < 1:
            raise ValueError("Decode window steps and batch size must be positive")
        keys = [
            (layer, signal)
            for layer in sorted(self._layer_ids)
            for signal in ("attention_output", "post_residual")
            if self._signal_enabled(signal)
        ]
        if self.config.capture_lm_head_input:
            keys.append((None, "lm_head_input"))
        if not keys:
            raise ValueError("Decode capture requires at least one signal")
        row_bytes = len(keys) * hidden_size * dtype.itemsize
        scratch_bytes = row_bytes * max_batch_size
        capacity = min(
            max_batch_size * self.config.decode_window_steps,
            (self.config.decode_gpu_budget_bytes - scratch_bytes) // row_bytes,
        )
        if capacity < max_batch_size:
            raise ValueError("Decode GPU budget cannot fit graph outputs and one full batch")
        self.decode_keys = tuple(keys)
        self.decode_key_indices = {key: index for index, key in enumerate(keys)}
        self.decode_buffer = torch.empty(
            (len(keys), max_batch_size, hidden_size), dtype=dtype, device=device
        )
        self.decode_window = torch.empty(
            (len(keys), capacity, hidden_size), dtype=dtype, device=device
        )
        self.decode_stats.update(
            reserved_bytes=scratch_bytes + capacity * row_bytes, capacity_rows=capacity
        )

    def verify_decode_writes(self) -> None:
        if self.decode_written != set(self.decode_keys):
            raise RuntimeError(
                f"Decode hooks missing signals: {set(self.decode_keys) - self.decode_written}"
            )

    def finish_decode(self, batch: Batch, used_graph: bool, terminal: bool = False) -> None:
        """Snapshot values and CPU provenance before request counters are advanced."""
        if not self.enabled or not self.config.capture_decode:
            return
        if self.decode_buffer is None or self.decode_window is None:
            raise RuntimeError("Decode capture buffers were not initialized")
        if not used_graph:
            self.verify_decode_writes()
        count = batch.size
        start = len(self.decode_tokens)
        if start + count > self.decode_window.shape[1]:
            self.flush_decode()
            start = 0
        # Same-stream ordering prevents the next graph replay overwriting this snapshot.
        self.decode_window[:, start : start + count].copy_(self.decode_buffer[:, :count])
        for req in batch.reqs:
            position = req.cached_len
            if req.extend_len != 1 or position >= len(req.input_ids):
                raise RuntimeError("Decode provenance requires one known input token per request")
            prompt_length = req.max_device_len - req.output_len
            self.decode_tokens.append(
                DecodeToken(
                    req.uid,
                    int(req.input_ids[position]),
                    position,
                    position - prompt_length,
                    terminal,
                )
            )
        self.decode_steps += 1
        self.decode_stats["captured_rows"] += count
        self.decode_stats["terminal_rows"] += count if terminal else 0
        self.decode_stats["peak_rows"] = max(
            self.decode_stats["peak_rows"], len(self.decode_tokens)
        )
        self.decode_stats["graph_batches" if used_graph else "eager_batches"] += 1
        if (
            self.decode_steps >= self.config.decode_window_steps
            or len(self.decode_tokens) == self.decode_window.shape[1]
        ):
            self.flush_decode()

    def flush_decode(self) -> None:
        """Blocking offload and sink provide bounded memory and explicit backpressure."""
        if not self.decode_tokens:
            return
        assert self.decode_window is not None and self.config.decode_sink is not None
        # A CPU clone is needed for CPU tests as well as the normal CUDA-to-CPU copy.
        tensor = self.decode_window[:, : len(self.decode_tokens)].to("cpu", copy=True).contiguous()
        self.config.decode_sink(
            DecodeCaptureWindow(tensor, self.decode_keys, tuple(self.decode_tokens))
        )
        # Keep the window intact if the sink raises. Shutdown can retry the flush.
        self.decode_tokens.clear()
        self.decode_steps = 0
        self.decode_stats["windows"] += 1

    def _signal_enabled(self, signal: ActivationSignal) -> bool:
        if signal == "attention_output":
            return self.config.capture_attention_output
        if signal == "lm_head_input":
            return self.config.capture_lm_head_input
        return self.config.capture_post_residual

    @staticmethod
    def _provenance(batch: Batch) -> Tuple[torch.Tensor, Tuple[ActivationSpan, ...]]:
        token_parts: List[torch.Tensor] = []
        spans: List[ActivationSpan] = []
        tensor_offset = 0
        for req in batch.reqs:
            token_part = req.input_ids[req.cached_len : req.device_len].detach().clone()
            token_parts.append(token_part)
            next_offset = tensor_offset + req.extend_len
            spans.append(
                ActivationSpan(
                    request_id=req.uid,
                    tensor_start=tensor_offset,
                    tensor_end=next_offset,
                    token_start=req.cached_len,
                    token_end=req.device_len,
                )
            )
            tensor_offset = next_offset

        token_ids = torch.cat(token_parts) if token_parts else torch.empty(0, dtype=torch.int32)
        return token_ids, tuple(spans)

    def capture(
        self,
        signal: ActivationSignal,
        layer_id: int | None,
        tensor: torch.Tensor,
        batch: Batch,
    ) -> None:
        if batch.is_decode:
            # Python executes these copies during graph construction; replay executes
            # the recorded GPU copies. It must never append Python records in the graph.
            if self.config.capture_decode and (
                self.recording_decode_graph or (self.enabled and self.decode_forward_active)
            ):
                key = (layer_id, signal)
                if key in self.decode_key_indices:
                    assert self.decode_buffer is not None
                    if tensor.ndim != 2 or tensor.shape[0] > self.decode_buffer.shape[1]:
                        raise RuntimeError(f"Invalid decode tensor shape: {tuple(tensor.shape)}")
                    self.decode_buffer[self.decode_key_indices[key], : tensor.shape[0]].copy_(
                        tensor.detach()
                    )
                    self.decode_written.add(key)
            return
        if not self.enabled or not self.config.capture_prefill:
            return
        if not self._signal_enabled(signal):
            return
        if signal != "lm_head_input" and layer_id not in self._layer_ids:
            return

        token_ids, spans = self._provenance(batch)
        expected_tokens = sum(req.extend_len for req in batch.reqs)
        if tensor.ndim != 2 or tensor.shape[0] != expected_tokens:
            raise RuntimeError(
                f"Unexpected {signal} shape at layer {layer_id}: "
                f"{tuple(tensor.shape)}, expected ({expected_tokens}, hidden_size)"
            )

        # The source tensors are mutated by later fused residual/norm operations.
        # clone() records a device-to-device copy on the current inference stream.
        self._staged.append(
            _StagedActivation(
                signal=signal,
                layer_id=layer_id,
                phase=batch.phase,
                tensor=tensor.detach().clone(),
                token_ids=token_ids,
                spans=spans,
            )
        )

    def drain(self) -> List[CapturedActivation]:
        """Hand all staged captures to the caller and empty the queue.

        capture() only clones tensors on the GPU and appends them to `_staged`;
        nothing leaves the device during the forward pass. drain() copies each
        staged tensor to CPU (a blocking D2H copy), clears `_staged` so the GPU
        clones can be freed, and returns model-level records first, followed by
        layer records sorted by (layer, signal).
        """
        captures = [
            CapturedActivation(
                signal=record.signal,
                layer_id=record.layer_id,
                phase=record.phase,
                tensor=record.tensor.to(device="cpu"),
                token_ids=record.token_ids,
                spans=record.spans,
            )
            for record in self._staged
        ]
        self._staged.clear()
        return sorted(
            captures,
            key=lambda item: (-1 if item.layer_id is None else item.layer_id, item.signal),
        )
