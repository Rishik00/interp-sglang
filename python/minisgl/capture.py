from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal, Tuple

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


ActivationSignal = Literal["attention_output", "post_residual"]


@dataclass(frozen=True)
class ActivationCaptureConfig:
    """Select the layers and residual-space signals to capture during prefill."""

    layer_ids: Tuple[int, ...]
    capture_attention_output: bool = True
    capture_post_residual: bool = True


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
    layer_id: int
    phase: str
    tensor: torch.Tensor
    token_ids: torch.Tensor
    spans: Tuple[ActivationSpan, ...]


@dataclass(frozen=True)
class _StagedActivation:
    signal: ActivationSignal
    layer_id: int
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

    def _signal_enabled(self, signal: ActivationSignal) -> bool:
        if signal == "attention_output":
            return self.config.capture_attention_output
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
        layer_id: int,
        tensor: torch.Tensor,
        batch: Batch,
    ) -> None:
        # Decode may run under CUDA graphs. This first implementation intentionally
        # captures prefill only, which also keeps graph construction untouched.
        if not self.enabled or not batch.is_prefill or layer_id not in self._layer_ids:
            return
        if not self._signal_enabled(signal):
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
        clones can be freed, and returns the records sorted by (layer, signal).
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
        return sorted(captures, key=lambda item: (item.layer_id, item.signal))
