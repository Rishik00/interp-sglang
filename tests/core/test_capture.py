from __future__ import annotations

import torch
from minisgl.capture import ActivationCaptureConfig, ActivationCaptureManager
from minisgl.core import Batch, Req, SamplingParams


def _req(uid: int, token_ids: list[int], cached_len: int) -> Req:
    return Req(
        input_ids=torch.tensor(token_ids, dtype=torch.int32),
        table_idx=uid,
        cached_len=cached_len,
        output_len=1,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=1),
        cache_handle=None,  # type: ignore[arg-type]
    )


def test_capture_selection_provenance_and_staging_copy() -> None:
    manager = ActivationCaptureManager(ActivationCaptureConfig(layer_ids=(2,)))
    batch = Batch(
        reqs=[_req(10, [1, 2, 3], cached_len=1), _req(20, [4, 5], cached_len=0)],
        phase="prefill",
    )
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    manager.capture("post_residual", 1, source, batch)
    manager.capture("post_residual", 2, source, batch)
    manager.capture("attention_output", 2, source + 100, batch)
    source.fill_(-1)

    captures = manager.drain()
    assert [(item.layer_id, item.signal) for item in captures] == [
        (2, "attention_output"),
        (2, "post_residual"),
    ]
    assert captures[0].token_ids.tolist() == [2, 3, 4, 5]
    assert captures[0].spans[0].request_id == 10
    assert captures[0].spans[0].tensor_start == 0
    assert captures[0].spans[0].tensor_end == 2
    assert captures[0].spans[0].token_start == 1
    assert captures[0].spans[0].token_end == 3
    assert captures[0].spans[1].request_id == 20
    assert captures[0].spans[1].tensor_start == 2
    assert captures[0].spans[1].tensor_end == 4
    assert torch.equal(captures[1].tensor, torch.arange(12).reshape(4, 3))
    assert manager.drain() == []


def test_capture_skips_decode() -> None:
    manager = ActivationCaptureManager(ActivationCaptureConfig(layer_ids=(2,)))
    batch = Batch(reqs=[_req(10, [1, 2], cached_len=1)], phase="decode")

    manager.capture("attention_output", 2, torch.ones(1, 3), batch)

    assert manager.drain() == []


def test_capture_can_be_temporarily_disabled() -> None:
    manager = ActivationCaptureManager(ActivationCaptureConfig(layer_ids=(2,)))
    manager.enabled = False
    batch = Batch(reqs=[_req(10, [1, 2], cached_len=0)], phase="prefill")

    manager.capture("attention_output", 2, torch.ones(2, 3), batch)

    assert manager.drain() == []


def test_lm_head_input_is_opt_in_and_independent_of_layer_selection() -> None:
    batch = Batch(reqs=[_req(10, [1, 2, 3], cached_len=1)], phase="prefill")
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    disabled = ActivationCaptureManager(ActivationCaptureConfig(layer_ids=()))
    disabled.capture("lm_head_input", None, source, batch)
    assert disabled.drain() == []

    manager = ActivationCaptureManager(
        ActivationCaptureConfig(layer_ids=(), capture_lm_head_input=True)
    )
    manager.capture("lm_head_input", None, source, batch)
    source.fill_(-1)
    capture, = manager.drain()
    assert capture.signal == "lm_head_input"
    assert capture.layer_id is None
    assert torch.equal(capture.tensor, torch.arange(6).reshape(2, 3))
    assert capture.token_ids.tolist() == [2, 3]
    assert capture.spans[0].token_start == 1
    assert capture.spans[0].token_end == 3

    batch.phase = "decode"
    manager.capture("lm_head_input", None, torch.ones(1, 3), batch)
    assert manager.drain() == []


def test_model_level_and_layer_captures_can_be_drained_together() -> None:
    manager = ActivationCaptureManager(
        ActivationCaptureConfig(layer_ids=(2,), capture_lm_head_input=True)
    )
    batch = Batch(reqs=[_req(10, [1, 2], cached_len=0)], phase="prefill")
    manager.capture("post_residual", 2, torch.ones(2, 3), batch)
    manager.capture("lm_head_input", None, torch.ones(2, 3), batch)
    assert [(item.layer_id, item.signal) for item in manager.drain()] == [
        (None, "lm_head_input"), (2, "post_residual"),
    ]
