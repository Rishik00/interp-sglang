from __future__ import annotations

import pytest
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
    (capture,) = manager.drain()
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
        (None, "lm_head_input"),
        (2, "post_residual"),
    ]


def decode_req(uid: int, generated: list[int]) -> Req:
    req = _req(uid, [1, 2], cached_len=0)
    req.output_len = 100
    req.max_device_len = 102
    req.input_ids = torch.tensor([1, 2, *generated], dtype=torch.int32)
    req.device_len = len(req.input_ids)
    req.cached_len = req.device_len - 1
    return req


def stage_decode(manager, batch, value, terminal=False):
    manager.decode_forward_active = True
    manager.decode_written.clear()
    for layer, signal in manager.decode_keys:
        manager.capture(signal, layer, value, batch)
    manager.finish_decode(batch, used_graph=False, terminal=terminal)


def test_decode_windows_drop_padding_preserve_values_and_flush_partial():
    windows = []
    manager = ActivationCaptureManager(
        ActivationCaptureConfig(
            layer_ids=(2,),
            capture_decode=True,
            decode_window_steps=2,
            decode_gpu_budget_bytes=280,
            decode_sink=windows.append,
        )
    )
    manager.prepare_decode(4, 3, torch.float32, torch.device("cpu"))
    assert manager.decode_stats["reserved_bytes"] <= 280
    assert manager.decode_stats["capacity_rows"] == 7
    value = torch.arange(12).reshape(4, 3).float()
    first = Batch([decode_req(10, [7]), decode_req(20, [8])], "decode")
    stage_decode(manager, first, value)
    value.fill_(-99)  # Simulate replay overwriting its static outputs.
    second = Batch([decode_req(10, [7, 9])], "decode")
    stage_decode(manager, second, value, terminal=True)
    assert len(windows) == 1
    window = windows[0]
    assert window.tensor.shape == (2, 3, 3)
    assert window.tensor[0, :2].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert [(t.request_id, t.token_id, t.generated_index, t.terminal) for t in window.tokens] == [
        (10, 7, 0, False),
        (20, 8, 0, False),
        (10, 9, 1, True),
    ]
    stage_decode(manager, second, value)
    manager.flush_decode()
    assert len(windows) == 2
    assert manager.decode_stats["captured_rows"] == 4
    assert manager.decode_stats["peak_rows"] == 3
    assert manager.drain() == []


def test_decode_budget_and_missing_hooks_fail_explicitly():
    manager = ActivationCaptureManager(
        ActivationCaptureConfig(
            layer_ids=(2,),
            capture_decode=True,
            decode_gpu_budget_bytes=1,
            decode_sink=lambda window: None,
        )
    )
    with pytest.raises(ValueError, match="budget"):
        manager.prepare_decode(4, 3, torch.float32, torch.device("cpu"))
    manager = ActivationCaptureManager(
        ActivationCaptureConfig(
            layer_ids=(2,), capture_decode=True, decode_sink=lambda window: None
        )
    )
    manager.prepare_decode(4, 3, torch.float32, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="missing signals"):
        manager.finish_decode(Batch([decode_req(10, [7])], "decode"), used_graph=False)


def test_decode_sink_failure_retains_pending_window():
    def failed_sink(window):
        raise OSError("disk full")

    manager = ActivationCaptureManager(
        ActivationCaptureConfig(
            layer_ids=(),
            capture_lm_head_input=True,
            capture_decode=True,
            decode_window_steps=1,
            decode_sink=failed_sink,
        )
    )
    manager.prepare_decode(1, 3, torch.float32, torch.device("cpu"))
    with pytest.raises(OSError, match="disk full"):
        stage_decode(manager, Batch([decode_req(10, [7])], "decode"), torch.ones(1, 3))
    assert len(manager.decode_tokens) == 1
    assert manager.decode_stats["windows"] == 0


def test_gpu_sink_borrows_and_recycles_storage_without_offload():
    from utils import DecodeGpuWindowObserver

    observer = DecodeGpuWindowObserver()
    manager = ActivationCaptureManager(
        ActivationCaptureConfig(
            layer_ids=(),
            capture_lm_head_input=True,
            capture_decode=True,
            decode_window_steps=1,
            decode_offload_to_cpu=False,
            decode_sink=observer,
        )
    )
    manager.prepare_decode(1, 3, torch.float32, torch.device("cpu"))
    stage_decode(manager, Batch([decode_req(10, [7])], "decode"), torch.ones(1, 3))
    first = observer.last_window
    # CPU-backed test buffers expose the same ownership contract without a GPU.
    assert first.tensor.data_ptr() == manager.decode_window.data_ptr()
    stage_decode(
        manager,
        Batch([decode_req(10, [7, 8])], "decode"),
        torch.full((1, 3), 9.0),
        terminal=True,
    )
    assert observer.rows == 2 and observer.windows == 2
    assert observer.last_window.tensor.data_ptr() == first.tensor.data_ptr()
    assert observer.last_window.tensor.tolist() == [[[9.0, 9.0, 9.0]]]
    assert observer.last_window.tokens[0].generated_index == 1
    assert observer.last_window.tokens[0].terminal
    assert manager.decode_tokens == []


def test_graph_construction_writes_static_outputs_without_harvesting():
    windows = []
    manager = ActivationCaptureManager(
        ActivationCaptureConfig(
            layer_ids=(),
            capture_lm_head_input=True,
            capture_decode=True,
            decode_sink=windows.append,
        )
    )
    manager.prepare_decode(4, 3, torch.float32, torch.device("cpu"))
    manager.enabled = False
    manager.recording_decode_graph = True
    batch = Batch([decode_req(-1, [0])], "decode")
    manager.capture("lm_head_input", None, torch.ones(1, 3), batch)
    manager.verify_decode_writes()
    manager.flush_decode()
    assert windows == []
    assert manager.decode_tokens == []
