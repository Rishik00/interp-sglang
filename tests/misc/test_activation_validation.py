import json
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from activation_validation import (
    ACTIVATION_METRICS,
    capture_hf_prefill,
    capture_points,
    compare_activation_l2,
    copy_trace_tensor,
    install_hf_operation,
    prefill_spans,
    summarize_referee_populations,
    validate_engine_capture_spans,
)


def test_referee_pooling_uses_tokens_and_strict_thresholds():
    def prompt(row, engine, hf):
        def metrics(values):
            return {"token_relative_error": values, "token_l2": values}

        return {
            "dataset_row": row,
            "token_ids": [10] * len(engine),
            "points": [
                {
                    "layer": None,
                    "signal": "lm_head_input",
                    "engine_vs_fp32": metrics(engine),
                    "hf_fp16_vs_fp32": metrics(hf),
                }
            ],
        }

    result = summarize_referee_populations(
        [prompt(1, [0.1], [0.2]), prompt(2, [0.0, 0.02, 0.04], [0.01, 0.03, 0.05])]
    )
    pooled = result["summaries"][-1]
    assert pooled["engine_median"] == pytest.approx(0.03)
    assert pooled["engine_p99"] == pytest.approx(0.0982)
    assert pooled["engine_p99_9"] == pytest.approx(0.09982)
    assert pooled["engine_above_2_percent"] == 2
    assert pooled["engine_above_5_percent"] == 1
    assert pooled["hf_fp16_above_2_percent"] == 3
    assert pooled["hf_fp16_above_5_percent"] == 1
    assert pooled["engine_closer"] == 4
    assert pooled["worst_20_overlap"] == 4
    assert pooled["points_engine_max_worse"] == 0


def test_hf_replacement_restores_norm_and_activation_without_changing_weights(monkeypatch):
    import activation_validation as validation

    monkeypatch.setitem(
        sys.modules,
        "transformers.models.qwen3.modeling_qwen3",
        SimpleNamespace(
            Qwen3RotaryEmbedding=None, apply_rotary_pos_emb=None, eager_attention_forward=None
        ),
    )
    monkeypatch.setattr(validation, "hf_rmsnorm", lambda hidden, weight, epsilon: hidden * weight)

    def make_norm():
        return SimpleNamespace(
            weight=torch.ones(4),
            eps=1e-6,
            forward=lambda hidden, residual=None: (hidden, residual),
            forward_inplace=lambda hidden: None,
        )

    original_activation = lambda gate_up: gate_up
    block = SimpleNamespace(
        input_layernorm=make_norm(),
        post_attention_layernorm=make_norm(),
        self_attn=SimpleNamespace(q_norm=make_norm(), k_norm=make_norm()),
        mlp=SimpleNamespace(act_fn=original_activation),
    )
    model = SimpleNamespace(norm=make_norm(), layers=SimpleNamespace(op_list=[block]))
    llm = SimpleNamespace(engine=SimpleNamespace(model=SimpleNamespace(model=model)))
    original_norm = block.input_layernorm.forward
    weight = block.input_layernorm.weight
    restore = install_hf_operation(llm, "norm", None, 2)
    hidden, residual = torch.ones(2, 4), torch.ones(2, 4)
    normalized, summed = block.input_layernorm.forward(hidden, residual)
    assert torch.equal(summed, torch.full((2, 4), 2.0))
    assert torch.equal(normalized, summed)
    assert block.input_layernorm.weight is weight
    restore()
    assert block.input_layernorm.forward is original_norm
    assert torch.equal(hidden, torch.ones(2, 4))
    assert torch.equal(residual, torch.ones(2, 4))
    restore = install_hf_operation(llm, "silu", None, 2)
    actual = block.mlp.act_fn(torch.cat([hidden, residual], dim=-1))
    assert torch.equal(actual, torch.nn.functional.silu(hidden) * residual)
    restore()
    assert block.mlp.act_fn is original_activation


def test_trace_copy_preserves_tokens_and_survives_inplace_reuse():
    original = torch.arange(24, dtype=torch.float16).reshape(1, 3, 2, 4)
    expected = original.reshape(3, 8).clone()
    snapshot = copy_trace_tensor(original.transpose(1, 2), 3, heads_first=True)
    original.zero_()
    assert torch.equal(snapshot, expected)
    assert snapshot.device.type == "cpu"
    assert torch.equal(copy_trace_tensor((expected, None), 3), expected)


def test_l2_keeps_token_location_magnitude_and_zero_reference_norm():
    expected = torch.zeros(3, 2, dtype=torch.float16)
    actual = expected.clone()
    actual[1] = torch.tensor([3, 4])
    actual[2] = torch.tensor([0, 12])
    metrics = compare_activation_l2(actual, expected)
    assert metrics == {
        "token_l2": [0.0, 5.0, 12.0],
        "prompt_l2": 13.0,
        "hf_token_norm": [0.0, 0.0, 0.0],
        "token_relative_error": [None, None, None],
    }
    assert compare_activation_l2(expected, expected) == {
        "token_l2": [0.0, 0.0, 0.0],
        "prompt_l2": 0.0,
        "hf_token_norm": [0.0, 0.0, 0.0],
        "token_relative_error": [None, None, None],
    }
    with pytest.raises(ValueError, match="shapes"):
        compare_activation_l2(actual, expected[:2])
    with pytest.raises(ValueError, match="dtypes"):
        compare_activation_l2(actual, expected.float())
    actual[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        compare_activation_l2(actual, expected)


def test_relative_error_uses_the_same_token_hf_norm_without_an_epsilon():
    expected = torch.tensor([[3.0, 4.0], [1e-8, 0.0]], dtype=torch.float32)
    actual = 2 * expected
    result = compare_activation_l2(actual, expected)
    assert result["hf_token_norm"] == pytest.approx([5.0, 1e-8])
    assert result["token_l2"] == result["hf_token_norm"]
    assert result["token_relative_error"] == [1.0, 1.0]


def test_larger_workload_uses_new_scan_budget_instead_of_saved_small_sample(monkeypatch):
    import modal_validate_activations as runner

    config = SimpleNamespace(model_type="qwen3", _commit_hash="pinned-revision")
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=lambda model_id: config),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
        ),
    )
    saved = {
        "prompts": [None] * 50,
        "seed": 42,
        "revision": "dataset-revision",
        "max_input_tokens": 3072,
        "scanned_rows": 4096,
    }

    def prepare_sample(tokenizer, **settings):
        if settings["scan_rows"] == 4096:
            raise ValueError("Length bucket 3 has only 21 candidates; need 25")
        assert settings == {
            "count": 100,
            "seed": 42,
            "revision": "dataset-revision",
            "max_tokens": 3072,
            "scan_rows": 8192,
        }
        return {"prompts": [None] * 100}

    monkeypatch.setattr(runner, "load_wildchat_prompts", prepare_sample)
    revision, workload = runner.prepare_model_workload.local("Qwen/Qwen3-1.7B", saved, 100)
    assert revision == workload["tokenizer_revision"] == "pinned-revision"
    assert len(workload["prompts"]) == 100


def test_prefill_spans_cover_whole_and_partial_final_chunks():
    assert prefill_spans(513, "whole") == [(0, 513)]
    assert prefill_spans(513, "chunked") == [(0, 256), (256, 512), (512, 513)]
    assert prefill_spans(256, "chunked") == [(0, 256)]
    with pytest.raises(ValueError):
        prefill_spans(3, "whole", 0)
    with pytest.raises(ValueError):
        prefill_spans(0, "whole")


@pytest.mark.parametrize("layers,points", [(16, 33), (28, 57), (36, 73)])
def test_capture_plan_has_all_block_points_and_one_final_norm(layers, points):
    plan = capture_points(layers)
    assert len(plan) == len(set(plan)) == points
    assert plan[-3:] == [
        (layers - 1, "attention_output"),
        (layers - 1, "post_residual"),
        (None, "lm_head_input"),
    ]


class SyntheticProjection(torch.nn.Module):
    def forward(self, hidden):
        return hidden * 2


class SyntheticBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = SyntheticProjection()

    def forward(self, hidden):
        return hidden + self.self_attn.o_proj(hidden) + 1


class SyntheticReference(torch.nn.Module):
    """CPU arithmetic fixture, with cache-dependent states and real module hooks."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([SyntheticBlock(), SyntheticBlock()])
        self.model.norm = torch.nn.Identity()
        self.calls = []

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        cache_position,
        past_key_values,
        use_cache,
        logits_to_keep,
    ):
        start = 0 if past_key_values is None else past_key_values.get_seq_length()
        stop = start + input_ids.shape[1]
        assert attention_mask.tolist() == [[1] * stop]
        assert cache_position.tolist() == list(range(start, stop))
        assert position_ids.tolist() == [list(range(start, stop))]
        assert use_cache and logits_to_keep == 1
        prefix_sum = 0 if past_key_values is None else past_key_values.total
        cumulative = input_ids.cumsum(dim=1) + prefix_sum
        hidden = cumulative.float().unsqueeze(-1).repeat(1, 1, 2)
        for block in self.model.layers:
            hidden = block(hidden)
        self.model.norm(hidden)
        self.calls.append((start, stop))
        return SimpleNamespace(
            past_key_values=SimpleNamespace(
                get_seq_length=lambda: stop, total=int(cumulative[0, -1])
            )
        )


def test_hf_chunked_carries_cache_and_positions_and_matches_whole_at_every_point():
    # No model/tokenizer download: the fixture verifies call boundaries and hook semantics.
    reference = SyntheticReference()
    ids = [2, 3, 5, 7, 11]
    whole = capture_hf_prefill(reference, ids, "whole", device="cpu")
    chunked = capture_hf_prefill(reference, ids, "chunked", chunk_size=2, device="cpu")
    assert reference.calls == [(0, 5), (0, 2), (2, 4), (4, 5)]
    assert set(whole) == set(capture_points(2))
    for point in whole:
        assert torch.equal(whole[point], chunked[point])
        assert whole[point].shape == (5, 2)
    assert torch.equal(whole[(0, "attention_output")][:, 0], torch.tensor([4, 10, 20, 34, 56]))
    assert torch.equal(whole[(0, "post_residual")][:, 0], torch.tensor([7, 16, 31, 52, 85]))
    assert not any(module._forward_hooks for module in reference.modules())


def test_engine_capture_rejects_gaps_duplicates_missing_points_and_wrong_boundaries():
    points = capture_points(1)
    spans = [(0, 2), (2, 3)]
    captures = [
        SimpleNamespace(
            layer_id=layer,
            signal=signal,
            phase="prefill",
            tensor=torch.zeros(stop - start, 2),
            spans=[
                SimpleNamespace(
                    request_id=0,
                    token_start=start,
                    token_end=stop,
                    tensor_start=0,
                    tensor_end=stop - start,
                )
            ],
        )
        for layer, signal in points
        for start, stop in spans
    ]
    validate_engine_capture_spans(captures, points, spans)
    with pytest.raises(ValueError, match="boundaries"):
        validate_engine_capture_spans(captures + [captures[0]], points, spans)
    with pytest.raises(ValueError, match="points"):
        validate_engine_capture_spans(captures[:-2], points, spans)
    with pytest.raises(ValueError, match="boundaries"):
        validate_engine_capture_spans(captures, points, [(0, 3)])
    captures[0].spans[0].request_id = 9
    with pytest.raises(ValueError, match="one request"):
        validate_engine_capture_spans(captures, points, spans)


def saved_result_fixture():
    workload = {
        "prompts": [
            {"dataset_row": 1, "token_ids": [10, 20]},
            {"dataset_row": 2, "token_ids": [30, 40]},
        ]
    }
    points = capture_points(1)
    setup = {
        "type": "setup",
        "model": "fixture",
        "model_revision": "revision",
        "mode": "whole",
        "prompts": 2,
        "num_layers": 1,
        "metrics": ["token_l2", "prompt_l2"],
        "capture_points": [{"layer": layer, "signal": signal} for layer, signal in points],
    }
    records = [setup] + [
        {
            "type": "prompt",
            **prompt,
            "spans": [[0, 2]],
            "rows": [
                {"layer": layer, "signal": signal, "token_l2": [3.0, 4.0], "prompt_l2": 5.0}
                for layer, signal in points
            ],
        }
        for prompt in workload["prompts"]
    ]
    return workload, records


def test_saved_artifact_requires_every_prompt_point_and_consistent_l2(tmp_path):
    from modal_validate_activations import verify_saved_run

    workload, records = saved_result_fixture()
    path = tmp_path / "results.jsonl"

    def write_records():
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    write_records()
    assert verify_saved_run(path, workload, "fixture", "revision", "whole", 256) == 2
    final = records.pop()
    write_records()
    with pytest.raises(ValueError, match="Incomplete saved run: 1/2"):
        verify_saved_run(path, workload, "fixture", "revision", "whole", 256)
    records.append(final)
    records[-1]["rows"][0]["prompt_l2"] = 6.0
    write_records()
    with pytest.raises(ValueError, match="inconsistent"):
        verify_saved_run(path, workload, "fixture", "revision", "whole", 256)
    records[-1]["rows"][0]["prompt_l2"] = 5.0
    records[-1]["token_ids"] = [99, 40]
    write_records()
    with pytest.raises(ValueError, match="identity"):
        verify_saved_run(path, workload, "fixture", "revision", "whole", 256)


def test_saved_artifact_checks_hf_norm_and_relative_error(tmp_path):
    from modal_validate_activations import verify_saved_run

    workload, records = saved_result_fixture()
    records[0]["metrics"] = ACTIVATION_METRICS
    for record in records[1:]:
        for row in record["rows"]:
            row.update(hf_token_norm=[6.0, 0.0], token_relative_error=[0.5, None])
    path = tmp_path / "relative-results.jsonl"

    def write_records():
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    write_records()
    assert verify_saved_run(path, workload, "fixture", "revision", "whole", 256) == 2
    records[-1]["rows"][0]["token_relative_error"][0] = 0.1
    write_records()
    with pytest.raises(ValueError, match="Relative error is inconsistent"):
        verify_saved_run(path, workload, "fixture", "revision", "whole", 256)
    records[-1]["rows"][0]["token_relative_error"][0] = 0.5
    records[-1]["rows"][0]["hf_token_norm"][0] = -1.0
    write_records()
    with pytest.raises(ValueError, match="nonnegative"):
        verify_saved_run(path, workload, "fixture", "revision", "whole", 256)


def test_detached_connection_loss_cannot_write_complete(tmp_path, monkeypatch):
    import modal_validate_activations as runner

    workload, _ = saved_result_fixture()

    @contextmanager
    def suppress_connection_loss(**kwargs):
        try:
            yield
        except ConnectionError:
            pass

    def lose_connection():
        raise ConnectionError("Disconnected while waiting for GPU result")

    monkeypatch.setattr(sys, "argv", ["modal_validate_activations.py"])
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(runner, "hugging_face_modal_secrets", lambda: [])
    monkeypatch.setattr(runner.modal, "enable_output", nullcontext)
    monkeypatch.setattr(runner.app, "run", suppress_connection_loss)
    monkeypatch.setattr(
        runner,
        "prepare_model_workload",
        SimpleNamespace(
            with_options=lambda **kwargs: SimpleNamespace(
                remote=lambda *args: ("revision", workload)
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "validate_prefill",
        SimpleNamespace(
            with_options=lambda **kwargs: SimpleNamespace(
                spawn=lambda *args: SimpleNamespace(object_id="fake-call-id", get=lose_connection)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="0/1 verified runs"):
        runner.main()
    assert json.loads((tmp_path / "completion.json").read_text())["status"] != "complete"
    assert json.loads((tmp_path / "jobs.json").read_text())[0]["call_id"] == "fake-call-id"
