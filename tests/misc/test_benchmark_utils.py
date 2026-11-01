import pytest
import torch

from utils import capture_layers, compare_activations


def test_request_timing_separates_queue_and_prefill_from_decode():
    from utils import decode_request_timing

    early = decode_request_timing(0, 50, [2.0, 2.25, 2.5])
    queued = decode_request_timing(1, 100, [12.0, 12.25, 12.5])
    assert early["decode_tokens_per_second"] == queued["decode_tokens_per_second"] == 4.0
    assert early["mean_inter_token_ms"] == 250.0
    assert queued["time_to_first_token_seconds"] == 12.0
    assert queued["completion_seconds"] == 12.5
    assert queued["end_to_end_tokens_per_second"] == 3 / 12.5
    single = decode_request_timing(2, 1, [1.0])
    assert single["decode_tokens_per_second"] is None
    assert single["mean_inter_token_ms"] is None
    with pytest.raises(ValueError, match="ordered"):
        decode_request_timing(3, 1, [1.0, 0.5])


@pytest.mark.parametrize("depth", [1, 2, 16, 28, 32, 64])
def test_capture_layers_cover_valid_model_depth(depth):
    layers = capture_layers(depth)
    assert layers[0] == 0
    assert layers[-1] == depth - 1
    assert len(layers) == len(set(layers))
    assert all(0 <= layer < depth for layer in layers)


def test_activation_check_rejects_one_bad_token_despite_good_global_metrics():
    expected = torch.arange(1, 9).float().repeat(4096, 1)
    actual = expected.clone()
    actual[100] *= -1
    thresholds = {"relative_l2_max": 0.05, "cosine_min": 0.999, "minimum_token_cosine_min": 0.98}
    metrics = compare_activations(actual, expected, thresholds)
    assert metrics["relative_l2"] < thresholds["relative_l2_max"]
    assert metrics["cosine"] > thresholds["cosine_min"]
    assert metrics["minimum_token_cosine"] < 0
    assert not metrics["passed"]
    assert compare_activations(expected, expected, thresholds)["passed"]


class LengthTokenizer:
    chat_template = None

    def encode(self, text):
        return list(range(int(text.split(":")[0])))


@pytest.mark.parametrize("count", [8, 50])
def test_dataset_sample_is_reproducible_balanced_and_does_not_truncate(count):
    from collections import Counter

    from utils import sample_wildchat_rows

    def row(text, **extra):
        return {"conversation": [{"role": "user", "content": text, "language": "English"}], **extra}

    rows = [row(f"{length}:{i}") for length in (100, 500, 1500, 2500) for i in range(16)]
    # These must never enter the sample, including an overlong prompt.
    rows += [row("100:toxic", toxic=True), row("500:redacted", redacted=True), row("4000:long")]
    rows += [rows[0]]
    a, scanned = sample_wildchat_rows(rows, LengthTokenizer(), count, 42, 3072)
    b, _ = sample_wildchat_rows(rows, LengthTokenizer(), count, 42, 3072)
    assert a == b
    assert scanned == len(rows)
    assert Counter(p["length_bucket"] for p in a) == {
        bucket: count // 4 + (bucket < count % 4) for bucket in range(4)
    }
    assert len({p["prompt_sha256"] for p in a}) == count
    for p in a:
        assert p["dataset_row"] < 64
        assert len(p["token_ids"]) == int(p["user_prompt"].split(":")[0])


def test_dataset_sample_fails_if_natural_long_examples_are_missing():
    from utils import sample_wildchat_rows

    with pytest.raises(ValueError, match="Length bucket"):
        sample_wildchat_rows([], LengthTokenizer(), 8, 42, 3072)


def test_reconstruction_rejects_missing_cached_prefix():
    from types import SimpleNamespace

    from utils import reconstruct_capture

    span = SimpleNamespace(request_id=0, tensor_start=0, tensor_end=2, token_start=2, token_end=4)
    capture = SimpleNamespace(
        layer_id=0,
        signal="post_residual",
        phase="prefill",
        spans=[span],
        token_ids=torch.tensor([12, 13]),
        tensor=torch.ones(2, 8),
    )
    with pytest.raises(ValueError, match="gap or overlap"):
        reconstruct_capture([capture], 0, "post_residual", 0, [10, 11, 12, 13])


def test_decode_gzip_roundtrip_is_bit_exact_and_checks_corruption(tmp_path):
    from minisgl.capture import DecodeCaptureWindow, DecodeToken

    from utils import DecodeGzipWriter, read_decode_request, read_decode_window

    values = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 1.2, -3], dtype=torch.float16
    ).reshape(1, 2, 3)
    writer = DecodeGzipWriter(tmp_path / "captures")
    writer(
        DecodeCaptureWindow(
            values,
            ((None, "lm_head_input"),),
            (DecodeToken(42, 7, 2, 0), DecodeToken(42, 8, 3, 1, True)),
        )
    )
    index = writer.close()
    recovered, tokens, keys = read_decode_request(writer.directory, index, 42)
    assert torch.equal(recovered.view(torch.uint8), values.view(torch.uint8))
    assert [token["token_id"] for token in tokens] == [7, 8]
    assert keys == [(None, "lm_head_input")]
    broken = dict(index["windows"][0], sha256="incorrect")
    with pytest.raises(ValueError, match="checksum"):
        read_decode_window(writer.directory, broken)
