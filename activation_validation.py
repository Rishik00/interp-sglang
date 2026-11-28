"""Shared prefill modes, capture points and activation L2 comparisons."""

from __future__ import annotations

ACTIVATION_METRICS = ["token_l2", "prompt_l2", "hf_token_norm", "token_relative_error"]


def summarize_referee_populations(prompts):
    """Summarize token ratios, pooling actual tokens rather than prompt summaries."""
    import torch

    if not prompts:
        raise ValueError("No referee prompts supplied")
    keys = [(row["layer"], row["signal"]) for row in prompts[0]["points"]]
    for prompt in prompts:
        if [(row["layer"], row["signal"]) for row in prompt["points"]] != keys:
            raise ValueError("Referee capture points differ across prompts")
        for point in prompt["points"]:
            for name in ("engine_vs_fp32", "hf_fp16_vs_fp32"):
                values = point[name]["token_relative_error"]
                if len(values) != len(prompt["token_ids"]) or any(
                    value is None for value in values
                ):
                    raise ValueError("Referee relative errors are incomplete or undefined")
    summaries, point_rows = [], []
    populations = [(str(prompt["dataset_row"]), [prompt]) for prompt in prompts] + [
        ("pooled", prompts)
    ]
    for population, selected in populations:
        point_statistics = []
        final = None
        for index, (layer, signal) in enumerate(keys):
            errors, raw, identities = {}, {}, []
            for prompt in selected:
                identities.extend(
                    (prompt["dataset_row"], position)
                    for position in range(len(prompt["token_ids"]))
                )
            for side, name in (("engine", "engine_vs_fp32"), ("hf_fp16", "hf_fp16_vs_fp32")):
                errors[side] = torch.tensor(
                    [
                        value
                        for prompt in selected
                        for value in prompt["points"][index][name]["token_relative_error"]
                    ],
                    dtype=torch.float64,
                )
                raw[side] = torch.tensor(
                    [
                        value
                        for prompt in selected
                        for value in prompt["points"][index][name]["token_l2"]
                    ],
                    dtype=torch.float64,
                )
            row = {"population": population, "layer": layer, "signal": signal}
            for side in errors:
                row[f"{side}_median"] = float(torch.quantile(errors[side], 0.5))
                row[f"{side}_max"] = float(errors[side].max())
            point_statistics.append(row)
            if signal == "lm_head_input":
                final = {"population": population, "tokens": len(identities)}
                worst = {}
                for side in errors:
                    quantiles = torch.quantile(
                        errors[side], torch.tensor([0.5, 0.99, 0.999], dtype=torch.float64)
                    )
                    for label, value in zip(("median", "p99", "p99_9"), quantiles.tolist()):
                        final[f"{side}_{label}"] = value
                    final[f"{side}_max"] = float(errors[side].max())
                    final[f"{side}_above_2_percent"] = int((errors[side] > 0.02).sum())
                    final[f"{side}_above_5_percent"] = int((errors[side] > 0.05).sum())
                    ranked = sorted(
                        zip(errors[side].tolist(), identities), key=lambda pair: (-pair[0], pair[1])
                    )
                    worst[side] = {identity for _, identity in ranked[:20]}
                final["engine_closer"] = int((raw["engine"] < raw["hf_fp16"]).sum())
                final["hf_fp16_closer"] = int((raw["hf_fp16"] < raw["engine"]).sum())
                final["equal_distance"] = int((raw["engine"] == raw["hf_fp16"]).sum())
                final["worst_20_overlap"] = len(worst["engine"] & worst["hf_fp16"])
        if final is None:
            raise ValueError("Referee final normalized state is missing")
        for statistic in ("max", "median"):
            final[f"points_engine_{statistic}_worse"] = sum(
                row[f"engine_{statistic}"] > row[f"hf_fp16_{statistic}"] for row in point_statistics
            )
            final[f"points_hf_fp16_{statistic}_worse"] = sum(
                row[f"hf_fp16_{statistic}"] > row[f"engine_{statistic}"] for row in point_statistics
            )
            final[f"points_{statistic}_equal"] = sum(
                row[f"engine_{statistic}"] == row[f"hf_fp16_{statistic}"]
                for row in point_statistics
            )
        final["capture_points"] = len(keys)
        summaries.append(final)
        point_rows.extend(point_statistics)
    return {
        "summaries": summaries,
        "point_statistics": point_rows,
        "quantile_interpolation": "linear",
        "thresholds": "strictly greater than 0.02 and 0.05",
        "worst_20_identity": "dataset row plus token position; ties ordered by that identity",
    }


def relative_error_from_norms(token_l2, hf_token_norm):
    """Divide CPU FP64 values without an epsilon; a zero HF norm is undefined."""
    import math

    if len(token_l2) != len(hf_token_norm):
        raise ValueError("Token errors and HF norms have different lengths")
    if any(not math.isfinite(value) or value < 0 for value in [*token_l2, *hf_token_norm]):
        raise ValueError("Token errors and HF norms must be finite and nonnegative")
    return [None if norm == 0 else error / norm for error, norm in zip(token_l2, hf_token_norm)]


def prefill_spans(token_count: int, mode: str, chunk_size: int = 256):
    """Define identical token boundaries for engine and HF; whole is one span."""
    if token_count < 1 or chunk_size < 1:
        raise ValueError("Token count and chunk size must be positive")
    if mode not in {"whole", "chunked"}:
        raise ValueError(f"Unknown prefill mode: {mode}")
    step = token_count if mode == "whole" else chunk_size
    return [(start, min(start + step, token_count)) for start in range(0, token_count, step)]


def capture_points(num_layers: int):
    """Two points per complete block plus one model-level final norm output."""
    if num_layers < 1:
        raise ValueError("Number of layers must be positive")
    return [
        (layer, signal)
        for layer in range(num_layers)
        for signal in ("attention_output", "post_residual")
    ] + [(None, "lm_head_input")]


def compare_activation_l2(actual, expected):
    """Return raw L2, per-token HF norm, and relative error against HF.

    Inputs are [tokens, hidden width]. Subtraction and reductions run in FP64
    on CPU. Relative error is raw token L2 divided by its HF norm, with no
    epsilon. A zero HF norm produces null relative error, including 0/0.
    """
    import torch

    if actual.shape != expected.shape or actual.ndim != 2 or actual.numel() == 0:
        raise ValueError("Expected equal, nonempty [tokens, hidden width] activation shapes")
    if actual.dtype != expected.dtype:
        raise ValueError("Activation dtypes differ")
    actual = actual.detach().to(device="cpu", dtype=torch.float64)
    expected = expected.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError("Cannot compare nonfinite activations")
    difference = actual - expected
    token_l2 = torch.linalg.vector_norm(difference, dim=-1).tolist()
    hf_token_norm = torch.linalg.vector_norm(expected, dim=-1).tolist()
    return {
        "token_l2": token_l2,
        "prompt_l2": float(torch.linalg.vector_norm(difference)),
        "hf_token_norm": hf_token_norm,
        "token_relative_error": relative_error_from_norms(token_l2, hf_token_norm),
    }


def capture_hf_prefill(reference, token_ids, mode: str, chunk_size: int = 256, device="cuda"):
    """Hook the same points on HF, with cache and absolute positions in both modes.

    Every chunk sees the full prefix through past_key_values; it is not an
    independent short prompt. Clone hook outputs before later in-place writes.
    """
    import torch

    spans = prefill_spans(len(token_ids), mode, chunk_size)
    points = capture_points(len(reference.model.layers))
    pieces = {point: [] for point in points}
    handles = []

    def make_hook(point):
        def save_output(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.ndim != 3 or value.shape[0] != 1:
                raise ValueError("HF capture must have shape [1, tokens, hidden width]")
            pieces[point].append(value.detach().squeeze(0).to("cpu").clone())

        return save_output

    # Whole-block output is before the model's final norm, including the last block.
    try:
        for layer_index, block in enumerate(reference.model.layers):
            handles.append(
                block.self_attn.o_proj.register_forward_hook(
                    make_hook((layer_index, "attention_output"))
                )
            )
            handles.append(block.register_forward_hook(make_hook((layer_index, "post_residual"))))
        handles.append(
            reference.model.norm.register_forward_hook(make_hook((None, "lm_head_input")))
        )
        cache = None
        with torch.inference_mode():
            for chunk_index, (start, stop) in enumerate(spans):
                positions = torch.arange(start, stop, device=device)
                result = reference(
                    input_ids=torch.tensor([token_ids[start:stop]], device=device),
                    attention_mask=torch.ones((1, stop), dtype=torch.long, device=device),
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = result.past_key_values
                if cache is None:
                    raise ValueError("HF did not return a KV cache")
                if cache.get_seq_length() != stop:
                    raise ValueError("HF KV cache does not cover the complete prompt prefix")
                for point, chunks in pieces.items():
                    if len(chunks) != chunk_index + 1 or chunks[-1].shape[0] != stop - start:
                        raise ValueError(f"HF capture missing, duplicated or misaligned at {point}")
        return {point: torch.cat(chunks, dim=0) for point, chunks in pieces.items()}
    finally:
        for handle in handles:
            handle.remove()


def validate_engine_capture_spans(captures, points, expected_spans):
    """Reject missing/extra points, duplicates, unexpected requests and chunk boundaries."""
    if {(capture.layer_id, capture.signal) for capture in captures} != set(points):
        raise ValueError("Engine capture points differ from the agreed capture plan")
    for point in points:
        spans = []
        for capture in captures:
            if (capture.layer_id, capture.signal) != point:
                continue
            if capture.phase != "prefill":
                raise ValueError("Expected prefill captures only")
            for span in capture.spans:
                if span.request_id != 0:
                    raise ValueError("Expected one request at a time")
                if (
                    span.tensor_end - span.tensor_start != span.token_end - span.token_start
                    or span.tensor_start < 0
                    or span.tensor_end > capture.tensor.shape[0]
                ):
                    raise ValueError("Engine capture tensor span is invalid")
                spans.append((span.token_start, span.token_end))
        if sorted(spans) != expected_spans:
            raise ValueError(f"Engine chunk boundaries differ from HF at {point}: {spans}")


def copy_trace_tensor(value, token_count, heads_first=False):
    """Own an immutable CPU snapshot, preserving token order across head layouts."""
    if isinstance(value, tuple):
        value = value[0]
    if heads_first:
        value = value.transpose(1, 2)
    return value.detach().reshape(token_count, -1).cpu().clone()


def hf_rmsnorm(hidden, weight, epsilon):
    """Call the installed HF Qwen3 implementation with the engine's loaded weight."""
    from types import SimpleNamespace

    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    return Qwen3RMSNorm.forward(SimpleNamespace(weight=weight, variance_epsilon=epsilon), hidden)


def install_hf_operation(llm, family, config, token_count):
    """Replace one family on current engine inputs; restore every temporary edit."""
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as functional
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    if family not in {"norm", "qkv", "rope", "attention", "silu"}:
        raise ValueError(f"Unknown HF replacement family: {family}")
    restores = []

    def replace(owner, attribute, replacement):
        restores.append((owner, attribute, getattr(owner, attribute)))
        setattr(owner, attribute, replacement)

    model = llm.engine.model.model
    blocks = model.layers.op_list
    if family == "norm":
        # Preserve the deferred residual interface and its standard capture points.
        for owner in [
            model.norm,
            *[
                norm
                for block in blocks
                for norm in (block.input_layernorm, block.post_attention_layernorm)
            ],
        ]:

            def normalize(hidden, residual=None, operator=owner):
                summed = hidden if residual is None else residual + hidden
                return hf_rmsnorm(summed, operator.weight, operator.eps), summed

            replace(owner, "forward", normalize)
        for block in blocks:
            for owner in (block.self_attn.q_norm, block.self_attn.k_norm):

                def normalize_inplace(hidden, operator=owner):
                    hidden.copy_(hf_rmsnorm(hidden, operator.weight, operator.eps))

                replace(owner, "forward_inplace", normalize_inplace)
    elif family == "qkv":
        for block in blocks:
            owner, attention = block.self_attn.qkv_proj, block.self_attn.attn
            sizes = [attention.qo_attn_dim, attention.kv_attn_dim, attention.kv_attn_dim]

            def project(hidden, operator=owner, output_sizes=sizes):
                if operator.bias is not None:
                    raise ValueError("This Qwen3 experiment expects no projection biases")
                return torch.cat(
                    [
                        functional.linear(hidden.unsqueeze(0), weight)[0]
                        for weight in operator.weight.split(output_sizes, dim=0)
                    ],
                    dim=-1,
                )

            replace(owner, "forward", project)
    elif family == "rope":
        # All engine blocks share a rotary object. Replace it once and use HF's
        # own frequency calculation, FP16 cos/sin, and rotary arithmetic.
        rotary = Qwen3RotaryEmbedding(config).to("cuda")
        positions = torch.arange(token_count, device="cuda").unsqueeze(0)
        dummy = torch.empty(1, token_count, 1, device="cuda", dtype=torch.float16)
        cosine, sine = rotary(dummy, positions)

        def rotate(positions, query, key):
            if not torch.equal(positions, torch.arange(token_count, device=positions.device)):
                raise ValueError("HF rotary replacement expects cold whole-prompt positions")
            query_heads = query.view(1, token_count, -1, config.head_dim).transpose(1, 2)
            key_heads = key.view(1, token_count, -1, config.head_dim).transpose(1, 2)
            rotated_query, rotated_key = apply_rotary_pos_emb(query_heads, key_heads, cosine, sine)
            query.copy_(rotated_query.transpose(1, 2).reshape_as(query))
            key.copy_(rotated_key.transpose(1, 2).reshape_as(key))
            return query, key

        seen = set()
        for block in blocks:
            owner = block.self_attn.attn.rotary
            if id(owner) not in seen:
                replace(owner, "forward", rotate)
                seen.add(id(owner))
    elif family == "attention":
        backend = llm.engine.attn_backend
        module = SimpleNamespace(
            num_key_value_groups=config.num_attention_heads // config.num_key_value_heads,
            training=False,
        )
        mask = torch.full(
            (token_count, token_count),
            torch.finfo(torch.float16).min,
            device="cuda",
            dtype=torch.float16,
        ).triu_(diagonal=1)[None, None]

        def attend(query, key, value, layer_id, batch):
            if len(batch.reqs) != 1 or any(
                request.cached_len != 0 or request.extend_len != token_count
                for request in batch.reqs
            ):
                raise ValueError("HF attention replacement expects one cold whole-prompt request")
            backend.kvcache.store_kv(key, value, batch.out_loc, layer_id)
            query_heads = query.view(1, token_count, -1, config.head_dim).transpose(1, 2)
            key_heads = key.view(1, token_count, -1, config.head_dim).transpose(1, 2)
            value_heads = value.view(1, token_count, -1, config.head_dim).transpose(1, 2)
            output, _ = eager_attention_forward(
                module,
                query_heads,
                key_heads,
                value_heads,
                mask,
                scaling=config.head_dim**-0.5,
                dropout=0.0,
            )
            return output.reshape(token_count, config.num_attention_heads, config.head_dim)

        replace(backend, "forward", attend)
    else:

        def activate(gate_up):
            gate, up = gate_up.chunk(2, dim=-1)
            return functional.silu(gate) * up

        for block in blocks:
            replace(block.mlp, "act_fn", activate)

    def restore():
        for owner, attribute, original in reversed(restores):
            setattr(owner, attribute, original)
        restores.clear()

    return restore


def install_engine_trace(llm, token_count):
    """Temporarily observe the original operations, cloning before in-place reuse."""
    import torch
    from sgl_interp.core import get_global_ctx

    trace, restores, metadata = {}, [], {}

    def save(key, value):
        if key in trace:
            raise ValueError(f"Repeated diagnostic capture: {key}")
        trace[key] = copy_trace_tensor(value, token_count)

    def wrap(owner, name, observer):
        original = getattr(owner, name)

        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            observer(args, result)
            return result

        setattr(owner, name, observe)
        restores.append((owner, name, original))

    model = llm.engine.model.model
    wrap(model.embed_tokens, "forward", lambda args, result: save((None, "embedding"), result))
    for index, block in enumerate(model.layers.op_list):

        def norm_output(args, result, layer=index, prefix="input"):
            save((layer, f"{prefix}_norm"), result[0])
            save((layer, "block_input" if prefix == "input" else "attention_residual"), result[1])

        wrap(block.input_layernorm, "forward", norm_output)
        wrap(
            block.post_attention_layernorm,
            "forward",
            lambda args, result, layer=index: norm_output(args, result, layer, "post"),
        )
        attention = block.self_attn

        def qkv_output(args, result, layer=index, operator=attention.attn):
            for name, part in zip(
                ("q_raw", "k_raw", "v_raw"),
                result.split(
                    [operator.qo_attn_dim, operator.kv_attn_dim, operator.kv_attn_dim], dim=-1
                ),
            ):
                save((layer, name), part)

        wrap(attention.qkv_proj, "forward", qkv_output)
        for name in ("q", "k"):
            wrap(
                getattr(attention, f"{name}_norm"),
                "forward_inplace",
                lambda args, result, layer=index, signal=f"{name}_norm": save(
                    (layer, signal), args[0]
                ),
            )
        wrap(
            attention.o_proj,
            "forward",
            lambda args, result, layer=index: save((layer, "attention_output"), result),
        )

        def gate_up_output(args, result, layer=index):
            gate, up = result.chunk(2, dim=-1)
            save((layer, "gate"), gate)
            save((layer, "up"), up)

        wrap(block.mlp.gate_up_proj, "forward", gate_up_output)
        wrap(
            block.mlp,
            "act_fn",
            lambda args, result, layer=index: save((layer, "gated_activation"), result),
        )
        wrap(
            block.mlp.down_proj,
            "forward",
            lambda args, result, layer=index: save((layer, "mlp_output"), result),
        )
    wrap(model.norm, "forward", lambda args, result: save((None, "lm_head_input"), result[0]))
    backend = llm.engine.attn_backend
    original_attention = backend.forward

    def observe_attention(query, key, value, layer_id, batch):
        positions = batch.positions.detach().cpu()
        if not torch.equal(positions, torch.arange(token_count)):
            raise ValueError("Engine token positions are not the whole prompt's absolute positions")
        if any(
            request.cached_len != 0 or request.extend_len != token_count for request in batch.reqs
        ):
            raise ValueError("Expected cold, unchunked prefill")
        save((layer_id, "q_rope"), query)
        save((layer_id, "k_rope"), key)
        output = original_attention(query, key, value, layer_id, batch)
        save((layer_id, "attention_context"), output)
        metadata["positions"] = positions.tolist()
        return output

    backend.forward = observe_attention
    restores.append((backend, "forward", original_attention))
    original_plan = backend.prefill_wrapper.plan

    def observe_plan(*args, **kwargs):
        if not kwargs.get("causal") or kwargs["pos_encoding_mode"] != "NONE":
            raise ValueError("Engine attention plan is not causal with external RoPE")
        if kwargs["qo_indptr"].cpu().tolist() != [0, token_count]:
            raise ValueError("Engine query spans differ")
        if kwargs["paged_kv_indptr"].cpu().tolist() != [0, token_count]:
            raise ValueError("Engine KV spans differ")
        metadata["causal"] = True
        metadata["kv_indices"] = kwargs["paged_kv_indices"].cpu().tolist()
        if len(set(metadata["kv_indices"])) != token_count:
            raise ValueError("Repeated KV-cache locations")
        return original_plan(*args, **kwargs)

    backend.prefill_wrapper.plan = observe_plan
    restores.append((backend.prefill_wrapper, "plan", original_plan))

    def restore():
        for owner, name, original in reversed(restores):
            setattr(owner, name, original)
        restores.clear()

    # Avoid retaining the engine through a global context after shutdown.
    assert get_global_ctx() is llm.engine.ctx
    return trace, metadata, restore


def install_hf_trace(reference, token_count):
    """Observe the matching HF modules and its original rotary/attention functions."""
    import torch
    import transformers.models.qwen3.modeling_qwen3 as modeling

    trace, handles, metadata = {}, [], {}
    active_layer = [None]

    def save(key, value, heads_first=False):
        if key in trace:
            raise ValueError(f"Repeated HF diagnostic capture: {key}")
        trace[key] = copy_trace_tensor(value, token_count, heads_first)

    def hook(module, key, before=False):
        if before:
            handles.append(
                module.register_forward_pre_hook(lambda module, args: save(key, args[0]))
            )
        else:
            handles.append(
                module.register_forward_hook(lambda module, args, output: save(key, output))
            )

    hook(reference.model.embed_tokens, (None, "embedding"))
    for index, block in enumerate(reference.model.layers):

        def enter_block(module, args, layer=index):
            active_layer[0] = layer

        handles.append(block.register_forward_pre_hook(enter_block))
        hook(block.input_layernorm, (index, "block_input"), before=True)
        hook(block.input_layernorm, (index, "input_norm"))
        for name in ("q", "k", "v"):
            hook(getattr(block.self_attn, f"{name}_proj"), (index, f"{name}_raw"))
        for name in ("q", "k"):
            hook(getattr(block.self_attn, f"{name}_norm"), (index, f"{name}_norm"))
        hook(block.self_attn.o_proj, (index, "attention_context"), before=True)
        hook(block.self_attn.o_proj, (index, "attention_output"))
        hook(block.post_attention_layernorm, (index, "attention_residual"), before=True)
        hook(block.post_attention_layernorm, (index, "post_norm"))
        hook(block.mlp.gate_proj, (index, "gate"))
        hook(block.mlp.up_proj, (index, "up"))
        hook(block.mlp.down_proj, (index, "gated_activation"), before=True)
        hook(block.mlp.down_proj, (index, "mlp_output"))
        hook(block, (index, "post_residual"))
    hook(reference.model.norm, (None, "lm_head_input"))
    original_rope = modeling.apply_rotary_pos_emb
    original_attention = modeling.eager_attention_forward

    def observe_rope(*args, **kwargs):
        result = original_rope(*args, **kwargs)
        save((active_layer[0], "q_rope"), result[0], heads_first=True)
        save((active_layer[0], "k_rope"), result[1], heads_first=True)
        return result

    def observe_attention(module, query, key, value, attention_mask, scaling, **kwargs):
        if attention_mask is None:
            raise ValueError("Expected an explicit HF causal mask")
        mask = attention_mask[0, 0].detach().cpu()
        blocked = torch.triu(torch.ones(token_count, token_count, dtype=torch.bool), diagonal=1)
        if not bool((mask[blocked] < -10000).all() and (mask[~blocked] == 0).all()):
            raise ValueError("HF mask is not the expected whole-prompt causal mask")
        metadata["causal"] = True
        return original_attention(module, query, key, value, attention_mask, scaling, **kwargs)

    modeling.apply_rotary_pos_emb = observe_rope
    modeling.eager_attention_forward = observe_attention

    def restore():
        for handle in handles:
            handle.remove()
        modeling.apply_rotary_pos_emb = original_rope
        modeling.eager_attention_forward = original_attention

    return trace, metadata, restore


def replay_engine_operations(hf_trace, weights, rope_cache, config):
    """Run each engine operation on its HF inputs, without accumulated upstream error."""
    import torch
    import torch.nn.functional as functional
    from flashinfer import (
        BatchPrefillWithPagedKVCacheWrapper,
        apply_rope_with_cos_sin_cache_inplace,
        fused_add_rmsnorm,
        rmsnorm,
    )
    from sgl_interp.layers import silu_and_mul

    count = hf_trace[(None, "embedding")].shape[0]
    results = {}
    positions = torch.arange(count, device="cuda")
    rope_cache = rope_cache.to("cuda")
    # Match the serving attention wrapper: NHD, page size one, FA2, causal, external RoPE.
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    attention = BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
    attention.plan(
        qo_indptr=torch.tensor([0, count], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, count], dtype=torch.int32),
        paged_kv_indices=torch.arange(count, device="cuda", dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([1], dtype=torch.int32),
        num_qo_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim_qk=config.head_dim,
        page_size=1,
        pos_encoding_mode="NONE",
        seq_lens=torch.tensor([count], dtype=torch.int32),
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        causal=True,
    )

    def save(key, value):
        results[key] = value.detach().reshape(count, -1).cpu().clone()

    def tensor(layer, name):
        return hf_trace[(layer, name)].to("cuda")

    def normalize(branch, residual, weight):
        if residual is None:
            return rmsnorm(branch, weight, config.rms_norm_eps), branch
        fused_add_rmsnorm(branch, residual, weight, config.rms_norm_eps)
        return branch, residual

    with torch.inference_mode():
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            local_weights = {
                name.removeprefix(prefix): value.to("cuda")
                for name, value in weights.items()
                if name.startswith(prefix)
            }
            if layer == 0:
                branch, residual = tensor(None, "embedding"), None
            else:
                branch, residual = (
                    tensor(layer - 1, "mlp_output"),
                    tensor(layer - 1, "attention_residual"),
                )
            normalized, summed = normalize(
                branch, residual, local_weights["input_layernorm.weight"]
            )
            save((layer, "input_norm"), normalized)
            save((layer, "block_input"), summed)
            if layer > 0:
                save((layer - 1, "post_residual"), summed)
            merged = functional.linear(
                tensor(layer, "input_norm"), local_weights["self_attn.qkv_proj.weight"]
            )
            sizes = [hf_trace[(layer, name)].shape[1] for name in ("q_raw", "k_raw", "v_raw")]
            for name, value in zip(("q_raw", "k_raw", "v_raw"), merged.split(sizes, dim=-1)):
                save((layer, name), value)
            for name in ("q", "k"):
                value = tensor(layer, f"{name}_raw").view(count, -1, config.head_dim)
                save(
                    (layer, f"{name}_norm"),
                    rmsnorm(
                        value, local_weights[f"self_attn.{name}_norm.weight"], config.rms_norm_eps
                    ),
                )
            query, key = tensor(layer, "q_norm"), tensor(layer, "k_norm")
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=config.head_dim,
                cos_sin_cache=rope_cache,
            )
            save((layer, "q_rope"), query)
            save((layer, "k_rope"), key)
            query = tensor(layer, "q_rope").view(count, config.num_attention_heads, config.head_dim)
            key = tensor(layer, "k_rope").view(
                count, 1, config.num_key_value_heads, config.head_dim
            )
            value = tensor(layer, "v_raw").view_as(key)
            save((layer, "attention_context"), attention.run(query, (key, value)))
            save(
                (layer, "attention_output"),
                functional.linear(
                    tensor(layer, "attention_context"), local_weights["self_attn.o_proj.weight"]
                ),
            )
            normalized, summed = normalize(
                tensor(layer, "attention_output"),
                tensor(layer, "block_input"),
                local_weights["post_attention_layernorm.weight"],
            )
            save((layer, "post_norm"), normalized)
            save((layer, "attention_residual"), summed)
            gate_up = functional.linear(
                tensor(layer, "post_norm"), local_weights["mlp.gate_up_proj.weight"]
            )
            for name, value in zip(("gate", "up"), gate_up.chunk(2, dim=-1)):
                save((layer, name), value)
            save(
                (layer, "gated_activation"),
                silu_and_mul(torch.cat([tensor(layer, "gate"), tensor(layer, "up")], dim=-1)),
            )
            save(
                (layer, "mlp_output"),
                functional.linear(
                    tensor(layer, "gated_activation"), local_weights["mlp.down_proj.weight"]
                ),
            )
        normalized, summed = normalize(
            tensor(config.num_hidden_layers - 1, "mlp_output"),
            tensor(config.num_hidden_layers - 1, "attention_residual"),
            weights["model.norm.weight"].to("cuda"),
        )
        save((None, "lm_head_input"), normalized)
        save((config.num_hidden_layers - 1, "post_residual"), summed)
    return results
