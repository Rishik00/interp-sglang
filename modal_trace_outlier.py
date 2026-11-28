"""Deep dive on one outlier token: where engine and HF first differ, and what grows it.

Works on one (prompt, token) picked from a saved engine-vs-HF run. Four stages, run in
order; each later stage reuses the tensors the "trace" stage saved on the Modal volume:

  trace     (T4)  engine and HF traced at 18 points per block, plus each engine
                  operation replayed on HF's own inputs
  profile   (T4)  swap one operation family at a time for HF's version
  relative  (CPU) relative error (L2 / HF vector norm) for the trace and the swaps
  fp32      (T4)  one HF FP32 forward: is engine FP16 or HF FP16 closer at this token?

Run:
    # 1. Pick the target. Default: the token with the largest final-state L2 in the run.
    python modal_trace_outlier.py --pick-target \
        --run artifacts/matched-prefill/Qwen--Qwen3-1.7B-whole-20261128T023424Z.jsonl \
        --workload artifacts/matched-prefill/Qwen--Qwen3-1.7B-workload-20261128T023424Z.json
    # 2. Run the stages in order.
    python modal_trace_outlier.py --stage trace
    python modal_trace_outlier.py --stage profile
    python modal_trace_outlier.py --stage relative
    python modal_trace_outlier.py --stage fp32

Dense Qwen3 only (the detailed hooks and operation swaps are written for Qwen3).
Outputs land in artifacts/matched-prefill/ as <stage prefix>-<time>.json, with a
<stage prefix>-completion.json pointer that the next stage reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import modal

from activation_validation import (
    capture_points,
    compare_activation_l2,
    hf_rmsnorm,
    install_engine_trace,
    install_hf_operation,
    install_hf_trace,
    relative_error_from_norms,
    replay_engine_operations,
    validate_engine_capture_spans,
)
from config import LOCAL_REPO, REMOTE_CACHE, app, cache_volume, dependency_image
from utils import hugging_face_modal_secrets, mem_cleanup, reconstruct_capture

OUTPUT_DIR = LOCAL_REPO / "artifacts/matched-prefill"
TARGET_PATH = OUTPUT_DIR / "outlier-target.json"
# Stage name -> artifact prefix. The prefixes match the 2026-11-28 artifacts.
STAGE_PREFIX = {
    "trace": "outlier",
    "profile": "operation-profile",
    "relative": "relative-outlier",
    "fp32": "fp32-referee",
}


# Stage "trace" (T4). Runs the engine twice (plain, then with extra observation hooks
# that must not change the 57 standard captures), then HF FP16 eager once with matching
# hooks: 506 points per token. Then replays every engine operation on HF's own saved
# inputs, which shows the error each operation adds by itself. Saves engine.pt, hf.pt
# and shared-input-engine.pt on the Modal volume; downloads only the L2 summary.
@app.function(
    image=dependency_image,
    gpu="T4",
    cpu=2,
    memory=65536,
    timeout=3600,
    volumes={str(REMOTE_CACHE): cache_volume},
    single_use_containers=True,
)
def diagnose_outlier(target, output_name, profile=False, reference_trace_path=None):
    """Trace one original prompt, verify capture neutrality, and replay on HF inputs."""
    if profile:
        return profile_outlier_operations(target, output_name, reference_trace_path)
    import torch
    from huggingface_hub import snapshot_download
    from sgl_interp.capture import ActivationCaptureConfig
    from sgl_interp.core import SamplingParams
    from sgl_interp.layers.rotary import get_rope
    from sgl_interp.llm import LLM
    from transformers import AutoConfig, AutoModelForCausalLM

    snapshot = snapshot_download(
        target["model"],
        revision=target["model_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    config = AutoConfig.from_pretrained(snapshot)
    if config.model_type != "qwen3":
        raise ValueError("Detailed operation tracing currently targets dense Qwen3")
    ids = target["prompt"]["token_ids"]
    position = target["token_position"]
    assert ids[position] == target["token_id"]
    llm = LLM(
        model_path=snapshot,
        dtype=torch.float16,
        cache_type="naive",
        max_extend_tokens=len(ids),
        max_running_req=1,
        cuda_graph_max_bs=1,
        max_seq_len_override=4096,
        memory_ratio=0.45,
        use_pynccl=False,
        activation_capture=ActivationCaptureConfig(
            layer_ids=tuple(range(config.num_hidden_layers)), capture_lm_head_input=True
        ),
    )
    sampling = SamplingParams(top_k=1, temperature=0, max_tokens=1, ignore_eos=True)
    points = capture_points(config.num_hidden_layers)
    controls = []
    try:
        # First run is uninstrumented. A second run adds observations only; their
        # standard captures must be exactly equal before interpreting the trace.
        llm.generate([ids], sampling)
        baseline = llm.take_activation_captures()
        weights = {
            name: value.detach().cpu().clone()
            for name, value in llm.engine.model.state_dict().items()
            if name.startswith(("model.layers.", "model.norm."))
        }
        rope_cache = (
            llm.engine.model.model.layers.op_list[0]
            .self_attn.attn.rotary._cos_sin_cache.cpu()
            .clone()
        )
        engine_trace, engine_metadata, restore_engine = install_engine_trace(llm, len(ids))
        try:
            llm.generate([ids], sampling)
            captured = llm.take_activation_captures()
        finally:
            restore_engine()
        validate_engine_capture_spans(captured, points, [(0, len(ids))])
        for layer, signal in points:
            actual = reconstruct_capture(captured, layer, signal, 0, ids)
            before = reconstruct_capture(baseline, layer, signal, 0, ids)
            metrics = compare_activation_l2(actual, before)
            if metrics["prompt_l2"] != 0:
                raise ValueError(
                    f"Instrumentation changed the standard capture at {(layer, signal)}"
                )
            if (layer, signal) in engine_trace:
                if compare_activation_l2(engine_trace[(layer, signal)], actual)["prompt_l2"] != 0:
                    raise ValueError("Diagnostic copy differs from the standard capture")
            engine_trace[(layer, signal)] = actual.clone()
            controls.append(
                {
                    "layer": layer,
                    "signal": signal,
                    "token_l2": metrics["token_l2"],
                    "prompt_l2": metrics["prompt_l2"],
                }
            )
        baseline.clear()
        captured.clear()
    finally:
        llm.shutdown()
    del llm, actual, before
    get_rope.cache_clear()
    mem_cleanup()

    # Execute HF's original full-prompt forward with its standard mask and hooks.
    reference = (
        AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=torch.float16, attn_implementation="eager"
        )
        .eval()
        .to("cuda")
    )
    hf_trace, hf_metadata, restore_hf = install_hf_trace(reference, len(ids))
    try:
        with torch.inference_mode():
            positions = torch.arange(len(ids), device="cuda")
            output = reference(
                input_ids=torch.tensor([ids], device="cuda"),
                attention_mask=torch.ones(1, len(ids), device="cuda", dtype=torch.long),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                use_cache=True,
                logits_to_keep=1,
            )
        assert output.past_key_values.get_seq_length() == len(ids)
    finally:
        restore_hf()
    del reference, output
    mem_cleanup()
    if engine_trace.keys() != hf_trace.keys():
        raise ValueError(
            f"Detailed capture point mismatch: {engine_trace.keys() ^ hf_trace.keys()}"
        )
    print(
        "Detailed engine/HF captures complete; replaying operations with identical HF inputs.",
        flush=True,
    )
    replay = replay_engine_operations(hf_trace, weights, rope_cache, config)

    def compare_trace(trace):
        rows = []
        for (layer, signal), actual_value in trace.items():
            metrics = compare_activation_l2(actual_value, hf_trace[(layer, signal)])
            rows.append(
                {
                    "layer": layer,
                    "signal": signal,
                    **metrics,
                    "target_token_l2": metrics["token_l2"][position],
                }
            )
        return rows

    rows = compare_trace(engine_trace)
    replay_rows = compare_trace(replay)
    final = next(row for row in rows if row["signal"] == "lm_head_input")
    summary = {
        "model": target["model"],
        "model_revision": target["model_revision"],
        "dataset_row": target["dataset_row"],
        "input_tokens": len(ids),
        "token_position": position,
        "token_id": ids[position],
        "token_piece": llm_token_piece(snapshot, ids[position]),
        "mode": "whole",
        "dtype": "float16",
        "metric_dtype": "float64",
        "saved_target_token_l2": target["token_l2"],
        "rerun_target_token_l2": final["target_token_l2"],
        "engine_metadata": engine_metadata,
        "hf_metadata": hf_metadata,
        "controls": controls,
        "rows": rows,
        "shared_input_rows": replay_rows,
    }
    # Keep actual tensors on the persistent volume; download only the L2 report.
    directory = Path(str(REMOTE_CACHE)) / "matched-prefill" / output_name
    directory.mkdir(parents=True, exist_ok=True)
    tensor_paths = []
    for name, values in (
        ("engine", engine_trace),
        ("hf", hf_trace),
        ("shared-input-engine", replay),
    ):
        path = directory / f"{name}.pt"
        torch.save(values, path)
        tensor_paths.append(str(path.relative_to(Path(str(REMOTE_CACHE)))))
    summary["tensor_paths"] = tensor_paths
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    cache_volume.commit()
    return {
        "remote_path": str(path.relative_to(Path(str(REMOTE_CACHE)))),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensor_paths": tensor_paths,
    }


def llm_token_piece(snapshot, token_id):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot).convert_ids_to_tokens(token_id)


# Stage "profile" (T4, called through diagnose_outlier). Swaps one operation family
# at a time (norm, qkv, rope, attention, silu) across all blocks for HF's version, then
# restores the engine and checks it reproduces the baseline exactly. Also probes block 0
# input normalization directly, to show where the first rounding difference comes from.
def profile_outlier_operations(target, output_name, reference_trace_path):
    """Measure one HF operation replacement at a time against the saved reference."""
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from sgl_interp.capture import ActivationCaptureConfig
    from sgl_interp.core import SamplingParams
    from sgl_interp.llm import LLM
    from transformers import AutoConfig

    ids, position = target["prompt"]["token_ids"], target["token_position"]
    reference_path = Path(str(REMOTE_CACHE)) / reference_trace_path
    hf_trace = torch.load(reference_path, map_location="cpu", weights_only=True)
    if hf_trace[(None, "embedding")].shape[0] != len(ids):
        raise ValueError("Reference trace length differs from the target prompt")
    snapshot = snapshot_download(
        target["model"],
        revision=target["model_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    config = AutoConfig.from_pretrained(snapshot)
    if config.model_type != "qwen3" or ids[position] != target["token_id"]:
        raise ValueError("Profile expects the original Qwen3 target")
    llm = LLM(
        model_path=snapshot,
        dtype=torch.float16,
        cache_type="naive",
        max_extend_tokens=len(ids),
        max_running_req=1,
        cuda_graph_max_bs=1,
        max_seq_len_override=4096,
        memory_ratio=0.45,
        use_pynccl=False,
        activation_capture=ActivationCaptureConfig(
            layer_ids=tuple(range(config.num_hidden_layers)), capture_lm_head_input=True
        ),
    )
    import flashinfer

    sampling = SamplingParams(top_k=1, temperature=0, max_tokens=1, ignore_eos=True)
    points, cases = capture_points(config.num_hidden_layers), []

    def metrics(actual, expected):
        result = compare_activation_l2(actual.cpu(), expected.cpu())
        return {**result, "target_token_l2": result["token_l2"][position]}

    try:
        # Probe the first normalization directly, with no upstream differences.
        # Moving the cast isolates one arithmetic choice; unit weights remove it.
        with torch.inference_mode():
            operator = llm.engine.model.model.layers.op_list[0].input_layernorm
            hidden = hf_trace[(None, "embedding")].to("cuda")
            engine_output, _ = operator.forward(hidden.clone())
            hf_output = hf_rmsnorm(hidden, operator.weight, operator.eps)
            normalized = hidden.float() * torch.rsqrt(
                hidden.float().square().mean(-1, keepdim=True) + operator.eps
            )
            cast_after_weight = (normalized * operator.weight.float()).to(hidden.dtype)
            unit_weight = torch.ones_like(operator.weight)
            norm_probe = {
                "engine_vs_hf": metrics(engine_output, hf_output),
                "engine_vs_cast_after_weight": metrics(engine_output, cast_after_weight),
                "engine_vs_hf_unit_weight": metrics(
                    operator.rmsnorm(hidden, unit_weight, operator.eps),
                    hf_rmsnorm(hidden, unit_weight, operator.eps),
                ),
                "hf_vs_saved_hf": metrics(hf_output, hf_trace[(0, "input_norm")]),
            }
        del hidden, engine_output, hf_output, normalized, cast_after_weight, unit_weight, operator
        baseline = None
        for family in ("baseline", "norm", "qkv", "rope", "attention", "silu", "restored_baseline"):
            print(f"Running operation profile: {family}", flush=True)
            restore = (
                None
                if family in {"baseline", "restored_baseline"}
                else install_hf_operation(llm, family, config, len(ids))
            )
            try:
                llm.generate([ids], sampling)
                captures = llm.take_activation_captures()
            finally:
                if restore is not None:
                    restore()
            validate_engine_capture_spans(captures, points, [(0, len(ids))])
            rows = []
            current = {}
            for layer, signal in points:
                actual = reconstruct_capture(captures, layer, signal, 0, ids)
                current[(layer, signal)] = actual
                rows.append(
                    {"layer": layer, "signal": signal, **metrics(actual, hf_trace[(layer, signal)])}
                )
            if family == "baseline":
                baseline = current
                keyed = {(row["layer"], row["signal"]): row for row in rows}
                for saved in target["saved_trace"]:
                    repeated = keyed[(saved["layer"], saved["signal"])]
                    if (
                        repeated["target_token_l2"] != saved["token_l2"]
                        or repeated["prompt_l2"] != saved["prompt_l2"]
                    ):
                        raise ValueError("Profile baseline did not reproduce the saved run")
            elif family == "restored_baseline":
                if any(
                    metrics(current[point], baseline[point])["prompt_l2"] != 0 for point in points
                ):
                    raise ValueError("Restoring operations did not recover the original baseline")
            cases.append({"family": family, "rows": rows})
            del captures, actual, current
    finally:
        llm.shutdown()
    summary = {
        "model": target["model"],
        "model_revision": target["model_revision"],
        "dataset_row": target["dataset_row"],
        "token_position": position,
        "token_id": ids[position],
        "input_tokens": len(ids),
        "mode": "whole",
        "dtype": "float16",
        "metric_dtype": "float64",
        "reference_trace_path": reference_trace_path,
        "norm_probe": norm_probe,
        "cases": cases,
        "restored_baseline_exact": True,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "flashinfer": flashinfer.__version__,
        },
    }
    path = Path(str(REMOTE_CACHE)) / "matched-prefill" / output_name / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    cache_volume.commit()
    return {
        "remote_path": str(path.relative_to(Path(str(REMOTE_CACHE)))),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# Stage "relative" (Modal CPU). Recomputes every traced point from the saved tensors
# and turns raw L2 into relative error (L2 divided by the HF vector norm), for the trace
# and for every operation-swap case. No model forward.
@app.function(
    image=dependency_image,
    cpu=2,
    memory=65536,
    timeout=1800,
    volumes={str(REMOTE_CACHE): cache_volume},
)
def summarize_outlier_relative(target, output_name, reference_trace_path, profile_path):
    """Read saved tensors on Modal CPU; no model forward or local tensor download."""
    import torch

    torch.set_num_threads(2)
    directory = Path(str(REMOTE_CACHE)) / Path(reference_trace_path).parent
    engine = torch.load(directory / "engine.pt", map_location="cpu", weights_only=True)
    hf = torch.load(directory / "hf.pt", map_location="cpu", weights_only=True)
    profile = json.loads((Path(str(REMOTE_CACHE)) / profile_path).read_text())
    if engine.keys() != hf.keys() or any(
        profile[key] != target[key]
        for key in ("model", "model_revision", "dataset_row", "token_position", "token_id")
    ):
        raise ValueError("Saved tensors/profile do not describe the requested target")
    position, count = target["token_position"], len(target["prompt"]["token_ids"])
    points, standard, norms = [], [], {}
    standard_keys = {(row["layer"], row["signal"]) for row in target["saved_trace"]}
    destination = Path(str(REMOTE_CACHE)) / "matched-prefill" / output_name
    destination.mkdir(parents=True, exist_ok=True)
    token_path = destination / "token-metrics.jsonl"

    def target_metrics(layer, signal, errors, reference_norms, width):
        relative = relative_error_from_norms(errors, reference_norms)
        return {
            "layer": layer,
            "signal": signal,
            "vector_width": width,
            "hf_token_norm": reference_norms[position],
            "token_l2": errors[position],
            "token_relative_error": relative[position],
        }

    # Recompute every detailed point from the actual tensors, retaining full
    # per-token numerical results remotely and downloading the small summaries.
    with token_path.open("w", encoding="utf-8") as handle:
        for (layer, signal), actual in engine.items():
            expected = hf[(layer, signal)]
            if (
                actual.shape[0] != count
                or actual.device.type != "cpu"
                or expected.device.type != "cpu"
            ):
                raise ValueError("Expected complete CPU tensors for this prompt")
            metrics = compare_activation_l2(actual, expected)
            handle.write(
                json.dumps({"layer": layer, "signal": signal, **metrics}, allow_nan=False) + "\n"
            )
            row = target_metrics(
                layer, signal, metrics["token_l2"], metrics["hf_token_norm"], expected.shape[1]
            )
            points.append(row)
            if (layer, signal) in standard_keys:
                valid = [value for value in metrics["token_relative_error"] if value is not None]
                standard.append(
                    {
                        **row,
                        "median_relative_error": float(
                            torch.quantile(torch.tensor(valid, dtype=torch.float64), 0.5)
                        )
                        if valid
                        else None,
                        "max_relative_error": max(valid) if valid else None,
                        "undefined_relative_error_tokens": count - len(valid),
                    }
                )
                norms[(layer, signal)] = metrics["hf_token_norm"]
                baseline = next(case for case in profile["cases"] if case["family"] == "baseline")
                previous = next(
                    value
                    for value in baseline["rows"]
                    if (value["layer"], value["signal"]) == (layer, signal)
                )
                if metrics["token_l2"] != previous["token_l2"]:
                    raise ValueError("Recomputed baseline L2 differs from the saved profile")
    if {(row["layer"], row["signal"]) for row in standard} != standard_keys:
        raise ValueError("Relative-error summary is missing standard capture points")

    # Each saved replacement error uses the same HF denominator at the same
    # token and capture point. No GPU rerun is required for these ratios.
    operation_points = []
    for case in profile["cases"]:
        for row in case["rows"]:
            key = row["layer"], row["signal"]
            if len(row["token_l2"]) != count:
                raise ValueError("Saved profile has incomplete token L2")
            operation_points.append(
                {
                    "family": case["family"],
                    **target_metrics(*key, row["token_l2"], norms[key], hf[key].shape[1]),
                }
            )
    if torch.cuda.is_initialized():
        raise ValueError("Relative-error postprocessing unexpectedly initialized CUDA")
    summary = {
        "model": target["model"],
        "model_revision": target["model_revision"],
        "dataset_row": target["dataset_row"],
        "token_position": position,
        "token_id": target["token_id"],
        "input_tokens": count,
        "compute": "Modal CPU",
        "metric_dtype": "float64",
        "zero_hf_norm_policy": "undefined: null, no epsilon",
        "points": points,
        "standard_points": standard,
        "operation_points": operation_points,
        "token_metrics_path": str(token_path.relative_to(Path(str(REMOTE_CACHE)))),
    }
    path = destination / "summary.json"
    path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    cache_volume.commit()
    return {
        "remote_path": str(path.relative_to(Path(str(REMOTE_CACHE)))),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "token_metrics_path": summary["token_metrics_path"],
    }


# Stage "fp32" (T4). One HF FP32 eager forward with TF32 off, compared on CPU with the
# saved engine FP16 and HF FP16 traces: which FP16 side is closer to FP32, per token.
@app.function(
    image=dependency_image,
    gpu="T4",
    cpu=2,
    memory=65536,
    timeout=3600,
    volumes={str(REMOTE_CACHE): cache_volume},
    single_use_containers=True,
)
def run_fp32_referee(target, output_name, reference_trace_path):
    """One FP32 HF forward, followed by CPU comparisons with both saved FP16 traces."""
    import torch

    # Disable TF32 before any CUDA use and keep autocast disabled for the one
    # forward. Do not initialize or run the engine or an FP16 model here.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    ids, position = target["prompt"]["token_ids"], target["token_position"]
    source = Path(str(REMOTE_CACHE)) / Path(reference_trace_path).parent
    saved_setup = json.loads((source / "summary.json").read_text())
    if any(
        saved_setup[key] != target[key]
        for key in ("model", "model_revision", "dataset_row", "token_position", "token_id")
    ):
        raise ValueError("Saved FP16 trace belongs to a different target")
    engine = torch.load(source / "engine.pt", map_location="cpu", weights_only=True)
    hf_half = torch.load(source / "hf.pt", map_location="cpu", weights_only=True)
    if engine.keys() != hf_half.keys() or any(
        engine[key].shape != hf_half[key].shape
        or engine[key].shape[0] != len(ids)
        or engine[key].dtype != torch.float16
        or hf_half[key].dtype != torch.float16
        for key in engine
    ):
        raise ValueError("Saved FP16 trace shapes or dtypes differ from the requested workload")
    destination = Path(str(REMOTE_CACHE)) / "matched-prefill" / output_name
    destination.mkdir(parents=True, exist_ok=True)
    setup = {
        "model": target["model"],
        "model_revision": target["model_revision"],
        "dataset_row": target["dataset_row"],
        "token_position": position,
        "token_id": target["token_id"],
        "input_tokens": len(ids),
        "mode": "whole",
        "reference_dtype": "float32",
        "comparison_dtype": "float16",
        "metric_dtype": "float64",
        "attention": "eager",
        "forward_calls": 0,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "source_trace_directory": str(source),
        "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    }
    snapshot = snapshot_download(
        target["model"],
        revision=target["model_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    try:
        reference = (
            AutoModelForCausalLM.from_pretrained(
                snapshot, torch_dtype=torch.float32, attn_implementation="eager"
            )
            .eval()
            .to("cuda")
        )
        if any(parameter.dtype != torch.float32 for parameter in reference.parameters()):
            raise ValueError("FP32 reference contains non-FP32 parameters")
        setup["memory_after_model_load"] = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "free_bytes": torch.cuda.mem_get_info()[0],
            "total_bytes": torch.cuda.mem_get_info()[1],
        }
        setup["tf32_before_forward"] = {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        if any(setup["tf32_before_forward"].values()):
            raise ValueError("TF32 must be disabled")
        print(
            f"Foreground check passed: FP32 model loaded on {setup['gpu']}; "
            f"free GPU memory {setup['memory_after_model_load']['free_bytes'] / 2**30:.2f} GiB; "
            "TF32 off. Starting the only forward pass.",
            flush=True,
        )
        trace, metadata, restore = install_hf_trace(reference, len(ids))
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                positions = torch.arange(len(ids), device="cuda")
                setup["forward_calls"] += 1
                output = reference(
                    input_ids=torch.tensor([ids], device="cuda"),
                    attention_mask=torch.ones(1, len(ids), device="cuda", dtype=torch.long),
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    use_cache=True,
                    logits_to_keep=1,
                )
            if output.past_key_values.get_seq_length() != len(ids):
                raise ValueError("FP32 cache does not cover the whole prompt")
        finally:
            restore()
        setup["tf32_after_forward"] = {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        setup["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        setup["capture_metadata"] = metadata
        if trace.keys() != hf_half.keys() or any(
            value.dtype != torch.float32 for value in trace.values()
        ):
            raise ValueError("FP32 trace point coverage or dtype differs from the FP16 reference")
        torch.save(trace, destination / "hf-fp32.pt")
        (destination / "forward-metadata.json").write_text(json.dumps(setup, indent=2))
        cache_volume.commit()
        del reference, output
        mem_cleanup()
    except torch.cuda.OutOfMemoryError:
        setup.update(
            status="out_of_memory",
            error="HF FP32 did not fit on the T4. Stopped without changing GPU or settings.",
        )
        path = destination / "summary.json"
        path.write_text(json.dumps(setup, indent=2))
        cache_volume.commit()
        return {
            "remote_path": str(path.relative_to(Path(str(REMOTE_CACHE)))),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    # Cast the saved values to CPU FP64 before comparison. Casting FP16/FP32
    # values upwards is exact and avoids downcasting either side of the metric.
    rows, final_tokens, final_counts = [], [], {}
    points = [(row["layer"], row["signal"]) for row in target["saved_trace"]]
    for layer, signal in points:
        key = layer, signal
        expected = trace[key].to(device="cpu", dtype=torch.float64)
        engine_metrics = compare_activation_l2(engine[key].double(), expected)
        hf_metrics = compare_activation_l2(hf_half[key].double(), expected)
        row = {
            "layer": layer,
            "signal": signal,
            "hf_fp32_token_norm": engine_metrics["hf_token_norm"][position],
            "engine_token_l2": engine_metrics["token_l2"][position],
            "hf_fp16_token_l2": hf_metrics["token_l2"][position],
            "engine_relative_error": engine_metrics["token_relative_error"][position],
            "hf_fp16_relative_error": hf_metrics["token_relative_error"][position],
        }
        for name, metrics in (("engine", engine_metrics), ("hf_fp16", hf_metrics)):
            valid = [value for value in metrics["token_relative_error"] if value is not None]
            row[f"{name}_median_relative_error"] = (
                float(torch.quantile(torch.tensor(valid, dtype=torch.float64), 0.5))
                if valid
                else None
            )
            row[f"{name}_max_relative_error"] = max(valid) if valid else None
            row[f"{name}_undefined_relative_errors"] = len(ids) - len(valid)
        rows.append(row)
        if signal == "lm_head_input":
            final_counts = {"engine_closer": 0, "hf_fp16_closer": 0, "equal_distance": 0}
            for token_position in range(len(ids)):
                engine_error = engine_metrics["token_l2"][token_position]
                hf_error = hf_metrics["token_l2"][token_position]
                closer = (
                    "engine_closer"
                    if engine_error < hf_error
                    else "hf_fp16_closer"
                    if hf_error < engine_error
                    else "equal_distance"
                )
                final_counts[closer] += 1
                final_tokens.append(
                    {
                        "token_position": token_position,
                        "token_id": ids[token_position],
                        "hf_fp32_token_norm": engine_metrics["hf_token_norm"][token_position],
                        "engine_token_l2": engine_error,
                        "hf_fp16_token_l2": hf_error,
                        "engine_relative_error": engine_metrics["token_relative_error"][
                            token_position
                        ],
                        "hf_fp16_relative_error": hf_metrics["token_relative_error"][
                            token_position
                        ],
                        "closer": closer,
                    }
                )
    setup.update(
        status="complete",
        rows=rows,
        final_tokens=final_tokens,
        final_counts=final_counts,
        fp32_trace_path=str((destination / "hf-fp32.pt").relative_to(Path(str(REMOTE_CACHE)))),
    )
    path = destination / "summary.json"
    path.write_text(json.dumps(setup, allow_nan=False), encoding="utf-8")
    cache_volume.commit()
    return {
        "remote_path": str(path.relative_to(Path(str(REMOTE_CACHE)))),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fp32_trace_path": setup["fp32_trace_path"],
    }


# Build the target file from a saved run. Input: the run JSONL written by
# modal_validate_activations.py and its workload JSON. Output: outlier-target.json with
# the prompt, the token, and that token's saved L2 at all 57 points (each stage checks
# that it reproduces these exactly before trusting anything else).
def pick_target(run_path, workload_path, dataset_row=None, token_position=None):
    workload = json.loads(Path(workload_path).read_text(encoding="utf-8"))
    prompts = {prompt["dataset_row"]: prompt for prompt in workload["prompts"]}
    best = None
    with Path(run_path).open(encoding="utf-8") as handle:
        setup = json.loads(next(handle))
        for line in handle:
            record = json.loads(line)
            if dataset_row is not None and record["dataset_row"] != dataset_row:
                continue
            final = next(row for row in record["rows"] if row["signal"] == "lm_head_input")
            positions = (
                [token_position] if token_position is not None else range(len(final["token_l2"]))
            )
            for position in positions:
                value = final["token_l2"][position]
                if best is None or value > best[0]:
                    best = (value, record, position)
    if best is None:
        raise ValueError("No matching prompt in the saved run")
    value, record, position = best
    prompt = prompts[record["dataset_row"]]
    if prompt["token_ids"] != record["token_ids"]:
        raise ValueError("Saved run and workload disagree on the prompt's token IDs")
    target = {
        "model": setup["model"],
        "model_revision": setup["model_revision"],
        "dataset_row": record["dataset_row"],
        "token_position": position,
        "token_id": record["token_ids"][position],
        "input_tokens": len(record["token_ids"]),
        "token_l2": value,
        "prompt": prompt,
        "saved_trace": [
            {
                "layer": row["layer"],
                "signal": row["signal"],
                "token_l2": row["token_l2"][position],
                "prompt_l2": row["prompt_l2"],
            }
            for row in record["rows"]
        ],
        "workload_metadata": {key: val for key, val in workload.items() if key != "prompts"},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_PATH.write_text(json.dumps(target, indent=2), encoding="utf-8")
    print(
        f"Target: row {target['dataset_row']}, token {position} of {target['input_tokens']}, "
        f"final-state L2 {value:.6f}. Saved {TARGET_PATH}",
        flush=True,
    )


# Launch one stage, download its summary, check it against the target, and only then
# write <prefix>-completion.json. Stages after "trace" read the trace's saved HF tensors.
def run_stage(stage):
    import time

    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    prefix = STAGE_PREFIX[stage]
    reference_trace_path = None
    if stage != "trace":
        reference = json.loads((OUTPUT_DIR / "outlier-completion.json").read_text())
        if reference["status"] != "complete":
            raise ValueError("Run --stage trace first")
        saved = json.loads(Path(reference["result"]).read_text())
        if any(
            saved[key] != target[key]
            for key in ("model", "model_revision", "dataset_row", "token_position", "token_id")
        ):
            raise ValueError("The completed trace belongs to a different target")
        reference_trace_path = next(
            path for path in reference["manifest"]["tensor_paths"] if path.endswith("/hf.pt")
        )
    name = prefix + "-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_path = OUTPUT_DIR / f"{name}.json"
    finished = False
    with modal.enable_output(), app.run(detach=stage != "fp32"):
        if stage == "fp32":
            call = run_fp32_referee.with_options(secrets=hugging_face_modal_secrets()).spawn(
                target, name, reference_trace_path
            )
        elif stage == "relative":
            completed_profile = json.loads(
                (OUTPUT_DIR / "operation-profile-completion.json").read_text()
            )
            if completed_profile["status"] != "complete":
                raise ValueError("Run --stage profile before --stage relative")
            call = summarize_outlier_relative.spawn(
                target, name, reference_trace_path, completed_profile["manifest"]["remote_path"]
            )
        else:
            call = diagnose_outlier.with_options(secrets=hugging_face_modal_secrets()).spawn(
                target, name, stage == "profile", reference_trace_path
            )
        (OUTPUT_DIR / f"{prefix}-job.json").write_text(
            json.dumps({"call_id": call.object_id, "output": str(output_path)}, indent=2)
        )
        manifest = call.get()
        with output_path.open("wb") as handle:
            for chunk in cache_volume.read_file(manifest["remote_path"]):
                handle.write(chunk)
        if (
            output_path.stat().st_size != manifest["bytes"]
            or hashlib.sha256(output_path.read_bytes()).hexdigest() != manifest["sha256"]
        ):
            raise ValueError("Diagnostic artifact checksum mismatch")
        result = json.loads(output_path.read_text())
        if stage == "fp32" and result["status"] == "out_of_memory":
            raise RuntimeError(result["error"])
        check_stage_result(stage, target, result)
        finished = True
    # Detached Modal contexts can swallow connection loss, so check outside the context.
    if not finished:
        raise RuntimeError("Diagnostic disconnected before verified completion")
    (OUTPUT_DIR / f"{prefix}-completion.json").write_text(
        json.dumps(
            {"status": "complete", "result": str(output_path), "manifest": manifest}, indent=2
        )
    )
    print(f"Stage {stage} complete: {output_path}", flush=True)


def check_stage_result(stage, target, result):
    """Reject a stage result that is for another target or is missing points."""
    expected_points = {(row["layer"], row["signal"]) for row in target["saved_trace"]}
    if (
        result["model"] != target["model"]
        or result["model_revision"] != target["model_revision"]
        or result["dataset_row"] != target["dataset_row"]
        or result["input_tokens"] != len(target["prompt"]["token_ids"])
        or result["token_id"] != target["token_id"]
        or result["token_position"] != target["token_position"]
    ):
        raise ValueError("Diagnostic control checks failed")
    if stage == "fp32":
        if (
            result["status"] != "complete"
            or result["forward_calls"] != 1
            or [(row["layer"], row["signal"]) for row in result["rows"]]
            != [(row["layer"], row["signal"]) for row in target["saved_trace"]]
            or len(result["final_tokens"]) != len(target["prompt"]["token_ids"])
            or sum(result["final_counts"].values()) != len(result["final_tokens"])
            or any(result["tf32_before_forward"].values())
            or any(result["tf32_after_forward"].values())
        ):
            raise ValueError("FP32 result is incomplete or settings differ")
    elif stage == "relative":
        detailed_points = 18 * ((len(expected_points) - 1) // 2) + 2
        if (
            len(result["points"]) != detailed_points
            or len(result["standard_points"]) != len(expected_points)
            or {(row["layer"], row["signal"]) for row in result["standard_points"]}
            != expected_points
            or len(result["operation_points"]) != 7 * len(expected_points)
        ):
            raise ValueError("Relative-error summary is incomplete")
    elif stage == "profile":
        families = ["baseline", "norm", "qkv", "rope", "attention", "silu", "restored_baseline"]
        if (
            [case["family"] for case in result["cases"]] != families
            or not result["restored_baseline_exact"]
            or any(
                {(row["layer"], row["signal"]) for row in case["rows"]} != expected_points
                or len(case["rows"]) != len(expected_points)
                for case in result["cases"]
            )
        ):
            raise ValueError("Operation profile is incomplete")
    elif (
        len(result["controls"]) != len(expected_points)
        or {(row["layer"], row["signal"]) for row in result["controls"]} != expected_points
        or any(row["prompt_l2"] != 0 for row in result["controls"])
        or not result["engine_metadata"]["causal"]
        or not result["hf_metadata"]["causal"]
    ):
        raise ValueError("Diagnostic capture controls failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pick-target", action="store_true", help="Write outlier-target.json")
    parser.add_argument("--run", help="Saved run JSONL from modal_validate_activations.py")
    parser.add_argument("--workload", help="That run's workload JSON")
    parser.add_argument("--row", type=int, help="Dataset row (default: worst final-state token)")
    parser.add_argument("--position", type=int, help="Token position within --row")
    parser.add_argument("--stage", choices=list(STAGE_PREFIX))
    args = parser.parse_args()
    if args.pick_target:
        if not (args.run and args.workload):
            parser.error("--pick-target needs --run and --workload")
        pick_target(args.run, args.workload, args.row, args.position)
    elif args.stage:
        try:
            run_stage(args.stage)
        except Exception as error:
            (OUTPUT_DIR / f"{STAGE_PREFIX[args.stage]}-completion.json").write_text(
                json.dumps(
                    {"status": "failed", "error": str(error), "traceback": traceback.format_exc()},
                    indent=2,
                )
            )
            raise
    else:
        parser.error("Pass --pick-target or --stage")


if __name__ == "__main__":
    main()
