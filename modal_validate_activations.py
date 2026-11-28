"""Engine-vs-HF activation comparison on many prompts, plus a loaded-weight audit.

Feeds identical saved token IDs to the engine (FP16, FlashInfer, fused kernels) and to
HF Transformers (FP16, eager attention), captures the same points on both (attention
output and completed-block residual for every block, plus the final normalized hidden
state: 57 points for Qwen3-1.7B), and saves per-token L2, HF vector norm and relative
error (L2 / HF norm) at every point. Metrics are computed in FP64 on CPU.

Run (GPU on Modal):
    # 100 WildChat prompts, whole-prompt prefill on both sides (the 2026-11-28 run)
    python modal_validate_activations.py --model-id Qwen/Qwen3-1.7B --mode whole --num-prompts 100
    # The same with 256-token chunked prefill, on the engine and on HF
    python modal_validate_activations.py --mode chunked --chunk-size 256
    # Check that the engine installs exactly HF's weights (no prompts)
    python modal_validate_activations.py --audit-weights --model-id Qwen/Qwen3-1.7B Qwen/Qwen3-4B

Outputs, in artifacts/matched-prefill/: <model>-workload-<time>.json (prompts and token
IDs), <model>-<mode>-<time>.jsonl (one setup line, then one line per prompt) and
completion.json. The weight audit writes artifacts/weight-audit/<model>.json.
A saved run feeds modal_trace_outlier.py (one-token deep dive) and modal_fp32_referee.py
(which implementation is closer to FP32).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

import modal

from activation_validation import (
    ACTIVATION_METRICS,
    capture_hf_prefill,
    capture_points,
    compare_activation_l2,
    prefill_spans,
    relative_error_from_norms,
    validate_engine_capture_spans,
)
from config import LOCAL_REPO, REMOTE_CACHE, app, cache_volume, dependency_image
from utils import (
    hugging_face_modal_secrets,
    load_wildchat_prompts,
    mem_cleanup,
    reconstruct_capture,
)

# Earlier saved workloads, read for their sampling settings (seed, dataset revision,
# length limit) and engine memory ratio. They live in the local, gitignored artifacts/.
SAVED_WORKLOADS = {
    "Qwen/Qwen3-1.7B": "validate-Qwen--Qwen3-1.7B-20261101T230126Z.json",
    "Qwen/Qwen3-4B": "validate-Qwen--Qwen3-4B-20261101T231832Z.json",
    "meta-llama/Llama-3.2-1B": "validate-meta-llama--Llama-3.2-1B-20261101T225510Z.json",
    "meta-llama/Llama-3.2-3B": "validate-meta-llama--Llama-3.2-3B-20261101T230227Z.json",
}
OUTPUT_DIR = LOCAL_REPO / "artifacts/matched-prefill"


# Workload preparation (Modal CPU). Pins the checkpoint revision and, when a larger
# sample is requested, draws it from WildChat with the saved seed and filters. Both
# implementations then receive these saved token IDs; nothing is retokenized later.
@app.function(image=dependency_image, timeout=600, volumes={str(REMOTE_CACHE): cache_volume})
def prepare_model_workload(model_id, saved_workload, num_prompts, scan_rows=8192):
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_id)
    if config.model_type not in {"llama", "qwen3"}:
        raise ValueError("Matched capture currently supports dense Llama and Qwen3 only")
    if not config._commit_hash:
        raise ValueError("Cannot pin model revision")
    revision = config._commit_hash
    workload = saved_workload
    # Prepare any larger sample once on Modal CPU; both implementations receive
    # the resulting saved IDs rather than tokenizing independently during inference.
    if num_prompts is not None and num_prompts != len(saved_workload["prompts"]):
        workload = load_wildchat_prompts(
            AutoTokenizer.from_pretrained(model_id, revision=revision),
            count=num_prompts,
            seed=saved_workload["seed"],
            revision=saved_workload["revision"],
            max_tokens=saved_workload["max_input_tokens"],
            scan_rows=scan_rows,
        )
        workload["tokenizer_revision"] = revision
    return revision, workload


# Main comparison (T4). Runs every prompt through the engine with capture on (one
# sampled token, so prefill only), releases it, then runs HF FP16 eager on the same IDs
# with hooks at the same 57 points. Writes one JSONL line per prompt with per-token L2,
# HF norm and relative error at every point, on the Modal volume.
@app.function(
    image=dependency_image,
    gpu="T4",
    cpu=2,
    memory=65536,
    timeout=3600,
    volumes={str(REMOTE_CACHE): cache_volume},
    single_use_containers=True,
)
def validate_prefill(
    model_id, revision, workload, mode, chunk_size, memory_ratio, source, output_name
):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from sgl_interp.capture import ActivationCaptureConfig
    from sgl_interp.core import SamplingParams
    from sgl_interp.layers.rotary import get_rope
    from sgl_interp.llm import LLM
    from transformers import AutoConfig, AutoModelForCausalLM

    # Read the same pinned snapshot in both loaders, and never retokenize saved IDs.
    snapshot = snapshot_download(
        model_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]
    )
    config = AutoConfig.from_pretrained(snapshot)
    if config.model_type not in {"llama", "qwen3"}:
        raise ValueError("Capture hooks have not been audited for this architecture")
    samples = workload["prompts"]
    if not samples or any(not prompt["token_ids"] for prompt in samples):
        raise ValueError("Saved workload must contain nonempty pretokenized prompts")
    maximum_length = max(len(prompt["token_ids"]) for prompt in samples)
    if maximum_length >= 4096:
        raise ValueError("Workload leaves no room for the one prefill-produced output token")
    budget = maximum_length if mode == "whole" else chunk_size
    prefill_spans(maximum_length, mode, chunk_size)
    points = capture_points(config.num_hidden_layers)
    llm = LLM(
        model_path=snapshot,
        dtype=torch.float16,
        cache_type="naive",
        max_extend_tokens=budget,
        max_running_req=1,
        cuda_graph_max_bs=1,
        max_seq_len_override=4096,
        memory_ratio=memory_ratio,
        use_pynccl=False,
        activation_capture=ActivationCaptureConfig(
            layer_ids=tuple(range(config.num_hidden_layers)), capture_lm_head_input=True
        ),
    )
    saved = []
    attention_backend = type(llm.engine.attn_backend).__name__
    try:
        # One request at a time prevents another request consuming the chunk budget.
        # A single sampled token triggers prompt processing without a decode forward.
        for prompt in samples:
            ids = prompt["token_ids"]
            llm.generate(
                [ids], SamplingParams(top_k=1, temperature=0, max_tokens=1, ignore_eos=True)
            )
            captures = llm.take_activation_captures()
            validate_engine_capture_spans(
                captures, points, prefill_spans(len(ids), mode, chunk_size)
            )
            saved.append((prompt, captures))
    finally:
        llm.shutdown()
    del llm
    get_rope.cache_clear()
    mem_cleanup()
    reference = (
        AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=torch.float16, attn_implementation="eager"
        )
        .eval()
        .to("cuda")
    )
    if len(reference.model.layers) != config.num_hidden_layers:
        raise ValueError("HF block count differs from capture plan")
    setup = {
        "type": "setup",
        "model": model_id,
        "model_revision": revision,
        "mode": mode,
        "chunk_size": chunk_size if mode == "chunked" else None,
        "engine_prefill_budget": budget,
        "prompts": len(samples),
        "num_layers": config.num_hidden_layers,
        "capture_points": [{"layer": layer, "signal": signal} for layer, signal in points],
        "dtype": "float16",
        "metric_dtype": "float64",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "hf_attention": "eager",
        "engine_attention": attention_backend,
        "hf_use_cache": True,
        "engine_cache": "naive",
        "batch_size": 1,
        "normalization": "original fused residual-add/RMSNorm",
        "metrics": ACTIVATION_METRICS,
        "source": source,
        "dataset": {key: value for key, value in workload.items() if key != "prompts"},
    }
    # Persist large results on Modal instead of sending them through a live generator
    # connection. The detached worker can finish even if the local connection drops.
    remote_path = f"matched-prefill/{output_name}"
    output_path = Path(str(REMOTE_CACHE)) / remote_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(setup, allow_nan=False) + "\n")
            for prompt, captures in saved:
                ids = prompt["token_ids"]
                expected = capture_hf_prefill(reference, ids, mode, chunk_size)
                rows = []
                for layer, signal in points:
                    actual = reconstruct_capture(captures, layer, signal, 0, ids)
                    target = expected[(layer, signal)]
                    if actual.shape != (len(ids), config.hidden_size):
                        raise ValueError("Engine capture shape differs from the full prompt")
                    rows.append(
                        {"layer": layer, "signal": signal, **compare_activation_l2(actual, target)}
                    )
                record = {
                    "type": "prompt",
                    "dataset_row": prompt["dataset_row"],
                    "token_ids": ids,
                    "spans": prefill_spans(len(ids), mode, chunk_size),
                    "rows": rows,
                }
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                captures.clear()
    finally:
        cache_volume.commit()
    return {
        "model": model_id,
        "model_revision": revision,
        "mode": mode,
        "prompts": len(samples),
        "remote_path": remote_path,
        "bytes": output_path.stat().st_size,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def map_reference_weight(name, reference_state):
    """Return the runtime key and row slice, using HF shapes, not loader helpers."""
    for group, merged in (
        (("q_proj", "k_proj", "v_proj"), "qkv_proj"),
        (("gate_proj", "up_proj"), "gate_up_proj"),
    ):
        for slot, projection in enumerate(group):
            marker = f".{projection}."
            if marker in name:
                sizes = [
                    reference_state[name.replace(marker, f".{part}.")].shape[0] for part in group
                ]
                start = sum(sizes[:slot])
                return name.replace(marker, f".{merged}."), start, start + sizes[slot]
    return name, None, None


def compare_tensors(actual, expected):
    """Compare all elements; bound FP64 scratch memory with flat chunks."""
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"passed": False, "shape_or_dtype_mismatch": True}
    actual_flat, expected_flat = actual.reshape(-1), expected.reshape(-1)
    different = 0
    error_squared = reference_squared = max_absolute = 0.0
    finite = True
    for start in range(0, actual.numel(), 262144):
        left = actual_flat[start : start + 262144].double()
        right = expected_flat[start : start + 262144].double()
        finite &= bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        different += int((left != right).sum())
        difference = left - right
        error_squared += float(difference.square().sum())
        reference_squared += float(right.square().sum())
        max_absolute = max(max_absolute, float(difference.abs().max()))
    return {
        "passed": different == 0 and finite,
        "finite": finite,
        "different_elements": different,
        "elements": actual.numel(),
        "max_absolute_error": max_absolute,
        "relative_l2": (error_squared / reference_squared) ** 0.5
        if reference_squared
        else (0.0 if error_squared == 0 else None),
    }


# --audit-weights (T4, then Modal CPU). Loads the serving engine, copies its installed
# weights to CPU, then loads HF FP16 from the same pinned snapshot. Undoes the engine's
# merged Q/K/V and gate/up using HF tensor shapes and requires exact equality for every
# tensor, plus equal per-block norm epsilons and head counts. No prompts are run.
@app.function(
    image=dependency_image,
    gpu="T4",
    cpu=4,
    memory=65536,
    timeout=3600,
    volumes={str(REMOTE_CACHE): cache_volume},
    single_use_containers=True,
)
def audit_model(model_id, source):
    import gc
    from dataclasses import asdict
    from importlib.metadata import version

    import torch
    from huggingface_hub import snapshot_download
    from sgl_interp.llm import LLM
    from sgl_interp.models import ModelConfig
    from transformers import AutoConfig, AutoModelForCausalLM

    # Resolve one revision, then force BOTH actual loaders to that local snapshot.
    config = AutoConfig.from_pretrained(model_id)
    revision = config._commit_hash
    assert revision, "Cannot pin checkpoint revision"
    snapshot = snapshot_download(
        model_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]
    )
    llm = LLM(
        model_path=snapshot,
        dtype=torch.float16,
        cache_type="naive",
        cuda_graph_max_bs=1,
        max_extend_tokens=256,
        max_running_req=2,
        max_seq_len_override=4096,
        memory_ratio=0.75 if "4B" in model_id else 0.45,
        use_pynccl=False,
    )
    try:
        runtime = llm.engine.model
        # Read the model actually installed in the serving engine, not loader output.
        engine_state = {
            name: tensor.detach().cpu().clone() for name, tensor in runtime.state_dict().items()
        }
        tied = runtime.lm_head.tied_embedding is runtime.model.embed_tokens
        if tied:
            engine_state["lm_head.weight"] = (
                runtime.model.embed_tokens.weight.detach().cpu().clone()
            )
        runtime_layers = [
            {
                "input_eps": layer.input_layernorm.eps,
                "post_attention_eps": layer.post_attention_layernorm.eps,
                "q_eps": layer.self_attn.q_norm.eps,
                "k_eps": layer.self_attn.k_norm.eps,
                "head_dim": layer.self_attn.attn.head_dim,
                "q_heads": layer.self_attn.attn.num_qo_heads,
                "kv_heads": layer.self_attn.attn.num_kv_heads,
            }
            for layer in runtime.model.layers.op_list
        ]
        backend = type(llm.engine.attn_backend).__name__
    finally:
        llm.shutdown()
    del runtime, llm
    gc.collect()
    torch.cuda.empty_cache()

    # HF loads the exact same pinned files with serving FP16 conversion, on Modal CPU.
    reference, loading_info = AutoModelForCausalLM.from_pretrained(
        snapshot,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    reference.eval()
    reference_state = reference.state_dict()
    rows, used_keys, covered_elements = [], set(), {}
    for name, expected in reference_state.items():
        runtime_key, start, stop = map_reference_weight(name, reference_state)
        row = {
            "hf_name": name,
            "engine_name": runtime_key,
            "row_start": start,
            "row_stop": stop,
            "hf_shape": list(expected.shape),
            "hf_dtype": str(expected.dtype),
        }
        if runtime_key not in engine_state:
            row.update(passed=False, missing=True)
        else:
            used_keys.add(runtime_key)
            covered_elements[runtime_key] = covered_elements.get(runtime_key, 0) + expected.numel()
            actual = engine_state[runtime_key]
            if start is not None:
                actual = actual[start:stop]
            row.update(
                engine_shape=list(actual.shape),
                engine_dtype=str(actual.dtype),
                **compare_tensors(actual, expected),
            )
        rows.append(row)
    extra_keys = sorted(engine_state.keys() - used_keys)
    coverage_errors = [
        key for key in used_keys if covered_elements[key] != engine_state[key].numel()
    ]

    # Check the effective per-block scalar settings against instantiated HF modules.
    setting_rows = []
    for index, (actual, block) in enumerate(zip(runtime_layers, reference.model.layers)):
        expected = {
            "input_eps": block.input_layernorm.variance_epsilon,
            "post_attention_eps": block.post_attention_layernorm.variance_epsilon,
            "q_eps": block.self_attn.q_norm.variance_epsilon,
            "k_eps": block.self_attn.k_norm.variance_epsilon,
            "head_dim": block.self_attn.head_dim,
            "q_heads": reference.config.num_attention_heads,
            "kv_heads": reference.config.num_key_value_heads,
        }
        setting_rows.append(
            {"layer": index, "engine": actual, "hf": expected, "passed": actual == expected}
        )
    passed = (
        all(row["passed"] for row in rows)
        and not extra_keys
        and not coverage_errors
        and len(runtime_layers) == len(reference.model.layers)
        and all(row["passed"] for row in setting_rows)
        and not any(
            loading_info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        )
    )
    return {
        "model": model_id,
        "revision": revision,
        "source": source,
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "gpu": torch.cuda.get_device_name(),
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "engine_attention": backend,
        "hf_attention": "eager",
        "engine_config": asdict(ModelConfig.from_hf(reference.config)),
        "hf_config": reference.config.to_dict(),
        "tied_embeddings": tied,
        "hf_loading_info": loading_info,
        "extra_engine_keys": extra_keys,
        "coverage_errors": coverage_errors,
        "rows": rows,
        "layer_settings": setting_rows,
        "passed": passed,
        "compared_tensors": len(rows),
        "failed_tensors": sum(not row["passed"] for row in rows),
        "compared_elements_including_aliases": sum(row.get("elements", 0) for row in rows),
    }


def verify_saved_run(path, workload, model_id, revision, mode, chunk_size):
    """Re-read the artifact before claiming success; errors are integrity failures only."""
    with path.open(encoding="utf-8") as handle:
        setup = json.loads(next(handle))
        if (
            setup["type"] != "setup"
            or setup["model"] != model_id
            or setup["model_revision"] != revision
            or setup["mode"] != mode
            or setup["prompts"] != len(workload["prompts"])
            or setup["metrics"] not in (["token_l2", "prompt_l2"], ACTIVATION_METRICS)
        ):
            raise ValueError("Saved setup does not match the requested run")
        points = capture_points(setup["num_layers"])
        if setup["capture_points"] != [
            {"layer": layer, "signal": signal} for layer, signal in points
        ]:
            raise ValueError("Saved capture plan is incomplete")
        prompt_count = 0
        for prompt in workload["prompts"]:
            line = next(handle, None)
            if line is None:
                raise ValueError(
                    f"Incomplete saved run: {prompt_count}/{len(workload['prompts'])} prompts"
                )
            record = json.loads(line)
            if (
                record["type"] != "prompt"
                or record["dataset_row"] != prompt["dataset_row"]
                or record["token_ids"] != prompt["token_ids"]
                or record["spans"]
                != [
                    list(span) for span in prefill_spans(len(prompt["token_ids"]), mode, chunk_size)
                ]
            ):
                raise ValueError("Saved prompt identity or token spans differ")
            if [(row["layer"], row["signal"]) for row in record["rows"]] != points:
                raise ValueError("Saved prompt has missing, duplicate or reordered capture points")
            for row in record["rows"]:
                if set(row) != {"layer", "signal", *setup["metrics"]}:
                    raise ValueError("Unexpected error metric in saved run")
                values = row["token_l2"]
                if (
                    len(values) != len(prompt["token_ids"])
                    or not all(math.isfinite(value) and value >= 0 for value in values)
                    or not math.isfinite(row["prompt_l2"])
                    or row["prompt_l2"] < 0
                ):
                    raise ValueError("Saved L2 values are incomplete or invalid")
                if not math.isclose(
                    math.sqrt(math.fsum(value * value for value in values)),
                    row["prompt_l2"],
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise ValueError("Prompt L2 is inconsistent with per-token L2")
                if "hf_token_norm" in row:
                    if len(row["hf_token_norm"]) != len(values):
                        raise ValueError("Saved HF norms are incomplete")
                    expected_relative = relative_error_from_norms(values, row["hf_token_norm"])
                    if row["token_relative_error"] != expected_relative:
                        raise ValueError(
                            "Relative error is inconsistent with token L2 and HF norms"
                        )
            prompt_count += 1
        if next(handle, None) is not None:
            raise ValueError("Saved run contains unexpected extra records")
    return prompt_count


def source_fingerprint():
    """SHA256 of the code uploaded to Modal, saved with each result."""
    digest = hashlib.sha256()
    for path in [
        Path(__file__),
        LOCAL_REPO / "activation_validation.py",
        LOCAL_REPO / "utils.py",
        LOCAL_REPO / "config.py",
        *sorted((LOCAL_REPO / "python/sgl_interp").rglob("*.py")),
    ]:
        digest.update(str(path.relative_to(LOCAL_REPO)).encode())
        digest.update(path.read_bytes())
    return {"working_tree_sha256": digest.hexdigest()}


def run_weight_audit(model_ids):
    """One fresh container per model, one after another; prints a summary table."""
    directory = LOCAL_REPO / "artifacts/weight-audit"
    directory.mkdir(parents=True, exist_ok=True)
    source = source_fingerprint()
    results = []
    with modal.enable_output(), app.run(detach=True):
        for model_id in model_ids:
            result = audit_model.with_options(secrets=hugging_face_modal_secrets()).remote(
                model_id, source
            )
            results.append(result)
            (directory / f"{model_id.split('/')[-1]}.json").write_text(json.dumps(result, indent=2))
    if len(results) != len(model_ids):
        raise RuntimeError(f"Weight audit incomplete: {len(results)}/{len(model_ids)} models")
    print("model | revision | tensors | failed | extra engine tensors | block settings | pass")
    for result in results:
        print(
            f"{result['model']} | {result['revision']} | {result['compared_tensors']} | "
            f"{result['failed_tensors']} | {len(result['extra_engine_keys'])} | "
            f"{all(row['passed'] for row in result['layer_settings'])} | {result['passed']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", nargs="+", choices=list(SAVED_WORKLOADS), default=["Qwen/Qwen3-1.7B"]
    )
    parser.add_argument("--mode", nargs="+", choices=["whole", "chunked"], default=["whole"])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--num-prompts",
        type=int,
        help="Prepare this many WildChat prompts on Modal before inference",
    )
    parser.add_argument("--dataset-scan-rows", type=int, default=8192)
    parser.add_argument(
        "--audit-weights",
        action="store_true",
        help="Instead of comparing activations, check the engine loads exactly HF's weights",
    )
    args = parser.parse_args()
    if args.audit_weights:
        run_weight_audit(args.model_id)
        return
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.num_prompts is not None and args.num_prompts < 4:
        parser.error("--num-prompts must be at least 4 for the four length ranges")
    if args.dataset_scan_rows < 1:
        parser.error("--dataset-scan-rows must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = source_fingerprint()
    options = {"secrets": hugging_face_modal_secrets()}
    import time

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    paths = []
    jobs = []
    expected_runs = len(args.model_id) * len(set(args.mode))
    (OUTPUT_DIR / "completion.json").write_text(json.dumps({"status": "running"}))
    with modal.enable_output(), app.run(detach=True):
        for model_id in args.model_id:
            previous = json.loads(
                (LOCAL_REPO / "artifacts/modal-traces" / SAVED_WORKLOADS[model_id]).read_text()
            )
            if previous["model"] != model_id:
                raise ValueError("Saved workload belongs to another model")
            # Prepare once per model so all modes share checkpoint and input IDs.
            revision, workload = prepare_model_workload.with_options(**options).remote(
                model_id, previous["workload"], args.num_prompts, args.dataset_scan_rows
            )
            workload_path = OUTPUT_DIR / f"{model_id.replace('/', '--')}-workload-{timestamp}.json"
            workload_path.write_text(json.dumps(workload, indent=2), encoding="utf-8")
            for mode in dict.fromkeys(args.mode):
                path = OUTPUT_DIR / f"{model_id.replace('/', '--')}-{mode}-{timestamp}.jsonl"
                call = validate_prefill.with_options(**options).spawn(
                    model_id,
                    revision,
                    workload,
                    mode,
                    args.chunk_size,
                    previous["memory_ratio"],
                    source,
                    path.name,
                )
                # Keep the call ID so a disconnected client can retrieve the same
                # completed job instead of spending GPU time running it again.
                jobs.append(
                    {
                        "call_id": call.object_id,
                        "output": str(path),
                        "workload": str(workload_path),
                        "model_revision": revision,
                        "model": model_id,
                        "mode": mode,
                        "chunk_size": args.chunk_size,
                    }
                )
                (OUTPUT_DIR / "jobs.json").write_text(json.dumps(jobs, indent=2))
                manifest = call.get()
                path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
                # Transfer the durable artifact only after the worker has completed.
                with path.open("wb") as handle:
                    for chunk in cache_volume.read_file(manifest["remote_path"]):
                        handle.write(chunk)
                if (
                    path.stat().st_size != manifest["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]
                ):
                    raise ValueError("Downloaded artifact size or SHA256 differs from Modal")
                verify_saved_run(path, workload, model_id, revision, mode, args.chunk_size)
                paths.append(str(path))
    # Detached Modal contexts can suppress connection-loss exceptions. This must
    # sit OUTSIDE that context so an interrupted job can never be called complete.
    if len(paths) != expected_runs:
        raise RuntimeError(f"Incomplete validation: {len(paths)}/{expected_runs} verified runs")
    (OUTPUT_DIR / "completion.json").write_text(
        json.dumps({"status": "complete", "runs": paths, "metrics": ACTIVATION_METRICS}, indent=2)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--audit-weights" not in sys.argv:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "completion.json").write_text(
                json.dumps(
                    {"status": "failed", "error": str(error), "traceback": traceback.format_exc()},
                    indent=2,
                )
            )
        raise
