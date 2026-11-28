"""FP32 check: measure engine FP16 and HF FP16 eager against HF FP32 eager.

Engine-versus-HF-FP16 distance cannot say which side is wrong, because HF FP16 is
itself a rounded version of the model. This script runs, per prompt, one engine FP16
pass, one HF FP16 eager pass and one HF FP32 eager pass (TF32 off) on a Modal T4,
then asks which FP16 implementation is closer to FP32 at every token and capture point.

Run (GPU on Modal, then a local summary of the saved numbers):
    python modal_fp32_referee.py \
        --workload artifacts/matched-prefill/Qwen--Qwen3-1.7B-workload-20261128T023424Z.json \
        --rows 2802 2555 2197 540 5632 1359

Re-summarize a finished batch locally, with no Modal calls:
    python modal_fp32_referee.py --summarize artifacts/matched-prefill/fp32-referee-prompts-<time>

Outputs, in the batch directory: one row-<dataset row>.jsonl per prompt with per-token
norms, raw L2 and relative error for all three pairs; summary.json;
final-state-summary.csv (per prompt and pooled); capture-point-summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import modal

from activation_validation import (
    capture_hf_prefill,
    capture_points,
    compare_activation_l2,
    relative_error_from_norms,
    summarize_referee_populations,
    validate_engine_capture_spans,
)
from config import LOCAL_REPO, REMOTE_CACHE, app, cache_volume, dependency_image
from utils import hugging_face_modal_secrets, mem_cleanup, reconstruct_capture

OUTPUT_DIR = LOCAL_REPO / "artifacts/matched-prefill"


# Per-token metrics for one prompt. Inputs: three traces keyed by (layer, signal), each
# [tokens, width]. All three relative errors divide by the HF FP32 norm of the same token.
def save_referee_token_metrics(target, output_name, traces, metadata, num_layers):
    import torch

    points = capture_points(num_layers)
    if any(set(trace) != set(points) for trace in traces.values()):
        raise ValueError("Referee traces must contain exactly the standard capture points")
    path = Path(str(REMOTE_CACHE)) / "matched-prefill" / f"{output_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    setup = {
        "type": "setup",
        "model": target["model"],
        "model_revision": target["model_revision"],
        "dataset_row": target["prompt"]["dataset_row"],
        "token_ids": target["prompt"]["token_ids"],
        "mode": "whole",
        "num_layers": num_layers,
        "metric_dtype": "float64",
        "relative_error_denominator": "HF FP32 token norm for all three comparisons",
        **metadata,
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(setup, allow_nan=False) + "\n")
        for layer, signal in points:
            # Casting FP16/FP32 up to FP64 is exact, so neither side is rounded down.
            values = {
                name: trace[(layer, signal)].to(device="cpu", dtype=torch.float64)
                for name, trace in traces.items()
            }
            comparisons = {
                "engine_vs_fp32": compare_activation_l2(values["engine"], values["fp32"]),
                "hf_fp16_vs_fp32": compare_activation_l2(values["hf_fp16"], values["fp32"]),
                "engine_vs_hf_fp16": compare_activation_l2(values["engine"], values["hf_fp16"]),
            }
            norms = comparisons["engine_vs_fp32"]["hf_token_norm"]
            record = {
                "type": "point",
                "layer": layer,
                "signal": signal,
                "engine_token_norm": torch.linalg.vector_norm(values["engine"], dim=-1).tolist(),
                "hf_fp16_token_norm": comparisons["engine_vs_hf_fp16"]["hf_token_norm"],
                "hf_fp32_token_norm": norms,
            }
            for name, metrics in comparisons.items():
                record[name] = {
                    "token_l2": metrics["token_l2"],
                    "token_relative_error": relative_error_from_norms(metrics["token_l2"], norms),
                }
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    cache_volume.commit()
    return {
        "remote_path": str(path.relative_to(Path(str(REMOTE_CACHE)))),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# GPU job for one prompt: engine FP16, then HF FP16 eager, then HF FP32 eager, each
# exactly once, whole-prompt prefill, batch size one. Each model is released before
# the next loads. No warm-up forwards. TF32 is disabled before any CUDA work.
@app.function(
    image=dependency_image,
    gpu="T4",
    cpu=2,
    memory=65536,
    timeout=3600,
    volumes={str(REMOTE_CACHE): cache_volume},
    single_use_containers=True,
)
def run_prompt_referee(target, output_name):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    from huggingface_hub import snapshot_download
    from sgl_interp.capture import ActivationCaptureConfig
    from sgl_interp.core import SamplingParams
    from sgl_interp.layers.rotary import get_rope
    from sgl_interp.llm import LLM
    from transformers import AutoConfig, AutoModelForCausalLM

    ids = target["prompt"]["token_ids"]
    snapshot = snapshot_download(
        target["model"],
        revision=target["model_revision"],
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
    )
    config = AutoConfig.from_pretrained(snapshot)
    if config.model_type not in {"llama", "qwen3"}:
        raise ValueError("Capture hooks have been audited for dense Llama and Qwen3 only")
    num_layers = config.num_hidden_layers
    points = capture_points(num_layers)
    traces, calls, memory = {}, {}, {}
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
            layer_ids=tuple(range(num_layers)), capture_lm_head_input=True
        ),
    )
    try:
        # One sampled token triggers the whole-prompt prefill without a decode forward.
        llm.generate([ids], SamplingParams(top_k=1, temperature=0, max_tokens=1, ignore_eos=True))
        calls["engine_fp16"] = 1
        captures = llm.take_activation_captures()
        validate_engine_capture_spans(captures, points, [(0, len(ids))])
        traces["engine"] = {
            point: reconstruct_capture(captures, *point, 0, ids) for point in points
        }
        captures.clear()
    finally:
        llm.shutdown()
    del llm
    get_rope.cache_clear()
    mem_cleanup()

    for name, dtype in (("hf_fp16", torch.float16), ("fp32", torch.float32)):
        reference = (
            AutoModelForCausalLM.from_pretrained(
                snapshot, torch_dtype=dtype, attn_implementation="eager"
            )
            .eval()
            .to("cuda")
        )
        if any(parameter.dtype != dtype for parameter in reference.parameters()):
            raise ValueError("HF model precision differs from requested settings")
        torch.cuda.reset_peak_memory_stats()
        print(
            f"Row {target['prompt']['dataset_row']}: {name} loaded; "
            f"free GPU memory {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB. One forward next.",
            flush=True,
        )
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
            raise ValueError("TF32 must remain off")
        with torch.autocast("cuda", enabled=False):
            traces[name] = capture_hf_prefill(reference, ids, "whole")
        calls[name] = 1
        memory[name] = torch.cuda.max_memory_allocated()
        del reference
        mem_cleanup()
    return save_referee_token_metrics(
        target,
        output_name,
        traces,
        {
            "gpu": torch.cuda.get_device_name(),
            "new_forward_calls": calls,
            "peak_allocated_bytes": memory,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        num_layers,
    )


# Local summary of a finished batch: reads the downloaded row-*.jsonl files (numbers
# only, no tensors), checks each one, and pools actual tokens across prompts.
def summarize_batch(batch_directory):
    batch_directory = Path(batch_directory)
    progress = json.loads((batch_directory / "progress.json").read_text())
    prompts, revisions = [], set()
    for run in progress:
        path = batch_directory / Path(run["path"]).name
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["manifest"]["sha256"]:
            raise ValueError(f"{path.name} failed checksum verification")
        with path.open(encoding="utf-8") as handle:
            setup = json.loads(next(handle))
            points = [json.loads(line) for line in handle]
        num_layers = (len(points) - 1) // 2
        if [(point["layer"], point["signal"]) for point in points] != capture_points(num_layers):
            raise ValueError(f"{path.name} has missing capture points")
        if setup.get("matmul_allow_tf32") or setup.get("cudnn_allow_tf32"):
            raise ValueError(f"{path.name}: TF32 was enabled")
        if setup["mode"] != "whole":
            raise ValueError(f"{path.name}: expected whole-prompt prefill")
        # Row 2802 in the 2026-11-28 batch reused saved traces, so it has no new forwards.
        expected_calls = {} if setup.get("reused") else {"engine_fp16": 1, "hf_fp16": 1, "fp32": 1}
        if setup["new_forward_calls"] != expected_calls:
            raise ValueError(f"{path.name}: unexpected number of forward passes")
        for point in points:
            norms = point["hf_fp32_token_norm"]
            if len(norms) != len(setup["token_ids"]):
                raise ValueError(f"{path.name}: incomplete FP32 norm coverage")
            for name in ("engine_vs_fp32", "hf_fp16_vs_fp32", "engine_vs_hf_fp16"):
                if point[name]["token_relative_error"] != relative_error_from_norms(
                    point[name]["token_l2"], norms
                ):
                    raise ValueError(f"{path.name}: relative error does not use the FP32 norm")
        revisions.add(setup["model_revision"])
        prompts.append(
            {"dataset_row": setup["dataset_row"], "token_ids": setup["token_ids"], "points": points}
        )
    if len(revisions) != 1:
        raise ValueError("Prompts in one batch must share a checkpoint revision")
    prompts.sort(key=lambda prompt: prompt["dataset_row"])
    result = summarize_referee_populations(prompts)
    (batch_directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    for filename, key in (
        ("final-state-summary.csv", "summaries"),
        ("capture-point-summary.csv", "point_statistics"),
    ):
        with (batch_directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    print_final_state_table(result)
    return result


def print_final_state_table(result):
    """Final normalized state, engine / HF FP16, relative error in percent."""
    print("population | tokens | median % | p99 % | p99.9 % | max % | >2% | >5% | closer")
    for row in result["summaries"]:
        cells = [
            f"{100 * row['engine_' + stat]:.4f} / {100 * row['hf_fp16_' + stat]:.4f}"
            for stat in ("median", "p99", "p99_9", "max")
        ]
        counts = [
            f"{row['engine_' + key]} / {row['hf_fp16_' + key]}"
            for key in ("above_2_percent", "above_5_percent")
        ]
        closer = f"{row['engine_closer']} / {row['hf_fp16_closer']}"
        print(" | ".join([row["population"], str(row["tokens"]), *cells, *counts, closer]))


# Launch loop: one fresh T4 container per prompt, shortest prompt first so a memory
# failure shows up early. Prompts run one after another; there are no retries, so an
# out-of-memory error stops the batch instead of changing GPU or settings.
def run_referee(workload_path, model_id, dataset_rows):
    import time

    workload = json.loads(Path(workload_path).read_text(encoding="utf-8"))
    selected = {
        prompt["dataset_row"]: prompt
        for prompt in workload["prompts"]
        if prompt["dataset_row"] in set(dataset_rows)
    }
    missing = set(dataset_rows) - set(selected)
    if missing:
        raise ValueError(f"Rows not in the saved workload: {sorted(missing)}")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    batch_directory = OUTPUT_DIR / f"fp32-referee-prompts-{timestamp}"
    batch_directory.mkdir(parents=True)
    jobs, completed = [], []
    ordered = sorted(selected, key=lambda row: len(selected[row]["token_ids"]))
    with modal.enable_output(), app.run(detach=False):
        for row in ordered:
            target = {
                "model": model_id,
                "model_revision": workload["tokenizer_revision"],
                "prompt": selected[row],
            }
            print(f"Starting row {row}, {len(selected[row]['token_ids'])} tokens.", flush=True)
            call = run_prompt_referee.with_options(secrets=hugging_face_modal_secrets()).spawn(
                target, f"fp32-referee-prompts-{timestamp}/row-{row}"
            )
            jobs.append({"dataset_row": row, "call_id": call.object_id})
            (batch_directory / "jobs.json").write_text(json.dumps(jobs, indent=2))
            manifest = call.get()
            path = batch_directory / f"row-{row}.jsonl"
            with path.open("wb") as handle:
                for chunk in cache_volume.read_file(manifest["remote_path"]):
                    handle.write(chunk)
            if (
                path.stat().st_size != manifest["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]
            ):
                raise ValueError("Downloaded prompt metrics failed checksum verification")
            completed.append({"dataset_row": row, "path": str(path), "manifest": manifest})
            (batch_directory / "progress.json").write_text(json.dumps(completed, indent=2))
            print(f"Row {row} complete and downloaded.", flush=True)
    if len(completed) != len(ordered):
        raise RuntimeError(f"Batch disconnected after {len(completed)}/{len(ordered)} rows")
    summarize_batch(batch_directory)
    print(f"Saved results in {batch_directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload", help="Saved pretokenized workload JSON")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--rows", type=int, nargs="+", help="Dataset rows from the workload")
    parser.add_argument("--summarize", help="Re-summarize a finished batch directory locally")
    args = parser.parse_args()
    if args.summarize:
        summarize_batch(args.summarize)
    elif args.workload and args.rows:
        run_referee(args.workload, args.model_id, args.rows)
    else:
        parser.error("Pass --workload and --rows, or --summarize")


if __name__ == "__main__":
    main()
