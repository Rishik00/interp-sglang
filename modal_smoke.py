"""Run a small Mini-SGLang inference and retrieve a Python call trace from Modal.

The expensive CUDA/PyTorch/dependency environment is cached as a Modal Image.
The local Mini-SGLang checkout is mounted separately, so Python source edits do
not rebuild or repackage the image.

Examples:
    modal run modal_smoke.py
    modal run modal_smoke.py --graph
    modal run modal_smoke.py --no-trace
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

import modal

APP_NAME = "mini-sglang-smoke"
MODEL_ID = "Qwen/Qwen3-0.6B"
CAPTURE_LAYERS = (1, 6, 13, 20, 27)
PYTORCH_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel"
REMOTE_REPO = PurePosixPath("/workspace/mini-sglang")
REMOTE_CACHE = PurePosixPath("/cache")
LOCAL_REPO = Path(__file__).resolve().parent
PROJECT_DEPENDENCIES = (
    "accelerate",
    "msgpack",
    "sgl_kernel>=0.3.17.post1",
    "transformers>=4.56.0,<=4.57.3",
    "flashinfer-python>=0.5.3",
    "pyzmq",
    "uvicorn",
    "fastapi",
    "prompt_toolkit",
    "openai",
    "apache-tvm-ffi>=0.1.4",
    "nvidia-cutlass-dsl==4.3.1",
)


dependency_image = (
    modal.Image.from_registry(PYTORCH_IMAGE)
    .entrypoint([])
    .apt_install("git", "libnuma1")
    .uv_pip_install(*PROJECT_DEPENDENCIES)
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "PYTHONPATH": f"{REMOTE_REPO}:{REMOTE_REPO}/python",
            "PYTHONUNBUFFERED": "1",
            "MINISGL_DISABLE_OVERLAP_SCHEDULING": "1",
            "HF_HOME": f"{REMOTE_CACHE}/huggingface",
            "FLASHINFER_WORKSPACE_BASE": f"{REMOTE_CACHE}/flashinfer",
            "TVM_FFI_CACHE_DIR": f"{REMOTE_CACHE}/tvm-ffi",
            "TORCH_EXTENSIONS_DIR": f"{REMOTE_CACHE}/torch-extensions",
        }
    )
    .add_local_dir(
        LOCAL_REPO,
        remote_path=str(REMOTE_REPO),
        copy=False,
        ignore=[
            ".git/**",
            ".venv/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            "**/__pycache__/**",
            "artifacts/**",
        ],
    )
)

cache_volume = modal.Volume.from_name("mini-sglang-cache", create_if_missing=True)
app = modal.App(APP_NAME)


class _Tee(io.TextIOBase):
    """Write trace output to Modal logs and to the returned log file."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class _MiniSGLangCallTracer:
    """Print Python call/return events from the important Mini-SGLang modules."""

    MODULE_PREFIXES = (
        "minisgl.scheduler",
        "minisgl.engine",
        "minisgl.models",
        "minisgl.layers",
        "minisgl.attention",
        "minisgl.kvcache",
    )

    def __init__(self, output: TextIO, max_events: int = 25_000):
        self.output = output
        self.max_events = max_events
        self.events = 0
        self.depth = 0
        self._truncated = False

    def _emit(self, text: str) -> None:
        if self.events >= self.max_events:
            if not self._truncated:
                print(
                    f"[trace] truncated after {self.max_events} events",
                    file=self.output,
                    flush=True,
                )
                self._truncated = True
            return
        print(text, file=self.output, flush=True)
        self.events += 1

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        del arg
        if event not in ("call", "return"):
            return

        module = frame.f_globals.get("__name__", "")
        if not module.startswith(self.MODULE_PREFIXES):
            return

        qualname = frame.f_code.co_qualname
        if event == "call":
            self._emit(f"[call] {'  ' * self.depth}{module}.{qualname}")
            self.depth += 1
        else:
            self.depth = max(0, self.depth - 1)
            self._emit(f"[return] {'  ' * self.depth}{module}.{qualname}")


def _software_report(torch: Any) -> None:
    import transformers

    print(f"[env] python={sys.version.split()[0]}")
    print(f"[env] torch={torch.__version__}")
    print(f"[env] transformers={transformers.__version__}")
    print(f"[env] cuda_runtime={torch.version.cuda}")


def _cuda_report(torch: Any) -> None:
    import flashinfer

    print(f"[env] flashinfer={getattr(flashinfer, '__version__', 'unknown')}")
    print(f"[env] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"[env] gpu={properties.name}")
        print(f"[env] compute_capability={properties.major}.{properties.minor}")
        print(f"[env] gpu_memory_bytes={properties.total_memory}")


@app.function(
    image=dependency_image,
    gpu="T4",
    cpu=2,
    memory=8192,
    timeout=20 * 60,
    volumes={str(REMOTE_CACHE): cache_volume},
)
def run_smoke(
    prompt: str,
    max_tokens: int,
    trace: bool,
    graph: bool,
    capture: bool,
) -> dict[str, Any]:
    """Run deterministic offline inference and return its complete trace log."""
    os.chdir(REMOTE_REPO)
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    log_path = Path("/tmp") / f"mini-sglang-{run_id}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee = _Tee(sys.__stdout__, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"[run] id={run_id}")
            print(f"[run] model={MODEL_ID}")
            print(f"[run] graph={graph}")
            print(f"[run] trace={trace}")
            print(f"[run] capture={capture}")
            print(f"[run] capture_layers={CAPTURE_LAYERS if capture else ()}")
            print(f"[run] prompt_chars={len(prompt)}")
            print(f"[run] max_tokens={max_tokens}")

            import torch
            from minisgl.capture import ActivationCaptureConfig
            from minisgl.core import SamplingParams
            from minisgl.llm import LLM

            _software_report(torch)
            tracer = _MiniSGLangCallTracer(tee) if trace else None
            llm = None
            result = None
            capture_payload = None
            error = None
            started = time.perf_counter()
            try:
                llm = LLM(
                    model_path=MODEL_ID,
                    dtype=torch.float16,
                    cache_type="naive",
                    cuda_graph_max_bs=1 if graph else 0,
                    max_extend_tokens=512,
                    max_running_req=1,
                    max_seq_len_override=512,
                    memory_ratio=0.60,
                    use_pynccl=False,
                    activation_capture=(
                        ActivationCaptureConfig(layer_ids=CAPTURE_LAYERS) if capture else None
                    ),
                )
                if not graph:
                    # GraphRunner.pad_batch() dereferences dummy_req even when
                    # padding is zero and CUDA graphs are disabled. Keep this
                    # upstream eager-mode bug out of the baseline harness.
                    llm.engine.graph_runner.dummy_req = llm.engine.dummy_req
                    print("[smoke] applied eager GraphRunner.dummy_req shim")
                _cuda_report(torch)
                sampling = SamplingParams(
                    top_k=1,
                    temperature=0.0,
                    ignore_eos=True,
                    max_tokens=max_tokens,
                )
                print("[smoke] starting generation")
                if tracer is not None:
                    sys.setprofile(tracer)
                    threading.setprofile(tracer)
                result = llm.generate([prompt], sampling)
                sys.setprofile(None)
                threading.setprofile(None)
                captures = llm.take_activation_captures()
                if capture:
                    expected_keys = {
                        (layer_id, signal)
                        for layer_id in CAPTURE_LAYERS
                        for signal in ("attention_output", "post_residual")
                    }
                    actual_keys = {(item.layer_id, item.signal) for item in captures}
                    if actual_keys != expected_keys:
                        raise RuntimeError(
                            f"Unexpected captures: got {sorted(actual_keys)}, "
                            f"expected {sorted(expected_keys)}"
                        )
                    serializable_captures = []
                    for item in captures:
                        if not torch.isfinite(item.tensor).all():
                            raise RuntimeError(
                                f"Non-finite values in {item.signal} at layer {item.layer_id}"
                            )
                        print(
                            f"[capture] layer={item.layer_id} signal={item.signal} "
                            f"shape={tuple(item.tensor.shape)} dtype={item.tensor.dtype} "
                            f"phase={item.phase}"
                        )
                        serializable_captures.append(
                            {
                                "layer_id": item.layer_id,
                                "signal": item.signal,
                                "phase": item.phase,
                                "tensor": item.tensor,
                                "token_ids": item.token_ids,
                                "spans": [
                                    {
                                        "request_id": span.request_id,
                                        "tensor_start": span.tensor_start,
                                        "tensor_end": span.tensor_end,
                                        "token_start": span.token_start,
                                        "token_end": span.token_end,
                                    }
                                    for span in item.spans
                                ],
                            }
                        )
                    capture_buffer = io.BytesIO()
                    torch.save(
                        {
                            "model": MODEL_ID,
                            "layers": CAPTURE_LAYERS,
                            "captures": serializable_captures,
                        },
                        capture_buffer,
                    )
                    capture_payload = capture_buffer.getvalue()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                print(f"[smoke] result={result!r}")
                print(f"[smoke] elapsed_seconds={elapsed:.6f}")
                print("[smoke] status=PASS")
            except BaseException:
                error = traceback.format_exc()
                print("[smoke] status=FAIL")
                print(error)
            finally:
                sys.setprofile(None)
                threading.setprofile(None)
                if llm is not None:
                    try:
                        llm.shutdown()
                    except BaseException:
                        shutdown_error = traceback.format_exc()
                        print("[smoke] shutdown_status=FAIL")
                        print(shutdown_error)
                        error = error or shutdown_error
                try:
                    cache_volume.commit()
                except BaseException:
                    commit_error = traceback.format_exc()
                    print("[smoke] cache_commit_status=FAIL")
                    print(commit_error)
                    error = error or commit_error

    return {
        "run_id": run_id,
        "model": MODEL_ID,
        "graph": graph,
        "trace": trace,
        "capture": capture,
        "capture_payload": capture_payload,
        "result": result,
        "error": error,
        "log": log_path.read_text(encoding="utf-8"),
    }


@app.local_entrypoint()
def main(
    prompt: str = "The capital of France is",
    max_tokens: int = 3,
    trace: bool = True,
    graph: bool = False,
    capture: bool = False,
) -> None:
    """Run the Modal smoke test and save its remote log in this checkout."""
    response = run_smoke.remote(
        prompt=prompt,
        max_tokens=max_tokens,
        trace=trace,
        graph=graph,
        capture=capture,
    )
    output_dir = LOCAL_REPO / "artifacts" / "modal-traces"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{response['run_id']}.log"
    output_path.write_text(response["log"], encoding="utf-8")

    print(f"Result: {response['result']}")
    print(f"Trace:  {output_path}")
    if response["capture_payload"] is not None:
        capture_path = output_dir / f"{response['run_id']}.activations.pt"
        capture_path.write_bytes(response["capture_payload"])
        print(f"Activations: {capture_path}")
    if response["error"] is not None:
        raise RuntimeError(f"Modal smoke test failed; full trace saved to {output_path}")
