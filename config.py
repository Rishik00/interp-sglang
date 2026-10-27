
import modal
from pathlib import Path, PurePosixPath

app = modal.App("mini-sglang-capture-benchmark")
CAPTURE_LAYERS = (1, 6, 13, 20, 27)
MODEL_ID = "Qwen/Qwen3-0.6B"
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

PROMPTS = (
    "What is the capital of France?",
    "How many legs does a spider have?",
    "Who wrote the play Romeo and Juliet?",
    "What is the boiling point of water in Celsius at sea level?",
    "What is the largest planet in our solar system?",
    "What gas do plants absorb from the air during photosynthesis?",
    "How many continents are there on Earth?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
    "What is the square root of 144?",
)