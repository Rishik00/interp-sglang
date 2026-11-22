from __future__ import annotations

import torch
from sglang_interp.kernel import test_tensor
from sglang_interp.utils import call_if_main


@call_if_main()
def main():
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    test_tensor(x, y)
