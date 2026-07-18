"""Runtime and hardware provenance recorded with claim-bearing results."""

from __future__ import annotations

import platform
import sys

import numpy as np
import torch


def runtime_provenance() -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "gpu_names": (
            [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
        "byteorder": sys.byteorder,
    }
