"""RNG-state isolation (checkpoint-selection protocol §3).

``preserve_rng_states`` snapshots every global random stream (Python,
NumPy, torch CPU, all torch CUDA devices) plus any named torch.Generators,
runs the wrapped block, and restores everything exactly. Wrapping
mid-training evaluation in this guard guarantees that enabling, disabling,
or re-scheduling screening cannot change later training batches, diffusion
noise, timesteps, updates, or the final checkpoint sequence
(tests/protocol/test_rng_preservation.py verifies this end-to-end).
"""

from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def preserve_rng_states(named_generators: dict[str, torch.Generator] | None = None):
    named_generators = named_generators or {}
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_cpu_state = torch.get_rng_state()
    torch_cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    generator_states = {
        name: g.get_state().clone() for name, g in named_generators.items()
    }
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_states is not None:
            torch.cuda.set_rng_state_all(torch_cuda_states)
        for name, g in named_generators.items():
            g.set_state(generator_states[name])
