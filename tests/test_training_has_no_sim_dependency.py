"""Training must work with simulator packages completely unavailable.

Runs a subprocess where importing mani_skill / gymnasium / sapien raises,
then imports every training-side module AND runs a few real gradient steps
of both trainers on a fabricated dataset.
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

BLOCKED_TRAINING_SCRIPT = r"""
import sys
sys.path.insert(0, {src!r})

import importlib.abc


class SimBlocker(importlib.abc.MetaPathFinder):
    BLOCKED = ("mani_skill", "gymnasium", "sapien")

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"BLOCKED simulator import during training: {{fullname}}")
        return None


sys.meta_path.insert(0, SimBlocker())

# Every module used by training must import cleanly with sim blocked.
import actsemble.data.reader
import actsemble.data.torch_dataset
import actsemble.data.validation
import actsemble.policies.diffusion.policy
import actsemble.components.action_chunk_compatibility
from actsemble.training.train_diffusion_policy import train_diffusion_policy
from actsemble.training.train_component import train_component

# And training must RUN with sim blocked.
sys.path.insert(0, {tests!r})
from conftest import make_episodes, make_metadata, TINY_POLICY_CFG, TINY_COMPONENT_CFG
from actsemble.data.writer import write_dataset

dataset = {tmp!r} + "/blocked.h5"
write_dataset(dataset, make_episodes(4, T=30), make_metadata())

out = train_diffusion_policy(
    policy_cfg=TINY_POLICY_CFG, dataset_path=dataset,
    output_dir={tmp!r} + "/p", max_steps=3, device="cpu",
)
assert out["steps"] == 3
out = train_component(
    component_cfg=TINY_COMPONENT_CFG, dataset_path=dataset,
    output_dir={tmp!r} + "/c", max_steps=3, device="cpu",
)
assert out["steps"] == 3
print("TRAINING_WITHOUT_SIM_OK")
"""


def test_training_runs_with_simulator_blocked(tmp_path):
    script = BLOCKED_TRAINING_SCRIPT.format(
        src=str(SRC),
        tests=str(Path(__file__).resolve().parent),
        tmp=str(tmp_path),
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "TRAINING_WITHOUT_SIM_OK" in proc.stdout


def test_sim_modules_are_not_imported_by_training_modules():
    """Static guard: the training import graph must not reach sim adapters."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(SRC)!r});"
                "import actsemble.training.train_diffusion_policy;"
                "import actsemble.training.train_component;"
                "bad = [m for m in sys.modules if m.split('.')[0] in "
                "('mani_skill', 'gymnasium', 'sapien')];"
                "bad += [m for m in sys.modules if m.startswith('actsemble.sim')];"
                "assert not bad, f'sim modules imported: {bad}';"
                "print('IMPORT_GRAPH_CLEAN')"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "IMPORT_GRAPH_CLEAN" in proc.stdout
