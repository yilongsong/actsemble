#!/usr/bin/env python
"""Simple GPU-aware job pool for Phase 0A pipelines.

Reads a JSON list of jobs [{"args": [...], "log": "path"}], runs them with
N workers, worker i pinned to GPU (i % num_gpus). Each job is a
scripts/phase0a.py invocation. Failed jobs are reported at the end;
exit code is nonzero if any failed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_job(job: dict, gpu: int) -> tuple[dict, int]:
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    script = job.get("script", "scripts/phase0a.py")  # per-job script override
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [sys.executable, str(REPO / script), *job["args"]],
            stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO,
        )
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[runner] {status} gpu{gpu} {' '.join(job['args'])} ({dt/60:.1f} min)", flush=True)
    return job, proc.returncode


def main() -> int:
    jobs_file, workers, num_gpus = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    jobs = json.loads(Path(jobs_file).read_text())
    print(f"[runner] {len(jobs)} jobs, {workers} workers, {num_gpus} GPUs", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for i, job in enumerate(jobs):
            futures.append(pool.submit(run_job, job, i % num_gpus))
        for f in futures:
            job, rc = f.result()
            if rc != 0:
                failures.append(job)
    if failures:
        print(f"[runner] {len(failures)} FAILED job(s):", flush=True)
        for j in failures:
            print(f"  {' '.join(j['args'])} (log: {j['log']})", flush=True)
        return 1
    print("[runner] all jobs succeeded", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
