"""Repository provenance for audit trails."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def current_git_commit(cwd: str | Path | None = None) -> str | None:
    """The current git commit hash, or None when not in a git repository.

    Protocol audit trails record this; run experiments from a committed
    tree so results are attributable to exact code.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_provenance(cwd: str | Path | None = None) -> dict:
    """Content-sensitive Git provenance, including dirty and untracked files.

    ``source_tree_hash`` identifies the actual source state used by a run; the
    commit hash alone does not when experiments are launched from a dirty tree.
    """
    root = Path(cwd) if cwd else Path.cwd()
    commit = current_git_commit(root)

    def is_source_path(rel: str) -> bool:
        path = Path(rel)
        if path.parts and path.parts[0] in {
            "src",
            "scripts",
            "configs",
            "tests",
            "docs",
            ".github",
            "requirements",
        }:
            return True
        return (
            len(path.parts) == 1
            and path.suffix in {".py", ".toml", ".yaml", ".yml", ".md", ".txt"}
            or rel in {".gitignore"}
        )

    try:
        status_run = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"git_commit": commit, "git_dirty": None, "source_tree_hash": None}
    if status_run.returncode != 0:
        return {"git_commit": commit, "git_dirty": None, "source_tree_hash": None}

    status = status_run.stdout.decode("utf-8", errors="surrogateescape")
    changed = [line[3:] for line in status.splitlines() if is_source_path(line[3:])]
    tracked_changed = [
        rel
        for rel in changed
        if not (root / rel).is_file()
        or not any(
            line.startswith("?? ") and line[3:] == rel for line in status.splitlines()
        )
    ]
    diff_bytes = b""
    if tracked_changed:
        try:
            diff_run = subprocess.run(
                ["git", "diff", "--binary", "HEAD", "--", *tracked_changed],
                cwd=root,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"git_commit": commit, "git_dirty": None, "source_tree_hash": None}
        if diff_run.returncode != 0:
            return {"git_commit": commit, "git_dirty": None, "source_tree_hash": None}
        diff_bytes = diff_run.stdout

    digest = hashlib.sha256()
    digest.update((commit or "no-commit").encode())
    digest.update(diff_bytes)
    untracked = []
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        rel = line[3:]
        path = root / rel
        if is_source_path(rel) and path.is_file():
            untracked.append(rel)
    for rel in sorted(untracked):
        digest.update(rel.encode("utf-8", errors="surrogateescape"))
        digest.update((root / rel).read_bytes())
    return {
        "git_commit": commit,
        "git_dirty": bool(changed),
        "source_tree_hash": digest.hexdigest(),
        "changed_files": changed,
    }
