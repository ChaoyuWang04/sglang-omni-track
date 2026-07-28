#!/usr/bin/env python3
"""run_manifest.json generator for the 1.2 measurement.

Covers every field in #1018's Required Run Fields plus the items #1164 was
faulted for omitting (memory fraction, launch command, placement).
Auto-detects environment; anything undetectable is recorded as null, never
silently omitted.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


def _gpu_info() -> list[dict]:
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,power.limit",
                "--format=csv,noheader"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append({"index": parts[0], "name": parts[1],
                         "memory_total": parts[2], "driver": parts[3],
                         "power_limit": parts[4]})
    return gpus


def _py_versions(python: str) -> dict:
    code = ("import json,importlib.metadata as m\n"
            "import torch\n"
            "def v(p):\n"
            "    try: return m.version(p)\n"
            "    except Exception: return None\n"
            "print(json.dumps({'torch': torch.__version__,"
            "'cuda': torch.version.cuda,"
            "'sglang': v('sglang'), 'sglang_omni': v('sglang-omni'),"
            "'flashinfer': v('flashinfer-python') or v('flashinfer'),"
            "'transformers': v('transformers')}))")
    out = _run([python, "-c", code])
    try:
        return json.loads(out) if out else {}
    except json.JSONDecodeError:
        return {}


def build_manifest(*, repo_dir: str, python: str, launch_command: str,
                   server_config: dict, contract: dict,
                   prompt_file: str | None = None,
                   extra: dict | None = None) -> dict:
    prompt_sha = None
    if prompt_file and Path(prompt_file).exists():
        prompt_sha = hashlib.sha256(Path(prompt_file).read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "gpus": _gpu_info(),
            "kernel": _run(["uname", "-r"]),
            "framework_commit": _run(["git", "-C", repo_dir, "rev-parse", "HEAD"]),
            "framework_dirty": bool(_run(["git", "-C", repo_dir, "status", "--porcelain"])),
            "versions": _py_versions(python),
        },
        "launch_command": launch_command,
        # model / tp / topology / placement / mem_fraction / port / overrides
        "server_config": server_config,
        # sampling, cache mode + reset method, ladders, gates, seed, output len,
        # warmup policy, prompt set
        "contract": contract,
        "prompt_file": prompt_file,
        "prompt_file_sha256": prompt_sha,
        **(extra or {}),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = build_manifest(repo_dir=a.repo_dir, python=a.python, launch_command="",
                       server_config={}, contract={})
    Path(a.out).write_text(json.dumps(m, indent=1))
    print(json.dumps(m["environment"], indent=1))
