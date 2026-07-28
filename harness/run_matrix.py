#!/usr/bin/env python3
"""Matrix orchestrator for the 1.2 measurement (and its 5090 dry-run).

Runs the full pipeline: canonical baseline ladder (2 identical arms) ->
A/A arms (cap A) on the A/B ladder -> K paired A/B repeats -> manifest.
Server lifecycle per arm: own process group (setsid), health-gated start,
group kill + GPU-orphan verification on teardown (never --skip-gpu-cleanup).

Cache contract: radix/prefix cache disabled at launch => every request is
cold by construction (recorded in the manifest as the reset method).

  dry-run (5090, Qwen3-1.7B, plain sglang):
    run_matrix.py --profile dry --run-dir results/i1024_1p2/dryrun1
  H100 (sgl-omni serve): --profile real  (launch template; see runbook)
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).parent
WORKSPACE = HERE.parent
VENV_PY = str(WORKSPACE / "sglang-omni/.venv/bin/python")


def log(msg: str) -> None:
    print(f"[run_matrix {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- server ----

def launch_cmd(profile: str, cfg: dict, cap: int, port: int) -> list[str]:
    if profile == "dry":
        d = cfg["dry_run"]
        return [VENV_PY, "-m", "sglang.launch_server",
                "--model-path", d["model"],
                "--port", str(port),
                "--max-running-requests", str(cap),
                "--mem-fraction-static", str(d["mem_fraction_static"]),
                "--disable-radix-cache"]
    # real: sgl-omni serve (H100). Verified locally via merged-config probes
    # (evidence/serve_config_probes.md): --thinker-max-running-requests (added by
    # #1135) targets the thinker cap; the full pipeline config carries
    # decode_log_interval; CLI overrides apply on top of --config.
    d = cfg
    return [VENV_PY, "-m", "sglang_omni.cli", "serve",
            "--model-path", d["model"]["path"],
            "--text-only",
            "--thinker-tp-size", str(d["model"]["tensor_parallel"]),
            "--thinker-gpus", "0,1",
            "--mem-fraction-static", str(d["serving"]["mem_fraction_static"]),
            "--thinker-max-running-requests", str(cap),
            "--config", str(WORKSPACE / d["serving"]["pipeline_config"]),
            "--port", str(port)]


def wait_health(base_url: str, timeout_s: float = 600.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                log(f"server healthy after {time.time() - t0:.0f}s")
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"server not healthy after {timeout_s}s")


def gpu_orphans() -> list[tuple[str, str]]:
    """(pid, process_name) of compute processes from our venv still on GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    orphans = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and "sglang-omni/.venv" in parts[1]:
            orphans.append((parts[0], parts[1]))
    return orphans


class Server:
    def __init__(self, cmd: list[str], log_path: Path, base_url: str):
        self.cmd, self.log_path, self.base_url = cmd, log_path, base_url
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # The sgl-omni launcher silently falls back to a random port when the
        # requested one is still held (previous arm's socket draining after
        # SIGTERM) — health checks would then poll a dead port for 10 minutes.
        # Wait for the port to be released before launching.
        # Mirror the launcher's own availability test (plain bind, no
        # SO_REUSEADDR): after an arm's teardown the port has no listener but
        # TIME_WAIT pairs still make bind() fail for up to ~60s, which is
        # exactly what triggers the fallback.
        port = int(self.base_url.rsplit(":", 1)[1])
        for _ in range(40):
            try:
                with socket.socket() as s:
                    s.bind(("", port))
                break
            except OSError:
                log(f"port {port} not yet bindable (TIME_WAIT drain), waiting ...")
                time.sleep(3)
        else:
            raise RuntimeError(f"port {port} still unbindable after 120s")
        logf = self.log_path.open("w")
        log(f"launch: {' '.join(self.cmd)} (log -> {self.log_path})")
        # venv bin must be on PATH: flashinfer's JIT shells out to `ninja`
        env = dict(os.environ)
        venv_bin = str(Path(VENV_PY).parent)
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        # Never let the launcher silently serve on a fallback port — fail fast
        # so the guard above (or a real conflict) surfaces in seconds, not 10 min.
        env["SGLANG_OMNI_STRICT_PORT"] = "1"
        self.proc = subprocess.Popen(
            self.cmd, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True, env=env)  # own process group
        try:
            wait_health(self.base_url)
        except Exception:
            log("health wait failed — tearing down the just-launched server")
            self.stop()
            raise

    def stop(self) -> None:
        if self.proc is None:
            return
        pgid = os.getpgid(self.proc.pid)
        log(f"teardown: SIGTERM process group {pgid}")
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            log("SIGTERM timeout, escalating to SIGKILL on group")
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        self.proc = None
        # GPU-orphan verification (identity-checked: venv path filter only)
        for _ in range(20):
            if not gpu_orphans():
                log("GPU clean")
                return
            time.sleep(3)
        for pid, name in gpu_orphans():
            log(f"orphan worker pid={pid} ({name[:80]}) — SIGKILL (venv-verified)")
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(3)
        if gpu_orphans():
            raise RuntimeError(f"GPU still held after cleanup: {gpu_orphans()}")


# ---------------------------------------------------------------- levels ----

def run_level(base_url: str, model: str, section: dict, level: int,
              out_path: Path, label: str, output_tokens: int,
              seed: int | None) -> None:
    req_cfg = section.get("requests") or section
    cmd = [VENV_PY, str(HERE / "bench_client.py"),
           "--base-url", base_url, "--model", model,
           "--prompt-file", str(WORKSPACE / "harness/prompts_200w_96.jsonl"),
           "--concurrency", str(level),
           "--num-requests", str(req_cfg["requests_per_level"]),
           "--warmup", str(req_cfg.get("warmup_requests", 4)),
           "--output-tokens", str(output_tokens),
           "--temperature", "0.0",
           "--out", str(out_path), "--label", label]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, check=True)


def run_ladder(server: Server, model: str, section: dict, levels: list[int],
               out_dir: Path, tag: str, output_tokens: int,
               seed: int | None) -> None:
    for level in levels:
        run_level(server.base_url, model, section, level,
                  out_dir / f"c{level}.json", f"{tag}-c{level}",
                  output_tokens, seed)


# ------------------------------------------------------------------ main ----

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["dry", "real"], required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=str(HERE / "config_1p2.yaml"))
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--stages", default="baseline,aa,ab",
                   help="comma list: baseline,aa,ab")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    dry = a.profile == "dry"
    section = cfg["dry_run"] if dry else cfg
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    port = a.port or (30801 if dry else cfg["serving"]["port"])
    base_url = f"http://127.0.0.1:{port}"
    model = section["model"] if dry else cfg["model"]["path"]
    output_tokens = section["output_tokens"] if dry else cfg["workload"]["output_tokens"]
    seed = None  # seedless greedy contract (calibration mode differs; see runbook)
    arms = section["arms"]
    cap_a = arms["A"]["max_running_requests"]
    cap_b = arms["B"]["max_running_requests"]
    ladders = section["ladders"]
    repeats = section["repeats"]
    stages = a.stages.split(",")
    log_dir = run_dir / "server_logs"

    if gpu_orphans():
        raise RuntimeError(f"GPU not clean before start: {gpu_orphans()}")

    telemetry = cfg.get("telemetry", {}).get("event_recorder", {})
    record_events = (not dry) and telemetry.get("enable", False)

    def with_server(cap: int, name: str, fn) -> None:
        srv = Server(launch_cmd(a.profile, cfg, cap, port),
                     log_dir / f"{name}.log", base_url)
        srv.start()
        # Event recorder on for every real arm (uniform overhead across
        # baseline/AA/AB, matching the calibration & audio stages).
        if record_events:
            event_dir = (run_dir / "events" / name).resolve()
            event_dir.mkdir(parents=True, exist_ok=True)
            requests.post(f"{base_url}{telemetry['start_route']}",
                          json={"run_id": name, "event_dir": str(event_dir)},
                          timeout=30).raise_for_status()
        try:
            fn(srv)
        finally:
            if record_events:
                try:
                    requests.post(f"{base_url}{telemetry['stop_route']}",
                                  json={}, timeout=30)
                except requests.RequestException:
                    log(f"WARN: stop_request_profile failed for {name}")
            srv.stop()

    # 1) canonical baseline ladder, two identical arms (default cap = arm B)
    if "baseline" in stages:
        for arm in (1, 2):
            log(f"=== baseline arm{arm} (cap {cap_b}) ===")
            with_server(cap_b, f"baseline_arm{arm}", lambda srv, arm=arm: run_ladder(
                srv, model, section, ladders["baseline"],
                run_dir / "baseline" / f"arm{arm}", f"baseline{arm}",
                output_tokens, seed))

    # 2) A/A arms on the A/B ladder (cap A = the rollback/old default)
    if "aa" in stages:
        for arm in range(1, repeats["aa_arms"] + 1):
            log(f"=== A/A arm{arm} (cap {cap_a}) ===")
            with_server(cap_a, f"aa_arm{arm}", lambda srv, arm=arm: run_ladder(
                srv, model, section, ladders["admission_ab"],
                run_dir / "aa" / f"arm{arm}", f"aa{arm}",
                output_tokens, seed))

    # 3) K paired A/B repeats (adjacent in time within each pair)
    if "ab" in stages:
        for k in range(1, repeats["ab_pairs"] + 1):
            for cap in (cap_a, cap_b):
                log(f"=== A/B pair{k} cap{cap} ===")
                with_server(cap, f"ab_pair{k}_cap{cap}", lambda srv, k=k, cap=cap: run_ladder(
                    srv, model, section, ladders["admission_ab"],
                    run_dir / "ab" / f"pair{k}" / f"cap{cap}", f"ab{k}-cap{cap}",
                    output_tokens, seed))

    # 4) manifest
    sys.path.insert(0, str(HERE))
    from manifest import build_manifest  # noqa: E402
    manifest = build_manifest(
        repo_dir=str(WORKSPACE / "sglang-omni"),
        python=VENV_PY,
        launch_command=" ".join(launch_cmd(a.profile, cfg, cap_b, port)),
        server_config={
            "profile": a.profile, "model": model,
            "tensor_parallel": 1 if dry else cfg["model"]["tensor_parallel"],
            "gpus": section.get("gpu") if dry else cfg["model"]["hardware"],
            "placement": None if dry else cfg["model"]["placement"],
            "mem_fraction_static": section["mem_fraction_static"] if dry
                                   else cfg["serving"]["mem_fraction_static"],
            "port": port,
            "arms": arms,
        },
        contract={
            "sampling": "seedless greedy (temperature=0, no seed parameter; "
                        "server_args_builder pins random_seed=123, unused under greedy)",
            "canonical_baseline_arm": (
                f"B (cap {cap_b} — current main default post-#1135; "
                f"cap {cap_a} kept as documented rollback). Pending Wenyao "
                "confirmation per contract open question 1."),
            "cache_mode": "cold", "cache_reset_method": "disable-radix-cache at launch",
            "prefix_hit_state": "0 by construction (radix cache disabled)",
            "ladders": ladders, "repeats": repeats,
            "output_tokens": output_tokens,
            "warmup_policy": f"{(section.get('requests') or section).get('warmup_requests', 4)} "
                             "requests per level, same concurrency, excluded from aggregates",
            "gates": cfg["gates"],
        },
        prompt_file=str(WORKSPACE / "harness/prompts_200w_96.jsonl"),
        extra={"dry_run": dry},
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"manifest written; matrix complete under {run_dir}")


if __name__ == "__main__":
    main()
