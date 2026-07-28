#!/usr/bin/env python3
"""Constant in-flight concurrency streaming benchmark client (text -> text).

Targets an OpenAI-compatible /v1/chat/completions endpoint (sglang or
sglang-omni). Records per-request TTFT, total latency, completion text and
token counts; failed requests are recorded, never dropped. Warmup requests
run first at the same concurrency and are excluded from all aggregates.

Usage (normally driven by run_matrix.py):
  bench_client.py --base-url http://localhost:8000 --model MODEL \
      --prompt-file prompts_200w_96.jsonl --concurrency 16 \
      --num-requests 96 --warmup 4 --output-tokens 128 --out level.json
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--num-requests", type=int, required=True)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None,
                   help="omit for the seedless greedy contract")
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--audios", default=None,
                   help="comma-separated server-readable audio paths; requests "
                        "cycle over them (audio->text modality). Sent as the "
                        "top-level 'audios' field with empty message content "
                        "per docs/basic_usage/qwen3_omni.md.")
    return p.parse_args()


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def build_payload(args, prompt: dict, audio_path: str | None) -> dict:
    """Chat-completions payload. Audio requests use the sglang-omni extension:
    top-level `audios` (server-readable paths) with empty message content when
    all semantic content comes from the audio (docs/basic_usage/qwen3_omni.md)."""
    payload = {
        "model": args.model,
        "messages": [{"role": "user",
                      "content": "" if audio_path else prompt["prompt"]}],
        "max_tokens": args.output_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if audio_path:
        payload["audios"] = [audio_path]
        payload["modalities"] = ["text"]
    if args.seed is not None:
        payload["seed"] = args.seed
    return payload


async def one_request(session, args, prompt: dict, request_id: str,
                      audio_path: str | None = None) -> dict:
    payload = build_payload(args, prompt, audio_path)
    rec = {
        "request_id": request_id,
        "prompt_id": prompt["prompt_id"],
        "audio": audio_path,
        "success": False,
        "http_status": None,
        "ttft_s": None,
        "latency_s": None,
        "completion_text": "",
        "completion_tokens": None,
        "prompt_tokens": None,
        "error": None,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{args.base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=args.timeout_s),
        ) as resp:
            rec["http_status"] = resp.status
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:500]}"
                rec["latency_s"] = time.perf_counter() - t0
                return rec
            chunks: list[str] = []
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    rec["completion_tokens"] = obj["usage"].get("completion_tokens")
                    rec["prompt_tokens"] = obj["usage"].get("prompt_tokens")
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = time.perf_counter() - t0
                        chunks.append(piece)
            rec["latency_s"] = time.perf_counter() - t0
            rec["completion_text"] = "".join(chunks)
            rec["success"] = rec["ttft_s"] is not None
            if not rec["success"]:
                rec["error"] = "stream ended with no content delta"
    except Exception as e:  # noqa: BLE001 — every failure is a recorded data point
        rec["latency_s"] = time.perf_counter() - t0
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


async def run_pool(session, args, jobs: list[dict], phase: str) -> tuple[list, float]:
    """Constant in-flight pool: `concurrency` workers pull from one queue."""
    audios = [a for a in (args.audios or "").split(",") if a]
    queue: asyncio.Queue = asyncio.Queue()
    for i, prompt in enumerate(jobs):
        queue.put_nowait((i, prompt))
    results: list[dict] = []

    async def worker():
        while True:
            try:
                i, prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            audio = audios[i % len(audios)] if audios else None
            rec = await one_request(session, args, prompt, f"{phase}-{i:04d}",
                                    audio_path=audio)
            results.append(rec)

    t0 = time.perf_counter()
    workers = [asyncio.create_task(worker())
               for _ in range(min(args.concurrency, len(jobs)))]
    await asyncio.gather(*workers)
    wall = time.perf_counter() - t0
    results.sort(key=lambda r: r["request_id"])
    return results, wall


async def main_async(args):
    prompts = load_prompts(args.prompt_file)
    measured_jobs = [prompts[i % len(prompts)] for i in range(args.num_requests)]
    warmup_jobs = [prompts[(args.num_requests + i) % len(prompts)]
                   for i in range(args.warmup)]

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        warmup_results = []
        if warmup_jobs:
            warmup_results, _ = await run_pool(session, args, warmup_jobs, "warmup")
        measured, wall = await run_pool(session, args, measured_jobs, "req")

    ok = [r for r in measured if r["success"]]
    total_completion_tokens = sum(r["completion_tokens"] or 0 for r in ok)
    out = {
        "label": args.label,
        "config": {
            "base_url": args.base_url, "model": args.model,
            "concurrency": args.concurrency, "num_requests": args.num_requests,
            "warmup": args.warmup, "output_tokens": args.output_tokens,
            "temperature": args.temperature, "seed": args.seed,
            "prompt_file": args.prompt_file, "streaming": True,
            "audios": args.audios,
        },
        "wall_s": wall,
        "attempted": len(measured),
        "succeeded": len(ok),
        "failed": len(measured) - len(ok),
        "output_throughput_tok_s": (total_completion_tokens / wall) if wall > 0 else 0.0,
        "warmup_succeeded": sum(1 for r in warmup_results if r["success"]),
        "per_request": measured,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[bench_client] {args.label or args.out}: {len(ok)}/{len(measured)} ok, "
          f"wall {wall:.2f}s, {out['output_throughput_tok_s']:.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
