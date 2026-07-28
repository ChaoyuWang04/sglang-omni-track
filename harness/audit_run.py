#!/usr/bin/env python3
"""Mechanical completeness + health audit for a 1.2 measurement run.

Verifies — deterministically, against config_1p2.yaml — that the run tree is
COMPLETE and every artifact is HEALTHY. Run this BEFORE declaring the run done
and BEFORE packing. Exit 0 = all checks pass; exit 1 = failures listed.

Usage: audit_run.py --run-dir results/i1024_1p2/h100_r1 [--config harness/config_1p2.yaml]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

FAILS: list[str] = []
WARNS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  ❌ {msg}")


def warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  ⚠️  {msg}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def check_level_json(path: Path, expect_requests: int, min_success: int,
                     output_tokens: int, streaming_usage: bool = True) -> None:
    if not path.exists():
        fail(f"缺文件: {path}")
        return
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        fail(f"{path}: JSON 解析失败 {e}")
        return
    name = str(path)
    if d.get("attempted") != expect_requests:
        fail(f"{name}: attempted={d.get('attempted')} != {expect_requests}")
    if d.get("failed", 0) > 0:
        fail(f"{name}: failed={d['failed']}（失败请求需在报告中单列并归因）")
    if d.get("succeeded", 0) < min_success:
        fail(f"{name}: succeeded={d.get('succeeded')} < {min_success}")
    per = d.get("per_request", [])
    ok_reqs = [r for r in per if r.get("success")]
    if len(per) != d.get("attempted"):
        fail(f"{name}: per_request 条数 {len(per)} != attempted")
    null_ttft = sum(1 for r in ok_reqs if r.get("ttft_s") is None)
    if null_ttft:
        fail(f"{name}: {null_ttft} 条成功请求 ttft 为空（流式解析问题）")
    if streaming_usage:
        null_tok = sum(1 for r in ok_reqs if not r.get("completion_tokens"))
        if null_tok:
            fail(f"{name}: {null_tok} 条成功请求缺 completion_tokens"
                 "（include_usage 未生效，吞吐数字不可信）")
        short = sum(1 for r in ok_reqs
                    if r.get("completion_tokens") and r["completion_tokens"] < output_tokens * 0.5)
        if short > len(ok_reqs) * 0.2:
            warn(f"{name}: {short}/{len(ok_reqs)} 请求输出 tokens < 50% 上限"
                 "（提前 EOS 偏多，吞吐口径与 #1135 可能不可比，报告需注明）")
    if d.get("wall_s", 0) <= 0:
        fail(f"{name}: wall_s 异常")


def check_server_log(path: Path, expect_cap: int | None) -> None:
    if not path.exists():
        fail(f"缺 server 日志: {path}（KV/graph 遥测唯一来源，缺=该 arm 遥测作废）")
        return
    text = path.read_text(errors="replace")
    name = path.name
    for pat, why in [
        (r"Dead stage process", "stage 死亡——该 arm 起该档及之后全部作废"),
        (r"scheduler crashed", "scheduler 崩溃"),
        (r"out of memory|OutOfMemoryError", "OOM"),
    ]:
        if re.search(pat, text):
            fail(f"{name}: 日志含 '{pat}' —— {why}")
    m = re.search(r"Capture cuda graph bs \[([^\]]+)\]", text)
    if not m:
        fail(f"{name}: 无 'Capture cuda graph bs' 行")
    elif expect_cap is not None:
        bs = [int(x) for x in m.group(1).split(",")]
        if max(bs) != expect_cap:
            fail(f"{name}: graph capture max bs={max(bs)} != cap {expect_cap}"
                 "（cap 覆盖未生效！该 arm 全部作废）")
    decode_lines = len(re.findall(r"Decode batch", text))
    if decode_lines == 0:
        fail(f"{name}: 零条 'Decode batch' 行（decode_log_interval 未生效或负载没跑到）")
    tracebacks = len(re.findall(r"Traceback \(most recent call last\)", text))
    if tracebacks:
        warn(f"{name}: 含 {tracebacks} 个 Traceback（逐一人工判读是否影响数据）")


def check_events(event_root: Path, subdirs: list[str]) -> None:
    for sub in subdirs:
        d = event_root / sub
        files = list(d.glob("events_*.jsonl")) if d.exists() else []
        if not files:
            fail(f"events/{sub}: 无事件文件（遥测缺失）")
            continue
        blob = "".join(f.read_text(errors="replace") for f in files)
        for ev in ("thinker_lookahead_decision", "thinker_lookahead_launch",
                   "thinker_lookahead_resolve"):
            if ev not in blob:
                fail(f"events/{sub}: 缺 {ev}（补丁未生效或 recorder 没开）")
        if '"use_lookahead": true' not in blob and '"use_lookahead":true' not in blob:
            warn(f"events/{sub}: 无 use_lookahead=true 的 decision"
                 "（c>=16 下不该发生，检查 async decode 是否开启）")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=str(Path(__file__).parent / "config_1p2.yaml"))
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    run = Path(a.run_dir)
    arms = cfg["arms"]
    cap_a = arms["A"]["max_running_requests"]
    cap_b = arms["B"]["max_running_requests"]
    ladders = cfg["ladders"]
    K = cfg["repeats"]["ab_pairs"]
    req = cfg["requests"]["requests_per_level"]
    min_s = cfg["requests"]["min_success_per_level"]
    out_tok = cfg["workload"]["output_tokens"]
    calib = cfg["calibration"]

    print("== 1. 校准 (calib/) ==")
    for cap in calib["arms"]:
        for c in calib["concurrency"]:
            check_level_json(run / "calib" / f"cap{cap}" / f"c{c}.json",
                             calib["prompts"], min_s, calib["output_tokens"])
    print("== 2. Canonical baseline (baseline/, 2 arms) ==")
    for arm in (1, 2):
        for c in ladders["baseline"]:
            check_level_json(run / "baseline" / f"arm{arm}" / f"c{c}.json",
                             req, min_s, out_tok)
    print("== 3. A/A (aa/, 2 arms) ==")
    for arm in (1, 2):
        for c in ladders["admission_ab"]:
            check_level_json(run / "aa" / f"arm{arm}" / f"c{c}.json",
                             req, min_s, out_tok)
    print(f"== 4. A/B (ab/, K={K} 对) ==")
    for k in range(1, K + 1):
        for cap in (cap_a, cap_b):
            for c in ladders["admission_ab"]:
                check_level_json(run / "ab" / f"pair{k}" / f"cap{cap}" / f"c{c}.json",
                                 req, min_s, out_tok)
    print("== 5. audio->text (audio/) ==")
    for c in cfg["native_audio_baseline"]["ladder"]:
        check_level_json(run / "audio" / f"c{c}.json", req, min_s, out_tok)

    print("== 6. server 日志健康 ==")
    log_dir = run / "server_logs"
    expected_logs = (
        [(f"calib_cap{cap}", cap) for cap in calib["arms"]]
        + [(f"baseline_arm{i}", cap_b) for i in (1, 2)]
        + [(f"aa_arm{i}", cap_a) for i in (1, 2)]
        + [(f"ab_pair{k}_cap{cap}", cap)
           for k in range(1, K + 1) for cap in (cap_a, cap_b)]
        + [(f"events_probe_cap{cap}", cap) for cap in (cap_a, cap_b)]
        + [("audio_cap64", cap_b)]
    )
    for name, cap in expected_logs:
        check_server_log(log_dir / f"{name}.log", cap)

    print("== 7. 遥测事件 ==")
    # Recorder measured at 25-30% throughput cost when active (anomalies.md #6),
    # so perf arms run recorder-OFF; lookahead events come from the dedicated
    # events_probe stage only (one c32 probe per cap, non-perf).
    check_events(run / "events",
                 [f"events_probe_cap{cap}" for cap in (cap_a, cap_b)])

    print("== 8. manifest ==")
    mpath = run / "manifest.json"
    if not mpath.exists():
        fail("缺 manifest.json")
    else:
        m = json.loads(mpath.read_text())
        env = m.get("environment", {})
        for key, val in [("gpus", env.get("gpus")),
                         ("framework_commit", env.get("framework_commit")),
                         ("versions.torch", (env.get("versions") or {}).get("torch")),
                         ("prompt_file_sha256", m.get("prompt_file_sha256"))]:
            if not val:
                fail(f"manifest 缺 {key}")
        if not env.get("framework_dirty"):
            fail("manifest framework_dirty 应为 true（遥测补丁在树上）；"
                 "false 说明补丁没 apply，事件数据来源存疑")
        if len(env.get("gpus", [])) != 2:
            fail(f"manifest gpus 数={len(env.get('gpus', []))} != 2")
    if not (run / "applied_patch.diff").exists():
        warn("缺 applied_patch.diff（pack 阶段生成；若已 pack 可忽略）")

    print()
    print(f"===== 审计结果: {len(FAILS)} FAIL / {len(WARNS)} WARN =====")
    if FAILS:
        print("有 FAIL 项：该 run 不完整或不健康，逐项修复/补跑后重新审计。")
        print("补跑指引：单阶段 `bash scripts/h100_run.sh --fg <stage>`；"
              "单 arm 作废时删除该 arm 目录后重跑对应 stage。")
        sys.exit(1)
    print("完整性与健康检查全部通过。接着跑 analyze.py 出 gate 报告，然后 pack。")


if __name__ == "__main__":
    main()
