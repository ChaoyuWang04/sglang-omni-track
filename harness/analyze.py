#!/usr/bin/env python3
"""Gate arithmetic + report generator for the 1.2 measurement.

Implements #1018 §1.2 verbatim gates against the run tree produced by
run_matrix.py:

  <run_dir>/
    manifest.json
    baseline/arm{1,2}/c{L}.json          canonical baseline ladder (A/A pair)
    aa/arm{1,2}/c{L}.json                A/A arms for the A/B ladder (cap16)
    ab/pair{k}/cap{16,64}/c{L}.json      K paired A/B repeats
    server_logs/...                      scraped stage logs (telemetry appendix)

Gate verdict style follows #1025: "gate was X, measured Y -> verdict".
All thresholds come from config_1p2.yaml — nothing is hardcoded here.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import yaml

# sglang decode log (installed sglang: observability/scheduler_metrics_mixin.py)
DECODE_LINE = re.compile(
    r"Decode batch.*?#running-req: (?P<running>\d+), #token: (?P<token>\d+), "
    r"token usage: (?P<usage>[\d.]+), .*?"
    r"(?:cuda graph|graph): (?P<graph>\w+), "
    r"gen throughput \(token/s\): (?P<gen_tput>[\d.]+), "
    r"#queue-req: (?P<queue>\d+)")


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round(q / 100 * (len(vals) - 1))))
    return vals[idx]


def level_metrics(path: Path) -> dict:
    data = json.loads(path.read_text())
    ok = [r for r in data["per_request"] if r["success"]]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    lats = [r["latency_s"] for r in ok if r["latency_s"] is not None]
    return {
        "attempted": data["attempted"],
        "succeeded": data["succeeded"],
        "failed": data["failed"],
        "output_throughput": data["output_throughput_tok_s"],
        "output_token_count": sum(r["completion_tokens"] or 0 for r in ok),
        "ttft_p50": pctl(ttfts, 50), "ttft_p95": pctl(ttfts, 95),
        "latency_p50": pctl(lats, 50), "latency_p95": pctl(lats, 95),
        "wall_s": data["wall_s"],
        "per_request": data["per_request"],
    }


def load_ladder(dirpath: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(dirpath.glob("c*.json")):
        level = int(f.stem[1:])
        out[level] = level_metrics(f)
    return out


def rel_delta(a: float, b: float) -> float | None:
    """Relative delta of b vs a, in percent (positive = b larger)."""
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a * 100.0


def noise_band(arm1: dict, arm2: dict, metrics: list[str]) -> dict:
    """Per-level, per-metric |relative delta| between two identical arms."""
    band = {}
    for level in sorted(set(arm1) & set(arm2)):
        band[level] = {}
        for m in metrics:
            d = rel_delta(arm1[level].get(m), arm2[level].get(m))
            band[level][m] = abs(d) if d is not None else None
    return band


def token_diff(a: dict, b: dict) -> dict:
    """Per-prompt output comparison between two arms at one level."""
    def by_prompt(level: dict) -> dict:
        return {r["prompt_id"]: r for r in level["per_request"] if r["success"]}
    pa, pb = by_prompt(a), by_prompt(b)
    common = sorted(set(pa) & set(pb))
    text_mismatch = [p for p in common
                     if pa[p]["completion_text"] != pb[p]["completion_text"]]
    tok_mismatch = [p for p in common
                    if pa[p]["completion_tokens"] != pb[p]["completion_tokens"]]
    return {"compared": len(common),
            "text_mismatches": len(text_mismatch),
            "token_count_mismatches": len(tok_mismatch),
            "mismatched_prompt_ids": text_mismatch[:20]}


def parse_decode_log(path: Path) -> dict:
    rows = [m.groupdict() for m in
            (DECODE_LINE.search(l) for l in path.read_text(errors="replace").splitlines())
            if m]
    if not rows:
        return {"decode_lines": 0}
    usages = [float(r["usage"]) for r in rows]
    graph_true = sum(1 for r in rows if r["graph"] == "True")
    return {
        "decode_lines": len(rows),
        "token_usage_max": max(usages),
        "token_usage_p50": pctl(usages, 50),
        "running_req_max": max(int(r["running"]) for r in rows),
        "queue_req_max": max(int(r["queue"]) for r in rows),
        "cuda_graph_hit_rate": graph_true / len(rows),
    }


def fmt(v, nd=3):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def analyze(run_dir: Path, cfg: dict, dry_run: bool) -> str:
    section = cfg["dry_run"] if dry_run else cfg
    gates = cfg["gates"]
    lines: list[str] = []
    verdicts: list[bool] = []

    def emit(s=""):
        lines.append(s)

    emit(f"# 1.2 gate report — `{run_dir.name}`" + (" (DRY RUN — numbers are meaningless)" if dry_run else ""))
    emit()

    # ---- Baseline gate -------------------------------------------------
    required = gates["baseline"]["required_fields"]
    metric_map = {"ttft": "ttft_p50", "total_latency": "latency_p50",
                  "output_throughput": "output_throughput",
                  "request_count": "succeeded",
                  "output_token_count": "output_token_count"}
    base_arms = sorted((run_dir / "baseline").glob("arm*"))
    emit("## Baseline gate")
    emit()
    emit('Gate (verbatim #1018 §1.2): "Every concurrency level completes under the '
         'declared contract and reports TTFT, total latency, output throughput, '
         'request count, and output token count."')
    emit()
    baseline_ok = True
    req_cfg = section.get("requests") or section
    min_success = req_cfg["min_success_per_level"]
    for arm_dir in base_arms:
        ladder = load_ladder(arm_dir)
        emit(f"### {arm_dir.name}")
        emit()
        emit("| c | attempted | succeeded | failed | TTFT p50/p95 (s) | latency p50/p95 (s) | out tok/s | out tokens |")
        emit("|---:|---:|---:|---:|---|---|---:|---:|")
        for level, m in ladder.items():
            emit(f"| {level} | {m['attempted']} | {m['succeeded']} | {m['failed']} "
                 f"| {fmt(m['ttft_p50'])} / {fmt(m['ttft_p95'])} "
                 f"| {fmt(m['latency_p50'])} / {fmt(m['latency_p95'])} "
                 f"| {fmt(m['output_throughput'], 1)} | {m['output_token_count']} |")
            missing = [f for f in required if m.get(metric_map[f]) in (None, 0) and f != "request_count"]
            if m["failed"] > 0 or m["succeeded"] < min_success:
                baseline_ok = False
                emit(f"  - ❌ c{level}: failed={m['failed']}, succeeded={m['succeeded']}")
            if missing:
                baseline_ok = False
                emit(f"  - ❌ c{level}: missing fields {missing}")
        emit()
    emit(f"**Baseline gate: gate was “all levels complete + 5 fields reported”, "
         f"measured {'all levels clean' if baseline_ok else 'violations above'} → "
         f"{'PASS' if baseline_ok else 'FAIL'}**")
    verdicts.append(baseline_ok)
    emit()

    # ---- A/A noise band ------------------------------------------------
    metrics = ["output_throughput", "ttft_p95", "latency_p95"]
    aa1 = load_ladder(run_dir / "aa" / "arm1")
    aa2 = load_ladder(run_dir / "aa" / "arm2")
    band = noise_band(aa1, aa2, metrics)
    emit("## A/A noise band (two identical arms, A/B ladder)")
    emit()
    emit("| c | Δ out tok/s (%) | Δ TTFT p95 (%) | Δ latency p95 (%) |")
    emit("|---:|---:|---:|---:|")
    for level, row in band.items():
        emit(f"| {level} | {fmt(row['output_throughput'], 2)} "
             f"| {fmt(row['ttft_p95'], 2)} | {fmt(row['latency_p95'], 2)} |")
    emit()
    max_noise = max((v for row in band.values() for v in row.values() if v is not None),
                    default=None)
    noise_cap = gates["noise"]["e2e_max_noise_pct"]
    noise_ok = max_noise is not None and max_noise <= noise_cap
    emit(f"**Promotion-capability: gate was “baseline noise ≤ {noise_cap}% e2e”, "
         f"measured max {fmt(max_noise, 2)}% → "
         f"{'PASS' if noise_ok else 'FAIL — benchmark must be stabilized'}**")
    verdicts.append(noise_ok)
    emit()

    # A/A output-mismatch baseline: identical config, greedy — any mismatch
    # here is run-to-run nondeterminism, the denominator for the A/B output
    # checks (dry-run finding: batch-composition-dependent argmax flips).
    emit("### A/A output-mismatch baseline (identical arms)")
    emit()
    aa_mismatch_total = 0
    for level in sorted(set(aa1) & set(aa2)):
        d = token_diff(aa1[level], aa2[level])
        aa_mismatch_total += d["text_mismatches"]
        emit(f"- c{level}: {d['compared']} compared, {d['text_mismatches']} text "
             f"mismatches, {d['token_count_mismatches']} token-count mismatches")
    emit()
    if aa_mismatch_total == 0:
        emit("**Output-mismatch baseline = 0 → the strict zero-mismatch gate "
             "applies to the A/B as written.**")
    else:
        emit(f"**⚠️ Output-mismatch baseline = {aa_mismatch_total} under identical "
             "config — greedy outputs are not run-to-run stable on this setup. "
             "A/B mismatches at or below this baseline cannot be attributed to "
             "the cap change; the zero-mismatch gate needs a tolerance "
             "definition (flagged in the contract).**")
    emit()

    # ---- Admission gate -------------------------------------------------
    arms_cfg = section["arms"]
    cap_a = arms_cfg["A"]["max_running_requests"]
    cap_b = arms_cfg["B"]["max_running_requests"]
    pair_dirs = sorted((run_dir / "ab").glob("pair*"))
    pairs = [(load_ladder(p / f"cap{cap_a}"), load_ladder(p / f"cap{cap_b}"))
             for p in pair_dirs]
    improve_levels = gates["admission"]["improve"]["levels"] if not dry_run else section["ladders"]["admission_ab"][-1:]
    regress_levels = gates["admission"]["regression"]["levels"] if not dry_run else section["ladders"]["admission_ab"][:1]
    max_reg = gates["admission"]["regression"]["max_regression_pct"]

    emit(f"## Admission gate (cap {cap_a} vs cap {cap_b}, K={len(pairs)} paired repeats)")
    emit()
    emit("### Improvement at high concurrency — both readings of the #1018 wording")
    emit()
    emit('"improves beyond the noise band across three paired repeats" is read two '
         "ways and both are reported; the strict per-repeat reading is the binding "
         "one unless Wenyao rules otherwise.")
    emit()
    admission_ok = True
    for level in improve_levels:
        b = band.get(level, {}).get("output_throughput")
        deltas = [rel_delta(pa[level]["output_throughput"], pb[level]["output_throughput"])
                  for pa, pb in pairs if level in pa and level in pb]
        valid = [d for d in deltas if d is not None]
        each_ok = (len(valid) == len(pairs) and b is not None
                   and all(d > b for d in valid))
        mean_delta = statistics.mean(valid) if valid else None
        agg_ok = mean_delta is not None and b is not None and mean_delta > b
        admission_ok &= each_ok
        emit(f"- c{level}: gate was “Δthroughput > noise band {fmt(b, 2)}%”, measured "
             f"{[fmt(d, 2) + '%' for d in deltas]} "
             f"→ strict per-repeat: {'PASS' if each_ok else 'FAIL'}; "
             f"aggregate (mean {fmt(mean_delta, 2)}%): {'PASS' if agg_ok else 'FAIL'}")
    emit()
    emit(f"### Regression check at low concurrency (any metric worse than {max_reg}% ⇒ FAIL; worst repeat counts)")
    emit()
    for level in regress_levels:
        for metric in gates["admission"]["regression"]["metrics"]:
            deltas = [rel_delta(pa[level][metric], pb[level][metric])
                      for pa, pb in pairs if level in pa and level in pb]
            deltas = [d for d in deltas if d is not None]
            if metric == "output_throughput":
                worst = min(deltas) if deltas else None      # decrease = regression
                bad = worst is not None and worst < -max_reg
            else:
                worst = max(deltas) if deltas else None      # increase = regression
                bad = worst is not None and worst > max_reg
            admission_ok &= not bad
            emit(f"- c{level} {metric}: gate was “|regression| ≤ {max_reg}%”, "
                 f"measured worst {fmt(worst, 2)}% → {'FAIL' if bad else 'PASS'}")
    emit()

    # ---- Output checks ---------------------------------------------------
    emit("### Output checks (greedy token equivalence, cap A vs cap B)")
    emit()
    mismatch_total = 0
    for k, (pa, pb) in enumerate(pairs, 1):
        for level in sorted(set(pa) & set(pb)):
            d = token_diff(pa[level], pb[level])
            mismatch_total += d["text_mismatches"]
            flag = "" if d["text_mismatches"] == 0 else f" ⚠️ prompts {d['mismatched_prompt_ids']}"
            emit(f"- pair{k} c{level}: {d['compared']} compared, "
                 f"{d['text_mismatches']} text mismatches, "
                 f"{d['token_count_mismatches']} token-count mismatches{flag}")
    output_ok = mismatch_total == 0
    emit()
    emit(f"**Output checks: gate was “zero mismatches”, measured {mismatch_total} → "
         f"{'PASS' if output_ok else 'FAIL — blocks promotion'}**")
    admission_ok &= output_ok
    verdicts.append(admission_ok)
    emit()
    emit(f"**Admission gate overall: {'PASS' if admission_ok else 'FAIL'}**")
    emit()

    # ---- Telemetry appendix ----------------------------------------------
    log_dir = run_dir / "server_logs"
    if log_dir.exists():
        emit("## Telemetry appendix (scraped decode logs)")
        emit()
        emit("| log | decode lines | KV usage max/p50 | #running max | #queue max | graph hit |")
        emit("|---|---:|---|---:|---:|---:|")
        for lf in sorted(log_dir.glob("*.log")):
            t = parse_decode_log(lf)
            if t["decode_lines"]:
                emit(f"| {lf.name} | {t['decode_lines']} "
                     f"| {fmt(t['token_usage_max'], 2)}/{fmt(t['token_usage_p50'], 2)} "
                     f"| {t['running_req_max']} | {t['queue_req_max']} "
                     f"| {fmt(t['cuda_graph_hit_rate'], 2)} |")
            else:
                emit(f"| {lf.name} | 0 | — | — | — | — |")
        emit()

    emit(f"---\n**Overall: {'ALL GATES PASS' if all(verdicts) else 'GATE FAILURES PRESENT'}**")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=str(Path(__file__).parent / "config_1p2.yaml"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    report = analyze(Path(a.run_dir), cfg, a.dry_run)
    out = Path(a.out) if a.out else Path(a.run_dir) / "gate_report.md"
    out.write_text(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
