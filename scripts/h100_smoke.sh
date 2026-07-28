#!/usr/bin/env bash
# 冒烟：serve 起服 → merged-config 复核 → 1 条文本请求 → lookahead 事件真机首验 → 干净退场
# 被 h100_setup.sh 调用；也可单独执行。产物: logs/smoke.log, logs/smoke_events/
set -uo pipefail
cd "$(dirname "$0")/.."
WS=$(pwd)
VENV="$WS/sglang-omni/.venv"
export PATH="$VENV/bin:$PATH"
LOG="$WS/logs/smoke.log"
PORT=8300
fail() { echo "[SMOKE] FAIL: $1" >&2; cleanup; exit 1; }
ok()   { echo "[SMOKE] OK: $1"; }

cleanup() {
  [ -n "${SRV_PID:-}" ] && kill -- -"$(ps -o pgid= -p $SRV_PID | tr -d ' ')" 2>/dev/null
  sleep 5
  for pid in $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader \
               | grep "sglang-omni/.venv" | cut -d, -f1); do
    kill -9 "$pid" 2>/dev/null && echo "[SMOKE] killed orphan $pid"
  done
}

setsid "$VENV/bin/python" -m sglang_omni.cli serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only \
  --thinker-tp-size 2 --thinker-gpus 0,1 --mem-fraction-static 0.85 \
  --thinker-max-running-requests 64 \
  --config harness/pipeline_1p2.yaml --port $PORT --log-level debug \
  > "$LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 200); do
  curl -sf localhost:$PORT/health >/dev/null 2>&1 && break
  kill -0 $SRV_PID 2>/dev/null || fail "server 进程退出（tail logs/smoke.log）"
  sleep 3
done
curl -sf localhost:$PORT/health >/dev/null || fail "10min 内未 healthy"
ok "server healthy"

grep -q "max_running_requests: 64" "$LOG" || fail "merged config 缺 cap 64"
grep -q "decode_log_interval: 8" "$LOG" || fail "merged config 缺 decode_log_interval"
grep -q "disable_radix_cache: true" "$LOG" || fail "merged config 缺 disable_radix_cache"
ok "merged config 三 override 落位"

# 事件采集开 → 并发 4 请求（lookahead 需 bs>=2）→ 关
mkdir -p logs/smoke_events
curl -sf -X POST localhost:$PORT/start_request_profile -H 'Content-Type: application/json' \
  -d "{\"run_id\":\"smoke\",\"event_dir\":\"$WS/logs/smoke_events\"}" >/dev/null || fail "start_request_profile"
"$VENV/bin/python" harness/bench_client.py --base-url http://localhost:$PORT \
  --model qwen3-omni --prompt-file harness/prompts_200w_96.jsonl \
  --concurrency 4 --num-requests 8 --warmup 0 --output-tokens 32 \
  --out logs/smoke_bench.json --label smoke || fail "bench_client 请求"
curl -sf -X POST localhost:$PORT/stop_request_profile -H 'Content-Type: application/json' -d '{}' >/dev/null

grep -hl thinker_lookahead_decision logs/smoke_events/events_*.jsonl >/dev/null 2>&1 \
  || fail "事件文件无 thinker_lookahead_decision（检查补丁/async 开关）"
for ev in thinker_lookahead_launch thinker_lookahead_resolve; do
  grep -hq "$ev" logs/smoke_events/events_*.jsonl || fail "缺事件 $ev"
done
ok "lookahead 三事件真机落盘"

grep -q "Decode batch" "$LOG" && ok "decode 日志行存在（KV/graph 遥测源）" \
  || echo "[SMOKE] WARN: 未见 Decode batch 行（请求太短属可能，跑矩阵时复核）"

cleanup
APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -n "$APPS" ] && fail "退场后 GPU 未清空: $APPS" || ok "GPU 清空"
echo "[SMOKE] ===== 冒烟全过 ====="
