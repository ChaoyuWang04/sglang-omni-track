#!/usr/bin/env bash
# 一键正式实验：校准(#1135 对表) → 正式矩阵(baseline/A/A/A/B) → audio sweep → 判定 → 打包
# 默认 nohup 后台执行，实时日志 logs/h100_run.log。
#   启动:   bash scripts/h100_run.sh            (后台，立即返回)
#   看进度: tail -f logs/h100_run.log
#   前台跑: bash scripts/h100_run.sh --fg
#   跑部分: bash scripts/h100_run.sh --fg calibration|matrix|audio|analyze|pack
set -uo pipefail
cd "$(dirname "$0")/.."
WS=$(pwd)
VENV="$WS/sglang-omni/.venv"
export PATH="$VENV/bin:$PATH"   # JIT kernel 编译需要 venv 里的 ninja
PORT=8300
RUN=results/i1024_1p2/h100_r1
mkdir -p logs "$RUN"

# ---------- 后台自挂 ----------
if [ "${1:-}" != "--fg" ]; then
  nohup bash "$0" --fg "${@:1}" > logs/h100_run.log 2>&1 &
  echo $! > logs/h100_run.pid
  echo "已后台启动 (pid $(cat logs/h100_run.pid))"
  echo "看进度:  tail -f logs/h100_run.log"
  echo "杀任务:  kill -- -\$(ps -o pgid= -p \$(cat logs/h100_run.pid) | tr -d ' ')  # 之后跑 scripts/h100_smoke.sh 里的 orphan 清理"
  exit 0
fi
shift  # 去掉 --fg
STAGES="${*:-calibration matrix events_probe audio analyze pack}"

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
fail() { stamp "FAIL: $1"; exit 1; }

serve_up() {  # $1=cap  $2=log名
  # 等端口真正可 bind（TIME_WAIT 排空）；STRICT_PORT 禁止 launcher 静默换端口
  for i in $(seq 1 40); do
    "$VENV/bin/python" -c "import socket;s=socket.socket();s.bind(('',$PORT))" 2>/dev/null && break
    stamp "port $PORT TIME_WAIT 排空中 ..."; sleep 3
  done
  SGLANG_OMNI_STRICT_PORT=1 \
  setsid "$VENV/bin/python" -m sglang_omni.cli serve \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only \
    --thinker-tp-size 2 --thinker-gpus 0,1 --mem-fraction-static 0.85 \
    --thinker-max-running-requests "$1" \
    --config harness/pipeline_1p2.yaml --port $PORT \
    > "$RUN/server_logs/$2.log" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 200); do
    curl -sf localhost:$PORT/health >/dev/null 2>&1 && return 0
    kill -0 $SRV_PID 2>/dev/null || fail "server 死于启动 ($2)"
    sleep 3
  done
  fail "server 10min 未 healthy ($2)"
}
serve_down() {
  [ -n "${SRV_PID:-}" ] && kill -- -"$(ps -o pgid= -p $SRV_PID | tr -d ' ')" 2>/dev/null
  for i in $(seq 1 20); do
    APPS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
    case "$APPS" in *sglang-omni/.venv*) sleep 3 ;; *) return 0 ;; esac
  done
  for pid in $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader \
               | grep "sglang-omni/.venv" | cut -d, -f1); do
    kill -9 "$pid" && stamp "killed orphan $pid"
  done
  sleep 3
}
bench() {  # $1=并发 $2=out $3=label [$4 extra args...]
  "$VENV/bin/python" harness/bench_client.py --base-url http://localhost:$PORT \
    --model qwen3-omni --prompt-file harness/prompts_200w_96.jsonl \
    --concurrency "$1" --num-requests 96 --warmup 4 --output-tokens 128 \
    --out "$2" --label "$3" "${@:4}" || fail "bench $3"
}

mkdir -p "$RUN/server_logs"

# ========== Stage: calibration（#1135 对表，seed 1234 streaming） ==========
if grep -qw calibration <<<"$STAGES"; then
  stamp "===== CALIBRATION (#1135 replication) ====="
  for CAP in 16 64; do
    serve_up $CAP "calib_cap$CAP"
    # recorder 全程关（实测开着 -25~30% 吞吐，见 anomalies.md #6）；事件证据由 events_probe 阶段单独提供
    # seedless（2026-07-27 Chaoyu 批准的契约偏离）：API seed 会触发
    # lookahead_eligible 的 sampling_seed 门（#1047），把测量踢回同步 decode 路径；
    # greedy 下 seed 不影响 token 选择，seedless 才复现 #1135 表的实际 async 条件。
    # 详见 results/.../anomalies.md #4。seed1234 同步路径数据已归档 calib_seed1234_syncpath/。
    for C in 1 16 32 48 64; do
      bench $C "$RUN/calib/cap$CAP/c$C.json" "calib-cap$CAP-c$C"
    done
    serve_down
  done
  "$VENV/bin/python" - <<'PYEOF' || fail "校准对表偏差超 ±25% 门限，停（不进矩阵，人工复核）"
import json, yaml, pathlib
cfg = yaml.safe_load(open('harness/config_1p2.yaml'))
ref = cfg['calibration']['reference']['output_tok_s']
bad = []
for i, cap in enumerate(('16','64')):
    for c in (1,16,32,48,64):
        got = json.load(open(f'results/i1024_1p2/h100_r1/calib/cap{cap}/c{c}.json'))['output_throughput_tok_s']
        want = ref[f'c{c}'][0 if cap=='16' else 1]
        drift = abs(got-want)/want*100
        print(f'cap{cap} c{c}: got {got:.0f} vs #1135 {want} ({drift:+.1f}%)')
        if drift > 25: bad.append((cap,c,drift))
assert not bad, f'>25% drift: {bad}'
print('calibration aligned with #1135 table')
PYEOF
  stamp "calibration done"
fi

# ========== Stage: matrix（正式：baseline → A/A → A/B，seedless greedy） ==========
if grep -qw matrix <<<"$STAGES"; then
  stamp "===== FORMAL MATRIX (run_matrix real) ====="
  "$VENV/bin/python" harness/run_matrix.py --profile real --run-dir "$RUN" --port $PORT \
    || fail "run_matrix"
  stamp "matrix done"
fi

# ========== Stage: events_probe（lookahead 事件取证；非性能数据） ==========
# 契约要求记录 lookahead decision/launch/resolve 事件，但 recorder 开启实测拖慢
# 吞吐 25-30%（anomalies.md #6），故性能臂全部关 recorder，事件在此独立采集：
# 每 cap 一次 c32 探测（bs>=2 稳定触发 lookahead），产物仅作行为证据，不入性能口径。
if grep -qw events_probe <<<"$STAGES"; then
  stamp "===== LOOKAHEAD EVENT PROBES (recorder ON; NOT perf data) ====="
  for CAP in 16 64; do
    serve_up $CAP "events_probe_cap$CAP"
    curl -sf -X POST localhost:$PORT/start_request_profile -H 'Content-Type: application/json' \
      -d "{\"run_id\":\"probe$CAP\",\"event_dir\":\"$WS/$RUN/events/events_probe_cap$CAP\"}" >/dev/null
    bench 32 "$RUN/events_probe/cap$CAP/c32.json" "events-probe-cap$CAP-c32"
    curl -sf -X POST localhost:$PORT/stop_request_profile -H 'Content-Type: application/json' -d '{}' >/dev/null
    serve_down
  done
  stamp "events_probe done"
fi

# ========== Stage: audio（audio->text baseline，3 fixtures） ==========
if grep -qw audio <<<"$STAGES"; then
  stamp "===== AUDIO->TEXT BASELINE ====="
  FIX="$WS/harness/fixtures"
  serve_up 64 "audio_cap64"
  # recorder 关（见 anomalies.md #6）
  for C in 1 16 32 48; do
    bench $C "$RUN/audio/c$C.json" "audio-c$C" \
      --audios "$FIX/fixture_short.wav,$FIX/fixture_medium.wav,$FIX/fixture_long.wav"
  done
  serve_down
  stamp "audio done"
fi

# ========== Stage: analyze ==========
if grep -qw analyze <<<"$STAGES"; then
  stamp "===== ANALYZE ====="
  "$VENV/bin/python" harness/analyze.py --run-dir "$RUN" || fail "analyze"
  stamp "gate report -> $RUN/gate_report.md"
fi

# ========== Stage: pack ==========
if grep -qw pack <<<"$STAGES"; then
  stamp "===== PACK ====="
  ( cd sglang-omni && git diff > "../$RUN/applied_patch.diff" )
  tar czf i1024_1p2_h100_raw.tgz "$RUN" logs/ harness/fixtures/fixtures.json
  stamp "打包 -> i1024_1p2_h100_raw.tgz ($(du -h i1024_1p2_h100_raw.tgz | cut -f1))，记得 rsync 回本地"
fi

stamp "===== ALL STAGES DONE ====="
