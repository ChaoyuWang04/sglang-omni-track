#!/usr/bin/env bash
# H100 一键环境安装 + 自检（RunPod EU 节点，直连无镜像）
# 用法: 在 workspace 仓库根目录执行  bash scripts/h100_setup.sh
# 全程幂等：重复执行只补缺失步骤。每步输出 [SETUP] OK/FAIL 标记。
set -uo pipefail
cd "$(dirname "$0")/.."
WS=$(pwd)
LOG_DIR="$WS/logs"; mkdir -p "$LOG_DIR"
fail() { echo "[SETUP] FAIL: $1" >&2; exit 1; }
ok()   { echo "[SETUP] OK: $1"; }

# ---- 0. 硬件门槛（不满足立即退出，别浪费租金） -------------------------
command -v nvidia-smi >/dev/null || fail "nvidia-smi 不存在"
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c H100 || true)
[ "$GPUS" -ge 2 ] || fail "需要 2x H100，当前: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
CUDA_MAJ=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
[ "${CUDA_MAJ:-0}" -ge 13 ] || fail "宿主机驱动 CUDA=$CUDA_MAJ < 13，torch cu130 wheel 跑不了——换 CUDA 13.x 宿主机"
# 先落盘再 grep 文件——pipefail 下 `nvidia-smi | grep -q` 会因 SIGPIPE 误判
nvidia-smi topo -m > "$LOG_DIR/topo.txt"
grep -qE 'NV[0-9]+' "$LOG_DIR/topo.txt" || fail "GPU0-GPU1 无 NVLink（topo -m 无 NV*，见 logs/topo.txt）"
# sgl-omni 每卡跑多个进程（thinker rank + encoder stages 共卡），Exclusive 模式必死
CMODE=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | sort -u | tr '\n' ';')
case "$CMODE" in
  Default*) : ;;
  *) nvidia-smi -c DEFAULT 2>/dev/null && ok "compute mode 已改回 DEFAULT" \
       || fail "GPU compute mode=$CMODE（需 Default：多进程共卡）。容器内改不动就换宿主机" ;;
esac
ok "2x H100 + CUDA $CUDA_MAJ + NVLink + compute mode Default"

# 逐卡真建 CUDA context——点名（device_count/compute mode）查不出坏卡：
# 判例 2026-07-27，某 pod 的 GPU1 被驱动标记 Recovery Action=Reset，任何进程都无法在
# 其上建 context，症状是 TP2 起服"挂死在 Init torch distributed"（tp1 秒死、tp0 空等
# 10 分钟超时）——白烧数小时租金才定位。
nvidia-smi -q > "$LOG_DIR/gpu_full.txt" 2>&1
if grep -i 'recovery action' "$LOG_DIR/gpu_full.txt" | grep -qvi 'none'; then
  fail "某卡 GPU Recovery Action != None（坏卡，换机器）：$(grep -i 'recovery action' "$LOG_DIR/gpu_full.txt" | tr '\n' ';')"
fi

# ---- 1. 基础工具 -------------------------------------------------------
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git curl) || fail "git 安装"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv 安装"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
ok "git + uv"

# ---- 2. sglang-omni 主仓库（clone 进 workspace，布局与本地一致） --------
if [ ! -d sglang-omni/.git ]; then
  git clone --depth 50 https://github.com/sgl-project/sglang-omni.git || fail "clone sglang-omni"
fi
cd sglang-omni
git rev-parse HEAD > "$LOG_DIR/framework_commit.txt"
ok "sglang-omni @ $(git rev-parse --short HEAD)"

# ---- 3. venv + 依赖（pin 栈镜像 CI） ------------------------------------
[ -d .venv ] || uv venv --python 3.12 || fail "uv venv"
uv pip install -e . || fail "pip install -e ."
uv pip install ninja aiohttp pyyaml requests hf_transfer soundfile numpy pytest || fail "辅助依赖"
.venv/bin/python -c "import torch; assert torch.cuda.device_count()==2, torch.cuda.device_count(); print(torch.__version__, torch.version.cuda)" \
  > "$LOG_DIR/torch_check.txt" 2>&1 || fail "torch 自检: $(cat "$LOG_DIR/torch_check.txt")"
ok "venv: $(cat "$LOG_DIR/torch_check.txt" | head -1)"

# 逐卡真建 context + 分配显存（承接第 0 步的 Recovery Action 检查；坏卡在这里必现）
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python -c \
    "import torch; torch.cuda.set_device(0); x=torch.zeros(1<<20, device='cuda'); \
     torch.cuda.synchronize(); print(x.numel())" > "$LOG_DIR/gpu${i}_ctx.txt" 2>&1 \
    || fail "GPU$i 无法建立 CUDA context（坏卡，换机器）：$(tail -3 "$LOG_DIR/gpu${i}_ctx.txt")"
done
ok "两卡 CUDA context 体检通过"

# ---- 4. 权重（后台下载，~60GB） -----------------------------------------
export HF_HUB_ENABLE_HF_TRANSFER=1
if ! .venv/bin/hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --include 'config.json' >/dev/null 2>&1; then
  fail "HF 连通性"
fi
nohup .venv/bin/hf download Qwen/Qwen3-Omni-30B-A3B-Instruct > "$LOG_DIR/weights_download.log" 2>&1 &
WEIGHTS_PID=$!
ok "权重下载后台启动 (pid $WEIGHTS_PID, tail -f logs/weights_download.log)"

# ---- 5. 遥测补丁 + 单测 --------------------------------------------------
if ! git diff --quiet 2>/dev/null || git apply --check ../patches/lookahead_telemetry.diff 2>/dev/null; then
  git apply ../patches/lookahead_telemetry.diff 2>/dev/null && ok "补丁 applied" || {
    grep -q thinker_lookahead_decision sglang_omni/scheduling/omni_scheduler.py \
      && ok "补丁已在（幂等跳过）" || fail "补丁 apply"; }
fi
.venv/bin/python -m pytest -q tests/unit_test/pipeline/test_async_decode.py \
  tests/unit_test/qwen3_omni/test_thinker_lookahead_eligible.py \
  tests/unit_test/qwen3_omni/test_thinker_async_decode_flag.py \
  > "$LOG_DIR/unit_tests.txt" 2>&1 && ok "async 单测: $(tail -1 "$LOG_DIR/unit_tests.txt" | head -c60)" \
  || fail "async 单测（本地基线 45 passed）: $(tail -3 "$LOG_DIR/unit_tests.txt")"
cd "$WS"

# ---- 6. audio fixtures（repo 里已有则跳过；否则 EU 直连下载构建） --------
if [ -f harness/fixtures/fixtures.json ]; then
  ok "fixtures 已随仓库携带"
else
  mkdir -p data/librispeech
  [ -f data/librispeech/test-clean.tar.gz ] || \
    curl -L --retry 3 -o data/librispeech/test-clean.tar.gz \
      https://www.openslr.org/resources/12/test-clean.tar.gz || fail "LibriSpeech 下载"
  sglang-omni/.venv/bin/python harness/build_audio_fixtures.py \
    --tar data/librispeech/test-clean.tar.gz --out harness/fixtures || fail "fixture 构建"
fi
sglang-omni/.venv/bin/python harness/validate_fixtures.py --fixtures harness/fixtures \
  > "$LOG_DIR/fixture_validation.txt" 2>&1 && ok "fixtures 过服务端加载路径" \
  || fail "fixture 验证: $(cat "$LOG_DIR/fixture_validation.txt")"

# ---- 7. 等权重 → 冒烟（/health + 1 请求 + lookahead 事件真机首验） ------
wait $WEIGHTS_PID || fail "权重下载"
ok "权重就绪"
bash scripts/h100_smoke.sh || fail "冒烟（详见 logs/smoke.log）"

echo
echo "[SETUP] ===== 全部通过，可执行 bash scripts/h100_run.sh ====="
