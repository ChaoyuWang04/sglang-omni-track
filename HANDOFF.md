# HANDOFF — sglang-omni #1018 §1.2 测量执行（交接文档）

> 写于 2026-07-27 21:3x (+08:00)，由本地（5090 工作站）会话交接给 RunPod pod 上的新会话。
> 本仓库是精简 deploy 快照；完整笔记/证据/社区上下文在 Chaoyu 本地私有仓库，不在此处。
> 涉及社区人际与发布策略的问题：**问 Chaoyu，不要推断**。

## 1. 任务是什么

sgl-project/sglang-omni 的 Qwen3-Omni serving 性能 RFC（issue **#1018**）第 **1.2** 节
"Thinker admission cap and baseline qualification" 还有两个未勾的 gate checkbox，
对应的测量没人做。维护者（GitHub: edwingao28）已通过私下渠道向 Chaoyu 绿灯放行，
验收方式 = 做完把结果给他看。产出全部先落草稿，Chaoyu 亲自发布。

要跑的东西（依据 #1018 §1.2 逐字，机器可读版 = `harness/config_1p2.yaml`）：

1. **校准**：复现 PR #1135 的表（96 prompts / 200 words in / 128 out / **seed 1234** /
   streaming / mem-fraction 0.85 / c{1,16,32,48,64} × cap{16,64}）。数字对齐（±25% 内、
   趋势一致）才进正式矩阵；对不上就停，报环境差异。参照数字已写死在 config 的
   `calibration.reference`。
2. **Canonical baseline 重建**：c{1,8,16,32}，cap64（当前 main 默认），两个 identical arms，
   seedless greedy，冷缓存，每档 ≥32 成功请求。
3. **A/A 噪声带**：cap16 两个 identical arms，c{1,16,32,48}——先跑，作分母。
4. **Admission A/B**：cap16 vs cap64，c{1,16,32,48}，K=3 配对重复，遥测全开。
5. **audio→text baseline**：三个 LibriSpeech fixture（仓库自带 `harness/fixtures/`），
   c{1,16,32,48}。

Gate 算术已内置 `harness/analyze.py`（改进档逐次超噪声带 + 聚合双读法；c1/c16 回归
≤5%；A/A output-mismatch 基线先行；#1025 文风 "gate was X, measured Y → verdict"）。

## 2a. 当前状态（更新于 2026-07-28 ~00:40 +08:00：✅ 全部测量完成，已打包待回传）

**测量全部完成**：校准 10/10 过门（最差 -12.5%）→ baseline/A-A/A-B/audio 全 96/96 →
audit_run.py **0 FAIL / 0 WARN** → gate_report.md 已出。核心结论：admission 提升
c32 +37.6~45.8% / c48 +36.5~42.8%（噪声带 ≤0.34%，K=3 逐次+聚合双 PASS）；三个字面
FAIL 如实保留交维护者裁量（c16 尾延迟百分比在 ~10ms 绝对量级、c48 三 prompt 贪心
发散 9/1152）。产物 `i1024_1p2_h100_raw.tgz`（20M，sha256 cf75a4c1...af8f），内含
`draft_1018_comment.md` 草稿。**待办：Chaoyu rsync 回本地 → 审草稿 → 发布 #1018。**
运行全程 11 条异常台账见 `results/i1024_1p2/h100_r1/anomalies.md`。

---

## 2b. 历史状态（测量设计定稿过程，已完结）

校准两度未过门并完成完整归因（细节见 `results/i1024_1p2/h100_r1/anomalies.md` #4–#6）：
1. **seed 1234 会触发 `lookahead_eligible` 的 `sampling_seed` 门（#1047）强制同步 decode**
   （慢 30-40%）——校准改 seedless（Chaoyu 批准；greedy 下 seed 不影响输出）。
2. **event recorder 开启实测 -25~30% 吞吐**（emit 在 launch/resolve 关键路径上，吃掉
   async lookahead 的 host 重叠裕量）——逐 commit 二分（dc64a6b/3f97d73/d690b55/1793a22
   全部 ~2000-2750 tok/s @c32/48）证明**无代码回归**，纯 recorder 开销。
   测量设计定稿：性能臂全部 recorder 关；新增 `events_probe` 阶段单独取证 lookahead 事件。
   两条都是可报给上游的发现（#1135 表用 API seed 复现不了 + 遥测实现不可在测性能时开启）。

当前：`h100_run.sh`（calibration → matrix → events_probe → audio → analyze → pack）
在最终配置下运行中。诊断数据在 `results/i1024_1p2/h100_r1/calib_diag_*/`，
弃用的两轮校准归档在 `calib_seed1234_syncpath/` 与 `calib_seedless_recorderon/`。

## 2b. 旧状态（换机过程，已解决）

**✅ 第三次换机成功。**本 pod 两卡 UUID（GPU-39cb8abe... / GPU-f027c525...）均非坏卡
（GPU-e54a5561...），Recovery Action 两卡 None，逐卡真建 context 体检通过。宿主 uptime
仍是 68 天、PCI 位号 87/90——疑似同一宿主机但换到了另外两张好卡，不影响使用。
环境全部从持久卷 `/workspace` 继承（venv、66G 权重、补丁），setup 幂等复检通过。

**冒烟曾在 `/start_request_profile` 422 上失败，根因是 harness 传输层 bug：**
smoke/run 脚本的 `curl -d` 未带 `Content-Type: application/json`，FastAPI Pydantic body
解析拒收（form-urlencoded）。已修复 6 处 curl 调用（smoke×2、run.sh×4）。另发现
`run_matrix.py` 的 real profile 从未接事件遥测（dry-run 的 plain sglang 无此端点所以
一直没暴露），已补：baseline/AA/AB 每个 arm 统一开 event recorder（与校准/audio 条件
一致，产物落 `$RUN/events/<arm名>/`）。两处修改均为 harness 插桩/传输层，不动测量契约。

修复后冒烟全过：server healthy / merged config 三 override 落位 / 8 请求 bench 通过 /
lookahead 三事件真机落盘 / Decode batch 遥测源存在 / 退场 GPU 清空。
正式实验 `scripts/h100_run.sh` 已后台启动，进度见 `logs/h100_run.log`。

---

以下为换机前的定位记录（仍有效）：

**冒烟失败已定位根因：第一台 pod 的 GPU1 宿主机层故障，Chaoyu 决定换机。**
完整证据链见 `logs/gpu1_diagnosis.md`（换机前应已 rsync 回本地）。结论摘要：

- GPU1 被驱动标记 `GPU Recovery Action : Reset`，任何进程（含脱离框架的最小 torch
  进程、ctypes 直调 libcuda 的 `cuCtxCreate`→错误 999）都无法在其上创建 CUDA context。
  GPU0 完全正常。非 sglang-omni 栈内问题，NCCL 从未跑到初始化。
- 此前观察到的"挂死在 `Init torch distributed begin`"是次生现象：tp1（GPU1）进程
  启动即死，tp0 在分布式初始化处永远等不到 rank 1，10 分钟后被超时杀掉。
- 教训：setup 自检的"点名"（device_count、compute mode）不足以发现此类故障，
  必须每卡真正建 context 分配显存才算数。

**新机器上的第一步（在跑任何 setup/smoke 之前）——逐卡 CUDA context 体检：**

```bash
for i in 0 1; do CUDA_VISIBLE_DEVICES=$i python3 -c "
import torch; torch.cuda.set_device(0)
x=torch.zeros(1024,device='cuda'); print('gpu$i ok:', x.sum().item())"; done
nvidia-smi -q | grep -i 'recovery action'   # 两卡都必须是 None
```

（若新机 venv 未就绪，用系统 python3 + 任何带 torch 的环境均可；ctypes 方案见
`logs/gpu1_diagnosis.md`。）两卡体检通过后按顺序：`h100_setup.sh` → `h100_smoke.sh`。

- 换机后环境需全部重建（venv、权重 ~60GB、lookahead 补丁、fixtures），
  `scripts/h100_setup.sh` 一条龙负责；预算已在第一台 pod 上消耗数小时，注意 $60 上限。
- 旧机已验证过的事实（应仍适用，但新机需复核）：H100 上 sglang 默认 attention
  backend = **fa3**（与 5090 的 flashinfer 不同，属预期，记进 manifest 即可）；
  启动时 HF 匿名限速警告无害（权重下载完成后走本地 cache）。
- 判读框架（留作后续排障参考）：错误在 NCCL/CUDA 层（peer access/P2P）→ 宿主机层
  问题，可用 `NCCL_P2P_DISABLE=1` 作诊断（只许诊断，禁用于正式测量——会改变性能
  语义）；Python 层（import/断言/OOM）→ 栈内问题，本地可修。

## 3. 仓库地图

```
CLAUDE.md            ← 你已读的铁律
HANDOFF.md           ← 本文件
harness/
  config_1p2.yaml    ← 测量契约唯一机器可读源（gate/阶梯/采样/缓存全在此）
  pipeline_1p2.yaml  ← sgl-omni serve 的完整 pipeline config（decode_log_interval=8 +
                        disable_radix_cache=true 已内置；serve 用 --config 指向它）
  bench_client.py    ← 压测 client（恒定在途并发/streaming TTFT/--audios/失败单列）
  run_matrix.py      ← 正式矩阵编排（baseline→A/A→A/B，server 起停+GPU 验证内置）
  analyze.py         ← gate 判定 + markdown 报告（跑完矩阵后执行）
  manifest.py        ← run_manifest.json 生成（环境自动探测）
  prompts_200w_96.jsonl  ← 固定 prompt set（sha256 见 manifest）
  fixtures/          ← 三个 audio fixture + fixtures.json（utterance ID + sha256）
patches/
  lookahead_telemetry.diff  ← 遥测埋点（setup 已 apply；README 有验证记录）
scripts/
  h100_setup.sh / h100_smoke.sh / h100_run.sh   ← 三个入口
```

## 4. 操作手册

```bash
# 冒烟（当前卡在这，先修它）
bash scripts/h100_smoke.sh          # 产物: logs/smoke.log, logs/smoke_events/

# 冒烟过了之后的正式实验（自动 nohup 后台，SSH 断连无影响）
bash scripts/h100_run.sh            # tail -f logs/h100_run.log
bash scripts/h100_run.sh --fg matrix   # 单独重跑某阶段: calibration/matrix/audio/analyze/pack

# server 手动起（调试用；参数即 canonical 契约，勿改）
sglang-omni/.venv/bin/python -m sglang_omni.cli serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --text-only \
  --thinker-tp-size 2 --thinker-gpus 0,1 --mem-fraction-static 0.85 \
  --thinker-max-running-requests 64 --config harness/pipeline_1p2.yaml \
  --port 8300 --log-level debug

# 事件遥测开关（低开销事件模式）
curl -X POST localhost:8300/start_request_profile -d '{"run_id":"x","event_dir":"/abs/dir"}'
curl -X POST localhost:8300/stop_request_profile -d '{}'
# lookahead 三事件: thinker_lookahead_decision/launch/resolve（补丁提供，bs>=2 才触发）

# GPU 清理验证（每次 server 退出后）
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
```

遥测取数一览：KV pressure / #running-req / #queue-req / cuda graph 命中 = server 日志的
`Decode batch` 行（每 8 decode 步一行，analyze.py 会刮）；TTFT = client 侧逐请求；
lookahead 事件 = event recorder JSONL。**没有 /metrics 端点，stage 日志是唯一来源，
所以 server 日志文件必须全程落盘保留。**

## 5. 完成判据与移交

全部跑完后：`results/i1024_1p2/h100_r1/` 下应有 calib/ baseline/ aa/ ab/ audio/ events/
server_logs/ manifest.json gate_report.md applied_patch.diff，run.sh 的 pack 阶段会打成
`i1024_1p2_h100_raw.tgz`。提醒用户 rsync 回本地后，**用户负责**：审报告草稿、发布到
#1018、联系维护者验收。你的工作在 tgz 落地 + 向用户汇报 gate 结论时结束。

若中途机器要回收/预算触线：优先保 `results/` 和 `logs/` 的 rsync，实验可以下次续
（run_matrix 的产物按 arm 分文件，已完成的 arm 数据独立有效）。
