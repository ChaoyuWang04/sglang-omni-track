# ACCEPTANCE — 结果验收自查（跑完 ≠ 做完；过完本文件才算完）

> 面向 pod 侧执行会话。设计动机：远端只有你一人经手数据，任何"跑完了但不全/不对"
> 的问题若在下机后才发现，代价是重新租机重跑。所以：**pack 之前，本文件的每一节
> 必须过一遍**。机器能查的都在 `harness/audit_run.py` 里；人要判的列在 §3。

## 0. 心智模型：什么样算"成功交付"

不是"脚本跑完没报错"，而是同时满足：
1. **完整**：五个板块（calib/baseline/aa/ab/audio）的每个 arm × 每档 × 每次重复都有数据文件；
2. **健康**：每个文件里 96/96 请求成功、字段齐全；每个 server 日志无 stage 死亡、
   graph capture 与 cap 一致、decode 遥测行存在；事件文件三类 lookahead 事件齐备；
3. **可信**：校准对上 #1135 表、A/A 噪声带 ≤5%、异常都有归因注记；
4. **可复核**：manifest 完整 + 补丁 diff 存档 + 全部原始日志在 tgz 里——
   本地侧要能不问你任何问题就重建每个数字的来源。

## 1. 运行中健康监控（不要跑完才看）

正式跑起来后，每 20–30 分钟扫一眼：

```bash
tail -20 logs/h100_run.log                 # 阶段推进正常？FAIL 立即停
ls results/i1024_1p2/h100_r1/**/c*.json | wc -l    # 产物数在涨
nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader
                                            # 温度 <85C；throttle reasons 应为 0x...0000
df -h /workspace | tail -1                 # 磁盘余量 >20GB
grep -c 'Dead stage\|scheduler crashed' results/i1024_1p2/h100_r1/server_logs/*.log 2>/dev/null | grep -v ':0' || echo clean
```

**红旗即停**（停 = 让 run.sh 跑完当前 arm 后 kill，保留现场，向用户汇报）：
- 同一 arm 连续两次起服失败；任何 `Dead stage process`；
- 某档 wall 时间比同 arm 相邻档异常放大 5×+（可能僵尸负载/降频）；
- GPU throttle reasons 非零（热/功率降频会污染吞吐数据——记录时间段，该段数据标疑）。

## 2. 机器自查（必跑，FAIL 清零才继续）

```bash
sglang-omni/.venv/bin/python harness/audit_run.py --run-dir results/i1024_1p2/h100_r1
```

它检查：54 个 level JSON 的存在性与字段健康（attempted=96/failed=0/ttft 非空/
completion_tokens 非空——最后这项缺失说明 include_usage 没生效，吞吐全废）、
15 个 server 日志的 capture-bs-与-cap 一致性（**cap 覆盖没生效 = 整个 A/B 白跑**，
这是最贵的失败模式）与 stage 死亡扫描、事件文件三类 lookahead 事件、manifest 完整性
（含 framework_dirty=true 佐证补丁在树）。FAIL 有补跑指引。

## 3. 人工判读项（脚本查不了的）

按顺序过，每项在最终汇报里写一行结论：

- [ ] **校准对表**：`calib` 各档吞吐与 config `calibration.reference` 的方向与量级一致？
      c32/c48 上 cap64 显著优于 cap16 的形态是否重现（参照 +8~12% 吞吐、TTFT p95 从秒级降到 <0.5s）？
      漂移 >25% → 停，报环境差异，勿跑结论。
- [ ] **A/A 噪声带**：`analyze.py` 报告里 max noise ≤5%？超了 = benchmark 不具备
      promotion 资格（#1018 原文规则），只能报"需稳定化"，不能出 gate 结论。
- [ ] **A/A output-mismatch 基线**：=0 则严格零容忍 gate 适用；>0 则 A/B 的 mismatch
      要对照基线解读（报告已自动生成该段，确认它有数据而不是空表）。
- [ ] **提前 EOS 检查**：audit 若 WARN 大量请求输出远短于 128 tokens，吞吐口径与
      #1135（固定 128 out）不可比——在报告注明比例。
- [ ] **audio 档异常**：长 fixture（191s）在高并发下若有失败，失败模式记录了吗
      （HTTP 码 + server 日志对应行）？失败不可静默丢弃。
- [ ] **时序完整性**：`ab/pairN` 内 cap16/cap64 是相邻时间跑的（配对语义）？
      中途有没有插入过手动调试请求污染某个 arm？有就标记该 arm。
- [ ] **异常台账**：跑全程遇到的每个意外（重试、超时、throttle、磁盘告警）都记进
      `results/i1024_1p2/h100_r1/anomalies.md` 了？没有异常也写"无"。

## 4. 判定与打包

```bash
sglang-omni/.venv/bin/python harness/analyze.py --run-dir results/i1024_1p2/h100_r1
# 读 gate_report.md：每个 gate 的 "gate was X, measured Y → verdict" 都言之有据？
bash scripts/h100_run.sh --fg pack
sha256sum i1024_1p2_h100_raw.tgz          # 记进最终汇报
```

## 5. 完成汇报模板（发给用户的最后一条消息，照此结构）

```
## 1.2 测量完成汇报
- 审计: audit_run.py 0 FAIL / N WARN（WARN 逐条: …）
- 校准: 对表结论一句话 + 最大漂移%
- Baseline gate: PASS/FAIL + 依据
- A/A 噪声带: max X%（≤5% ✓/✗）; output-mismatch 基线: N
- Admission gate: 逐次读法 PASS/FAIL, 聚合读法 PASS/FAIL; c1/c16 回归最差值
- Output checks: N mismatches (vs A/A 基线 M)
- audio→text: 各档成功率 + 异常
- 异常台账: K 条（摘要）
- 产物: i1024_1p2_h100_raw.tgz (X GB, sha256 …) —— 请立即 rsync 回本地
- GPU 时消耗: ~X h（预算余量 $Y）
```

## 6. 升级求助判据（出现即停手汇报，别自行发挥）

- 校准漂移 >25% / A/A 噪声 >5% / 任何 gate 的判定依据你自己不确信
- 同因失败重试 2 次未解
- 预算消耗过半而矩阵未过半
- 想改 harness/config 里任何契约参数（阶梯/采样/缓存/输出长度/mem-fraction）——
  一律先问，这些定义来自上游 RFC，改了数据即作废
