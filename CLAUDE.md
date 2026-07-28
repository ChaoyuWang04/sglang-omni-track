# CLAUDE.md — RunPod H100 测量执行会话

你在一台租用的 2×H100 RunPod 上，任务是执行 sglang-omni Qwen3-Omni 的 #1018 §1.2 剩余测量。
**开始任何工作前，先完整阅读 `HANDOFF.md`**（项目状态、当前卡点、测量契约、操作规程），
**宣布任何阶段"完成"前，先过 `ACCEPTANCE.md`**（运行监控、机器自查 `harness/audit_run.py`、
人工判读清单、完成汇报模板）。"脚本跑完没报错"≠"做完"。

## 铁律（违反会造成真实损失）

1. **GitHub 只读**：本会话绝不发 issue/PR/comment。一切对外产物写成本地待发草稿，由用户（Chaoyu）审核后亲自发布。
2. **计费意识**：这台机器按小时烧钱（~$6/h）。预算硬上限 $60。排查问题优先级：先看日志再重试；连续两次同因失败就停下来向用户汇报，不要无限重试烧钱。
3. **进程清理**：任何 server 起停后必须验证 GPU 清空（`nvidia-smi --query-compute-apps`）；sgl-omni stage worker 会孤儿存活占卡。kill 前必须核实进程身份（路径含 `sglang-omni/.venv` 才可杀）。
4. **数据即时回传**：每个阶段产物完成后提醒用户 rsync 回本地。云机随时可能被回收。
5. **测量纪律**：性能数字只在 `harness/config_1p2.yaml` 声明的契约下有效。不要临时改参数"看看效果"——任何偏离契约的改动都会使数据作废。校准（seed 1234）与正式矩阵（seedless greedy）的采样设置不同，是有意的，勿"统一"。
6. **溯源**：写进报告的每个数字必须能指回 `results/` 下的原始 JSON 或 server 日志文件。

## 快速上下文

- 规格唯一依据：GitHub issue sgl-project/sglang-omni#1018 的 §1.2 节 + Benchmark Contract（`harness/config_1p2.yaml` 是它的机器可读版，代码零硬编码）
- 三个入口脚本：`scripts/h100_setup.sh`（环境+自检）、`scripts/h100_smoke.sh`（冒烟）、`scripts/h100_run.sh`（正式实验，自动 nohup 后台，`tail -f logs/h100_run.log`）
- 当前卡点与调试状态：见 `HANDOFF.md` §"当前状态"
