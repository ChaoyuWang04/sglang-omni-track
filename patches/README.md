# lookahead_telemetry.diff

基线: upstream main dc64a6b2 (#1135)。telemetry-only，行为零改动。
三个事件: thinker_lookahead_decision {use_lookahead,bs,min_bs} / thinker_lookahead_launch {bs} / thinker_lookahead_resolve {bs,query_hit_total,query_miss_total}(累计计数器，逐步差分)。
全部以 recorder.is_active() 门控（未开 profiler 时热循环零额外开销）+ 防御式字段访问。

本地验证 (2026-07-27): 45/45 async 单测过（含 test_async_decode / thinker_lookahead_eligible / thinker_async_decode_flag）；黑盒 recorder round-trip 三事件 schema 正确（stage 绑定解析为 thinker）；black --check 干净; git apply --check 在 bench 分支干净。

用法: cd sglang-omni && git apply ../patches/lookahead_telemetry.diff（manifest 会记录 framework_dirty=true + 此补丁）。可独立成小 PR：[Qwen3-Omni Perf] Add lookahead decision/launch/resolve events, Part of #1022。
