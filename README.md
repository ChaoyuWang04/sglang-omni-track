# sglang-omni measurement harness (deploy snapshot)

Measurement tooling for sglang-omni Qwen3-Omni perf work (#1018 item 1.2).
Deploy snapshot — generated from a private workspace; no history.

## Usage (2x H100, CUDA 13.x host)

```bash
bash scripts/h100_setup.sh   # env + weights + patch + fixtures + smoke (idempotent)
bash scripts/h100_run.sh     # background run; tail -f logs/h100_run.log
```

Stages: `bash scripts/h100_run.sh --fg calibration|matrix|audio|analyze|pack`
