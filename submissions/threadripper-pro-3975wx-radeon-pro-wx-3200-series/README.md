# threadripper-pro-3975wx-radeon-pro-wx-3200-series — benchmark submission

| | |
|---|---|
| Host | threadripper-pro-3975wx-radeon-pro-wx-3200-series |
| OS | linux |
| CPU | AMD Ryzen Threadripper PRO 3975WX 32-Cores (32C/64T) |
| GPU | — |
| Memory | 61.6 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | AMD Radeon Pro WX 3200 Series (RADV POLARIS12) | 32 | 0.85 | 74.99 | ok |
| cpu:0 | AMD Ryzen Threadripper PRO 3975WX 32-Cores | 32 | 1.5 | 49.43 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 32 | ok (22 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 32 | ok (17 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 32 | ok (20 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | 32 | ok (8 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 32 | ok (22 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 32 | ok (19 pts) | ok |
