# threadripper-pro-7975wx-radeon-pro-w7600 — benchmark submission

| | |
|---|---|
| Host | threadripper-pro-7975wx-radeon-pro-w7600 |
| OS | linux |
| CPU | AMD Ryzen Threadripper PRO 7975WX 32-Cores (32C/64T) |
| GPU | — |
| Memory | 122.7 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | AMD Radeon Pro W7600 (RADV NAVI33) | 32 | 7.27 | 186.92 | ok |
| cpu:0 | AMD Ryzen Threadripper PRO 7975WX 32-Cores | 32 | 1.71 | 174.11 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 32 | ok (22 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 32 | ok (22 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 32 | ok (20 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | 32 | ok (20 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 32 | ok (22 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 32 | ok (22 pts) | ok |
