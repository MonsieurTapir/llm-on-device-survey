# ryzen-7-255 — benchmark submission

| | |
|---|---|
| Host | ryzen-7-255 |
| OS | linux |
| CPU | AMD Ryzen 7 255 w/ Radeon 780M Graphics (8C/16T) |
| GPU | — |
| Memory | 13.4 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | AMD Radeon Graphics (RADV PHOENIX) | 8 | 3.51 | 64.04 | ok |
| cpu:0 | AMD Ryzen 7 255 w/ Radeon 780M Graphics | 8 | 0.41 | 51.98 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 8 | ok (20 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 8 | ok (22 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 8 | ok (15 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | 8 | ok (20 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 8 | ok (22 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 8 | ok (22 pts) | ok |
