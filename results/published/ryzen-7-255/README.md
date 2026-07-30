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

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| vulkan:0 | AMD Radeon Graphics (RADV PHOENIX) | 3.5 | 63.97 | ok |
| cpu:0 | AMD Ryzen 7 255 w/ Radeon 780M Graphics | 0.41 | 48.36 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | ok (17 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | ok (21 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | ok (14 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | ok (20 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | ok (21 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | ok (21 pts) | ok |
