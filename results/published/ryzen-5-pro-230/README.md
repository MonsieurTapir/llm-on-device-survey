# ryzen-5-pro-230 — benchmark submission

| | |
|---|---|
| Host | ryzen-5-pro-230 |
| OS | windows |
| CPU | AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics (6C/12T) |
| GPU | — |
| Memory | 15.2 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | AMD Radeon 760M Graphics | 6 | 1.59 | 34.38 | ok |
| cpu:0 | AMD Ryzen 5 PRO 230 w/ Radeon 760M Graphics     | 6 | 0.63 | 31.56 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 6 | ok (14 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 6 | ok (22 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 6 | ok (11 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | 6 | ok (20 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 6 | ok (21 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 6 | ok (22 pts) | ok |
