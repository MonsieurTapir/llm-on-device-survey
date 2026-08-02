# ryzen-7-9800x3d-rtx-3080 — benchmark submission

| | |
|---|---|
| Host | ryzen-7-9800x3d-rtx-3080 |
| OS | windows |
| CPU | AMD Ryzen 7 9800X3D 8-Core Processor (8C/16T) |
| GPU | NVIDIA GeForce RTX 3080 |
| Memory | 61.7 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (9 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | NVIDIA GeForce RTX 3080 | 8 | 76.19 | 660.28 | ok |
| vulkan:1 | AMD Radeon(TM) Graphics | 8 | 0.42 | 62.84 | ok |
| cpu:0 | AMD Ryzen 7 9800X3D 8-Core Processor            | 8 | 2.35 | 51.56 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 8 | ok (22 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 8 | ok (22 pts) | ok |
| Ministral3-3B | q4 | vulkan:1 | 8 | ok (11 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 8 | ok (20 pts) | errored |
| Qwen3.5-4B | q4 | vulkan:0 | 8 | ok (20 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:1 | 8 | ok (9 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 8 | ok (22 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 8 | ok (22 pts) | ok |
| gemma4-E2B | q4 | vulkan:1 | 8 | ok (12 pts) | ok |
