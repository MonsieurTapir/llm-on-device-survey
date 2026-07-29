# ryzen-9-9950x-rtx-5080 — benchmark submission

| | |
|---|---|
| Host | ryzen-9-9950x-rtx-5080 |
| OS | linux |
| CPU | AMD Ryzen 9 9950X 16-Core Processor (16C/32T) |
| GPU | NVIDIA GeForce RTX 5080 |
| Memory | 60.4 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## ggml  (9 runs)

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| vulkan:0 | NVIDIA GeForce RTX 5080 | 132.05 | 816.49 | ok |
| vulkan:1 | AMD Ryzen 9 9950X 16-Core Processor (RADV RAPHAEL_MENDOCINO) | 0.49 | 64.73 | ok |
| cpu:0 | AMD Ryzen 9 9950X 16-Core Processor | 1.6 | 36.51 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | ok (19 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | ok (19 pts) | ok |
| Ministral3-3B | q4 | vulkan:1 | ok (11 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | ok (18 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | ok (18 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:1 | ok (9 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | ok (19 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | ok (19 pts) | ok |
| gemma4-E2B | q4 | vulkan:1 | ok (15 pts) | ok |
