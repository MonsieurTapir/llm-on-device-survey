# ryzen-9-9950x-rtx-5080 — benchmark submission

| | |
|---|---|
| Host | monsieurtapir-workstation |
| OS | linux |
| CPU | AMD Ryzen 9 9950X 16-Core Processor (16C/32T) |
| GPU | NVIDIA GeForce RTX 5080 |
| Memory | 60.4 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (9 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | NVIDIA GeForce RTX 5080 | 16 | 108.22 | 809.95 | ok |
| vulkan:1 | AMD Ryzen 9 9950X 16-Core Processor (RADV RAPHAEL_MENDOCINO) | 16 | 0.68 | 64.83 | ok |
| cpu:0 | AMD Ryzen 9 9950X 16-Core Processor | 16 | 1.57 | 36.09 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 16 | ok (22 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 16 | ok (22 pts) | ok |
| Ministral3-3B | q4 | vulkan:1 | 16 | ok (15 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 16 | ok (20 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | 16 | ok (20 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:1 | 16 | ok (11 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 16 | ok (22 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 16 | ok (22 pts) | ok |
| gemma4-E2B | q4 | vulkan:1 | 16 | ok (20 pts) | ok |
