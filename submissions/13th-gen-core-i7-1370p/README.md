# 13th-gen-core-i7-1370p — benchmark submission

| | |
|---|---|
| Host | 13th-gen-core-i7-1370p |
| OS | linux |
| CPU | 13th Gen Intel(R) Core(TM) i7-1370P (14C/20T) |
| GPU | — |
| Memory | 31 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | Intel(R) Iris(R) Xe Graphics (RPL-P) | 14 | 0.76 | 67.24 | ok |
| cpu:0 | 13th Gen Intel(R) Core(TM) i7-1370P | 14 | 0.26 | 29.38 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 14 | ok (11 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 14 | ok (19 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 14 | ok (9 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | 14 | ok (16 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 14 | ok (15 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 14 | ok (14 pts) | ok |
