# core-ultra-5-125u — benchmark submission

| | |
|---|---|
| Host | core-ultra-5-125u |
| OS | linux |
| CPU | Intel(R) Core(TM) Ultra 5 125U (12C/14T) |
| GPU | — |
| Memory | 15.1 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## llamacpp  (6 runs)

| provider | device | threads | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|---|
| vulkan:0 | Intel(R) Graphics (MTL) | 12 | 0.47 | 56.86 | ok |
| cpu:0 | Intel(R) Core(TM) Ultra 5 125U | 12 | 0.26 | 52.15 | ok |

| model | quant | provider | threads | sweep | job |
|---|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | 12 | ok (9 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | 12 | ok (15 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | 12 | ok (7 pts) | too_slow |
| Qwen3.5-4B | q4 | vulkan:0 | 12 | ok (14 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | 12 | ok (12 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | 12 | ok (9 pts) | ok |
