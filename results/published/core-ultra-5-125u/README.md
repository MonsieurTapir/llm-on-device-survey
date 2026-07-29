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

## ggml  (6 runs)

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| vulkan:0 | Intel(R) Graphics (MTL) | 0.47 | 56.37 | ok |
| cpu:0 | Intel(R) Core(TM) Ultra 5 125U | 0.26 | 52.83 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | ok (8 pts) | ok |
| Ministral3-3B | q4 | vulkan:0 | ok (13 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | ok (7 pts) | ok |
| Qwen3.5-4B | q4 | vulkan:0 | ok (12 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | ok (12 pts) | ok |
| gemma4-E2B | q4 | vulkan:0 | ok (8 pts) | ok |
