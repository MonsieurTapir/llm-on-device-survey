# apple-m5-pro — benchmark submission

| | |
|---|---|
| Host | apple-m5-pro |
| OS | macos |
| CPU | Apple M5 Pro (18C/18T) |
| GPU | Apple M5 Pro |
| Memory | 48 GB |
| Sampling | job: 1 spawns; sweep/probe: adaptive per point |

Status legend: `ok` (measured) · `too_slow` (backstop killed / below the floor) · `errored` (crash/OOM, no sample) · `skipped` · `unhealthy` (brain-check failed).

## ggml  (6 runs)

| provider | device | gemm TFLOP/s | d2d GB/s | probe |
|---|---|---|---|---|
| mtl:0 | Apple M5 Pro | 7.27 | 140.68 | ok |
| cpu:0 | Apple M5 Pro | 1.12 | 182.04 | ok |

| model | quant | provider | sweep | job |
|---|---|---|---|---|
| Ministral3-3B | q4 | cpu:0 | ok (14 pts) | ok |
| Ministral3-3B | q4 | mtl:0 | ok (19 pts) | ok |
| Qwen3.5-4B | q4 | cpu:0 | ok (14 pts) | ok |
| Qwen3.5-4B | q4 | mtl:0 | ok (18 pts) | ok |
| gemma4-E2B | q4 | cpu:0 | ok (19 pts) | ok |
| gemma4-E2B | q4 | mtl:0 | ok (19 pts) | ok |
