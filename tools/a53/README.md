# Measurement tooling for small ARMv8.0 boards (Cortex-A53 class)

Used to pick and tune the Q4R4 kernels and to produce the numbers in
[docs/performance.md](../../docs/performance.md). Nothing here knows about a
particular device: copy the binaries and models yourself (or with
`xfer_verified.sh`) and run on the target.

| file | runs on | what |
|---|---|---|
| `../../c/tests/bench_a53.c` | target | read bandwidth with/without `PRFM` (roofline input), int8 `SMLAL` MAC peak, GEMV kernel candidates on weights much larger than the caches, the real Q4R4 driver (decode and prefill), hot GEMM variants. Every candidate is checked against a scalar reference first. Build line in the file header. |
| `rssrun.c` | target | `rssrun CMD ARGS...`: peak RSS (`ru_maxrss`) and wall time of CMD; the child gets `oom_score_adj=1000`. `aarch64-linux-gnu-gcc -O2 -static rssrun.c -o rssrun` |
| `run_matrix.sh` | target | engine x QBITS x THREADS matrix (+ optional `llama-bench` rows), interleaved repetitions, memory sentinel, optional wait-while-busy guard; see the header |
| `parse_table.py` | host | `run_matrix.sh` log -> markdown table (median over repetitions) |
| `run_configs.sh` | target | labelled configurations (any engine + environment, `EMBEDS=`/`PROMPT_IDS=` runs, `llama-bench` rows), interleaved repetitions, memory sentinel, wait-while-busy guard; see the header |
| `parse_runs.py` | host | `run_configs.sh` log(s) -> markdown table, medians with every run listed |
| `xfer_verified.sh` | host | chunked, rate-limited, sha256-verified copy to the target; stops at the first connection failure |

Typical session (int4 container of LFM2.5-350M, 1/2/4 threads, three
repetitions):

    # host
    SNAP=LFM2.5-350M QBITS=4 SAVE_PACKED=LFM2.5-350M-q4r4 ./lfm2    # pack once
    TARGET=user@board ./xfer_verified.sh LFM2.5-350M-q4r4/model.safetensors /data/moty/LFM2.5-350M-q4r4
    # target
    ENGINE=./lfm2 SNAP=LFM2.5-350M-q4r4 MODEL=LFM2.5 QBITS_LIST=4 REPS=3 \
      PROMPT_FILE=prompt.txt ./run_matrix.sh > lfm.log 2>&1
    # host
    python3 parse_table.py lfm.log

Measurement hygiene that mattered on a shared board: interleave the
configurations you compare (background load drifts by tens of percent),
warm the cpufreq governor before timing (`bench_a53` does), and keep
several hundred MB of MemAvailable headroom — a model sized to the last
50 MB makes the whole system thrash.
