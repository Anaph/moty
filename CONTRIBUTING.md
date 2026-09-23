# Contributing

Keep changes focused and preserve Moty's dependency-free default CPU path.

## Branches

- **`main`** is the stable branch. It's what users clone, and it stays known-good.
- **`dev`** is the integration branch. **Open your PR against `dev`.** Reviewed PRs
  land there first; once a batch is tested and stable, the maintainer fast-forwards
  it into `main`. This keeps `main` clean instead of taking every PR one at a time.

Every PR — on either branch — is reviewed for a clean build (0 warnings), the
engine validation (REF-mode token match where applicable), and its own targeted
validation before merge.

## Local checks

Run the lightweight checks locally:

```sh
make check
```

`make -C c check` remains available for scripts that already run from the
engine directory.

This performs one portable CPU build and the test suite. It does not download
a model. The test suite needs cmake ≥ 3.24 and a C++ compiler (GoogleTest glue
only — all test logic is C); the first configure fetches a pinned gtest over
the network unless a system GTest is installed.

Benchmark reports should include the commit, exact commands, hardware and
storage details, warm-up policy, run count, and median throughput.

## Kernels and quantization changes

- Every new or changed `hw/` kernel keeps a scalar `*_ref` reference and a
  test against it. Kernels that only have a fast path on aarch64 (Q4R4,
  `hw/hw_ops.h`) are compared on x86 through the references only: also run
  the standalone runners (`-DQ4R4_TEST_MAIN`, `-DOPS_TEST_MAIN`, build
  line in `c/tests/q4r4_tests.c`) on an ARM board or under
  `qemu-aarch64 -cpu cortex-a53`, as the CI `arm-simd` job does.
- Changes that affect numerics of a quantized path come with quality
  numbers, not only speed: perplexity and top-1 agreement against HF
  transformers f32 on `tools/ref/ppl_text_long.txt` (`PPL=`, `PPL_OUT=`,
  see `tools/ref/README.md`). f32 paths must keep REF-mode token matches
  and max |Δlogit| around 1e-5 (`REF_LOGITS=`).
- On shared or loaded boards (anything also running its own workload),
  interleave the configurations being compared and report medians of
  interleaved repetitions; back-to-back runs drift by tens of percent.
  `tools/a53/` has the helpers.
