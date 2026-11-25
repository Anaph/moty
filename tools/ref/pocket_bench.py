#!/usr/bin/env python3
"""Run paired Pocket peak-RSS and throughput measurements on macOS."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import tempfile
from time import monotonic

METRICS = ("pp", "tps", "ttft_ms", "request_ms", "rss_mib")
PROFILES = {
    "text": (12, 512, "0.1065", "64"),
    "256": (12, 512, "0.108", "256"),
    "1536": (4, 2048, "0.145", "384"),
    "3584": (12, 4096, "0.202", "384"),
    "3584-speed": (12, 4096, "0.207", None),
}
INPUTS = Path(__file__).with_name("pocket_bench_inputs.json")
MEASURE_ENV = {
    "SNAP", "GGUF", "PROMPT", "PROMPT_IDS", "MEM_GB", "MEM_FRAC", "PREFILL_CHUNK",
    "KV_BITS", "MMAP", "HEAD_TOPK", "PPL", "PPL_OUT", "REF", "REF_LOGITS",
    "THREADS_DECODE", "IGNORE_EOS", "MOTY_POOL_SPIN_US",
}


def profile_settings(workload):
    _, _, budget, packed_batch = PROFILES[workload]
    settings = dict(MEM_GB=budget, MOTY_Q41_FILE_WINDOWS="1", MOTY_Q41_SPLIT_DOWN="1",
                    MOTY_Q41_CACHE_HASH="parallel", MOTY_Q41_PANEL_PREFILL="1",
                    MOTY_Q41_PANEL_DUAL="1", MOTY_Q41_SHORT_PREFILL="1",
                    MOTY_Q41_PREFILL_PAR="1", MOTY_POOL_SPIN_US="100",
                    MOTY_Q41_PACKED="stream")
    if packed_batch:
        settings.update(MOTY_Q41_PACKED="stream-all", MOTY_Q41_WINDOWS="4",
                        MOTY_Q41_RESIDENT_PREFIX="0", MOTY_Q41_RESIDENT_DOWN="slot",
                        MOTY_Q41_READ_AHEAD="3", MOTY_Q41_INDEXED_HEAD="1",
                        MOTY_Q41_STREAM_PREFILL=packed_batch, MOTY_Q41_PP_ATTN="4")
    if workload in ("text", "1536", "3584"):
        settings["MOTY_Q41_PREFILL_ATTN"] = "panel"
    if workload.startswith("3584") or workload == "1536":
        settings["MOTY_Q41_AUTO_CHUNK"] = "512"
    return settings


def run(binary, env):
    start = monotonic()
    with subprocess.Popen(["/usr/bin/time", "-l", binary], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        first_byte = proc.stdout.read(1)
        ttft = monotonic() - start
        _, err = proc.communicate()
    elapsed = monotonic() - start
    log = err.decode(errors="replace")
    if proc.returncode or not first_byte:
        raise RuntimeError(log)
    pp = re.search(r"prefill (\d+) tok in .*? \(([\d.]+) tok/s\)", log)
    tps = re.search(r"decode \d+ tok in .*? \(([\d.]+) tok/s\)", log)
    rss = re.search(r"(\d+)\s+maximum resident set size", log)
    tokens = re.search(r"^(?:\d+ +)+$", log, re.MULTILINE)
    if not all((pp, tps, rss, tokens)):
        raise RuntimeError(f"Missing benchmark data\n{log}")
    return (tokens[0].split(), int(pp[1]), float(pp[2]), float(tps[1]),
            ttft * 1000, elapsed * 1000, int(rss[1]) / 2**20)


def summarize(rows):
    ids, prompt, pp, tps, ttft, elapsed, rss = zip(*rows)
    return dict(zip(METRICS, (sum(prompt) / sum(n / rate for n, rate in zip(prompt, pp)),
                              sum(map(len, ids)) / sum(len(t) / rate for t, rate in zip(ids, tps)),
                              sum(ttft) / len(rows), sum(elapsed) / len(rows), max(rss))))


def compare(workload, binaries, env, pairs, fixtures, fixture_dir):
    threads, context, _, _ = PROFILES[workload]
    common = dict(env, THREADS=str(threads), CTX=str(context))
    if workload == "text":
        cases = [dict(PROMPT=prompt, NGEN="64") for prompt in fixtures["text"]]
    else:
        n = workload.split("-")[0]
        ids = fixture_dir / f"{n}.json"
        ids.write_text(json.dumps({"ids": fixtures[n]}))
        cases = [dict(PROMPT_IDS=str(ids), NGEN="128", IGNORE_EOS="1")]
    options = ({}, profile_settings(workload))
    results = [[], []]
    for repeat in range(-1, pairs):
        runs = [[], []]
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for case in cases:
            for i in order:
                runs[i].append(run(binaries[i], dict(common, **case, **options[i])))
            if runs[0][-1][0] != runs[1][-1][0]:
                raise RuntimeError(f"{workload} generated different token IDs in pair {repeat}")
            if workload != "text" and (len(runs[0][-1][0]), runs[0][-1][1]) != (128, int(n)):
                raise RuntimeError(f"{workload} returned an unexpected token count")
        if repeat >= 0:
            for i in order:
                results[i].append(summarize(runs[i]))
    baseline, candidate = results
    return dict(generated_tokens=sum(len(row[0]) for row in runs[0]),
                baseline=baseline, candidate=candidate,
                peak_mib=[max(row["rss_mib"] for row in side) for side in results],
                paired_change_pct={key: round(100 * median(b[key] / a[key] - 1
                                                          for a, b in zip(baseline, candidate)), 2)
                                   for key in METRICS[:-1]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", type=Path,
                        help="Saved native Q4_1 executable, defaults to the candidate in resident mode")
    parser.add_argument("--workload", choices=PROFILES, action="append")
    parser.add_argument("--pairs", type=int, default=7)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    bins = [(args.baseline or args.candidate).resolve(), args.candidate.resolve()]
    model = args.model.resolve()

    def sha256(path):
        with path.open("rb") as file:
            return hashlib.file_digest(file, "sha256").hexdigest()

    data = {"sha256": dict(model=sha256(model), baseline=sha256(bins[0]), candidate=sha256(bins[1]))}
    env = {k: v for k, v in os.environ.items() if k not in MEASURE_ENV
           and not k.startswith(("MOTY_Q41_", "MOTY_TEST_"))}
    env.update(GGUF=str(model), QBITS="-1", Q41_ACT="f16", TEMP="0", TOKENS="1")
    fixtures = json.loads(INPUTS.read_text())
    with tempfile.TemporaryDirectory(prefix="moty-pocket-") as temp:
        for workload in args.workload or list(PROFILES):
            data[workload] = compare(workload, bins, env, args.pairs, fixtures, Path(temp))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
