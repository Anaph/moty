#!/bin/sh
# run_matrix.sh — benchmark matrix on the target (runs ON the device).
#
#   ENGINE=./lfm2 SNAP=LFM2.5-350M MODEL=LFM QBITS_LIST="8 4" THREADS_LIST="1 2 4" \
#   REPS=3 PROMPT_FILE=prompt.txt [NGEN=64] [EXTRA_ENV="EMBED=disk"] \
#   [LLAMA_BENCH=./llama-bench GGUF=model-Q4_0.gguf] [MEM_MIN_KB=150000] \
#   [WAIT_WHILE="pattern"] ./run_matrix.sh > run.log 2>&1
#   python3 parse_table.py run.log
#
# Every run goes through rssrun (peak RSS, wall time, oom_score_adj=1000).
# Repetitions are interleaved (rep-major) so drifting background load hits
# all configurations alike. MEM_MIN_KB: a sentinel SIGKILLs the benchmark if
# MemAvailable drops below it (a device thrashing into its watchdog is worse
# than a lost run). WAIT_WHILE: before each run, wait while a process whose
# command line matches the pattern is running (shared devices).
: "${ENGINE:?}" "${SNAP:?}" "${PROMPT_FILE:?}"
MODEL=${MODEL:-$(basename "$SNAP")}
P="$(cat "$PROMPT_FILE")"
(
  while :; do
    a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$a" -lt "${MEM_MIN_KB:-150000}" ]; then
      echo "SENTINEL $(date +%T): MemAvailable ${a} kB -> kill benchmark"
      pkill -9 -f "$(basename "$ENGINE")|llama-bench"
    fi
    sleep 1
  done
) &
SPID=$!; trap 'kill $SPID' EXIT
waitfree() {
  [ -n "${WAIT_WHILE:-}" ] || return 0
  while ps w | grep -E "$WAIT_WHILE" | grep -v grep >/dev/null; do echo "(waiting: $WAIT_WHILE)"; sleep 20; done
}
echo "start $(date) loadavg $(cat /proc/loadavg) | $(grep MemAvailable /proc/meminfo)"
for rep in $(seq 1 "${REPS:-3}"); do
  for q in ${QBITS_LIST:-4}; do for t in ${THREADS_LIST:-1 2 4}; do
    waitfree
    echo "== moty $MODEL QBITS=$q ${EXTRA_ENV:-} THREADS=$t rep=$rep | MemAvailable $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
    env ${EXTRA_ENV:-} SNAP="$SNAP" QBITS=$q THREADS=$t CTX=${CTX:-512} NGEN=${NGEN:-64} TEMP=0 CHAT_TEMPLATE=0 \
        IGNORE_EOS=1 PROMPT="$P" ./rssrun "$ENGINE" 2>&1 \
      | grep -E "load [0-9.]+s|prefill [0-9]+ tok|rssrun|OOM|rror" | sed -E 's/.*(load [0-9.]+s).*/\1/'
  done; done
  if [ -n "${LLAMA_BENCH:-}" ]; then
    np=$(printf '%s' "$P" | wc -w)   # llama-bench takes a token count, not text
    for t in ${THREADS_LIST:-1 2 4}; do
      waitfree
      echo "== llama.cpp $MODEL Q4_0 THREADS=$t rep=$rep"
      ./rssrun "$LLAMA_BENCH" -m "$GGUF" -p "${LLAMA_P:-$np}" -n "${NGEN:-64}" -t "$t" -r 1 -o csv 2>&1 | grep -E '^"|rssrun' | tail -3
    done
  fi
done
echo "done $(date) loadavg $(cat /proc/loadavg)"
