#!/bin/sh
# run_configs.sh — interleaved runs of labelled configurations (runs ON the device).
#
#   REPS=3 PROMPT_FILE=prompt.txt [NGEN=64] [MEM_MIN_KB=150000] [WAIT_WHILE="pattern"] \
#   ./run_configs.sh "label;engine;ENV=v ENV2=v" ["label;llama-bench;<llama-bench args>"] ... > run.log 2>&1
#   python3 parse_runs.py run.log
#
# Each configuration: a label, an engine binary (./lfm2, ./qwen, or
# llama-bench) and its environment (moty: SNAP=, QBITS=, THREADS=,
# THREADS_DECODE=, EMBEDS=, PROMPT_IDS=, ...; llama-bench: its arguments,
# run with -p 66 -n NGEN). moty runs get the prompt of PROMPT_FILE unless
# PROMPT_IDS= is in their environment, and CTX=512 TEMP=0 CHAT_TEMPLATE=0
# IGNORE_EOS=1 NGEN=$NGEN. Repetitions are interleaved (rep-major). Every run
# goes through rssrun (peak RSS, wall). MEM_MIN_KB: a sentinel SIGKILLs the
# engines (by process name, busybox killall) when MemAvailable drops below
# it. WAIT_WHILE: wait while a process matching the pattern runs (shared device).
: "${PROMPT_FILE:?}"
P="$(cat "$PROMPT_FILE")"
NAMES=""
for spec in "$@"; do b=${spec#*;}; b=${b%%;*}; NAMES="$NAMES $(basename "$b")"; done
(
  while :; do
    a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$a" -lt "${MEM_MIN_KB:-150000}" ]; then
      echo "SENTINEL $(date +%T): MemAvailable ${a} kB -> kill$NAMES"
      killall -9 $NAMES 2>/dev/null
    fi
    sleep 1
  done
) &
SPID=$!; trap 'kill $SPID' EXIT
waitfree() {
  [ -n "${WAIT_WHILE:-}" ] || return 0
  while ps w | grep -E "$WAIT_WHILE" | grep -v grep >/dev/null; do echo "(waiting: $WAIT_WHILE)"; sleep 20; done
}
echo "start $(date +%T) loadavg $(cat /proc/loadavg)"
r=1
while [ $r -le "${REPS:-3}" ]; do
  for spec in "$@"; do
    lab=${spec%%;*}; rest=${spec#*;}; bin=${rest%%;*}; envs=${rest#*;}
    waitfree
    echo "== $lab rep=$r | MemAvail $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB | load $(cut -d' ' -f1 /proc/loadavg)"
    case $(basename "$bin") in
    llama-bench)
      ./rssrun "$bin" $envs -p "${LLAMA_P:-66}" -n "${NGEN:-64}" -r 1 -o md 2>&1 | grep -E "pp[0-9]+|tg[0-9]+|rssrun|rror" ;;
    *)
      env $envs CTX=${CTX:-512} TEMP=0 CHAT_TEMPLATE=0 IGNORE_EOS=1 NGEN=${NGEN:-64} PROMPT="$P" ./rssrun "$bin" 2>&1 \
        | grep -E "prefill [0-9]+ tok|rssrun|prof-op|OOM|rror|SENTINEL| load [0-9.]+s" ;;
    esac
  done
  r=$((r + 1))
done
echo "end $(date +%T)"
