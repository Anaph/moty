#!/bin/sh
# VisionPsy decoder on the board: moty_cycle speed/TTFT/RSS (2 reps x 2 cycles per container)
# and per-tile greedy ids from the qwen CLI (parse with board_lead.py <log> <refs.json>).
# Needs in the cwd: containers, moty_cycle, qwen, tiles/*.f32, vpsy_ids.txt, vpsy_prompt_ids.json; ../rssrun.
# runs ON camera 139 in /mnt/user_data/moty-bench/vpsy
cd /mnt/user_data/moty-bench/vpsy
( while :; do a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    if [ "$a" -lt 150000 ]; then echo "SENTINEL $(date +%T) MemAvailable $a kB"; killall -9 moty_cycle qwen 2>/dev/null; fi
    sleep 1; done ) &
SPID=$!; trap 'kill $SPID' EXIT
busy() { ps w | grep -E "rkllm_probe|vpsy-test" | grep -v grep; }
echo "start $(date +%T) load $(cat /proc/loadavg)"
for r in ${REPS:-1 2}; do for c in ${CONTAINERS:-vpsy-q8all vpsy-q4}; do
  if busy; then echo "AGENT-B BUSY, stop"; exit 1; fi
  echo "== $c rep $r | MemAvail $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB | load $(cut -d' ' -f1 /proc/loadavg)"
  ../rssrun ./moty_cycle $c 2 tiles/t00_034140.npu.f32 4 3 64 1000 vpsy_ids.txt 49152 2>&1 | grep -E "cycle|rssrun|^The|rror|SENTINEL"
done; done
for c in ${CONTAINERS:-vpsy-q8all vpsy-q4}; do for t in tiles/*.f32; do
  if busy; then echo "AGENT-B BUSY, stop"; exit 1; fi
  echo "== ids $c $(basename $t)"
  SNAP=$c QBITS=4 THREADS=4 THREADS_DECODE=3 TEMP=0 EMBEDS=$t EMBEDS_TOKEN=49152 PROMPT_IDS=vpsy_prompt_ids.json NGEN=64 IGNORE_EOS=1 TOKENS=1 ./qwen 2>&1 >/dev/null \
    | awk 'NF>=32 { ok=1; for(i=1;i<=NF;i++) if ($i !~ /^-?[0-9]+$/) ok=0; if (ok) print "IDS " $0 }'
done; done
echo "end $(date +%T)"
