#!/bin/sh
# vl_pipeline.sh <encoder.rknn> <image.ppm> <lm container> <prompt_ids.json> [ngen=64]
# NPU encoder (rknn_encode) -> moty lfm2 with the rows injected; reports the
# encoder time, time to first token (encoder + LM load-less prefill), decode
# speed, peak RSS and MemAvailable. Env: RKNN_LIB, THREADS (4), THREADS_DECODE (3).
enc=$1; img=$2; lm=$3; ids=$4; ngen=${5:-64}
tmp=${TMPDIR:-/tmp}/vl_pipeline.$$; mkdir -p $tmp
ms() { awk '{printf "%d", $1 * 1000}' /proc/uptime; }   # busybox date has no %N
t0=$(ms)
./rknn_encode "$enc" "$img" $tmp/img.f32 2> $tmp/enc.txt || { cat $tmp/enc.txt; exit 1; }
t1=$(ms)
SNAP=$lm QBITS=4 THREADS=${THREADS:-4} THREADS_DECODE=${THREADS_DECODE:-3} TEMP=0 NGEN=$ngen \
  EMBEDS=$tmp/img.f32 PROMPT_IDS=$ids ./rssrun ./lfm2 > $tmp/answer.txt 2> $tmp/lm.txt
t2=$(ms)
grep rknn_encode $tmp/enc.txt
grep -E "load [0-9.]+s|prefill [0-9]+ tok|peak_rss" $tmp/lm.txt | sed 's/^.*\(load [0-9.]*s\).*$/\1/'
echo "wall: encoder step $((t1 - t0)) ms, LM step $((t2 - t1)) ms (10 ms resolution)"
echo "ANSWER: $(tr '\n' ' ' < $tmp/answer.txt)"
rm -rf $tmp
