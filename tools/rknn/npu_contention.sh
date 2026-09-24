#!/bin/sh
# npu_contention.sh — does a VL encoder on the NPU slow the board's own
# detector (and vice versa)? Runs ON the device.
#
#   RKNN_LIB=<librknnrt.so> DET_MODEL=<detector .rknn> DET_BYTES=<its input size in bytes> \
#   VL_MODEL=<encoder .rknn> VL_IMAGE=<tile.ppm> [DET_FRAMES=60] [VL_RUNS=3] ./npu_contention.sh
#
# The detector's input is random bytes of DET_BYTES (only timing matters;
# e.g. 960x544x3 = 1566720). Prints the encode timings of each model alone
# and of both at once, and the NPU load samples while both run.
: "${RKNN_LIB:?}" "${DET_MODEL:?}" "${DET_BYTES:?}" "${VL_MODEL:?}" "${VL_IMAGE:?}"
export RKNN_LIB
tmp=${TMPDIR:-/tmp}/npu_contention.$$; mkdir -p "$tmp"
head -c "$DET_BYTES" /dev/urandom > "$tmp/det.rgb"
load() { for f in /proc/rknpu/load /sys/kernel/debug/rknpu/load; do [ -r "$f" ] && { cat "$f"; return; }; done; echo "(no NPU load file)"; }
echo "== detector alone (${DET_FRAMES:-60} frames)"
./rknn_encode "$DET_MODEL" "$tmp/det.rgb" "$tmp/det.f32" "${DET_FRAMES:-60}" 2>&1 | grep -o "encode best.*runs)"
echo "== VL encoder alone (${VL_RUNS:-3} runs)"
./rknn_encode "$VL_MODEL" "$VL_IMAGE" "$tmp/vl.f32" "${VL_RUNS:-3}" 2>&1 | grep -o "encode best.*runs)"
echo "== both at once"
( ./rknn_encode "$VL_MODEL" "$VL_IMAGE" "$tmp/vl.f32" "${VL_RUNS:-3}" 2>&1 | grep -o "encode best.*runs)" | sed 's/^/VL: /' ) &
sleep 2
( for i in 1 2 3 4 5 6 7 8; do load; sleep 0.5; done ) &
./rknn_encode "$DET_MODEL" "$tmp/det.rgb" "$tmp/det.f32" "${DET_FRAMES:-60}" 2>&1 | grep -o "encode best.*runs)" | sed 's/^/detector: /'
wait
rm -rf "$tmp"
