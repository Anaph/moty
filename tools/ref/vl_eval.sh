#!/bin/sh
# vl_eval.sh — the LFM2.5-VL int4 layout study on image inputs
# (docs/models/lfm2.5-vl-450m.md, docs/performance.md §5.14), HF fake-quant.
#
#   SNAP=<LFM2.5-VL-450M snapshot> PROMPT_IDS=<prompt_ids.json> TILES=<dir>[,<dir>...] \
#   CALIB_TILES=<dir> [CALIB_TEXT=calib_text.txt] [OUT=vl_eval] [PY=python3] ./vl_eval.sh
#
# TILES: the evaluation tiles (NPU encoder rows, raw f32 [256][hidden], *.f32);
# CALIB_TILES: other tiles for VL-aware GPTQ (hf_vl_rows.py output), disjoint
# from TILES. Writes OUT/<run>.log and .json (per tile answers) and the
# invented-object counts (vl_halluc.py). Hours on a desktop CPU for 33 tiles.
set -e
: "${SNAP:?}" "${PROMPT_IDS:?}" "${TILES:?}" "${CALIB_TILES:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}; OUT=${OUT:-vl_eval}; CALIB_TEXT=${CALIB_TEXT:-$HERE/calib_text.txt}
mkdir -p "$OUT"
L815='layers\.8\.|layers\.9\.|layers\.1[0-5]\.'
Q="$PY $HERE/hf_vlq.py $SNAP $PROMPT_IDS $TILES --greedy"
# round-to-nearest layouts (no calibration), the f32-weight ceiling and int8 everywhere
$Q --out "$OUT/rtn.json" 'f32@a8' 'sym4g32@a8' 'sym4g32+Q=head@a8' "sym4g32+Q=$L815@a8" \
   'sym4g32+Q=head|layers\.1[0-5]\.@a8' 'sym8g32@a8' > "$OUT/rtn.log" 2>&1
# the text recipe (GPTQ calibrated on text + Q8R4 layers 0-1, docs/performance.md §5.8)
$Q --calib-text "$CALIB_TEXT" --out "$OUT/gptq_text.json" \
   'sym4g32:gptq+Q=layers\.0\.|layers\.1\.@a8' > "$OUT/gptq_text.log" 2>&1
# VL-aware GPTQ: text + the text positions of image sequences
$Q --calib-text "$CALIB_TEXT" --calib-tiles "$CALIB_TILES" --calib-img none --out "$OUT/gptq_vl.json" \
   'sym4g32:gptq@a8' 'sym4g32:gptq+Q=head@a8' 'sym4g32:gptq+Q=head|w2@a8' "sym4g32:gptq+Q=$L815@a8" \
   'sym4g32:gptq+Q=head|layers\.1[0-5]\.@a8' > "$OUT/gptq_vl.log" 2>&1
grep -h "MB/tok" "$OUT"/rtn.log "$OUT"/gptq_text.log "$OUT"/gptq_vl.log
$PY "$HERE/vl_halluc.py" "$OUT"/rtn.json "$OUT"/gptq_text.json "$OUT"/gptq_vl.json
