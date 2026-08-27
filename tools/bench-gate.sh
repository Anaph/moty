#!/bin/sh
# bench-gate.sh — la soglia di sicurezza della modularizzazione (M0).
# Correttezza SEMPRE; performance solo su macchina ferma (loadavg < 1.5).
# Uso:  tools/bench-gate.sh [lfm2_baseline_tok_s] [qwen_baseline_tok_s]
# Exit 0 = verde; 1 = fallito (stampa il perché).
set -u
CDIR=$(cd "$(dirname "$0")/../c" && pwd)
MODELS=${MOTY_MODELS:-/workspace/models}
LFM2_GGUF="$MODELS/LFM2.5-8B-A1B-Q4_K_M.gguf"
QWEN_GGUF="$MODELS/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
LFM2_BASE=${1:-57.0}
QWEN_BASE=${2:-20.5}
TOL=0.97          # -3%
FAILED=0

say() { printf '[gate] %s\n' "$*"; }
fail() { say "FAIL: $*"; FAILED=1; }

[ -f "$LFM2_GGUF" ] || { say "SKIP perf: $LFM2_GGUF assente"; PERF=0; }
[ -f "$QWEN_GGUF" ] || { say "SKIP perf: $QWEN_GGUF assente"; PERF=0; }
PERF=${PERF:-1}

# ---- 1. correttezza: suite completa portable + native ----
say "make check (portable + 118 test)..."
( cd "$CDIR" && make check >/dev/null 2>&1 ) || fail "make check"
say "make test-native..."
( cd "$CDIR" && make test-native >/dev/null 2>&1 ) || fail "make test-native"

# ---- 2. build motori + greedy smokes ----
say "rebuild engines..."
( cd "$CDIR" && make clean >/dev/null 2>&1 && make all >/dev/null 2>&1 ) || fail "build motori"

if [ "$PERF" = 1 ]; then
  say "LFM2 greedy smoke..."
  out=$(cd "$CDIR" && GGUF="$LFM2_GGUF" PROMPT="Say hello" NGEN=32 TEMP=0 CHAT_TEMPLATE=1 \
        MOTY_NO_OMP_TUNE=1 timeout 90 ./moty 2>/dev/null | head -c 4000)
  echo "$out" | grep -qi "hello" || fail "LFM2 greedy senza 'hello'"

  say "Qwen greedy smoke..."
  out=$(cd "$CDIR" && GGUF="$QWEN_GGUF" PROMPT="The capital of France is" NGEN=8 TEMP=0 \
        CHAT_TEMPLATE=0 MOTY_NO_OMP_TUNE=1 timeout 150 ./moty 2>/dev/null | head -c 40)
  echo "$out" | grep -q " Paris" || fail "Qwen greedy senza ' Paris'"

  # ---- 3. performance: solo a macchina ferma, mediana di 3 ----
  LOAD=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0)
  if awk "BEGIN{exit !($LOAD < 1.5)}"; then
    for M in LFM2 QWEN; do
      eval GGUF="\$${M}_GGUF"; eval BASE="\$${M}_BASE"
      case $M in LFM2) P='epic poem about sea storms';; *) P='essay about navigation history';; esac
      say "$M: 3 run da 384 token..."
      best=0
      for i in 1 2 3; do
        line=$(cd "$CDIR" && GGUF="$GGUF" PROMPT="Write a long detailed $P" NGEN=384 TEMP=0.7 \
               CHAT_TEMPLATE=1 MOTY_NO_OMP_TUNE=1 timeout 300 ./moty 2>&1 >/dev/null | grep decode | tail -1)
        tps=$(echo "$line" | sed -n 's/.*decode [0-9]* tok in [0-9.]*s (\([0-9.]*\) tok\/s).*/\1/p')
        [ -n "$tps" ] || { fail "$M: nessuna riga decode ($line)"; tps=0; }
        awk "BEGIN{exit !($tps > $best)}" && best=$tps
      done
      say "$M best=$best tok/s (baseline $BASE, min $(awk "BEGIN{printf \"%.1f\", $BASE*$TOL}"))"
      awk "BEGIN{exit !($best >= $BASE*$TOL)}" || fail "$M $best < $(awk "BEGIN{printf \"%.1f\", $BASE*$TOL}")"
    done
  else
    say "SKIP perf: loadavg=$LOAD >= 1.5 (macchina condivisa)"
  fi
fi

[ "$FAILED" = 0 ] && say "VERDE" || say "ROSSO"
exit $FAILED
