#!/bin/bash
# xfer_verified.sh LOCALFILE REMOTEDIR
#
# Copies LOCALFILE to $TARGET:REMOTEDIR in 128 MB chunks, rate-limited
# (RATE_KBIT, default 24000 = ~3 MB/s), checking the sha256 of every chunk
# on both sides and of the joined file. Stops (exit 2) at the first
# connection failure instead of retrying: small embedded boards can drop
# off the network under a long unthrottled copy, and a blind retry loop
# only makes that worse. The executable bit of LOCALFILE is preserved.
#
#   TARGET=user@host [SSH_OPTS="..."] ./xfer_verified.sh model.safetensors /data/models/x
set -u
f=$1; rdir=$2; name=$(basename "$f")
: "${TARGET:?set TARGET=user@host}"
RATE=${RATE_KBIT:-24000}
SSH="ssh ${SSH_OPTS:-}"; SCP="scp -O ${SSH_OPTS:-}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
full=$(sha256sum "$f" | cut -d' ' -f1)
split -b 128M -d -a 3 "$f" "$tmp/p."
parts="$rdir/.parts_$name"
timeout 60 $SSH "$TARGET" "mkdir -p '$parts' && rm -f '$parts'/*" || { echo "CONN-FAIL mkdir"; exit 2; }
for p in "$tmp"/p.*; do
  pn=$(basename "$p"); want=$(sha256sum "$p" | cut -d' ' -f1)
  timeout 300 $SCP -l "$RATE" "$p" "$TARGET:$parts/$pn" || { echo "CONN-FAIL scp $pn"; exit 2; }
  got=$(timeout 60 $SSH "$TARGET" "sha256sum '$parts/$pn'" | cut -d' ' -f1) || { echo "CONN-FAIL sha $pn"; exit 2; }
  [ "$got" = "$want" ] || { echo "SHA-MISMATCH $pn"; exit 3; }
done
mode=""; [ -x "$f" ] && mode="&& chmod +x '$rdir/$name'"
got=$(timeout 300 $SSH "$TARGET" "cat '$parts'/p.* > '$rdir/$name' $mode && rm -rf '$parts' && sha256sum '$rdir/$name'" | cut -d' ' -f1) \
  || { echo "CONN-FAIL join"; exit 2; }
[ "$got" = "$full" ] && echo "OK $name $full" || { echo "FULL-SHA-MISMATCH $name"; exit 3; }
