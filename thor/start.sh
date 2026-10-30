#!/bin/bash
# Launch llama-server for DeepSeek-V4.1-Flash-1017-REAP192 (GGUF L3) on one Jetson AGX Thor.
#
#   thor/start.sh                 text + vision, 16K context, port 8080
#   DSPARK=1 thor/start.sh        add the DSpark drafter (speculative decoding)
#   VISION=0 PORT=9000 thor/start.sh
#
# Settings (environment):
#   MODEL_DIR   directory holding the 3 L3 shards (+ mmproj, dspark sidecar)   default ~/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF
#   ENGRAM_DIR  directory holding engram_L1.v41rows / engram_L14.v41rows (NVMe) default $MODEL_DIR/engram_rows
#   CTX         context length (the fork clamps to 16384 until the long-context mask is validated)  default 16384
#   VISION      1 = load the mmproj if present                                default 1 (0 with DSPARK=1)
#   DSPARK      1 = speculative decoding with the DSpark sidecar               default 0
#   DRAFT_N     tokens drafted per step with DSPARK=1                          default 3
#   HOST PORT   listen address                                                 default 127.0.0.1 8080
#   EXTRA       further llama-server arguments, appended verbatim
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BIN:-$here/build-thor/bin}
MODEL_DIR=${MODEL_DIR:-$HOME/models/DeepSeek-V4.1-Flash-1017-REAP192-GGUF}
ENGRAM_DIR=${ENGRAM_DIR:-$MODEL_DIR/engram_rows}
CTX=${CTX:-16384}
DSPARK=${DSPARK:-0}
VISION=${VISION:-$([ "$DSPARK" = 1 ] && echo 0 || echo 1)}    # DSpark was measured text-only
DRAFT_N=${DRAFT_N:-3}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}

die() { echo "start.sh: $*" >&2; exit 1; }

model=$MODEL_DIR/DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf
[ -x "$BIN/llama-server" ] || die "no $BIN/llama-server; run thor/build.sh first"
for i in 1 2 3; do
    [ -f "$MODEL_DIR/DeepSeek-V4.1-Flash-1017-REAP192-L3-0000$i-of-00003.gguf" ] || die "missing shard $i in $MODEL_DIR"
done

# Engram: two ~101 GB row files read with O_DIRECT. They must be on local NVMe; USB disks stall decode.
for l in 1 14; do
    [ -f "$ENGRAM_DIR/engram_L$l.v41rows" ] || die "missing $ENGRAM_DIR/engram_L$l.v41rows (see thor/README.md, 'Engram row files')"
done
case $(realpath "$ENGRAM_DIR") in /media/*|/mnt/usb*) die "ENGRAM_DIR is on removable media; copy the row files to NVMe" ;; esac

# Thor has unified memory: a second model-loading process beside this one can hard-reboot the box.
for p in llama-server llama-cli llama-mtmd-cli v41_server v41_decode vllm; do
    pgrep -x "$p" >/dev/null && die "$p is already running; stop it first (unified memory)"
done
avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$avail" -lt 100 ]; then
    echo "start.sh: only $avail GiB available; dropping page cache" >&2
    sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$avail" -ge 96 ] || die "only $avail GiB available, need ~96"
fi

args=(-m "$model" -ngl 999 -c "$CTX" -np 1 --jinja --host "$HOST" --port "$PORT")

mmproj=$MODEL_DIR/mmproj-DeepSeek-V4.1-Flash-1017-bf16.gguf
if [ "$VISION" = 1 ]; then
    if [ -f "$mmproj" ]; then args+=(--mmproj "$mmproj"); else echo "start.sh: no mmproj in $MODEL_DIR, text only" >&2; fi
fi

if [ "$DSPARK" = 1 ]; then
    draft=$MODEL_DIR/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
    [ -f "$draft" ] || die "DSPARK=1 but no $draft"
    args+=(-md "$draft" -ngld 999 --spec-type draft-dspark --spec-draft-n-max "$DRAFT_N")
fi

export LLAMA_ENGRAM_ROWFILE_DIR=$ENGRAM_DIR
export LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
echo "start.sh: llama-server ${args[*]} ${EXTRA:-}" >&2
# shellcheck disable=SC2086
exec "$BIN/llama-server" "${args[@]}" ${EXTRA:-}
