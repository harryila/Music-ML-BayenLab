#!/bin/bash
# Tuplet-aware continue-train: warm-start from ssl_classical_clean ep13 and continue the SAME masked-SSL
# recipe (50/50 real ASAP / unpaired classical scores), but upweight the rare NON-DYADIC (tuplet) buckets
# of the duration/offset/downbeat cross-entropy. Diagnosis (LAB_REPORT.md §11): ssl_classical_clean's
# tuplet head COLLAPSED (emits 0 tuplets on 9/14 ASAP pieces; NoteDuration is the dominant error stream,
# 0.654 vs the released model's 0.192 on Haydn). The released model emits tuplets across the board, so the
# gap is a training-objective problem, not data-ceiling and not field-wide-hard. Upweighting tuplet buckets
# un-collapses the head while warm-start preserves the pitch/structure streams already at parity.
set -uo pipefail
cd /root/Music-ML-BayenLab || exit 1
REPO="$PWD"

TUPLET_WEIGHT="${TUPLET_WEIGHT:-5.0}"
TUPLET_GAMMA="${TUPLET_GAMMA:-0.0}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-15}"
WARMUP="${WARMUP:-300}"
STAGE="${STAGE:-ssl_tuplet5}"

# Reshape runs need the tuplet_rate-augmented manifest; default to the plain one otherwise.
MAN="${MANIFEST:-$REPO/data/pairs_classical_clean_manifest.csv}"
INIT="$REPO/MIDI2ScoreTransformer/checkpoints/ssl_classical_clean/ssl_classical_clean-epoch=13-val/total=0.5125.ckpt"
OUT="$REPO/MIDI2ScoreTransformer/checkpoints/$STAGE"
LOG="/root/$STAGE.log"

[ -f "$MAN" ]  || { echo "MISSING manifest: $MAN"; exit 1; }
[ -f "$INIT" ] || { echo "MISSING init ckpt: $INIT"; exit 1; }
# Guard: --tuplet-gamma>0 silently NO-OPS unless the manifest carries a tuplet_rate column
# (the dataset's `'tuplet_rate' in columns` check). Fail loudly so a "reshape" run can't
# secretly degrade to a plain warm-start (this ambiguity affected ssl_reshape_g1/g2).
if [ "$TUPLET_GAMMA" != "0.0" ] && [ "$TUPLET_GAMMA" != "0" ]; then
  head -1 "$MAN" | grep -q "tuplet_rate" || {
    echo "ERROR: --tuplet-gamma=$TUPLET_GAMMA but manifest has no 'tuplet_rate' column: $MAN"
    echo "       Use MANIFEST=\$REPO/data/pairs_classical_clean_tuplrate.csv for reshape runs."; exit 1; }
fi
mkdir -p "$OUT"
pkill -9 -f '[t]rain.py fit' 2>/dev/null; sleep 2
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO/MIDI2ScoreTransformer/midi2scoretransformer"
setsid nohup "$REPO/venv311/bin/python" -u \
  "$REPO/MIDI2ScoreTransformer/midi2scoretransformer/train.py" fit \
  --stage "$STAGE" --dataset-type ssl --real-fraction 0.5 \
  --manifest "$MAN" --data-dir "$REPO/MIDI2ScoreTransformer/data/" \
  --init-ckpt "$INIT" --autoregressive \
  --tuplet-weight "$TUPLET_WEIGHT" --tuplet-gamma "$TUPLET_GAMMA" \
  --lr "$LR" --max-epochs "$EPOCHS" --warmup-steps "$WARMUP" \
  --batch-size 32 --seq-length 512 --precision bf16-mixed --num-workers 10 \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "$STAGE launched (pid $!) -> $LOG   [tuplet_weight=$TUPLET_WEIGHT tuplet_gamma=$TUPLET_GAMMA lr=$LR epochs=$EPOCHS warmup=$WARMUP manifest=$(basename "$MAN")]"
