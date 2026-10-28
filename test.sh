#!/bin/bash
# Evaluate a trained checkpoint on the test set of a fold (splits/fold<N>.csv, --fold 0 by default).
# FRAME_DIR points at the frame folder, CKPT at the checkpoint; extra arguments go to run.py.
FRAME_DIR="${FRAME_DIR:-data/frames}"
CKPT="${CKPT:-models/best_model_wts.pt}"
python run.py \
    --frame_dir "$FRAME_DIR" \
    --ckpt "$CKPT" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode eval \
    "$@"
