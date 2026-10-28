#!/bin/bash
# Evaluate a trained checkpoint on the test split that training saved (splits.npy).
# CKPT points at the checkpoint; extra arguments are passed on to run.py.
CKPT="${CKPT:-models/best_model_wts.pt}"
python run.py \
    --ckpt "$CKPT" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode eval \
    "$@"
