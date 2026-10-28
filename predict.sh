#!/bin/bash
# Save a checkpoint's val/test class probabilities (for fuse.py), averaged over two clips per
# video. FRAME_DIR points at the frame (or flow) folder, CKPT at the checkpoint, OUT at the .npz
# to write; extra arguments go to run.py (pass the same model flags the checkpoint was trained
# with, and --fold / --modality as used).
FRAME_DIR="${FRAME_DIR:-data/frames}"
CKPT="${CKPT:-models/best.pt}"
OUT="${OUT:-predictions.npz}"
python run.py \
    --frame_dir "$FRAME_DIR" \
    --ckpt "$CKPT" \
    --predictions "$OUT" \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 16 \
    --mode predict \
    "$@"
