#!/bin/bash
# Train the LRCN on the UCF50 frames written by preprocess.py.
# FRAME_DIR points at that frame folder; extra arguments are passed on to run.py
# (for example: FRAME_DIR=/data/frames bash train.sh --n_epochs 10 --run_name my_run).
FRAME_DIR="${FRAME_DIR:-data/frames}"
python run.py \
    --frame_dir "$FRAME_DIR" \
    --train_size 0.75 \
    --test_size 0.15 \
    --model_type lrcn \
    --n_classes 50 \
    --fr_per_vid 16 \
    --batch_size 4 \
    --mode train \
    "$@"
