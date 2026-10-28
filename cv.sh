#!/bin/bash
# Cross-validation over the group folds: for every fold given on the command line (default 0 to
# 4) train the RGB stream and the flow stream with the final recipe, save each stream's
# probabilities (two clips per video) and fuse them. With STREAMS=3 the 3D CNN stream (the
# hand-over's '3dcnn' model type) is trained too and a three-stream fusion is written next to the
# two-stream one. Then cv_summary.py reports the per-fold numbers and their mean.
#   FRAME_DIR=data/frames FLOW_DIR=data/flow OUT_DIR=cv bash cv.sh 1 2 3 4
#   FRAME_DIR=data/frames FLOW_DIR=data/flow OUT_DIR=cv STREAMS=3 bash cv.sh 0 1 2 3 4
# Each fold writes OUT_DIR/fold<k>/{rgb_best.pt, flow_best.pt, rgb.npz, flow.npz, fusion.json}
# and, with STREAMS=3, {3dcnn_best.pt, 3dcnn.npz, fusion3.json}.
set -e
FRAME_DIR="${FRAME_DIR:-data/frames}"
FLOW_DIR="${FLOW_DIR:-data/flow}"
OUT_DIR="${OUT_DIR:-cv}"
STREAMS="${STREAMS:-2}"
FOLDS="${@:-0 1 2 3 4}"

# The recipe of improvements 1 to 3 (see the README); the RGB stream adds the bidirectional LSTM.
RECIPE="--batch_size 16 --n_epochs 30 --cnn_backbone resnet50 --temporal_pooling late_fusion
        --rnn_hidden_size 256 --freeze_bn --freeze_stages 1 --backbone_lr_mult 0.1
        --learning_rate 3e-4 --optimizer adamw --weight_decay 0.05 --scheduler cosine
        --warmup_epochs 2 --dropout 0.5 --feature_dropout 0.5 --label_smoothing 0.1 --aug strong"
MODEL="--cnn_backbone resnet50 --temporal_pooling late_fusion --rnn_hidden_size 256"
# The 3D CNN stream: the same optimiser settings and augmentation; the LSTM flags do not apply.
RECIPE3D="--model_type 3dcnn --cnn_backbone r3d_18 --batch_size 16 --n_epochs 30 --freeze_bn
          --freeze_stages 1 --backbone_lr_mult 0.1 --learning_rate 3e-4 --optimizer adamw
          --weight_decay 0.05 --scheduler cosine --warmup_epochs 2 --dropout 0.5
          --label_smoothing 0.1 --aug strong"

for k in $FOLDS; do
    out="$OUT_DIR/fold$k"
    mkdir -p "$out"
    echo "== fold $k: RGB stream =="
    rm -rf models
    FRAME_DIR="$FRAME_DIR" bash train.sh $RECIPE --bidirectional --rnn_n_layers 2 --fold "$k" \
        --run_name "cv_fold${k}_rgb"
    cp models/best.pt "$out/rgb_best.pt"
    echo "== fold $k: flow stream =="
    rm -rf models
    FRAME_DIR="$FLOW_DIR" bash train.sh $RECIPE --modality flow --fold "$k" \
        --run_name "cv_fold${k}_flow"
    cp models/best.pt "$out/flow_best.pt"
    echo "== fold $k: predictions and fusion =="
    FRAME_DIR="$FRAME_DIR" CKPT="$out/rgb_best.pt" OUT="$out/rgb.npz" \
        bash predict.sh $MODEL --bidirectional --rnn_n_layers 2 --fold "$k"
    FRAME_DIR="$FLOW_DIR" CKPT="$out/flow_best.pt" OUT="$out/flow.npz" \
        bash predict.sh $MODEL --modality flow --fold "$k"
    python fuse.py --members "$out/rgb.npz" "$out/flow.npz" --names rgb flow --out "$out/fusion.json"
    if [ "$STREAMS" = "3" ]; then
        echo "== fold $k: 3D CNN stream =="
        rm -rf models
        FRAME_DIR="$FRAME_DIR" bash train.sh $RECIPE3D --fold "$k" --run_name "cv_fold${k}_3dcnn"
        cp models/best.pt "$out/3dcnn_best.pt"
        FRAME_DIR="$FRAME_DIR" CKPT="$out/3dcnn_best.pt" OUT="$out/3dcnn.npz" \
            bash predict.sh --model_type 3dcnn --cnn_backbone r3d_18 --fold "$k"
        python fuse.py --members "$out/rgb.npz" "$out/flow.npz" "$out/3dcnn.npz" \
            --names rgb flow 3dcnn --out "$out/fusion3.json"
    fi
done
python cv_summary.py --out_dir "$OUT_DIR"
