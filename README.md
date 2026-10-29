# Assignment 4 - Video Action Recognition

Rajbir S Chahal

The assignment code implements an LRCN - a ResNet turns each frame into a feature vector, an LSTM reads the sequence of vectors, a classifier names the action.
The code was modified to use UCF50 dataset, log to Weights & Biases, debugged and improved. The W&B project <https://wandb.ai/bigbendai/jhu705643-a4-ucf50-lrcn> has every run's train, validation and test loss and accuracy per epoch.

## Results

The results are reported on a group-disjoint split: no UCF50 group (the clips cut from the same video) is placed in more than one of the train, validation and test sets.
The final models are evaluated with 5-fold cross-validation (see Protocol below); the development runs use fold 0.

| step | commit | run | val | test |
|---|---|---|---:|---:|
| assignment code (clip-level split, LSTM bug), 10 epochs | 0d0caeb | R1 | 90.13 | 88.63 |
| data leak fixed (group split), 10 epochs | c2f4e15 | R2 | 65.85 | 67.09 |
| LSTM bug fixed, 10 epochs | 48bacb4 | R3 | 80.68 | 80.32 |
| data pipeline fixed, 30 epochs: **the fixed baseline** | e6d5f0f | R4 | 83.86 | 82.34 |
| improvement 1: fine-tuning recipe + ResNet50 | 9025edd | R5 | 86.87 | 88.88 |
| improvement 2: bidirectional two-layer LSTM | d39cc57 | R6 | 85.55 | 89.48 |
| improvement 3: optical-flow stream | 431b516 | R7 | 78.99 | 78.06 |
| improvement 3: two-stream fusion of R6 and R7 (two clips per video) | 431b516 | – | 90.62 | 92.56 |

Five-fold cross-validation of the final models (four interleaved clips per video, val/test
frames at 256, equal-weight fusion in the log domain; see Protocol):

| fold | RGB LRCN | flow LRCN | **two-stream LRCN** | 3D CNN (extension) | three streams |
|---|---|---|---|---|---|
| 0 | 87.05 / 90.16 | 81.24 / 82.27 | 90.62 / 94.06 | 92.87 / 92.79 | 93.43 / 95.04 |
| 1 | 88.93 / 87.10 | 80.07 / 81.36 | 92.99 / 92.54 | 94.10 / 92.10 | 94.65 / 94.11 |
| 2 | 88.14 / 88.61 | 83.24 / 78.74 | 92.28 / 91.80 | 94.54 / 91.65 | 93.22 / 92.71 |
| 3 | 86.25 / 90.65 | 81.73 / 81.83 | 91.53 / 94.04 | 92.84 / 92.99 | 93.60 / 94.80 |
| 4 | 85.88 / 87.63 | 80.04 / 85.29 | 91.34 / 93.63 | 94.35 / 93.56 | 93.41 / 94.95 |
| **mean ± std (test)** | 88.83 ± 1.38 | 81.90 ± 2.09 | **93.22 ± 0.90** | 92.61 ± 0.67 | **94.32 ± 0.87** |

Each cell in the table above displays the validation / test accuracy in percent of the checkpoint from the epoch with the best validation accuracy, under the evaluation settings above.

**The two-stream LRCN reaches 93.2% test accuracy on average over the five folds (91.8 to 94.1).**

**Adding the 3D CNN stream, the extension described below, raises the fusion to 94.3%.** 


## Code fixes

| problem | how it was found | fix (commit) |
|---|---|---|
| `run_training.py` imported a function that does not exist; `preprocess.py` was missing; torchvision's `pretrained=` argument and `ReduceLROnPlateau(verbose=)` were removed; `--pretrained False` was the string `'False'` (truthy); `resnet152` built a resnet34; `train.sh` pointed at HMDB51 with 51 classes; `test.sh` loaded a file training never wrote | the tracebacks, pylint on the assignment code, reading the scripts | `preprocess.py` (32 frames per clip), `weights=` API, real boolean flag, `map_location`, UCF50 wrappers, pinned requirements (0d0caeb) |
| **the data leak**: clips were split at random, but clips of one group share the person, background and viewpoint, so 99% of the test clips had a near-twin in the training set; validation tracked test epoch by epoch (R1) | counting test clips whose group also appears in train: 99.2% | split by group, dealt into 5 folds per class, five CSVs committed, an overlap check on every load (c2f4e15): R1 88.6 → R2 67.1 test |
| **the significant bug**: `nn.LSTM` built without `batch_first` was fed `(batch, 1, features)` one frame at a time, so it read the batch axis as time; videos influenced each other's predictions and no video's frames were modelled as a sequence | printing the hidden state shape in `forward()`: `(layers, 1, hidden)` instead of `(layers, batch, hidden)` | `batch_first=True`, CNN on all frames at once, one LSTM call per batch; a test that perturbs one video and checks the others' logits (48bacb4): R2 67.1 → R3 80.3 test |
| frames listed with `glob` (directory order, so shuffled); only the first 16 files used; augmentation drawn per frame instead of per clip; no loader workers (GPU idle 75%, 8 min/epoch); `test.py` clashed with pytest and Python's `test` package | reading `__getitem__` and the collate function; `nvidia-smi` | sorted frames; training draws one random frame per segment of the 32 stored frames each epoch, val/test use fixed evenly spaced frames; one random flip/shift per clip; 16 workers (1 min/epoch); `evaluate.py`; `best.pt`/`last.pt`; W&B confusion matrix (e6d5f0f) |
| slow forward pass | timing | bf16 autocast and channels-last: 55 s → 40 s per training pass, no accuracy change (15304bf) |


## Model-level improvements 

**1. The paper's LRCN head and a transfer-learning recipe, ResNet50** (9025edd). The baseline
fine-tunes the whole ImageNet ResNet and the untrained LSTM with one learning rate, re-estimates
BatchNorm from four clips per batch and classifies from the last frame's output; it memorises
the training clips. R4 → R5: 82.3 → 88.9 test. The changes:

- **The paper's head.** The classifier is applied to every frame's LSTM output and the logits
  are averaged ("predicts the video class at each time step and we average these predictions");
  LSTM with 256 units, as in the paper.
- **Transfer learning.** The pretrained backbone is changed gently: its BatchNorm statistics are
  frozen, its stem and first stage are frozen, and it trains at a tenth of the learning rate of
  the new LSTM and classifier.
- **Optimiser.** AdamW with weight decay, cosine learning-rate schedule with two warm-up epochs.
- **Regularisation.** Dropout 0.5 on the frame features and on the classifier; label smoothing
  0.1; a random crop and brightness/contrast jitter drawn once per clip.
- **Capacity.** ResNet50 backbone instead of the baseline's ResNet34; batch size 16.


**2. Bidirectional two-layer LSTM** (d39cc57). The baseline has one LSTM layer that reads the 
sixteen frames in order, so its state at any frame
knows only the frames before it, and it has one layer to turn the frame features into a
decision. R5 → R6: 88.9 → 89.5 test (within the run-to-run spread). The changes:

- **Bidirectional.** A second LSTM reads the frames in reverse and its output is joined to the
  forward one at every frame, so each frame's output reflects the whole clip. The clip is
  classified after the fact, so there is no reason to hide the later frames.
- **Two layers.** A second LSTM layer reads the first layer's outputs, so it can form patterns
  of patterns over time, with dropout between the layers to limit memorising.


**3. Two-stream model: an optical-flow LRCN fused with the RGB LRCN** (431b516). The RGB LRCN
sees appearance, which can mislead it when two actions happen in similar scenes; it has no
direct view of motion. Flow alone 78.1 test; R6 + R7 on fold 0: 89.5 → 92.6 test. The changes:

- **Optical flow.** `preprocess.py --modality flow` computes the optical flow between each
  stored frame and the next (Farneback, OpenCV) and stores it as an image: x and y displacement
  centred at 128, magnitude as the third channel, as in the paper.
- **A second LRCN on the flow images.** The same model and recipe trained on the flow images
  instead of the colour frames, with no horizontal flip (it would reverse the motion). It sees
  only what moved.
- **Fusion.** `predict.sh` averages each stream's probabilities over interleaved 16-frame clips
  of a video; `fuse.py` combines the two streams with equal weights in the log domain (mean of
  the log-probabilities, the product rule). The streams are trained with label smoothing, so
  their probabilities are flat and a plain mean lets the weaker flow stream pull a confident
  RGB vote; the product rewards the class both streams rate well. Chosen on validation over the
  probability mean: higher on four folds, tied on one (mean validation 90.59 → 91.41; test
  91.55 → 92.48, recorded only).


**Extension: 3D CNN model type** (47b0a0c). The given code has
`--model_type 3dcnn`, 112x112 transforms with the Kinetics normalisation and `collate_fn_r3d_18`,
but no 3D model behind them. Alone it averages 92.6% test; fusing all three streams averages
94.3%, the best result of this project. The changes:

- **The model.** `models.VideoCNN` completes that path with torchvision's R3D-18, a ResNet whose
  convolutions run over space and time, pretrained on the Kinetics-400 video dataset. It sees
  motion directly, so it complements the two LRCN streams.
- **Same recipe, same protocol.** Fine-tuned with the recipe of improvement 1, evaluated with
  the same folds and settings, and fused with the two LRCN streams by `fuse.py`.


## Protocol

- **Split by group.** Within each class the 25 groups are shuffled once (seed 0) and dealt into
  5 folds. For fold k, its groups are the test set, the next 2 groups the validation set and the
  remaining groups (about 18) the training set. `make_splits.py` writes `splits/fold0.csv` to
  `fold4.csv` (committed; about 4,800 / 530 / 1,330 clips each); `run.py --fold k` reads one and
  refuses to run if any group is in two sets. Every group is tested exactly once over the five
  folds.
- **Selection on validation only.** The checkpoint kept is the one from the epoch with the best
  validation accuracy. The test set is evaluated every epoch for the W&B curves and reported
  once, at that epoch. Fusion weights are fixed and equal; no weight search, no ensembles of
  repeated runs, nothing chosen on test.
- **Development on fold 0, final models on all folds.** Runs R1 to R7 use fold 0. The final RGB,
  flow and 3D CNN models are trained on each of the five folds (`cv.sh`); the headline is the
  mean test accuracy over the folds with its standard deviation (`cv_summary.py`).
- **Four clips per video at evaluation.** Training uses 16 of 32 stored frames per clip. For the
  fold table, 64 frames are stored per video (`preprocess.py --n_frames 64`) and the predictions
  of four interleaved 16-frame clips are averaged (`predict.sh --n_views 4`), as the paper
  averages over clips. The checkpoints are unchanged.
- **Val/test frames at 256.** The LRCN streams are evaluated on frames resized to 256 x 256
  instead of the training size 224 (the 3D CNN at 128 instead of 112; `--eval_size`, set in
  `cv.sh`). The training augmentation crops 49 to 100% of the frame and resizes it to 224, so
  the network trains on a zoomed-in view on average; the full frame resized to 224 at test time
  shows everything smaller than in training, and a larger test frame evens that out (the
  train/test resolution discrepancy, Touvron et al., NeurIPS 2019). The backbones take any size,
  so the checkpoints are unchanged. Chosen on validation: the two-stream validation mean goes
  from 91.41 to 91.75 (up on three folds, tied on one, down 0.75 on fold 0); the test mean,
  recorded only, from 92.48 to 93.22. At the training size the same checkpoints give
  92.48 ± 0.96 (two streams) and 93.90 ± 0.97 (three streams).
- **Hardware.** One NVIDIA A100 80 GB; PyTorch 2.12.1 with CUDA 12.6, Python 3.10. A 30-epoch
  LRCN run takes 25 to 30 minutes, a 3D CNN run 7 minutes.


## How to run

```bash
pip install -r requirements.txt

# frames (32 per clip) and optical-flow images at the same positions, for training
python preprocess.py --video_dir data/UCF50 --frame_dir data/frames --workers 8
python preprocess.py --video_dir data/UCF50 --frame_dir data/flow --modality flow --workers 8
# 64 per clip for the four-clip evaluation (optional; with the 32-frame folders use --n_views 2)
python preprocess.py --video_dir data/UCF50 --frame_dir data/frames64 --n_frames 64 --workers 8
python preprocess.py --video_dir data/UCF50 --frame_dir data/flow64 --n_frames 64 --modality flow --workers 8
python make_splits.py --frame_dir data/frames           # splits/fold0.csv ... fold4.csv

# the fixed baseline (assignment design and hyper-parameters), fold 0
FRAME_DIR=data/frames bash train.sh --fold 0 --run_name baseline

# the final streams on fold 0 (the recipe flags are listed in cv.sh)
RECIPE="--batch_size 16 --cnn_backbone resnet50 --temporal_pooling late_fusion --rnn_hidden_size 256
        --freeze_bn --freeze_stages 1 --backbone_lr_mult 0.1 --learning_rate 3e-4 --optimizer adamw
        --weight_decay 0.05 --scheduler cosine --warmup_epochs 2 --dropout 0.5 --feature_dropout 0.5
        --label_smoothing 0.1 --aug strong"
FRAME_DIR=data/frames bash train.sh $RECIPE --bidirectional --rnn_n_layers 2 --fold 0 --run_name rgb
FRAME_DIR=data/flow   bash train.sh $RECIPE --modality flow --fold 0 --run_name flow

# test accuracy of a checkpoint; probabilities of each stream; fusion
FRAME_DIR=data/frames CKPT=models/best.pt bash test.sh --cnn_backbone resnet50 \
    --temporal_pooling late_fusion --rnn_hidden_size 256 --bidirectional --rnn_n_layers 2 --fold 0
FRAME_DIR=data/frames64 CKPT=rgb_best.pt  OUT=rgb.npz  bash predict.sh --cnn_backbone resnet50 \
    --temporal_pooling late_fusion --rnn_hidden_size 256 --bidirectional --rnn_n_layers 2 \
    --fold 0 --n_views 4 --eval_size 256
FRAME_DIR=data/flow64   CKPT=flow_best.pt OUT=flow.npz bash predict.sh --cnn_backbone resnet50 \
    --temporal_pooling late_fusion --rnn_hidden_size 256 --modality flow \
    --fold 0 --n_views 4 --eval_size 256
python fuse.py --members rgb.npz flow.npz --names rgb flow --out fusion.json

# all five folds (add STREAMS=3 for the 3D CNN stream), then the fold table of Results
FRAME_DIR=data/frames FLOW_DIR=data/flow EVAL_FRAME_DIR=data/frames64 EVAL_FLOW_DIR=data/flow64 \
    OUT_DIR=cv bash cv.sh 0 1 2 3 4
python cv_summary.py --out_dir cv

# tests and lint
python -m pytest -q tests          # 31 tests, synthetic data, no GPU
python -m pylint *.py tests/*.py   # 10.00/10
```

`train.sh` holds the assignment's data settings (16 frames per clip, batch 4); anything after it
is passed to `run.py`, and a later flag overrides an earlier one. Training writes
`models/best.pt` and `models/last.pt`.


## Tests and lint

`tests/` has 31 pytest tests on synthetic data: the group split (no shared group, every class in
every set, every group tested once over the folds, CSV round trip); the LSTM fix (perturbing one
video of a batch must not change the others' logits, which fails on the assignment model);
frame order, evenly spaced and random-per-segment sampling, interleaved evaluation views,
per-clip transforms, the evaluation size, the collate function; the model options (late fusion,
frozen BatchNorm, frozen stage, parameter groups, bidirectional LSTM, the 3D CNN); the flow
image encoding; the fusion (product rule). pylint scores 10.00/10 with `.pylintrc` setting the
line length to 100 and declaring OpenCV's members; no message category is disabled.

Every module, class and function has a docstring; pylint's documentation checks are enabled.

## Files

| file | role |
|---|---|
| `preprocess.py` | videos → frame folders (`--modality rgb` or `flow`) |
| `make_splits.py` | the group-level fold splits → `splits/fold<k>.csv` |
| `video_datasets.py` | `VideoDataset`, the group split, split CSVs, collate functions |
| `utils.py` | frame and flow extraction, transforms, DataLoaders |
| `models.py` | `LRCN` and `VideoCNN` |
| `train.py` | the training loop with W&B logging and checkpoints |
| `evaluate.py` | accuracy, probabilities, reports (was `test.py`) |
| `fuse.py` | score fusion of streams |
| `run.py` | entry point: `--mode train`, `eval`, `predict` |
| `train.sh`, `test.sh`, `predict.sh`, `cv.sh`, `cv_summary.py` | wrappers and cross-validation |
| `tests/` | the pytest suite |
