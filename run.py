"""
Module: run.py

This module is the main entry point for training or evaluating a video classification model.
It uses command-line arguments to configure the experiment, including dataset paths, model parameters,
and training hyperparameters. Depending on the selected mode ('train' or 'eval'), it performs
the following:

- Train mode:
    - Loads the dataset from a directory structure.
    - Splits the dataset into training, validation, and test sets.
    - Creates PyTorch Datasets and DataLoaders for training and validation.
    - Initializes the model (e.g., LRCN) and defines the loss function, optimizer, and learning rate scheduler.
    - Runs the training loop and saves the best model weights.

- Eval mode:
    - Loads the pre-generated dataset splits.
    - Creates a DataLoader for the test set.
    - Loads the trained model checkpoint.
    - Evaluates the model on the test set and prints overall test accuracy.

- Predict mode:
    - Like eval, but saves the softmax probabilities of the checkpoint on the validation and the
      test set to an .npz file (averaged over two interleaved clips per video, as the paper
      averages over clips), the input of the two-stream fusion in fuse.py.
    
The module also includes a helper function for parsing command-line arguments.
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR

import wandb

from video_datasets import VideoDataset, split_overlap, read_split_csv
from utils import transform_stats, compose_data_transforms, train_val_dloaders, test_dloaders
from models import LRCN, VideoCNN
from train import train
from evaluate import evaluate, predict, get_test_report

def args_parser():
    """
    Parse command-line arguments for configuring the video classification training or evaluation.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
        
    Arguments include:
        -fd/--frame_dir: Directory for storing video frames.
        --fold: Which cross-validation fold to use (default 0); the split is read from
            splits/fold<N>.csv written by make_splits.py.
        -fpv/--fr_per_vid: Number of frames per video to consider (default 16).
        -nc/--n_classes: Number of classes for the classification task (required).
        -c/--ckpt: Path for loading a trained model checkpoint.
        -mt/--model_type: Model type, either '3dcnn' or 'lrcn' (default 'lrcn').
        -cnn/--cnn_backbone: Backbone CNN for 2D feature extraction (default 'resnet34').
        -p/--pretrained: Whether to use a pretrained CNN backbone (default True).
        -rhs/--rnn_hidden_size: Number of neurons in the RNN/LSTM hidden layer (default 100).
        -rnl/--rnn_n_layers: Number of RNN/LSTM layers (default 1).
        -m/--mode: Mode of operation: 'train' or 'eval' (required).
        -bs/--batch_size: Mini-batch size (required).
        -d/--dropout: Dropout rate for regularization (default 0.1).
        -lr/--learning_rate: Learning rate for training (default 3e-5).
        -ne/--n_epochs: Number of training epochs (default 30).
    """
    parser = argparse.ArgumentParser(description='Video Classification Training')

    parser.add_argument('-fd', '--frame_dir', help='Directory for storing video frames')
    parser.add_argument('--fold', type=int, default=0,
                        help='cross-validation fold: the split is read from <split_dir>/fold<N>.csv')
    parser.add_argument('--split_dir', default='splits', help='folder of the fold CSVs')
    parser.add_argument('-fpv', '--fr_per_vid', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--modality', choices=['rgb', 'flow'], default='rgb',
                        help='what the frame folders hold: RGB frames or optical-flow images '
                             '(flow disables the horizontal flip and the colour jitter)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='DataLoader worker processes that decode the frames (0: main process)')
    parser.add_argument('-nc', '--n_classes', type=int, help='Number of classes for the classification task', required=True)

    parser.add_argument('-c', '--ckpt', help='Path for loading trained model checkpoints')
    parser.add_argument('-mt', '--model_type', choices=['lrcn', '3dcnn'], default='lrcn',
                        help="'lrcn' (ResNet + LSTM, 224x224) or '3dcnn' (R3D-18, 112x112)")
    parser.add_argument('-cnn', '--cnn_backbone', default='resnet34',
                        help='LRCN: resnet18, resnet34, resnet50, resnet101, resnet152; '
                             '3dcnn: r3d_18, r2plus1d_18')
    # A plain default=True argument receives the *string* 'False' from the command line, which is
    # truthy; BooleanOptionalAction gives real --pretrained / --no-pretrained flags.
    parser.add_argument('-p', '--pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='Use pretrained 2D CNN backbone')
    parser.add_argument('-rhs', '--rnn_hidden_size', type=int, default=100, help='Number of neurons in the RNN/LSTM hidden layer')
    parser.add_argument('-rnl', '--rnn_n_layers', type=int, default=1, help='Number of RNN/LSTM layers')
    # Fine-tuning recipe (all off by default = the hand-over's design)
    parser.add_argument('--temporal_pooling', choices=['last', 'late_fusion'], default='last',
                        help="classify the last LSTM output, or every frame's output averaged")
    parser.add_argument('--freeze_bn', action=argparse.BooleanOptionalAction, default=False,
                        help="keep the backbone's BatchNorm layers in eval mode while training")
    parser.add_argument('--freeze_stages', type=int, default=0,
                        help='backbone stages kept fixed: 1 = stem + layer1 ... 4 = all')
    parser.add_argument('--feature_dropout', type=float, default=0.0,
                        help='dropout on the per-frame CNN features before the LSTM')
    parser.add_argument('--bidirectional', action=argparse.BooleanOptionalAction, default=False,
                        help='bidirectional LSTM (forward and backward reads, concatenated)')

    parser.add_argument('-m', '--mode', type=str, default='train', required=True,
                        choices=['train', 'eval', 'predict'],
                        help="'train', 'eval' (test accuracy) or 'predict' (save probabilities)")
    parser.add_argument('--predictions', default='predictions.npz',
                        help='where --mode predict saves the val/test probabilities')
    parser.add_argument('-bs', '--batch_size', type=int, help='Mini-batch size', required=True)
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout rate for regularization')
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5, help='Learning rate for model training')
    parser.add_argument('-ne', '--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--backbone_lr_mult', type=float, default=1.0,
                        help='learning rate multiplier for the pretrained backbone (e.g. 0.1)')
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--scheduler', choices=['plateau', 'cosine'], default='plateau',
                        help='halve the LR when val loss stalls (hand-over) or cosine over the run')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='epochs of linear LR warm-up before the cosine schedule')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--aug', choices=['basic', 'strong'], default='basic',
                        help="training augmentation: the hand-over's flip + shift, "
                             "or also a random crop and brightness/contrast jitter")

    parser.add_argument('--wandb_project', default='jhu705643-a4-ucf50-lrcn',
                        help='Weights & Biases project')
    parser.add_argument('--run_name', default=None,
                        help='Weights & Biases run name (default: chosen by W&B)')

    args = parser.parse_args()
    return args

def load_split(fold, frame_dir, split_dir='splits'):
    """
    Read the split CSV of a fold, check that no group is in two sets, and print the sizes.

    Args:
        fold (int): The fold number; the file is <split_dir>/fold<fold>.csv.
        frame_dir (str): The frame directory the CSV's clip paths are relative to.
        split_dir (str): The folder of the fold CSVs. Default is 'splits'.

    Returns:
        dict: Dictionary with keys 'train', 'val' and 'test' (lists of (path, label)).
    """
    csv_path = os.path.join(split_dir, f'fold{fold}.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'{csv_path} not found: run make_splits.py first')
    splits = read_split_csv(csv_path, frame_dir)
    shared = split_overlap(splits)
    if shared:
        raise RuntimeError(f'{len(shared)} groups appear in more than one split, '
                           f'e.g. {sorted(shared)[:3]}')
    print(f'Loaded {csv_path}: ' + ', '.join(f'{name} {len(clips)}' for name, clips in splits.items()))
    return splits


def make_optimizer_and_scheduler(model, args):
    """
    Build the optimizer and the learning rate scheduler from the command-line choices.

    Args:
        model (LRCN): The model whose parameters are optimised.
        args (argparse.Namespace): learning_rate, backbone_lr_mult, optimizer, weight_decay,
            scheduler, warmup_epochs, n_epochs.

    Returns:
        tuple: (optimizer, scheduler).
    """
    groups = model.parameter_groups(args.learning_rate, args.backbone_lr_mult)
    if args.optimizer == 'adamw':
        opt = optim.AdamW(groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        opt = optim.Adam(groups, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.scheduler == 'plateau':
        return opt, ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    # Cosine decay to zero over the epochs after the warm-up; the warm-up ramps the learning
    # rate from 10% to 100% so the randomly initialised layers do not disturb the pretrained
    # backbone in the first epochs.
    cosine = CosineAnnealingLR(opt, T_max=max(1, args.n_epochs - args.warmup_epochs))
    if args.warmup_epochs == 0:
        return opt, cosine
    warmup = LinearLR(opt, start_factor=0.1, total_iters=args.warmup_epochs)
    return opt, SequentialLR(opt, [warmup, cosine], milestones=[args.warmup_epochs])


def main(args):
    """
    Main function to execute training or evaluation based on the parsed command-line arguments.

    For training:
        - Loads the dataset from the specified frame directory.
        - Splits the dataset into training, validation, and test sets.
        - Saves the splits for later use.
        - Creates the training and validation DataLoaders.
        - Initializes the model (LRCN) with specified hyperparameters.
        - Defines the loss function, optimizer, and learning rate scheduler.
        - Runs the training loop and saves the best model weights.

    For evaluation:
        - Loads the saved dataset splits.
        - Creates the test DataLoader.
        - Loads the trained model checkpoint.
        - Evaluates the model on the test set and prints the overall test accuracy.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Dataset parameters
    frame_dir = args.frame_dir
    fr_per_vid = args.fr_per_vid
    n_classes = args.n_classes

    # Model parameters
    model_type = args.model_type
    rnn_hidden_size = args.rnn_hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    pretrained = args.pretrained
    cnn_backbone = args.cnn_backbone
    ckpt = args.ckpt

    # Training parameters
    mode = args.mode
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load transformation statistics and create data augmentation transforms
    h, w, mean, std = transform_stats(model_type)
    tr_transforms, val_ts_transforms = compose_data_transforms(h, w, mean, std,
                                                               strong=args.aug == 'strong',
                                                               modality=args.modality)

    # Initialize the model: the LRCN, or the 3D CNN of the hand-over's '3dcnn' model type
    if model_type == '3dcnn':
        model = VideoCNN(n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone,
                         freeze_bn=args.freeze_bn, freeze_stages=args.freeze_stages,
                         dropout_rate=dropout)
    else:
        model = LRCN(hidden_size=rnn_hidden_size, n_layers=rnn_n_layers, dropout_rate=dropout,
                     n_classes=n_classes, pretrained=pretrained, cnn_model=cnn_backbone,
                     temporal_pooling=args.temporal_pooling, freeze_bn=args.freeze_bn,
                     freeze_stages=args.freeze_stages, feature_dropout=args.feature_dropout,
                     bidirectional=args.bidirectional)

    if mode == 'train':
        # Load this fold's split (written once by make_splits.py and committed)
        splits = load_split(args.fold, frame_dir, args.split_dir)
        tr_split, val_split, ts_split = splits['train'], splits['val'], splits['test']

        # Create PyTorch Datasets and DataLoaders for train, validation and test. The test loader is
        # only used to log test loss/accuracy per epoch; model selection uses validation.
        # Training draws a random clip of each video every epoch; val and test use a fixed clip
        tr_dataset = VideoDataset(tr_split, fr_per_vid, tr_transforms, random_clip=True)
        val_dataset = VideoDataset(val_split, fr_per_vid, val_ts_transforms)
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = train_val_dloaders(tr_dataset, val_dataset, batch_size, model_type,
                                         args.num_workers)
        dataloaders.update(test_dloaders(ts_dataset, batch_size, model_type, args.num_workers))

        # Define the loss function, optimizer, and learning rate scheduler
        loss_func = nn.CrossEntropyLoss(reduction='sum', label_smoothing=args.label_smoothing)
        opt, lr_scheduler = make_optimizer_and_scheduler(model, args)
        os.makedirs("./models", exist_ok=True)
        optim_model_dir = './models'

        # Main training procedure, with every epoch logged to Weights & Biases
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        model.to(device)
        class_names = sorted(os.listdir(frame_dir))
        model, loss_hist, acc_hist = train(dataloaders, model, loss_func, opt, lr_scheduler, device,
                                           optim_model_dir, n_epochs, class_names)
        wandb.finish()

    elif mode == 'eval':
        # Load this fold's split and evaluate on its test set
        ts_split = load_split(args.fold, frame_dir, args.split_dir)['test']

        # Create PyTorch Dataset and DataLoader for the test set
        ts_dataset = VideoDataset(ts_split, fr_per_vid, val_ts_transforms)
        dataloaders = test_dloaders(ts_dataset, batch_size, model_type, args.num_workers)

        # Load the trained model checkpoint
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.to(device)
        targets, outputs, accuracy = evaluate(model, dataloaders['test'], device)

        print(f'The overall test accuracy is {100 * accuracy:.4f}%.')
        # Per-class recall, worst classes first, from the class folder names
        class_names = sorted(os.listdir(frame_dir))
        report = get_test_report(targets, outputs, class_names)
        print('Lowest per-class recall:')
        for name in sorted(class_names, key=lambda n: report[n]['recall'])[:5]:
            print(f"  {name:20s} recall {100 * report[name]['recall']:.1f}%  "
                  f"(n={int(report[name]['support'])})")

    elif mode == 'predict':
        # Softmax probabilities of the checkpoint on the val and test sets, averaged over the
        # two interleaved 16-frame clips of each video, saved for the two-stream fusion
        splits = load_split(args.fold, frame_dir, args.split_dir)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.to(device)
        saved = {'class_names': np.array(sorted(os.listdir(frame_dir))), 'ckpt': args.ckpt}
        for name in ('val', 'test'):
            probs = 0
            for shifted in (False, True):
                dataset = VideoDataset(splits[name], fr_per_vid, val_ts_transforms, shifted=shifted)
                loader = test_dloaders(dataset, batch_size, model_type, args.num_workers)['test']
                clip_probs, targets = predict(model, loader, device)
                probs = probs + clip_probs / 2
            saved[f'probs_{name}'], saved[f'targets_{name}'] = probs, targets
            saved[f'clips_{name}'] = np.array([os.path.relpath(path, frame_dir)
                                               for path, _ in splits[name]])
            print(f'{name}: accuracy {100 * np.mean(probs.argmax(axis=1) == targets):.2f}%')
        np.savez(args.predictions, **saved)
        print(f'Probabilities saved to {args.predictions}')

    else:
        raise ValueError('The mode argument must be "train", "eval" or "predict".')

if __name__ == "__main__":
    args = args_parser()
    main(args)
