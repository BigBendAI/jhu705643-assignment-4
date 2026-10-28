"""
Module: make_splits.py

Write the group-level cross-validation splits of the frame directory to CSV files, one per fold
(splits/fold0.csv ... splits/fold4.csv). The CSVs are committed, so every training run and every
evaluation uses exactly the same clips in each set, on any machine.

Usage:
    python make_splits.py --frame_dir data/frames --out_dir splits --n_folds 5 --seed 0
"""

import argparse
import os

from video_datasets import load_dataset, dataset_split, split_overlap, write_split_csv


def main():
    """Parse the arguments and write one split CSV per fold."""
    parser = argparse.ArgumentParser(description='Write the group-level split of every fold')
    parser.add_argument('--frame_dir', required=True, help='Frame folders, one per class')
    parser.add_argument('--out_dir', default='splits', help='Where the CSV files go')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of test folds')
    parser.add_argument('--n_val_groups', type=int, default=2, help='Validation groups per class')
    parser.add_argument('--seed', type=int, default=0, help='Seed of the group shuffle')
    args = parser.parse_args()

    vid_dataset, _ = load_dataset(args.frame_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for fold in range(args.n_folds):
        parts = dataset_split(vid_dataset, fold, args.n_folds, args.n_val_groups, args.seed)
        splits = dict(zip(('train', 'val', 'test'), parts))
        if split_overlap(splits):
            raise RuntimeError(f'fold {fold}: a group appears in two sets')
        csv_path = os.path.join(args.out_dir, f'fold{fold}.csv')
        write_split_csv(csv_path, splits, args.frame_dir)
        print(f"{csv_path}: train {len(splits['train'])}, val {len(splits['val'])}, "
              f"test {len(splits['test'])} clips")


if __name__ == '__main__':
    main()
