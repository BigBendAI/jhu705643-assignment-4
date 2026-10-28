"""
Module: fuse.py

Two-stream score fusion: average the class probabilities of the RGB LRCN and the optical-flow
LRCN (the LRCN paper, Table 1; lecture: two-stream networks). The inputs are the
predictions.npz files written by ``run.py --mode predict``, one per stream, each holding the
softmax probabilities and the labels of the validation and the test clips.

The streams are averaged with equal weights, the paper's choice. The validation accuracy of the
fusion is printed next to the test accuracy; nothing is tuned on the test set.

Usage:
    python fuse.py --members rgb_predictions.npz flow_predictions.npz --names rgb flow \\
        --out fusion.json
"""

import argparse
import json

import numpy as np


def load_predictions(path):
    """
    Load one predictions file (from run.py --mode predict) into a dict of arrays.

    Args:
        path (str): The .npz file.

    Returns:
        dict: probs_val, targets_val, clips_val, probs_test, targets_test, clips_test, ...
    """
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def check_same_clips(members):
    """
    Make sure every member holds the same clips in the same order with the same labels.

    Args:
        members (list): Prediction dicts as returned by load_predictions.

    Raises:
        ValueError: If two members disagree on a split.
    """
    for split in ('val', 'test'):
        for other in members[1:]:
            if not np.array_equal(members[0][f'targets_{split}'], other[f'targets_{split}']):
                raise ValueError(f'members disagree on the {split} labels (different folds?)')
            if not np.array_equal(members[0][f'clips_{split}'], other[f'clips_{split}']):
                raise ValueError(f'members disagree on the {split} clip order')


def accuracy(probs, targets):
    """
    Top-1 accuracy of a probability matrix.

    Args:
        probs (ndarray): (n_clips, n_classes) probabilities.
        targets (ndarray): (n_clips,) true labels.

    Returns:
        float: Fraction of clips whose most probable class is the true one.
    """
    return float(np.mean(probs.argmax(axis=1) == targets))


def fuse(members, names, weights=None):
    """
    Accuracy of every member alone and of their weighted probability average, on val and test.

    Args:
        members (list): Prediction dicts as returned by load_predictions.
        names (list): A name per member, for the report.
        weights (list, optional): One weight per member; equal weights when omitted.

    Returns:
        dict: {'weights': {name: w}, 'members': {name: {'val': acc, 'test': acc}},
               'fused': {'val': acc, 'test': acc}}.
    """
    check_same_clips(members)
    weights = [1.0] * len(members) if weights is None else [float(w) for w in weights]
    report = {'weights': dict(zip(names, weights)), 'members': {}, 'fused': {}}
    for name, member in zip(names, members):
        report['members'][name] = {
            split: accuracy(member[f'probs_{split}'], member[f'targets_{split}'])
            for split in ('val', 'test')}
    for split in ('val', 'test'):
        total = sum(w * m[f'probs_{split}'] for w, m in zip(weights, members)) / sum(weights)
        report['fused'][split] = accuracy(total, members[0][f'targets_{split}'])
    return report


def main():
    """Fuse the given prediction files, print the accuracies and save them as JSON."""
    parser = argparse.ArgumentParser(description='Two-stream score fusion')
    parser.add_argument('--members', nargs='+', required=True, help='predictions.npz files')
    parser.add_argument('--names', nargs='+', required=True, help='one name per member')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='one weight per member (default: equal)')
    parser.add_argument('--out', default='fusion.json', help='where to save the accuracies')
    args = parser.parse_args()
    if len(args.members) != len(args.names):
        raise ValueError('--members and --names must have the same length')

    report = fuse([load_predictions(path) for path in args.members], args.names, args.weights)
    print(f"{'model':24s} {'val':>8s} {'test':>8s}")
    for name, acc in report['members'].items():
        print(f"{name:24s} {100 * acc['val']:7.2f}% {100 * acc['test']:7.2f}%")
    print(f"{'fused (equal weights)':24s} {100 * report['fused']['val']:7.2f}% "
          f"{100 * report['fused']['test']:7.2f}%")
    with open(args.out, 'w', encoding='utf-8') as out_file:
        json.dump(report, out_file, indent=2)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
