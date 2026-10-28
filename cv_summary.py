"""
Module: cv_summary.py

Print the per-fold accuracies of the cross-validation (the fusion.json files written by cv.sh)
and their mean and standard deviation: the headline number of this repository.

Usage:
    python cv_summary.py --out_dir cv
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    """Read every fold's fusion.json below --out_dir and print the table and the mean."""
    parser = argparse.ArgumentParser(description='Cross-validation summary')
    parser.add_argument('--out_dir', default='cv', help='folder with fold<k>/fusion.json files')
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, 'fold*', 'fusion.json'))):
        with open(path, encoding='utf-8') as json_file:
            report = json.load(json_file)
        fold = os.path.basename(os.path.dirname(path))
        rows.append((fold, report['members']['rgb'], report['members']['flow'], report['fused']))
    if not rows:
        raise FileNotFoundError(f'no fold*/fusion.json below {args.out_dir}')

    print(f"{'fold':6s} {'rgb val':>8s} {'rgb test':>9s} {'flow val':>9s} {'flow test':>10s} "
          f"{'fused val':>10s} {'fused test':>11s}")
    for fold, rgb, flow, fused in rows:
        print(f"{fold:6s} {100 * rgb['val']:8.2f} {100 * rgb['test']:9.2f} "
              f"{100 * flow['val']:9.2f} {100 * flow['test']:10.2f} "
              f"{100 * fused['val']:10.2f} {100 * fused['test']:11.2f}")
    val = np.array([100 * fused['val'] for _, _, _, fused in rows])
    test = np.array([100 * fused['test'] for _, _, _, fused in rows])
    print(f"{'mean':6s} {'':8s} {'':9s} {'':9s} {'':10s} {val.mean():10.2f} {test.mean():11.2f}")
    print(f"{'std':6s} {'':8s} {'':9s} {'':9s} {'':10s} {val.std():10.2f} {test.std():11.2f}")
    print(f'\nTwo-stream LRCN, {len(rows)}-fold group cross-validation: '
          f'test accuracy {test.mean():.2f} +- {test.std():.2f} (validation {val.mean():.2f})')


if __name__ == '__main__':
    main()
