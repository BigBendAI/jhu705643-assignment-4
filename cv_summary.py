"""
Module: cv_summary.py

Print the per-fold accuracies of the cross-validation (the fusion.json files written by cv.sh)
and their mean and standard deviation: the headline number of this repository. When a fold also
has a fusion3.json (the 3D CNN stream added, cv.sh with STREAMS=3) those columns are shown too.

Usage:
    python cv_summary.py --out_dir cv
"""

import argparse
import glob
import json
import os

import numpy as np


def load(path):
    """Read a fusion JSON, or return None when the file does not exist."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as json_file:
        return json.load(json_file)


def main():
    """Read every fold's fusion files below --out_dir and print the table and the means."""
    parser = argparse.ArgumentParser(description='Cross-validation summary')
    parser.add_argument('--out_dir', default='cv', help='folder with fold<k>/fusion.json files')
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, 'fold*', 'fusion.json'))):
        fold = os.path.basename(os.path.dirname(path))
        rows.append((fold, load(path), load(os.path.join(os.path.dirname(path), 'fusion3.json'))))
    if not rows:
        raise FileNotFoundError(f'no fold*/fusion.json below {args.out_dir}')
    with_3d = all(three is not None for _, _, three in rows)

    columns = [('rgb', lambda two, three: two['members']['rgb']),
               ('flow', lambda two, three: two['members']['flow']),
               ('rgb+flow', lambda two, three: two['fused'])]
    if with_3d:
        columns += [('3dcnn', lambda two, three: three['members']['3dcnn']),
                    ('rgb+flow+3dcnn', lambda two, three: three['fused'])]
    header = ''.join(f"{name + ' val':>16s}{name + ' test':>17s}" for name, _ in columns)
    print(f"{'fold':6s}" + header)
    for fold, two, three in rows:
        line = f'{fold:6s}'
        for _, pick in columns:
            acc = pick(two, three)
            line += f"{100 * acc['val']:16.2f}{100 * acc['test']:17.2f}"
        print(line)
    for stat, func in (('mean', np.mean), ('std', np.std)):
        line = f'{stat:6s}'
        for _, pick in columns:
            for split in ('val', 'test'):
                line += f"{func([100 * pick(two, three)[split] for _, two, three in rows]):16.2f}"
        print(line)
    test = [100 * two['fused']['test'] for _, two, _ in rows]
    print(f'\nTwo-stream LRCN (rgb + flow), {len(rows)}-fold group cross-validation: '
          f'test accuracy {np.mean(test):.2f} +- {np.std(test):.2f}')
    if with_3d:
        test3 = [100 * three['fused']['test'] for _, _, three in rows]
        print(f'With the 3D CNN stream (rgb + flow + 3dcnn): '
              f'test accuracy {np.mean(test3):.2f} +- {np.std(test3):.2f}')


if __name__ == '__main__':
    main()
