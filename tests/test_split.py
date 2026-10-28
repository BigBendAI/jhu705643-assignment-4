"""Tests for the group-level cross-validation split in video_datasets.py."""

import pytest

from video_datasets import (clip_group, dataset_split, split_overlap,
                            write_split_csv, read_split_csv)


def fake_dataset(n_classes=3, n_groups=25, n_clips=4):
    """A UCF50-like path -> label dictionary without any files on disk."""
    dataset = {}
    for label in range(n_classes):
        for group in range(1, n_groups + 1):
            for clip in range(1, n_clips + 1):
                dataset[f'/frames/Class{label}/v_Class{label}_g{group:02d}_c{clip:02d}'] = label
    return dataset


def as_dict(parts):
    """(train, val, test) lists -> dict."""
    return dict(zip(('train', 'val', 'test'), parts))


def test_clip_group_reads_class_and_group():
    """The class name and group number come from the clip folder name."""
    assert clip_group('/x/Basketball/v_Basketball_g07_c03') == ('Basketball', 7)
    with pytest.raises(ValueError):
        clip_group('/x/Basketball/clip_without_group')


def test_fold_keeps_groups_apart_and_every_class_in_every_set():
    """No group is shared between sets, every class is in every set, counts follow the folds."""
    dataset = fake_dataset()
    splits = as_dict(dataset_split(dataset, fold=0))
    assert sum(len(v) for v in splits.values()) == len(dataset)
    assert not split_overlap(splits)
    for clips in splits.values():
        assert {label for _, label in clips} == set(range(3))
    # 25 groups per class, 5 folds: 5 test groups, 2 validation groups, 18 training groups
    assert len(splits['test']) == 3 * 5 * 4
    assert len(splits['val']) == 3 * 2 * 4
    assert len(splits['train']) == 3 * 18 * 4


def test_folds_cover_every_group_once_as_test():
    """Over the five folds every group is in the test set exactly once."""
    dataset = fake_dataset()
    seen = []
    for fold in range(5):
        seen += sorted({clip_group(p) for p, _ in dataset_split(dataset, fold)[2]})
    assert sorted(seen) == sorted({clip_group(p) for p in dataset})   # each group once


def test_split_is_reproducible_and_seed_dependent():
    """The same seed gives the same split; another seed gives a different one."""
    dataset = fake_dataset()
    assert dataset_split(dataset, 0, seed=0) == dataset_split(dataset, 0, seed=0)
    assert dataset_split(dataset, 0, seed=0) != dataset_split(dataset, 0, seed=1)
    with pytest.raises(ValueError):
        dataset_split(dataset, fold=5)


def test_split_overlap_detects_a_shared_group():
    """A group that appears in two sets is reported."""
    splits = {'train': [('/f/A/v_A_g01_c01', 0)], 'val': [('/f/A/v_A_g02_c01', 0)],
              'test': [('/f/A/v_A_g01_c02', 0)]}
    assert split_overlap(splits) == {('A', 1)}


def test_csv_round_trip(tmp_path):
    """Writing and reading the CSV gives the split back, relative to any frame directory."""
    splits = as_dict(dataset_split(fake_dataset(n_classes=2, n_groups=10), fold=1))
    csv_path = tmp_path / 'fold1.csv'
    write_split_csv(csv_path, splits, '/frames')
    assert read_split_csv(csv_path, '/frames') == splits
    assert read_split_csv(csv_path, '/elsewhere')['test'][0][0].startswith('/elsewhere/')
