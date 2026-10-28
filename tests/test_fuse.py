"""Tests for the two-stream score fusion in fuse.py."""

import numpy as np
import pytest

from fuse import fuse, load_predictions, check_same_clips


def member(probs_val, probs_test, targets, clips):
    """A prediction dict with the same targets and clips for val and test."""
    return {'probs_val': np.array(probs_val, dtype=np.float32),
            'probs_test': np.array(probs_test, dtype=np.float32),
            'targets_val': np.array(targets), 'targets_test': np.array(targets),
            'clips_val': np.array(clips), 'clips_test': np.array(clips)}


def test_fusion_averages_probabilities():
    """Each stream is right on one clip; their average is right on both."""
    targets, clips = [0, 1], ['A/a', 'B/b']
    rgb = member([[0.9, 0.1], [0.6, 0.4]], [[0.9, 0.1], [0.6, 0.4]], targets, clips)
    flow = member([[0.4, 0.6], [0.1, 0.9]], [[0.4, 0.6], [0.1, 0.9]], targets, clips)
    report = fuse([rgb, flow], ['rgb', 'flow'])
    assert report['members']['rgb'] == {'val': 0.5, 'test': 0.5}
    assert report['members']['flow'] == {'val': 0.5, 'test': 0.5}
    assert report['fused'] == {'val': 1.0, 'test': 1.0}
    assert report['weights'] == {'rgb': 1.0, 'flow': 1.0}
    weighted = fuse([rgb, flow], ['rgb', 'flow'], weights=[1.0, 0.0])
    assert weighted['fused'] == {'val': 0.5, 'test': 0.5}


def test_members_must_hold_the_same_clips():
    """A member with different clips or labels is rejected."""
    rgb = member([[1, 0]], [[1, 0]], [0], ['A/a'])
    other_clip = member([[1, 0]], [[1, 0]], [0], ['A/b'])
    other_label = member([[1, 0]], [[1, 0]], [1], ['A/a'])
    with pytest.raises(ValueError):
        check_same_clips([rgb, other_clip])
    with pytest.raises(ValueError):
        check_same_clips([rgb, other_label])


def test_load_predictions_round_trip(tmp_path):
    """An npz written like run.py --mode predict loads back with every key."""
    rgb = member([[1, 0]], [[0, 1]], [0], ['A/a'])
    np.savez(tmp_path / 'p.npz', class_names=np.array(['A', 'B']), **rgb)
    loaded = load_predictions(tmp_path / 'p.npz')
    assert set(loaded) == set(rgb) | {'class_names'}
    assert np.array_equal(loaded['probs_test'], rgb['probs_test'])
