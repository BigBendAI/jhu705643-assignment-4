"""Tests for VideoDataset and the collate function: frame order and count, clip transforms."""

import os

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from video_datasets import VideoDataset, collate_fn_rnn
from utils import compose_data_transforms


def write_clip(clip_dir, n_frames, size=(8, 6), step=7):
    """Write n_frames JPEGs whose red channel encodes the frame index (as index * step)."""
    os.makedirs(clip_dir)
    for idx in range(n_frames):
        image = Image.new('RGB', size, (idx * step, 0, 0))
        image.save(os.path.join(clip_dir, f'frame_{idx:03d}.jpg'))


def red_levels(clip, step=7):
    """The mean red value of each frame of a uint8 clip tensor, rounded to the encoded index."""
    return [int(round(frame[0].float().mean().item() / step)) for frame in clip]


@pytest.fixture(name='frames')
def frames_fixture(tmp_path):
    """Two clips: one with 32 stored frames, one with only 12."""
    write_clip(tmp_path / 'A' / 'v_A_g01_c01', 32)
    write_clip(tmp_path / 'A' / 'v_A_g01_c02', 12)
    return tmp_path


def test_frames_come_in_time_order_evenly_spaced(frames):
    """glob gives directory order; the dataset sorts the files and takes 16 evenly spaced ones."""
    dataset = VideoDataset([(str(frames / 'A' / 'v_A_g01_c01'), 0)], 16)
    clip, label = dataset[0]
    assert label == 0
    assert clip.shape == (16, 3, 6, 8) and clip.dtype == torch.uint8
    assert red_levels(clip) == [round(i * 31 / 15) for i in range(16)]
    assert red_levels(dataset[0][0]) == red_levels(dataset[0][0])      # deterministic


def test_random_clip_draws_one_frame_per_segment(frames):
    """With 32 stored frames and 16 wanted, each pick comes from its own segment of two frames."""
    dataset = VideoDataset([(str(frames / 'A' / 'v_A_g01_c01'), 0)], 16, random_clip=True)
    seen = set()
    for _ in range(20):
        levels = red_levels(dataset[0][0])
        assert len(levels) == 16 and levels == sorted(levels)
        assert all(level // 2 == segment for segment, level in enumerate(levels))
        seen.add(tuple(levels))
    assert len(seen) > 1                      # different draws on different loads


def test_short_clip_is_stretched_to_fr_per_vid(frames):
    """A clip with fewer stored frames still yields fr_per_vid frames, evenly spaced, in order."""
    dataset = VideoDataset([(str(frames / 'A' / 'v_A_g01_c02'), 1)], 16, random_clip=True)
    levels = red_levels(dataset[0][0])
    assert len(levels) == 16 and levels == sorted(levels) and levels[0] == 0 and levels[-1] == 11


def test_missing_frames_raise(tmp_path):
    """An empty clip directory is an error, not a silently dropped sample."""
    os.makedirs(tmp_path / 'A' / 'v_A_g01_c01')
    with pytest.raises(FileNotFoundError):
        _ = VideoDataset([(str(tmp_path / 'A' / 'v_A_g01_c01'), 0)], 16)[0]


def test_train_transform_treats_the_clip_as_one(frames):
    """Every frame of a clip gets the same random flip and shift, and the output is normalised."""
    write_clip(frames / 'B' / 'v_B_g01_c01', 16, size=(32, 32), step=15)
    train_tf, val_tf = compose_data_transforms(24, 24, [0.5] * 3, [0.25] * 3)
    # Make the frames of this clip fully identical with a transform of our own, then check that
    # the random transform keeps them identical (same parameters for the whole clip).
    same = v2.Compose([v2.Lambda(lambda x: x[:1].expand(x.shape[0], -1, -1, -1).clone()), train_tf])
    clip, _ = VideoDataset([(str(frames / 'B' / 'v_B_g01_c01'), 0)], 16, same)[0]
    assert clip.shape == (16, 3, 24, 24) and clip.dtype == torch.float32
    assert all(torch.equal(clip[0], frame) for frame in clip)
    clip, _ = VideoDataset([(str(frames / 'B' / 'v_B_g01_c01'), 0)], 16, val_tf)[0]
    assert clip.shape == (16, 3, 24, 24) and clip.min() < 0 < clip.max()


def test_collate_stacks_clips(frames):
    """The collate function returns (batch, T, C, H, W) and a label tensor."""
    dataset = VideoDataset([(str(frames / 'A' / 'v_A_g01_c01'), 0),
                            (str(frames / 'A' / 'v_A_g01_c02'), 1)], 16)
    clips, labels = collate_fn_rnn([dataset[0], dataset[1]])
    assert clips.shape == (2, 16, 3, 6, 8)
    assert labels.tolist() == [0, 1]


def test_interleaved_views(frames):
    """With 32 stored frames and 2 views, view 1 takes the second frame of each pair."""
    clip_list = [(str(frames / 'A' / 'v_A_g01_c01'), 0)]
    first = red_levels(VideoDataset(clip_list, 16)[0][0])
    second = red_levels(VideoDataset(clip_list, 16, view=1, n_views=2)[0][0])
    assert first == [round(i * 31 / 15) for i in range(16)]
    assert all(b >= a for a, b in zip(first, second)) and second[-1] == 31
    assert sum(b > a for a, b in zip(first, second)) == 15   # only the last frame coincides
    # with no spare frames every view equals the plain clip
    short = [(str(frames / 'A' / 'v_A_g01_c02'), 0)]
    assert (red_levels(VideoDataset(short, 16, view=3, n_views=4)[0][0])
            == red_levels(VideoDataset(short, 16)[0][0]))
    with pytest.raises(ValueError):
        VideoDataset(clip_list, 16, view=4, n_views=4)


def test_four_views_of_sixty_four_frames_are_interleaved():
    """With 64 stored frames, the 4 views are 16 distinct frames each, interleaved in time."""
    views = [VideoDataset([], 16, view=v, n_views=4).frame_indices(64) for v in range(4)]
    assert all(len(set(v)) == 16 for v in views)
    for earlier, later in zip(views, views[1:]):
        assert all(b >= a for a, b in zip(earlier, later))
        assert sum(b > a for a, b in zip(earlier, later)) >= 15
    assert len({i for v in views for i in v}) >= 60      # together they cover almost all frames
