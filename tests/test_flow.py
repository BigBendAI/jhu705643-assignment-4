"""Tests for the optical-flow images and the flow-specific transforms in utils.py."""

import numpy as np
import pytest
from torchvision.transforms import v2

from utils import flow_to_image, compose_data_transforms


def test_flow_to_image_encoding():
    """Zero motion is (128, 128, 0); one pixel of x motion adds 16 grey levels; values clip.

    The magnitude channel is truncated to a whole grey level, as the uint8 cast does.
    """
    flow = np.zeros((4, 4, 2), dtype=np.float32)
    flow[0, 0] = (1.0, -2.0)
    flow[1, 1] = (100.0, 0.0)
    image = flow_to_image(flow)
    assert image.dtype == np.uint8 and image.shape == (4, 4, 3)
    assert tuple(image[2, 2]) == (128, 128, 0)
    assert tuple(image[0, 0]) == (144, 96, int(16 * np.sqrt(5)))
    assert tuple(image[1, 1]) == (255, 128, 255)


@pytest.mark.parametrize('strong', [False, True])
def test_flow_transforms_have_no_flip_or_jitter(strong):
    """The RGB pipeline flips (and jitters when strong); the flow pipeline never does."""
    rgb, _ = compose_data_transforms(8, 8, [0.5] * 3, [0.5] * 3, strong=strong, modality='rgb')
    flow, _ = compose_data_transforms(8, 8, [0.5] * 3, [0.5] * 3, strong=strong, modality='flow')
    rgb_types = {type(t) for t in rgb.transforms}
    flow_types = {type(t) for t in flow.transforms}
    assert v2.RandomHorizontalFlip in rgb_types and v2.RandomHorizontalFlip not in flow_types
    assert (v2.ColorJitter in rgb_types) == strong and v2.ColorJitter not in flow_types
    assert v2.Normalize in flow_types and v2.ToDtype in flow_types
    with pytest.raises(ValueError):
        compose_data_transforms(8, 8, [0.5] * 3, [0.5] * 3, modality='depth')
