"""Tests for the LRCN model in models.py: every video is classified from its own frame sequence."""

import torch

from models import LRCN


def small_lrcn():
    """A small untrained LRCN in eval mode (no dropout, no download)."""
    torch.manual_seed(0)
    model = LRCN(hidden_size=8, n_layers=1, dropout_rate=0.1, n_classes=5, pretrained=False,
                 cnn_model='resnet18')
    return model.eval()


def clips(batch_size=4, time_steps=3, size=32):
    """A random batch of clips shaped (batch, time, channels, height, width)."""
    torch.manual_seed(1)
    return torch.rand(batch_size, time_steps, 3, size, size)


def test_output_shape():
    """One row of logits per video, one column per class."""
    assert small_lrcn()(clips()).shape == (4, 5)


def test_videos_in_a_batch_do_not_influence_each_other():
    """Changing the last video of a batch must leave the other videos' logits unchanged.

    The hand-over fed the LSTM (batch, 1, features) without batch_first, so it treated the batch
    as a sequence and the logits of video 0 depended on videos 1 to 3.
    """
    model, x = small_lrcn(), clips()
    with torch.no_grad():
        before = model(x)
        x[3] = torch.rand_like(x[3])
        after = model(x)
    assert torch.allclose(before[:3], after[:3], atol=1e-5)
    assert not torch.allclose(before[3], after[3], atol=1e-5)


def test_batched_and_single_video_logits_agree():
    """A video gets the same logits whether it is classified alone or inside a batch."""
    model, x = small_lrcn(), clips()
    with torch.no_grad():
        batched = model(x)
        single = torch.cat([model(x[i:i + 1]) for i in range(x.shape[0])])
    assert torch.allclose(batched, single, atol=1e-5)


def test_frame_order_matters():
    """The LSTM reads the frames as a sequence, so reversing them changes the logits."""
    model, x = small_lrcn(), clips()
    with torch.no_grad():
        forward_order = model(x)
        reversed_order = model(x.flip(1))
    assert not torch.allclose(forward_order, reversed_order, atol=1e-5)
