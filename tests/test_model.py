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


def test_late_fusion_averages_per_frame_logits():
    """late_fusion gives one row of logits per video and differs from last-step classification."""
    torch.manual_seed(0)
    model = LRCN(hidden_size=8, n_layers=1, dropout_rate=0.1, n_classes=5, pretrained=False,
                 cnn_model='resnet18', temporal_pooling='late_fusion').eval()
    x = clips()
    with torch.no_grad():
        fused = model(x)
        model.temporal_pooling = 'last'
        last = model(x)
    assert fused.shape == (4, 5)
    assert not torch.allclose(fused, last)


def test_freeze_bn_keeps_backbone_batchnorm_in_eval_mode():
    """In training mode the backbone's BatchNorm layers stay in eval mode when freeze_bn is set."""
    model = LRCN(hidden_size=8, n_layers=1, dropout_rate=0.1, n_classes=5, pretrained=False,
                 cnn_model='resnet18', freeze_bn=True).train()
    assert model.training and model.rnn.training
    assert not any(m.training for m in model.base_model.modules()
                   if isinstance(m, torch.nn.BatchNorm2d))
    plain = LRCN(hidden_size=8, n_layers=1, dropout_rate=0.1, n_classes=5, pretrained=False,
                 cnn_model='resnet18').train()
    assert all(m.training for m in plain.base_model.modules()
               if isinstance(m, torch.nn.BatchNorm2d))


def test_freeze_stages_and_parameter_groups():
    """freeze_stages=1 fixes stem + layer1; the groups hold each trainable parameter once."""
    model = LRCN(hidden_size=8, n_layers=1, dropout_rate=0.1, n_classes=5, pretrained=False,
                 cnn_model='resnet18', freeze_stages=1)
    assert not any(p.requires_grad for p in model.base_model.layer1.parameters())
    assert all(p.requires_grad for p in model.base_model.layer2.parameters())
    groups = model.parameter_groups(1e-3, 0.1)
    assert len(groups) == 2
    head, backbone = groups[0], groups[1]
    assert head['lr'] == 1e-3 and abs(backbone['lr'] - 1e-4) < 1e-12
    grouped = [id(p) for g in (head, backbone) for p in g['params']]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(grouped) == sorted(trainable) and len(grouped) == len(set(grouped))
    new_layers = list(model.rnn.parameters()) + list(model.fc.parameters())
    assert {id(p) for p in head['params']} == {id(p) for p in new_layers}


def test_bidirectional_two_layer_lstm():
    """Two stacked bidirectional layers still give one row of logits per video."""
    torch.manual_seed(0)
    model = LRCN(hidden_size=8, n_layers=2, dropout_rate=0.5, n_classes=5, pretrained=False,
                 cnn_model='resnet18', bidirectional=True, temporal_pooling='late_fusion').eval()
    assert model.rnn.bidirectional and model.rnn.num_layers == 2 and model.rnn.dropout == 0.5
    assert model.fc.in_features == 16
    x = clips()
    with torch.no_grad():
        batched = model(x)
        single = torch.cat([model(x[i:i + 1]) for i in range(x.shape[0])])
    assert batched.shape == (4, 5)
    assert torch.allclose(batched, single, atol=1e-5)
